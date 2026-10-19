# Standard library imports
import logging
from pathlib import Path

# Third party imports
import torch
from torchvision.ops import batched_nms
from transformers.trainer_utils import get_last_checkpoint

# Local folder imports
from .dataset import canvas_size

logger = logging.getLogger(__name__)


class RatioSampler(torch.utils.data.Sampler):
    """
    Samples from multiple datasets with specific ratios.
    Anchors the epoch length to the FIRST dataset (index 0).
    For example, ratios=[9, 1] means 9 samples from dataset 0 for every 1 from dataset 1.
    Dataset 1 will be oversampled or subsampled to exactly match Dataset 0's needs.
    """

    def __init__(self, dataset_lengths, ratios):
        self.dataset_lengths = dataset_lengths
        self.ratios = ratios

        self.offsets = [0]
        for i in range(len(dataset_lengths) - 1):
            self.offsets.append(self.offsets[-1] + dataset_lengths[i])

        # Base length is strictly determined by the FIRST dataset (the anchor).
        # This guarantees 1 full pass of dataset 0. Other datasets follow the ratio.
        self.base_len = max(1, dataset_lengths[0] // ratios[0])
        self.total_size = sum([self.base_len * r for r in ratios])

    def __iter__(self):
        iters = []
        for i, l in enumerate(self.dataset_lengths):
            if l == 0:
                raise ValueError(f"Dataset at index {i} has length 0.")

            num_samples = self.base_len * self.ratios[i]

            if num_samples > l:
                # Oversample: repeat the dataset enough times to satisfy the ratio
                repeats = (num_samples // l) + 1
                perm = torch.cat([torch.randperm(l) for _ in range(repeats)])[
                    :num_samples
                ]
            else:
                # Undersample: randomly pick the exact number of samples needed
                # (Silently drops the remainder of this specific epoch)
                perm = torch.randperm(l)[:num_samples]

            iters.append(iter(perm.tolist()))

        # Yield indices following the strict ratio sequence
        for _ in range(self.base_len):
            for i, r in enumerate(self.ratios):
                for _ in range(r):
                    yield next(iters[i]) + self.offsets[i]

    def __len__(self):
        return self.total_size


def omr_collate_fn(batch):
    """Collate images and targets into a dictionary for Trainer.

    Returns:
        dict: {
            "pixel_values": (B, 3, H, W) float32 tensor
            "labels": list[dict] of length B, annotations
        }
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return {"pixel_values": images, "labels": list(targets)}


def resolve_weights(checkpoint_path: str) -> Path | None:
    """The weights file of a checkpoint: an HF Trainer output dir (its last
    checkpoint-NNNN), a run dir, or a direct path to the weights."""
    ckpt = Path(checkpoint_path)
    if ckpt.is_dir():
        last = get_last_checkpoint(str(ckpt))
        if last is not None:
            ckpt = Path(last)
    for name in ("model.safetensors", "pytorch_model.bin"):
        weights = (ckpt / name) if ckpt.is_dir() else ckpt
        if weights.exists():
            return weights
    return None


def _checkpoint_tensors(checkpoint_path: str, prefix: str = "") -> dict:
    """A detector checkpoint's weights as shapes only (meta tensors), except
    the small ``input_wh`` buffer, which is read for its value."""
    weights = resolve_weights(checkpoint_path)
    if weights is None:
        raise FileNotFoundError(f"no weights under {checkpoint_path}")
    if weights.suffix != ".safetensors":
        return torch.load(weights, map_location="cpu", weights_only=False)
    # Third party imports
    from safetensors import safe_open

    tensors = {}
    with safe_open(str(weights), "pt") as f:
        for k in f.keys():
            if not k.startswith(prefix):
                continue
            if k == f"{prefix}input_wh":
                tensors[k] = f.get_tensor(k)
            else:
                tensors[k] = torch.empty(f.get_slice(k).get_shape(), device="meta")
    return tensors


def checkpoint_backbone(checkpoint_path: str, prefix: str = "") -> str:
    """The backbone a detector checkpoint holds (``prefix``: 'detector.' for
    GNN wrapper checkpoints), read from the weight names and shapes only."""
    # Local folder imports
    from .model import detect_backbone

    return detect_backbone(_checkpoint_tensors(checkpoint_path, prefix), prefix=prefix)


def checkpoint_input_size(checkpoint_path: str, prefix: str = "") -> tuple[int, int]:
    """The (width, height) canvas a detector checkpoint was trained at."""
    # Local folder imports
    from .model import detector_layout

    return detector_layout(_checkpoint_tensors(checkpoint_path, prefix), prefix=prefix)["input_size"]


def detector_for_checkpoint(checkpoint_path: str, num_classes: int, prefix: str = "", **kwargs):
    """An (unloaded) MusicDetector shaped like the checkpoint's detector:
    the same backbone, detection levels, neck and input canvas, no pretrained
    download. Load it with load_checkpoint. The layout read from the weights
    wins over ``kwargs`` (a mismatch would only fail at load time)."""
    # Local folder imports
    from .model import MusicDetector, detect_backbone, detector_layout

    tensors = _checkpoint_tensors(checkpoint_path, prefix)
    backbone = detect_backbone(tensors, prefix=prefix)
    layout = detector_layout(tensors, prefix=prefix)
    for key, value in layout.items():
        given = kwargs.get(key)
        if given is None:
            continue
        given = canvas_size(given) if key == "input_size" else given
        if (tuple(given) if isinstance(value, tuple) else given) != value:
            logger.warning(f"  {key}={kwargs[key]} ignored: the checkpoint has {value}")
    kwargs.update(layout)
    width, height = layout["input_size"]
    logger.info(f"  Detector: {backbone}, {len(kwargs.get('out_indices', (0, 1, 2)))} levels, {width}x{height} input")
    return MusicDetector(num_classes=num_classes, backbone=backbone, pretrained=False, **kwargs)


def complete_legacy_state(model, state: dict) -> dict:
    """Detectors from before the canvas and scale ranges were stored keep the
    model's: square 640 (what detector_for_checkpoint gives them) and the
    default 3-level ranges they were trained with."""
    for key, value in model.state_dict().items():
        if key.endswith(("input_wh", "level_areas")) and key not in state:
            state[key] = value
    return state


def load_checkpoint(
    model, checkpoint_path: str, device: torch.device, eval: bool = True
):
    """Resolve a checkpoint path and load weights into the model.

    Handles:
      - HF Trainer output dirs (finds the last checkpoint-NNNN subdirectory)
      - Direct paths to model.safetensors or pytorch_model.bin
      - Single-run directories containing weight files

    Args:
        eval: If True (default), sets the model to eval mode. Set to False for training.
    """
    ckpt = Path(checkpoint_path)
    if ckpt.is_dir():
        last = get_last_checkpoint(str(ckpt))
        if last is not None:
            ckpt = Path(last)
            logger.info(f"  Using last checkpoint: {ckpt}")

    for name in ("model.safetensors", "pytorch_model.bin"):
        weights = (ckpt / name) if ckpt.is_dir() else ckpt
        if weights.exists():
            if weights.suffix == ".safetensors":
                # Third party imports
                from safetensors.torch import load_file

                state = load_file(weights, device=str(device))
            else:
                state = torch.load(weights, map_location=device, weights_only=False)
            complete_legacy_state(model, state)
            model.load_state_dict(state)
            logger.info(f"  Loaded weights: {weights}")
            break
    else:
        logger.warning("No weights file found — using random weights.")

    model.to(device)
    if eval:
        model.eval()


def decode_model_outputs(outputs, conf_thresh, iou_thresh, input_size, multi_label=True):
    """
    Decodes multi-scale raw model outputs into absolute pixel bounding boxes
    on the canvas (``input_size``: int or (width, height)).
    Handles dynamic batch sizes and dynamic feature map resolutions.

    ``multi_label`` emits every class over the threshold at a cell (sharing
    the cell's box); False keeps only its argmax. Training marks all the
    classes centred on a cell positive, and with the argmax coincident boxes
    (measure/staff/layer of a single staff, mRest over its whole-rest glyph,
    a one-accidental keySig) suppress each other.

    Returns:
        list[dict]: A list of length BatchSize, containing dicts with "boxes", "scores", and "labels".
    """
    batch_size = outputs[0]["cls"].shape[0]
    device = outputs[0]["cls"].device
    canvas_w, canvas_h = canvas_size(input_size)

    all_boxes = [[] for _ in range(batch_size)]
    all_scores = [[] for _ in range(batch_size)]
    all_labels = [[] for _ in range(batch_size)]

    for out in outputs:
        cls_map = out["cls"].sigmoid()
        reg_map = out["reg"]

        B, _, H, W = reg_map.shape

        if multi_label:
            pos = (cls_map >= conf_thresh).nonzero(as_tuple=False)
            if pos.numel() == 0:
                continue
            b_idx, cls_idx, y_idx, x_idx = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
            scores = cls_map[b_idx, cls_idx, y_idx, x_idx]
        else:
            max_scores, max_classes = cls_map.max(dim=1)
            mask = max_scores >= conf_thresh
            pos = mask.nonzero(as_tuple=False)

            if pos.numel() == 0:
                continue

            b_idx, y_idx, x_idx = pos[:, 0], pos[:, 1], pos[:, 2]

            scores = max_scores[b_idx, y_idx, x_idx]
            cls_idx = max_classes[b_idx, y_idx, x_idx]

        tx = reg_map[b_idx, 0, y_idx, x_idx].sigmoid()
        ty = reg_map[b_idx, 1, y_idx, x_idx].sigmoid()
        tw = reg_map[b_idx, 2, y_idx, x_idx].exp()
        th = reg_map[b_idx, 3, y_idx, x_idx].exp()

        cx_n = (tx + x_idx) / W
        cy_n = (ty + y_idx) / H

        cx = cx_n * canvas_w
        cy = cy_n * canvas_h
        bw = tw * canvas_w
        bh = th * canvas_h

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        for b in b_idx.unique().tolist():
            sel = b_idx == b
            all_boxes[b].append(boxes[sel])
            all_scores[b].append(scores[sel])
            all_labels[b].append(cls_idx[sel])

    batch_preds = []
    for b in range(batch_size):
        if len(all_boxes[b]) > 0:
            b_boxes = torch.cat(all_boxes[b], dim=0)
            b_scores = torch.cat(all_scores[b], dim=0)
            b_labels = torch.cat(all_labels[b], dim=0)

            keep = batched_nms(b_boxes, b_scores, b_labels, iou_thresh)

            batch_preds.append(
                {
                    "boxes": b_boxes[keep],
                    "scores": b_scores[keep],
                    "labels": b_labels[keep],
                }
            )
        else:
            batch_preds.append(
                {
                    "boxes": torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                }
            )

    return batch_preds
