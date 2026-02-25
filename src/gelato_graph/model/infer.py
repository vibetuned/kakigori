"""PDF Inference Script for EdgeMusicDetector.

Renders every page of a PDF, runs the trained model on each, and outputs:
  - page_NNNN.png  — raw rendered page image
  - page_NNNN.json — detections in dataset annotation format

Usage:
    infer-model --checkpoint checkpoints/my_run --pdf score.pdf --output-dir out/
"""

import argparse
import json
import logging
from pathlib import Path

from transformers.trainer_utils import get_last_checkpoint

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .model import EdgeMusicDetector
from .dataset import _letterbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF → PIL pages
# ---------------------------------------------------------------------------

def pdf_to_pages(pdf_path: Path, dpi: int = 300) -> list[Image.Image]:
    """Render each page of a PDF to a PIL RGB image."""
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError:
        raise ImportError(
            "pypdfium2 is required for PDF rendering. Install it with:\n"
            "  uv add pypdfium2"
        )

    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    pages = []
    for i, page in enumerate(doc):
        bitmap = page.render(scale=scale, rotation=0)
        pages.append(bitmap.to_pil().convert("RGB"))
        logger.info(f"  Rendered page {i + 1}/{len(doc)}")
    return pages


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(image: Image.Image, input_size: int, device: torch.device) -> tuple[torch.Tensor, dict]:
    """Letterbox + normalise a PIL image. Returns (tensor, padding metadata)."""
    orig_w, orig_h = image.size
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2

    img_lb, _ = _letterbox(image, input_size, torch.zeros(0, 4))
    tensor = TF.to_tensor(img_lb).unsqueeze(0).to(device)  # (1, 3, H, W)

    meta = {
        "orig_w": orig_w, "orig_h": orig_h,
        "scale": scale, "pad_x": pad_x, "pad_y": pad_y,
        "input_size": input_size,
    }
    return tensor, meta


# ---------------------------------------------------------------------------
# Post-processing: decode + NMS
# ---------------------------------------------------------------------------

def decode_outputs(
    outputs: list[dict],
    meta: dict,
    conf_thresh: float,
    class_list: list[str],
) -> list[dict]:
    """Decode multi-scale head outputs into detections in original image coords."""
    detections = []
    s = meta["input_size"]

    for out in outputs:
        cls_map = out["cls"].sigmoid()   # (1, C, H, W)
        reg_map = out["reg"]             # (1, 4, H, W)

        max_scores, max_classes = cls_map[0].max(dim=0)  # (H, W)
        pos = (max_scores >= conf_thresh).nonzero(as_tuple=False)  # (N, 2)

        if pos.numel() == 0:
            continue

        # --- Vectorised decode (stays on GPU) ---
        scores = max_scores[pos[:, 0], pos[:, 1]]        # (N,)
        cls_idx = max_classes[pos[:, 0], pos[:, 1]]      # (N,)

        cx_n = reg_map[0, 0, pos[:, 0], pos[:, 1]]
        cy_n = reg_map[0, 1, pos[:, 0], pos[:, 1]]
        bw_n = reg_map[0, 2, pos[:, 0], pos[:, 1]].abs()
        bh_n = reg_map[0, 3, pos[:, 0], pos[:, 1]].abs()

        cx_orig = (cx_n * s - meta["pad_x"]) / meta["scale"]
        cy_orig = (cy_n * s - meta["pad_y"]) / meta["scale"]
        bw_orig = bw_n * s / meta["scale"]
        bh_orig = bh_n * s / meta["scale"]

        x1 = (cx_orig - bw_orig / 2).clamp(min=0.0)
        y1 = (cy_orig - bh_orig / 2).clamp(min=0.0)
        x2 = (cx_orig + bw_orig / 2).clamp(max=float(meta["orig_w"]))
        y2 = (cy_orig + bh_orig / 2).clamp(max=float(meta["orig_h"]))

        # Filter degenerate boxes on GPU before moving to CPU
        valid = (x2 > x1) & (y2 > y1)
        if not valid.any():
            continue

        x1, y1, x2, y2 = x1[valid].tolist(), y1[valid].tolist(), x2[valid].tolist(), y2[valid].tolist()
        scores = scores[valid].tolist()
        cls_idx = cls_idx[valid].tolist()

        detections.extend([
            {
                "class": class_list[int(ci)],
                "score": round(sc, 4),
                "bbox": [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)],
            }
            for sc, ci, bx1, by1, bx2, by2 in zip(scores, cls_idx, x1, y1, x2, y2)
        ])

    return detections


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return (inter / union).item() if union > 0 else 0.0


def nms(detections: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Class-agnostic NMS."""
    if not detections:
        return []
    boxes = torch.tensor([[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]] for d in detections])
    scores = torch.tensor([d["score"] for d in detections])
    try:
        from torchvision.ops import nms as tv_nms
        keep = tv_nms(boxes, scores, iou_thresh).tolist()
    except Exception:
        keep = []
        order = scores.argsort(descending=True).tolist()
        while order:
            i = order.pop(0)
            keep.append(i)
            b = boxes[i]
            order = [j for j in order if _iou(boxes[j], b) < iou_thresh]
    return [detections[i] for i in keep]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run EdgeMusicDetector on a PDF score.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory.")
    parser.add_argument("--config", type=str, default="gelato_config.json")
    parser.add_argument("--output-dir", type=str, default="inference_out")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PDF page rendering.")
    parser.add_argument("--use-bottom-up", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load config ---
    with open(args.config) as f:
        config = json.load(f)
    class_list = config["target_classes"]
    num_classes = len(class_list)

    # --- Load model ---
    logger.info(f"Loading model from {args.checkpoint}")
    model = EdgeMusicDetector(num_classes=num_classes, use_bottom_up=args.use_bottom_up)

    ckpt = Path(args.checkpoint)
    if ckpt.is_dir():
        # Prefer the last HF Trainer checkpoint inside the directory
        last = get_last_checkpoint(str(ckpt))
        if last is not None:
            ckpt = Path(last)
            logger.info(f"  Using last checkpoint: {ckpt}")
        # else: treat the directory itself as the checkpoint (single-run dir)

    # Look for weights file inside the resolved checkpoint directory
    for name in ("model.safetensors", "pytorch_model.bin"):
        weights = (ckpt / name) if ckpt.is_dir() else ckpt
        if weights.exists():
            if weights.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(weights, device=str(device))
            else:
                state = torch.load(weights, map_location=device, weights_only=False)
            model.load_state_dict(state)
            logger.info(f"  Loaded weights: {weights}")
            break
    else:
        logger.warning("No weights file found — using random weights.")
    model.to(device).eval()

    # --- Render PDF ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rendering PDF: {args.pdf}")
    pages = pdf_to_pages(Path(args.pdf), dpi=args.dpi)

    with torch.inference_mode():
        pbar = tqdm(enumerate(pages), total=len(pages), unit="page", desc="Inferring")
        for page_idx, page_image in pbar:
            stem = f"page_{page_idx + 1:04d}"
            img_path = output_dir / f"{stem}.png"
            page_image.save(img_path)

            tensor, meta = preprocess(page_image, args.input_size, device)
            outputs = model(tensor)
            detections = decode_outputs(outputs, meta, args.conf_thresh, class_list)
            detections = nms(detections, args.iou_thresh)
            pbar.set_postfix(dets=len(detections))

            # One JSON per page — matches dataset annotation format
            ann_data = {
                "image": img_path.name,
                "width": page_image.width,
                "height": page_image.height,
                "annotations": [
                    {"class": d["class"], "score": d["score"], "bbox": d["bbox"]}
                    for d in detections
                ],
            }
            json_path = output_dir / f"{stem}.json"
            with open(json_path, "w") as f:
                json.dump(ann_data, f, indent=2)

    logger.info(f"Done. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
