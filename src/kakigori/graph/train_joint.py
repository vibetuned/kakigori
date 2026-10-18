"""Joint detection + graph training (stage C of the 0.2.0 roadmap).

ONE model, TWO losses on the SAME forward pass:

    L = det_weight * (cls_w * L_focal + reg_w * L_ciou)      [per page]
      + edge_weight * mean_systems(L_edge)                    [per system]

The backbone+neck features are computed once per page; the detection heads
consume them for the detection loss while the RoI extractor + GATv2 consume
them for the per-system edge loss (accumulated over every system, then
backpropagated together — the detection loss anchors the shared features,
which is what makes unfreezing safe and kills the two-neck deployment).

Two variants, selected by `unfreeze`:
  - "neck": PANet neck + detection heads + GNN train, ConvNeXt frozen
  - "full": everything trains

Warm start (`fine_tune`) from a GNN wrapper checkpoint (its detector copy
already carries the adapted neck); detection heads that were frozen during
the GNN phases resume training here.

Usage:
    train-joint --train-config conf-lite/train_joint_neck.yaml
"""

# Standard library imports
import os
import json
import logging
import argparse
from pathlib import Path

# Third party imports
import torch
from transformers import TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

# Local imports
from .trainer import GNNTrainer, compute_gnn_metrics
from .dataset import OMRFullPageDataset
from .utils import omr_collate_fn, split_into_systems
from .heuristics import generate_axis_aware_edges, map_gt_to_candidates
from .model import GraphVisualExtractor, ScoreGraphReconstructor, GNNPhase2Model
from kakigori.vision.trainer import OMRTrainer
from kakigori.vision.losses import DynamicFocalLoss, CIoULoss
from kakigori.vision.utils import load_checkpoint
from kakigori.vision.model import MusicDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class JointModel(GNNPhase2Model):
    """GNN wrapper whose detection HEADS also train — they carry the
    detection loss that anchors the shared backbone/neck features."""

    def __init__(self, detector, roi_extractor, gnn, unfreeze="neck"):
        super().__init__(detector, roi_extractor, gnn, unfreeze=unfreeze)
        for p in self.detector.heads.parameters():
            p.requires_grad = True


class JointTrainer(GNNTrainer):
    """Extends the per-system edge loss with the page-level detection loss."""

    def __init__(self, *args, det_weight=0.1, edge_weight=10.0,
                 cls_weight=1.0, reg_weight=5.0, base_gamma=2.0,
                 max_gamma=3.0, scale_ranges=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.det_weight = det_weight
        self.edge_weight = edge_weight
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.scale_ranges = (
            scale_ranges if scale_ranges is not None
            else OMRTrainer.DEFAULT_SCALE_RANGES
        )
        self.det_focal = DynamicFocalLoss(base_gamma=base_gamma, max_gamma=max_gamma)
        self.det_ciou = CIoULoss()

    def log(self, logs: dict, start_time: float | None = None) -> None:
        """Merge the per-loss components (train/det_cls, train/det_reg,
        train/edge_loss) into every Trainer log call — GNNTrainer has no
        such override, so without this they are set and silently dropped."""
        if getattr(self, "_custom_metrics", None):
            logs.update(self._custom_metrics)
            self._custom_metrics = {}
        super().log(logs, start_time=start_time)

    def _detection_loss(self, model, fused_features, boxes_list, labels_list,
                        input_size, device):
        """Detection loss on already-computed neck features (per page batch)."""
        outputs = []
        for feature, head in zip(fused_features, model.detector.heads):
            cls_out, reg_out = head(feature)
            outputs.append({"cls": cls_out, "reg": reg_out})

        # GNN dataset boxes are absolute xyxy on the letterboxed canvas;
        # build_targets wants normalized cxcywh
        targets_list = []
        for boxes, labels in zip(boxes_list, labels_list):
            if boxes.numel():
                cxcywh = torch.stack([
                    (boxes[:, 0] + boxes[:, 2]) / 2.0 / input_size,
                    (boxes[:, 1] + boxes[:, 3]) / 2.0 / input_size,
                    (boxes[:, 2] - boxes[:, 0]) / input_size,
                    (boxes[:, 3] - boxes[:, 1]) / input_size,
                ], dim=1)
            else:
                cxcywh = boxes.new_zeros((0, 4))
            targets_list.append({"boxes": cxcywh, "labels": labels})

        num_classes = model.detector.heads[0].cls_branch[-1].out_channels
        feature_sizes = [(o["cls"].shape[2], o["cls"].shape[3]) for o in outputs]
        cls_targets, reg_targets, obj_masks = OMRTrainer.build_targets(
            self, targets_list, feature_sizes, num_classes, device
        )

        max_steps = self.state.max_steps
        progress = self.state.global_step / max_steps if max_steps > 0 else 0.0

        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)
        for scale_idx, out in enumerate(outputs):
            total_cls = total_cls + self.det_focal(
                out["cls"], cls_targets[scale_idx], progress=progress
            )
            mask = obj_masks[scale_idx]
            if mask.any():
                pos = mask.nonzero(as_tuple=False)
                b_idx, r_idx, c_idx = pos[:, 0], pos[:, 1], pos[:, 2]
                pred_pos = out["reg"][b_idx, :, r_idx, c_idx]
                tgt_pos = reg_targets[scale_idx][b_idx, :, r_idx, c_idx]
                fw, fh = out["reg"].shape[3], out["reg"].shape[2]
                pred_cx = (pred_pos[:, 0].sigmoid() + c_idx) / fw
                pred_cy = (pred_pos[:, 1].sigmoid() + r_idx) / fh
                pred_w = torch.exp(pred_pos[:, 2].clamp(max=10.0))
                pred_h = torch.exp(pred_pos[:, 3].clamp(max=10.0))
                tgt_cx = (tgt_pos[:, 0] + c_idx) / fw
                tgt_cy = (tgt_pos[:, 1] + r_idx) / fh
                pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)
                tgt_boxes = torch.stack([tgt_cx, tgt_cy, tgt_pos[:, 2], tgt_pos[:, 3]], dim=1)
                total_reg = total_reg + self.det_ciou(pred_boxes, tgt_boxes)

        return self.cls_weight * total_cls + self.reg_weight * total_reg, total_cls, total_reg

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        images = inputs["images"]
        boxes_list = inputs["boxes"]
        labels_list = inputs["labels"]
        edges_list = inputs["edges"]
        # Detection supervision needs the FULL annotation set: graph-filtered
        # boxes lack classes that never become graph nodes (note, barlines) —
        # supervising on them unlearns those classes (note AP 0.93 -> 0.01)
        det_boxes_list = inputs.get("det_boxes", boxes_list)
        det_labels_list = inputs.get("det_labels", labels_list)
        device = self.args.device
        class_to_idx = {c: i for i, c in enumerate(self.class_list)}
        unfreeze = getattr(model, "unfreeze", "neck")

        edge_loss = 0.0
        det_loss_sum = torch.tensor(0.0, device=device)
        det_cls_sum, det_reg_sum = 0.0, 0.0
        all_logits, all_targets = [], []
        valid_systems, n_pages = 0, 0

        for i in range(len(images)):
            img = images[i].to(device)
            abs_boxes = boxes_list[i].to(device)
            labels = labels_list[i].to(device)
            page_edges = edges_list[i].to(device)
            if len(abs_boxes) < 2:
                continue

            # Jitter feeds ONLY the GNN branch (RoI robustness). The
            # detection targets must stay exact: jittered regression targets
            # destroyed localization (mAP@.75 0.58 -> 0.21) in the first
            # joint run before this split.
            clean_boxes = abs_boxes
            if model.training and self.box_jitter > 0:
                abs_boxes = abs_boxes + torch.randn_like(abs_boxes) * self.box_jitter
                abs_boxes[:, 2:] = torch.maximum(abs_boxes[:, 2:], abs_boxes[:, :2] + 1.0)

            # Shared features — grads per unfreeze mode
            if unfreeze == "full":
                features = model.detector.backbone(img.unsqueeze(0))
            else:
                with torch.no_grad():
                    features = model.detector.backbone(img.unsqueeze(0))
            fused_features = model.detector.neck(features)
            feat_dict = {str(idx): feat for idx, feat in enumerate(fused_features)}

            # --- Detection loss (page-level, anchors the shared features)
            input_size = img.shape[-1]
            det_loss, det_cls, det_reg = self._detection_loss(
                model, fused_features,
                [det_boxes_list[i].to(device)], [det_labels_list[i].to(device)],
                input_size, device,
            )
            det_loss_sum = det_loss_sum + det_loss
            det_cls_sum += float(det_cls.detach())
            det_reg_sum += float(det_reg.detach())
            n_pages += 1

            # --- Edge loss (accumulated over every system on the page)
            for sys_data in split_into_systems(abs_boxes, labels, page_edges, class_to_idx):
                sys_abs_boxes = sys_data["abs_boxes"]
                sys_labels = sys_data["labels"]
                if len(sys_abs_boxes) < 2:
                    continue
                roi_feats = model.roi_extractor(feat_dict, [sys_abs_boxes], img.shape[-2:])
                sx1, sy1, _, _ = sys_data["system_bbox"]
                sys_rel_boxes = sys_abs_boxes.clone()
                sys_rel_boxes[:, [0, 2]] -= sx1
                sys_rel_boxes[:, [1, 3]] -= sy1
                norm_boxes = sys_rel_boxes / img.shape[-1]
                class_embeds = model.gnn.class_embedding(sys_labels)
                x = torch.cat([roi_feats, norm_boxes, class_embeds], dim=1)
                candidates = generate_axis_aware_edges(sys_rel_boxes, sys_labels, class_to_idx)
                if candidates.shape[1] == 0:
                    continue
                y_targets = map_gt_to_candidates(candidates, sys_data["edge_targets"])
                edge_logits = model.gnn(x, candidates)
                edge_loss = edge_loss + self.loss_fn(edge_logits, y_targets)
                valid_systems += 1

                if return_outputs or not model.training:
                    all_logits.append(edge_logits)
                    all_targets.append(y_targets)
                    if not model.training:
                        self.topology_evaluator.update(
                            edge_index=candidates, gt_edges=y_targets,
                            pred_edges=torch.argmax(edge_logits, dim=1),
                            num_nodes=len(sys_abs_boxes),
                        )

        if valid_systems > 0:
            edge_loss = edge_loss / valid_systems
        else:
            edge_loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0
        if n_pages > 0:
            det_loss_sum = det_loss_sum / n_pages

        total_loss = self.edge_weight * edge_loss + self.det_weight * det_loss_sum

        self._custom_metrics = {
            "train/det_cls": det_cls_sum / max(1, n_pages),
            "train/det_reg": det_reg_sum / max(1, n_pages),
            "train/edge_loss": float(edge_loss.detach()) if torch.is_tensor(edge_loss) else edge_loss,
        }

        if return_outputs:
            if all_logits:
                outputs = {"logits": torch.cat(all_logits, dim=0),
                           "labels": torch.cat(all_targets, dim=0)}
            else:
                outputs = {"logits": torch.empty((0, 6), device=device),
                           "labels": torch.empty(0, device=device)}
            return total_loss, outputs
        return total_loss


def parse_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--train-config", type=str,
                             default="conf-lite/train_joint_neck.yaml")
    conf_args, remaining = conf_parser.parse_known_args()
    yaml_defaults = {}
    if os.path.exists(conf_args.train_config):
        import yaml
        with open(conf_args.train_config) as f:
            yaml_defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(description="Joint detection+graph trainer",
                                     parents=[conf_parser])
    parser.add_argument("--img-dir", type=str, required=False)
    parser.add_argument("--ann-dir", type=str, required=False)
    parser.add_argument("--graph-dir", type=str, required=False)
    parser.add_argument("--config", type=str, default="conf-lite/config.json")
    parser.add_argument("--detector-checkpoint", type=str, default=None)
    parser.add_argument("--fine-tune", type=str, default=None,
                        help="GNN wrapper checkpoint to warm-start from")
    parser.add_argument("--unfreeze", type=str, default="neck", choices=["neck", "full"])
    parser.add_argument("--vision-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4,
                        help="LR for the detection heads (re-aligning with the adapted neck)")
    parser.add_argument("--box-jitter", type=float, default=0.0)
    parser.add_argument("--det-weight", type=float, default=0.1)
    parser.add_argument("--edge-weight", type=float, default=10.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=5.0)
    parser.add_argument("--base-gamma", type=float, default=2.0)
    parser.add_argument("--max-gamma", type=float, default=3.0)
    parser.add_argument("--output-dir", type=str, default="checkpoints_joint")
    parser.add_argument("--logging-dir", type=str, default="runs_joint")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=2500)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(**yaml_defaults)
    args, _ = parser.parse_known_args(remaining)
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = sorted(d for d in output_root.iterdir()
                  if d.is_dir() and d.name.startswith("run_"))
    if args.resume and runs:
        run_dir = runs[-1]
        logger.info(f"Resuming from existing run: {run_dir}")
    else:
        n = int(runs[-1].name.split("_")[1]) + 1 if runs else 1
        run_dir = output_root / f"run_{n:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created new run directory: {run_dir}")

    os.environ["TENSORBOARD_LOGGING_DIR"] = args.logging_dir + "/" + run_dir.name

    with open(args.config) as f:
        cfg = json.load(f)
    class_list = cfg["target_classes"]
    class_remap = cfg.get("class_remap") or {}
    num_classes = len(class_list)
    if class_remap:
        logger.info(f"class_remap active: {len(class_remap)} classes -> "
                    f"{sorted(set(class_remap.values()))}")

    full_ds = OMRFullPageDataset(
        img_dir=args.img_dir, json_dir=args.ann_dir,
        graph_dir=args.graph_dir, class_list=class_list,
        class_remap=class_remap,
    )
    train_size = int(0.95 * len(full_ds))
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_ds, [train_size, len(full_ds) - train_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MusicDetector(num_classes=num_classes)
    if args.detector_checkpoint:
        load_checkpoint(detector, args.detector_checkpoint,
                        device=torch.device("cpu"), eval=False)

    gnn = ScoreGraphReconstructor(
        node_in_dim=256 + 4 + 32, num_classes=num_classes, class_embed_dim=32)
    model = JointModel(detector, GraphVisualExtractor(), gnn, unfreeze=args.unfreeze)

    if args.fine_tune:
        logger.info(f"Warm-starting wrapper from {args.fine_tune}")
        load_checkpoint(model, args.fine_tune, device=torch.device("cpu"), eval=False)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Joint model ({args.unfreeze}): {n_train/1e6:.1f}M trainable params, "
                f"det_weight={args.det_weight}, edge_weight={args.edge_weight}")

    training_args = TrainingArguments(
        output_dir=str(run_dir), run_name=run_dir.name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="steps", eval_steps=args.eval_steps,
        logging_steps=args.logging_steps, save_steps=args.save_steps,
        max_steps=args.max_steps, seed=args.seed,
        remove_unused_columns=False, report_to="tensorboard",
        save_total_limit=3, logging_first_step=True,
        lr_scheduler_type="cosine", warmup_ratio=0.1,
        bf16=True, dataloader_pin_memory=True,
    )

    # Three LR groups: GNN head at full lr; detection HEADS at a mid lr
    # (they start misaligned with the GNN-adapted neck and must re-converge
    # within the run); backbone/neck at the gentle vision lr.
    head_params = [p for p in model.detector.heads.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    vision_params = [p for p in model.detector.parameters()
                     if p.requires_grad and id(p) not in head_ids]
    gnn_params = list(model.gnn.parameters()) + list(model.roi_extractor.parameters())
    optimizer = torch.optim.AdamW([
        {"params": gnn_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
        {"params": vision_params, "lr": args.vision_lr},
    ])

    alpha_weights = torch.tensor(
        [0.4, 1.0, 2.0, 1.0, 3.0, 1.5], dtype=torch.float32
    ).to(device)  # same as train-gnn
    trainer = JointTrainer(
        alpha_weights,
        class_list,
        args.box_jitter,
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=omr_collate_fn,
        compute_metrics=compute_gnn_metrics,
        det_weight=args.det_weight, edge_weight=args.edge_weight,
        cls_weight=args.cls_weight, reg_weight=args.reg_weight,
        base_gamma=args.base_gamma, max_gamma=args.max_gamma,
        optimizers=(optimizer, None),
    )

    resume_ckpt = None
    if args.resume:
        # Walk checkpoints newest-to-oldest and take the first VALID one:
        # trainer_state.json present AND model.safetensors deserializable.
        # A power loss mid-save leaves a truncated safetensors next to a
        # complete trainer_state.json ("header too small" crash-loop).
        from safetensors import safe_open
        candidates = sorted(
            (d for d in run_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")),
            key=lambda d: int(d.name.split("-")[-1]), reverse=True,
        )
        for cand in candidates:
            if not (cand / "trainer_state.json").exists():
                continue
            try:
                with safe_open(cand / "model.safetensors", framework="pt"):
                    pass
            except Exception as e:
                logger.warning(f"Skipping corrupt checkpoint {cand.name}: {e}")
                continue
            resume_ckpt = str(cand)
            logger.info(f"Checkpoint detected, resuming training at {resume_ckpt}.")
            break

    logger.info("Starting Joint Detection+Graph Training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model()
    torch.save(model.gnn.state_dict(), run_dir / "gnn_final.pth")


if __name__ == "__main__":
    main()
