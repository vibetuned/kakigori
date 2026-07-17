"""Modular Training Script for MusicDetector OMR model.

Uses Hugging Face Transformers Trainer for robust, standard implementation.
"""

# Standard library imports
import os
import math
import json
import logging
import argparse
from pathlib import Path

# Third party imports
import torch
from torch import nn
from transformers import TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

# Local folder imports
from .model import MusicDetector
from .utils import RatioSampler, omr_collate_fn, load_checkpoint
from .dataset import OMRDataset
from .trainer import OMRTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    # First, parse configuration file path
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument(
        "--train-config",
        type=str,
        default="conf/train.yaml",
        help="Path to YAML training configuration.",
    )
    conf_args, remaining_argv = conf_parser.parse_known_args()

    yaml_defaults = {}
    if os.path.exists(conf_args.train_config):
        # Third party imports
        import yaml

        with open(conf_args.train_config, "r") as f:
            yaml_defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(description="OMR Trainer", parents=[conf_parser])

    # Data args
    parser.add_argument("--img-dir", type=str, default="data/dataset-small-render/imgs")
    parser.add_argument(
        "--ann-dir", type=str, default="data/dataset-small-render/annotations"
    )
    parser.add_argument(
        "--synthetic-img-dir", type=str, default="data/synthetic-small/img"
    )
    parser.add_argument(
        "--synthetic-ann-dir", type=str, default="data/synthetic-small/annotations"
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=int,
        default=4,
        help="ratio of synthetic to real (default 4)",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="use synthetic data merged with real data",
    )
    parser.add_argument("--config", type=str, default="conf/config.json")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model args
    parser.add_argument("--use-bottom-up", action="store_true")
    parser.add_argument(
        "--out-indices",
        type=int,
        nargs=3,
        default=[1, 2, 3],
        help="Three backbone feature map indices to extract (default: 1 2 3).",
    )

    # --- Class expansion args ---
    parser.add_argument("--old-num-classes", type=int, default=None, help="Number of classes in the pre-trained weights.")
    parser.add_argument("--freeze-epochs", type=int, default=10, help="Number of epochs to train with a frozen backbone.")
    parser.add_argument("--lr-unfrozen", type=float, default=1e-5, help="Learning rate for stage 2 (unfrozen).")

    # Trainer args
    parser.add_argument("--reg-weight", type=float, default=5.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--base-gamma", type=float, default=2.0)
    parser.add_argument("--max-gamma", type=float, default=4.0)
    parser.add_argument(
        "--scale-ranges",
        type=float,
        nargs=6,
        default=None,
        help="Six floats defining 3 (min, max) area ranges for scale assignment, "
        "e.g. 0.0 0.0002 0.0002 0.002 0.002 2.0 (default).",
    )

    # Training args (subset of common ones, others can be passed via unknown args if needed)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--logging-dir",
        type=str,
        default="runs",
        help="Root directory for TensorBoard run logs.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-detect and resume from the last checkpoint.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint to resume from.",
    )
    parser.add_argument(
        "--fine-tune",
        type=str,
        default=None,
        help="Path to a checkpoint to fine-tune from (loads weights only, starts new run).",
    )

    parser.set_defaults(**yaml_defaults)
    args, unknown = parser.parse_known_args(remaining_argv)
    return args


def main():
    args = parse_args()

    set_seed(args.seed)

    # --- Resolve Output Run Directory ---
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.resume or args.resume_from_checkpoint:
        # Find the latest existing run directory to resume from
        existing_runs = sorted(
            [
                d
                for d in output_root.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ],
            key=lambda d: d.name,
        )
        if existing_runs:
            run_dir = existing_runs[-1]
            logger.info(f"Resuming from existing run: {run_dir}")
        else:
            run_dir = output_root / "run_001"
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"No existing runs found in {output_root}, creating {run_dir}"
            )
    else:
        # Create a new numbered run directory
        existing_runs = sorted(
            [
                d
                for d in output_root.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ],
            key=lambda d: d.name,
        )
        if existing_runs:
            last_num = int(existing_runs[-1].name.split("_")[1])
            next_num = last_num + 1
        else:
            next_num = 1
        run_dir = output_root / f"run_{next_num:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created new run directory: {run_dir}")

    # Set TensorBoard logging dir via env var (logging_dir kwarg is deprecated)
    os.environ["TENSORBOARD_LOGGING_DIR"] = args.logging_dir + "/" + run_dir.name

    # --- Load Configuration ---
    with open(args.config) as f:
        config = json.load(f)
    class_list = config["target_classes"]
    num_classes = len(class_list)
    logger.info(f"Training with {num_classes} classes from {args.config}")

    # --- Prepare Datasets ---
    full_ds = OMRDataset(
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        class_list=class_list,
        input_size=args.input_size,
        augment=True,
    )

    train_dataset = torch.utils.data.Subset(full_ds, range(len(full_ds)))

    custom_sampler = None
    if args.use_synthetic:
        logger.info(f"Loading synthetic dataset with ratio 1:{args.synthetic_ratio}...")
        synthetic_ds = OMRDataset(
            img_dir=args.synthetic_img_dir,
            ann_dir=args.synthetic_ann_dir,
            class_list=class_list,
            input_size=args.input_size,
            augment=True,
        )
        synth_subset = torch.utils.data.Subset(synthetic_ds, range(len(synthetic_ds)))

        if args.synthetic_ratio > 0:
            train_dataset = torch.utils.data.ConcatDataset(
                [synth_subset, train_dataset]
            )
            dataset_lengths = [len(synthetic_ds), len(full_ds)]
            ratios = [1, args.synthetic_ratio]
            custom_sampler = RatioSampler(dataset_lengths, ratios)
            logger.info(f"Using synthetic data with ratio 1:{args.synthetic_ratio}")

        elif args.synthetic_ratio < 0:
            train_dataset = torch.utils.data.ConcatDataset(
                [train_dataset, synth_subset]
            )
            dataset_lengths = [len(full_ds), len(synthetic_ds)]
            ratio = abs(args.synthetic_ratio)
            ratios = [ratio, 1]
            custom_sampler = RatioSampler(dataset_lengths, ratios)
            logger.info(f"Using synthetic data with ratio {ratio}:1")

        else:
            logger.warning("Invalid synthetic ratio. Using real data only.")

    # --- Initialize Model ---
    # 1. Initialize with old architecture size
    model = MusicDetector(
        num_classes=args.old_num_classes,
        use_bottom_up=args.use_bottom_up,
        out_indices=tuple(args.out_indices),
    )
    device = torch.device("cpu")
    load_checkpoint(model, args.fine_tune, device=device, eval=False)

    # 2. Surgically expand the classification heads
    in_channels = 128
    for head in model.heads:
        old_conv = head.cls_branch[-1]
        new_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        with torch.no_grad():
            new_conv.weight[:args.old_num_classes] = old_conv.weight
            new_conv.bias[:args.old_num_classes] = old_conv.bias
            
            # Apply focal bias prior to the new classes
            pi = 0.01
            focal_bias = -math.log((1.0 - pi) / pi)
            new_conv.bias[args.old_num_classes:].fill_(focal_bias)
            
        head.cls_branch[-1] = new_conv

    # 3. Freeze backbone and neck
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.neck.parameters():
        param.requires_grad = False

    # --- Initialize Training Arguments ---
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,  # Important!
        report_to="tensorboard",
        save_total_limit=3,
        logging_first_step=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # Use the first 10% of training steps to warm up the LR
        bf16=True,  # Change to bf16=True if you have an RTX 3000/4000 series GPU
        dataloader_pin_memory=True,  # Speeds up CPU-to-GPU data transfer
    )

    # --- Parse scale ranges from flat list into list of tuples ---
    scale_ranges = None
    if args.scale_ranges is not None:
        sr = args.scale_ranges
        scale_ranges = [(sr[0], sr[1]), (sr[2], sr[3]), (sr[4], sr[5])]

    # --- Stage-aware crash resume ---
    # The supervisor (train.sh) relaunches after segfaults; without resume a
    # crash restarts the whole two-stage schedule. A marker file records
    # that stage 2 began; each stage resumes from its own checkpoints, and
    # partial checkpoints (no trainer_state.json) are discarded.
    import shutil

    def _last_complete_checkpoint(directory):
        ckpt = get_last_checkpoint(str(directory))
        while ckpt is not None and not (Path(ckpt) / "trainer_state.json").exists():
            logger.warning(f"Discarding incomplete checkpoint: {ckpt}")
            shutil.rmtree(ckpt)
            ckpt = get_last_checkpoint(str(directory))
        return ckpt

    stage2_marker = run_dir / "STAGE2_STARTED"
    stage2_dir = run_dir / "stage2"

    if not stage2_marker.exists():
        logger.info(f"--- STAGE 1: Training frozen network for {args.freeze_epochs} epochs ---")
        training_args.num_train_epochs = args.freeze_epochs

        trainer = OMRTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=omr_collate_fn,
            scale_ranges=scale_ranges,
            base_gamma=args.base_gamma,
            max_gamma=args.max_gamma,
            custom_sampler=custom_sampler,
        )
        trainer.train(resume_from_checkpoint=_last_complete_checkpoint(run_dir))
        stage2_marker.touch()
    else:
        logger.info("--- STAGE 1 already completed (marker found), skipping ---")
        # Carry the latest weights forward: stage-2 checkpoint if one exists,
        # otherwise the last stage-1 checkpoint
        stage2_ckpt = _last_complete_checkpoint(stage2_dir) if stage2_dir.exists() else None
        if stage2_ckpt is None:
            load_checkpoint(model, str(run_dir), device=torch.device("cpu"), eval=False)

    logger.info("--- STAGE 2: Unfreezing network for fine-tuning ---")
    for param in model.parameters():
        param.requires_grad = True

    remaining_epochs = args.epochs - args.freeze_epochs
    training_args.num_train_epochs = remaining_epochs
    training_args.learning_rate = args.lr_unfrozen
    stage2_dir.mkdir(parents=True, exist_ok=True)
    training_args.output_dir = str(stage2_dir)

    # We must re-instantiate the Trainer so it builds a new optimizer with the unfrozen parameters
    trainer_unfrozen = OMRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=omr_collate_fn,
        scale_ranges=scale_ranges,
        base_gamma=args.base_gamma,
        max_gamma=args.max_gamma,
        custom_sampler=custom_sampler,
    )
    trainer_unfrozen.train(resume_from_checkpoint=_last_complete_checkpoint(stage2_dir))
    trainer_unfrozen.save_model()


if __name__ == "__main__":
    main()
