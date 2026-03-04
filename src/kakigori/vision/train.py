"""Modular Training Script for MusicDetector OMR model.

Uses Hugging Face Transformers Trainer for robust, standard implementation.
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
    model = MusicDetector(
        num_classes=num_classes,
        use_bottom_up=args.use_bottom_up,
        out_indices=tuple(args.out_indices),
    )

    if args.fine_tune:
        logger.info(
            f"Loading weights from {args.fine_tune} for fine-tuning (forcing CPU to save VRAM)..."
        )
        device = torch.device("cpu")
        load_checkpoint(model, args.fine_tune, device=device, eval=False)

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
        # --- THE FIX: Cosine Warmup Schedule ---
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # Use the first 10% of training steps to warm up the LR
        # NEW: The Speed & VRAM Cheat Code
        fp16=True,  # Change to bf16=True if you have an RTX 3000/4000 series GPU
        dataloader_pin_memory=True,  # Speeds up CPU-to-GPU data transfer
    )

    # --- Parse scale ranges from flat list into list of tuples ---
    scale_ranges = None
    if args.scale_ranges is not None:
        sr = args.scale_ranges
        scale_ranges = [(sr[0], sr[1]), (sr[2], sr[3]), (sr[4], sr[5])]

    # --- Initialize Trainer ---
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

    # --- Start Training ---
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif args.resume:
        last_checkpoint = get_last_checkpoint(str(run_dir))
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
            checkpoint = last_checkpoint
        else:
            logger.warning(f"No checkpoint found in {run_dir}, starting from scratch.")

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
