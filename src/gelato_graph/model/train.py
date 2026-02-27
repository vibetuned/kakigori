"""Modular Training Script for EdgeMusicDetector OMR model.

Uses Hugging Face Transformers Trainer for robust, standard implementation.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import torch
from transformers import TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from .model import EdgeMusicDetector
from .dataset import OMRDataset
from .collator import omr_collate_fn
from .trainer import OMRTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="OMR Trainer")
    
    # Data args
    parser.add_argument("--img-dir", type=str, default="data/dataset-small-render/imgs")
    parser.add_argument("--ann-dir", type=str, default="data/dataset-small-render/annotations")
    parser.add_argument("--config", type=str, default="gelato_config.json")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Model args
    parser.add_argument("--use-bottom-up", action="store_true")
    
    # Training args (subset of common ones, others can be passed via unknown args if needed)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--logging-dir", type=str, default="runs", help="Root directory for TensorBoard run logs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Auto-detect and resume from the last checkpoint.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a specific checkpoint to resume from.")

    args, unknown = parser.parse_known_args()

    set_seed(args.seed)

    # --- Resolve Output Run Directory ---
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.resume or args.resume_from_checkpoint:
        # Find the latest existing run directory to resume from
        existing_runs = sorted(
            [d for d in output_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda d: d.name,
        )
        if existing_runs:
            run_dir = existing_runs[-1]
            logger.info(f"Resuming from existing run: {run_dir}")
        else:
            run_dir = output_root / "run_001"
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"No existing runs found in {output_root}, creating {run_dir}")
    else:
        # Create a new numbered run directory
        existing_runs = sorted(
            [d for d in output_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
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

    # --- Initialize Model ---
    model = EdgeMusicDetector(
        num_classes=num_classes,
        use_bottom_up=args.use_bottom_up,
    )

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
    )

    # --- Initialize Trainer ---
    trainer = OMRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=omr_collate_fn,
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