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
    parser.add_argument("--epochs", type=int, default=100)
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
        output_dir=args.output_dir,
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
        logging_dir=f"{args.output_dir}/runs",
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
        if os.path.isdir(args.output_dir):
            last_checkpoint = get_last_checkpoint(args.output_dir)
            if last_checkpoint is not None:
                logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
                checkpoint = last_checkpoint
            else:
                logger.warning(f"No checkpoint found in {args.output_dir}, starting from scratch.")
    
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()