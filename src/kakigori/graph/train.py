"""Modular Training Script for GNN Phase 2 Model.

Uses Hugging Face Transformers Trainer for robust, standard implementation.
"""

import os
import json
import logging
import argparse
from pathlib import Path

import torch
from transformers import TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

# Import your components
from .trainer import GNNTrainer, compute_gnn_metrics
from .dataset import OMRFullPageDataset
from .utils import omr_collate_fn
from .model import GraphVisualExtractor, ScoreGraphReconstructor, GNNPhase2Model
from kakigori.vision.utils import load_checkpoint
from kakigori.vision.model import MusicDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    # First, parse configuration file path
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument(
        "--train-config",
        type=str,
        default="conf/train_gnn.yaml",
        help="Path to YAML training configuration.",
    )
    conf_args, remaining_argv = conf_parser.parse_known_args()

    yaml_defaults = {}
    if os.path.exists(conf_args.train_config):
        import yaml
        with open(conf_args.train_config, "r") as f:
            yaml_defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(description="GNN Phase 2 Trainer", parents=[conf_parser])
    
    # Data args
    parser.add_argument("--img-dir", type=str, default="data/output_imgs")
    parser.add_argument("--ann-dir", type=str, default="data/output_annotations")
    parser.add_argument("--graph-dir", type=str, default="data/output_graphs")
    parser.add_argument("--config", type=str, default="conf/gelato_config.json")
    
    # Pre-trained Vision Backbone args
    parser.add_argument("--detector-checkpoint", type=str, help="Path to pre-trained Detector weights")
    
    # Trainer args
    parser.add_argument("--output-dir", type=str, default="checkpoints_gnn")
    parser.add_argument("--logging-dir", type=str, default="runs_gnn", help="Root directory for TensorBoard run logs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2) # 2 is standard for full-page graphs
    parser.add_argument("--lr", type=float, default=1e-3)    # 1e-3 is ideal for GATv2 initialization
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Auto-detect and resume from the last checkpoint.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a specific checkpoint to resume from.")
    parser.add_argument("--fine-tune", type=str, default=None, help="Path to a checkpoint to fine-tune from.")

    parser.set_defaults(**yaml_defaults)
    args, unknown = parser.parse_known_args(remaining_argv)
    return args

def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve Output Run Directory
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
            logger.warning(f"No existing runs found in {output_root}, creating {run_dir}")
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

    # Load Configuration
    with open(args.config) as f:
        config = json.load(f)
    class_list = config["target_classes"]
    num_classes = len(class_list)
    # Prepare Datasets
    full_ds = OMRFullPageDataset(
        img_dir=args.img_dir,
        json_dir=args.ann_dir,
        graph_dir=args.graph_dir,
        class_list=class_list
    )
    
    # Split 90/10 for Train/Val
    train_size = int(0.95 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_ds, [train_size, val_size])

    # Initialize Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detector = MusicDetector(
        num_classes=num_classes,
    )

    if args.detector_checkpoint:
        logger.info(
            f"Loading weights from {args.detector_checkpoint} for fine-tuning (forcing CPU to save VRAM)..."
        )
        c_device = torch.device("cpu")
        load_checkpoint(detector, args.detector_checkpoint, device=c_device, eval=False)
    else:
        raise ValueError("Please provide a path to the pre-trained Detector weights.")
    
    roi_extractor = GraphVisualExtractor()
    
    # In train.py, update the GNN initialization:
    class_embed_dim = 32
    node_in_dim = 256 + 4 + class_embed_dim # RoI (256) + Coords (4) + ClassEmbed (32)
    
    gnn_model = ScoreGraphReconstructor(
        node_in_dim=node_in_dim, 
        num_classes=num_classes,
        class_embed_dim=class_embed_dim
    )
    
    # Wrap them for the Trainer
    model_wrapper = GNNPhase2Model(detector, roi_extractor, gnn_model)

    if args.fine_tune:
        logger.info(f"Loading weights from {args.fine_tune} for fine-tuning...")
        # Since only the GNN is trainable, we load weights for the whole wrapper
        # The vision model is already loaded and frozen
        model_wrapper.load_state_dict(torch.load(args.fine_tune, map_location=device), strict=False)

    # Initialize Training Arguments
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_steps=1000,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        seed=args.seed,
        remove_unused_columns=False, # Essential so the Trainer doesn't delete the 'edges' column
        report_to="tensorboard",
        save_total_limit=3,
        logging_first_step=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True, 
        dataloader_pin_memory=True,
    )

    # Define Focal Loss class weights
    alpha_weights = torch.tensor([0.05, 1.0, 2.5, 1.0, 4.0], dtype=torch.float32)
    alpha_weights = alpha_weights.to(device)

    # Initialize Custom Trainer
    trainer = GNNTrainer(
        alpha_weights=alpha_weights,
        model=model_wrapper,
        class_list=class_list,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=omr_collate_fn,
        compute_metrics=compute_gnn_metrics,
    )

    # Start Training
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

    logger.info("Starting Phase 2 Graph Training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save the final GNN weights independently of the wrapper
    torch.save(gnn_model.state_dict(), run_dir / "gnn_final.pth")

if __name__ == "__main__":
    main()