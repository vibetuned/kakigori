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
from .model import GraphVisualExtractor, ScoreGraphReconstructor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="GNN Phase 2 Trainer")
    
    # Data args
    parser.add_argument("--img-dir", type=str, default="data/output_imgs")
    parser.add_argument("--ann-dir", type=str, default="data/output_annotations")
    parser.add_argument("--graph-dir", type=str, default="data/output_graphs")
    parser.add_argument("--config", type=str, default="conf/gelato_config.json")
    
    # Pre-trained Vision Backbone args
    parser.add_argument("--roi-checkpoint", type=str, required=True, help="Path to pre-trained RoI Extractor weights")
    
    # Trainer args
    parser.add_argument("--output-dir", type=str, default="checkpoints_gnn")
    parser.add_argument("--logging-dir", type=str, default="runs_gnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2) # 2 is standard for full-page graphs
    parser.add_argument("--lr", type=float, default=1e-3)    # 1e-3 is ideal for GATv2 initialization
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve Output Run Directory
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    # (Assuming the run_dir resolution logic from your train.py is applied here)
    run_dir = output_root / "run_001" 
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load Configuration
    with open(args.config) as f:
        config = json.load(f)
    class_list = config["target_classes"]
    
    # Prepare Datasets
    full_ds = OMRFullPageDataset(
        img_dir=args.img_dir,
        json_dir=args.ann_dir,
        graph_dir=args.graph_dir,
        class_list=class_list
    )
    
    # Split 90/10 for Train/Val
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_ds, [train_size, val_size])

    # Initialize Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    roi_extractor = GraphVisualExtractor()
    roi_extractor.load_state_dict(torch.load(args.roi_checkpoint, map_location=device))
    
    node_in_dim = 256 + 4 + 32 # RoI + Coords + ClassEmbed
    gnn_model = ScoreGraphReconstructor(node_in_dim=node_in_dim)
    
    # Wrap them for the Trainer
    model_wrapper = GNNPhase2Model(roi_extractor, gnn_model)

    # Initialize Training Arguments
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        seed=args.seed,
        remove_unused_columns=False, # Essential so the Trainer doesn't delete the 'edges' column
        report_to="tensorboard",
        save_total_limit=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True, 
        dataloader_pin_memory=True,
    )

    # Define Focal Loss class weights
    alpha_weights = torch.tensor([0.05, 1.0, 2.5, 1.0, 4.0], dtype=torch.float32)

    # Initialize Custom Trainer
    trainer = GNNTrainer(
        alpha_weights=alpha_weights,
        model=model_wrapper,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=omr_collate_fn,
        compute_metrics=compute_gnn_metrics,
    )

    # Start Training
    logger.info("Starting Phase 2 Graph Training...")
    trainer.train()
    
    # Save the final GNN weights independently of the wrapper
    torch.save(gnn_model.state_dict(), run_dir / "gnn_final.pth")

if __name__ == "__main__":
    main()