# Standard library imports
import unittest
from unittest.mock import MagicMock

# Third party imports
import torch
import torch.nn as nn
from transformers import TrainingArguments

# First party imports
from kakigori.vision.trainer import OMRTrainer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                "cls_branch": nn.Sequential(nn.Conv2d(1, 91, 1))
            })
        ])
    def forward(self, x):
        return [{
            "cls": torch.randn(x.size(0), 91, 10, 10),
            "reg": torch.randn(x.size(0), 4, 10, 10)
        }]

class TestOMRTrainer(unittest.TestCase):
    def test_compute_loss_logging(self):
        # 1. Setup minimal model and trainer
        model = DummyModel()
        
        args = TrainingArguments(output_dir="/tmp/test_trainer", report_to="none")
        trainer = OMRTrainer(model=model, args=args)
        
        # Mock trainer control to trigger logging
        trainer.control = MagicMock()
        trainer.control.should_log = True
        trainer.log = MagicMock()
        
        # 2. Prepare mock inputs
        inputs = {
            "pixel_values": torch.randn(1, 3, 320, 320),
            "labels": [{
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "labels": torch.tensor([0])
            }]
        }
        
        # 3. Call compute_loss
        loss = trainer.compute_loss(model, inputs)
        
        # 4. Verify self.log was called
        trainer.log.assert_called_once()
        log_call_args = trainer.log.call_args[0][0]
        self.assertIn("train/cls_loss", log_call_args)
        self.assertIn("train/reg_loss", log_call_args)
        print("\nSuccess: self.log was called with correct keys.")

if __name__ == "__main__":
    unittest.main()
