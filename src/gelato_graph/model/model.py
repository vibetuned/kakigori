import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .layers import PANetNeck, DecoupledHead    

class EdgeMusicDetector(nn.Module):
    def __init__(self, num_classes=91, use_bottom_up=False):
        super().__init__()
        
        # 1. The Encoder
        self.backbone = timm.create_model(
            'mobilenetv5_300m.gemma3n', #'convnext_base.dinov3_lvd1689m', 
            pretrained=True, 
            features_only=True,
            #out_indices=(1, 2, 3) # Adjust indices based on desired feature resolution for dino models 
            out_indices=(2, 3, 4) # Adjust indices based on desired feature resolution
        )
        
        # Calculate the actual channel outputs of those specific layers
        dummy_input = torch.randn(1, 3, 256, 256)
        backbone_channels = [f.shape[1] for f in self.backbone(dummy_input)]
        
        # 2. The Neck
        hidden_dim = 128
        self.neck = PANetNeck(backbone_channels, hidden_dim, use_bottom_up=use_bottom_up)
        
        # 3. The Heads (One for each feature scale)
        # We use a ModuleList because we need a head for the micro, mid, and macro features
        self.heads = nn.ModuleList([
            DecoupledHead(hidden_dim, num_classes) for _ in range(3)
        ])

    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.neck(features)
        
        outputs = []
        for feature, head in zip(fused_features, self.heads):
            cls_out, reg_out = head(feature)
            outputs.append({'cls': cls_out, 'reg': reg_out})
            
        return outputs