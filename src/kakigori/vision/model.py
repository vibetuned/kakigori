# Third party imports
import torch
import torch.nn as nn

# Local folder imports
from .layers import PANetNeck, DecoupledHead
from .regla import VARIANTS, ReGLA

DEFAULT_BACKBONE = "regla_m"
# The 0.1.x detectors (released weights included): ImageNet-pretrained, RGB
LEGACY_BACKBONE = "convnext_base.dinov3_lvd1689m"


def timm_backbone(name: str, out_indices, pretrained: bool = True) -> nn.Module:
    """A pretrained timm feature extractor; grey pages are repeated to RGB.

    The timm module is used as is, so 0.1.x checkpoints (keys under
    ``backbone.stages_0...``) load unchanged.
    """
    # Local import: timm is only needed for the legacy backbones
    import timm

    model = timm.create_model(name, out_indices=out_indices, pretrained=pretrained, features_only=True)
    model.channels = model.feature_info.channels()

    def grey_to_rgb(_module, args):
        x = args[0]
        return (x.expand(-1, 3, -1, -1),) if x.shape[1] == 1 else None

    model.register_forward_pre_hook(grey_to_rgb)
    return model


def make_backbone(backbone: str, out_indices, pretrained: bool = True) -> nn.Module:
    """``regla_t`` ... ``regla_x`` (one grey channel, trained from scratch) or a
    timm model name (e.g. the legacy ConvNeXt DINOv3)."""
    if backbone in VARIANTS:
        return ReGLA(backbone, in_chans=1, out_indices=out_indices)
    return timm_backbone(backbone, out_indices, pretrained=pretrained)


def detect_backbone(state_dict: dict, prefix: str = "") -> str:
    """The backbone a detector checkpoint was trained with, from its keys:
    ReGLA's stage widths and depths, or the legacy timm ConvNeXt."""
    head = f"{prefix}backbone."
    keys = [k[len(head):] for k in state_dict if k.startswith(head)]
    if any(k.startswith(("stages_", "stem_")) for k in keys):
        return LEGACY_BACKBONE
    widths, depths = [], []
    i = 0
    while any(k.startswith(f"stages.{i}.") for k in keys):
        blocks = {k.split(".")[3] for k in keys if k.startswith(f"stages.{i}.blocks.")}
        depths.append(len(blocks))
        # first depthwise conv of the stage's first block: ELRF or attention
        first = f"{head}stages.{i}.blocks.0." + ("dw3.weight" if i < 2 else "cpe.0.weight")
        widths.append(state_dict[first].shape[0])
        i += 1
    for name, (v_depths, v_dims) in VARIANTS.items():
        if tuple(depths) == v_depths[: len(depths)] and tuple(widths) == v_dims[: len(widths)]:
            return name
    raise ValueError(f"checkpoint backbone not recognised (stage depths {depths}, widths {widths})")


class MusicDetector(nn.Module):
    def __init__(self, num_classes=91, use_bottom_up=False, out_indices=(0, 1, 2),
                 backbone=DEFAULT_BACKBONE, pretrained=True):
        super().__init__()
        self.backbone_name = backbone

        # 1. The Encoder: grey pages in, features at strides 4/8/16 out
        self.backbone = make_backbone(backbone, tuple(out_indices), pretrained=pretrained)
        backbone_channels = self.backbone.channels

        # 2. The Neck
        hidden_dim = 128
        self.neck = PANetNeck(
            backbone_channels, hidden_dim, use_bottom_up=use_bottom_up
        )

        # 3. The Heads (One for each feature scale)
        # We use a ModuleList because we need a head for the micro, mid, and macro features
        self.heads = nn.ModuleList(
            [DecoupledHead(hidden_dim, num_classes) for _ in range(3)]
        )

    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.neck(features)

        outputs = []
        for feature, head in zip(fused_features, self.heads):
            cls_out, reg_out = head(feature)
            outputs.append({"cls": cls_out, "reg": reg_out})

        return outputs
