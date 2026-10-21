# Third party imports
import torch
import torch.nn as nn

# Local folder imports
from .layers import PANetNeck, DecoupledHead
from .regla import VARIANTS, ReGLA
from .dataset import canvas_size

DEFAULT_BACKBONE = "regla_m"

# The box areas each detection level is trained on, finest level first, as
# fractions of a 640x640 canvas (pixel area / 640**2): plain normalized areas
# on the square 640 canvas, the same sizes in pixels on any other. Four
# levels add stride 32 for the page-scale boxes (system-staff, system and
# page-margin start at ~1e-2, where measure/staff/layer end)
DEFAULT_SCALE_RANGES = {
    3: ((0.0, 0.0001), (0.0001, 0.001), (0.001, 2.0)),
    4: ((0.0, 0.0001), (0.0001, 0.001), (0.001, 0.01), (0.01, 2.0)),
}
# The 0.1.x detectors (released weights included): ImageNet-pretrained, RGB
LEGACY_BACKBONE = "convnext_base.dinov3_lvd1689m"

# A target cell regresses one box per slot. With a single slot, 21% of the
# lite training boxes shared their cell with another box of their level and
# lost their regression target to it (measure/staff/layer of one staff,
# a notehead under its note, a beam centred on a note, keySig over its one
# accidental, system over its only row). Structure classes get a slot each,
# composites (boxes drawn around other annotated glyphs) one together, and
# every other glyph slot 0: 0.4% of the boxes still share a slot.
STRUCTURE_SLOT_CLASSES = ("system", "system-staff", "measure", "staff", "layer", "page-margin")
COMPOSITE_SLOT_CLASSES = ("note", "chord", "keySig", "meterSig", "mRest", "tuplet")


def regression_slots(class_list) -> list[int]:
    """Each class's regression slot (see STRUCTURE_SLOT_CLASSES): numbered
    over the classes present, slot 0 for the plain glyphs."""
    slots = {}
    for name in STRUCTURE_SLOT_CLASSES:
        if name in class_list:
            slots[name] = len(slots) + 1
    composite = len(slots) + 1
    present = [name for name in COMPOSITE_SLOT_CLASSES if name in class_list]
    slots.update({name: composite for name in present})
    return [slots.get(name, 0) for name in class_list]


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


def detector_layout(state_dict: dict, prefix: str = "") -> dict:
    """What a detector checkpoint was built with, beyond its backbone: the
    detection levels (ReGLA stages used), the PANet bottom-up path, and the
    input canvas (legacy checkpoints have no ``input_wh``: square 640)."""
    keys = [k[len(prefix):] for k in state_dict if k.startswith(prefix)]
    levels = len({k.split(".")[1] for k in keys if k.startswith("heads.")})
    layout = {
        "use_bottom_up": any(k.startswith("neck.fuse_bu_") for k in keys),
        "input_size": (640, 640),
    }
    if not any(k.startswith(("backbone.stages_", "backbone.stem_")) for k in keys):
        # ReGLA builds stages up to the last one it returns, all of them used
        layout["out_indices"] = tuple(range(levels))
    if f"{prefix}input_wh" in state_dict:
        width, height = (int(v) for v in state_dict[f"{prefix}input_wh"].tolist())
        layout["input_size"] = (width, height)
    if f"{prefix}reg_slot" in state_dict:
        layout["reg_slots"] = [int(v) for v in state_dict[f"{prefix}reg_slot"].tolist()]
    return layout


class MusicDetector(nn.Module):
    def __init__(self, num_classes=91, use_bottom_up=False, out_indices=(0, 1, 2),
                 backbone=DEFAULT_BACKBONE, pretrained=True, input_size=640, scale_ranges=None,
                 reg_slots=None):
        super().__init__()
        self.backbone_name = backbone
        self.out_indices = tuple(out_indices)

        # Each class's regression slot (regression_slots; None: one shared
        # slot, the pre-slot detectors), stored with the weights
        reg_slots = list(reg_slots) if reg_slots is not None else [0] * num_classes
        if len(reg_slots) != num_classes:
            raise ValueError(f"{len(reg_slots)} regression slots for {num_classes} classes")
        self.register_buffer("reg_slot", torch.tensor(reg_slots, dtype=torch.long))
        num_slots = max(reg_slots) + 1

        # The canvas the detector is trained and run at, (width, height): it
        # travels with the weights so every tool letterboxes pages the same way
        self.register_buffer("input_wh", torch.zeros(2, dtype=torch.long))
        self.set_input_size(input_size)
        # ...and so do the box areas each level is trained on, which every
        # later training stage (fine-tune, consolidation, joint) must reuse
        self.register_buffer("level_areas", torch.zeros(len(out_indices), 2))
        self.set_scale_ranges(scale_ranges or DEFAULT_SCALE_RANGES[len(out_indices)])

        # 1. The Encoder: grey pages in, features at strides 4/8/16(/32) out
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
            [DecoupledHead(hidden_dim, num_classes, num_slots) for _ in range(len(out_indices))]
        )

    @property
    def input_size(self) -> tuple[int, int]:
        """(width, height) of the input canvas."""
        width, height = self.input_wh.tolist()
        return (int(width), int(height))

    def set_input_size(self, input_size) -> None:
        """Set the canvas; also retargets a trained detector (fine-tuning at
        another resolution). Both sides must divide by the coarsest stride."""
        width, height = canvas_size(input_size)
        stride = 2 ** (max(self.out_indices) + 2)  # stage i has stride 4 * 2**i
        if width % stride or height % stride:
            raise ValueError(f"input size {width}x{height} must be a multiple of the coarsest stride {stride}")
        self.input_wh.copy_(torch.tensor([width, height]))

    @property
    def scale_ranges(self) -> list[tuple[float, float]]:
        """(min, max) box area per level, see DEFAULT_SCALE_RANGES."""
        return [(float(lo), float(hi)) for lo, hi in self.level_areas.tolist()]

    def set_scale_ranges(self, scale_ranges) -> None:
        if len(scale_ranges) != self.level_areas.shape[0]:
            raise ValueError(
                f"{len(scale_ranges)} scale ranges for {self.level_areas.shape[0]} detection levels"
            )
        self.level_areas.copy_(torch.tensor(scale_ranges, dtype=torch.float32))

    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.neck(features)

        outputs = []
        for feature, head in zip(fused_features, self.heads):
            cls_out, reg_out = head(feature)
            # "reg": 4 channels (tx, ty, log w, log h) per slot; "reg_slot"
            # tells decoders and losses which slot holds each class's box
            outputs.append({"cls": cls_out, "reg": reg_out, "reg_slot": self.reg_slot})

        return outputs
