"""ReGLA backbone: efficient convolutions + ReLU gated linear attention.

After Li et al., "ReGLA: Efficient Receptive-Field Modeling with Gated Linear
Attention Network" (arXiv 2602.05262, 2026): a stem to 1/4 resolution, then
four stages (1/4, 1/8, 1/16, 1/32) joined by strided convolutions. Stages 1-2
stack ELRF blocks (local features, a 7x7 receptive field from cascaded 3x3 and
5x5 depthwise convolutions around an FFN); stages 3-4 stack ReLU attention
blocks (a convolutional position encoding, RGMA global attention with linear
complexity, a depthwise 3x3 convolution).

From the paper (text, Fig. 2, Tables 1 and 7):

* variants and their per-stage depths and channels (``VARIANTS``);
* ELRF: DW3x3(C) -> 1x1(4C) -> GELU -> DW5x5(4C) -> 1x1(C) -> GELU, residual;
* attention block: CPE (DW3x3 + norm, residual) -> RGMA -> DW3x3 + GELU
  (residual); the paper's "Conv.k" are depthwise convolutions;
* RGMA (Eq. 3-6, Fig. 2): v = sigmoid(W_v x) is both the attention value and
  the gate ("full gate", the best ablation); Q, K = ReLU(W_qk x);
  attn = Q (K^T v) / (Q (K^T 1)); out = x + Norm(attn * v).

Not specified in the paper (no code or supplement was available), chosen here:

* stem: two 3x3 stride-2 convolutions (to C1/2, then C1), norm + GELU;
* downsampling: 3x3 stride-2 convolution + norm;
* norm: channel LayerNorm by default (``norm="ln"``). kakigori's graph
  trainers run the backbone one page at a time, where BatchNorm statistics
  are those of a single image; ``norm="bn"`` is available;
* single-head attention with Q/K width C (the paper calls multi-head
  reshaping a cost; ``heads`` is configurable);
* layer scale (init 1e-2) on every residual branch, for training from scratch;
* input: one channel (grey pages) by default, scaled to [-1, 1].

Features come out after each requested stage (``out_indices``, 0-based, as in
timm), and stages past the last requested one are not built.
"""

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# (depths, channels) per stage
VARIANTS = {
    "regla_t": ((2, 2, 16, 6), (32, 64, 128, 256)),
    "regla_s": ((2, 2, 19, 6), (32, 64, 160, 320)),
    "regla_m": ((2, 2, 22, 6), (48, 96, 192, 384)),
    "regla_l": ((3, 3, 31, 9), (64, 128, 256, 448)),
    "regla_x": ((4, 4, 43, 12), (64, 128, 336, 512)),
}


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of a (B, C, H, W) map."""

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "ln":
        return LayerNorm2d(channels, eps=1e-6)
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"unknown norm {kind!r} (ln or bn)")


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((channels, 1, 1), init))

    def forward(self, x):
        return x * self.gamma


def conv_norm_act(c_in, c_out, k, stride, norm, act=True):
    layers = [
        nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
        make_norm(norm, c_out),
    ]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ELRF(nn.Module):
    """Efficient Large Receptive Field block: DW3x3 -> FFN with a DW5x5 inside."""

    def __init__(self, dim: int, norm: str, expansion: int = 4, layer_scale: float = 1e-2):
        super().__init__()
        hidden = dim * expansion
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm1 = make_norm(norm, dim)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dw5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.norm2 = make_norm(norm, hidden)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.act = nn.GELU()
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x):
        h = self.act(self.fc1(self.norm1(self.dw3(x))))
        h = self.act(self.fc2(self.norm2(self.dw5(h))))
        return x + self.scale(h)


class RGMA(nn.Module):
    """ReLU Gated Modulated Attention: x + Norm(ReLUAttn(Q, K, v) * v),
    v = sigmoid(W_v x) serving as both value and gate."""

    def __init__(self, dim: int, norm: str, heads: int = 1, qk_dim: int | None = None,
                 layer_scale: float = 1e-2, eps: float = 1e-6):
        super().__init__()
        qk_dim = qk_dim or dim
        if dim % heads or qk_dim % heads:
            raise ValueError(f"dim {dim} and qk_dim {qk_dim} must divide into {heads} heads")
        self.heads, self.qk_dim, self.eps = heads, qk_dim, eps
        self.v = nn.Conv2d(dim, dim, 1)
        self.qk = nn.Conv2d(dim, 2 * qk_dim, 1)
        self.norm = make_norm(norm, dim)
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x):
        b, c, h, w = x.shape
        n, heads = h * w, self.heads
        v = torch.sigmoid(self.v(x))
        q, k = F.relu(self.qk(x)).chunk(2, dim=1)
        # linear attention in fp32: the normalizer divides sums over all tokens
        with torch.autocast(device_type=x.device.type, enabled=False):
            qf = q.float().reshape(b, heads, -1, n)
            kf = k.float().reshape(b, heads, -1, n)
            vf = v.float().reshape(b, heads, -1, n)
            kv = torch.einsum("bhdn,bhcn->bhdc", kf, vf)
            num = torch.einsum("bhdn,bhdc->bhcn", qf, kv)
            den = torch.einsum("bhdn,bhd->bhn", qf, kf.sum(dim=-1)).unsqueeze(2) + self.eps
            attn = (num / den).reshape(b, c, h, w).to(x.dtype)
        return x + self.scale(self.norm(attn * v))


class AttentionBlock(nn.Module):
    """CPE (DW3x3 + norm, residual) -> RGMA -> DW3x3 + GELU (residual)."""

    def __init__(self, dim: int, norm: str, heads: int = 1, layer_scale: float = 1e-2):
        super().__init__()
        self.cpe = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False), make_norm(norm, dim)
        )
        self.rgma = RGMA(dim, norm, heads=heads, layer_scale=layer_scale)
        self.local = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.GELU())
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x):
        x = x + self.cpe(x)
        x = self.rgma(x)
        return x + self.scale(self.local(x))


class Stage(nn.Module):
    def __init__(self, c_in: int, c_out: int, depth: int, kind: str, norm: str,
                 downsample: bool, heads: int, layer_scale: float):
        super().__init__()
        self.downsample = conv_norm_act(c_in, c_out, 3, 2, norm, act=False) if downsample else nn.Identity()
        block = (lambda: ELRF(c_out, norm, layer_scale=layer_scale)) if kind == "elrf" else (
            lambda: AttentionBlock(c_out, norm, heads=heads, layer_scale=layer_scale)
        )
        self.blocks = nn.Sequential(*(block() for _ in range(depth)))

    def forward(self, x):
        return self.blocks(self.downsample(x))


class ReGLA(nn.Module):
    """Multi-scale feature extractor; ``channels`` lists the channels of the
    returned maps (like timm's ``feature_info.channels()``)."""

    def __init__(self, variant: str = "regla_m", in_chans: int = 1, out_indices=(0, 1, 2),
                 norm: str = "ln", heads: int = 1, layer_scale: float = 1e-2,
                 depths=None, dims=None):
        super().__init__()
        if depths is None or dims is None:
            if variant not in VARIANTS:
                raise ValueError(f"unknown ReGLA variant {variant!r}: {list(VARIANTS)}")
            depths, dims = VARIANTS[variant]
        self.out_indices = tuple(out_indices)
        self.in_chans = in_chans
        n_stages = max(self.out_indices) + 1
        self.stem = nn.Sequential(
            conv_norm_act(in_chans, dims[0] // 2, 3, 2, norm),
            conv_norm_act(dims[0] // 2, dims[0], 3, 2, norm),
        )
        self.stages = nn.ModuleList(
            Stage(
                dims[max(i - 1, 0)], dims[i], depths[i],
                "elrf" if i < 2 else "attention", norm,
                downsample=i > 0, heads=heads, layer_scale=layer_scale,
            )
            for i in range(n_stages)
        )
        self.channels = [dims[i] for i in self.out_indices]
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.shape[1] != self.in_chans:
            # RGB pages into a grey model (or the reverse): same luminance
            x = x.mean(dim=1, keepdim=True).expand(-1, self.in_chans, -1, -1)
        x = self.stem(x * 2.0 - 1.0)
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                features.append(x)
        return features
