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
* ELRF: DW3x3 -> 1x1 to 4C -> DW5x5 -> 1x1 back to C, residual (a 7x7
  receptive field from two small depthwise convolutions);
* attention block: CPE (DW3x3 + norm, residual) -> RGMA -> a local 3x3
  convolution (residual);
* RGMA (Eq. 3-6, Fig. 2): v = sigmoid(W_v x) is both the attention value and
  the gate ("full gate", the best ablation); Q, K = ReLU(W_qk x);
  attn = Q (K^T v) / (Q (K^T 1)), gated by v, normalized, residual.

Departures from Fig. 2:

* ELRF is a linear bottleneck: GELU after fc1 and after DW5x5 + norm, none
  after the final 1x1, so the residual branch can push activations down as
  well as up (the figure has a GELU after the last 1x1);
* RGMA normalizes its input (pre-norm, for the 22-31 block deep stage 3) and
  ends with a 1x1 output projection, likely the unpublished piece: without it
  and with a depthwise local module the counts were 72% of Table 1, the
  projection brings regla_m to 88%, the separable local module below to 106%
  (all variants: 104-111%);
* RGMA scales Q and K by 1/sqrt(N) before the token sums: the factor cancels
  in num / den, and without it K^T v and sum(K) overflow fp16 at stride-4
  token counts (25,600 tokens), as on the iPhone ANE;
* the local module after RGMA is separable (DW3x3 + norm + GELU + 1x1), not
  the paper's depthwise "Conv.3": the 1x1 mixes channels, which stages 3-4
  otherwise only do in the attention projections (worth an ablation: the
  paper preferred depthwise on ImageNet).

Not specified in the paper (no code or supplement was available), chosen here:

* stem: two 3x3 stride-2 convolutions (to C1/2, then C1), norm + GELU;
* downsampling: 3x3 stride-2 convolution + norm;
* norm: channel LayerNorm by default (``norm="ln"``). kakigori's graph
  trainers keep the detector in eval mode and run it one page at a time,
  where BatchNorm statistics are those of a single image; ``norm="bn"`` is
  available. ``LayerNorm2d`` uses the fused kernel in training and a
  permute-free form when traced or exported (Core ML / ANE);
* single-head attention with Q/K width C (the paper calls multi-head
  reshaping a cost; ``heads`` is configurable);
* layer scale (init 1e-2) on every residual branch, for training from scratch;
* input: one channel (grey pages) by default, scaled to [-1, 1].

Features come out after each requested stage (``out_indices``, 0-based, as in
timm), and stages past the last requested one are not built: regla_m stages
1-3 are 4.63M parameters (the kakigori detector 4.97M, against 60.5M with
ConvNeXt-B); a batch-4 640px bf16 training step on an RTX 5090 takes 33.5 ms
and 3.58 GiB (ConvNeXt-B: 41.2 ms, 5.70 GiB).
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

def _exporting() -> bool:
    """Traced (Core ML via torch.jit.trace), exported (torch.export) or ONNX."""
    return torch.jit.is_tracing() or torch.compiler.is_exporting() or torch.onnx.is_in_onnx_export()


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of an NCHW map.

    Training and inference use PyTorch's fused kernel behind a permute: under
    bf16 autocast it runs in fp32 and saves one tensor for backward. The
    hand-written form (mean and variance over dim 1) saves every intermediate
    and computes its statistics in bf16 (55.6 vs 33.7 ms, 4.80 vs 3.58 GiB per
    regla_m training step, twice the error); it is only used when the model is
    traced or exported, because Core ML / the ANE handle it without permutes.
    Both read the same (1, C, 1, 1) parameters.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _exporting():
            mean = x.mean(dim=1, keepdim=True)
            centered = x - mean
            var = centered.square().mean(dim=1, keepdim=True)
            return centered * torch.rsqrt(var + self.eps) * self.weight + self.bias
        y = F.layer_norm(
            x.permute(0, 2, 3, 1), (x.shape[1],), self.weight.view(-1), self.bias.view(-1), self.eps
        )
        return y.permute(0, 3, 1, 2)

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
        self.act1 = nn.GELU()
        self.dw5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.norm2 = make_norm(norm, hidden)
        self.act2 = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.fc1(self.norm1(self.dw3(x))))
        # Activate after DW5x5 + Norm2, leaving fc2 as a linear bottleneck projection
        h = self.fc2(self.act2(self.norm2(self.dw5(h))))
        return x + self.scale(h)


class RGMA(nn.Module):
    """ReLU Gated Modulated Attention with pre-norm, scale-invariant FP16 safety,
    and output projection."""

    def __init__(self, dim: int, norm: str, heads: int = 1, qk_dim: int | None = None,
                 layer_scale: float = 1e-2, eps: float = 1e-6):
        super().__init__()
        qk_dim = qk_dim or dim
        if dim % heads or qk_dim % heads:
            raise ValueError(f"dim {dim} and qk_dim {qk_dim} must divide into {heads} heads")
        self.heads, self.qk_dim, self.eps = heads, qk_dim, eps
        self.pre_norm = make_norm(norm, dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.qk = nn.Conv2d(dim, 2 * qk_dim, 1)
        self.post_norm = make_norm(norm, dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n, heads = h * w, self.heads
        d = self.qk_dim // heads
        c_h = c // heads

        xn = self.pre_norm(x)
        v = torch.sigmoid(self.v(xn))
        q, k = F.relu(self.qk(xn)).chunk(2, dim=1)

        # Scale k by 1/sqrt(n) so kv and k.sum() never overflow FP16 on iPhone ANE;
        # the factor cancels identically in num / den.
        inv_sqrt_n = n ** -0.5
        q_r = q.reshape(b * heads, d, n) * inv_sqrt_n
        k_r = k.reshape(b * heads, d, n) * inv_sqrt_n
        v_r = v.reshape(b * heads, c_h, n)

        with torch.autocast(device_type=x.device.type, enabled=False):
            qf, kf, vf = q_r.float(), k_r.float(), v_r.float()
            kv = kf @ vf.transpose(-2, -1)                      # (B*H, d, c_h)
            num = kv.transpose(-2, -1) @ qf                     # (B*H, c_h, N)
            k_sum = kf.sum(dim=-1, keepdim=True)                # (B*H, d, 1)
            den = (k_sum.transpose(-2, -1) @ qf) + self.eps     # (B*H, 1, N)
            attn = (num / den).reshape(b, c, h, w).to(x.dtype)

        out = self.proj(self.post_norm(attn * v))
        return x + self.scale(out)


class AttentionBlock(nn.Module):
    """CPE (DW3x3 + norm, residual) -> RGMA -> Separable Conv3x3 (residual)."""

    def __init__(self, dim: int, norm: str, heads: int = 1, layer_scale: float = 1e-2):
        super().__init__()
        self.cpe = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            make_norm(norm, dim),
        )
        self.rgma = RGMA(dim, norm, heads=heads, layer_scale=layer_scale)
        # Depthwise 3x3 + Norm + GELU + Pointwise 1x1 so channels mix and residual is linear
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            make_norm(norm, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
