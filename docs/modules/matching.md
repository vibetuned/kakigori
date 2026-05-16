# matching/ — module map

## Pipeline overview

The `matching/` subpackage contains a self-contained flow-matching generative model intended to learn a velocity field that transports a degraded music-score image (`x_0`) to its clean counterpart (`x_1`) via straight-line interpolation. It provides both a U-Net and a DiT (Diffusion Transformer) backbone, along with a training step and an Euler sampler — the typical building blocks for image-to-image restoration of degraded scores in an OMR preprocessing stage.

This subpackage is **exploratory / currently unused**: it has no `__init__.py`, and a search across `src/kakigori/` (and the wider repo) finds no imports of `kakigori.matching`, `from .matching`, or `from ..matching`. It also has a latent bug — `models.py` references `nn` and `torch` without importing them — which further suggests it has not been wired into any executable path yet.

## Modules

### `layers.py`
**Role:** Low-level neural building blocks shared by the two backbones in `models.py`. Implements sinusoidal time conditioning, a time-conditioned convolutional residual block for the U-Net, and an AdaLN-modulated transformer block for the DiT, plus a `modulate(x, shift, scale)` helper.
**Key classes:** `SinusoidalTimeEmbedding`, `UNetBlock`, `DiTBlock` (plus the `modulate` function).
**Used by:** `matching/models.py` (imports `UNetBlock`, `SinusoidalTimeEmbedding`). Not referenced outside the `matching/` subpackage.

### `models.py`
**Role:** Defines the two velocity-prediction networks used by the flow-matching training/sampling loop. Both take a noisy/interpolated image `x_t` and a timestep `t` and return a predicted velocity field of the same shape as the input image.
**Key classes:** `FlowMatchingUNet` (encoder–bottleneck–decoder U-Net with skip connections and time conditioning), `FlowMatchingDiT` (patch-based transformer with learnable positional embeddings, AdaLN time conditioning, and unpatchify back to image space).
**Used by:** Nothing in the codebase imports this module. Note: the file uses `nn.*` and `torch.*` without importing `torch`/`torch.nn`, so it will not even import successfully as-is.

### `steps.py`
**Role:** Training and inference routines for the flow-matching objective. `flow_matching_train_step` samples `t ~ U(0,1)`, builds the linear interpolant `x_t = (1-t)·x_0 + t·x_1`, regresses the model output against the constant target velocity `x_1 - x_0` under MSE, and runs an optimizer step. `euler_sample` performs straight Euler integration from a degraded image to a restored one over `num_steps`.
**Key functions:** `flow_matching_train_step(model, degraded_images, clean_images, optimizer)`, `euler_sample(model, degraded_image, num_steps=10)`.
**Used by:** Not imported anywhere in `src/kakigori/`, tests, or scripts.
