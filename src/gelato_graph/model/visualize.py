"""Interactive Model Visualizer — tkinter GUI for EdgeMusicDetector.

Inspired by ``dataset/visualize_dataset.py``, this script provides:

* Multi-image navigation (directory scan or single image)
* Three visualisation modes selectable via radio buttons:
    1. **Heatmap** — per-class sigmoid probabilities overlaid as a colourmap
    2. **Feature Maps** — PANet neck hook with max-projected heatmaps
    3. **Bounding Boxes** — decoded detections (reuses ``infer.decode_outputs``)
* Hierarchical class visibility toggles (``gelato_config.json`` + ``hierarchy.json``)
* Confidence threshold slider
* Background inference with per-image result caching (model forward is slow)
"""

import argparse
import json
import logging
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageTk
from transformers.trainer_utils import get_last_checkpoint

from .model import EdgeMusicDetector
from .infer import preprocess, decode_outputs, nms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Stable colour palette ─────────────────────────────────────────────────
_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#fabed4", "#469990", "#dcbeff",
    "#9A6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#a9a9a9", "#e6beff", "#ffe119",
]

def _class_color(cls_name: str) -> str:
    return _COLORS[sum(ord(c) for c in cls_name) % len(_COLORS)]


# ── Jet‐like colormap implemented in pure Python/numpy ────────────────────
def _jet_rgba(data: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Map a [0, 1] float array to RGBA uint8 via a jet-like colormap."""
    r = np.clip(1.5 - np.abs(data * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(data * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(data * 4 - 1), 0, 1)
    a = np.where(data > 0.01, alpha, 0.0)
    return (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)


class ModelVisualizer:
    """Interactive tkinter GUI for visualising EdgeMusicDetector outputs."""

    # ── Construction ──────────────────────────────────────────────────────
    def __init__(
        self,
        master: tk.Tk,
        model: EdgeMusicDetector,
        device: torch.device,
        img_dir: str,
        hierarchy_path: str,
        config_path: str,
        input_size: int = 640,
        conf_thresh: float = 0.3,
        iou_thresh: float = 0.5,
    ):
        self.master = master
        self.master.title("Gelato Graph — Model Output Visualizer")
        self.model = model
        self.device = device
        self.input_size = input_size
        self.iou_thresh = iou_thresh

        # Paths
        self.img_dir = Path(img_dir)
        self.hierarchy_path = Path(hierarchy_path)
        self.config_path = Path(config_path)

        # Image list
        self.image_paths = sorted(
            list(self.img_dir.rglob("*.png"))
            + list(self.img_dir.rglob("*.jpg"))
            + list(self.img_dir.rglob("*.jpeg"))
        )
        self.current_index = 0

        # Config
        self.class_list = self._load_class_list()           # ordered list for decode_outputs
        self.target_classes = set(self.class_list)           # set for fast membership checks
        self.hierarchy = self._load_hierarchy()

        # Caches: key = image path → dict with outputs, feature_maps, detections …
        self._cache: dict[Path, dict] = {}
        self._inference_lock = threading.Lock()

        # Feature‐map hook storage
        self._captured_features: list[torch.Tensor] = []

        # Scale labels for the 3 PANet outputs
        self._scale_labels = ["P3 (Micro)", "P4 (Mid)", "P5 (Macro)"]

        # UI state variables
        self.visibility_vars: dict[str, tk.BooleanVar] = {}
        self.mode_var = tk.StringVar(value="bbox")  # bbox | heatmap | features
        self.scale_var = tk.StringVar(value="all")   # all | 0 | 1 | 2
        self.conf_var = tk.DoubleVar(value=conf_thresh)

        self._setup_ui()
        self._bind_keys()
        self._register_hook()

        if self.image_paths:
            self._load_image()
        else:
            logger.warning(f"No images found in {self.img_dir}")

    # ── Config loaders (mirrored from dataset visualizer) ─────────────────
    def _load_class_list(self) -> list[str]:
        """Load the ordered class list from gelato_config.json."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return json.load(f).get("target_classes", [])
            except Exception as e:
                logger.error(f"Failed to read {self.config_path}: {e}")
        return []

    def _load_hierarchy(self) -> dict[str, list[str]]:
        hierarchy: dict[str, list[str]] = {}
        if self.hierarchy_path.exists():
            try:
                with open(self.hierarchy_path) as f:
                    hierarchy = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read {self.hierarchy_path}: {e}")

        assigned = {c for g in hierarchy.values() for c in g}
        missing = self.target_classes - assigned
        if missing:
            hierarchy["Uncategorized"] = sorted(missing)

        # Keep only targeted classes
        return {
            grp: [c for c in cls_list if c in self.target_classes or not self.target_classes]
            for grp, cls_list in hierarchy.items()
            if any(c in self.target_classes for c in cls_list) or not self.target_classes
        }

    # ── Feature‐map forward hook ──────────────────────────────────────────
    def _register_hook(self):
        def _hook(module, inp, out):
            self._captured_features = [fm.detach().cpu() for fm in out]
        self._hook_handle = self.model.neck.register_forward_hook(_hook)

    # ── UI construction ───────────────────────────────────────────────────
    def _setup_ui(self):
        # Root paned window
        self.main_paned = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # --- Sidebar ---
        sidebar = ttk.Frame(self.main_paned, width=300, padding=10)
        self.main_paned.add(sidebar, weight=0)

        # Mode selector
        ttk.Label(sidebar, text="Visualization Mode", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        for text, val in [("Bounding Boxes", "bbox"), ("Heatmaps", "heatmap"), ("Feature Maps", "features")]:
            ttk.Radiobutton(sidebar, text=text, variable=self.mode_var, value=val,
                            command=self._redraw).pack(anchor=tk.W, padx=5)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Scale level selector (P3 / P4 / P5 / All)
        ttk.Label(sidebar, text="Scale Level", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        scale_frame = ttk.Frame(sidebar)
        scale_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Radiobutton(scale_frame, text="All", variable=self.scale_var, value="all",
                        command=self._redraw).pack(side=tk.LEFT)
        for idx, label in enumerate(self._scale_labels):
            ttk.Radiobutton(scale_frame, text=label, variable=self.scale_var, value=str(idx),
                            command=self._redraw).pack(side=tk.LEFT)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Confidence slider
        ttk.Label(sidebar, text="Confidence Threshold", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.conf_slider = ttk.Scale(sidebar, from_=0.01, to=0.99, variable=self.conf_var,
                                     orient=tk.HORIZONTAL, command=self._on_conf_change)
        self.conf_slider.pack(fill=tk.X, padx=5, pady=(0, 3))
        self.conf_label = ttk.Label(sidebar, text=f"{self.conf_var.get():.2f}")
        self.conf_label.pack(anchor=tk.W, padx=5)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Class toggles
        ttk.Label(sidebar, text="Class Visibility", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))

        canvas_sb = tk.Canvas(sidebar, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas_sb.yview)
        self.scrollable = ttk.Frame(canvas_sb)

        self.scrollable.bind("<Configure>", lambda e: canvas_sb.configure(scrollregion=canvas_sb.bbox("all")))
        canvas_sb.create_window((0, 0), window=self.scrollable, anchor="nw")
        canvas_sb.configure(yscrollcommand=scrollbar.set)

        canvas_sb.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for group_name, classes in self.hierarchy.items():
            grp_frame = ttk.Frame(self.scrollable)
            grp_frame.pack(fill=tk.X, anchor=tk.W, pady=(5, 0))

            grp_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(grp_frame, text=group_name, variable=grp_var,
                            command=lambda gv=grp_var, cl=classes: self._toggle_group(gv, cl)).pack(anchor=tk.W)

            items = ttk.Frame(self.scrollable)
            items.pack(fill=tk.X, anchor=tk.W, padx=(20, 0))

            for cls in classes:
                var = tk.BooleanVar(value=True)
                self.visibility_vars[cls] = var
                ttk.Checkbutton(items, text=cls, variable=var, command=self._redraw).pack(anchor=tk.W)

        # --- Canvas area ---
        canvas_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(canvas_frame, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="gray20", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom bar
        nav = ttk.Frame(canvas_frame, padding=5)
        nav.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_info = ttk.Label(nav, text="0 / 0 : No Images", font=("Arial", 10, "bold"))
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        self.lbl_status = ttk.Label(nav, text="", font=("Arial", 9, "italic"))
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        ttk.Button(nav, text="Next >", command=self._next_image).pack(side=tk.RIGHT)
        ttk.Button(nav, text="< Prev", command=self._prev_image).pack(side=tk.RIGHT)

        self.master.bind("<Configure>", self._on_resize)

    # ── Group toggle ──────────────────────────────────────────────────────
    def _toggle_group(self, group_var: tk.BooleanVar, classes: list[str]):
        state = group_var.get()
        for cls in classes:
            if cls in self.visibility_vars:
                self.visibility_vars[cls].set(state)
        self._redraw()

    # ── Key bindings ──────────────────────────────────────────────────────
    def _bind_keys(self):
        self.master.bind("<Right>", lambda e: self._next_image())
        self.master.bind("<Left>", lambda e: self._prev_image())

    # ── Navigation ────────────────────────────────────────────────────────
    def _next_image(self):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self._load_image()

    def _prev_image(self):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self._load_image()

    # ── Confidence slider callback ────────────────────────────────────────
    def _on_conf_change(self, _event=None):
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")
        # Re-decode detections when threshold changes (no re-inference needed)
        path = self.image_paths[self.current_index] if self.image_paths else None
        if path and path in self._cache:
            cached = self._cache[path]
            if "raw_outputs" in cached:
                cached["detections"] = nms(
                    decode_outputs(cached["raw_outputs"], cached["meta"], self.conf_var.get(), self.class_list),
                    self.iou_thresh,
                )
        self._redraw()

    # ── Image loading + background inference ──────────────────────────────
    def _load_image(self):
        if not self.image_paths:
            return

        path = self.image_paths[self.current_index]
        self.lbl_info.config(text=f"Image {self.current_index + 1} / {len(self.image_paths)} : {path.name}")

        self.original_image = Image.open(path).convert("RGB")

        if path in self._cache:
            self.lbl_status.config(text="✓ Cached")
            self._redraw()
        else:
            self.lbl_status.config(text="⏳ Running inference…")
            self._redraw()  # show raw image immediately
            threading.Thread(target=self._run_inference, args=(path,), daemon=True).start()

    def _run_inference(self, path: Path):
        """Run model forward pass in a background thread, then schedule redraw."""
        with self._inference_lock:
            if path in self._cache:
                self.master.after(0, self._on_inference_done, path)
                return

            try:
                image = Image.open(path).convert("RGB")
                tensor, meta = preprocess(image, self.input_size, self.device)

                self._captured_features = []
                with torch.inference_mode():
                    raw_outputs = self.model(tensor)

                # Move raw outputs to CPU for caching
                raw_cpu = []
                for out in raw_outputs:
                    raw_cpu.append({
                        "cls": out["cls"].detach().cpu(),
                        "reg": out["reg"].detach().cpu(),
                    })

                detections = nms(
                    decode_outputs(raw_outputs, meta, self.conf_var.get(), self.class_list),
                    self.iou_thresh,
                )

                self._cache[path] = {
                    "raw_outputs": raw_cpu,
                    "meta": meta,
                    "feature_maps": list(self._captured_features),  # copy
                    "detections": detections,
                }
            except Exception as e:
                logger.error(f"Inference failed for {path}: {e}")
                self._cache[path] = {"error": str(e)}

            self.master.after(0, self._on_inference_done, path)

    def _on_inference_done(self, path: Path):
        """Called on the main thread after background inference completes."""
        # Only redraw if we're still looking at the same image
        if self.image_paths and self.image_paths[self.current_index] == path:
            cached = self._cache.get(path, {})
            if "error" in cached:
                self.lbl_status.config(text=f"⚠ Error: {cached['error'][:60]}")
            else:
                self.lbl_status.config(text="✓ Cached")
            self._redraw()

    # ── Main redraw ───────────────────────────────────────────────────────
    def _redraw(self, _event=None):
        if not hasattr(self, "original_image"):
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 800

        ow, oh = self.original_image.size
        scale = min(cw / ow, ch / oh)
        new_w, new_h = int(ow * scale), int(oh * scale)
        if new_w <= 0 or new_h <= 0:
            return

        resized = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        path = self.image_paths[self.current_index] if self.image_paths else None
        cached = self._cache.get(path, {}) if path else {}
        mode = self.mode_var.get()
        visible_classes = {cls for cls, var in self.visibility_vars.items() if var.get()}

        if cached and "error" not in cached:
            if mode == "bbox":
                resized = self._draw_bboxes(resized, cached, visible_classes)
            elif mode == "heatmap":
                resized = self._draw_heatmaps(resized, cached, visible_classes, new_w, new_h)
            elif mode == "features":
                resized = self._draw_feature_maps(resized, cached, new_w, new_h)

        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        x_off = (cw - new_w) // 2
        y_off = (ch - new_h) // 2
        self.canvas.create_image(x_off, y_off, anchor=tk.NW, image=self.tk_image)

    # ── Drawing helpers ───────────────────────────────────────────────────
    def _draw_bboxes(self, img: Image.Image, cached: dict, visible: set[str]) -> Image.Image:
        draw = ImageDraw.Draw(img)
        ow, oh = self.original_image.size
        iw, ih = img.size
        sx, sy = iw / ow, ih / oh

        for det in cached.get("detections", []):
            cls = det["class"]
            if cls not in visible:
                continue
            x1, y1, x2, y2 = det["bbox"]
            color = _class_color(cls)
            draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=color, width=2)
            label = f"{cls} {det.get('score', 0):.2f}"
            draw.text((x1 * sx, max(0, y1 * sy - 12)), label, fill=color)
        return img

    def _crop_letterbox(self, hm_pil: Image.Image, meta: dict) -> Image.Image:
        """Crop the letterbox padding from a heatmap upscaled to input_size."""
        s = meta["input_size"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]
        # The content region inside the padded canvas
        content_w = int(round(meta["orig_w"] * meta["scale"]))
        content_h = int(round(meta["orig_h"] * meta["scale"]))
        # Upscale heatmap to full input_size, then crop
        hm_full = hm_pil.resize((s, s), Image.Resampling.BILINEAR)
        return hm_full.crop((pad_x, pad_y, pad_x + content_w, pad_y + content_h))

    def _draw_heatmaps(self, img: Image.Image, cached: dict, visible: set[str],
                       disp_w: int, disp_h: int) -> Image.Image:
        """Overlay per-class sigmoid heatmaps, filtering by selected scale level."""
        raw_outputs = cached.get("raw_outputs", [])
        meta = cached.get("meta")
        if not raw_outputs or not meta:
            return img

        # Determine which scales to include
        sel = self.scale_var.get()
        if sel == "all":
            indices = range(len(raw_outputs))
        else:
            indices = [int(sel)]

        # Build a combined heatmap: max over all visible classes and selected scales
        combined = None
        for si in indices:
            if si >= len(raw_outputs):
                continue
            out = raw_outputs[si]
            cls_probs = out["cls"].sigmoid()  # (1, C, H, W)
            for ci, cls_name in enumerate(self.class_list):
                if cls_name not in visible:
                    continue
                hm = cls_probs[0, ci].numpy()  # (H, W)
                if combined is None:
                    combined = np.zeros_like(hm)
                # Upsample to a common size (use the largest feature map size for quality)
                if hm.shape != combined.shape:
                    hm = np.array(Image.fromarray(hm).resize(
                        (combined.shape[1], combined.shape[0]), Image.Resampling.BILINEAR))
                combined = np.maximum(combined, hm)

        if combined is None:
            return img

        # Threshold floor
        thresh = self.conf_var.get()
        combined = np.where(combined >= thresh, combined, 0.0)

        # Crop out letterbox padding, then resize to display size
        hm_pil = Image.fromarray(combined)
        hm_cropped = self._crop_letterbox(hm_pil, meta)
        hm_display = hm_cropped.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        hm_arr = np.array(hm_display, dtype=np.float32)

        overlay = Image.fromarray(_jet_rgba(hm_arr), "RGBA")
        base = img.convert("RGBA")
        base = Image.alpha_composite(base, overlay)
        return base.convert("RGB")

    def _draw_feature_maps(self, img: Image.Image, cached: dict,
                           disp_w: int, disp_h: int) -> Image.Image:
        """Overlay max-projected PANet feature maps, filtering by selected scale."""
        features = cached.get("feature_maps", [])
        meta = cached.get("meta")
        if not features or not meta:
            return img

        # Determine which scales to include
        sel = self.scale_var.get()
        if sel == "all":
            indices = range(len(features))
        else:
            indices = [int(sel)]

        # Create a composite: max over channels per scale, then max over selected scales
        combined = None
        for si in indices:
            if si >= len(features):
                continue
            fm = features[si]
            # fm: (1, C, H, W)
            hm = torch.max(fm.squeeze(0), dim=0)[0].numpy()  # (H, W)
            hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
            # Crop out letterbox padding, then resize to display resolution
            hm_pil = Image.fromarray(hm)
            hm_cropped = self._crop_letterbox(hm_pil, meta)
            hm_display = hm_cropped.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
            hm_arr = np.array(hm_display, dtype=np.float32)
            if combined is None:
                combined = hm_arr
            else:
                combined = np.maximum(combined, hm_arr)

        if combined is None:
            return img

        overlay = Image.fromarray(_jet_rgba(combined, alpha=0.6), "RGBA")
        base = img.convert("RGBA")
        base = Image.alpha_composite(base, overlay)
        return base.convert("RGB")

    # ── Resize handler ────────────────────────────────────────────────────
    def _on_resize(self, event):
        if event.widget == self.canvas:
            self._redraw()


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Interactive model output visualizer for EdgeMusicDetector."
    )
    parser.add_argument("--image", type=str, default=None, help="Path to a single image (alternative to --img_dir)")
    parser.add_argument("--img_dir", type=str, default="data/output_imgs", help="Directory containing images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="gelato_config.json", help="Path to gelato_config.json")
    parser.add_argument("--hierarchy", type=str, default="data/hierarchy.json", help="Path to hierarchy.json")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--use-bottom-up", action="store_true", help="Enable PANet bottom-up path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load config ---
    with open(args.config) as f:
        config = json.load(f)
    class_list = config["target_classes"]
    num_classes = len(class_list)

    # --- Load model ---
    logger.info(f"Loading model from {args.checkpoint}")
    model = EdgeMusicDetector(num_classes=num_classes, use_bottom_up=args.use_bottom_up)

    ckpt = Path(args.checkpoint)
    if ckpt.is_dir():
        last = get_last_checkpoint(str(ckpt))
        if last is not None:
            ckpt = Path(last)
        logger.info(f"  Using last checkpoint: {ckpt}")

    for name in ("model.safetensors", "pytorch_model.bin"):
        weights = (ckpt / name) if ckpt.is_dir() else ckpt
        if weights.exists():
            if weights.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(weights, device=str(device))
            else:
                state = torch.load(weights, map_location=device, weights_only=False)
            model.load_state_dict(state)
            logger.info(f"  Loaded weights: {weights}")
            break
    else:
        logger.warning("No weights file found — using random weights.")
    model.to(device).eval()

    # Handle single image mode: create a temp directory pointer
    img_dir = args.img_dir
    if args.image:
        img_path = Path(args.image)
        img_dir = str(img_path.parent)

    root = tk.Tk()
    root.geometry("1400x900")
    app = ModelVisualizer(
        root,
        model=model,
        device=device,
        img_dir=img_dir,
        hierarchy_path=args.hierarchy,
        config_path=args.config,
        input_size=args.input_size,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
