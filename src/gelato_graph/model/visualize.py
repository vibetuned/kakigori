import sys
import json
import logging
import argparse
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers.trainer_utils import get_last_checkpoint

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QSplitter, QScrollArea, QLabel, 
                               QPushButton, QCheckBox, QGraphicsView, QGraphicsScene,
                               QRadioButton, QButtonGroup, QSlider, QGraphicsPixmapItem)
from PySide6.QtGui import QPixmap, QColor, QPen, QFont, QImage, QShortcut, QKeySequence
from PySide6.QtCore import Qt, QRectF, QObject, Signal

# Assuming these are in your local modules as per your original script
from .model import EdgeMusicDetector
from .infer import preprocess, decode_outputs, nms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#fabed4", "#469990", "#dcbeff",
    "#9A6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#a9a9a9", "#e6beff", "#ffe119",
]

def _class_color(cls_name: str) -> QColor:
    return QColor(_COLORS[sum(ord(c) for c in cls_name) % len(_COLORS)])

def _jet_rgba(data: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    r = np.clip(1.5 - np.abs(data * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(data * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(data * 4 - 1), 0, 1)
    a = np.where(data > 0.01, alpha, 0.0)
    return (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)

def pil2qpixmap(pil_img: Image.Image) -> QPixmap:
    """Helper to convert PIL RGBA images to QPixmap for rendering."""
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", "RGBA")
    qim = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qim)


class InferenceSignals(QObject):
    """Custom signals to communicate securely from the background thread to the UI."""
    finished = Signal(Path)
    error = Signal(Path, str)


class ResizableGraphicsView(QGraphicsView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)


class ModelVisualizer(QMainWindow):
    def __init__(self, model, device, img_dir, hierarchy_path, config_path, input_size=640, conf_thresh=0.3, iou_thresh=0.5):
        super().__init__()
        self.setWindowTitle("Gelato Graph — Model Output Visualizer (PySide6)")
        self.resize(1400, 900)
        
        self.model = model
        self.device = device
        self.input_size = input_size
        self.iou_thresh = iou_thresh
        self.current_conf_thresh = conf_thresh

        self.img_dir = Path(img_dir)
        self.hierarchy_path = Path(hierarchy_path)
        self.config_path = Path(config_path)

        self.image_paths = sorted(list(self.img_dir.rglob("*.png")) + list(self.img_dir.rglob("*.jpg")) + list(self.img_dir.rglob("*.jpeg")))
        self.current_index = 0

        self.class_list = self._load_class_list()
        self.target_classes = set(self.class_list)
        self.hierarchy = self._load_hierarchy()

        self._cache = {}
        self._inference_lock = threading.Lock()
        self._captured_features = []
        self._scale_labels = ["P3 (Micro)", "P4 (Mid)", "P5 (Macro)"]

        # Threading signals
        self.worker_signals = InferenceSignals()
        self.worker_signals.finished.connect(self._on_inference_done)
        self.worker_signals.error.connect(self._on_inference_error)

        # UI State tracking
        self.class_checkboxes = {}
        self.drawn_bboxes = []       # Stores [{class: 'name', items: [rect, text]}]
        self.base_pixmap_item = None # The original image
        self.overlay_pixmap_item = None # The heatmap/feature overlay

        self._register_hook()
        self._setup_ui()

        if self.image_paths:
            self._load_image()
        else:
            logger.warning(f"No images found in {self.img_dir}")

    # --- Config Loaders & Hooks (Identical logic to original) ---
    def _load_class_list(self):
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return json.load(f).get("target_classes", [])
            except Exception as e:
                logger.error(f"Failed to read config: {e}")
        return []

    def _load_hierarchy(self):
        hierarchy = {}
        if self.hierarchy_path.exists():
            try:
                with open(self.hierarchy_path) as f:
                    hierarchy = json.load(f)
            except Exception: pass

        assigned = {c for g in hierarchy.values() for c in g}
        missing = self.target_classes - assigned
        if missing: hierarchy["Uncategorized"] = sorted(missing)

        return {
            grp: [c for c in cls_list if c in self.target_classes or not self.target_classes]
            for grp, cls_list in hierarchy.items()
            if any(c in self.target_classes for c in cls_list) or not self.target_classes
        }

    def _register_hook(self):
        def _hook(module, inp, out):
            self._captured_features = [fm.detach().cpu() for fm in out]
        self._hook_handle = self.model.neck.register_forward_hook(_hook)

    # --- UI Setup ---
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- Sidebar ---
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        
        # Mode Selector
        sidebar_layout.addWidget(self._create_header("Visualization Mode"))
        self.mode_group = QButtonGroup(self)
        self.mode_vars = {"bbox": QRadioButton("Bounding Boxes"), "heatmap": QRadioButton("Heatmaps"), "features": QRadioButton("Feature Maps")}
        self.mode_vars["bbox"].setChecked(True)
        for i, (val, rb) in enumerate(self.mode_vars.items()):
            self.mode_group.addButton(rb, i)
            sidebar_layout.addWidget(rb)
            rb.toggled.connect(self._redraw)

        # Scale Level
        sidebar_layout.addWidget(self._create_header("Scale Level", top_margin=15))
        scale_layout = QHBoxLayout()
        self.scale_group = QButtonGroup(self)
        rb_all = QRadioButton("All")
        rb_all.setChecked(True)
        self.scale_group.addButton(rb_all, -1)
        scale_layout.addWidget(rb_all)
        rb_all.toggled.connect(self._redraw)

        for i, label in enumerate(self._scale_labels):
            rb = QRadioButton(label)
            self.scale_group.addButton(rb, i)
            scale_layout.addWidget(rb)
            rb.toggled.connect(self._redraw)
        sidebar_layout.addLayout(scale_layout)

        # Confidence Slider
        sidebar_layout.addWidget(self._create_header("Confidence Threshold", top_margin=15))
        conf_layout = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(int(self.current_conf_thresh * 100))
        self.lbl_conf = QLabel(f"{self.current_conf_thresh:.2f}")
        
        self.conf_slider.valueChanged.connect(self._on_conf_change)
        
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.lbl_conf)
        sidebar_layout.addLayout(conf_layout)

        # Class Toggles
        sidebar_layout.addWidget(self._create_header("Class Visibility", top_margin=15))
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignTop)

        for group_name, classes in self.hierarchy.items():
            grp_cb = QCheckBox(group_name)
            grp_cb.setChecked(True)
            grp_cb.toggled.connect(lambda checked, cb=grp_cb, cl=classes: self._toggle_group(checked, cl))
            scroll_layout.addWidget(grp_cb)
            
            for cls in classes:
                cb = QCheckBox(cls)
                cb.setChecked(True)
                cb.setStyleSheet("margin-left: 20px;")
                cb.toggled.connect(self._redraw)
                self.class_checkboxes[cls] = cb
                scroll_layout.addWidget(cb)

        scroll_area.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll_area)
        splitter.addWidget(sidebar_widget)

        # --- Canvas Area ---
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.scene = QGraphicsScene()
        self.view = ResizableGraphicsView(self.scene)
        self.view.setStyleSheet("background-color: #2b2b2b; border: none;")
        canvas_layout.addWidget(self.view)

        # Bottom Bar
        nav_layout = QHBoxLayout()
        self.lbl_info = QLabel("0 / 0 : No Images")
        self.lbl_info.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("font-style: italic; color: #aaaaaa;")

        btn_prev = QPushButton("< Prev")
        btn_next = QPushButton("Next >")
        btn_prev.clicked.connect(self._prev_image)
        btn_next.clicked.connect(self._next_image)

        nav_layout.addWidget(self.lbl_info)
        nav_layout.addWidget(self.lbl_status)
        nav_layout.addStretch()
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        
        canvas_layout.addLayout(nav_layout)
        splitter.addWidget(canvas_container)
        splitter.setSizes([350, 1050])

        # Shortcuts mapping arrow keys globally
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self._next_image)
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self._prev_image)

    def _create_header(self, text, top_margin=0):
        lbl = QLabel(text)
        lbl.setFont(QFont("Arial", 11, QFont.Bold))
        lbl.setContentsMargins(0, top_margin, 0, 5)
        return lbl

    def _toggle_group(self, checked, classes):
        for cls in classes:
            if cls in self.class_checkboxes:
                cb = self.class_checkboxes[cls]
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)
        self._redraw()

    def _on_conf_change(self, value):
        self.current_conf_thresh = value / 100.0
        self.lbl_conf.setText(f"{self.current_conf_thresh:.2f}")
        
        path = self.image_paths[self.current_index] if self.image_paths else None
        if path and path in self._cache and "raw_outputs" in self._cache[path]:
            cached = self._cache[path]
            cached["detections"] = nms(decode_outputs(cached["raw_outputs"], cached["meta"], self.current_conf_thresh, self.class_list), self.iou_thresh)
        self._redraw()

    # --- Navigation & Background Inference ---
    def _next_image(self):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self._load_image()

    def _prev_image(self):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self._load_image()

    def _load_image(self):
        if not self.image_paths: return
        
        path = self.image_paths[self.current_index]
        self.lbl_info.setText(f"Image {self.current_index + 1} / {len(self.image_paths)} : {path.name}")

        # Reset Scene
        self.scene.clear()
        self.drawn_bboxes.clear()
        
        pixmap = QPixmap(str(path))
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.base_pixmap_item = self.scene.addPixmap(pixmap)
        
        # Prepare overlay layer (empty initially)
        self.overlay_pixmap_item = self.scene.addPixmap(QPixmap())
        self.overlay_pixmap_item.setZValue(1) # Ensure it renders above the base image

        self.original_w, self.original_h = pixmap.width(), pixmap.height()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        if path in self._cache:
            self.lbl_status.setText("✓ Cached")
            self._redraw()
        else:
            self.lbl_status.setText("⏳ Running inference…")
            threading.Thread(target=self._run_inference, args=(path,), daemon=True).start()

    def _run_inference(self, path):
        with self._inference_lock:
            if path in self._cache:
                self.worker_signals.finished.emit(path)
                return
            try:
                image = Image.open(path).convert("RGB")
                tensor, meta = preprocess(image, self.input_size, self.device)

                self._captured_features = []
                with torch.inference_mode():
                    raw_outputs = self.model(tensor)

                raw_cpu = [{"cls": out["cls"].detach().cpu(), "reg": out["reg"].detach().cpu()} for out in raw_outputs]
                detections = nms(decode_outputs(raw_outputs, meta, self.current_conf_thresh, self.class_list), self.iou_thresh)

                self._cache[path] = {
                    "raw_outputs": raw_cpu,
                    "meta": meta,
                    "feature_maps": list(self._captured_features),
                    "detections": detections,
                }
                self.worker_signals.finished.emit(path)
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                self.worker_signals.error.emit(path, str(e))

    def _on_inference_error(self, path, error_msg):
        if self.image_paths and self.image_paths[self.current_index] == path:
            self.lbl_status.setText(f"⚠ Error: {error_msg[:60]}")

    def _on_inference_done(self, path):
        if self.image_paths and self.image_paths[self.current_index] == path:
            self.lbl_status.setText("✓ Cached")
            
            # Initial Setup of BBoxes natively in Qt (only done once per image load)
            cached = self._cache[path]
            for det in cached.get("detections", []):
                cls = det["class"]
                x1, y1, x2, y2 = det["bbox"]
                color = _class_color(cls)
                pen = QPen(color)
                pen.setWidth(max(1, int(self.original_w * 0.002)))
                
                rect_item = self.scene.addRect(QRectF(x1, y1, x2-x1, y2-y1), pen)
                rect_item.setZValue(2)
                
                text_item = self.scene.addText(f"{cls} {det.get('score', 0):.2f}")
                text_item.setDefaultTextColor(color)
                text_item.setPos(x1, max(0, y1 - 25))
                text_item.setZValue(2)
                
                self.drawn_bboxes.append({'class': cls, 'items': [rect_item, text_item]})
            
            self._redraw()

    # --- Rendering Updates ---
    def _crop_letterbox(self, hm_pil, meta):
        s, pad_x, pad_y = meta["input_size"], meta["pad_x"], meta["pad_y"]
        content_w = int(round(meta["orig_w"] * meta["scale"]))
        content_h = int(round(meta["orig_h"] * meta["scale"]))
        hm_full = hm_pil.resize((s, s), Image.Resampling.BILINEAR)
        return hm_full.crop((pad_x, pad_y, pad_x + content_w, pad_y + content_h))

    def _redraw(self):
        if not self.base_pixmap_item: return
        
        path = self.image_paths[self.current_index]
        cached = self._cache.get(path)
        if not cached: return

        mode_id = self.mode_group.checkedId()
        mode = "bbox" if mode_id == 0 else "heatmap" if mode_id == 1 else "features"
        visible_classes = {cls for cls, cb in self.class_checkboxes.items() if cb.isChecked()}

        # 1. Update BBox Visibility
        for ann in self.drawn_bboxes:
            is_visible = (mode == "bbox") and (ann['class'] in visible_classes)
            for item in ann['items']:
                item.setVisible(is_visible)

        # 2. Update Overlay Visibility and Content
        if mode == "bbox":
            self.overlay_pixmap_item.setVisible(False)
            return
            
        self.overlay_pixmap_item.setVisible(True)
        scale_idx = self.scale_group.checkedId() # -1 is 'All', 0 is P3, etc.
        indices = range(len(cached.get("raw_outputs", []))) if scale_idx == -1 else [scale_idx]

        combined = None
        if mode == "heatmap":
            for si in indices:
                if si >= len(cached["raw_outputs"]): continue
                cls_probs = cached["raw_outputs"][si]["cls"].sigmoid()
                for ci, cls_name in enumerate(self.class_list):
                    if cls_name not in visible_classes: continue
                    hm = cls_probs[0, ci].numpy()
                    if combined is None: combined = np.zeros_like(hm)
                    if hm.shape != combined.shape:
                        hm = np.array(Image.fromarray(hm).resize((combined.shape[1], combined.shape[0]), Image.Resampling.BILINEAR))
                    combined = np.maximum(combined, hm)
                    
            if combined is not None:
                combined = np.where(combined >= self.current_conf_thresh, combined, 0.0)

        elif mode == "features":
            for si in indices:
                if si >= len(cached["feature_maps"]): continue
                hm = torch.max(cached["feature_maps"][si].squeeze(0), dim=0)[0].numpy()
                hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
                if combined is None: combined = hm
                else:
                    hm_resized = np.array(Image.fromarray(hm).resize((combined.shape[1], combined.shape[0]), Image.Resampling.BILINEAR))
                    combined = np.maximum(combined, hm_resized)

        if combined is not None:
            hm_pil = Image.fromarray(combined)
            hm_cropped = self._crop_letterbox(hm_pil, cached["meta"])
            
            # Key difference: Resize exactly to the original image dimensions, Qt handles the display scaling
            hm_display = hm_cropped.resize((self.original_w, self.original_h), Image.Resampling.BILINEAR)
            
            overlay_rgba = Image.fromarray(_jet_rgba(np.array(hm_display, dtype=np.float32), alpha=0.6 if mode=="features" else 0.65), "RGBA")
            self.overlay_pixmap_item.setPixmap(pil2qpixmap(overlay_rgba))
        else:
            self.overlay_pixmap_item.setPixmap(QPixmap())


def main():
    # Setup args, device, and load PyTorch model exactly as you had in your script...
    # For testing the UI layout immediately, you can replace the model load with `model = None`
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
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
  
    window = ModelVisualizer(model, device, args.img_dir, args.hierarchy, args.config, args.input_size, args.conf_thresh, args.iou_thresh)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()