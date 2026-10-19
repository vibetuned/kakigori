# Standard library imports
import sys
import json
import logging
import argparse
import threading
from pathlib import Path

# Third party imports
import numpy as np
import torch
from PIL import Image
from PySide6.QtGui import QPen, QFont, QColor, QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtCore import Qt, QRectF, Signal, QObject
from torchvision.ops import box_iou
from PySide6.QtWidgets import (
    QLabel,
    QSlider,
    QWidget,
    QCheckBox,
    QSplitter,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QApplication,
    QButtonGroup,
    QRadioButton,
    QGraphicsView,
    QGraphicsScene,
)

# Local folder imports
from .infer import preprocess
from .utils import detector_for_checkpoint, load_checkpoint, decode_model_outputs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#800000",
    "#aaffc3",
    "#808000",
    "#000075",
    "#a9a9a9",
    "#e6beff",
    "#ffe119",
]

# Ground-truth comparison colours
_FP_COLOR = QColor("#ff1744")  # detection matching nothing
_CONFUSED_COLOR = QColor("#d500f9")  # detection over a box of another class
_FN_COLOR = QColor("#ff9100")  # ground-truth box nobody found

MODES = ("bbox", "heatmap", "features", "compare")


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


def match_page(preds: list, gts: list, iou_thresh: float = 0.5) -> dict:
    """Match detections to ground truth, class by class, highest score first.

    Returns ``tp`` (pred, gt) pairs, ``fp`` (pred, confused gt or None): a
    detection over a ground-truth box of ANOTHER class is a confusion, which
    says more than a plain false positive, and ``fn`` (missed gt boxes).
    """
    tp, fp = [], []
    matched = set()
    iou = None
    if preds and gts:
        iou = box_iou(
            torch.tensor([p["bbox"] for p in preds], dtype=torch.float32),
            torch.tensor([g["bbox"] for g in gts], dtype=torch.float32),
        )
    for i in sorted(range(len(preds)), key=lambda i: -preds[i]["score"]):
        pred = preds[i]
        best, best_iou = None, iou_thresh
        if iou is not None:
            for j, gt in enumerate(gts):
                if j not in matched and gt["class"] == pred["class"] and iou[i, j] >= best_iou:
                    best, best_iou = j, float(iou[i, j])
        if best is not None:
            matched.add(best)
            tp.append((pred, gts[best]))
            continue
        confused = None
        if iou is not None:
            j = int(iou[i].argmax())
            if iou[i, j] >= iou_thresh and gts[j]["class"] != pred["class"]:
                confused = gts[j]
        fp.append((pred, confused))
    fn = [gt for j, gt in enumerate(gts) if j not in matched]
    return {"tp": tp, "fp": fp, "fn": fn}


def page_scores(match: dict, visible: set | None = None) -> dict:
    """Counts, precision, recall and F1 of a matched page (optionally only
    the visible classes: a TP/FP counts by its detected class, a FN by its
    ground-truth class)."""
    keep = (lambda c: True) if visible is None else (lambda c: c in visible)
    tp = sum(1 for p, _ in match["tp"] if keep(p["class"]))
    fp = sum(1 for p, _ in match["fp"] if keep(p["class"]))
    confused = sum(1 for p, g in match["fp"] if g is not None and keep(p["class"]))
    fn = sum(1 for g in match["fn"] if keep(g["class"]))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "confused": confused, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


class InferenceSignals(QObject):
    """Custom signals to communicate securely from the background thread to the UI."""

    finished = Signal(Path)
    error = Signal(Path, str)
    progress = Signal(int, int)
    ranked = Signal(list)


class ResizableGraphicsView(QGraphicsView):
    """Custom view that supports zooming and panning open images."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom = 0

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.15
            self._zoom += 1
        else:
            factor = 1 / 1.15
            self._zoom -= 1
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() and not self.scene().sceneRect().isEmpty() and self._zoom == 0:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)


class ModelVisualizer(QMainWindow):
    def __init__(
        self,
        model,
        device,
        img_dir,
        hierarchy_path,
        config_path,
        input_size=640,
        conf_thresh=0.3,
        iou_thresh=0.5,
        ann_dir=None,
        image_paths=None,
        multi_label=False,
    ):
        super().__init__()
        backbone = getattr(model, "backbone_name", "detector")
        self.setWindowTitle(f"Kakigori — Model Output Visualizer ({backbone})")
        self.resize(1400, 900)

        self.model = model
        self.device = device
        self.input_size = input_size
        self.iou_thresh = iou_thresh
        self.current_conf_thresh = conf_thresh
        self.multi_label = multi_label

        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir) if ann_dir else None
        self.hierarchy_path = Path(hierarchy_path)
        self.config_path = Path(config_path)

        if image_paths is not None:
            self.image_paths = [Path(p) for p in image_paths]
        else:
            self.image_paths = sorted(
                list(self.img_dir.rglob("*.png"))
                + list(self.img_dir.rglob("*.jpg"))
                + list(self.img_dir.rglob("*.jpeg"))
            )
        self.current_index = 0

        self.class_list = self._load_class_list()
        self.target_classes = set(self.class_list)
        self.hierarchy = self._load_hierarchy()

        self._cache = {}
        self._gt_cache = {}
        self._inference_lock = threading.Lock()
        self._captured_neck = []
        self._captured_backbone = []
        self._scale_labels = ["Fine", "Mid", "Coarse"]

        # Threading signals
        self.worker_signals = InferenceSignals()
        self.worker_signals.finished.connect(self._on_inference_done)
        self.worker_signals.error.connect(self._on_inference_error)
        self.worker_signals.progress.connect(self._on_rank_progress)
        self.worker_signals.ranked.connect(self._on_ranked)

        # UI State tracking
        self.class_checkboxes = {}
        self.drawn_bboxes = []  # [{class: 'name', items: [rect, text]}]
        self.drawn_compare = []  # [{class, kind: tp/fp/confused/fn, items}]
        self.base_pixmap_item = None  # The original image
        self.overlay_pixmap_item = None  # The heatmap/feature overlay

        self._register_hooks()
        self._setup_ui()

        if self.image_paths:
            self._load_image()
        else:
            logger.warning(f"No images found in {self.img_dir}")

    # --- Config Loaders & Hooks ---
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
            except Exception:
                pass

        assigned = {c for g in hierarchy.values() for c in g}
        missing = self.target_classes - assigned
        if missing:
            hierarchy["Uncategorized"] = sorted(missing)

        return {
            grp: [
                c
                for c in cls_list
                if c in self.target_classes or not self.target_classes
            ]
            for grp, cls_list in hierarchy.items()
            if any(c in self.target_classes for c in cls_list)
            or not self.target_classes
        }

    def _register_hooks(self):
        """Capture the fused neck maps and the raw backbone stages (ReGLA's
        stage outputs, or the timm features of a legacy checkpoint)."""

        def _neck_hook(module, inp, out):
            self._captured_neck = [fm.detach().cpu() for fm in out]

        def _backbone_hook(module, inp, out):
            self._captured_backbone = [fm.detach().float().cpu() for fm in out]

        self._hook_handles = [
            self.model.neck.register_forward_hook(_neck_hook),
            self.model.backbone.register_forward_hook(_backbone_hook),
        ]

    def _load_gt(self, path):
        """Ground-truth boxes of a page (target classes only), or None."""
        if self.ann_dir is None:
            return None
        if path not in self._gt_cache:
            ann_path = self.ann_dir / f"{path.stem}.json"
            gts = []
            if ann_path.exists():
                with open(ann_path) as f:
                    for ann in json.load(f).get("annotations", []):
                        if ann.get("class") in self.target_classes:
                            gts.append({"class": ann["class"], "bbox": [float(v) for v in ann["bbox"]]})
            self._gt_cache[path] = gts
        return self._gt_cache[path]

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
        self.mode_vars = {
            "bbox": QRadioButton("Bounding Boxes"),
            "heatmap": QRadioButton("Heatmaps"),
            "features": QRadioButton("Feature Maps"),
            "compare": QRadioButton("Compare with Ground Truth"),
        }
        self.mode_vars["bbox"].setChecked(True)
        self.mode_vars["compare"].setEnabled(self.ann_dir is not None)
        if self.ann_dir is None:
            self.mode_vars["compare"].setToolTip("Start with --ann-dir to compare against annotations")
        for i, (val, rb) in enumerate(self.mode_vars.items()):
            self.mode_group.addButton(rb, i)
            sidebar_layout.addWidget(rb)
            rb.toggled.connect(self._redraw)

        # Feature source (feature-map mode)
        feat_layout = QHBoxLayout()
        feat_layout.addWidget(QLabel("Features from:"))
        self.feature_group = QButtonGroup(self)
        for i, label in enumerate(("Neck", "Backbone")):
            rb = QRadioButton(label)
            rb.setChecked(i == 0)
            self.feature_group.addButton(rb, i)
            feat_layout.addWidget(rb)
            rb.toggled.connect(self._redraw)
        sidebar_layout.addLayout(feat_layout)

        # Ground-truth comparison options
        self.errors_only = QCheckBox("Errors only (hide matched boxes)")
        self.errors_only.setEnabled(self.ann_dir is not None)
        self.errors_only.toggled.connect(self._redraw)
        sidebar_layout.addWidget(self.errors_only)
        self.lbl_scores = QLabel("")
        self.lbl_scores.setWordWrap(True)
        self.lbl_scores.setStyleSheet("color: #444444;")
        sidebar_layout.addWidget(self.lbl_scores)
        legend = QLabel(
            "<span style='color:#ff1744'>■</span> false positive &nbsp; "
            "<span style='color:#d500f9'>■</span> wrong class &nbsp; "
            "<span style='color:#ff9100'>■</span> missed"
        )
        legend.setVisible(self.ann_dir is not None)
        sidebar_layout.addWidget(legend)
        self.btn_rank = QPushButton("Rank pages, worst F1 first")
        self.btn_rank.setEnabled(self.ann_dir is not None)
        self.btn_rank.setToolTip("Runs the detector on every page, then orders them by F1 at the current threshold")
        self.btn_rank.clicked.connect(self._rank_pages)
        sidebar_layout.addWidget(self.btn_rank)

        # Scale Level
        sidebar_layout.addWidget(self._create_header("Scale Level", top_margin=15))
        scale_layout = QHBoxLayout()
        self.scale_group = QButtonGroup(self)
        rb_all = QRadioButton("All")
        rb_all.setChecked(True)
        self.scale_group.addButton(rb_all, -1)
        scale_layout.addWidget(rb_all)
        rb_all.toggled.connect(self._redraw)

        self.scale_buttons = []
        for i, label in enumerate(self._scale_labels):
            rb = QRadioButton(label)
            self.scale_group.addButton(rb, i)
            scale_layout.addWidget(rb)
            rb.toggled.connect(self._redraw)
            self.scale_buttons.append(rb)
        sidebar_layout.addLayout(scale_layout)

        # Confidence Slider
        sidebar_layout.addWidget(
            self._create_header("Confidence Threshold", top_margin=15)
        )
        conf_layout = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(int(self.current_conf_thresh * 100))
        self.lbl_conf = QLabel(f"{self.current_conf_thresh:.2f}")

        self.conf_slider.valueChanged.connect(self._on_conf_change)

        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.lbl_conf)
        sidebar_layout.addLayout(conf_layout)

        # Argmax keeps one class per cell; coincident boxes (measure/staff/layer
        # of one staff, mRest over its whole-rest glyph) need every class
        self.multi_label_cb = QCheckBox("Multi-label decode (every class per cell)")
        self.multi_label_cb.setChecked(self.multi_label)
        self.multi_label_cb.toggled.connect(self._on_multi_label_change)
        sidebar_layout.addWidget(self.multi_label_cb)

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
            grp_cb.toggled.connect(
                lambda checked, cb=grp_cb, cl=classes: self._toggle_group(checked, cl)
            )
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

    def _visible_classes(self):
        return {cls for cls, cb in self.class_checkboxes.items() if cb.isChecked()}

    # --- Detections ---
    def _decode(self, raw_outputs, meta, conf_thresh):
        """Detections in page pixels from raw head outputs (letterbox undone)."""
        preds = decode_model_outputs(
            raw_outputs, conf_thresh, self.iou_thresh, self.input_size, multi_label=self.multi_label
        )[0]
        boxes = preds["boxes"].float().clone()
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - meta["pad_x"]) / meta["scale_x"]).clamp(0.0, float(meta["orig_w"]))
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - meta["pad_y"]) / meta["scale_y"]).clamp(0.0, float(meta["orig_h"]))
        return [
            {
                "class": self.class_list[int(c)],
                "score": round(float(s), 4),
                "bbox": [round(float(v), 1) for v in box],
            }
            for box, s, c in zip(boxes, preds["scores"], preds["labels"])
        ]

    def _infer(self, path):
        """Run the detector on one page and cache everything the views need."""
        image = Image.open(path).convert("RGB")
        tensor, meta = preprocess(image, self.input_size, self.device)
        self._captured_neck, self._captured_backbone = [], []
        with torch.inference_mode():
            raw_outputs = self.model(tensor)
        raw_cpu = [
            {"cls": out["cls"].detach().float().cpu(), "reg": out["reg"].detach().float().cpu()}
            for out in raw_outputs
        ]
        entry = {
            "raw_outputs": raw_cpu,
            "meta": meta,
            "neck_maps": list(self._captured_neck),
            "backbone_maps": list(self._captured_backbone),
            "detections": self._decode(raw_cpu, meta, self.current_conf_thresh),
        }
        gts = self._load_gt(path)
        if gts is not None:
            entry["match"] = match_page(entry["detections"], gts, self.iou_thresh)
        return entry

    def _update_bbox_items(self):
        """Safely destroys old QGraphicsItems and draws new ones based on current cache."""
        for ann in self.drawn_bboxes + self.drawn_compare:
            for item in ann["items"]:
                self.scene.removeItem(item)
        self.drawn_bboxes.clear()
        self.drawn_compare.clear()

        path = self.image_paths[self.current_index]
        cached = self._cache.get(path)
        if not cached:
            return
        width = max(1, int(self.original_w * 0.002))

        # Plain detections (bbox mode)
        for det in cached.get("detections", []):
            cls = det["class"]
            x1, y1, x2, y2 = det["bbox"]
            color = _class_color(cls)
            pen = QPen(color)
            pen.setWidth(width)

            rect_item = self.scene.addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)
            rect_item.setZValue(2)

            text_item = self.scene.addText(f"{cls} {det.get('score', 0):.2f}")
            text_item.setDefaultTextColor(color)
            text_item.setPos(x1, max(0, y1 - 25))
            text_item.setZValue(2)

            self.drawn_bboxes.append({"class": cls, "items": [rect_item, text_item]})

        # Ground-truth comparison (compare mode)
        match = cached.get("match")
        if match is None:
            return

        def add(cls, kind, bbox, color, label, pen_width, style=Qt.SolidLine):
            x1, y1, x2, y2 = bbox
            pen = QPen(color)
            pen.setWidth(pen_width)
            pen.setStyle(style)
            items = [self.scene.addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)]
            if label:
                text = self.scene.addText(label)
                text.setDefaultTextColor(color)
                text.setPos(x1, max(0, y1 - 25))
                items.append(text)
            for item in items:
                item.setZValue(3 if kind != "tp" else 2)
            self.drawn_compare.append({"class": cls, "kind": kind, "items": items})

        for pred, _gt in match["tp"]:
            color = QColor(_class_color(pred["class"]))
            color.setAlpha(110)
            add(pred["class"], "tp", pred["bbox"], color, None, width)
        for pred, confused in match["fp"]:
            if confused is None:
                add(pred["class"], "fp", pred["bbox"], _FP_COLOR,
                    f"FP {pred['class']} {pred['score']:.2f}", width * 2, Qt.DashLine)
            else:
                add(pred["class"], "confused", pred["bbox"], _CONFUSED_COLOR,
                    f"{pred['class']} {pred['score']:.2f} ← gt {confused['class']}", width * 2)
        for gt in match["fn"]:
            add(gt["class"], "fn", gt["bbox"], _FN_COLOR, f"missed {gt['class']}", width * 3)

    def _on_conf_change(self, value):
        self.current_conf_thresh = value / 100.0
        self.lbl_conf.setText(f"{self.current_conf_thresh:.2f}")
        self._redecode_current()

    def _on_multi_label_change(self, checked):
        self.multi_label = checked
        self._redecode_current()

    def _redecode_current(self):
        """Decode the cached raw outputs of the shown page again (threshold or
        decode mode changed) and redraw."""
        path = self.image_paths[self.current_index] if self.image_paths else None
        if path and path in self._cache:
            cached = self._cache[path]
            cached["detections"] = self._decode(cached["raw_outputs"], cached["meta"], self.current_conf_thresh)
            gts = self._load_gt(path)
            if gts is not None:
                cached["match"] = match_page(cached["detections"], gts, self.iou_thresh)
            self._update_bbox_items()

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
        if not self.image_paths:
            return

        path = self.image_paths[self.current_index]
        self.lbl_info.setText(
            f"Image {self.current_index + 1} / {len(self.image_paths)} : {path.name}"
        )

        # Reset Scene
        self.scene.clear()
        self.drawn_bboxes.clear()
        self.drawn_compare.clear()

        pixmap = QPixmap(str(path))
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.base_pixmap_item = self.scene.addPixmap(pixmap)

        # Prepare overlay layer (empty initially)
        self.overlay_pixmap_item = self.scene.addPixmap(QPixmap())
        self.overlay_pixmap_item.setZValue(1)  # Ensure it renders above the base image

        self.original_w, self.original_h = pixmap.width(), pixmap.height()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        if path in self._cache:
            self.lbl_status.setText("✓ Cached")
            # threshold or decode mode may have changed since it was decoded
            self._redecode_current()
        else:
            self.lbl_status.setText("⏳ Running inference…")
            threading.Thread(
                target=self._run_inference, args=(path,), daemon=True
            ).start()

    def _run_inference(self, path):
        with self._inference_lock:
            if path in self._cache:
                self.worker_signals.finished.emit(path)
                return
            try:
                self._cache[path] = self._infer(path)
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
            self._update_bbox_items()
            self._redraw()

    # --- Ranking pages against the ground truth ---
    def _rank_pages(self):
        self.btn_rank.setEnabled(False)
        threading.Thread(target=self._rank_worker, daemon=True).start()

    def _rank_worker(self):
        paths = list(self.image_paths)
        scored = []
        for i, path in enumerate(paths):
            with self._inference_lock:
                if path not in self._cache:
                    try:
                        self._cache[path] = self._infer(path)
                    except Exception as e:
                        logger.error(f"Inference failed on {path.name}: {e}")
                        continue
                cached = self._cache[path]
                if cached["detections"] is None or "match" not in cached:
                    continue
                # Rank at the current threshold
                cached["detections"] = self._decode(cached["raw_outputs"], cached["meta"], self.current_conf_thresh)
                cached["match"] = match_page(cached["detections"], self._load_gt(path), self.iou_thresh)
                scored.append((page_scores(cached["match"])["f1"], str(path)))
            if i % 10 == 0 or i == len(paths) - 1:
                self.worker_signals.progress.emit(i + 1, len(paths))
        scored.sort()
        self.worker_signals.ranked.emit(scored)

    def _on_rank_progress(self, done, total):
        self.lbl_status.setText(f"⏳ Ranking pages… {done} / {total}")

    def _on_ranked(self, scored):
        self.btn_rank.setEnabled(True)
        if not scored:
            self.lbl_status.setText("Nothing to rank")
            return
        self.image_paths = [Path(p) for _, p in scored]
        mean_f1 = sum(f for f, _ in scored) / len(scored)
        self.current_index = 0
        self._load_image()
        self.lbl_status.setText(
            f"Ranked {len(scored)} pages, worst first (mean F1 {mean_f1:.3f}, worst {scored[0][0]:.3f})"
        )

    # --- Rendering Updates ---
    def _crop_letterbox(self, hm_pil, meta):
        s, pad_x, pad_y = meta["input_size"], meta["pad_x"], meta["pad_y"]
        content_w = int(round(meta["orig_w"] * meta["scale"]))
        content_h = int(round(meta["orig_h"] * meta["scale"]))
        hm_full = hm_pil.resize((s, s), Image.Resampling.BILINEAR)
        return hm_full.crop((pad_x, pad_y, pad_x + content_w, pad_y + content_h))

    def _update_scores_label(self, cached, visible):
        match = cached.get("match")
        if match is None:
            self.lbl_scores.setText("")
            return
        s = page_scores(match, visible)
        self.lbl_scores.setText(
            f"<b>This page</b> (IoU ≥ {self.iou_thresh:.2f}, conf ≥ {self.current_conf_thresh:.2f}, visible classes): "
            f"{s['tp']} found, {s['fp']} false positives ({s['confused']} wrong class), "
            f"{s['fn']} missed — precision {s['precision']:.3f}, recall {s['recall']:.3f}, F1 {s['f1']:.3f}"
        )

    def _redraw(self):
        if not self.base_pixmap_item:
            return

        path = self.image_paths[self.current_index]
        cached = self._cache.get(path)
        if not cached:
            return

        mode = MODES[max(0, self.mode_group.checkedId())]
        visible_classes = self._visible_classes()
        self._update_scores_label(cached, visible_classes)

        # Scale buttons name the strides once they are known
        maps = cached.get("neck_maps") or []
        for rb, fm in zip(self.scale_buttons, maps):
            rb.setText(f"1/{round(self.input_size / fm.shape[-1])}")

        # 1. Box visibility: plain detections, or the ground-truth comparison
        for ann in self.drawn_bboxes:
            is_visible = (mode == "bbox") and (ann["class"] in visible_classes)
            for item in ann["items"]:
                item.setVisible(is_visible)
        hide_tp = self.errors_only.isChecked()
        for ann in self.drawn_compare:
            is_visible = (
                mode == "compare"
                and ann["class"] in visible_classes
                and not (hide_tp and ann["kind"] == "tp")
            )
            for item in ann["items"]:
                item.setVisible(is_visible)

        # 2. Update Overlay Visibility and Content
        if mode in ("bbox", "compare"):
            self.overlay_pixmap_item.setVisible(False)
            return

        self.overlay_pixmap_item.setVisible(True)
        scale_idx = self.scale_group.checkedId()
        indices = (
            range(len(cached.get("raw_outputs", [])))
            if scale_idx == -1
            else [scale_idx]
        )

        combined = None
        if mode == "heatmap":
            for si in indices:
                if si >= len(cached["raw_outputs"]):
                    continue
                cls_probs = cached["raw_outputs"][si]["cls"].sigmoid()
                for ci, cls_name in enumerate(self.class_list):
                    if cls_name not in visible_classes:
                        continue
                    hm = cls_probs[0, ci].numpy()
                    if combined is None:
                        combined = np.zeros_like(hm)
                    if hm.shape != combined.shape:
                        hm = np.array(
                            Image.fromarray(hm).resize(
                                (combined.shape[1], combined.shape[0]),
                                Image.Resampling.BILINEAR,
                            )
                        )
                    combined = np.maximum(combined, hm)

            if combined is not None:
                combined = np.where(combined >= self.current_conf_thresh, combined, 0.0)

        elif mode == "features":
            source = "backbone_maps" if self.feature_group.checkedId() == 1 else "neck_maps"
            feature_maps = cached.get(source) or []
            for si in indices:
                if si >= len(feature_maps):
                    continue
                hm = torch.max(feature_maps[si].squeeze(0), dim=0)[0].numpy()
                hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
                if combined is None:
                    combined = hm
                else:
                    hm_resized = np.array(
                        Image.fromarray(hm).resize(
                            (combined.shape[1], combined.shape[0]),
                            Image.Resampling.BILINEAR,
                        )
                    )
                    combined = np.maximum(combined, hm_resized)

        if combined is not None:
            hm_pil = Image.fromarray(combined.astype(np.float32))
            hm_cropped = self._crop_letterbox(hm_pil, cached["meta"])

            hm_display = hm_cropped.resize(
                (self.original_w, self.original_h), Image.Resampling.BILINEAR
            )

            overlay_rgba = Image.fromarray(
                _jet_rgba(
                    np.array(hm_display, dtype=np.float32),
                    alpha=0.6 if mode == "features" else 0.65,
                ),
                "RGBA",
            )
            self.overlay_pixmap_item.setPixmap(pil2qpixmap(overlay_rgba))
        else:
            self.overlay_pixmap_item.setPixmap(QPixmap())


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    parser = argparse.ArgumentParser(
        description="Interactive model output visualizer for MusicDetector "
        "(any backbone: read from the checkpoint).",
        epilog="Validation check: --checkpoint checkpoints_lite/run_006 "
        "--img_dir data/validation-lite/imgs --ann-dir data/validation-lite/annotations "
        "--config conf-lite/config.json --hierarchy conf-lite/hierarchy.json",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image (alternative to --img_dir)",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="data/validation-lite/imgs",
        help="Directory containing images",
    )
    parser.add_argument(
        "--ann-dir",
        type=str,
        default=None,
        help="Ground-truth annotation JSONs (e.g. data/validation-lite/annotations): "
        "enables the comparison mode, per-page scores and page ranking",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config.json",
        help="Path to conf/config.json",
    )
    parser.add_argument(
        "--hierarchy",
        type=str,
        default="conf/hierarchy.json",
        help="Path to conf/hierarchy.json",
    )
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument(
        "--multi-label", action="store_true", help="Start with multi-label decoding (toggle in the sidebar)"
    )
    parser.add_argument(
        "--use-bottom-up", action="store_true", help="Enable PANet bottom-up path"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load config ---
    with open(args.config) as f:
        config = json.load(f)
    class_list = config["target_classes"]
    num_classes = len(class_list)

    # --- Load model (the backbone is read from the checkpoint) ---
    logger.info(f"Loading model from {args.checkpoint}")
    model = detector_for_checkpoint(args.checkpoint, num_classes, use_bottom_up=args.use_bottom_up)
    load_checkpoint(model, args.checkpoint, device)

    image_paths = [args.image] if args.image else None
    img_dir = Path(args.image).parent if args.image else args.img_dir
    window = ModelVisualizer(
        model,
        device,
        img_dir,
        args.hierarchy,
        args.config,
        args.input_size,
        args.conf_thresh,
        args.iou_thresh,
        ann_dir=args.ann_dir,
        image_paths=image_paths,
        multi_label=args.multi_label,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
