import sys
import json
import logging
import argparse
from pathlib import Path

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QSplitter, QScrollArea, QLabel, 
                               QPushButton, QCheckBox, QGraphicsView, QGraphicsScene)
from PySide6.QtGui import QPixmap, QColor, QPen, QFont
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtCore import Qt, QRectF

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ResizableGraphicsView(QGraphicsView):
    """Custom view that automatically scales the scene to fit the window on resize."""
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

class BBoxVisualizer(QMainWindow):
    def __init__(self, img_dir: str, ann_dir: str, hierarchy_path: str):
        super().__init__()
        self.setWindowTitle("Gelato Graph - PySide6 Annotations Visualizer")
        self.resize(1400, 900)
        
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.hierarchy_path = Path(hierarchy_path)
        
        self.image_paths = sorted(list(self.img_dir.rglob("*.png")) + list(self.img_dir.rglob("*.jpg")))
        self.current_index = 0
        
        self.target_classes = self.load_target_classes()
        self.hierarchy = self.load_hierarchy()
        
        self.class_checkboxes = {}  # Tracks individual class QCheckBox widgets
        self.drawn_annotations = [] # Tracks drawn items: [{'class': cls_name, 'items': [rect, text]}]
        
        self.setup_ui()
        
        if self.image_paths:
            self.load_image()
        else:
            logger.warning(f"No images found in {self.img_dir}")
            
    def load_target_classes(self) -> set:
        target = set()
        if Path("gelato_config.json").exists():
            try:
                with open("gelato_config.json", 'r') as f:
                    data = json.load(f)
                    target = set(data.get("target_classes", []))
            except Exception as e:
                logger.error(f"Failed to read gelato_config.json: {e}")
        return target
        
    def load_hierarchy(self) -> dict:
        hierarchy = {}
        if self.hierarchy_path.exists():
            try:
                with open(self.hierarchy_path, 'r') as f:
                    hierarchy = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read {self.hierarchy_path}: {e}")
        
        assigned_classes = {cls for group in hierarchy.values() for cls in group}
        missing_classes = self.target_classes - assigned_classes
        
        if missing_classes:
            hierarchy["Uncategorized"] = sorted(list(missing_classes))
            
        clean_hierarchy = {}
        for group, classes in hierarchy.items():
            valid_classes = [c for c in classes if c in self.target_classes or not self.target_classes]
            if valid_classes:
                clean_hierarchy[group] = valid_classes
                
        return clean_hierarchy

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Splitter handles draggable sidebar resizing perfectly
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- Sidebar ---
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        
        title = QLabel("Class Visibility Toggle")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        sidebar_layout.addWidget(title)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        
        # Build hierarchy checkboxes
        for group_name, classes in self.hierarchy.items():
            group_cb = QCheckBox(group_name)
            group_cb.setChecked(True)
            # Default argument capturing for lambda inside a loop
            group_cb.toggled.connect(lambda checked, cb=group_cb, cls_list=classes: self.toggle_group(checked, cls_list, cb))
            self.scroll_layout.addWidget(group_cb)
            
            for cls in classes:
                cls_cb = QCheckBox(cls)
                cls_cb.setChecked(True)
                cls_cb.setStyleSheet("margin-left: 20px;")
                cls_cb.toggled.connect(self.update_visibility)
                
                self.class_checkboxes[cls] = cls_cb
                self.scroll_layout.addWidget(cls_cb)
                
        scroll_area.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll_area)
        splitter.addWidget(sidebar_widget)
        
        # --- Canvas Viewer Area ---
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scene = QGraphicsScene()
        self.view = ResizableGraphicsView(self.scene)
        self.view.setStyleSheet("background-color: #2b2b2b; border: none;")
        
        canvas_layout.addWidget(self.view)
        
        # --- Bottom Navigation ---
        nav_layout = QHBoxLayout()
        self.lbl_info = QLabel("0 / 0 : No Images")
        self.lbl_info.setFont(QFont("Arial", 10, QFont.Bold))
        
        btn_prev = QPushButton("< Prev")
        btn_next = QPushButton("Next >")
        
        btn_prev.clicked.connect(self.prev_image)
        btn_next.clicked.connect(self.next_image)
        
        nav_layout.addWidget(self.lbl_info)
        nav_layout.addStretch()
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        
        canvas_layout.addLayout(nav_layout)
        splitter.addWidget(canvas_container)
        
        # Initial sidebar vs canvas width ratio
        splitter.setSizes([300, 1100])

        # --- Keyboard Shortcuts ---
        # Next Image: Right Arrow or 'd'
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_image)
        QShortcut(QKeySequence(Qt.Key_D), self, self.next_image)
        
        # Previous Image: Left Arrow or 'a'
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_image)
        QShortcut(QKeySequence(Qt.Key_A), self, self.prev_image)


    def toggle_group(self, checked, classes, group_cb):
        # Prevent the individual checkboxes from triggering a massive redraw loop
        for cls in classes:
            if cls in self.class_checkboxes:
                cb = self.class_checkboxes[cls]
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)
        self.update_visibility()

    def next_image(self):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.load_image()
            
    def prev_image(self):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.load_image()

    def get_color(self, cls_name: str) -> QColor:
        colors = ["#ff3333", "#3333ff", "#33cc33", "#ff9900", "#9933cc", "#33cccc", "#ff33cc", "#ffcc00", "#008080", "#ff66b2"]
        hex_color = colors[sum(ord(c) for c in cls_name) % len(colors)]
        return QColor(hex_color)

    def load_image(self):
        if not self.image_paths: return
        
        # Clear the old scene and memory references
        self.scene.clear()
        self.drawn_annotations.clear()
        
        img_path = self.image_paths[self.current_index]
        self.lbl_info.setText(f"Image {self.current_index + 1} / {len(self.image_paths)} : {img_path.name}")
        
        # Load the raw image natively
        pixmap = QPixmap(str(img_path))
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.scene.addPixmap(pixmap)
        
        ann_path = self.ann_dir / f"{img_path.stem}.json"
        annotations = []
        if ann_path.exists():
            try:
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                    annotations = data.get("annotations", [])
            except Exception as e:
                logger.error(f"Error reading {ann_path}: {e}")

        # Draw annotations ONCE and save references to their graphic objects
        for ann in annotations:
            cls = ann.get('class')
            x1, y1, x2, y2 = ann['bbox']
            
            color = self.get_color(cls)
            pen = QPen(color)
            pen.setWidth(max(1, int(pixmap.width() * 0.002))) # Scale line thickness dynamically
            
            # Create graphic items
            rect_item = self.scene.addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)
            text_item = self.scene.addText(cls)
            text_item.setDefaultTextColor(color)
            text_item.setPos(x1, max(0, y1 - 25))
            
            self.drawn_annotations.append({
                'class': cls,
                'items': [rect_item, text_item]
            })
        
        # Ensure items start with correct visibility based on sidebar
        self.update_visibility()
        
        # Fit view to scene automatically
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def update_visibility(self):
        # Determine what is currently checked in the UI
        visible_classes = {cls for cls, cb in self.class_checkboxes.items() if cb.isChecked()}
        
        # Instantly toggle the vector layer visibility (no image redrawing required)
        for ann_obj in self.drawn_annotations:
            is_visible = ann_obj['class'] in visible_classes
            for item in ann_obj['items']:
                item.setVisible(is_visible)

def main():
    parser = argparse.ArgumentParser(description="PySide6 Annotations Visualizer")
    parser.add_argument("--img_dir", type=str, default="data/output_imgs")
    parser.add_argument("--ann_dir", type=str, default="data/annotations")
    parser.add_argument("--hierarchy", type=str, default="data/hierarchy.json")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # Optional: Force a dark theme style if running on a light desktop
    app.setStyle("Fusion")
    
    window = BBoxVisualizer(args.img_dir, args.ann_dir, args.hierarchy)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()