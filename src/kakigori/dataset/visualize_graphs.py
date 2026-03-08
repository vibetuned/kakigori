# Standard library imports
import sys
import json
import logging
import argparse
from pathlib import Path

# Third party imports
import torch
from PySide6.QtGui import QPen, QFont, QColor, QPixmap, QShortcut, QKeySequence
from PySide6.QtCore import Qt, QLineF, QRectF
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QCheckBox,
    QSplitter,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QApplication,
    QGraphicsView,
    QGraphicsScene,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Edge class names mapping based on the parsers.py structural classes
EDGE_CLASSES = {
    1: "Class 1: Structural",
    2: "Class 2: Modifier",
    3: "Class 3: Temporal",
    4: "Class 4: Sync",
}

EDGE_COLORS = {
    1: QColor("#ff3333"),  # Red
    2: QColor("#33cc33"),  # Green
    3: QColor("#3333ff"),  # Blue
    4: QColor("#ff9900"),  # Orange
}

NODE_COLOR = QColor("#9933cc")  # Purple


class ResizableGraphicsView(QGraphicsView):
    """Custom view that supports zooming and panning."""

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


class GraphVisualizer(QMainWindow):
    def __init__(self, img_dir: str, ann_dir: str, graph_dir: str):
        super().__init__()
        self.setWindowTitle("Kakigori - PySide6 Graph Visualizer")
        self.resize(1400, 900)

        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.graph_dir = Path(graph_dir)

        self.image_paths = sorted(
            list(self.img_dir.rglob("*.png")) + list(self.img_dir.rglob("*.jpg"))
        )
        self.current_index = 0

        self.layers = ["Nodes"] + list(EDGE_CLASSES.values())
        self.layer_checkboxes = {}
        self.drawn_items = []  # Tracks drawn items: [{'layer': layer_name, 'items': [rect, text, line]}]

        self.setup_ui()

        if self.image_paths:
            self.load_image()
        else:
            logger.warning(f"No images found in {self.img_dir}")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Sidebar ---
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)

        title = QLabel("Graph Layers Toggle")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        sidebar_layout.addWidget(title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)

        # Build layer checkboxes
        for layer_name in self.layers:
            cb = QCheckBox(layer_name)
            cb.setChecked(True)

            # Optional: color coordinate the text for edges
            if layer_name != "Nodes":
                # Find the class ID
                for k, v in EDGE_CLASSES.items():
                    if v == layer_name:
                        color = EDGE_COLORS.get(k)
                        if color:
                            cb.setStyleSheet(
                                f"color: {color.name()}; font-weight: bold;"
                            )
                        break
            else:
                cb.setStyleSheet(f"color: {NODE_COLOR.name()}; font-weight: bold;")

            cb.toggled.connect(self.update_visibility)
            self.layer_checkboxes[layer_name] = cb
            self.scroll_layout.addWidget(cb)

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

        splitter.setSizes([300, 1100])

        # --- Keyboard Shortcuts ---
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_image)
        QShortcut(QKeySequence(Qt.Key_D), self, self.next_image)

        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_image)
        QShortcut(QKeySequence(Qt.Key_A), self, self.prev_image)

    def next_image(self):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.load_image()

    def prev_image(self):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.load_image()

    def load_image(self):
        if not self.image_paths:
            return

        self.scene.clear()
        self.drawn_items.clear()

        img_path = self.image_paths[self.current_index]
        self.lbl_info.setText(
            f"Image {self.current_index + 1} / {len(self.image_paths)} : {img_path.name}"
        )

        pixmap = QPixmap(str(img_path))
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.scene.addPixmap(pixmap)

        # Standardize stem ignoring suffixes like _page1 if necessary
        # Usually img: name_page1.png -> name_page1.json
        ann_path = self.ann_dir / f"{img_path.stem}.json"

        # Mxl files are name.mei -> name.pt, but graphs script might output name.pt
        # If image is name_page1.png, graph might be name.pt
        stem_no_page = img_path.stem.split("_page")[0]
        graph_path = self.graph_dir / f"{stem_no_page}.pt"
        if not graph_path.exists():
            graph_path = self.graph_dir / f"{img_path.stem}.pt"

        annotations = []
        if ann_path.exists():
            try:
                with open(ann_path, "r") as f:
                    data = json.load(f)
                    annotations = data.get("annotations", [])
            except Exception as e:
                logger.error(f"Error reading {ann_path}: {e}")

        # Map JSON bounds (Collision-safe to match parsers.py)
        node_map = {}
        for ann in annotations:
            if "id" in ann:
                base_id = ann["id"]
                if base_id in node_map:
                    # Recreate the exact pseudo-ID generated by the graph builder
                    pseudo_id = f"{base_id}_{ann['class']}_{len(node_map)}"
                    ann["id"] = pseudo_id
                    node_map[pseudo_id] = ann
                else:
                    node_map[base_id] = ann

        if graph_path.exists():
            try:
                graph_data = torch.load(graph_path, weights_only=False)
                edge_index = graph_data["edge_index"]
                y = graph_data["y"]
                node_ids = graph_data["node_ids"]

                pen_node = QPen(NODE_COLOR)
                pen_node.setWidth(max(1, int(pixmap.width() * 0.002)))

                # Draw Nodes
                for node_id in node_ids:
                    if node_id in node_map:
                        ann = node_map[node_id]
                        x1, y1, x2, y2 = ann["bbox"]
                        rect_item = self.scene.addRect(
                            QRectF(x1, y1, x2 - x1, y2 - y1), pen_node
                        )

                        text_item = self.scene.addText(ann.get("class", ""))
                        text_item.setDefaultTextColor(NODE_COLOR)
                        text_item.setPos(x1, max(0, y1 - 25))

                        self.drawn_items.append(
                            {"layer": "Nodes", "items": [rect_item, text_item]}
                        )

                # Draw Edges
                for i in range(edge_index.shape[1]):
                    u_idx = edge_index[0, i].item()
                    v_idx = edge_index[1, i].item()
                    edge_type = y[i].item()

                    if u_idx < len(node_ids) and v_idx < len(node_ids):
                        u_id = node_ids[u_idx]
                        v_id = node_ids[v_idx]

                        if u_id in node_map and v_id in node_map:
                            u_ann = node_map[u_id]
                            v_ann = node_map[v_id]

                            ux1, uy1, ux2, uy2 = u_ann["bbox"]
                            vx1, vy1, vx2, vy2 = v_ann["bbox"]

                            # Center points
                            ucx, ucy = (ux1 + ux2) / 2, (uy1 + uy2) / 2
                            vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2

                            color = EDGE_COLORS.get(edge_type, QColor("#ffffff"))
                            layer_name = EDGE_CLASSES.get(
                                edge_type, f"Unknown Type {edge_type}"
                            )

                            pen_edge = QPen(color)
                            pen_edge.setWidth(max(2, int(pixmap.width() * 0.0015)))

                            line_item = self.scene.addLine(
                                QLineF(ucx, ucy, vcx, vcy), pen_edge
                            )

                            self.drawn_items.append(
                                {"layer": layer_name, "items": [line_item]}
                            )

            except Exception as e:
                logger.error(f"Error mapping graph data from {graph_path}: {e}")
        else:
            logger.warning(f"No graph found for {img_path.name} at {graph_path}")

        self.update_visibility()
        self.view._zoom = 0
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def update_visibility(self):
        visible_layers = {
            layer for layer, cb in self.layer_checkboxes.items() if cb.isChecked()
        }

        for drawn_obj in self.drawn_items:
            is_visible = drawn_obj["layer"] in visible_layers
            for item in drawn_obj["items"]:
                item.setVisible(is_visible)


def main():
    parser = argparse.ArgumentParser(description="PySide6 Graph Annotations Visualizer")
    parser.add_argument(
        "--img-dir", "--img_dir", dest="img_dir", type=str, default="data/output_imgs"
    )
    parser.add_argument(
        "--ann-dir",
        "--ann_dir",
        dest="ann_dir",
        type=str,
        default="data/output_annotations",
    )
    parser.add_argument(
        "--graph-dir",
        "--graph_dir",
        dest="graph_dir",
        type=str,
        default="data/output_graphs",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = GraphVisualizer(args.img_dir, args.ann_dir, args.graph_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
