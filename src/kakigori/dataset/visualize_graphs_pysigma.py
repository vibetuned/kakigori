# Standard library imports
import os
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path

# Force Chromium to use GPU acceleration as requested by the user
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
    "--enable-gpu-rasterization --ignore-gpu-blocklist --enable-zero-copy --no-sandbox"
)

# Third party imports
import torch
import networkx as nx
import matplotlib.pyplot as plt
from ipysigma import Sigma
from PySide6.QtGui import QFont, QShortcut, QKeySequence
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QLabel,
    QSlider,
    QWidget,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QSplitter,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QApplication,
)
from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Edge class names mapping
EDGE_CLASSES = {
    1: "Class 1: Structural",
    2: "Class 2: Modifier",
    3: "Class 3: Temporal",
    4: "Class 4: Sync",
}

EDGE_COLORS = {
    1: "#ff3333",  # Red
    2: "#33cc33",  # Green
    3: "#3333ff",  # Blue
    4: "#ff9900",  # Orange
}
DEFAULT_EDGE_COLOR = "#9aa0a6"


def rgba_to_hex(color_tuple):
    r, g, b = [int(round(255 * c)) for c in color_tuple[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


class PySigmaVisualizer(QMainWindow):
    def __init__(self, img_dir: str, ann_dir: str, graph_dir: str):
        super().__init__()
        self.setWindowTitle("Kakigori - PySigma Graph Visualizer")
        self.resize(1400, 900)

        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.graph_dir = Path(graph_dir)

        # Allow iterating through the graph files directly instead of images
        self.graph_paths = sorted(list(self.graph_dir.glob("*.pt")))
        self.current_index = 0

        # Create a temporary directory for rendered PySigma HTML graphs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.html_counter = 0

        self.setup_ui()

        if self.graph_paths:
            self.load_graph()
        else:
            logger.warning(f"No graphs found in {self.graph_dir}")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # --- Sidebar Controls ---
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)

        title = QLabel("Graph Controls")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        sidebar_layout.addWidget(title)

        # Component Dropdown
        sidebar_layout.addWidget(QLabel("Component:"))
        self.cb_component = QComboBox()
        self.cb_component.addItem("All components", -1)
        self.cb_component.currentIndexChanged.connect(self.request_render)
        sidebar_layout.addWidget(self.cb_component)

        # Min Degree Slider
        sidebar_layout.addWidget(QLabel("Min degree:"))
        min_deg_layout = QHBoxLayout()
        self.slider_min_deg = QSlider(Qt.Horizontal)
        self.slider_min_deg.setMinimum(0)
        self.slider_min_deg.setMaximum(50)
        self.slider_min_deg.setValue(0)
        self.lbl_min_deg = QLabel("0")
        self.slider_min_deg.valueChanged.connect(self.update_min_deg_label)
        self.slider_min_deg.sliderReleased.connect(self.request_render)
        min_deg_layout.addWidget(self.slider_min_deg)
        min_deg_layout.addWidget(self.lbl_min_deg)
        sidebar_layout.addLayout(min_deg_layout)

        # Max Nodes Slider
        sidebar_layout.addWidget(QLabel("Max nodes:"))
        max_nodes_layout = QHBoxLayout()
        self.slider_max_nodes = QSlider(Qt.Horizontal)
        self.slider_max_nodes.setMinimum(20)
        self.slider_max_nodes.setMaximum(1000)
        self.slider_max_nodes.setValue(200)
        self.slider_max_nodes.setSingleStep(10)
        self.lbl_max_nodes = QLabel("200")
        self.slider_max_nodes.valueChanged.connect(self.update_max_nodes_label)
        self.slider_max_nodes.sliderReleased.connect(self.request_render)
        max_nodes_layout.addWidget(self.slider_max_nodes)
        max_nodes_layout.addWidget(self.lbl_max_nodes)
        sidebar_layout.addLayout(max_nodes_layout)

        # Keep Isolates Checkbox
        self.chk_isolates = QCheckBox("Keep isolated nodes")
        self.chk_isolates.setChecked(True)
        self.chk_isolates.stateChanged.connect(self.request_render)
        sidebar_layout.addWidget(self.chk_isolates)

        sidebar_layout.addStretch()
        splitter.addWidget(sidebar_widget)

        # --- Canvas Area ---
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        # WebEngineView for PySigma HTML
        self.web_view = QWebEngineView()
        # Enable settings to allow local HTML to load remote or local assets
        settings = self.web_view.settings()
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
        )
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)

        profile = self.web_view.page().profile()
        profile.setHttpUserAgent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        self.web_view.loadFinished.connect(self.on_load_finished)

        canvas_layout.addWidget(self.web_view, stretch=1)
        splitter.addWidget(canvas_container)

        splitter.setSizes([300, 1100])

        # --- Bottom Navigation ---
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(10, 10, 10, 10)
        self.lbl_info = QLabel("0 / 0 : No Images/Graphs")
        self.lbl_info.setFont(QFont("Arial", 10, QFont.Bold))

        btn_prev = QPushButton("< Prev")
        btn_next = QPushButton("Next >")

        btn_prev.clicked.connect(self.prev_item)
        btn_next.clicked.connect(self.next_item)

        nav_layout.addWidget(self.lbl_info)
        nav_layout.addStretch()
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)

        main_layout.addLayout(nav_layout)

        # --- Keyboard Shortcuts ---
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_item)
        QShortcut(QKeySequence(Qt.Key_D), self, self.next_item)

        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_item)
        QShortcut(QKeySequence(Qt.Key_A), self, self.prev_item)

    def update_min_deg_label(self, val):
        self.lbl_min_deg.setText(str(val))

    def update_max_nodes_label(self, val):
        self.lbl_max_nodes.setText(str(val))

    def request_render(self):
        # We wrap in a short delay or just call directly if G is ready
        if hasattr(self, "current_G") and self.current_G is not None:
            self.render_current_graph()

    def on_load_finished(self, ok):
        if ok:
            logger.info("Web page loaded successfully.")
        else:
            logger.error("Failed to load web page.")

    def next_item(self):
        if hasattr(self, "graph_paths") and self.graph_paths:
            self.current_index = (self.current_index + 1) % len(self.graph_paths)
            self.load_graph()

    def prev_item(self):
        if hasattr(self, "graph_paths") and self.graph_paths:
            self.current_index = (self.current_index - 1) % len(self.graph_paths)
            self.load_graph()

    def get_annotations_for_graph(self, graph_path):
        # We try to find the corresponding JSON annotation.
        # It could be `name.json` or `name_pageX.json`.
        ann_path = self.ann_dir / f"{graph_path.stem}.json"

        if not ann_path.exists():
            ann_paths = sorted(list(self.ann_dir.glob(f"{graph_path.stem}_page*.json")))
        else:
            ann_paths = [ann_path]

        annotations = []
        for path in ann_paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                        annotations.extend(data.get("annotations", []))
                except Exception as e:
                    logger.error(f"Error reading {path}: {e}")

        # Map JSON bounds (Collision-safe to match parsers.py)
        node_map = {}
        for ann in annotations:
            if "id" in ann:
                base_id = ann["id"]
                if base_id in node_map:
                    pseudo_id = f"{base_id}_{ann['class']}_{len(node_map)}"
                    ann["id"] = pseudo_id
                    node_map[pseudo_id] = ann
                else:
                    node_map[base_id] = ann
        return node_map

    def process_and_render_graph(
        self, graph_path, item_name, total_items, node_map=None
    ):
        self.lbl_info.setText(
            f"Item {self.current_index + 1} / {total_items} : {item_name}"
        )

        if not graph_path.exists():
            html_content = "<html><body><h2>Graph not found.</h2></body></html>"
            self.web_view.setHtml(html_content)
            logger.warning(f"No graph found at {graph_path}")
            self.current_G = None
            return

        try:
            data = torch.load(graph_path, map_location="cpu", weights_only=False)

            if isinstance(data, dict) and "edge_index" in data:
                edge_index = data["edge_index"]
                y = data.get("y", None)
                original_string_ids = data.get("node_ids", [])
            else:
                edge_index = data.edge_index
                y = getattr(data, "y", None)
                original_string_ids = getattr(data, "node_ids", [])

            if edge_index is None:
                raise ValueError("Loaded graph has no edge_index.")
            if not torch.is_tensor(edge_index):
                edge_index = torch.as_tensor(edge_index)

            if edge_index.numel() == 0:
                num_nodes = len(original_string_ids)
            else:
                num_nodes = int(edge_index.max(dim=1).values.max().item()) + 1
                num_nodes = max(num_nodes, len(original_string_ids))

            G = nx.DiGraph()
            G.add_nodes_from(range(num_nodes))

            edges = edge_index.t().tolist() if edge_index.numel() > 0 else []
            if edges:
                G.add_edges_from(edges)

            for n in G.nodes():
                if n < len(original_string_ids):
                    orig_id = original_string_ids[n]
                    node_class = "Unknown"
                    if node_map and orig_id in node_map:
                        node_class = node_map[orig_id].get("class", "Unknown")
                    G.nodes[n]["original_id"] = orig_id
                    G.nodes[n]["node_class"] = node_class
                else:
                    G.nodes[n]["original_id"] = str(n)
                    G.nodes[n]["node_class"] = "Unknown"

            edge_types = None
            if y is not None:
                y_tensor = y.detach().cpu() if hasattr(y, "detach") else y
                y_tensor = torch.as_tensor(y_tensor)
                if y_tensor.ndim >= 1 and len(edges) == int(y_tensor.shape[0]):
                    edge_types = y_tensor.tolist()

            components = (
                list(nx.weakly_connected_components(G)) if num_nodes > 0 else []
            )
            component_map = {}
            for component_index, nodes in enumerate(components):
                for node in nodes:
                    component_map[node] = component_index

            # Annotate full graph
            self.current_G = nx.DiGraph()
            self.current_G.add_nodes_from(G.nodes)
            self.current_G.add_edges_from(G.edges)

            max_deg = 0
            for node in self.current_G.nodes:
                comp = int(component_map.get(node, 0))
                deg = int(G.degree(node))
                if deg > max_deg:
                    max_deg = deg

                self.current_G.nodes[node]["label"] = (
                    G.nodes[node]["original_id"]
                    + " ("
                    + G.nodes[node]["node_class"]
                    + ")"
                )
                self.current_G.nodes[node]["node_class"] = G.nodes[node]["node_class"]
                self.current_G.nodes[node]["degree"] = deg
                self.current_G.nodes[node]["component"] = comp
                self.current_G.nodes[node]["size"] = max(
                    2.0, min(18.0, 2.2 + 0.9 * deg)
                )

            for edge_index_for, (u, v) in enumerate(edges):
                edge_type = 0
                if edge_types is not None and edge_index_for < len(edge_types):
                    edge_type = int(edge_types[edge_index_for])

                self.current_G[u][v]["edge_type"] = edge_type
                self.current_G[u][v]["label"] = EDGE_CLASSES.get(
                    edge_type, f"Class {edge_type}" if edge_type else "Unlabeled"
                )
                self.current_G[u][v]["color"] = EDGE_COLORS.get(
                    edge_type, DEFAULT_EDGE_COLOR
                )

            # Update sidebar UI ranges blockingly without triggering rendering yet
            self.cb_component.blockSignals(True)
            self.cb_component.clear()
            self.cb_component.addItem("All components", -1)
            for i in range(len(components)):
                self.cb_component.addItem(f"Component {i}", i)
            self.cb_component.blockSignals(False)

            self.slider_min_deg.blockSignals(True)
            self.slider_min_deg.setMaximum(max_deg)
            self.slider_min_deg.setValue(0)
            self.lbl_min_deg.setText("0")
            self.slider_min_deg.blockSignals(False)

            self.slider_max_nodes.blockSignals(True)
            self.slider_max_nodes.setMaximum(max(20, num_nodes))
            default_max = min(200, num_nodes)
            self.slider_max_nodes.setValue(default_max)
            self.lbl_max_nodes.setText(str(default_max))
            self.slider_max_nodes.blockSignals(False)

            # Render immediately
            self.render_current_graph()

        except Exception as e:
            logger.error(f"Error reading PySigma graph from {graph_path}: {e}")
            self.web_view.setHtml(
                f"<html><body><h2>Error read graph</h2><p>{e}</p></body></html>"
            )
            self.current_G = None

    def make_filtered_subgraph(
        self, G, component_value, min_degree, max_nodes, keep_isolates
    ):
        if component_value == -1:
            selected_nodes = list(G.nodes)
        else:
            selected_nodes = [
                n for n in G.nodes if G.nodes[n]["component"] == component_value
            ]

        selected_nodes = [
            n for n in selected_nodes if G.nodes[n]["degree"] >= int(min_degree)
        ]

        if not keep_isolates:
            selected_nodes = [n for n in selected_nodes if G.degree(n) > 0]

        if len(selected_nodes) > max_nodes:
            selected_nodes = sorted(
                selected_nodes,
                key=lambda n: G.nodes[n]["degree"],
                reverse=True,
            )[: int(max_nodes)]

        return G.subgraph(selected_nodes).copy()

    def render_current_graph(self):
        if not hasattr(self, "current_G") or self.current_G is None:
            return

        component_value = self.cb_component.currentData()
        min_degree = self.slider_min_deg.value()
        max_nodes = self.slider_max_nodes.value()
        keep_isolates = self.chk_isolates.isChecked()

        subgraph = self.make_filtered_subgraph(
            self.current_G, component_value, min_degree, max_nodes, keep_isolates
        )

        logger.info(
            f"Rendering: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges"
        )

        if subgraph.number_of_nodes() == 0:
            self.web_view.setHtml(
                "<html><body><h2>Graph is Empty after filtering.</h2></body></html>"
            )
            return

        # Coloring nodes by Class instead of components
        all_classes = sorted(
            list(
                set(
                    [
                        data.get("node_class", "Unknown")
                        for _, data in subgraph.nodes(data=True)
                    ]
                )
            )
        )
        num_classes = max(1, len(all_classes))
        palette = plt.get_cmap("tab20", num_classes)
        class_to_color = {
            cls_name: rgba_to_hex(palette(i % 20))
            for i, cls_name in enumerate(all_classes)
        }

        for node in subgraph.nodes:
            cls_name = subgraph.nodes[node].get("node_class", "Unknown")
            subgraph.nodes[node]["color"] = class_to_color.get(cls_name, "#999999")

        try:
            self.html_counter += 1
            html_path = os.path.join(
                self.temp_dir.name, f"graph_{self.html_counter}.html"
            )

            sigma_obj = Sigma(
                subgraph,
                node_label="label",
                node_size="size",
                node_color="color",
                edge_color="color",
                edge_label="label",
                default_edge_type="arrow",
                start_layout=True,
                height=800,
                background_color="#EEEEEE",
            )
            sigma_obj.to_html(html_path)

            self.web_view.setUrl(QUrl.fromLocalFile(html_path))
        except Exception as e:
            logger.error(f"Error rendering PySigma HTML: {e}")
            self.web_view.setHtml(
                f"<html><body><h2>Error rendering graph HTML</h2><p>{e}</p></body></html>"
            )

    def load_graph(self):
        if not hasattr(self, "graph_paths") or not self.graph_paths:
            return

        graph_path = self.graph_paths[self.current_index]
        node_map = self.get_annotations_for_graph(graph_path)

        self.process_and_render_graph(
            graph_path, graph_path.name, len(self.graph_paths), node_map=node_map
        )

    def closeEvent(self, event):
        # Clean up temporary directory on close
        self.temp_dir.cleanup()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="PySigma Graph Annotations Visualizer")
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

    window = PySigmaVisualizer(args.img_dir, args.ann_dir, args.graph_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
