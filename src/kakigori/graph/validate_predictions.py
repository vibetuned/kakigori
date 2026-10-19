# Standard library imports
import glob
import json
import logging
import argparse
from pathlib import Path

# Third party imports
import torch
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_file

# Local imports
from musicset.views.kern import MinimalHumdrumSerializer

from kakigori.graph.graph_io import graph_files as list_graphs, load_graph
from .graph_repair import repair_page_edges
from .validate_groundtruth import _load_and_resolve_nodes, _build_page_nodes
from .utils import split_into_systems
from .heuristics import generate_axis_aware_edges
from .model import GraphVisualExtractor, ScoreGraphReconstructor, GNNPhase2Model
import torchvision.transforms.functional as TF
from kakigori.vision.model import MusicDetector, detect_backbone, detector_layout
from kakigori.vision.utils import complete_legacy_state
from kakigori.vision.dataset import letterbox_image, page_to_canvas

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# NOTE: this drives the MINIMAL serializer. It relies almost entirely on the
# GNN's predicted edges; the only injected scaffold is system->measure (the
# GNN cannot predict it: system nodes are excluded from the per-system
# groups). Planned serializer iterations will add recovery heuristics —
# missing systems recovered from layer/measure evidence, missing measures
# recovered from note clusters, etc. Keep the serializer swappable here.


def _load_gnn(checkpoint_dir: str, device):
    """Build the wrapper and load a phase checkpoint (HF dir or run dir).

    The class count comes from the CHECKPOINT, not the current config:
    target_classes may have grown since the model was trained (appended
    classes keep old indices stable; their nodes are filtered at inference).
    """
    ckpts = sorted(
        glob.glob(f"{checkpoint_dir}/checkpoint-*"),
        key=lambda p: int(p.split("-")[-1]),
    )
    weights_path = f"{ckpts[-1]}/model.safetensors" if ckpts else f"{checkpoint_dir}/model.safetensors"
    weights = load_file(weights_path)
    num_classes = weights["gnn.class_embedding.weight"].shape[0]

    model = GNNPhase2Model(
        MusicDetector(
            num_classes=num_classes,
            backbone=detect_backbone(weights, prefix="detector."),
            pretrained=False,
            **detector_layout(weights, prefix="detector."),
        ),
        GraphVisualExtractor(),
        ScoreGraphReconstructor(
            node_in_dim=256 + 4 + 32, num_classes=num_classes, class_embed_dim=32
        ),
    )
    model.load_state_dict(complete_legacy_state(model, weights), strict=True)
    logger.info(f"Loaded GNN wrapper weights ({num_classes} classes): {weights_path}")
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_page_edges(model, page_nodes: list, img_path: Path,
                       class_to_idx: dict, device, input_size=None) -> list:
    """Run detector features + GNN over one page; return (u_id, v_id, cls).
    ``input_size``: the detector's own canvas unless given."""
    canvas, geo = letterbox_image(Image.open(img_path).convert("L"), input_size or model.detector.input_size)
    img = TF.to_tensor(canvas).to(device)

    # Classes appended to target_classes after a GNN was trained are unknown
    # to its embedding table — exclude those nodes from the GNN pass (the
    # serializer still sees them via the page nodes / other edges)
    num_known = model.gnn.class_embedding.num_embeddings
    page_nodes = [n for n in page_nodes if class_to_idx[n["class"]] < num_known]
    if len(page_nodes) < 2:
        return []

    node_ids = [n["id"] for n in page_nodes]
    boxes = page_to_canvas(torch.tensor([n["bbox"] for n in page_nodes], dtype=torch.float32), geo)
    boxes = boxes.to(device)
    labels = torch.tensor(
        [class_to_idx[n["class"]] for n in page_nodes], dtype=torch.long, device=device
    )

    fused = model.detector.neck(model.detector.backbone(img.unsqueeze(0)))
    feat_dict = {str(i): f for i, f in enumerate(fused)}

    empty_gt = torch.empty((0, 3), dtype=torch.long, device=device)
    predicted = []
    for group in split_into_systems(boxes, labels, empty_gt, class_to_idx):
        if len(group["abs_boxes"]) < 2:
            continue
        roi = model.roi_extractor(feat_dict, [group["abs_boxes"]], img.shape[-2:])
        rel = group["abs_boxes"].clone()
        sx1, sy1, _, _ = group["system_bbox"]
        rel[:, [0, 2]] -= sx1
        rel[:, [1, 3]] -= sy1
        x = torch.cat(
            # normalised by the canvas width, as in GNN training (img.shape[-1])
            [roi, rel / geo["canvas_w"], model.gnn.class_embedding(group["labels"])], dim=1
        )
        candidates = generate_axis_aware_edges(rel, group["labels"], class_to_idx)
        if candidates.shape[1] == 0:
            continue
        classes = model.gnn(x, candidates).argmax(dim=1)

        keep = classes > 0
        if not keep.any():
            continue
        cand = candidates[:, keep].cpu()
        cls = classes[keep].cpu()
        local_to_page = group["node_indices"].cpu()
        for j in range(cand.shape[1]):
            u = node_ids[local_to_page[cand[0, j]].item()]
            v = node_ids[local_to_page[cand[1, j]].item()]
            predicted.append((u, v, int(cls[j])))

    return predicted


def _inject_system_measure_edges(page_nodes: list, edges: list):
    """The one structural scaffold the GNN cannot predict (system nodes are
    excluded from system groups). Mirrors parsers.py's spatial fallback."""
    existing = {(u, v) for u, v, _ in edges}
    systems = [n for n in page_nodes if n["class"] == "system"]
    for measure in [n for n in page_nodes if n["class"] == "measure"]:
        m_cy = (measure["bbox"][1] + measure["bbox"][3]) / 2.0
        for sys_node in systems:
            if sys_node["bbox"][1] <= m_cy <= sys_node["bbox"][3]:
                if (sys_node["id"], measure["id"]) not in existing:
                    edges.append((sys_node["id"], measure["id"], 1))
                break


def process_single_graph(graph_path: Path, json_dir: Path, img_dir: Path,
                         out_dir: Path, node_roles: dict, model,
                         class_to_idx: dict, device, repair: bool = True) -> bool:
    try:
        graph_data = load_graph(graph_path)
        valid_node_ids = set(graph_data["node_ids"])
        json_filenames = graph_data.get("json_files", [])
        node_map, page_node_ids = _load_and_resolve_nodes(json_dir, json_filenames)
        if not node_map:
            return False

        # 1. Predict edges page by page
        all_edges = []
        pages = []
        for json_filename, page_ids in zip(json_filenames, page_node_ids):
            page_nodes = _build_page_nodes(node_map, page_ids, valid_node_ids)
            if not page_nodes:
                pages.append([])
                continue
            img_path = img_dir / f"{Path(json_filename).stem}.png"
            if not img_path.exists():
                logger.warning(f"Missing page image: {img_path.name}")
                pages.append([])
                continue
            page_edges = predict_page_edges(
                model, page_nodes, img_path, class_to_idx, device
            )
            _inject_system_measure_edges(page_nodes, page_edges)
            if repair:
                repair_page_edges(page_nodes, page_edges)
            all_edges.extend(page_edges)
            pages.append(page_nodes)

        if not all_edges:
            return False

        # 2. Serialize the predicted edges between known nodes
        known = set(graph_data["node_ids"])
        serializer = MinimalHumdrumSerializer(
            [(u, v, c) for u, v, c in all_edges if u in known and v in known], node_roles
        )
        for page_nodes in pages:
            if page_nodes:
                serializer.add_page(page_nodes)

        kern_stream = serializer.export_to_krn()
        if kern_stream.startswith("Error"):
            logger.warning(f"{graph_path.name}: {kern_stream}")
            return False

        with open(out_dir / f"{graph_path.stem}.krn", "w", encoding="utf-8") as f:
            f.write(kern_stream)
            f.write("\n")
        return True

    except Exception as e:
        logger.warning(f"Failed to process {graph_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end validation: GNN-predicted edges -> **kern. "
        "Same layout as validate-groundtruth but with model predictions "
        "replacing the ground-truth edge labels."
    )
    parser.add_argument("--graph_dir", type=str, required=True, help="Dir with graph files, .json or .pt (for node ids and page lists)")
    parser.add_argument("--json_dir", type=str, required=True, help="Dir with annotation .json files")
    parser.add_argument("--img_dir", type=str, required=True, help="Dir with page .png images")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for predicted .krn")
    parser.add_argument("--gnn_checkpoint", type=str, required=True, help="GNN wrapper run dir (e.g. checkpoints_gnn/run_003)")
    parser.add_argument("--roles_file", type=str, default="conf/structure.json")
    parser.add_argument("--config", type=str, default="conf/config.json")
    parser.add_argument("--no-repair", action="store_true",
                        help="Disable the spatial graph-repair heuristics (A/B baseline)")
    args = parser.parse_args()

    with open(args.roles_file) as f:
        node_roles = json.load(f)
    class_list = json.load(open(args.config))["target_classes"]
    class_to_idx = {c: i for i, c in enumerate(class_list)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_gnn(args.gnn_checkpoint, device)

    graph_dir, json_dir = Path(args.graph_dir), Path(args.json_dir)
    img_dir, out_dir = Path(args.img_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_files = list_graphs(graph_dir)
    logger.info(f"Predicting + serializing {len(graph_files)} scores...")
    success = 0
    for g_path in tqdm(graph_files, desc="End-to-end kern"):
        if process_single_graph(g_path, json_dir, img_dir, out_dir,
                                node_roles, model, class_to_idx, device,
                                repair=not args.no_repair):
            success += 1
    logger.info(f"Finished. Generated {success}/{len(graph_files)} predicted Humdrum files.")


if __name__ == "__main__":
    main()
