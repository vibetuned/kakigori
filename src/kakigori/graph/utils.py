import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

# Structure boxes must overlap their ground truth more than glyphs: a measure
# at IoU 0.3 may be half of another measure, a 3 px stem at 0.5 may be exact
STRUCTURE_MATCH_CLASSES = ("system", "system-staff", "measure", "staff", "layer")
STRUCTURE_MIN_IOU, GLYPH_MIN_IOU = 0.5, 0.3
# Classes matched as one: a detection of the right object with the wrong
# variant (beam8 for beam16: detected beams match a ground-truth beam of some
# class at a median IoU of 0.97, of their own class at 0.0-0.93) still takes
# its edges
MATCH_GROUP_PREFIXES = ("beam", "stem", "notehead")


def match_detections(det_boxes, det_labels, gt_boxes, gt_labels, class_list) -> dict:
    """{gt index: detection index}: per class (MATCH_GROUP_PREFIXES: per
    group), a Hungarian assignment on IoU, pairs below the class's minimum
    IoU left unmatched (xyxy boxes)."""
    structure = {class_list.index(c) for c in STRUCTURE_MATCH_CLASSES if c in class_list}
    group = torch.arange(len(class_list))
    for prefix in MATCH_GROUP_PREFIXES:
        members = [i for i, c in enumerate(class_list) if c.startswith(prefix)]
        if members:
            group[members] = members[0]
    gt_group, det_group = group[gt_labels.cpu()], group[det_labels.cpu()]
    matched = {}
    for g in torch.unique(gt_group).tolist():
        gi = torch.nonzero(gt_group == g).flatten()
        di = torch.nonzero(det_group == g).flatten()
        if len(gi) == 0 or len(di) == 0:
            continue
        iou = box_iou(gt_boxes[gi].float(), det_boxes[di].float()).cpu().numpy()
        rows, cols = linear_sum_assignment(-iou)
        min_iou = STRUCTURE_MIN_IOU if g in structure else GLYPH_MIN_IOU
        for r, c in zip(rows, cols):
            if iou[r, c] >= min_iou:
                matched[int(gi[r])] = int(di[c])
    return matched

def omr_collate_fn(batch):
    """
    Groups variable-sized page data without forcing them into strict multi-dimensional tensors.
    """
    images = []
    boxes_list = []
    labels_list = []
    edges_list = []
    file_names = []

    for item in batch:
        images.append(item["image"])
        boxes_list.append(item["boxes"])
        labels_list.append(item["labels"])
        edges_list.append(item["edges"])
        file_names.append(item["file_name"])

    # We can stack the images if they are padded to the exact same size,
    # but for full pages it is often safer to just return the list.
    # Assuming your model handles a list of image tensors:

    out = {
        "images": images,  # List of (3, H, W) tensors
        "boxes": boxes_list,  # List of (N, 4) tensors
        "labels": labels_list,  # List of (N) tensors
        "edges": edges_list,  # List of (E, 3) tensors
        "file_names": file_names,
    }
    # Full annotation boxes for detection supervision (joint training);
    # absent in older cached items, so pass through only when present
    if batch and "det_boxes" in batch[0]:
        out["det_boxes"] = [item["det_boxes"] for item in batch]
        out["det_labels"] = [item["det_labels"] for item in batch]
    # Per-node detection features and keep targets (stage D)
    for key in ("scores", "synthesized", "keep", "from_detections"):
        if batch and key in batch[0]:
            out[key] = [item[key] for item in batch]
    return out
def split_into_systems(abs_boxes, labels, gt_edges, class_to_idx):
    """
    Groups full-page boxes into systems and re-indexes the ground truth edges.
    """
    system_idx = class_to_idx["system"]
    system_mask = labels == system_idx

    system_boxes = abs_boxes[system_mask]
    system_boxes = system_boxes[system_boxes[:, 1].argsort()]  # Sort top-to-bottom

    primitive_mask = labels != system_idx
    primitive_boxes = abs_boxes[primitive_mask]
    primitive_labels = labels[primitive_mask]

    primitive_global_indices = torch.nonzero(primitive_mask, as_tuple=True)[0]

    system_groups = []

    for sys_box in system_boxes:
        sx1, sy1, sx2, sy2 = sys_box

        # Find primitives strictly inside this system's Y-boundaries
        centers_y = (primitive_boxes[:, 1] + primitive_boxes[:, 3]) / 2.0
        in_sys_mask = (centers_y >= sy1) & (centers_y <= sy2)

        if not in_sys_mask.any():
            continue

        sys_abs_boxes = primitive_boxes[in_sys_mask]
        sys_labels = primitive_labels[in_sys_mask]

        sys_original_indices = primitive_global_indices[in_sys_mask]

        # Vectorized edge re-indexing via a global->local lookup tensor. The
        # previous per-edge .item() loop forced thousands of GPU syncs per
        # training step and dominated the step time.
        sys_gt_tensor = torch.empty((0, 3), dtype=torch.long, device=abs_boxes.device)
        if gt_edges is not None and len(gt_edges) > 0:
            lut = torch.full(
                (abs_boxes.shape[0],), -1, dtype=torch.long, device=abs_boxes.device
            )
            lut[sys_original_indices] = torch.arange(
                len(sys_original_indices), device=abs_boxes.device
            )
            u_local = lut[gt_edges[:, 0]]
            v_local = lut[gt_edges[:, 1]]
            keep = (u_local >= 0) & (v_local >= 0)
            if keep.any():
                sys_gt_tensor = torch.stack(
                    [u_local[keep], v_local[keep], gt_edges[keep, 2]], dim=1
                )

        system_groups.append(
            {
                "abs_boxes": sys_abs_boxes,
                "labels": sys_labels,
                "edge_targets": sys_gt_tensor,
                "system_bbox": sys_box,
                # Page-level indices of this system's nodes: inference needs
                # them to map predicted candidate edges back to node ids
                "node_indices": sys_original_indices,
            }
        )

    return system_groups
def split_into_systems2(abs_boxes, labels, gt_edges, class_to_idx):
    """
    Groups full-page boxes into systems and re-indexes the ground truth edges.
    """
    system_idx = class_to_idx["system"]
    system_mask = labels == system_idx

    system_boxes = abs_boxes[system_mask]
    system_boxes = system_boxes[system_boxes[:, 1].argsort()]  # Sort top-to-bottom

    primitive_mask = ~system_mask
    primitive_boxes = abs_boxes[primitive_mask]
    primitive_labels = labels[primitive_mask]

    # Track original global indices so we can remap the edges
    primitive_global_indices = torch.nonzero(primitive_mask, as_tuple=True)[0]

    system_groups = []

    for sys_box in system_boxes:
        sx1, sy1, sx2, sy2 = sys_box

        # Find primitives strictly inside this system's Y-boundaries using their center point
        centers_y = (primitive_boxes[:, 1] + primitive_boxes[:, 3]) / 2.0
        in_sys_mask = (centers_y >= sy1) & (centers_y <= sy2)

        if not in_sys_mask.any():
            continue

        sys_abs_boxes = primitive_boxes[in_sys_mask]
        sys_labels = primitive_labels[in_sys_mask]

        # 1. Map Global Index -> Local System Index
        sys_original_indices = primitive_global_indices[in_sys_mask]
        global_to_local_map = {
            global_idx.item(): local_idx
            for local_idx, global_idx in enumerate(sys_original_indices)
        }

        # 2. Filter and Re-index Ground Truth Edges
        sys_gt_targets = []
        if gt_edges is not None and len(gt_edges) > 0:
            for u, v, edge_class in gt_edges:
                # Only keep the edge if BOTH nodes belong to this specific system
                if u in global_to_local_map and v in global_to_local_map:
                    local_u = global_to_local_map[u]
                    local_v = global_to_local_map[v]
                    sys_gt_targets.append([local_u, local_v, edge_class])

        sys_gt_tensor = (
            torch.tensor(sys_gt_targets, dtype=torch.long, device=abs_boxes.device)
            if sys_gt_targets
            else torch.empty((0, 3), dtype=torch.long, device=abs_boxes.device)
        )

        system_groups.append(
            {
                "abs_boxes": sys_abs_boxes,
                "labels": sys_labels,
                "edge_targets": sys_gt_tensor,
                "system_bbox": sys_box,
            }
        )

    return system_groups

def split_into_systems3(abs_boxes, labels, gt_edges, class_to_idx):
    """
    Groups full-page boxes into systems and re-indexes the ground truth edges.
    """
    system_idx = class_to_idx["system"]
    system_mask = labels == system_idx

    system_boxes = abs_boxes[system_mask]
    system_boxes = system_boxes[system_boxes[:, 1].argsort()]  # Sort top-to-bottom

    primitive_mask = ~system_mask
    primitive_boxes = abs_boxes[primitive_mask]
    primitive_labels = labels[primitive_mask]

    # Track original global indices so we can remap the edges
    primitive_global_indices = torch.nonzero(primitive_mask, as_tuple=True)[0]

    system_groups = []

    for sys_box in system_boxes:
        sx1, sy1, sx2, sy2 = sys_box

        # Find primitives strictly inside this system's Y-boundaries using their center point
        centers_y = (primitive_boxes[:, 1] + primitive_boxes[:, 3]) / 2.0
        in_sys_mask = (centers_y >= sy1) & (centers_y <= sy2)

        if not in_sys_mask.any():
            continue

        sys_abs_boxes = primitive_boxes[in_sys_mask]
        sys_labels = primitive_labels[in_sys_mask]

        # 1. Map Global Index -> Local System Index
        sys_original_indices = primitive_global_indices[in_sys_mask]
        global_to_local_map = {
            global_idx.item(): local_idx
            for local_idx, global_idx in enumerate(sys_original_indices)
        }

        # 2. Filter and Re-index Ground Truth Edges
        sys_gt_targets = []
        if gt_edges is not None and len(gt_edges) > 0:
            for u, v, edge_class in gt_edges:
                # THE FIX: Cast scalar tensors to Python integers!
                u_val = u.item()
                v_val = v.item()
                
                # Only keep the edge if BOTH nodes belong to this specific system
                if u_val in global_to_local_map and v_val in global_to_local_map:
                    local_u = global_to_local_map[u_val]
                    local_v = global_to_local_map[v_val]
                    sys_gt_targets.append([local_u, local_v, edge_class.item()])

        sys_gt_tensor = (
            torch.tensor(sys_gt_targets, dtype=torch.long, device=abs_boxes.device)
            if sys_gt_targets
            else torch.empty((0, 3), dtype=torch.long, device=abs_boxes.device)
        )

        system_groups.append(
            {
                "abs_boxes": sys_abs_boxes,
                "labels": sys_labels,
                "edge_targets": sys_gt_tensor,
                "system_bbox": sys_box,
            }
        )

    return system_groups

# --- Usage Example ---
# class_list = config["target_classes"]
# dataset = OMRFullPageDataset("data/output_imgs", "data/output_annotations", "data/output_graphs", class_list)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=omr_collate_fn)
