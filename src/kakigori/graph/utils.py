import torch

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

    return {
        "images": images,  # List of (3, H, W) tensors
        "boxes": boxes_list,  # List of (N, 4) tensors
        "labels": labels_list,  # List of (N) tensors
        "edges": edges_list,  # List of (E, 3) tensors
        "file_names": file_names,
    }
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
        global_to_local_map = {
            global_idx.item(): local_idx
            for local_idx, global_idx in enumerate(sys_original_indices)
        }

        sys_gt_targets = []
        if gt_edges is not None and len(gt_edges) > 0:
            for u, v, edge_class in gt_edges:
                # FIX: Extract scalar values from the tensors for dict lookup!
                u_val = u.item()
                v_val = v.item()
                cls_val = edge_class.item()
                
                if u_val in global_to_local_map and v_val in global_to_local_map:
                    local_u = global_to_local_map[u_val]
                    local_v = global_to_local_map[v_val]
                    sys_gt_targets.append([local_u, local_v, cls_val])

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
