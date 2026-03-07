def __getitem__(self, idx):
        sample = self.samples[idx]

        # --- A. Load Vision Data ---
        image = Image.open(sample["img"]).convert("RGB")
        img_tensor = TF.to_tensor(image)

        # --- B. Load Relational Data ---
        graph_data = torch.load(sample["graph"], weights_only=False)
        edge_index = graph_data["edge_index"]
        edge_labels = graph_data["y"]
        node_ids = graph_data["node_ids"]

        # --- C. Align Modalities ---
        with open(sample["json"], "r") as f:
            json_data = json.load(f)

        ann_map = {
            ann["id"]: ann for ann in json_data.get("annotations", []) if "id" in ann
        }

        boxes = []
        labels = []
        loaded_ids = set()

        # 1. Strictly load nodes present in the graph FIRST to preserve edge_index
        for i, node_id in enumerate(node_ids):
            if node_id in ann_map:
                ann = ann_map[node_id]
                boxes.append(ann["bbox"])
                labels.append(self.class_to_idx[ann["class"]])
                loaded_ids.add(node_id)
            else:
                # Fallback padding to prevent index shifting if an annotation was lost
                boxes.append([0.0, 0.0, 0.0, 0.0])
                labels.append(0)

        # 2. Append all visual-only nodes (like 'system', 'staff') missing from the graph
        for ann in json_data.get("annotations", []):
            if 'id' in ann and ann['id'] not in loaded_ids:
                if ann["class"] in self.class_to_idx:
                    boxes.append(ann["bbox"])
                    labels.append(self.class_to_idx[ann["class"]])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if edge_index.numel() > 0:
            edges_tensor = torch.cat([edge_index, edge_labels.unsqueeze(0)], dim=0).t()
        else:
            edges_tensor = torch.empty((0, 3), dtype=torch.long)

        return {
            "image": img_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "edges": edges_tensor,
            "file_name": sample["img"].name,
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