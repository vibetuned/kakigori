# Third party imports
# ... other imports ...
# Third party imports
import torch
from torch.optim import AdamW
from torch_geometric.data import Data


def slice_page_to_systems(
    full_image_tensor, gt_boxes, gt_labels, gt_edges, class_to_idx
):
    """
    Slices a full page into system-level crops, translates coordinates,
    and perfectly re-indexes the ground truth edges.
    """
    system_idx = class_to_idx["system"]  # From gelato_config.json
    system_mask = gt_labels == system_idx

    system_boxes = gt_boxes[system_mask]
    system_boxes = system_boxes[system_boxes[:, 1].argsort()]  # Sort top-to-bottom

    primitive_boxes = gt_boxes[~system_mask]
    primitive_labels = gt_labels[~system_mask]

    # We need to map the original full-page primitive index to its new system-level index
    # to rebuild the edge_index tensor correctly.
    full_to_system_index_map = {}

    system_datasets = []

    for sys_box in system_boxes:
        sx1, sy1, sx2, sy2 = sys_box

        # 1. Find primitives in this system
        centers_y = (primitive_boxes[:, 1] + primitive_boxes[:, 3]) / 2.0
        in_sys_mask = (centers_y >= sy1) & (centers_y <= sy2)

        sys_boxes = primitive_boxes[in_sys_mask].clone()
        sys_labels = primitive_labels[in_sys_mask]

        if len(sys_boxes) == 0:
            continue

        # 2. Map old indices to new indices for edge reconstruction
        original_indices = torch.nonzero(in_sys_mask, as_tuple=True)[0]
        current_sys_mapping = {
            old.item(): new for new, old in enumerate(original_indices)
        }
        full_to_system_index_map.update(current_sys_mapping)

        # 3. Translate spatial coordinates to relative system coordinates
        sys_boxes[:, 0] -= sx1
        sys_boxes[:, 1] -= sy1
        sys_boxes[:, 2] -= sx1
        sys_boxes[:, 3] -= sy1

        # 4. Crop the image tensor
        crop_x1, crop_y1, crop_x2, crop_y2 = map(int, [sx1, sy1, sx2, sy2])
        sys_crop = full_image_tensor[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

        # 5. Re-index the Ground Truth Edges
        # gt_edges is a list of tuples: (u_full_idx, v_full_idx, edge_class)
        sys_gt_edges = []
        for u, v, edge_class in gt_edges:
            if u in current_sys_mapping and v in current_sys_mapping:
                new_u = current_sys_mapping[u]
                new_v = current_sys_mapping[v]
                sys_gt_edges.append([new_u, new_v, edge_class])

        # Package it for the DataLoader
        system_datasets.append(
            {
                "image_crop": sys_crop,
                "boxes": sys_boxes,
                "labels": sys_labels,
                "edges": torch.tensor(sys_gt_edges, dtype=torch.long)
                if sys_gt_edges
                else torch.empty((0, 3), dtype=torch.long),
            }
        )

    return system_datasets


def train_phase3_end_to_end(
    detector, roi_extractor, gnn, full_page_dataloader, epochs=20
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = AdamW(
        [
            {"params": detector.parameters(), "lr": 1e-5},
            {"params": roi_extractor.parameters(), "lr": 5e-5},
            {"params": gnn.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )

    # [0: No Edge, 1: Struct, 2: Mod, 3: Temp, 4: Sync]
    alpha_weights = torch.tensor([0.05, 1.0, 2.5, 1.0, 4.0]).to(device)
    criterion = EdgeFocalLoss(alpha=alpha_weights)

    for epoch in range(epochs):
        detector.train()
        roi_extractor.train()
        gnn.train()

        # DataLoader yields FULL PAGES, not cropped systems
        for batch in full_page_dataloader:
            full_image = batch["image"].to(device)  # Shape: (1, C, H, W)
            abs_boxes = batch["boxes"].to(device)  # Absolute page coordinates
            labels = batch["labels"].to(device)
            gt_edges = batch["edges"]  # Full page edge list

            optimizer.zero_grad()

            # --- 1. VISION DOMAIN: Full Page Extraction ---
            # The detector stays in its native distribution!
            feat_maps = detector.extract_features(full_image)

            # We will accumulate the graph losses for all systems on this page
            page_graph_loss = 0.0

            # --- 2. BRIDGE: Isolate Systems ---
            # (Assuming a helper function that splits the page's boxes into system groups)
            # It returns the absolute boxes, labels, and the system's own bounding box
            system_groups = split_into_systems(
                abs_boxes, labels, gt_edges, class_to_idx
            )

            for sys_data in system_groups:
                sys_abs_boxes = sys_data["abs_boxes"]
                sys_labels = sys_data["labels"]
                sys_gt_targets = sys_data["edge_targets"]  # Re-indexed for this system
                sx1, sy1, sx2, sy2 = sys_data["system_bbox"]

                # A. Extract RoI Features using ABSOLUTE page coordinates
                roi_feats = roi_extractor(
                    feat_maps, [sys_abs_boxes], full_image.shape[-2:]
                )

                # B. Translate to RELATIVE coordinates purely for the GNN's spatial awareness
                sys_rel_boxes = sys_abs_boxes.clone()
                sys_rel_boxes[:, 0] -= sx1
                sys_rel_boxes[:, 1] -= sy1
                sys_rel_boxes[:, 2] -= sx1
                sys_rel_boxes[:, 3] -= sy1

                # --- 3. RELATIONAL DOMAIN: System Graph ---
                # Build Node Features (Visual + Relative Spatial + Class)
                class_embeds = gnn.class_embedding(sys_labels)
                node_x = torch.cat([roi_feats, sys_rel_boxes, class_embeds], dim=1)

                # Generate candidate edges using the relative boxes
                candidate_edge_index = generate_axis_aware_edges(
                    sys_rel_boxes, sys_labels
                )

                # GNN Forward & Loss accumulation
                edge_logits = gnn(node_x, candidate_edge_index)
                sys_loss = criterion(edge_logits, sys_gt_targets)

                page_graph_loss += sys_loss

            # --- 4. END-TO-END BACKPROPAGATION ---
            # The gradients flow from the combined system graphs, through the RoI extractor,
            # and gracefully update the full-page CNN feature maps.
            if page_graph_loss > 0:
                page_graph_loss.backward()
                optimizer.step()


# Third party imports
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader


def train_gnn_phase2(gnn_model, train_dataset, val_dataset, epochs=50, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model = gnn_model.to(device)

    # 1. PyG DataLoaders handle the complex batching of graphs automatically
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Optimizer for Phase 2 (Only GNN parameters)
    optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 3. Loss Function with your custom alpha weights
    alpha_weights = torch.tensor([0.05, 1.0, 2.5, 1.0, 4.0]).to(device)
    criterion = EdgeFocalLoss(alpha=alpha_weights, gamma=2.0, reduction="mean")

    for epoch in range(epochs):
        # --- TRAINING ---
        gnn_model.train()
        total_train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass: GNN predicts edge logits
            # batch.x shape: (Total Nodes in Batch, Features)
            # batch.edge_index shape: (2, Total Edges in Batch)
            edge_logits = gnn_model(batch.x, batch.edge_index)

            # Calculate focal loss against ground truth
            # batch.y shape: (Total Edges in Batch)
            loss = criterion(edge_logits, batch.y)

            # Backward pass & Optimize
            loss.backward()

            # Optional but highly recommended: Gradient clipping for GNN stability
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=2.0)

            optimizer.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION ---
        gnn_model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                edge_logits = gnn_model(batch.x, batch.edge_index)
                loss = criterion(edge_logits, batch.y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    return gnn_model


# Third party imports
import torch


def split_into_systems(abs_boxes, labels, gt_edges, class_to_idx):
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
