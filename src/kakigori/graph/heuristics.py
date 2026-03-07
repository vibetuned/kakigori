def generate_text_candidate_edges(boxes, labels, class_to_idx):
    """
    Generates candidate synchronization edges between Lyrics/Dynamics and Noteheads.
    boxes: Tensor of (N, 4) in [cx, cy, w, h] format
    labels: Tensor of (N) class indices
    """
    lyric_idx = class_to_idx["Lyric"]
    notehead_idx = class_to_idx["Notehead"]

    # Find the indices of all lyrics and all noteheads in this specific measure system
    lyric_indices = (labels == lyric_idx).nonzero(as_tuple=True)[0]
    notehead_indices = (labels == notehead_idx).nonzero(as_tuple=True)[0]

    candidate_edges = []

    for l_idx in lyric_indices:
        l_box = boxes[l_idx]
        l_cx, l_cy, l_w = l_box[0], l_box[1], l_box[2]

        # Define the X-axis boundaries of the lyric word
        l_left = l_cx - (l_w / 2)
        l_right = l_cx + (l_w / 2)

        best_note_idx = -1
        min_y_dist = float("inf")

        for n_idx in notehead_indices:
            n_box = boxes[n_idx]
            n_cx, n_cy = n_box[0], n_box[1]

            # Rule 1: The center of the notehead must fall within the horizontal span of the word
            # (or be extremely close to it)
            if l_left <= n_cx <= l_right:
                # Rule 2: The notehead must be ABOVE the lyric (smaller Y coordinate)
                y_dist = l_cy - n_cy
                if 0 < y_dist < min_y_dist:
                    min_y_dist = y_dist
                    best_note_idx = n_idx

        # If we found a valid notehead directly above this lyric, create a candidate edge
        if best_note_idx != -1:
            # Add bidirectional candidate edges for the GNN to evaluate
            candidate_edges.append([l_idx.item(), best_note_idx.item()])
            candidate_edges.append([best_note_idx.item(), l_idx.item()])

    if candidate_edges:
        return torch.tensor(candidate_edges, dtype=torch.long).t()
    else:
        return torch.empty((2, 0), dtype=torch.long)



import torch
from torch_geometric.nn import knn_graph

def generate_axis_aware_edges(boxes, labels, class_to_idx, k_neighbors=3):
    """
    Generates orthogonal candidate edges with multi-staff lane routing and a small KNN safety net.
    boxes: Tensor of (N, 4) in [x1, y1, x2, y2] relative coordinates.
    """
    num_nodes = boxes.shape[0]
    device = boxes.device
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = boxes[:, 2] - boxes[:, 0]
    
    # -----------------------------------------------------------------------
    # PHASE 1: Vertical Axis (Structural & Modifier)
    # -----------------------------------------------------------------------
    dx_matrix = torch.abs(cx.unsqueeze(1) - cx.unsqueeze(0)) 
    avg_w = (w.unsqueeze(1) + w.unsqueeze(0)) / 2.0 
    
    vertical_mask = dx_matrix < (avg_w * 1.5)
    vertical_mask.fill_diagonal_(False)
    vertical_edges = vertical_mask.nonzero(as_tuple=False).t()

    # -----------------------------------------------------------------------
    # PHASE 2: Horizontal Axis (Staff-Aware Temporal Flow)
    # -----------------------------------------------------------------------
    staff_idx = class_to_idx.get('staff', -1)
    staff_mask = (labels == staff_idx)
    staff_boxes = boxes[staff_mask]
    
    node_staff_assignment = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    if staff_boxes.shape[0] > 1:
        # Sort staves vertically so lane 0 is always the top staff
        staff_y_centers = (staff_boxes[:, 1] + staff_boxes[:, 3]) / 2.0
        sorted_staves_idx = torch.argsort(staff_y_centers)
        
        # Assign every node to the vertically closest staff lane
        dy_to_staves = torch.abs(cy.unsqueeze(1) - staff_y_centers[sorted_staves_idx].unsqueeze(0))
        node_staff_assignment = torch.argmin(dy_to_staves, dim=1)

    # Boolean mask preventing cross-staff connections
    same_staff_mask = node_staff_assignment.unsqueeze(1) == node_staff_assignment.unsqueeze(0)

    # Directional raycast: dx_dir[i, j] > 0 means j is to the right of i
    dx_dir = cx.unsqueeze(0) - cx.unsqueeze(1) 
    dy_abs = torch.abs(cy.unsqueeze(0) - cy.unsqueeze(1)) 

    # Heavily penalize vertical deviation so the ray shoots straight across the staff
    ray_distance = dx_dir + (dy_abs * 3.0)
    
    # Mask out leftward nodes, perfectly stacked nodes, and cross-staff nodes
    invalid_horizontal = (dx_dir <= 1.0) | (~same_staff_mask)
    ray_distance.masked_fill_(invalid_horizontal, float('inf'))

    # Top-K connection for polyphonic voices within the same staff lane
    k_right = min(3, num_nodes - 1)
    nearest_dists, nearest_right_idx = torch.topk(ray_distance, k=k_right, dim=1, largest=False)
    
    valid_right_mask = nearest_dists != float('inf')
    
    u_horiz = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, k_right)[valid_right_mask]
    v_horiz = nearest_right_idx[valid_right_mask]

    # Bidirectional temporal edges
    horizontal_edges_fwd = torch.stack([u_horiz, v_horiz], dim=0)
    horizontal_edges_rev = torch.stack([v_horiz, u_horiz], dim=0)

    # -----------------------------------------------------------------------
    # PHASE 3: KNN Safety Net & Combine
    # -----------------------------------------------------------------------
    pos = torch.stack([cx, cy], dim=1)
    k_knn = min(k_neighbors, num_nodes - 1)
    # knn_edges = knn_graph(pos, k=k_knn, loop=False)
    knn_edges = pure_pytorch_knn_graph(pos, k=k_knn, loop=False)
    combined_edges = torch.cat([
        vertical_edges, 
        horizontal_edges_fwd, 
        horizontal_edges_rev,
        knn_edges
    ], dim=1)
    
    # Strict deduplication
    candidate_edge_index = torch.unique(combined_edges, dim=1)

    return candidate_edge_index



def map_gt_to_candidates(candidate_edge_index, gt_targets):
    """
    Assigns the correct class label (0-4) to each proposed candidate edge.
    gt_targets: Tensor of (E, 3) where columns are [u, v, edge_class]
    """
    num_candidates = candidate_edge_index.shape[1]
    y_tensor = torch.zeros(
        num_candidates, dtype=torch.long, device=candidate_edge_index.device
    )

    if gt_targets.numel() == 0:
        return y_tensor  # All candidates are Class 0 (No Edge)

    # Build a fast dictionary of Ground Truth edges
    gt_dict = {
        (u.item(), v.item()): edge_class.item() for u, v, edge_class in gt_targets
    }

    for i in range(num_candidates):
        u = candidate_edge_index[0, i].item()
        v = candidate_edge_index[1, i].item()

        if (u, v) in gt_dict:
            y_tensor[i] = gt_dict[(u, v)]

    return y_tensor

import torch

def pure_pytorch_knn_graph(pos, k, loop=False):
    """
    A pure PyTorch replacement for torch_geometric.nn.knn_graph.
    pos: Tensor of shape (N, D) containing node coordinates.
    k: Number of nearest neighbors.
    loop: Whether to include self-loops.
    """
    num_nodes = pos.shape[0]
    device = pos.device
    
    # K cannot be larger than the available neighbors
    k = min(k, num_nodes - 1 if not loop else num_nodes)
    if k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # 1. Calculate pairwise Euclidean distance matrix (N, N)
    dist_matrix = torch.cdist(pos, pos, p=2.0)

    # 2. Mask out self-loops if requested
    if not loop:
        dist_matrix.fill_diagonal_(float('inf'))

    # 3. Get the indices of the top K smallest distances
    # largest=False means we want the smallest distances (nearest neighbors)
    _, topk_indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)

    # 4. Construct the edge_index tensor (Shape: [2, N * K])
    # The source nodes are simply [0, 0, 0, 1, 1, 1, ...]
    source_nodes = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, k).flatten()
    target_nodes = topk_indices.flatten()

    edge_index = torch.stack([source_nodes, target_nodes], dim=0)

    return edge_index