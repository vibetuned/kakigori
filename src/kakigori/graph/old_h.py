def generate_axis_aware_edges_old(boxes, labels, class_to_idx):
    """
    Generates candidate edges using Orthogonal Raycasting (Vertical Proximity + Horizontal Flow).
    boxes: Tensor of (N, 4) in [x1, y1, x2, y2] coordinates.
    labels: Tensor of (N) class indices (unused in this geometry-only pass, but kept for signature).
    """
    num_nodes = boxes.shape[0]
    device = boxes.device
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # 1. Extract geometric features
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    # -----------------------------------------------------------------------
    # PHASE 1: Vertical Axis (Structural & Modifier Edges)
    # -----------------------------------------------------------------------
    # Calculate pairwise absolute distance in X
    # dx_matrix[i, j] = distance between center of i and center of j
    dx_matrix = torch.abs(cx.unsqueeze(1) - cx.unsqueeze(0))  # Shape: (N, N)
    
    # Calculate pairwise average width
    avg_w = (w.unsqueeze(1) + w.unsqueeze(0)) / 2.0  # Shape: (N, N)
    
    # A structural group (Notehead + Stem + Accidental + Dot) lives in a tight vertical column.
    # We connect nodes if their X-centers are within 1.5x their average width.
    # This safely catches accidentals that sit just to the left of the notehead.
    vertical_mask = dx_matrix < (avg_w * 1.5)
    
    # Remove self-loops
    vertical_mask.fill_diagonal_(False)
    vertical_edges = vertical_mask.nonzero(as_tuple=False).t()  # Shape: (2, E_vert)

    # -----------------------------------------------------------------------
    # PHASE 2: Horizontal Axis (Temporal Flow Edges)
    # -----------------------------------------------------------------------
    # Calculate directed distance in X: dx_dir[i, j] = cx[j] - cx[i]
    # Positive means node 'j' is to the RIGHT of node 'i'.
    dx_dir = cx.unsqueeze(0) - cx.unsqueeze(1)  # Shape: (N, N)
    dy_abs = torch.abs(cy.unsqueeze(0) - cy.unsqueeze(1))  # Shape: (N, N)
    
    avg_h = (h.unsqueeze(1) + h.unsqueeze(0)) / 2.0
    
    # Valid targets for a horizontal raycast must be:
    # 1. To the right (dx > 0)
    # 2. On the same rough vertical staff level (dy < 3x average height). 
    #    This prevents connecting a treble clef note to a bass clef note in piano music.
    valid_right_mask = (dx_dir > 0) & (dy_abs < (avg_h * 3.0))

    # We want the *closest* node to the right. 
    # We heavily penalize Y-deviation so the ray shoots straight across.
    ray_distance = dx_dir + (dy_abs * 5.0)
    ray_distance[~valid_right_mask] = float('inf')

    # Find the index of the minimum distance candidate for each node
    nearest_right_idx = torch.argmin(ray_distance, dim=1)
    
    # Ensure a valid rightward candidate actually exists (e.g., the last note has no right neighbor)
    has_right_neighbor = valid_right_mask.any(dim=1)
    
    u_horiz = torch.arange(num_nodes, device=device)[has_right_neighbor]
    v_horiz = nearest_right_idx[has_right_neighbor]
    
    # Temporal flow needs to pass messages both forward and backward in the GNN
    horizontal_edges_fwd = torch.stack([u_horiz, v_horiz], dim=0)
    horizontal_edges_rev = torch.stack([v_horiz, u_horiz], dim=0)

    # -----------------------------------------------------------------------
    # PHASE 3: Combine and Deduplicate
    # -----------------------------------------------------------------------
    combined_edges = torch.cat([vertical_edges, horizontal_edges_fwd, horizontal_edges_rev], dim=1)
    
    # Guarantee strict uniqueness so GATv2Conv doesn't aggregate redundant messages
    candidate_edge_index = torch.unique(combined_edges, dim=1)

    return candidate_edge_index


# Third party imports
from torch_geometric.nn import knn_graph


def generate_axis_aware_edges_knn(boxes, labels, k_neighbors=15):
    """
    Generates candidate edges using K-Nearest Neighbors + Vertical X-Overlap.
    boxes: Tensor of (N, 4) in [x1, y1, x2, y2] relative coordinates.
    """
    num_nodes = boxes.shape[0]
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=boxes.device)

    # 1. Base Proximity Graph (KNN)
    # Captures sequential temporal flow and tight structural clusters (notehead + accidental)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    pos = torch.stack([cx, cy], dim=1)

    k = min(k_neighbors, num_nodes - 1)
    knn_edges = knn_graph(pos, k=k, loop=False)  # Shape: (2, E_knn)

    # 2. Vertical Raycasting (X-Overlap)
    # Captures elements perfectly stacked on top of each other (chords, far-away lyrics, high stems)
    # Vectorized check: max(x1_a, x1_b) < min(x2_a, x2_b)
    x1 = boxes[:, 0].unsqueeze(1)  # Shape: (N, 1)
    x2 = boxes[:, 2].unsqueeze(1)  # Shape: (N, 1)

    overlap_mask = torch.max(x1, x1.T) < torch.min(x2, x2.T)

    # Remove self-loops (a node overlaps with itself)
    overlap_mask.fill_diagonal_(False)

    # Convert boolean mask to edge indices
    vertical_edges = overlap_mask.nonzero(as_tuple=False).t()  # Shape: (2, E_vert)

    # 3. Combine and Deduplicate
    combined_edges = torch.cat([knn_edges, vertical_edges], dim=1)

    # Ensure edges are strictly unique so the GNN doesn't process redundant messages
    candidate_edge_index = torch.unique(combined_edges, dim=1)

    return candidate_edge_index

import torch

import torch
from torch_geometric.nn import knn_graph

def generate_axis_aware_edges_knn2(boxes, labels, k_neighbors=10):
    """
    Generates candidate edges using Orthogonal Raycasting + KNN Safety Net.
    boxes: Tensor of (N, 4) in [x1, y1, x2, y2] relative system coordinates.
    labels: Tensor of (N) class indices.
    """
    num_nodes = boxes.shape[0]
    device = boxes.device
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # Extract geometric features
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    # -----------------------------------------------------------------------
    # PHASE 1: Vertical Axis (Structural Groups & Stacked Chords)
    # -----------------------------------------------------------------------
    dx_matrix = torch.abs(cx.unsqueeze(1) - cx.unsqueeze(0)) 
    avg_w = (w.unsqueeze(1) + w.unsqueeze(0)) / 2.0 
    
    vertical_mask = dx_matrix < (avg_w * 1.5)
    vertical_mask.fill_diagonal_(False)
    vertical_edges = vertical_mask.nonzero(as_tuple=False).t() # Shape: (2, E_vert)

    # -----------------------------------------------------------------------
    # PHASE 2: Horizontal Axis (Forward Temporal Flow)
    # -----------------------------------------------------------------------
    # dx_dir[i, j] = cx[j] - cx[i] (Positive means 'j' is to the RIGHT of 'i')
    dx_dir = cx.unsqueeze(0) - cx.unsqueeze(1) 
    dy_abs = torch.abs(cy.unsqueeze(0) - cy.unsqueeze(1)) 

    # We penalize Y-deviation to prioritize straight lines, but we DO NOT strictly 
    # threshold it, allowing large melodic leaps (octaves) to still connect.
    ray_distance = dx_dir + (dy_abs * 3.0)
    
    # Mask out nodes that are to the left, or are literally stacked on top of each other (dx <= 1 pixel)
    ray_distance[dx_dir <= 1.0] = float('inf')

    # FIX: Use top-K to connect to multiple notes in the next chord (Polyphony)
    k_right = min(3, num_nodes - 1)
    
    # largest=False gets the SMALLEST ray distances
    nearest_dists, nearest_right_idx = torch.topk(ray_distance, k=k_right, dim=1, largest=False)
    
    # Filter out the 'inf' padding where a node didn't have K neighbors to its right
    valid_right_mask = nearest_dists != float('inf')
    
    u_horiz = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, k_right)[valid_right_mask]
    v_horiz = nearest_right_idx[valid_right_mask]

    # Temporal flow needs bidirectional candidate paths for the GNN
    horizontal_edges_fwd = torch.stack([u_horiz, v_horiz], dim=0)
    horizontal_edges_rev = torch.stack([v_horiz, u_horiz], dim=0)

    # -----------------------------------------------------------------------
    # PHASE 3: KNN Safety Net (Long-Range Modifiers like Slurs/Hairpins)
    # -----------------------------------------------------------------------
    pos = torch.stack([cx, cy], dim=1)
    k_knn = min(k_neighbors, num_nodes - 1)
    knn_edges = knn_graph(pos, k=k_knn, loop=False)

    # -----------------------------------------------------------------------
    # PHASE 4: Combine and Deduplicate
    # -----------------------------------------------------------------------
    combined_edges = torch.cat([
        vertical_edges, 
        horizontal_edges_fwd, 
        horizontal_edges_rev,
        knn_edges
    ], dim=1)
    
    # Guarantee strict uniqueness so GATv2Conv doesn't aggregate redundant messages
    candidate_edge_index = torch.unique(combined_edges, dim=1)

    return candidate_edge_index

import torch
from torch_geometric.nn import knn_graph

def generate_axis_aware_edges_knn3(boxes, labels, class_to_idx, k_neighbors=10):
    num_nodes = boxes.shape[0]
    device = boxes.device
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

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
    # A. Identify the Staves to create parallel temporal lanes
    staff_idx = class_to_idx.get('staff', -1)
    staff_mask = (labels == staff_idx)
    staff_boxes = boxes[staff_mask]
    
    # Default everything to lane 0 (in case no staves are detected)
    node_staff_assignment = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    num_staves = staff_boxes.shape[0]
    if num_staves > 1:
        # Sort staves vertically from top to bottom
        staff_y_centers = (staff_boxes[:, 1] + staff_boxes[:, 3]) / 2.0
        sorted_staves_idx = torch.argsort(staff_y_centers)
        
        # Assign every node to the vertically closest staff lane
        dy_to_staves = torch.abs(cy.unsqueeze(1) - staff_y_centers[sorted_staves_idx].unsqueeze(0))
        node_staff_assignment = torch.argmin(dy_to_staves, dim=1)

    # B. Create a boolean mask preventing cross-staff connections
    same_staff_mask = node_staff_assignment.unsqueeze(1) == node_staff_assignment.unsqueeze(0)

    # C. Calculate directional raycast
    dx_dir = cx.unsqueeze(0) - cx.unsqueeze(1) 
    dy_abs = torch.abs(cy.unsqueeze(0) - cy.unsqueeze(1)) 

    ray_distance = dx_dir + (dy_abs * 3.0)
    
    # D. Apply the masks: Invalid if to the left, stacked, OR in a different staff
    invalid_horizontal = (dx_dir <= 1.0) | (~same_staff_mask)
    ray_distance[invalid_horizontal] = float('inf')

    # E. Top-K connection for polyphonic voices within the same staff lane
    k_right = min(3, num_nodes - 1)
    nearest_dists, nearest_right_idx = torch.topk(ray_distance, k=k_right, dim=1, largest=False)
    
    valid_right_mask = nearest_dists != float('inf')
    
    u_horiz = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, k_right)[valid_right_mask]
    v_horiz = nearest_right_idx[valid_right_mask]

    horizontal_edges_fwd = torch.stack([u_horiz, v_horiz], dim=0)
    horizontal_edges_rev = torch.stack([v_horiz, u_horiz], dim=0)

    # -----------------------------------------------------------------------
    # PHASE 3: KNN Safety Net & Combine
    # -----------------------------------------------------------------------
    pos = torch.stack([cx, cy], dim=1)
    k_knn = min(k_neighbors, num_nodes - 1)
    knn_edges = knn_graph(pos, k=k_knn, loop=False)

    combined_edges = torch.cat([
        vertical_edges, 
        horizontal_edges_fwd, 
        horizontal_edges_rev,
        knn_edges
    ], dim=1)
    
    candidate_edge_index = torch.unique(combined_edges, dim=1)

    return candidate_edge_index