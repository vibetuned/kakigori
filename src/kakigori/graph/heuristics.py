

def generate_text_candidate_edges(boxes, labels, class_to_idx):
    """
    Generates candidate synchronization edges between Lyrics/Dynamics and Noteheads.
    boxes: Tensor of (N, 4) in [cx, cy, w, h] format
    labels: Tensor of (N) class indices
    """
    lyric_idx = class_to_idx['Lyric']
    notehead_idx = class_to_idx['Notehead']
    
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
        min_y_dist = float('inf')
        
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

from torch_geometric.nn import knn_graph

def generate_axis_aware_edges(boxes, labels, k_neighbors=15):
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
    knn_edges = knn_graph(pos, k=k, loop=False) # Shape: (2, E_knn)
    
    # 2. Vertical Raycasting (X-Overlap)
    # Captures elements perfectly stacked on top of each other (chords, far-away lyrics, high stems)
    # Vectorized check: max(x1_a, x1_b) < min(x2_a, x2_b)
    x1 = boxes[:, 0].unsqueeze(1) # Shape: (N, 1)
    x2 = boxes[:, 2].unsqueeze(1) # Shape: (N, 1)
    
    overlap_mask = torch.max(x1, x1.T) < torch.min(x2, x2.T)
    
    # Remove self-loops (a node overlaps with itself)
    overlap_mask.fill_diagonal_(False)
    
    # Convert boolean mask to edge indices
    vertical_edges = overlap_mask.nonzero(as_tuple=False).t() # Shape: (2, E_vert)
    
    # 3. Combine and Deduplicate
    combined_edges = torch.cat([knn_edges, vertical_edges], dim=1)
    
    # Ensure edges are strictly unique so the GNN doesn't process redundant messages
    candidate_edge_index = torch.unique(combined_edges, dim=1)
    
    return candidate_edge_index