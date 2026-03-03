def map_gt_to_candidates(candidate_edge_index, gt_targets):
    """
    Assigns the correct class label (0-4) to each proposed candidate edge.
    gt_targets: Tensor of (E, 3) where columns are [u, v, edge_class]
    """
    num_candidates = candidate_edge_index.shape[1]
    y_tensor = torch.zeros(num_candidates, dtype=torch.long, device=candidate_edge_index.device)
    
    if gt_targets.numel() == 0:
        return y_tensor # All candidates are Class 0 (No Edge)
        
    # Build a fast dictionary of Ground Truth edges
    gt_dict = {
        (u.item(), v.item()): edge_class.item() 
        for u, v, edge_class in gt_targets
    }
    
    for i in range(num_candidates):
        u = candidate_edge_index[0, i].item()
        v = candidate_edge_index[1, i].item()
        
        if (u, v) in gt_dict:
            y_tensor[i] = gt_dict[(u, v)]
            
    return y_tensor