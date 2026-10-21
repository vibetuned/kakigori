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

# Boxes drawn around a span of the timeline: they only connect to what lies
# under them, and are never the next event to the right of anything
SPAN_CLASSES = ("system", "system-staff", "measure", "staff", "layer", "tuplet",
                "beam8", "beam16", "beam32", "beamBroken")
CONTAINER_CLASSES = ("system", "system-staff", "measure", "staff", "layer")
# Temporal anchors: the events the temporal (class 3) edges join
ANCHOR_CLASSES = ("note", "chord", "mRest")
ANCHOR_PREFIXES = ("rest",)
# Staff context drawn ahead of the first measure's span (clef, key and time
# signatures) or on its edge (barlines): exempt from the span rule
SPAN_EXEMPT_PREFIXES = ("clef", "keySig", "keyAccid", "meterSig", "timeSig", "barline")


def _class_mask(labels, class_to_idx, names):
    idx = [class_to_idx[c] for c in names if c in class_to_idx]
    return torch.isin(labels, torch.tensor(idx, dtype=labels.dtype, device=labels.device))


def _lane_centres(boxes, labels, class_to_idx):
    """The vertical centre of each staff row of a system, top to bottom.

    Rows are the staff cells' centres clustered (a new row where the gap
    exceeds half a cell height): cells exist in training and at inference
    alike, while system-staff rows never reach the GT training graphs (a row
    shares its id with its system's annotation). Detected rows serve only
    where a group has no cells."""
    cells = _class_mask(labels, class_to_idx, ("staff",))
    if not cells.any():
        cells = _class_mask(labels, class_to_idx, ("system-staff",))
    if not cells.any():
        return boxes.new_zeros(0)
    cell_cy = ((boxes[cells, 1] + boxes[cells, 3]) / 2.0).sort().values
    gap = 0.5 * (boxes[cells, 3] - boxes[cells, 1]).median()
    starts = torch.ones_like(cell_cy, dtype=torch.bool)
    starts[1:] = (cell_cy[1:] - cell_cy[:-1]) > gap
    row_of = torch.cumsum(starts.long(), 0) - 1
    n_rows = int(row_of[-1]) + 1
    sums = torch.zeros(n_rows, dtype=cell_cy.dtype, device=cell_cy.device).index_add_(0, row_of, cell_cy)
    counts = torch.bincount(row_of, minlength=n_rows).to(cell_cy.dtype)
    return sums / counts


def generate_axis_aware_edges(boxes, labels, class_to_idx, k_neighbors=3):
    """
    Candidate edges for the GNN to classify, from three geometric rules.
    boxes: Tensor of (N, 4) in [x1, y1, x2, y2] relative coordinates.

    1. Vertical (structural and modifier links): pairs whose centres are
       within 1.5x their mean width horizontally. A pair with a span box
       (container, beam, tuplet) needs the narrower box's centre within the
       wider one's x-span (padded by a quarter of its height, at most 10 px):
       a layer is a measure wide, and its 1.5x window used to reach the beams
       and notes of the neighbouring measures (the GNN linked enough of them
       that 18% of the detection path's notes were emitted twice).
    2. Horizontal (temporal flow): from each box, a ray to the 3 nearest
       boxes to its right in the same staff lane, and the same ray over the
       temporal anchors alone. Lanes are the system's staff rows
       (_lane_centres); a lane per staff CELL split one row into several
       wherever the cells' centres differed by a pixel (74% of the detected
       rows). Containers are neither sources nor targets: their centres sit
       mid-measure on the staff line and took the ray's slots.
    3. A kNN safety net (k_neighbors), under the same span rule.
    """
    num_nodes = boxes.shape[0]
    device = boxes.device
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = boxes[:, 2] - boxes[:, 0]

    # --- 1. Vertical axis (structural & modifier) ---
    dx_matrix = torch.abs(cx.unsqueeze(1) - cx.unsqueeze(0))
    avg_w = (w.unsqueeze(1) + w.unsqueeze(0)) / 2.0
    vertical_mask = dx_matrix < (avg_w * 1.5)
    vertical_mask.fill_diagonal_(False)

    span = _class_mask(labels, class_to_idx, SPAN_CLASSES)
    ok = None
    if span.any():
        # j under i's span: the narrower box's centre sits within the wider
        # box's x-span, padded by a quarter of the wider box's height (capped
        # at 10 px so tall measure/staff boxes do not reach into neighbouring
        # measures; detected beams and layers end a little short of their
        # outer noteheads).
        pad = ((boxes[:, 3] - boxes[:, 1]) / 4.0).clamp(max=10.0)
        in_span = (cx.unsqueeze(0) >= (boxes[:, 0] - pad).unsqueeze(1)) & (
            cx.unsqueeze(0) <= (boxes[:, 2] + pad).unsqueeze(1)
        )
        exempt = _class_mask(labels, class_to_idx, [
            c for c in class_to_idx if c.startswith(SPAN_EXEMPT_PREFIXES)
        ])
        in_span |= exempt.unsqueeze(0)
        i_wider = w.unsqueeze(1) >= w.unsqueeze(0)
        pair_in_span = torch.where(i_wider, in_span, in_span.t())
        ok = ~(span.unsqueeze(1) | span.unsqueeze(0)) | pair_in_span
        vertical_mask &= ok
    vertical_edges = vertical_mask.nonzero(as_tuple=False).t()

    # --- 2. Horizontal axis (staff-lane temporal flow) ---
    lane = torch.zeros(num_nodes, dtype=torch.long, device=device)
    lane_cy = _lane_centres(boxes, labels, class_to_idx)
    if lane_cy.numel() > 1:
        lane = torch.argmin(torch.abs(cy.unsqueeze(1) - lane_cy.unsqueeze(0)), dim=1)
    same_staff_mask = lane.unsqueeze(1) == lane.unsqueeze(0)

    # Directional raycast: dx_dir[i, j] > 0 means j is to the right of i
    dx_dir = cx.unsqueeze(0) - cx.unsqueeze(1)
    dy_abs = torch.abs(cy.unsqueeze(0) - cy.unsqueeze(1))

    # Heavily penalize vertical deviation so the ray shoots straight across the staff
    ray_distance = dx_dir + (dy_abs * 3.0)

    # Mask out leftward and stacked nodes, other lanes, and containers
    container = _class_mask(labels, class_to_idx, CONTAINER_CLASSES)
    invalid_horizontal = (dx_dir <= 1.0) | (~same_staff_mask)
    invalid_horizontal |= container.unsqueeze(0) | container.unsqueeze(1)
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

    # The same ray over the temporal anchors only: temporal edges join events,
    # and a note's own notehead, stem and beam otherwise take the 3 slots
    # (temporal candidate recall 87.5 -> 99.8% on GT boxes, 81.5 -> 99.9% on
    # detections, for 5% more candidates)
    anchor = _class_mask(labels, class_to_idx, [
        c for c in class_to_idx if c in ANCHOR_CLASSES or c.startswith(ANCHOR_PREFIXES)
    ])
    if anchor.sum() > 1:
        anchor_distance = ray_distance.masked_fill(
            ~(anchor.unsqueeze(0) & anchor.unsqueeze(1)), float('inf')
        )
        k_anchor = min(3, num_nodes - 1)
        a_dists, a_idx = torch.topk(anchor_distance, k=k_anchor, dim=1, largest=False)
        a_valid = a_dists != float('inf')
        a_u = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, k_anchor)[a_valid]
        a_v = a_idx[a_valid]
        horizontal_edges_fwd = torch.cat([horizontal_edges_fwd, torch.stack([a_u, a_v])], dim=1)
        horizontal_edges_rev = torch.cat([horizontal_edges_rev, torch.stack([a_v, a_u])], dim=1)

    # --- 3. KNN safety net & combine ---
    pos = torch.stack([cx, cy], dim=1)
    k_knn = min(k_neighbors, num_nodes - 1)
    knn_edges = pure_pytorch_knn_graph(pos, k=k_knn, loop=False)
    if ok is not None and knn_edges.numel() > 0:
        knn_edges = knn_edges[:, ok[knn_edges[0], knn_edges[1]]]
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
    Assigns the correct class label (0-5) to each proposed candidate edge:
    0 none, 1 structural, 2 modifier, 3 temporal, 4 sync-text, 5 simultaneity.
    gt_targets: Tensor of (E, 3) where columns are [u, v, edge_class]

    Vectorized: (u, v) pairs are encoded as scalar keys and matched with
    searchsorted. The previous per-candidate .item() loop caused thousands
    of GPU syncs per training step and dominated the step time.
    """
    num_candidates = candidate_edge_index.shape[1]
    y_tensor = torch.zeros(
        num_candidates, dtype=torch.long, device=candidate_edge_index.device
    )

    if gt_targets.numel() == 0 or num_candidates == 0:
        return y_tensor  # All candidates are Class 0 (No Edge)

    gt = gt_targets.to(candidate_edge_index.device)
    n = int(max(candidate_edge_index.max(), gt[:, :2].max()).item()) + 1
    cand_keys = candidate_edge_index[0].long() * n + candidate_edge_index[1].long()
    gt_keys = gt[:, 0].long() * n + gt[:, 1].long()

    sort_idx = torch.argsort(gt_keys)
    sorted_keys = gt_keys[sort_idx]
    pos = torch.searchsorted(sorted_keys, cand_keys).clamp(max=len(sorted_keys) - 1)
    matched = sorted_keys[pos] == cand_keys
    y_tensor[matched] = gt[sort_idx][pos[matched], 2].long()

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