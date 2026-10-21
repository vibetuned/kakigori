import networkx as nx
import torch
import numpy as np

EVENT_NAMES = ("note", "chord", "mRest")
EVENT_PREFIXES = ("rest",)
PRUNE_NAMES = ("note", "chord", "mRest", "measure")
PRUNE_PREFIXES = ("rest", "barline")
BEAM_TUPLET_NAMES = ("beam8", "beam16", "beam32", "beamBroken", "tuplet")


class GraphTopologyEvaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.delta_components = []
        self.delta_triangles = []
        self.delta_density = []
        # OMR-specific structural & temporal counters
        self.gt_reached_events = 0
        self.pred_recalled_events = 0
        self.pred_reached_events = 0
        self.multi_staff_events = 0
        self.checked_tree_nodes = 0
        self.multi_owner_nodes = 0
        self.active_temporal_nodes = 0
        self.branch_temporal_nodes = 0
        # Keep-head (node pruning) counters on prunable classes
        self.prune_tp = 0
        self.prune_fp = 0
        self.prune_fn = 0
        self.prune_tn = 0

    def update(self, edge_index, gt_edges, pred_edges, num_nodes,
               node_labels=None, class_to_idx=None, node_logits=None, node_keep=None):
        """
        edge_index: (2, E) tensor of all candidate edges
        gt_edges: (E) tensor of ground truth classes (0-5)
        pred_edges: (E) tensor of predicted classes (0-5)
        """
        edges_np = edge_index.detach().cpu().numpy()
        gt_np = gt_edges.detach().cpu().numpy()
        pred_np = pred_edges.detach().cpu().numpy()

        # Filter out Class 0 (No Edge) to build the active graphs
        gt_mask = gt_np > 0
        pred_mask = pred_np > 0

        gt_active_edges = edges_np[:, gt_mask].T
        pred_active_edges = edges_np[:, pred_mask].T

        # Build NetworkX Undirected Graphs (useful for macro-topology)
        G_gt = nx.Graph()
        G_gt.add_nodes_from(range(num_nodes))
        G_gt.add_edges_from(gt_active_edges)

        G_pred = nx.Graph()
        G_pred.add_nodes_from(range(num_nodes))
        G_pred.add_edges_from(pred_active_edges)

        # 1. Connectivity Difference (Number of Subgraphs)
        gt_comps = nx.number_connected_components(G_gt)
        pred_comps = nx.number_connected_components(G_pred)
        self.delta_components.append(abs(gt_comps - pred_comps))

        # 2. Triangle Motifs
        gt_tris = sum(nx.triangles(G_gt).values()) // 3
        pred_tris = sum(nx.triangles(G_pred).values()) // 3
        self.delta_triangles.append(abs(gt_tris - pred_tris))

        # 3. Graph Density
        gt_dens = nx.density(G_gt)
        pred_dens = nx.density(G_pred)
        self.delta_density.append(abs(gt_dens - pred_dens))

        # 4. Temporal chain branching (Class 3: in-degree <= 1 and out-degree <= 1)
        t_edges = edges_np[:, pred_np == 3]
        if t_edges.shape[1] > 0:
            out_deg = np.bincount(t_edges[0], minlength=num_nodes)
            in_deg = np.bincount(t_edges[1], minlength=num_nodes)
            active = (out_deg > 0) | (in_deg > 0)
            branched = (out_deg > 1) | (in_deg > 1)
            self.active_temporal_nodes += int(active.sum())
            self.branch_temporal_nodes += int(branched.sum())

        # 5. Structural tree & staff-reachability metrics (Class 1)
        labels_np = node_labels.detach().cpu().numpy() if node_labels is not None else None
        if labels_np is not None and class_to_idx is not None and num_nodes > 0:
            staff_id = class_to_idx.get("staff", -1)
            layer_id = class_to_idx.get("layer", -1)
            event_ids = {
                idx for name, idx in class_to_idx.items()
                if name in EVENT_NAMES or name.startswith(EVENT_PREFIXES)
            }
            beam_tuplet_ids = {
                class_to_idx[name] for name in BEAM_TUPLET_NAMES if name in class_to_idx
            }
            container_ids = ({layer_id} if layer_id >= 0 else set()) | beam_tuplet_ids

            staff_nodes = np.flatnonzero(labels_np == staff_id)
            event_mask = np.isin(labels_np, list(event_ids))

            if len(staff_nodes) > 0 and event_mask.any():
                gt_s1 = edges_np[:, gt_np == 1]
                pred_s1 = edges_np[:, pred_np == 1]

                def _staff_reach(s1_edges):
                    adj = np.zeros((num_nodes, num_nodes), dtype=bool)
                    if s1_edges.shape[1] > 0:
                        adj[s1_edges[0], s1_edges[1]] = True
                    reach = np.zeros((num_nodes, len(staff_nodes)), dtype=bool)
                    reach[staff_nodes, np.arange(len(staff_nodes))] = True
                    for _ in range(5):
                        reach |= adj.T @ reach
                    return reach

                gt_reach = _staff_reach(gt_s1)[event_mask]
                pred_reach = _staff_reach(pred_s1)[event_mask]
                gt_has = gt_reach.sum(axis=1) >= 1
                pred_cnt = pred_reach.sum(axis=1)
                pred_has = pred_cnt >= 1

                self.gt_reached_events += int(gt_has.sum())
                self.pred_recalled_events += int((gt_has & pred_has).sum())
                self.pred_reached_events += int(pred_has.sum())
                self.multi_staff_events += int((pred_cnt >= 2).sum())

            # Multi-owner check on layers (<=1 staff), beams/tuplets (<=1 layer),
            # and events (<=1 layer/beam/tuplet container)
            pred_s1 = edges_np[:, pred_np == 1]
            if pred_s1.shape[1] > 0:
                u_cls = labels_np[pred_s1[0]]
                v_idx = pred_s1[1]

                if layer_id >= 0 and staff_id >= 0:
                    layer_nodes = np.flatnonzero(labels_np == layer_id)
                    staff_to_node = np.bincount(v_idx[u_cls == staff_id], minlength=num_nodes)
                    self.checked_tree_nodes += len(layer_nodes)
                    self.multi_owner_nodes += int((staff_to_node[layer_nodes] > 1).sum())

                if beam_tuplet_ids and layer_id >= 0:
                    bt_nodes = np.flatnonzero(np.isin(labels_np, list(beam_tuplet_ids)))
                    layer_to_node = np.bincount(v_idx[u_cls == layer_id], minlength=num_nodes)
                    self.checked_tree_nodes += len(bt_nodes)
                    self.multi_owner_nodes += int((layer_to_node[bt_nodes] > 1).sum())

                if container_ids and event_mask.any():
                    ev_nodes = np.flatnonzero(event_mask)
                    cont_mask = np.isin(u_cls, list(container_ids))
                    cont_to_node = np.bincount(v_idx[cont_mask], minlength=num_nodes)
                    self.checked_tree_nodes += len(ev_nodes)
                    self.multi_owner_nodes += int((cont_to_node[ev_nodes] > 1).sum())

        # 6. Keep-head (node pruning) precision/recall on false positives (keep == 0)
        if node_logits is not None and node_keep is not None:
            logits_np = node_logits.detach().float().cpu().numpy()
            keep_np = node_keep.detach().float().cpu().numpy()
            valid = keep_np >= 0
            if labels_np is not None and class_to_idx is not None:
                prune_ids = [
                    idx for name, idx in class_to_idx.items()
                    if name in PRUNE_NAMES or name.startswith(PRUNE_PREFIXES)
                ]
                valid = valid & np.isin(labels_np, prune_ids)
            if valid.any():
                pred_drop = logits_np[valid] < 0.0  # sigmoid < 0.5
                gt_drop = keep_np[valid] == 0.0
                self.prune_tp += int((pred_drop & gt_drop).sum())
                self.prune_fp += int((pred_drop & ~gt_drop).sum())
                self.prune_fn += int((~pred_drop & gt_drop).sum())
                self.prune_tn += int((~pred_drop & ~gt_drop).sum())

    def compute(self):
        out = {
            "Mean Δ Components": float(np.mean(self.delta_components)) if self.delta_components else 0.0,
            "Mean Δ Triangles": float(np.mean(self.delta_triangles)) if self.delta_triangles else 0.0,
            "Mean Δ Density": float(np.mean(self.delta_density)) if self.delta_density else 0.0,
        }
        if self.gt_reached_events > 0:
            out["Event Reach Recall"] = self.pred_recalled_events / self.gt_reached_events
        if self.pred_reached_events > 0:
            out["Multi Staff Event Rate"] = self.multi_staff_events / self.pred_reached_events
        if self.checked_tree_nodes > 0:
            out["Multi Owner Rate"] = self.multi_owner_nodes / self.checked_tree_nodes
        if self.active_temporal_nodes > 0:
            out["Temporal Branch Rate"] = self.branch_temporal_nodes / self.active_temporal_nodes
        total_prune = self.prune_tp + self.prune_fp + self.prune_fn + self.prune_tn
        if total_prune > 0:
            prec = self.prune_tp / max(1, self.prune_tp + self.prune_fp)
            rec = self.prune_tp / max(1, self.prune_tp + self.prune_fn)
            f1 = 2 * prec * rec / max(1e-8, prec + rec)
            out["Prune FP Precision"] = prec
            out["Prune FP Recall"] = rec
            out["Prune FP F1"] = f1
            out["Keep Node Acc"] = (self.prune_tp + self.prune_tn) / total_prune
        return out