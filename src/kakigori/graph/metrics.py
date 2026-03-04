import networkx as nx
import torch
import numpy as np

class GraphTopologyEvaluator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.delta_components = []
        self.delta_triangles = []
        self.delta_density = []

    def update(self, edge_index, gt_edges, pred_edges, num_nodes):
        """
        edge_index: (2, E) tensor of all candidate edges
        gt_edges: (E) tensor of ground truth classes (0-4)
        pred_edges: (E) tensor of predicted classes (0-4)
        """
        # Filter out Class 0 (No Edge) to build the active graphs
        gt_mask = gt_edges > 0
        pred_mask = pred_edges > 0
        
        gt_active_edges = edge_index[:, gt_mask].cpu().numpy().T
        pred_active_edges = edge_index[:, pred_mask].cpu().numpy().T
        
        # Build NetworkX Undirected Graphs (useful for macro-topology)
        G_gt = nx.Graph()
        G_gt.add_nodes_from(range(num_nodes))
        G_gt.add_edges_from(gt_active_edges)
        
        G_pred = nx.Graph()
        G_pred.add_nodes_from(range(num_nodes))
        G_pred.add_edges_from(pred_active_edges)
        
        # 1. Connectivity Difference (Number of Subgraphs)
        # If the model predicts too few components, it is merging distinct notes together.
        # If it predicts too many, it is shattering notes into orphaned stems/flags.
        gt_comps = nx.number_connected_components(G_gt)
        pred_comps = nx.number_connected_components(G_pred)
        self.delta_components.append(abs(gt_comps - pred_comps))
        
        # 2. Triangle Motifs
        # Triangles often represent dense, mutually modifying clusters (e.g., Notehead + Accidental + Stem)
        gt_tris = sum(nx.triangles(G_gt).values()) // 3
        pred_tris = sum(nx.triangles(G_pred).values()) // 3
        self.delta_triangles.append(abs(gt_tris - pred_tris))
        
        # 3. Graph Density
        gt_dens = nx.density(G_gt)
        pred_dens = nx.density(G_pred)
        self.delta_density.append(abs(gt_dens - pred_dens))

    def compute(self):
        return {
            "Mean Δ Components": np.mean(self.delta_components),
            "Mean Δ Triangles": np.mean(self.delta_triangles),
            "Mean Δ Density": np.mean(self.delta_density)
        }