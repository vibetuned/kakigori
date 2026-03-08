# Standard library imports
import json
import xml.etree.ElementTree as ET

class GroundTruthGraphBuilder:
    def __init__(self, mei_file, json_files, node_roles):
        self.mei_tree = ET.parse(mei_file)
        self.mei_root = self.mei_tree.getroot()
        self.ns = {"mei": "http://www.music-encoding.org/ns/mei"}
        self.roles = node_roles

        self.spatial_nodes = []
        if isinstance(json_files, str):
            json_files = [json_files]

        for j_file in json_files:
            with open(j_file, "r") as f:
                data = json.load(f)
                self.spatial_nodes.extend(data.get("annotations", []))

        self.node_map = {}
        self.gt_edges = [] # Initialize early so we can add fallback edgess

        for node in self.spatial_nodes:
            if "id" in node:
                base_id = node["id"]
                
                if base_id in self.node_map:
                    # ID Collision! Create a unique pseudo-ID for this sub-glyph
                    pseudo_id = f"{base_id}_{node['class']}_{len(self.node_map)}"
                    node["id"] = pseudo_id 
                    self.node_map[pseudo_id] = node
                    
                    # Automatically link this SMuFL glyph to its parent structural box
                    self.gt_edges.append((base_id, pseudo_id, 1))
                else:
                    self.node_map[base_id] = node

    def _get_id(self, element):
        return element.attrib.get("{http://www.w3.org/XML/1998/namespace}id")

    def _is_inside(self, inner_bbox, outer_bbox):
        """Checks if the center of the inner_bbox is contained within the outer_bbox."""
        cx = (inner_bbox[0] + inner_bbox[2]) / 2.0
        cy = (inner_bbox[1] + inner_bbox[3]) / 2.0
        return (outer_bbox[0] <= cx <= outer_bbox[2]) and (outer_bbox[1] <= cy <= outer_bbox[3])

    def build_edges(self):
        temporal = set(self.roles["temporal_anchors"])
        modifier = set(self.roles["modifiers"])
        sync = set(self.roles["synchronization_text"])
        context = set(self.roles["context_globals"])

        # 1. Temporal Edges (Class 3) - Left-to-Right sequence within layers
        for layer in self.mei_root.findall('.//mei:layer', self.ns):
            events = layer.findall('./*') 
            valid_sequence = []
            
            for ev in events:
                ev_id = self._get_id(ev)
                if ev_id in self.node_map:
                    cls = self.node_map[ev_id]['class']
                    if cls in temporal or cls in context:
                        valid_sequence.append(ev_id)
                        
            for i in range(len(valid_sequence) - 1):
                self.gt_edges.append((valid_sequence[i], valid_sequence[i+1], 3))

        # 2. Strict XML Hierarchy (Class 1, 2, 4)
        # This guarantees Measure -> Staff -> Layer -> Note regardless of bounding box overlaps
        # (Inside GroundTruthGraphBuilder.build_edges)
        parent_map = {c: p for p in self.mei_root.iter() for c in p}

        for child_el in self.mei_root.iter():
            child_id = self._get_id(child_el)
            if not child_id or child_id not in self.node_map: continue
            
            child_class = self.node_map[child_id]['class']
            curr, parent_id = child_el, None
            
            while curr in parent_map:
                p_el = parent_map[curr]
                p_id = self._get_id(p_el)
                if p_id and p_id in self.node_map:
                    parent_id = p_id
                    break
                curr = p_el
                
            if parent_id:
                if child_class in modifier: self.gt_edges.append((parent_id, child_id, 2))
                elif child_class in sync: self.gt_edges.append((parent_id, child_id, 4))
                else: self.gt_edges.append((parent_id, child_id, 1))

        # Spatial Fallback for System -> Measure
        systems = [n for n in self.spatial_nodes if n.get('class') == 'system']
        for measure in [n for n in self.spatial_nodes if n.get('class') == 'measure']:
            m_cy = (measure['bbox'][1] + measure['bbox'][3]) / 2
            for sys in systems:
                if sys['bbox'][1] <= m_cy <= sys['bbox'][3]:
                    self.gt_edges.append((sys['id'], measure['id'], 1))
                    break 

        # Spatial Fallback for Staff -> Clefs/KeySigs
        staves = [n for n in self.spatial_nodes if n.get('class') == 'staff']
        existing_children = {child for parent, child, edge_class in self.gt_edges}
        for node in self.spatial_nodes:
            if node['class'] in context and node['id'] not in existing_children:
                n_cy = (node['bbox'][1] + node['bbox'][3]) / 2.0
                for st in staves:
                    if st['bbox'][1] <= n_cy <= st['bbox'][3]:
                        self.gt_edges.append((st['id'], node['id'], 1))
                        break

        return self.gt_edges

    def get_pyg_labels(self, candidate_edge_index, node_id_list):
        import torch

        y = torch.zeros(candidate_edge_index.shape[1], dtype=torch.long)
        gt_edge_dict = {(u, v): edge_type for u, v, edge_type in self.gt_edges}
        for i in range(candidate_edge_index.shape[1]):
            u_idx = candidate_edge_index[0, i].item()
            v_idx = candidate_edge_index[1, i].item()
            u_id = node_id_list[u_idx]
            v_id = node_id_list[v_idx]
            if (u_id, v_id) in gt_edge_dict:
                y[i] = gt_edge_dict[(u_id, v_id)]
        return y