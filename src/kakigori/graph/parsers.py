# Standard library imports
import json
import xml.etree.ElementTree as ET

class GroundTruthGraphBuilder:
    def __init__(self, mei_file, json_files, node_roles):
        self.mei_tree = ET.parse(mei_file)
        self.mei_root = self.mei_tree.getroot()
        self.ns = {'mei': 'http://www.music-encoding.org/ns/mei'} 
        
        # Store the dynamic roles from structure.json
        self.roles = node_roles
        
        self.spatial_nodes = []
        if isinstance(json_files, str):
            json_files = [json_files]
            
        for j_file in json_files:
            with open(j_file, 'r') as f:
                data = json.load(f)
                self.spatial_nodes.extend(data.get('annotations', []))
            
        self.node_map = {node['id']: node for node in self.spatial_nodes if 'id' in node}
        self.gt_edges = []

    def _get_id(self, element):
        return element.attrib.get('{http://www.w3.org/XML/1998/namespace}id')

    def build_edges(self):
        """Dynamically traverses the MEI to build ground truth edges using structure.json."""
        temporal = set(self.roles["temporal_anchors"])
        structural = set(self.roles["structural_components"])
        modifier = set(self.roles["modifiers"])
        sync = set(self.roles["synchronization_text"])
        context = set(self.roles["context_globals"])

        # 1. Temporal Edges (Class 3) - Left-to-Right sequence
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

        # 2. Relational Edges (Parent-Child Hierarchy)
        parent_map = {c: p for p in self.mei_root.iter() for c in p}

        for child_el in self.mei_root.iter():
            child_id = self._get_id(child_el)
            if not child_id or child_id not in self.node_map: 
                continue
            
            child_class = self.node_map[child_id]['class']
            
            # Walk up to find the nearest valid MEI parent that ALSO exists in our visual node_map
            curr = child_el
            parent_id = None
            parent_class = None
            
            while curr in parent_map:
                p_el = parent_map[curr]
                p_id = self._get_id(p_el)
                if p_id and p_id in self.node_map:
                    parent_id = p_id
                    parent_class = self.node_map[p_id]['class']
                    break # Found the immediate visual parent
                curr = p_el
                
            if parent_id and parent_class:
                # Rule A: Parent is Anchor (e.g., Note -> Stem, Note -> Accid, Note -> Syl)
                if parent_class in temporal:
                    if child_class in structural:
                        self.gt_edges.append((parent_id, child_id, 1))
                    elif child_class in modifier:
                        self.gt_edges.append((parent_id, child_id, 2))
                    elif child_class in sync:
                        self.gt_edges.append((parent_id, child_id, 4))
                        
                # Rule B: Parent is Structural/Modifier holding an Anchor (e.g., Beam -> Note)
                elif child_class in temporal:
                    if parent_class in structural:
                        self.gt_edges.append((child_id, parent_id, 1)) 
                    elif parent_class in modifier:
                        self.gt_edges.append((child_id, parent_id, 2)) 
                    elif parent_class in temporal:
                        # e.g., Chord -> Note
                        self.gt_edges.append((parent_id, child_id, 1)) 

        # Deduplicate edges just in case MEI traversal hit redundancies
        self.gt_edges = list(set(self.gt_edges))
        return self.gt_edges

    def get_pyg_labels(self, candidate_edge_index, node_id_list):
        # ... (Keep your existing get_pyg_labels logic here) ...
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