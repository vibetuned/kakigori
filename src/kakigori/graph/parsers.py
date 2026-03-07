# Standard library imports
import json
import xml.etree.ElementTree as ET


class GroundTruthGraphBuilder:
    def __init__(self, mei_file, json_files, node_roles):
        self.mei_tree = ET.parse(mei_file)
        self.mei_root = self.mei_tree.getroot()
        self.ns = {"mei": "http://www.music-encoding.org/ns/mei"}

        # Store the dynamic roles from structure.json
        self.roles = node_roles

        self.spatial_nodes = []
        if isinstance(json_files, str):
            json_files = [json_files]

        for j_file in json_files:
            with open(j_file, "r") as f:
                data = json.load(f)
                self.spatial_nodes.extend(data.get("annotations", []))

        self.node_map = {
            node["id"]: node for node in self.spatial_nodes if "id" in node
        }
        self.gt_edges = []

    def _get_id(self, element):
        return element.attrib.get("{http://www.w3.org/XML/1998/namespace}id")

    def build_edges(self):
        """Dynamically traverses the MEI and spatial boxes to build ground truth edges."""
        temporal = set(self.roles["temporal_anchors"])
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

        # 2. Relational Edges (Parent-Child Hierarchy from MEI)
        parent_map = {c: p for p in self.mei_root.iter() for c in p}

        for child_el in self.mei_root.iter():
            child_id = self._get_id(child_el)
            if not child_id or child_id not in self.node_map: 
                continue
            
            child_class = self.node_map[child_id]['class']
            
            # Walk up to find the nearest visual parent
            curr = child_el
            parent_id = None
            parent_class = None
            
            while curr in parent_map:
                p_el = parent_map[curr]
                p_id = self._get_id(p_el)
                if p_id and p_id in self.node_map:
                    parent_id = p_id
                    parent_class = self.node_map[p_id]['class']
                    break
                curr = p_el
                
            if parent_id and parent_class:
                # Class 2 for Modifier (note -> accid)
                if child_class in modifier:
                    self.gt_edges.append((parent_id, child_id, 2))
                # Class 4 for Sync (note -> syl)
                elif child_class in sync:
                    self.gt_edges.append((parent_id, child_id, 4))
                # Class 1 for Structural Layout (measure -> staff -> layer -> note -> stem)
                else:
                    self.gt_edges.append((parent_id, child_id, 1))

        # 3. Macro-Layout System Linking (Spatial Fallback)
        # Because MEI uses <sb> milestones, systems don't enclose measures in the XML tree.
        # We link systems to measures using their bounding box geometry.
        systems = [n for n in self.spatial_nodes if n.get('class') == 'system']
        measures = [n for n in self.spatial_nodes if n.get('class') == 'measure']
        
        for measure in measures:
            mx1, my1, mx2, my2 = measure['bbox']
            m_cy = (my1 + my2) / 2
            
            for sys in systems:
                sx1, sy1, sx2, sy2 = sys['bbox']
                # If the measure's center Y falls within the system's vertical bounds
                if sy1 <= m_cy <= sy2:
                    self.gt_edges.append((sys['id'], measure['id'], 1))
                    break # A measure only belongs to one system

        # Deduplicate edges just in case MEI traversal hit redundancies
        # 4. Macro-Layout Context Linking (Orphaned Clefs and Signatures)
        staves = [n for n in self.spatial_nodes if n.get('class') == 'staff']
        
        for node in self.spatial_nodes:
            if node.get('class') in context:
                node_id = node['id']
                
                # Check if it was already linked during the MEI traversal
                has_parent = any(child == node_id for parent, child, edge_class in self.gt_edges)
                
                if not has_parent:
                    # Spatial Fallback: Link to the staff it physically sits on
                    nx1, ny1, nx2, ny2 = node['bbox']
                    n_cy = (ny1 + ny2) / 2.0
                    
                    for staff in staves:
                        sx1, sy1, sx2, sy2 = staff['bbox']
                        
                        # If the clef/signature's Y-center is within the staff's vertical bounds
                        if sy1 <= n_cy <= sy2:
                            # Class 1: Structural edge from Staff -> Context Marker
                            self.gt_edges.append((staff['id'], node_id, 1))
                            break
                        
        self.gt_edges = list(set(self.gt_edges))
        return self.gt_edges

    def get_pyg_labels(self, candidate_edge_index, node_id_list):
        # ... (Keep your existing get_pyg_labels logic here) ...
        # Third party imports
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
