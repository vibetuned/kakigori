# Standard library imports
import json
import xml.etree.ElementTree as ET


class GroundTruthGraphBuilder:
    def __init__(self, mei_file, json_files):
        self.mei_tree = ET.parse(mei_file)
        self.mei_root = self.mei_tree.getroot()
        # MEI usually has a namespace
        self.ns = {'mei': 'http://www.music-encoding.org/ns/mei'} 
        
        # Load spatial nodes from multiple files
        self.spatial_nodes = []
        if isinstance(json_files, str):
            json_files = [json_files]
            
        for j_file in json_files:
            with open(j_file, 'r') as f:
                data = json.load(f)
                self.spatial_nodes.extend(data.get('annotations', []))
            
        # Map IDs to their spatial node dicts for quick lookup
        self.node_map = {node['id']: node for node in self.spatial_nodes if 'id' in node}
        self.gt_edges = []

    def _get_id(self, element):
        """Extracts the xml:id from an MEI element."""
        return element.attrib.get('{http://www.w3.org/XML/1998/namespace}id')

    def build_edges(self):
        """Traverses the MEI to build ground truth edges."""
        # 1. Traverse layers to build Temporal (Class 3) edges
        for layer in self.mei_root.findall('.//mei:layer', self.ns):
            events = layer.findall('./*') # Get all children (notes, rests, chords)
            
            for i in range(len(events) - 1):
                current_event = events[i]
                next_event = events[i+1]
                
                curr_id = self._get_id(current_event)
                next_id = self._get_id(next_event)
                
                if curr_id in self.node_map and next_id in self.node_map:
                    # Class 3: Temporal Edge (Left-to-Right sequence)
                    self.gt_edges.append((curr_id, next_id, 3))

        # 2. Traverse notes to build Structural (Class 1) and Modifier (Class 2) edges
        for note in self.mei_root.findall('.//mei:note', self.ns):
            note_id = self._get_id(note)
            if not note_id or note_id not in self.node_map:
                continue

            # Look for children of the note (e.g., accidentals, articulations)
            for child in note:
                child_id = self._get_id(child)
                if not child_id or child_id not in self.node_map:
                    continue
                
                tag = child.tag.split('}')[-1] # Remove namespace
                
                # Class 1: Structural (Core components of the primitive)
                if tag in ['stem', 'flag', 'dot']:
                    self.gt_edges.append((note_id, child_id, 1))
                    
                # Class 2: Modifier (Accidentals, Articulations)
                elif tag in ['accid', 'artic', 'fermata']: 
                    self.gt_edges.append((note_id, child_id, 2))
        
        # 3. Traverse syllables for Synchronization (Class 4) edges
        # In MEI, <syl> is usually nested inside <verse>, which is inside <note> or <chord>
        # Standard ElementTree doesn't support 'ancestor::', so we build a parent map beforehand
        parent_map = {c: p for p in self.mei_root.iter() for c in p}
        
        for syl in self.mei_root.findall('.//mei:syl', self.ns):
            syl_id = self._get_id(syl)
            if not syl_id or syl_id not in self.node_map:
                continue
                
            # Traverse parents upwards until we find a full note element
            current = syl
            parent_note = None
            while current in parent_map:
                parent = parent_map[current]
                # Check tag (removing possible namespace)
                if parent.tag.endswith('note'):
                    parent_note = parent
                    break
                current = parent
            
            if parent_note is not None:
                parent_id = self._get_id(parent_note)
                if parent_id and parent_id in self.node_map:
                    self.gt_edges.append((parent_id, syl_id, 4))

        return self.gt_edges

    def get_pyg_labels(self, candidate_edge_index, node_id_list):
        """
        Maps the generated GT edges to the PyG candidate edge tensor.
        candidate_edge_index: Shape (2, E)
        node_id_list: List of string IDs corresponding to the PyG node indices.
        """
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