import networkx as nx

class HumdrumSerializer:
    def __init__(self, nodes, edge_index, edge_predictions):
        """
        nodes: List of dicts e.g., [{'id': 0, 'class': 'Notehead', 'cx': 150, 'cy': 80}, ...]
        edge_index: Shape (2, E)
        edge_predictions: Shape (E) -> 1: Structural, 2: Modifier, 3: Temporal, 4: Sync
        """
        self.G = nx.DiGraph()
        
        # 1. Add all raw primitives to the graph
        for node in nodes:
            self.G.add_node(node['id'], **node)
            
        # 2. Add only the valid predicted edges (Ignore Class 0: No Edge)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            edge_type = edge_predictions[i].item()
            
            if edge_type > 0:
                self.G.add_edge(u, v, type=edge_type)

import networkx as nx

def _collapse_primitives(self, node_roles):
    """
    Merges Structural (1) and Modifier (2) edges into Super-Nodes.
    Accounts for all dynamic classes defined in node_roles.
    """
    super_nodes = {}
    
    # Unpack the roles into fast lookup sets
    temporal_classes = set(node_roles["node_roles"]["temporal_anchors"])
    structural_classes = set(node_roles["node_roles"]["structural_components"])
    modifier_classes = set(node_roles["node_roles"]["modifiers"])
    
    # 1. Isolate only the structural and modifier edges
    subgraph_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d['type'] in [1, 2]]
    
    # Create an undirected graph for easy component finding
    primitive_graph = nx.Graph()
    primitive_graph.add_edges_from(subgraph_edges)
    
    # 2. CRITICAL: Add isolated temporal anchors (e.g., whole notes, rests without modifiers)
    # If they have no edges, they won't be in 'subgraph_edges'
    for node_id, data in self.G.nodes(data=True):
        if data['class'] in temporal_classes and not primitive_graph.has_node(node_id):
            primitive_graph.add_node(node_id)

    # 3. Find connected components (e.g., [accid, notehead, stem, flag] becomes one group)
    for component in nx.connected_components(primitive_graph):
        # Fetch the full node data from the main directed graph
        group = [self.G.nodes[n] for n in component]
        
        # Categorize the nodes in this component
        anchors = [n for n in group if n['class'] in temporal_classes]
        structurals = [n for n in group if n['class'] in structural_classes]
        modifiers = [n for n in group if n['class'] in modifier_classes]
        
        # If there is no temporal anchor in this cluster, it's an orphaned visual element (skip it)
        if not anchors:
            continue
            
        # For now, we assume one primary anchor per component. 
        # (If 'chord' is detected, it acts as the single anchor for multiple noteheads)
        primary_anchor = anchors[0] 
        
        super_node_id = f"super_{primary_anchor['id']}"
        
        # 4. Store the raw categorized data. 
        # We defer pitch/duration math to a later dedicated function.
        super_nodes[super_node_id] = {
            'id': super_node_id,
            'type': primary_anchor['class'], # e.g., 'notehead', 'rest', 'chord'
            'anchor_data': primary_anchor,
            'structurals': structurals,
            'modifiers': modifiers,
            'cx': primary_anchor['cx'], # Keep X coordinate for temporal sorting
            'cy': primary_anchor['cy']  # Keep Y coordinate for pitch calculation
        }
            
    return super_nodes

def _resolve_semantics(self, super_node):
    """
    Translates a grouped super_node into a **kern token (duration + pitch + modifiers).
    """
    anchor_type = super_node['type']
    
    # Extract just the class names for easy checking
    structurals = [n['class'] for n in super_node['structurals']]
    modifiers = [n['class'] for n in super_node['modifiers']]
    
    # --- 1. Resolve Rhythmic Duration ---
    # Default to a quarter note duration
    duration = '4' 
    
    if 'stem' not in structurals and anchor_type in ['notehead', 'chord']:
        # No stem usually means a whole note (semibreve)
        duration = '1'
    elif 'stem' in structurals:
        # Count the number of flags or beams to subdivide the duration
        # Note: In a real scenario, you'd count the exact number of intersecting beam/flag nodes
        num_tails = structurals.count('flag') + structurals.count('beam')
        
        if num_tails == 1:
            duration = '8'   # Eighth note
        elif num_tails == 2:
            duration = '16'  # Sixteenth note
        elif num_tails == 3:
            duration = '32'  # Thirty-second note
        else:
            duration = '4'   # Just a stem -> Quarter note (or Half note, see note below)

    # --- 2. Resolve Base Pitch / Rest ---
    kern_token = ""
    
    if anchor_type in ['rest', 'mRest']:
        # 'r' is the Humdrum representation for a rest
        kern_token = duration + 'r'
    else:
        # We will build this spatial math function next!
        # It needs the anchor_data['cy'] and the local staff lines.
        pitch_string = self._calculate_pitch(super_node['anchor_data']['cy'])
        kern_token = duration + pitch_string
        
        # --- 3. Apply Modifiers ---
        if 'accid' in modifiers:
            # We need sub-classification here (e.g., '#', '-', 'n')
            # Assuming a generic sharp for the placeholder
            kern_token += '#' 
            
        if 'fermata' in modifiers:
            kern_token += ';' # Humdrum syntax for fermata
            
        if 'artic' in modifiers:
            # Assuming a staccato dot for the placeholder
            kern_token += "'" # Humdrum syntax for staccato
            
        # You can continue adding modifier logic here (slurs, ties, etc.)
        
    return kern_token

import torch

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

def _calculate_pitch(self, notehead_cy, staff_bbox, active_clef="treble"):
    """
    Calculates the Humdrum **kern pitch string based on Y-coordinate interpolation.
    
    notehead_cy: The center-Y of the notehead or chord anchor.
    staff_bbox: Tuple of (x1, y1, x2, y2) for the staff this note belongs to.
    active_clef: String indicating the current clef ('treble', 'bass', 'alto', etc.)
    """
    _, y_top, _, y_bottom = staff_bbox
    staff_height = y_bottom - y_top
    
    # A standard staff has 8 steps between the top line and bottom line
    step_size = staff_height / 8.0
    
    # Calculate how many steps down from the top line the notehead is
    offset_from_top = notehead_cy - y_top
    slot = round(offset_from_top / step_size)
    
    # --- Pitch Mapping Arrays ---
    # Index 0 is the top line. Negative indices go above the staff.
    # Humdrum **kern pitch representations:
    # middle C = c; octave above = cc; octave below = C; two below = CC
    
    maps = {
        "treble": {
            -4: "ccc", -3: "bb", -2: "aa", -1: "gg", 0: "ff", # Above staff
            1: "ee", 2: "dd", 3: "cc", 4: "b", 5: "a",        # Upper half
            6: "g", 7: "f", 8: "e",                           # Lower half
            9: "d", 10: "c", 11: "B", 12: "A", 13: "G"        # Below staff
        },
        "bass": {
            -4: "e", -3: "d", -2: "c", -1: "B", 0: "A",       # Above staff
            1: "G", 2: "F", 3: "E", 4: "D", 5: "C",           # Upper half
            6: "BB", 7: "AA", 8: "GG",                        # Lower half
            9: "FF", 10: "EE", 11: "DD", 12: "CC", 13: "BBB"  # Below staff
        }
    }
    
    # Fallback to treble if clef is unknown, or cap the extremes if the slot goes wild
    current_map = maps.get(active_clef.lower(), maps["treble"])
    
    # Clamp the slot to our defined dictionary bounds to prevent KeyErrors on extreme ledger lines
    min_slot, max_slot = min(current_map.keys()), max(current_map.keys())
    clamped_slot = max(min_slot, min(slot, max_slot))
    
    base_pitch = current_map[clamped_slot]
    
    return base_pitch

class ContextTracker:
    def __init__(self):
        # Global State
        self.active_clef = "treble" 
        self.key_signature = {}       # e.g., {'f': '#', 'c': '#'}
        
        # Local State (Resets every measure)
        self.measure_accidentals = {} # e.g., {'g': '#'} 
        
        # Spatial Anchors
        self.staff_y_top = 0.0
        self.staff_y_bottom = 0.0

    def process_context_node(self, node, super_node=None, calculated_pitch=None):
        """
        Updates the running state based on the node class.
        """
        node_class = node['class']
        
        # 1. Handle Structural Anchors
        if node_class.startswith('barLine'):
            self._handle_barline(node)
            
        elif node_class.startswith('clef'):
            self._handle_clef(node)
            
        # 2. Handle Global Modifiers
        elif node_class.startswith('keySig'):
            self._handle_key_signature(node_class)
            
        # 3. Handle Local Modifiers (Inline Accidentals)
        elif super_node and calculated_pitch:
            modifiers = [m['class'] for m in super_node['modifiers']]
            for mod in modifiers:
                if mod.startswith('accid'):
                    self._handle_inline_accidental(mod, calculated_pitch)

    def _handle_barline(self, barline_node):
        """Resets local measure state and recalibrates vertical spatial anchors."""
        self.measure_accidentals.clear()
        
        _, y1, _, y2 = barline_node['bbox']
        self.staff_y_top = y1
        self.staff_y_bottom = y2

    def _handle_clef(self, clef_node):
        """Updates the active clef and acts as a spatial anchor."""
        clef_identity = clef_node['class'].split('-')[-1] # e.g., 'G', 'F', 'C'
        
        clef_map = {
            'G': 'treble',
            'F': 'bass',
            'C': 'alto' # Or tenor, depending on Y-position, but 'alto' is a safe default
        }
        self.active_clef = clef_map.get(clef_identity, 'treble')
        
        _, y1, _, y2 = clef_node['bbox']
        self.staff_y_top = y1
        self.staff_y_bottom = y2

    def _handle_key_signature(self, key_class):
        """
        Translates key signatures (e.g., 'keySig-3s' or 'keySig-2f') 
        into a dictionary of active accidentals based on the Circle of Fifths.
        """
        self.key_signature.clear()
        
        if '-' not in key_class:
            return # C Major / A Minor
            
        identity = key_class.split('-')[-1]
        count = int(identity[:-1]) # Extract the number (e.g., '3' from '3s')
        accid_type = identity[-1]  # Extract the type ('s' or 'f')
        
        # Order of sharps and flats
        sharps_order = ['f', 'c', 'g', 'd', 'a', 'e', 'b']
        flats_order = ['b', 'e', 'a', 'd', 'g', 'c', 'f']
        
        if accid_type == 's':
            for i in range(count):
                self.key_signature[sharps_order[i]] = '#'
        elif accid_type == 'f':
            for i in range(count):
                self.key_signature[flats_order[i]] = '-'

    def _handle_inline_accidental(self, accid_class, pitch_string):
        """
        Registers an inline accidental into the local measure state.
        In Humdrum (**kern), pitch strings encode the octave (e.g., 'cc', 'C').
        Strictly speaking, a measure accidental only applies to that specific octave.
        """
        identity = accid_class.split('-')[-1]
        
        accid_map = {
            'sharp': '#',
            'flat': '-',
            'nat': 'n',
            'dsharp': '##',
            'dflat': '--'
        }
        
        symbol = accid_map.get(identity, '')
        
        # Strip existing non-alphabetic characters (like duration numbers or old accidentals)
        # to isolate the pure pitch string (e.g., '4cc#' -> 'cc')
        pure_pitch = ''.join(filter(str.isalpha, pitch_string))
        
        if symbol:
            self.measure_accidentals[pure_pitch] = symbol

    def get_effective_accidental(self, pure_pitch):
        """
        Returns the active modifier for a given pitch.
        Checks the local measure state first, then the key signature.
        """
        # 1. Check local measure state (exact octave match)
        if pure_pitch in self.measure_accidentals:
            # If a natural 'n' is active, it cancels the accidental, so we return ''
            mod = self.measure_accidentals[pure_pitch]
            return '' if mod == 'n' else mod
            
        # 2. Check global key signature (pitch class match, ignoring octave)
        pitch_class = pure_pitch[0].lower() # 'cc' -> 'c', 'C' -> 'c'
        return self.key_signature.get(pitch_class, "")

def generate_kern_stream(system_nodes, super_nodes, node_roles):
    """
    Orchestrates the context tracking and semantic resolution to output Humdrum **kern.
    
    system_nodes: List of all spatial nodes (dicts) in this system, strictly sorted left-to-right by 'cx'.
    super_nodes: The dictionary of grouped primitives generated by _collapse_primitives().
    node_roles: The JSON dict categorizing node types.
    """
    tracker = ContextTracker()
    kern_stream = []
    
    # Fast lookup sets
    temporal_classes = set(node_roles["node_roles"]["temporal_anchors"])
    context_classes = set(node_roles["node_roles"]["context_globals"])
    context_classes.update(node_roles["node_roles"]["structural_components"]) # Include barlines

    for node in system_nodes:
        node_class = node['class']
        
        # --- 1. Update Global/Local Context ---
        if node_class in context_classes:
            tracker.process_context_node(node)
            
            # Humdrum also requires barlines and clefs to be printed in the output stream
            if node_class.startswith('barLine'):
                kern_stream.append("=") # Humdrum barline token
            elif node_class.startswith('clef'):
                # Translate to Humdrum clef syntax (e.g., *clefG2)
                clef_id = node_class.split('-')[-1]
                kern_stream.append(f"*clef{clef_id}")
                
            continue

        # --- 2. Process Musical Events (Temporal Anchors) ---
        if node_class in temporal_classes:
            super_node_id = f"super_{node['id']}"
            
            # Skip if the object detector found a notehead but the GNN failed to group it
            if super_node_id not in super_nodes:
                continue
                
            super_node = super_nodes[super_node_id]
            
            # A. Default to Rest logic
            if node_class in ['rest', 'mRest']:
                # Pass to semantic resolver (which we built previously)
                # Rests don't need pitch math, just duration from the structurals
                token = _resolve_semantics(super_node, pitch_string="")
                kern_stream.append(token)
                continue

            # B. Note/Chord logic
            # Calculate base pitch purely from spatial Y-coordinate and staff boundaries
            staff_bbox = (0, tracker.staff_y_top, 0, tracker.staff_y_bottom)
            raw_pitch = _calculate_pitch(node['cy'], staff_bbox, tracker.active_clef)
            
            # Check the tracker for active key signatures or measure-local accidentals
            effective_accidental = tracker.get_effective_accidental(raw_pitch)
            
            # Combine them (e.g., 'cc' + '#' = 'cc#')
            final_pitch_string = raw_pitch + effective_accidental
            
            # If the super node has an inline accidental, update the tracker for the rest of the measure
            tracker.process_context_node(node, super_node=super_node, calculated_pitch=final_pitch_string)
            
            # Resolve rhythmic duration and combine with the final pitch string
            kern_token = _resolve_semantics(super_node, pitch_string=final_pitch_string)
            
            kern_stream.append(kern_token)

    # Join the stream into a single column string (Humdrum format is tab/newline separated)
    return "\n".join(kern_stream)