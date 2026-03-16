import logging

logger = logging.getLogger(__name__)

# --- Kern encoding constants ---

DIATONIC = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Bottom staff line pitch for each clef: (diatonic_index, octave)
CLEF_BOTTOM_LINE = {
    "clefG": (2, 4),  # E4
    "clefF": (4, 2),  # G2
    "clefC": (2, 3),  # E3
}

REST_KERN = {
    "mRest": "1r",
    "restWhole": "1r", "restHalf": "2r", "restQuarter": "4r",
    "rest8th": "8r", "rest16th": "16r", "rest32nd": "32r",
}

NOTEHEAD_BASE_DURATION = {
    "noteheadWhole": 1,
    "noteheadHalf": 2,
    "noteheadBlack": 4,
}

FLAG_DURATION = {
    "flag8thUp": 8, "flag8thDown": 8,
    "flag16thUp": 16, "flag16thDown": 16,
}

STEM_DURATION = {
    "stem4": 4,
    "stem8": 8,
    "stem16": 16,
    "stem32": 32,
}

ACCIDENTAL_KERN = {
    "accidentalSharp": "#",
    "accidentalFlat": "-",
    "accidentalNatural": "n",
    "accidentalDoubleSharp": "##",
    "accidentalDoubleFlat": "--",
}

# Event-level classes (top-level temporal anchors we serialize)
EVENT_CLASSES = {"note", "chord", "mRest"} | set(REST_KERN.keys())

# Sub-glyph classes (children of notes, skip when collecting events)
SUB_GLYPH_CLASSES = set(NOTEHEAD_BASE_DURATION.keys())

# Valid kern durations (powers of 2)
VALID_DURATIONS = [1, 2, 4, 8, 16, 32]


def _nearest_duration(value: float) -> int:
    """Snap a raw duration value to the nearest standard kern duration."""
    return min(VALID_DURATIONS, key=lambda d: abs(d - value))


def _value_to_duration(value: float) -> tuple:
    """Convert a whole-note value to (kern_duration, dots). E.g. 0.75 -> (2, 1)."""
    for dur in VALID_DURATIONS:
        base = 1.0 / dur
        if abs(base - value) < 0.001:
            return dur, 0
        if abs(base * 1.5 - value) < 0.001:
            return dur, 1
        if abs(base * 1.75 - value) < 0.001:
            return dur, 2
    return _nearest_duration(round(1.0 / value)), 0


def _note_value(duration: int, dots: int) -> float:
    """Return the duration of a note in whole-note units (e.g. quarter=0.25)."""
    base = 1.0 / duration
    if dots > 0:
        base *= (2.0 - (0.5 ** dots))
    return base



class Measure:
    """Represents one measure's content within a single spine."""

    def __init__(self, measure_id: str):
        self.measure_id = measure_id
        self.tokens = []
        self.durations = []  # Duration in whole-note units, parallel to tokens
        self.cxs = []  # <--- NEW: Store spatial x-coordinates

    def add(self, token: str, duration: float = 0.25, cx: float = 0.0):
        """Add a note/rest token with its duration (in whole-note units)."""
        self.tokens.append(token)
        self.durations.append(duration)
        self.cxs.append(cx)

    def build(self) -> list:
        """Return the tokens for this measure. Uses null token if empty."""
        if not self.tokens:
            return ["."]
        return list(self.tokens)


class Spine:
    """Represents a single vertical column (spine) in a Humdrum file."""

    def __init__(self, spine_type="kern"):
        self.exclusive = f"**{spine_type}"
        self.head = []
        self.measures = []
        self.clef_type = None
        self.meter_num = 4
        self.meter_den = 4

    def add_to_head(self, token: str):
        """Append a tandem interpretation to the spine header."""
        self.head.append(token)

    def add_measure(self, measure: Measure):
        """Append a measure to this spine."""
        self.measures.append(measure)

    def build(self) -> list:
        """Assemble the complete token list for this spine."""
        tokens = [self.exclusive]
        tokens.extend(self.head)
        if self.measures:
            for i, measure in enumerate(self.measures, start=1):
                tokens.append(f"={i}")
                tokens.extend(measure.build())
            tokens.append("==")
        else:
            tokens.append("=1")
        tokens.append("*-")
        return tokens

    @classmethod
    def _get_system_descendants(cls, system_id, children_map):
        """Traverses the graph to find all nodes structurally attached to this system."""
        descendants = set()
        stack = [system_id]
        
        while stack:
            curr = stack.pop()
            for child, e_class in children_map.get(curr, []):
                if child not in descendants:
                    descendants.add(child)
                    stack.append(child)
                    
        return descendants

    @classmethod
    def _extract_key_signature(cls, staff_node, system_descendants, children_map, nodes_map) -> str:
        sy1, sy2 = staff_node['bbox'][1], staff_node['bbox'][3]
        target_keysig = None
        
        # 1. Find the parent 'keySig' node that visually sits on this staff
        for node_id in system_descendants:
            node = nodes_map.get(node_id)
            if node and node['class'].lower() == 'keysig':
                if sy1 - 20 <= node['cy'] <= sy2 + 20:
                    target_keysig = node_id
                    break
                    
        if not target_keysig:
            return "*k[]"

        staff_node_id = staff_node['id']

        if not staff_node_id:
            return "*k[]"

        # 2. Ask the graph for ONLY the accidentals explicitly linked to this keySig
        valid_accids = []
        
        for child_id, e_class in children_map.get(staff_node_id, []):
            child_node = nodes_map.get(child_id)
            if child_node and "keyAccid" in child_node['class']:
                valid_accids.append(child_node)

        sharps = sum(1 for a in valid_accids if "Sharp" in a['class'])
        flats = sum(1 for a in valid_accids if "Flat" in a['class'])

        if sharps == 0 and flats == 0:
            return "*k[]"

        sharps_order = ["f", "c", "g", "d", "a", "e", "b"]
        flats_order = ["b", "e", "a", "d", "g", "c", "f"]
        
        accids = []
        if sharps > 0:
            accids = [f"{n}#" for n in sharps_order[:min(sharps, 7)]]
        elif flats > 0:
            accids = [f"{n}-" for n in flats_order[:min(flats, 7)]]
            
        return f"*k[{''.join(accids)}]"

    @classmethod
    def _extract_meter_signature(cls, staff_node, system_descendants, children_map, nodes_map) -> str:
        sy1, sy2 = staff_node['bbox'][1], staff_node['bbox'][3]
        target_metersig = None
        
        # 1. Find the parent 'meterSig' node that visually sits on this staff
        for node_id in system_descendants:
            node = nodes_map.get(node_id)
            if node and node['class'].lower() == 'metersig':
                if sy1 - 20 <= node['cy'] <= sy2 + 20:
                    target_metersig = node_id
                    break
                    
        if not target_metersig:
            return "*M4/4" 

        # 2. Ask the graph for ONLY the digits explicitly linked to this meterSig
        time_sig_nodes = []
        for child_id, e_class in children_map.get(target_metersig, []):
            child_node = nodes_map.get(child_id)
            if child_node and child_node['class'].startswith("timeSig"):
                time_sig_nodes.append(child_node)

        if not time_sig_nodes:
            return "*M4/4"

        for node in time_sig_nodes:
            if node['class'] == "timeSigCommon": return "*M4/4"
            if node['class'] == "timeSigCutCommon": return "*M2/2"

        digit_nodes = [n for n in time_sig_nodes if n['class'].replace("timeSig", "").isdigit()]
        if not digit_nodes:
            return "*M4/4"
            
        if len(digit_nodes) == 1:
            val = digit_nodes[0]['class'].replace("timeSig", "")
            return f"*M{val}/4"

        # 3. Sort the retrieved digits geometrically
        digit_nodes.sort(key=lambda n: n['cy'])

        max_jump = 0
        split_idx = 1
        for i in range(1, len(digit_nodes)):
            jump = digit_nodes[i]['cy'] - digit_nodes[i-1]['cy']
            if jump > max_jump:
                max_jump = jump
                split_idx = i

        top_nodes = sorted(digit_nodes[:split_idx], key=lambda n: n['cx'])
        bottom_nodes = sorted(digit_nodes[split_idx:], key=lambda n: n['cx'])

        numerator = "".join([n['class'].replace("timeSig", "") for n in top_nodes])
        denominator = "".join([n['class'].replace("timeSig", "") for n in bottom_nodes])

        return f"*M{numerator}/{denominator}"

    @classmethod
    def create_from_measure(cls, system_id, measure_id, children_map, nodes_map):
        system_descendants = cls._get_system_descendants(system_id, children_map)
        
        staves = [
            v for v, e_class in children_map.get(measure_id, []) 
            if e_class == 1 and nodes_map[v].get('class') == 'staff'
        ]
        staves.sort(key=lambda s_id: nodes_map[s_id]['cy'])

        spines = []
        for index, st_id in enumerate(staves, start=1):
            spine = cls(spine_type="kern")
            
            spine.add_to_head(f"*part{index}")
            spine.add_to_head(f"*staff{index}")
            
            staff_node = nodes_map[st_id]
            
            # Use the graph-driven extractors!
            key_sig_found = cls._extract_key_signature(staff_node, system_descendants, children_map, nodes_map)
            meter_sig_found = cls._extract_meter_signature(staff_node, system_descendants, children_map, nodes_map)
            
            clef_found = "*"
            staff_elements = [v for v, e in children_map.get(st_id, []) if e == 1]
            for el_id in staff_elements:
                if nodes_map[el_id]['class'].startswith("clef"):
                    clef_found = nodes_map[el_id]['class'].replace("clef", "*clef")
                    break
                    
            if clef_found == "*":
                for node_id in system_descendants:
                    node = nodes_map.get(node_id)
                    if node and node['class'].startswith("clef"):
                        if (staff_node['bbox'][1] - 20 <= node['cy'] <= staff_node['bbox'][3] + 20):
                            clef_found = node['class'].replace("clef", "*clef")
                            break
            
            spine.add_to_head(clef_found)
            spine.add_to_head(key_sig_found)
            spine.add_to_head(meter_sig_found)
            spine.clef_type = clef_found.replace("*", "")
            spine.meter_num, spine.meter_den = cls._parse_meter(meter_sig_found)

            spines.append(spine)
            
        return spines

    @staticmethod
    def _parse_meter(meter_str: str) -> tuple:
        """Parse '*M4/4' into (4, 4). Returns (4, 4) as fallback."""
        try:
            m = meter_str.replace("*M", "")
            num, den = m.split("/")
            return int(num), int(den)
        except (ValueError, AttributeError):
            return 4, 4


class HumdrumContext:
    """Manages the collection of spines and handles the final text rendering."""
    
    def __init__(self):
        self.spines = []

    def add_spine(self, spine: Spine):
        self.spines.append(spine)

    def _synchronize_measures(self):
        """Align measures across spines using spatial horizontal coordinates (cx)."""
        if not self.spines:
            return

        # 1. Pad to the same number of measures across all spines
        max_measures = max(len(s.measures) for s in self.spines)
        for spine in self.spines:
            while len(spine.measures) < max_measures:
                spine.measures.append(Measure("_pad"))

        # 2. Cluster events into time slices based on spatial proximity
        TOLERANCE = 30.0  # Pixels. Events within this horizontal distance sync up.

        for m_idx in range(max_measures):
            all_cxs = []
            for spine in self.spines:
                all_cxs.extend(spine.measures[m_idx].cxs)

            if not all_cxs:
                # All spines have empty measures, ensure at least one null token
                for spine in self.spines:
                    if not spine.measures[m_idx].tokens:
                        spine.measures[m_idx].tokens = ["."]
                        spine.measures[m_idx].cxs = [0.0]
                continue

            # Sort and cluster the horizontal coordinates
            all_cxs.sort()
            clusters = []
            current_cluster = [all_cxs[0]]

            for cx in all_cxs[1:]:
                if cx - current_cluster[0] <= TOLERANCE:
                    current_cluster.append(cx)
                else:
                    # Average the positions in the cluster to find its center
                    clusters.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [cx]
            
            if current_cluster:
                clusters.append(sum(current_cluster) / len(current_cluster))

            # Merged timeline is now the sorted centers of our geometric clusters
            merged_timeline = sorted(clusters)

            # Rebuild each spine's tokens aligned to the geometric timeline
            for spine in self.spines:
                m = spine.measures[m_idx]
                time_to_token = {}

                # Map this spine's tokens to the nearest cluster
                for token, cx in zip(m.tokens, m.cxs):
                    nearest_cluster = min(merged_timeline, key=lambda c: abs(c - cx))
                    # Prevent overwriting if multiple tokens snap to the same cluster (e.g., a chord)
                    if nearest_cluster not in time_to_token:
                        time_to_token[nearest_cluster] = token
                    else:
                        time_to_token[nearest_cluster] += f" {token}"

                new_tokens = []
                for t in merged_timeline:
                    new_tokens.append(time_to_token.get(t, "."))

                m.tokens = new_tokens

    def merge_spines(self) -> str:
        """Builds all spines and transposes them into tab-separated horizontal rows."""
        if not self.spines:
            return ""

        self._synchronize_measures()

        built_columns = [spine.build() for spine in self.spines]
        lines = []
        total_rows = len(built_columns[0]) 

        for row_idx in range(total_rows):
            row_tokens = [col[row_idx] for col in built_columns]
            
            # Reverse the tokens to match Humdrum's bottom-to-top convention
            row_tokens.reverse() 
            
            lines.append("\t".join(row_tokens))

        return "\n".join(lines)
        
class MinimalHumdrumSerializer:
    """Serializes a PyG music graph into Humdrum **kern format, one page at a time."""

    def __init__(self, edge_index, edge_predictions, node_roles, pyg_node_ids):
        self.edge_index = edge_index
        self.edge_predictions = edge_predictions
        self.node_roles = node_roles
        self.pyg_node_ids = pyg_node_ids

        self.context = HumdrumContext()
        self._head_initialized = False

    def _build_children(self, nodes: dict) -> dict:
        """Build the children adjacency map scoped to a set of nodes."""
        children = {}
        node_ids = set(nodes.keys())

        for i in range(self.edge_index.shape[1]):
            u_str = self.pyg_node_ids[self.edge_index[0, i].item()]
            v_str = self.pyg_node_ids[self.edge_index[1, i].item()]
            e_class = self.edge_predictions[i].item()

            if e_class > 0 and u_str in node_ids and v_str in node_ids:
                if u_str not in children:
                    children[u_str] = []
                children[u_str].append((v_str, e_class))

        return children

    def add_page(self, page_nodes: list):
        """Process a single page's annotations and accumulate into the context."""
        nodes = {n['id']: n for n in page_nodes}
        children = self._build_children(nodes)

        # Find all systems on this page, sorted top-to-bottom
        systems = [n for n in nodes.values() if n['class'] == 'system']
        if not systems:
            return

        systems.sort(key=lambda s: s['cy'])

        for system in systems:
            # Get measures in this system, sorted left-to-right
            measures = [
                v for v, e_class in children.get(system['id'], [])
                if e_class == 1 and nodes.get(v, {}).get('class') == 'measure'
            ]
            if not measures:
                continue

            measures.sort(key=lambda m_id: nodes[m_id]['cx'])

            # Initialize spine headers from the first valid system/measure
            if not self._head_initialized:
                spines = Spine.create_from_measure(
                    system['id'], measures[0], children, nodes
                )
                if spines:
                    for spine in spines:
                        self.context.add_spine(spine)
                    self._head_initialized = True

            # Add each measure in this system to the corresponding spines
            if self._head_initialized:
                for measure_id in measures:
                    self._add_measure_to_spines(measure_id, children, nodes)

    def _add_measure_to_spines(self, measure_id: str, children: dict, nodes: dict):
        """Create a Measure for each staff, populate with note tokens, and append to the spine."""
        staves = [
            v for v, e_class in children.get(measure_id, [])
            if e_class == 1 and nodes.get(v, {}).get('class') == 'staff'
        ]
        staves.sort(key=lambda s_id: nodes[s_id].get('cy', 0))

        # 1. Grab all system-staff nodes to use their tight 5-line bounding boxes
        all_system_staves = [n for n in nodes.values() if n.get('class') == 'system-staff']

        for spine_idx, staff_id in enumerate(staves):
            if spine_idx >= len(self.context.spines):
                continue

            spine = self.context.spines[spine_idx]
            staff_node = nodes[staff_id]
            measure = Measure(measure_id)

            # 2. Find the closest system-staff geometrically
            staff_cy = staff_node.get('cy', 0.0)
            
            if all_system_staves:
                # Find the system-staff whose vertical center is closest to this staff's center
                best_ss = min(
                    all_system_staves, 
                    key=lambda ss: abs(((ss['bbox'][1] + ss['bbox'][3]) / 2.0) - staff_cy)
                )
                active_bbox = best_ss['bbox']
            else:
                active_bbox = staff_node.get('bbox', [0, 0, 0, 0])

            # 3. Collect events and analyze each one (passing the pristine active_bbox!)
            events = self._collect_staff_events(staff_id, children, nodes)
            event_infos = [
                self._analyze_event(ev, children, nodes, spine.clef_type, active_bbox)
                for ev in events
            ]

            # 4. Resolve ambiguous durations and mRest using the time signature
            self._resolve_durations(event_infos, spine.meter_num, spine.meter_den)
            for info in event_infos:
                if info['type'] == 'mrest':
                    measure_value = spine.meter_num / spine.meter_den
                    info['duration'], info['dots'] = _value_to_duration(measure_value)

            # 5. Serialize to kern tokens with durations and spatial cx
            for info in event_infos:
                dur = _note_value(info['duration'], info['dots'])
                cx = info.get('cx', 0.0)
                measure.add(self._info_to_kern(info), dur, cx)

            spine.add_measure(measure)

    # --- Event collection ---

    def _collect_staff_events(self, staff_id: str, children: dict, nodes: dict) -> list:
        """Find temporal-anchor events under a staff, ordered by Class 3 (Temporal) edges."""
        # 1. Collect all structural descendants via Class 1 edges
        descendants = set()
        stack = [staff_id]
        while stack:
            curr = stack.pop()
            for child_id, e_class in children.get(curr, []):
                if e_class == 1 and child_id not in descendants:
                    descendants.add(child_id)
                    stack.append(child_id)

        # 2. Filter down to valid top-level event temporal anchors
        raw_events = set()
        for node_id in descendants:
            node = nodes.get(node_id)
            if node and node['class'] in EVENT_CLASSES and node['class'] not in SUB_GLYPH_CLASSES:
                raw_events.add(node_id)

        # Exclude sub-components (like a 'note' inside a 'chord' or a 'rest' inside 'mRest').
        # If an event has a structural parent that is ALSO an event, drop the child.
        top_level_events = set(raw_events)
        for ev_id in raw_events:
            for child_id, e_class in children.get(ev_id, []):
                if e_class == 1 and child_id in top_level_events:
                    top_level_events.remove(child_id)

        # 3. Build a directed graph using ONLY Class 3 (Temporal) edges among these events
        event_ids = list(top_level_events)
        in_degree = {e: 0 for e in event_ids}
        out_edges = {e: [] for e in event_ids}

        for e_id in event_ids:
            for child_id, e_class in children.get(e_id, []):
                # Only follow temporal edges pointing to other top-level events we collected
                if e_class == 3 and child_id in in_degree:
                    out_edges[e_id].append(child_id)
                    in_degree[child_id] += 1

        # 4. Topological sort to establish the proper sequence
        # We tie-break with 'cx' (horizontal position) to gracefully handle parallel layers
        queue = [e for e in event_ids if in_degree[e] == 0]
        queue.sort(key=lambda x: nodes[x].get('cx', 0))

        sorted_events = []
        while queue:
            curr_id = queue.pop(0)
            sorted_events.append(nodes[curr_id])

            for neighbor in out_edges[curr_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            
            # Re-sort queue to maintain left-to-right flow for concurrent temporal chains
            queue.sort(key=lambda x: nodes[x].get('cx', 0))

        # 5. Fallback: If the graph has cyclical temporal errors or disconnected sub-graphs,
        # append the missing nodes sorted geometrically so we don't drop data.
        if len(sorted_events) < len(event_ids):
            missing = set(event_ids) - {e['id'] for e in sorted_events}
            missing_nodes = sorted([nodes[m] for m in missing], key=lambda x: x.get('cx', 0))
            sorted_events.extend(missing_nodes)

        return sorted_events

    # --- Event analysis (returns structured dicts) ---

    def _count_dots(self, node_id: str, children: dict, nodes: dict) -> int:
        """Count augmentation dots attached to a node using bbox aspect ratio."""
        dot_count = 0
        for child_id, _ in children.get(node_id, []):
            child = nodes.get(child_id)
            if child and child['class'] == 'dots':
                bbox = child.get('bbox', [0, 0, 0, 0])
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                # A single dot is roughly square; two dots side by side are much wider
                if height > 0 and width / height > 1.8:
                    dot_count += 2
                else:
                    dot_count += 1
        return dot_count

    def _analyze_event(self, event: dict, children: dict, nodes: dict,
                       clef_type: str, staff_bbox: list) -> dict:
        """Analyze a graph event and return a structured info dict."""
        cls = event['class']
        base_cx = event.get('cx', 0.0)

        if cls in REST_KERN:
            dur = int(REST_KERN[cls].replace('r', ''))
            dots = self._count_dots(event.get('id', ''), children, nodes)
            return {"type": "rest", "duration": dur, "dots": dots, "ambiguous": False, "cx": base_cx}

        if cls == "mRest":
            return {"type": "mrest", "duration": 1, "dots": 0, "ambiguous": False, "cx": base_cx}

        if cls == "note":
            return self._analyze_note(event, children, nodes, clef_type, staff_bbox)

        if cls == "chord":
            return self._analyze_chord(event, children, nodes, clef_type, staff_bbox)

        return {"type": "unknown", "duration": 4, "dots": 0, "ambiguous": False}

    def _analyze_note(self, note_node: dict, children: dict, nodes: dict,
                      clef_type: str, staff_bbox: list) -> dict:
        """Analyze a note node by traversing all its structural descendants."""
        # 1. Traverse and collect all structural descendants (noteheads, stems, accidentals)
        descendants = set()
        stack = [note_node['id']]
        while stack:
            curr = stack.pop()
            for child_id, e_class in children.get(curr, []):
                # Follow Class 1 (Structural) edges downward
                if e_class == 1 and child_id not in descendants:
                    descendants.add(child_id)
                    stack.append(child_id)

        duration = 4
        accidental = ""
        notehead_cy = note_node.get('cy', 0.0)
        notehead_cx = note_node.get('cx', 0.0)
        has_definitive_duration = False

        # 2. Analyze the collected components to determine the note's properties
        for desc_id in descendants:
            child = nodes.get(desc_id)
            if not child:
                continue
            
            cls = child.get('class', '')

            # Noteheads set the base duration and the Y-coordinate for pitch
            if cls in NOTEHEAD_BASE_DURATION:
                if not has_definitive_duration:
                    duration = NOTEHEAD_BASE_DURATION[cls]
                notehead_cy = child.get('cy', notehead_cy)
                notehead_cx = child.get('cx', notehead_cx)

            # Stems override noteheads with a more definitive duration
            if cls in STEM_DURATION:
                duration = STEM_DURATION[cls]
                has_definitive_duration = True

            # Flags override stems/noteheads
            if cls in FLAG_DURATION:
                duration = FLAG_DURATION[cls]
                has_definitive_duration = True

            # Grab accidentals if present
            if cls in ACCIDENTAL_KERN:
                accidental = ACCIDENTAL_KERN[cls]

        # 3. Check for dots (Note: You may need to adapt this depending on how dots connect)
        dot_count = self._count_dots(note_node['id'], children, nodes)

        is_ambiguous = (duration == 4 and not has_definitive_duration)
        pitch = self._position_to_kern_pitch(notehead_cy, clef_type, staff_bbox)

        return {
            "type": "note",
            "duration": duration,
            "pitch": pitch,
            "accidental": accidental,
            "dots": dot_count,
            "ambiguous": is_ambiguous,
            "cx": notehead_cx
        }

    def _analyze_chord(self, chord_node: dict, children: dict, nodes: dict,
                       clef_type: str, staff_bbox: list) -> dict:
        """Analyze a chord node and return structured info."""
        chord_children = children.get(chord_node['id'], [])

        notes = []
        shared_ambiguous = False
        shared_duration = 4
        shared_dots = 0

        for child_id, _ in chord_children:
            child = nodes.get(child_id)
            if child and child['class'] == 'note':
                note_info = self._analyze_note(child, children, nodes, clef_type, staff_bbox)
                notes.append(note_info)
                # All chord notes share the same duration — use the first one's info
                if not notes[1:]:
                    shared_duration = note_info['duration']
                    shared_dots = note_info['dots']
                    shared_ambiguous = note_info['ambiguous']
        chord_cx = notes[0]['cx'] if notes else chord_node.get('cx', 0.0)
        return {
            "type": "chord",
            "duration": shared_duration,
            "dots": shared_dots,
            "ambiguous": shared_ambiguous,
            "notes": notes,
            "cx": chord_cx
        }

    # --- Duration resolution ---

    @staticmethod
    def _resolve_durations(event_infos: list, meter_num: int, meter_den: int):
        """Resolve ambiguous durations using the time signature."""
        expected_total = meter_num / meter_den  # Measure length in whole-note units

        known_total = 0.0
        ambiguous_indices = []

        for i, info in enumerate(event_infos):
            if info['ambiguous']:
                ambiguous_indices.append(i)
            else:
                known_total += _note_value(info['duration'], info['dots'])

        if not ambiguous_indices:
            return

        remaining = expected_total - known_total
        if remaining <= 0:
            return  # Measure already full, leave as quarter notes

        each_value = remaining / len(ambiguous_indices)
        if each_value <= 0:
            return

        resolved_dur = _nearest_duration(round(1.0 / each_value))

        for idx in ambiguous_indices:
            event_infos[idx]['duration'] = resolved_dur
            event_infos[idx]['ambiguous'] = False
            # Propagate to chord sub-notes if applicable
            if event_infos[idx]['type'] == 'chord':
                for note in event_infos[idx].get('notes', []):
                    note['duration'] = resolved_dur
                    note['ambiguous'] = False

    # --- Kern string generation ---

    @staticmethod
    def _info_to_kern(info: dict) -> str:
        """Convert an event info dict to a kern token string."""
        if info['type'] == 'rest':
            token = f"{info['duration']}r"
            token += "." * info['dots']
            return token

        if info['type'] == 'mrest':
            token = f"{info['duration']}rr"
            token += "." * info['dots']
            return token

        if info['type'] == 'note':
            token = f"{info['duration']}{info['pitch']}{info.get('accidental', '')}"
            token += "." * info['dots']
            return token

        if info['type'] == 'chord':
            note_tokens = []
            for note in info.get('notes', []):
                t = f"{note['duration']}{note['pitch']}{note.get('accidental', '')}"
                t += "." * note['dots']
                note_tokens.append(t)
            note_tokens.sort()
            return " ".join(note_tokens) if note_tokens else "."

        return "."  # Unknown

    # --- Pitch estimation ---

    @staticmethod
    def _position_to_kern_pitch(cy: float, clef_type: str, staff_bbox: list) -> str:
        """Estimate kern pitch from vertical position on the 5-line staff."""
        y_top, y_bottom = staff_bbox[1], staff_bbox[3]
        staff_height = y_bottom - y_top
        if staff_height <= 0:
            return "c"

        # 
        # A standard 5-line staff has 4 spaces. 
        # The distance from the bottom line to the top line spans exactly 8 diatonic steps.
        half_step = staff_height / 8.0 
        step = round((y_bottom - cy) / half_step)

        # Robust clef matching (safely handles 'clefG2', 'clefF4', etc.)
        if "clefF" in clef_type:
            bottom_idx, bottom_oct = 4, 2  # Bass clef bottom line: G2
        elif "clefC" in clef_type:
            bottom_idx, bottom_oct = 3, 3  # Alto clef bottom line: F3
        else:
            bottom_idx, bottom_oct = 2, 4  # Default to Treble clef bottom line: E4

        total = bottom_idx + step

        # Calculate diatonic note index and octave
        note_idx = total % 7
        octave = bottom_oct + (total // 7)
        note_name = DIATONIC[note_idx]

        # Humdrum **kern octave formatting rules
        if octave >= 4:
            return note_name.lower() * (octave - 3)
        elif octave >= 1:
            return note_name.upper() * (4 - octave)
        else:
            return note_name.upper() * 3

    def export_to_krn(self) -> str:
        """Export the accumulated data as a single Humdrum **kern string."""
        if not self._head_initialized:
            return "Error: No valid systems found across any page."
        return self.context.merge_spines()