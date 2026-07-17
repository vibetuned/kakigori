import logging

logger = logging.getLogger(__name__)

# --- Kern encoding constants ---

DIATONIC = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Bottom staff line pitch for each clef: (diatonic_index, octave)
CLEF_BOTTOM_LINE = {
    "clefG": (2, 4),  # E4
    "clefF": (4, 2),  # G2
    "clefC": (2, 3),  # E3
    "clefG8vb": (2, 3),  # E3 — vocal tenor: treble sounding an octave down
    "clefG8va": (2, 5),  # E5 — treble sounding an octave up
}

# Mid-piece clef-change glyphs and the clef they switch to
CLEF_CHANGE_MAP = {
    "gClefChange": "clefG",
    "fClefChange": "clefF",
    "cClefChange": "clefC",
}

# Kern clef tokens with their standard staff line: renderers need the line
# number ('*clefG2', not '*clefG') or they may overwrite the key signature
CLEF_KERN = {
    "clefG": "*clefG2",
    "clefF": "*clefF4",
    "clefC": "*clefC3",
    "clefG8vb": "*clefGv2",  # kern 'v' = sounding an octave below
    "clefG8va": "*clefG^2",  # kern '^' = sounding an octave above
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

# Dynamic glyph classes and their **dynam spine text
DYNAMIC_KERN = {
    "dynamicPiano": "p",
    "dynamicMezzo": "m",
    "dynamicForte": "f",
    "dynamicSforzando": "s",
    "dynamicZ": "z",
    "dynamicPP": "pp",
    "dynamicMP": "mp",
    "dynamicMF": "mf",
    "dynamicFF": "ff",
    "dynamicFFF": "fff",
    "dynamicSforzato": "sfz",
    "dynamicForzando": "fz",
}

# Ornaments and articulations appended after the pitch in a kern token
ORNAMENT_KERN = {
    "trill": "T",
    "mordent": "M",
    "turn": "S",
    "fermata": ";",
    "arpeg": ":",
    "articStaccatoAbove": "'",
    "articTenutoAbove": "~",
    "articAccentAbove": "^",
    "articStaccatissimoAbove": "`",
    "articMarcatoAbove": "^^",
}

# Event-level classes (top-level temporal anchors we serialize)
EVENT_CLASSES = {"note", "chord", "mRest"} | set(REST_KERN.keys())

# Sub-glyph classes (children of notes, skip when collecting events)
SUB_GLYPH_CLASSES = set(NOTEHEAD_BASE_DURATION.keys())

# Valid kern durations (powers of 2)
VALID_DURATIONS = [1, 2, 4, 8, 16, 32]


def _find_staff_row(staff_node: dict, system_staves: list):
    """Match a measure-staff to the system-staff (tight 5-line box) it sits on.

    Uses vertical overlap rather than center distance: a staff's content bbox
    (ledger lines, dynamics...) can shift its center past a neighboring row,
    but it always fully covers its own 5 staff lines.
    """
    y1, y2 = staff_node['bbox'][1], staff_node['bbox'][3]
    best, best_overlap = None, 0.0
    for ss in system_staves:
        s1, s2 = ss['bbox'][1], ss['bbox'][3]
        overlap = min(y2, s2) - max(y1, s1)
        if overlap > best_overlap:
            best_overlap, best = overlap, ss
    return best


def _staff_sort_key(staff_node: dict, system_staves: list) -> float:
    """Stable top-to-bottom ordering key for staves within a measure."""
    row = _find_staff_row(staff_node, system_staves)
    if row is not None:
        return (row['bbox'][1] + row['bbox'][3]) / 2.0
    return (staff_node['bbox'][1] + staff_node['bbox'][3]) / 2.0


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


def _assign_staff(cy: float, staff_rows: list, staff_space: float) -> int:
    """Pick the staff a below-the-staff marking (dynamic, pedal) belongs to.

    Such markings are engraved BELOW their staff, so a glyph between two
    staves belongs to the one above it, even when the staff below is
    geometrically closer.
    """
    inside = [i for i, r in enumerate(staff_rows) if r[1] <= cy <= r[3]]
    if inside:
        return inside[0]
    above = [(cy - r[3], i) for i, r in enumerate(staff_rows) if r[3] <= cy]
    if above:
        dist, idx = min(above)
        if dist < 8 * staff_space:
            return idx
    return min(
        range(len(staff_rows)),
        key=lambda i: max(staff_rows[i][1] - cy, cy - staff_rows[i][3], 0.0)
    )


def _parse_key_accidentals(key_str: str) -> dict:
    """Parse a kern key signature like '*k[f#c#]' into {'F': '#', 'C': '#'}."""
    accids = {}
    start, end = key_str.find('['), key_str.rfind(']')
    if start == -1 or end <= start:
        return accids
    inner = key_str[start + 1:end]
    i = 0
    while i < len(inner):
        letter = inner[i].upper()
        i += 1
        acc = ""
        while i < len(inner) and inner[i] in "#-n":
            acc += inner[i]
            i += 1
        if letter in "ABCDEFG" and acc and acc != "n":
            accids[letter] = acc
    return accids



class Measure:
    """Represents one measure's content within a single spine."""

    def __init__(self, measure_id: str):
        self.measure_id = measure_id
        self.tokens = []
        self.durations = []  # Duration in whole-note units, parallel to tokens
        self.cxs = []  # Spatial x-coordinates, parallel to tokens
        self.ids = []  # Graph node IDs, parallel to tokens
        self.full_measure_rest = False  # Lone whole/measure rest fills this measure
        self.dynamics = []  # (cx, text) dynamic markings assigned to this staff
        self.dynam_tokens = None  # **dynam rows, aligned with tokens after sync
        self.pedals = []  # (cx, '*ped'/'*Xped') pedal marks assigned to this staff
        self.interps = []  # (cx, token) mid-piece clef/key/meter change rows

    def add(self, token: str, duration: float = 0.25, cx: float = 0.0, node_id=None):
        """Add a note/rest token with its duration (in whole-note units)."""
        self.tokens.append(token)
        self.durations.append(duration)
        self.cxs.append(cx)
        self.ids.append(node_id)

    def build(self) -> list:
        """Return the tokens for this measure. Uses null token if empty."""
        if not self.tokens:
            return ["."]
        return list(self.tokens)

    def build_dynam(self) -> list:
        """Return the **dynam rows for this measure, aligned with build()."""
        if self.dynam_tokens is not None:
            return list(self.dynam_tokens)
        return ["."] * len(self.build())


class Spine:
    """Represents a single vertical column (spine) in a Humdrum file."""

    def __init__(self, spine_type="kern"):
        self.exclusive = f"**{spine_type}"
        self.head = []
        self.measures = []
        self.clef_type = None
        self.meter_num = 4
        self.meter_den = 4
        self.key_accidentals = {}  # Letter -> kern accidental, e.g. {'F': '#'}
        self.pending_ties = {}  # Pitch -> accidental of a tie left open last measure
        self.key_token = "*k[]"  # Current key signature token, for change detection
        self.meter_token = "*M4/4"  # Current meter token, for change detection

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

    def has_dynamics(self) -> bool:
        return any(m.dynamics for m in self.measures)

    def build_dynam(self) -> list:
        """Assemble the companion **dynam column, row-aligned with build()."""
        tokens = ["**dynam"]
        for head_token in self.head:
            tokens.append(head_token if head_token.startswith("*part") else "*")
        if self.measures:
            for i, measure in enumerate(self.measures, start=1):
                tokens.append(f"={i}")
                tokens.extend(measure.build_dynam())
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

        return cls._keysig_token(target_keysig, staff_node['id'], children_map, nodes_map)

    @classmethod
    def _keysig_token(cls, keysig_id, staff_id, children_map, nodes_map) -> str:
        """Build the '*k[...]' token from a keySig node's accidentals."""
        # Collect the accidentals linked to this keySig or directly to the
        # staff (the spatial fallback in the graph builder attaches them
        # there). A staff can carry accids of SEVERAL signatures (courtesy
        # restatement before a system break), so fallback accids must sit
        # inside the keySig glyph's own horizontal span; and an accidental
        # linked both ways is only counted once.
        ks_bbox = nodes_map.get(keysig_id, {}).get('bbox')
        valid_accids = []
        seen = set()
        for child_id, e_class in children_map.get(keysig_id, []):
            child_node = nodes_map.get(child_id)
            if child_node and "keyAccid" in child_node['class']:
                seen.add(child_id)
                valid_accids.append(child_node)
        for child_id, e_class in children_map.get(staff_id, []):
            if child_id in seen:
                continue
            child_node = nodes_map.get(child_id)
            if not (child_node and "keyAccid" in child_node['class']):
                continue
            if ks_bbox:
                pad = ks_bbox[3] - ks_bbox[1]
                accid_cx = (child_node['bbox'][0] + child_node['bbox'][2]) / 2.0
                if not (ks_bbox[0] - pad <= accid_cx <= ks_bbox[2] + pad):
                    continue
            seen.add(child_id)
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

        return cls._metersig_token(target_metersig, children_map, nodes_map)

    @classmethod
    def _metersig_token(cls, metersig_id, children_map, nodes_map) -> str:
        """Build the '*M<num>/<den>' token from a meterSig node's digits."""
        # Ask the graph for ONLY the digits explicitly linked to this meterSig
        time_sig_nodes = []
        for child_id, e_class in children_map.get(metersig_id, []):
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
        system_staves = [n for n in nodes_map.values() if n.get('class') == 'system-staff']
        staves.sort(key=lambda s_id: _staff_sort_key(nodes_map[s_id], system_staves))

        spines = []
        for index, st_id in enumerate(staves, start=1):
            spine = cls(spine_type="kern")
            
            spine.add_to_head(f"*part{index}")
            spine.add_to_head(f"*staff{index}")
            
            staff_node = nodes_map[st_id]
            
            # Use the graph-driven extractors!
            key_sig_found = cls._extract_key_signature(staff_node, system_descendants, children_map, nodes_map)
            meter_sig_found = cls._extract_meter_signature(staff_node, system_descendants, children_map, nodes_map)
            
            clef_class = None
            staff_elements = [v for v, e in children_map.get(st_id, []) if e == 1]
            for el_id in staff_elements:
                if nodes_map[el_id]['class'].startswith("clef"):
                    clef_class = nodes_map[el_id]['class']
                    break

            if clef_class is None:
                for node_id in system_descendants:
                    node = nodes_map.get(node_id)
                    if node and node['class'].startswith("clef"):
                        if (staff_node['bbox'][1] - 20 <= node['cy'] <= staff_node['bbox'][3] + 20):
                            clef_class = node['class']
                            break

            clef_found = CLEF_KERN.get(clef_class, f"*{clef_class}" if clef_class else "*")
            spine.add_to_head(clef_found)
            spine.add_to_head(key_sig_found)
            spine.add_to_head(meter_sig_found)
            spine.clef_type = clef_class or ""
            spine.key_accidentals = _parse_key_accidentals(key_sig_found)
            spine.key_token = key_sig_found
            spine.meter_token = meter_sig_found
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

    def __init__(self, sync_groups=None):
        self.spines = []
        # Maps node ID -> sync group key. Events sharing a group are simultaneous
        # (Class 5 edges in the graph) and must land on the same Humdrum row.
        self.sync_groups = sync_groups or {}

    def add_spine(self, spine: Spine):
        self.spines.append(spine)

    def _slices_from_sync(self, measures: list):
        """Build time slices from Class 5 (sync) edge groups via topological sort.

        Returns (slices, slice_cxs) where each slice is a list (one entry per
        spine) of token lists and slice_cxs holds a representative horizontal
        position per slice — or None if the constraints are cyclic.
        """
        if not any(self.sync_groups.get(node_id) is not None
                   for m in measures for node_id in m.ids):
            return None  # No sync information in this measure

        # 1. Assign every token a slice key: its sync group, or a unique key
        slice_spines = {}  # key -> {spine_idx: [tokens]}
        slice_cx = {}      # key -> min cx (topo tie-breaker)
        spine_keys = []    # per spine: ordered list of slice keys

        for s_idx, m in enumerate(measures):
            keys = []
            for i, token in enumerate(m.tokens):
                node_id = m.ids[i] if i < len(m.ids) else None
                group = self.sync_groups.get(node_id)
                key = group if group is not None else ("solo", s_idx, i)
                slice_spines.setdefault(key, {}).setdefault(s_idx, []).append(token)
                cx = m.cxs[i] if i < len(m.cxs) else 0.0
                slice_cx[key] = min(slice_cx.get(key, cx), cx)
                keys.append(key)
            spine_keys.append(keys)

        # 2. Precedence constraints: within a spine, tokens appear in time order
        successors = {key: set() for key in slice_spines}
        in_degree = {key: 0 for key in slice_spines}
        for keys in spine_keys:
            for a, b in zip(keys, keys[1:]):
                if a != b and b not in successors[a]:
                    successors[a].add(b)
                    in_degree[b] += 1

        # 3. Kahn topological sort, tie-breaking by horizontal position
        ready = sorted((k for k in slice_spines if in_degree[k] == 0),
                       key=lambda k: slice_cx[k])
        ordered = []
        while ready:
            key = ready.pop(0)
            ordered.append(key)
            changed = False
            for succ in successors[key]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)
                    changed = True
            if changed:
                ready.sort(key=lambda k: slice_cx[k])

        if len(ordered) < len(slice_spines):
            return None  # Cyclic constraints; caller falls back to geometry

        n_spines = len(measures)
        slices = [
            [slice_spines[key].get(s_idx, []) for s_idx in range(n_spines)]
            for key in ordered
        ]
        return slices, [slice_cx[key] for key in ordered]

    @staticmethod
    def _slices_from_geometry(measures: list):
        """Fallback: cluster events into time slices by horizontal proximity.

        Returns (slices, slice_cxs) like _slices_from_sync.
        """
        TOLERANCE = 30.0  # Pixels. Events within this horizontal distance sync up.

        all_cxs = sorted(cx for m in measures for cx in m.cxs)
        if not all_cxs:
            return [[[] for _ in measures]], [0.0]

        clusters = []
        current_cluster = [all_cxs[0]]
        for cx in all_cxs[1:]:
            if cx - current_cluster[0] <= TOLERANCE:
                current_cluster.append(cx)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [cx]
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        merged_timeline = sorted(clusters)
        slices = [[[] for _ in measures] for _ in merged_timeline]
        for s_idx, m in enumerate(measures):
            for token, cx in zip(m.tokens, m.cxs):
                t_idx = min(range(len(merged_timeline)),
                            key=lambda j: abs(merged_timeline[j] - cx))
                slices[t_idx][s_idx].append(token)
        return slices, merged_timeline

    def _synchronize_measures(self):
        """Align measures across spines, preferring sync edges over geometry."""
        if not self.spines:
            return

        # 1. Pad to the same number of measures across all spines
        max_measures = max(len(s.measures) for s in self.spines)
        for spine in self.spines:
            while len(spine.measures) < max_measures:
                spine.measures.append(Measure("_pad"))

        # 2. Slice each measure column into simultaneous rows
        for m_idx in range(max_measures):
            measures = [spine.measures[m_idx] for spine in self.spines]

            if not any(m.tokens for m in measures):
                for m in measures:
                    m.tokens = ["."]
                    m.dynam_tokens = [self._dynam_row_text(m.dynamics)]
                self._insert_interp_rows(measures, [])
                continue

            # A full-measure rest starts at the measure's first beat even
            # though its glyph is drawn centered — keep it out of the cx-based
            # slicing and pin it to the first row afterwards
            rest_spines = [
                i for i, m in enumerate(measures)
                if m.full_measure_rest and len(m.tokens) == 1
            ]
            sliceable = [
                Measure("_rest") if i in rest_spines else m
                for i, m in enumerate(measures)
            ]

            sliced = self._slices_from_sync(sliceable)
            if sliced is None:
                sliced = self._slices_from_geometry(sliceable)
            slices, slice_cxs = sliced

            for i in rest_spines:
                slices[0][i] = [measures[i].tokens[0]]

            for s_idx, m in enumerate(measures):
                m.tokens = [
                    " ".join(sl[s_idx]) if sl[s_idx] else "."
                    for sl in slices
                ]
                m.dynam_tokens = self._place_dynamics(m.dynamics, slice_cxs)

            self._insert_interp_rows(measures, slice_cxs)

    @staticmethod
    def _dynam_row_text(dynamics: list) -> str:
        """Merge a measure's dynamic markings into a single row token."""
        if not dynamics:
            return "."
        return " ".join(text for _, text in sorted(dynamics))

    @staticmethod
    def _insert_interp_rows(measures: list, slice_cxs: list):
        """Insert interpretation rows shared by every spine.

        Pedals: '*ped' goes before the slice nearest the press point and
        '*Xped' after the slice nearest the release. Clef/key/meter changes
        go before the first slice drawn to the right of their glyph.
        Interpretation rows must be interpretation-only, so every other
        spine (and the **dynam columns) carries '*' there.
        """
        slots = {}
        for s_idx, m in enumerate(measures):
            for cx, text in m.pedals:
                if slice_cxs:
                    near = min(range(len(slice_cxs)),
                               key=lambda j: abs(slice_cxs[j] - cx))
                else:
                    near = 0
                slot = near if text == "*ped" else near + 1
                slots.setdefault(slot, []).append((cx, s_idx, text))
            for cx, text in m.interps:
                slot = sum(1 for scx in slice_cxs if scx < cx)
                slots.setdefault(slot, []).append((cx, s_idx, text))

        if not slots:
            return

        # Marks with the same token in the same slot share one row (e.g. a
        # meter change printed on every staff)
        slot_rows = {}
        for slot, entries in slots.items():
            rows, by_token = [], {}
            for cx, s_idx, token in sorted(entries):
                if token in by_token:
                    by_token[token][2].add(s_idx)
                else:
                    row = (cx, token, {s_idx})
                    by_token[token] = row
                    rows.append(row)
            slot_rows[slot] = rows

        n_rows = len(measures[0].tokens)
        for s_idx, m in enumerate(measures):
            new_tokens, new_dynams = [], []
            for row_idx in range(n_rows + 1):
                for cx, token, owners in slot_rows.get(row_idx, []):
                    new_tokens.append(token if s_idx in owners else "*")
                    new_dynams.append("*")
                if row_idx < n_rows:
                    new_tokens.append(m.tokens[row_idx])
                    new_dynams.append(m.dynam_tokens[row_idx])
            m.tokens, m.dynam_tokens = new_tokens, new_dynams

    @staticmethod
    def _place_dynamics(dynamics: list, slice_cxs: list) -> list:
        """Distribute (cx, text) dynamics onto the slice rows nearest to them."""
        n_rows = max(1, len(slice_cxs))
        row = ["."] * n_rows
        for cx, text in sorted(dynamics):
            if slice_cxs:
                idx = min(range(len(slice_cxs)), key=lambda j: abs(slice_cxs[j] - cx))
            else:
                idx = 0
            row[idx] = text if row[idx] == "." else f"{row[idx]} {text}"
        return row

    def merge_spines(self) -> str:
        """Builds all spines and transposes them into tab-separated horizontal rows."""
        if not self.spines:
            return ""

        self._synchronize_measures()

        # Humdrum orders spines bottom-to-top; each staff's **dynam column
        # rides directly to the right of its **kern column
        built_columns = []
        for spine in reversed(self.spines):
            built_columns.append(spine.build())
            if spine.has_dynamics():
                built_columns.append(spine.build_dynam())

        lines = []
        total_rows = len(built_columns[0])

        for row_idx in range(total_rows):
            lines.append("\t".join(col[row_idx] for col in built_columns))

        return "\n".join(lines)
        
class MinimalHumdrumSerializer:
    """Serializes a PyG music graph into Humdrum **kern format, one page at a time."""

    def __init__(self, edge_index, edge_predictions, node_roles, pyg_node_ids):
        self.edge_index = edge_index
        self.edge_predictions = edge_predictions
        self.node_roles = node_roles
        self.pyg_node_ids = pyg_node_ids

        self.context = HumdrumContext(self._build_sync_groups())
        self._head_initialized = False
        # Per-page map: slur/tie node ID -> [(anchored event ID, cx), ...]
        self._span_anchors = {}

    def _build_sync_groups(self) -> dict:
        """Union-find over Class 5 (sync) edges: maps node ID -> simultaneity group."""
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(self.edge_index.shape[1]):
            if self.edge_predictions[i].item() != 5:
                continue
            u = self.pyg_node_ids[self.edge_index[0, i].item()]
            v = self.pyg_node_ids[self.edge_index[1, i].item()]
            parent.setdefault(u, u)
            parent.setdefault(v, v)
            parent[find(u)] = find(v)

        return {node: find(node) for node in parent}

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

        # Median notehead height of the page: the grace test compares each
        # head against it, so "grace" means small relative to THIS score's
        # noteheads. Some render styles draw regular heads at ~0.8 staff
        # space, which a staff-space-only threshold misread as all-grace
        # (one validation file serialized every note with 'q': rhythm 0%).
        head_heights = sorted(
            n['bbox'][3] - n['bbox'][1]
            for n in page_nodes
            if n.get('class') in ('noteheadBlack', 'noteheadHalf', 'noteheadWhole')
        )
        self._page_median_head_h = (
            head_heights[len(head_heights) // 2] if head_heights else None
        )

        # Index spanning curves by their anchored events so the open/close
        # decision can compare the two end notes directly. Anchors are keyed
        # by (system row, cx): a tie across a system break ends on a note
        # whose cx is far LEFT of where it started, so raw cx ordering would
        # reverse the open/close roles.
        systems_sorted = sorted(
            (n for n in nodes.values() if n['class'] == 'system'),
            key=lambda s: s['bbox'][1]
        )

        def _anchor_key(node: dict) -> tuple:
            cy = node.get('cy', 0.0)
            row, best = 0, float('inf')
            for i, sys_node in enumerate(systems_sorted):
                dist = max(sys_node['bbox'][1] - cy, cy - sys_node['bbox'][3], 0.0)
                if dist < best:
                    best, row = dist, i
            return (row, node.get('cx', 0.0))

        self._span_anchors = {}
        for parent_id, child_list in children.items():
            for child_id, e_class in child_list:
                if e_class == 2 and nodes.get(child_id, {}).get('class') in ('slur', 'tie'):
                    parent = nodes.get(parent_id, {})
                    self._span_anchors.setdefault(child_id, []).append(
                        (parent_id, _anchor_key(parent))
                    )

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
        # 1. Grab all system-staff nodes to use their tight 5-line bounding boxes
        all_system_staves = [n for n in nodes.values() if n.get('class') == 'system-staff']

        staves = [
            v for v, e_class in children.get(measure_id, [])
            if e_class == 1 and nodes.get(v, {}).get('class') == 'staff'
        ]
        staves.sort(key=lambda s_id: _staff_sort_key(nodes[s_id], all_system_staves))

        # 2. Find each staff's system-staff row (tight 5-line bbox, by y-overlap)
        staff_rows = []
        for staff_id in staves:
            row = _find_staff_row(nodes[staff_id], all_system_staves)
            staff_rows.append(
                row['bbox'] if row is not None
                else nodes[staff_id].get('bbox', [0, 0, 0, 0])
            )

        # Dynamics and pedal spans hang off the measure node itself; split
        # them per staff row
        dynamics_per_staff = self._collect_dynamics(measure_id, children, nodes, staff_rows)
        pedals_per_staff = self._collect_pedals(measure_id, children, nodes, staff_rows)

        for spine_idx, staff_id in enumerate(staves):
            if spine_idx >= len(self.context.spines):
                continue

            spine = self.context.spines[spine_idx]
            measure = Measure(measure_id)
            measure.dynamics = dynamics_per_staff.get(spine_idx, [])
            measure.pedals = pedals_per_staff.get(spine_idx, [])
            active_bbox = staff_rows[spine_idx]

            # 3. Mid-piece signature changes: every system restates its
            # clef/keySig/meterSig on the staff — a differing value is a
            # change that takes effect here and becomes an interpretation row
            self._detect_signature_changes(spine, measure, staff_id, children, nodes)

            # 3b. Collect events and analyze each one; mid-measure clef
            # changes re-clef every event drawn to their right
            events, tuplet_ratios, beam_of, clef_marks = \
                self._collect_staff_events(staff_id, children, nodes)
            event_infos = [
                self._analyze_event(
                    ev, children, nodes,
                    self._clef_at(ev.get('cx', 0.0), clef_marks, spine.clef_type),
                    active_bbox,
                )
                for ev in events
            ]

            running_clef = spine.clef_type
            for cx, clef_type in clef_marks:
                if clef_type != running_clef:
                    measure.interps.append((cx, CLEF_KERN[clef_type]))
                    running_clef = clef_type
            spine.clef_type = running_clef

            # 3b. Mark beam groups (kern L/J) and use the beam class as a
            # duration fallback before tuplet scaling / ambiguity resolution
            self._apply_beams(events, event_infos, beam_of, nodes)

            # 3c. The stacked dots of simultaneous stacked voices may all be
            # attached to a single note — share them across the stack
            self._share_simultaneous_dots(event_infos)

            # 4. Scale tuplet durations: a triplet eighth is a kern '12'
            # (the scaled number is both the display token and the true value)
            for info, ev in zip(event_infos, events):
                ratio = tuplet_ratios.get(ev.get('id'))
                if ratio and not info.get('grace'):
                    num, numbase = ratio
                    scaled = max(1, round(info['duration'] * num / numbase))
                    info['duration'] = scaled
                    info['ambiguous'] = False
                    for sub in info.get('notes', []):
                        sub['duration'] = scaled

            # 5. Resolve ambiguous durations using the time signature. A whole
            # rest (or mRest) standing alone means a full measure of silence,
            # whatever the meter — restate it as the measure's true length.
            self._resolve_durations(event_infos, spine.meter_num, spine.meter_den)
            measure_value = spine.meter_num / spine.meter_den
            timed = [info for info in event_infos if not info.get('grace')]
            if (len(timed) == 1 and timed[0]['type'] in ('rest', 'mrest')
                    and timed[0]['duration'] == 1):
                timed[0]['duration'], timed[0]['dots'] = _value_to_duration(measure_value)
                measure.full_measure_rest = True
            for info in event_infos:
                if info['type'] == 'mrest':
                    info['duration'], info['dots'] = _value_to_duration(measure_value)

            # 6. Spell pitches absolutely: apply the key signature and
            # in-measure accidental carry-over (kern has no implicit keysig)
            spine.pending_ties = self._apply_key_signature(
                event_infos, spine.key_accidentals, spine.pending_ties
            )

            # 7. Serialize to kern tokens with durations, spatial cx and node ID
            for info in event_infos:
                # Grace notes take no time on the timeline
                dur = 0.0 if info.get('grace') else _note_value(info['duration'], info['dots'])
                cx = info.get('cx', 0.0)
                measure.add(self._info_to_kern(info), dur, cx, info.get('id'))

            spine.add_measure(measure)

    @staticmethod
    def _detect_signature_changes(spine, measure, staff_id: str,
                                  children: dict, nodes: dict):
        """Detect clef/key/meter restatements on a staff that differ from the
        spine's running state; emit them as interpretation rows and update
        the state so later measures use the new context."""
        for child_id, e_class in children.get(staff_id, []):
            if e_class != 1:
                continue
            node = nodes.get(child_id)
            if not node:
                continue
            cls_name = node['class']
            cx = node.get('cx', 0.0)

            if cls_name in CLEF_BOTTOM_LINE:
                if cls_name != spine.clef_type:
                    measure.interps.append((cx, CLEF_KERN[cls_name]))
                    spine.clef_type = cls_name
            elif cls_name.lower() == 'keysig':
                token = Spine._keysig_token(child_id, staff_id, children, nodes)
                if token != spine.key_token:
                    measure.interps.append((cx, token))
                    spine.key_token = token
                    spine.key_accidentals = _parse_key_accidentals(token)
            elif cls_name.lower() == 'metersig':
                token = Spine._metersig_token(child_id, children, nodes)
                if token != spine.meter_token:
                    measure.interps.append((cx, token))
                    spine.meter_token = token
                    spine.meter_num, spine.meter_den = Spine._parse_meter(token)

    @staticmethod
    def _clef_at(cx: float, clef_marks: list, base_clef: str) -> str:
        """The clef in effect at horizontal position cx, given the measure's
        cx-sorted clef changes and the clef the measure started with."""
        clef = base_clef
        for mark_cx, clef_type in clef_marks:
            if mark_cx <= cx:
                clef = clef_type
            else:
                break
        return clef

    @staticmethod
    def _collect_dynamics(measure_id: str, children: dict, nodes: dict,
                          staff_rows: list) -> dict:
        """Gather a measure's dynamic glyphs as {staff_index: [(cx, text)]}.

        Adjacent glyphs merge into one marking (e.g. 'f' + 'z' -> 'fz'); each
        marking is assigned to the staff row it sits closest to vertically.
        """
        per_staff = {}
        if not staff_rows:
            return per_staff

        glyphs = [
            nodes[child_id]
            for child_id, e_class in children.get(measure_id, [])
            if e_class == 1 and nodes.get(child_id, {}).get('class') in DYNAMIC_KERN
        ]
        if not glyphs:
            return per_staff

        glyphs.sort(key=lambda n: n.get('cx', 0.0))
        heights = [row[3] - row[1] for row in staff_rows]
        staff_space = max(1.0, (sum(heights) / len(heights)) / 4.0)

        clusters = []
        for glyph in glyphs:
            if clusters:
                prev = clusters[-1][-1]
                gap = glyph['bbox'][0] - prev['bbox'][2]
                if (gap < staff_space
                        and abs(glyph.get('cy', 0.0) - prev.get('cy', 0.0)) < 2 * staff_space):
                    clusters[-1].append(glyph)
                    continue
            clusters.append([glyph])

        for cluster in clusters:
            text = "".join(DYNAMIC_KERN[g['class']] for g in cluster)
            cx = (min(g['bbox'][0] for g in cluster)
                  + max(g['bbox'][2] for g in cluster)) / 2.0
            cy = sum(g.get('cy', 0.0) for g in cluster) / len(cluster)
            staff_idx = _assign_staff(cy, staff_rows, staff_space)
            per_staff.setdefault(staff_idx, []).append((cx, text))

        return per_staff

    @staticmethod
    def _collect_pedals(measure_id: str, children: dict, nodes: dict,
                        staff_rows: list) -> dict:
        """Gather a measure's pedal spans as {staff_index: [(cx, text)]}.

        A pedal annotation covers the whole press-to-release line, so its
        left edge becomes '*ped' and its right edge '*Xped'.
        """
        per_staff = {}
        if not staff_rows:
            return per_staff

        heights = [row[3] - row[1] for row in staff_rows]
        staff_space = max(1.0, (sum(heights) / len(heights)) / 4.0)

        for child_id, e_class in children.get(measure_id, []):
            node = nodes.get(child_id)
            if e_class == 1 and node and node.get('class') == 'pedal':
                x1, _, x2, _ = node['bbox']
                staff_idx = _assign_staff(node.get('cy', 0.0), staff_rows, staff_space)
                marks = per_staff.setdefault(staff_idx, [])
                marks.append((x1, "*ped"))
                marks.append((x2, "*Xped"))

        return per_staff

    def _share_simultaneous_dots(self, event_infos: list):
        """Share augmentation dots among simultaneous notes of equal duration.

        Stacked voices (e.g. three layers holding a double-dotted triad) are
        drawn with one dot column per row, but the graph may attach every dot
        glyph to a single note. Group events by sync group (or horizontal
        proximity) and give same-duration notes the group's max dot count.
        Different durations keep their own dots: a half against a dotted
        quarter at the same onset is legitimate.
        """
        groups = {}
        solo = []
        for info in event_infos:
            if info['type'] not in ('note', 'chord'):
                continue
            gid = self.context.sync_groups.get(info.get('id'))
            if gid is not None:
                groups.setdefault(gid, []).append(info)
            else:
                solo.append(info)

        # Cluster sync-less events by horizontal proximity
        solo.sort(key=lambda i: i.get('cx', 0.0))
        cluster = []
        for info in solo:
            if cluster and info.get('cx', 0.0) - cluster[-1].get('cx', 0.0) >= 30.0:
                groups[('cx', len(groups))] = cluster
                cluster = []
            cluster.append(info)
        if cluster:
            groups[('cx', len(groups))] = cluster

        for members in groups.values():
            if len(members) < 2:
                continue
            by_duration = {}
            for info in members:
                by_duration.setdefault((info['duration'], info.get('grace', False)),
                                       []).append(info)
            for same in by_duration.values():
                best = max(i['dots'] for i in same)
                for info in same:
                    info['dots'] = best
                    for sub in info.get('notes', []):
                        sub['dots'] = best

    @staticmethod
    def _apply_beams(events, event_infos: list, beam_of: dict, nodes: dict):
        """Mark kern beam start/end (L/J) on the first and last beamed event.

        The beam class (beam8, beam16...) also acts as a duration fallback for
        member notes whose stem/flag didn't resolve a definitive duration.
        """
        groups = {}
        for pos, ev in enumerate(events):
            beam_id = beam_of.get(ev.get('id'))
            if beam_id is not None and event_infos[pos]['type'] in ('note', 'chord'):
                groups.setdefault(beam_id, []).append(pos)

        for beam_id, positions in groups.items():
            beam_cls = nodes.get(beam_id, {}).get('class', '')
            digits = ''.join(ch for ch in beam_cls if ch.isdigit())
            if digits:
                for pos in positions:
                    info = event_infos[pos]
                    if info['ambiguous']:
                        info['duration'] = int(digits)
                        info['ambiguous'] = False
                        for sub in info.get('notes', []):
                            sub['duration'] = int(digits)
                            sub['ambiguous'] = False
            # A beam needs at least two events; anything less is likely noise
            if len(positions) >= 2:
                event_infos[min(positions)]['beam'] = 'L'
                event_infos[max(positions)]['beam'] = 'J'

    @staticmethod
    def _apply_key_signature(event_infos: list, key_accidentals: dict,
                             pending_ties: dict) -> dict:
        """Give every note its absolute chromatic spelling.

        Kern pitches are absolute: under *k[b-] a plain 'b' still means B
        natural. Notes without an explicit accidental glyph inherit the key
        signature, and an explicit glyph carries over to later notes on the
        same line/space until the end of the measure. A note closing a tie
        inherits the spelling of the note that opened it, even across the
        barline. Returns the ties left open for the next measure.
        """
        state = {}  # kern pitch (letter+octave) -> accidental active in this measure
        outgoing = {}  # Ties opened here, to be closed in the next measure
        for info in event_infos:
            if info['type'] == 'note':
                sub_notes = [info]
            elif info['type'] == 'chord':
                sub_notes = info.get('notes', [])
            else:
                continue
            for note in sub_notes:
                pitch = note.get('pitch')
                if not pitch:
                    continue
                closes_tie = (']' in note.get('suffix', '')
                              or ']' in info.get('suffix', ''))
                opens_tie = ('[' in note.get('prefix', '')
                             or '[' in info.get('prefix', ''))
                if note.get('accidental'):
                    state[pitch] = note['accidental']
                elif closes_tie and pitch in pending_ties:
                    note['accidental'] = pending_ties[pitch]
                elif pitch in state:
                    note['accidental'] = '' if state[pitch] == 'n' else state[pitch]
                else:
                    note['accidental'] = key_accidentals.get(pitch[0].upper(), '')
                if opens_tie:
                    outgoing[pitch] = note['accidental']
        return outgoing

    # --- Event collection ---

    def _collect_staff_events(self, staff_id: str, children: dict, nodes: dict) -> tuple:
        """Find temporal-anchor events under a staff, ordered by Class 3 (Temporal) edges.

        Returns (sorted_events, tuplet_ratios, beam_of, clef_marks) where
        tuplet_ratios maps event IDs to a (num, numbase) duration ratio for
        events inside tuplets, beam_of maps event IDs to their containing
        beam node ID, and clef_marks is a cx-sorted list of (cx, clef_type)
        mid-measure clef changes found under this staff.
        """
        # 1. Collect all structural descendants via Class 1 edges. Tuplets are
        # modifier-class containers (Class 2 edge from the layer) but hold
        # their notes via Class 1 edges, so descend through them as well,
        # remembering which tuplet an event belongs to. Beams are structural
        # containers (layer -> beam -> note/chord) and are traversed the same
        # way, remembering which beam an event belongs to.
        descendants = set()
        tuplet_of = {}
        beam_of = {}
        clef_marks = []
        stack = [(staff_id, None, None)]
        while stack:
            curr, tuplet_ctx, beam_ctx = stack.pop()
            for child_id, e_class in children.get(curr, []):
                if child_id in descendants:
                    continue
                child_cls = nodes.get(child_id, {}).get('class', '')
                is_tuplet_container = (e_class == 2 and child_cls == 'tuplet')
                if e_class == 1 or is_tuplet_container:
                    descendants.add(child_id)
                    if child_cls in CLEF_CHANGE_MAP:
                        clef_marks.append(
                            (nodes[child_id].get('cx', 0.0), CLEF_CHANGE_MAP[child_cls])
                        )
                    ctx = child_id if is_tuplet_container else tuplet_ctx
                    b_ctx = child_id if child_cls.startswith('beam') else beam_ctx
                    if ctx is not None:
                        tuplet_of[child_id] = ctx
                    if b_ctx is not None and b_ctx != child_id:
                        beam_of[child_id] = b_ctx
                    stack.append((child_id, ctx, b_ctx))
        clef_marks.sort()

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

        # 6. Derive tuplet duration ratios: a tuplet of N events plays N in the
        # time of the largest power of two below N (3:2, 5:4, 6:4, 7:4...)
        events_per_tuplet = {}
        for ev_id in top_level_events:
            t_id = tuplet_of.get(ev_id)
            if t_id is not None:
                events_per_tuplet.setdefault(t_id, []).append(ev_id)

        tuplet_ratios = {}
        for t_id, ev_ids in events_per_tuplet.items():
            num = len(ev_ids)
            if num < 3:
                continue
            numbase = 1
            while numbase * 2 < num:
                numbase *= 2
            for ev_id in ev_ids:
                tuplet_ratios[ev_id] = (num, numbase)

        return sorted_events, tuplet_ratios, beam_of, clef_marks

    # --- Event analysis (returns structured dicts) ---

    def _count_dots(self, node_id: str, children: dict, nodes: dict,
                    staff_space: float = 0.0) -> int:
        """Count augmentation dots attached to a node from its 'dots' glyph bbox.

        A single dot is roughly square; two dots side by side are much wider.
        On chords the dots of every chord note merge into one tall glyph, so
        the aspect ratio is useless there — compare the width against the
        staff space instead (one dot column is ~0.4 spaces, two are ~1.1).
        A node may also carry SEVERAL dots glyphs (one per chord-note row);
        they all show the same augmentation, so take the max, never the sum.
        """
        dot_count = 0
        for child_id, _ in children.get(node_id, []):
            child = nodes.get(child_id)
            if child and child['class'] == 'dots':
                bbox = child.get('bbox', [0, 0, 0, 0])
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                is_double = (
                    (height > 0 and width / height > 1.8)
                    or (staff_space > 0 and width > 0.55 * staff_space)
                )
                dot_count = max(dot_count, 2 if is_double else 1)
        return dot_count

    def _collect_ornaments(self, event: dict, children: dict, nodes: dict) -> tuple:
        """Collect kern prefix/suffix decorations from an event's modifier children.

        Spanning curves (slur, tie) are linked to both end notes in the graph;
        the leftmost anchored note opens the span, the other one closes it.
        """
        prefix, suffix = "", ""
        ev_id = event.get('id', '')
        for child_id, e_class in children.get(ev_id, []):
            if e_class != 2:
                continue
            child = nodes.get(child_id)
            if not child:
                continue
            ccls = child['class']
            if ccls in ORNAMENT_KERN:
                suffix += ORNAMENT_KERN[ccls]
            elif ccls in ('slur', 'tie'):
                open_ch, close_ch = ('(', ')') if ccls == 'slur' else ('[', ']')
                anchors = self._span_anchors.get(child_id, [])
                own_key = next((key for aid, key in anchors if aid == ev_id), None)
                others = [(key, aid) for aid, key in anchors if aid != ev_id]
                if others and own_key is not None:
                    is_start = (own_key, ev_id) <= min(others)
                else:
                    # Other end not on this page: fall back to curve geometry
                    d_left = abs(event.get('cx', 0.0) - child['bbox'][0])
                    d_right = abs(event.get('cx', 0.0) - child['bbox'][2])
                    is_start = d_left <= d_right
                if is_start:
                    prefix += open_ch
                else:
                    suffix += close_ch
        return prefix, suffix

    def _analyze_event(self, event: dict, children: dict, nodes: dict,
                       clef_type: str, staff_bbox: list) -> dict:
        """Analyze a graph event and return a structured info dict."""
        cls = event['class']
        base_cx = event.get('cx', 0.0)
        event_id = event.get('id')

        if cls in REST_KERN:
            dur = int(REST_KERN[cls].replace('r', ''))
            staff_space = (staff_bbox[3] - staff_bbox[1]) / 4.0
            dots = self._count_dots(event.get('id', ''), children, nodes, staff_space)
            return {"type": "rest", "duration": dur, "dots": dots, "ambiguous": False, "cx": base_cx, "id": event_id}

        if cls == "mRest":
            return {"type": "mrest", "duration": 1, "dots": 0, "ambiguous": False, "cx": base_cx, "id": event_id}

        if cls == "note":
            return self._analyze_note(event, children, nodes, clef_type, staff_bbox)

        if cls == "chord":
            return self._analyze_chord(event, children, nodes, clef_type, staff_bbox)

        return {"type": "unknown", "duration": 4, "dots": 0, "ambiguous": False, "id": event_id}

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

        accidental = ""
        notehead_cy = note_node.get('cy', 0.0)
        notehead_cx = note_node.get('cx', 0.0)
        notehead_dur = None
        stem_dur = None
        flag_dur = None
        is_grace = False
        staff_space = (staff_bbox[3] - staff_bbox[1]) / 4.0

        # 2. Analyze the collected components to determine the note's properties
        for desc_id in descendants:
            child = nodes.get(desc_id)
            if not child:
                continue

            cls = child.get('class', '')

            # Noteheads set the base duration and the Y-coordinate for pitch
            if cls in NOTEHEAD_BASE_DURATION:
                notehead_dur = NOTEHEAD_BASE_DURATION[cls]
                notehead_cy = child.get('cy', notehead_cy)
                notehead_cx = child.get('cx', notehead_cx)
                # Grace noteheads are drawn at ~0.75x scale. Test against
                # the staff space AND the page's median head height — some
                # render styles draw regular heads at only ~0.8 staff space,
                # so the absolute test alone flags whole scores as grace.
                notehead_h = child['bbox'][3] - child['bbox'][1]
                median_h = getattr(self, '_page_median_head_h', None)
                if (staff_space > 0 and notehead_h < 0.85 * staff_space
                        and (median_h is None or notehead_h < 0.85 * median_h)):
                    is_grace = True

            if cls in STEM_DURATION:
                stem_dur = STEM_DURATION[cls]

            if cls in FLAG_DURATION:
                flag_dur = FLAG_DURATION[cls]

        # Duration precedence: a half/whole notehead is authoritative (its
        # plain stem is annotated 'stem4' but says nothing about duration);
        # for black noteheads, flags beat stems beat the bare notehead
        duration = 4
        has_definitive_duration = False
        if notehead_dur in (1, 2):
            duration, has_definitive_duration = notehead_dur, True
        elif flag_dur is not None:
            duration, has_definitive_duration = flag_dur, True
        elif stem_dur is not None:
            duration, has_definitive_duration = stem_dur, True
        elif notehead_dur is not None:
            duration = notehead_dur

            # Grab accidentals if present
            if cls in ACCIDENTAL_KERN:
                accidental = ACCIDENTAL_KERN[cls]

        # Accidentals are modifiers, attached with Class 2 edges rather than
        # structural ones, so check the note's direct children as well
        for child_id, e_class in children.get(note_node['id'], []):
            if e_class != 2:
                continue
            child = nodes.get(child_id)
            if child and child['class'] in ACCIDENTAL_KERN:
                accidental = ACCIDENTAL_KERN[child['class']]

        # 3. Check for dots: usually attached to the note, occasionally to its notehead
        dot_count = self._count_dots(note_node['id'], children, nodes, staff_space)
        if dot_count == 0:
            for desc_id in descendants:
                if nodes.get(desc_id, {}).get('class', '') in NOTEHEAD_BASE_DURATION:
                    dot_count = self._count_dots(desc_id, children, nodes, staff_space)
                    if dot_count:
                        break

        is_ambiguous = (duration == 4 and not has_definitive_duration and not is_grace)
        pitch = self._position_to_kern_pitch(notehead_cy, clef_type, staff_bbox)
        prefix, suffix = self._collect_ornaments(note_node, children, nodes)

        return {
            "type": "note",
            "duration": duration,
            "pitch": pitch,
            "accidental": accidental,
            "dots": dot_count,
            "ambiguous": is_ambiguous,
            "grace": is_grace,
            "prefix": prefix,
            "suffix": suffix,
            "cx": notehead_cx,
            "id": note_node.get('id')
        }

    def _analyze_chord(self, chord_node: dict, children: dict, nodes: dict,
                       clef_type: str, staff_bbox: list) -> dict:
        """Analyze a chord node and return structured info."""
        chord_children = children.get(chord_node['id'], [])

        notes = []
        shared_ambiguous = False
        shared_duration = 4
        shared_dots = 0

        for child_id, e_class in chord_children:
            # Only structural children are chord members; temporal (3) and
            # sync (5) edges point to *other* events and must not be absorbed
            if e_class != 1:
                continue
            child = nodes.get(child_id)
            if child and child['class'] == 'note':
                note_info = self._analyze_note(child, children, nodes, clef_type, staff_bbox)
                notes.append(note_info)
                # All chord notes share the same duration — use the first one's info
                if not notes[1:]:
                    shared_duration = note_info['duration']
                    shared_dots = note_info['dots']
                    shared_ambiguous = note_info['ambiguous']

        # A chord's dots may attach to the chord node itself or to any single
        # member note (one glyph per row); they all show the same augmentation
        # — take the max and give it to every chord note's kern token
        staff_space = (staff_bbox[3] - staff_bbox[1]) / 4.0
        chord_dots = self._count_dots(chord_node['id'], children, nodes, staff_space)
        shared_dots = max([shared_dots, chord_dots] + [n['dots'] for n in notes])
        for note in notes:
            note['dots'] = shared_dots

        chord_cx = notes[0]['cx'] if notes else chord_node.get('cx', 0.0)
        prefix, suffix = self._collect_ornaments(chord_node, children, nodes)
        return {
            "type": "chord",
            "duration": shared_duration,
            "dots": shared_dots,
            "ambiguous": shared_ambiguous,
            "grace": any(n.get('grace') for n in notes),
            "prefix": prefix,
            "suffix": suffix,
            "notes": notes,
            "cx": chord_cx,
            "id": chord_node.get('id')
        }

    # --- Duration resolution ---

    @staticmethod
    def _resolve_durations(event_infos: list, meter_num: int, meter_den: int):
        """Resolve ambiguous durations using the time signature."""
        expected_total = meter_num / meter_den  # Measure length in whole-note units

        known_total = 0.0
        ambiguous_indices = []

        for i, info in enumerate(event_infos):
            if info.get('grace'):
                continue  # Grace notes consume no measure time
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
            if info.get('grace'):
                token += "q"
            token += "." * info['dots']
            token += info.get('beam', '')
            return f"{info.get('prefix', '')}{token}{info.get('suffix', '')}"

        if info['type'] == 'chord':
            note_tokens = []
            for note in info.get('notes', []):
                t = f"{note['duration']}{note['pitch']}{note.get('accidental', '')}"
                if note.get('grace'):
                    t += "q"
                t += "." * note['dots']
                t = f"{note.get('prefix', '')}{t}{note.get('suffix', '')}"
                note_tokens.append(t)
            note_tokens.sort()
            if not note_tokens:
                return "."
            return (f"{info.get('prefix', '')}{' '.join(note_tokens)}"
                    f"{info.get('beam', '')}{info.get('suffix', '')}")

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

        # Octave clefs shift the sounding pitch by a full octave
        if "8vb" in clef_type:
            bottom_oct -= 1
        elif "8va" in clef_type:
            bottom_oct += 1

        total = bottom_idx + step

        # Calculate diatonic note index and octave; clamp the octave so
        # degenerate staves (e.g. 1-line percussion) can't explode the range
        note_idx = total % 7
        octave = max(0, min(8, bottom_oct + (total // 7)))
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