import logging
import pprint
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)

class ContextTracker:
    def __init__(self):
        self.active_clefs = {}
        self.printed_clefs = {}  # Tracks what was actually output to the file
        self.measure_accidentals = defaultdict(dict)
        self.staff_bounds = {}

    def process_context_node(self, node, staff_id):
        """Updates internal state and returns a string if the serializer needs to print something."""
        node_class = node["class"]
        
        if node_class.startswith("barline"):
            self.measure_accidentals[staff_id].clear()
            return None
            
        elif node_class.startswith("clef"):
            # 1. Update the logical state for pitch calculation math
            clef_id = node_class.split("-")[-1]
            clef_map = {"G": "treble", "F": "bass", "C": "alto"}
            self.active_clefs[staff_id] = clef_map.get(clef_id, "treble")
            
            # 2. Check the serialization state: Do we need to print this?
            formatted_clef = node_class.replace("clef", "*clef")
            if self.printed_clefs.get(staff_id) != formatted_clef:
                self.printed_clefs[staff_id] = formatted_clef
                return formatted_clef  # Tell the serializer to print it!
            
            return None  # Already printed, stay silent

    def register_inline_accidental(self, accid_class, pure_pitch, staff_id):
        symbol = {"sharp": "#", "flat": "-", "nat": "n"}.get(accid_class.split("-")[-1], "")
        if symbol:
            self.measure_accidentals[staff_id][pure_pitch] = symbol

    def get_effective_accidental(self, pure_pitch, staff_id):
        mod = self.measure_accidentals[staff_id].get(pure_pitch, "")
        return "" if mod == "n" else mod


class HumdrumSerializer:
    def __init__(self, node_list, edge_index, edge_predictions, node_roles, pyg_node_ids):
        self.nodes = {n['id']: n for n in node_list}
        self.roles = node_roles
        self.tracker = ContextTracker()
        
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.nodes.keys())
        
        # Safely map PyG integers back to JSON string IDs
        for i in range(edge_index.shape[1]):
            u_str, v_str = pyg_node_ids[edge_index[0, i].item()], pyg_node_ids[edge_index[1, i].item()]
            e_class = edge_predictions[i].item()
            if e_class > 0 and u_str in self.nodes and v_str in self.nodes:
                self.G.add_edge(u_str, v_str, edge_class=e_class)

    def _derive_duration(self, components):
        classes = [n['class'] for n in components]
        if any("Whole" in c for c in classes) or "noteheadWhole" in classes: return "1"
        if any("Half" in c for c in classes) or "noteheadHalf" in classes: return "2"
        if any("16th" in c for c in classes): return "16"
        if any("8th" in c for c in classes) or "beam" in classes: return "8"
        return "4" 

    def _calculate_pitch(self, notehead_cy, staff_id):
        bounds = self.tracker.staff_bounds.get(staff_id, (0, 100))
        staff_height = bounds[1] - bounds[0]
        if staff_height <= 0: return "c4" 
        
        slot = round((notehead_cy - bounds[0]) / (staff_height / 8.0))
        maps = {
            "treble": {-2: "aa", -1: "gg", 0: "ff", 1: "ee", 2: "dd", 3: "cc", 4: "b", 5: "a", 6: "g", 7: "f", 8: "e", 9: "d", 10: "c"},
            "bass": {0: "A", 1: "G", 2: "F", 3: "E", 4: "D", 5: "C", 6: "BB", 7: "AA", 8: "GG", 9: "FF", 10: "EE"}
        }
        active_clef = self.tracker.active_clefs.get(staff_id, "treble")
        current_map = maps.get(active_clef, maps["treble"])
        return current_map[max(min(current_map.keys()), min(slot, max(current_map.keys())))]

    def _build_super_node(self, anchor_id):
        data = self.nodes[anchor_id]
        components, modifiers = [data], []
        for neighbor in self.G.successors(anchor_id):
            e_class = self.G.get_edge_data(anchor_id, neighbor)['edge_class']
            if e_class == 1: components.append(self.nodes[neighbor])
            elif e_class == 2: modifiers.append(self.nodes[neighbor])
        return {'class': data['class'], 'node': data, 'components': components, 'modifiers': modifiers}

    def _format_token(self, sn, staff_id):
        cls = sn['class']
        if "rest" in cls.lower():
            return f"{self._derive_duration(sn['components'])}r"
        if cls in self.roles["node_roles"].get("temporal_anchors", []):
            duration = self._derive_duration(sn['components'])
            pure_pitch = self._calculate_pitch(sn['node']['cy'], staff_id)
            for mod in sn['modifiers']:
                if mod['class'].startswith("accid"):
                    self.tracker.register_inline_accidental(mod['class'], pure_pitch, staff_id)
            accid = self.tracker.get_effective_accidental(pure_pitch, staff_id)
            mod_string = "".join(["'" if "Staccato" in m['class'] else ";" if "fermata" in m['class'].lower() else "." if "dot" in m['class'] else "" for m in sn['modifiers']])
            return f"{duration}{pure_pitch}{accid}{mod_string}"
        return "."

    def export_to_krn(self):
        kern_lines = []
        systems = sorted([n for n in self.nodes.values() if n['class'] == 'system'], key=lambda s: s['cy'])
        
        sorted_measures = []
        for sys in systems:
            sys_measures = [self.nodes[v] for u, v, d in self.G.edges(data=True) if u == sys['id'] and d['edge_class'] == 1 and v in self.nodes]
            sorted_measures.extend(sorted(sys_measures, key=lambda m: m['cx']))
            
        if not sorted_measures:
            sorted_measures = sorted([n for n in self.nodes.values() if n['class'] == 'measure'], key=lambda m: (round(m['cy'] / 200), m['cx']))

        # Determine column count
        spine_count = 1
        for m in sorted_measures:
            staves = [v for u, v, d in self.G.edges(data=True) if u == m['id'] and d['edge_class'] == 1 and self.nodes[v]['class'] == 'staff']
            if len(staves) > spine_count:
                spine_count = len(staves)
                
        kern_lines.append("\t".join(["**kern"] * spine_count))

        # FIX: Track the last printed clef per column (spine_idx)
        last_printed_clef = {}

        for m in sorted_measures:
            staves = sorted([v for u, v, d in self.G.edges(data=True) if u == m['id'] and d['edge_class'] == 1 and self.nodes[v]['class'] == 'staff'], key=lambda s_id: self.nodes[s_id]['cy'])
            
            measure_headers = [""] * spine_count
            time_steps = defaultdict(lambda: ["."] * spine_count)
            
            for spine_idx, st_id in enumerate(staves):
                self.tracker.staff_bounds[st_id] = (self.nodes[st_id]['bbox'][1], self.nodes[st_id]['bbox'][3])
                staff_children = [v for u, v, d in self.G.edges(data=True) if u == st_id and d['edge_class'] == 1]
                
                for child_id in staff_children:
                    cls = self.nodes[child_id]['class']
                    
                    if cls.startswith("clef"):
                        # The tracker handles the math AND tells us if we should print
                        clef_to_print = self.tracker.process_context_node(self.nodes[child_id], st_id)
                        if clef_to_print:
                            measure_headers[spine_idx] = clef_to_print
                        
                        # Only print if it is a NEW clef for this specific spine
                        formatted_clef = cls.replace("clef", "*clef")
                        if last_printed_clef.get(spine_idx) != formatted_clef:
                            measure_headers[spine_idx] = formatted_clef
                            last_printed_clef[spine_idx] = formatted_clef
                    
                    elif cls == 'layer':
                        layer_children = [v for u, v, d in self.G.edges(data=True) if u == child_id and d['edge_class'] == 1 and v in self.nodes]
                        for event_id in layer_children:
                            event = self.nodes[event_id]
                            rounded_x = round(event['cx'] / 5.0) * 5.0 
                            token = self._format_token(self._build_super_node(event_id), st_id)
                            
                            if token != ".": 
                                if time_steps[rounded_x][spine_idx] == ".":
                                    time_steps[rounded_x][spine_idx] = token
                                else:
                                    time_steps[rounded_x][spine_idx] += f" {token}"
                    
                    elif cls in self.roles["node_roles"].get("temporal_anchors", []) or "rest" in cls.lower():
                        event = self.nodes[child_id]
                        rounded_x = round(event['cx'] / 5.0) * 5.0
                        token = self._format_token(self._build_super_node(child_id), st_id)
                        
                        if token != ".":
                            if time_steps[rounded_x][spine_idx] == ".":
                                time_steps[rounded_x][spine_idx] = token
                            else:
                                time_steps[rounded_x][spine_idx] += f" {token}"

            if any(measure_headers):
                kern_lines.append("\t".join([h if h else "*" for h in measure_headers]))
                
            for x_pos in sorted(time_steps.keys()):
                kern_lines.append("\t".join(time_steps[x_pos]))
                
            kern_lines.append("\t".join(["="] * spine_count))

        kern_lines.append("\t".join(["*-"] * spine_count))
        return "\n".join(kern_lines)

class Spine:
    """Represents a single vertical column in the Humdrum file."""
    
    def __init__(self, spine_type="kern", comments=None):
        self.tokens = []
        self.tokens.append(f"**{spine_type}")
        
        if comments:
            for comment in comments:
                self.tokens.append(comment)

    def add(self, token: str):
        self.tokens.append(token)

    def build(self) -> list:
        self.tokens.append("*-")
        return self.tokens

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
            
        # 2. Ask the graph for ONLY the accidentals explicitly linked to this keySig
        valid_accids = []
        for child_id, e_class in children_map.get(target_keysig, []):
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
            
            spine.add(f"*part{index}")
            spine.add(f"*staff{index}")
            
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
            
            spine.add(clef_found)
            spine.add(key_sig_found)
            spine.add(meter_sig_found)
            spine.add("1r")  
            
            spines.append(spine)
            
        return spines


class HumdrumContext:
    """Manages the collection of spines and handles the final text rendering."""
    
    def __init__(self):
        self.spines = []

    def add_spine(self, spine: Spine):
        self.spines.append(spine)

    def merge_spines(self) -> str:
        """Builds all spines and transposes them into tab-separated horizontal rows."""
        if not self.spines:
            return ""

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
    def __init__(self, node_list, edge_index, edge_predictions, node_roles, pyg_node_ids):
        self.nodes = {n['id']: n for n in node_list}
        
        self.children = {}
        for i in range(edge_index.shape[1]):
            u_str = pyg_node_ids[edge_index[0, i].item()]
            v_str = pyg_node_ids[edge_index[1, i].item()]
            e_class = edge_predictions[i].item()
            
            if e_class > 0 and u_str in self.nodes and v_str in self.nodes:
                if u_str not in self.children:
                    self.children[u_str] = []
                self.children[u_str].append((v_str, e_class))

    def export_to_krn(self):
        context = HumdrumContext()
        # 1. Locate the first system and its first measure
        systems = [n for n in self.nodes.values() if n['class'] == 'system']
        if not systems: return "Error: No systems found."
        
        systems.sort(key=lambda s: s['cy'])
        first_system = systems[0]

        measures = [v for v, e_class in self.children.get(first_system['id'], []) if e_class == 1]
        if not measures: return "Error: No measures found in the first system."
        first_measure = measures[0]

        # 2. Use the Spine class method to do the heavy lifting
        spines = Spine.create_from_measure(first_system['id'], first_measure, self.children, self.nodes)
        if not spines: return "Error: No staves found in the first measure."

        # 3. Add them to the context
        for spine in spines:
            context.add_spine(spine)

        for spine in spines:
            spine.add("4b")  # The simple rest placeholder

        # 4. Merge and export
        return context.merge_spines()