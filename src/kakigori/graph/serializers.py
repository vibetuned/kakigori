import logging
import pprint
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)



class Spine:
    """Represents a single vertical column (spine) in a Humdrum file."""

    def __init__(self, spine_type="kern"):
        self.exclusive = f"**{spine_type}"
        self.head = []

    def add_to_head(self, token: str):
        """Append a tandem interpretation to the spine header."""
        self.head.append(token)

    def build(self) -> list:
        """Assemble the complete token list for this spine."""
        tokens = [self.exclusive]
        tokens.extend(self.head)
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

            # Future steps: iterate over measures and add notes to spines

    def export_to_krn(self) -> str:
        """Export the accumulated data as a single Humdrum **kern string."""
        if not self._head_initialized:
            return "Error: No valid systems found across any page."
        return self.context.merge_spines()