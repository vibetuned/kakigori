class ContextTracker:
    def __init__(self):
        # Global State
        self.active_clef = "treble"
        self.key_signature = {}  # e.g., {'f': '#', 'c': '#'}

        # Local State (Resets every measure)
        self.measure_accidentals = {}  # e.g., {'g': '#'}

        # Spatial Anchors
        self.staff_y_top = 0.0
        self.staff_y_bottom = 0.0

    def process_context_node(self, node, super_node=None, calculated_pitch=None):
        """
        Updates the running state based on the node class.
        """
        node_class = node["class"]

        # 1. Handle Structural Anchors
        if node_class.startswith("barLine"):
            self._handle_barline(node)

        elif node_class.startswith("clef"):
            self._handle_clef(node)

        # 2. Handle Global Modifiers
        elif node_class.startswith("keySig"):
            self._handle_key_signature(node_class)

        # 3. Handle Local Modifiers (Inline Accidentals)
        elif super_node and calculated_pitch:
            modifiers = [m["class"] for m in super_node["modifiers"]]
            for mod in modifiers:
                if mod.startswith("accid"):
                    self._handle_inline_accidental(mod, calculated_pitch)

    def _handle_barline(self, barline_node):
        """Resets local measure state and recalibrates vertical spatial anchors."""
        self.measure_accidentals.clear()

        _, y1, _, y2 = barline_node["bbox"]
        self.staff_y_top = y1
        self.staff_y_bottom = y2

    def _handle_clef(self, clef_node):
        """Updates the active clef and acts as a spatial anchor."""
        clef_identity = clef_node["class"].split("-")[-1]  # e.g., 'G', 'F', 'C'

        clef_map = {
            "G": "treble",
            "F": "bass",
            "C": "alto",  # Or tenor, depending on Y-position, but 'alto' is a safe default
        }
        self.active_clef = clef_map.get(clef_identity, "treble")

        _, y1, _, y2 = clef_node["bbox"]
        self.staff_y_top = y1
        self.staff_y_bottom = y2

    def _handle_key_signature(self, key_class):
        """
        Translates key signatures (e.g., 'keySig-3s' or 'keySig-2f')
        into a dictionary of active accidentals based on the Circle of Fifths.
        """
        self.key_signature.clear()

        if "-" not in key_class:
            return  # C Major / A Minor

        identity = key_class.split("-")[-1]
        count = int(identity[:-1])  # Extract the number (e.g., '3' from '3s')
        accid_type = identity[-1]  # Extract the type ('s' or 'f')

        # Order of sharps and flats
        sharps_order = ["f", "c", "g", "d", "a", "e", "b"]
        flats_order = ["b", "e", "a", "d", "g", "c", "f"]

        if accid_type == "s":
            for i in range(count):
                self.key_signature[sharps_order[i]] = "#"
        elif accid_type == "f":
            for i in range(count):
                self.key_signature[flats_order[i]] = "-"

    def _handle_inline_accidental(self, accid_class, pitch_string):
        """
        Registers an inline accidental into the local measure state.
        In Humdrum (**kern), pitch strings encode the octave (e.g., 'cc', 'C').
        Strictly speaking, a measure accidental only applies to that specific octave.
        """
        identity = accid_class.split("-")[-1]

        accid_map = {
            "sharp": "#",
            "flat": "-",
            "nat": "n",
            "dsharp": "##",
            "dflat": "--",
        }

        symbol = accid_map.get(identity, "")

        # Strip existing non-alphabetic characters (like duration numbers or old accidentals)
        # to isolate the pure pitch string (e.g., '4cc#' -> 'cc')
        pure_pitch = "".join(filter(str.isalpha, pitch_string))

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
            return "" if mod == "n" else mod

        # 2. Check global key signature (pitch class match, ignoring octave)
        pitch_class = pure_pitch[0].lower()  # 'cc' -> 'c', 'C' -> 'c'
        return self.key_signature.get(pitch_class, "")
