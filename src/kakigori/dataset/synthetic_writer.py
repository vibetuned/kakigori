# Standard library imports
import random
import logging
import argparse
import multiprocessing
from enum import Enum
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from fractions import Fraction
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
import copy

try:
    # Third party imports
    import music21
    from music21 import environment

    environment.UserSettings()["warnings"] = 0
except ImportError:
    pass

# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
# Classical 2-octave fingerings for major scales
# Fingers: 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
# RH = Right Hand (ascending), LH = Left Hand (ascending)
SCALE_FINGERINGS = {
    # White key scales - standard pattern
    "C": {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "G": {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "D": {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "A": {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "E": {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "B": {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [4, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1]},
    "F-": {"RH": [2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2], "LH": [4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4]}, # Gb
    "F": {"RH": [1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "B-": {"RH": [2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4], "LH": [3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3]}, # Bb
    "E-": {"RH": [3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3], "LH": [3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3]}, # Eb
    "A-": {"RH": [3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3], "LH": [3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3]}, # Ab
    "D-": {"RH": [2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2], "LH": [3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3]}, # Db
}

# Classical 2-octave fingerings for harmonic minor scales
HARMONIC_MINOR_FINGERINGS = {
    "A":  {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "E":  {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "B":  {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [4, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1]},
    "D":  {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "G":  {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "C":  {"RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "F":  {"RH": [1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4], "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]},
    "F#": {"RH": [2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2], "LH": [4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4]},
    "C#": {"RH": [3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3], "LH": [3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3]},
}

class CircleOfFifths:
    def __init__(self):
        self.majors = ['C', 'G', 'D', 'A', 'E', 'B', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F']
        self.minors = ['Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'Ebm', 'Bbm', 'Fm', 'Cm', 'Gm', 'Dm']
        self.diminisheds = ['C°', 'G°', 'D°', 'A°', 'E°', 'B°', 'Gb°', 'Db°', 'Ab°', 'Eb°', 'Bb°', 'F°']
    def get_neighbors(self, chord):
        """
        Auto-detects the chord type, finds its parent major key, 
        and returns the 6 diatonic neighbors.
        """
        # 1. Auto-detect chord and find its parent Major index
        if chord in self.diminisheds:
            dim_index = self.diminisheds.index(chord)
            parent_major_index = (dim_index - 5) % 12
        elif chord in self.minors:
            parent_major_index = self.minors.index(chord)
        elif chord in self.majors:
            parent_major_index = self.majors.index(chord)
        else:
            raise ValueError(f"Chord {chord} not recognized.")

        # 2. Gather all 7 diatonic chords for that key cluster
        left = (parent_major_index - 1) % 12
        right = (parent_major_index + 1) % 12
        dim_index = (parent_major_index + 5) % 12

        all_diatonic_chords = {
            'I': self.majors[parent_major_index],
            'IV': self.majors[left],
            'V': self.majors[right],
            'vi': self.minors[parent_major_index],
            'ii': self.minors[left],
            'iii': self.minors[right],
            'vii°': self.diminisheds[dim_index]
        }

        # 3. Filter out the current chord so we strictly return the 6 neighbors
        neighbors = {role: name for role, name in all_diatonic_chords.items() if name != chord}
        
        return neighbors

    def get_random_neighbor(self, chord):
        neighbors_dict = self.get_neighbors(chord)
        if not neighbors_dict:
            raise ValueError(f"Chord {chord} not recognized.")
            
        return random.choice(list(neighbors_dict.values()))

# =============================================================================
# Chord Types and Intervals
# =============================================================================

# Chord intervals in semitones from root
# Supports 3-note and 4-note voicings
CHORD_INTERVALS = {
    # Triads (3 notes)
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    # Seventh chords (4 notes)
    "maj7": [0, 4, 7, 11],
    "dom7": [0, 4, 7, 10],
    "min7": [0, 3, 7, 10],
    "dim7": [0, 3, 6, 9],
    "half_dim7": [0, 3, 6, 10],  # m7b5
    # Sixth chords (4 notes)
    "maj6": [0, 4, 7, 9],
    "min6": [0, 3, 7, 9],
    # Add chords (4 notes)
    "add9": [0, 4, 7, 14],  # Root + major triad + 9th
}

# Diatonic chords for major keys (I, ii, iii, IV, V, vi, vii°)
# Each entry: (scale_degree, chord_quality)
DIATONIC_CHORDS_DEGREES = [
    (0, "major"),  # I
    (2, "minor"),  # ii
    (4, "minor"),  # iii
    (5, "major"),  # IV
    (7, "major"),  # V
    (9, "minor"),  # vi
    (11, "dim"),  # vii°
]

# Roman numeral names for reference
ROMAN_NUMERALS = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]

# =============================================================================
# Advanced Music Generation (Native Music21)
# =============================================================================


class ArpeggioPattern(Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"
    BROKEN = "broken"
    ALBERTI = "alberti"


# User starting keys
STARTING_KEYS = ["C", "G", "D", "F", "Bb", "A", "E", "Eb", "Ab"]
MINOR_STARTING_KEYS = ["A", "E", "B", "D", "G", "C", "F", "F#", "C#"]
TIME_SIGNATURES = [
    "1/4",
    "2/4",
    "4/4",
    "6/8",
    "8/8",
    "C",
    "cut",
    "16/8",
    "16/16",
    "16/12",
    "9/8",
    "12/8",
]
FAILING_ARTICULATIONS = [
    music21.articulations.Staccato,
    music21.articulations.Accent,
    music21.articulations.Staccatissimo,
    music21.articulations.Tenuto,
    music21.articulations.StrongAccent,  # Marcato
]
FAILING_ORNAMENTS = [
    music21.expressions.Trill,
    music21.expressions.Mordent,
    music21.expressions.Turn,
]
FAILING_DYNAMICS = ["p", "pp", "z", "sfz", "fz", "mf", "f"]


def get_chord_pitches(
    key_str: str, degree: int, quality: str, inversion: int = 0, octave: int = 4
) -> List[music21.pitch.Pitch]:
    """Build a chord natively in music21 based on degree/quality, returning Pitches."""
    # We use music21 RomanNumeral for exact diatonic/secondary handling
    # Convert our (degree, quality) mapping into roman numeral strings
    # degree 0=I, 2=ii, 4=iii, 5=IV, 7=V, 9=vi, 11=vii°
    rn_str = ROMAN_NUMERALS[
        DIATONIC_CHORDS_DEGREES.index((degree, quality))
        if (degree, quality) in DIATONIC_CHORDS_DEGREES
        else 0
    ]

    if quality == "dom7":
        rn_str += "7"
    elif quality == "maj7":
        rn_str += "M7"
    elif quality == "min7":
        rn_str += "m7"
    elif quality == "sus4":
        rn_str = rn_str.replace("°", "").replace("m", "") + "sus4"

    try:
        rn = music21.roman.RomanNumeral(rn_str, key_str)
        # Apply standard inversion
        if inversion == 1:
            rn.inversion(1)
        elif inversion == 2:
            rn.inversion(2)
        elif inversion == 3:
            rn.inversion(3)

        # Shift all notes to the target octave
        pitches = list(rn.pitches)
        # RomanNumeral usually builds around obj.root() usually octave 4/3.
        # Shift them up/down to target the requested octave roughly.
        shift = octave - pitches[0].implicitOctave
        return [p.transpose(shift * 12) for p in pitches]
    except:
        # Fallback to a basic triad
        c = music21.chord.Chord(["C", "E", "G"])
        return [p.transpose((octave - 4) * 12) for p in c.pitches]


def calculate_voice_leading(
    prev_pitches: List[music21.pitch.Pitch], next_pitches: List[music21.pitch.Pitch]
) -> int:
    """Calculate absolute semitone differences between voices."""
    if not prev_pitches:
        return 0
    # Simple calculation: map voices 1-to-1 and sum differences
    dist = 0
    for p1, p2 in zip(prev_pitches, next_pitches):
        dist += abs(p1.midi - p2.midi)
    return dist


def choose_best_inversion(
    key_str: str,
    prev_pitches: List[music21.pitch.Pitch],
    next_deg: int,
    next_qual: str,
    octave: int,
) -> Tuple[int, List[music21.pitch.Pitch]]:
    if not prev_pitches:
        return 0, get_chord_pitches(key_str, next_deg, next_qual, 0, octave)

    best_inv = 0
    best_dist = float("inf")
    best_pitches = []

    # Try 0, 1, 2 inversions for triads (or 3 for 7ths)
    max_inv = 3 if "7" in next_qual else 2

    for inv in range(max_inv + 1):
        pitches = get_chord_pitches(key_str, next_deg, next_qual, inv, octave)
        dist = calculate_voice_leading(prev_pitches, pitches)
        if dist < best_dist:
            best_dist = dist
            best_inv = inv
            best_pitches = pitches

    return best_inv, best_pitches



def _get_fingering_key(key_obj: music21.key.Key, fingering_dict: dict = None) -> str:
    """Map a music21 Key to a fingering dictionary key.

    Handles enharmonic lookup (e.g. Bb -> B-, F# -> enharmonic Gb -> F-).
    Falls back to 'C' (or 'A' for minor dicts) if no match is found.
    """
    if fingering_dict is None:
        fingering_dict = SCALE_FINGERINGS
    tonic_name = key_obj.tonic.name
    if tonic_name in fingering_dict:
        return tonic_name
    enharmonic = key_obj.tonic.getEnharmonic().name
    if enharmonic in fingering_dict:
        return enharmonic
    # Fallback to first key in dict
    return next(iter(fingering_dict))


def _build_scale_pitches(key_obj: music21.key.Key, hand: str) -> List[music21.pitch.Pitch]:
    """Build a 2-octave ascending + descending scale for the given hand.

    RH (treble): starts at octave 4 (middle C region), goes up 2 octaves then back.
    LH (bass):   starts at octave 2, goes up 2 octaves then back.
    """
    scale = music21.scale.MajorScale(key_obj.tonic)
    start_octave = 4 if hand == "RH" else 2

    # Get ascending pitches for 2 octaves (15 notes: degree 1 to 15)
    ascending: List[music21.pitch.Pitch] = []
    for i in range(1, 16):  # 15 scale degrees = 2 full octaves
        p = scale.pitchFromDegree(i)
        # Assign correct octave
        octave_offset = (i - 1) // 7
        degree_in_octave = ((i - 1) % 7) + 1
        p.octave = start_octave + octave_offset
        # Fix: pitchFromDegree may give a pitch in wrong octave relative to tonic
        # Ensure monotonically ascending
        if ascending and p.midi <= ascending[-1].midi:
            p.octave += 1
        ascending.append(p)

    # Descending = reverse of ascending, skip the top note (already played)
    descending = list(reversed(ascending[:-1]))

    return ascending + descending


def _build_harmonic_minor_pitches(key_obj: music21.key.Key, hand: str) -> List[music21.pitch.Pitch]:
    """Build a 2-octave ascending + descending harmonic minor scale.

    Uses music21.scale.HarmonicMinorScale (raised 7th degree).
    """
    scale = music21.scale.HarmonicMinorScale(key_obj.tonic)
    start_octave = 4 if hand == "RH" else 2

    ascending: List[music21.pitch.Pitch] = []
    for i in range(1, 16):
        p = scale.pitchFromDegree(i)
        octave_offset = (i - 1) // 7
        p.octave = start_octave + octave_offset
        if ascending and p.midi <= ascending[-1].midi:
            p.octave += 1
        ascending.append(p)

    descending = list(reversed(ascending[:-1]))
    return ascending + descending


def _build_melodic_minor_pitches(key_obj: music21.key.Key, hand: str) -> List[music21.pitch.Pitch]:
    """Build a 2-octave melodic minor scale.

    Ascending: raised 6th and 7th (MelodicMinorScale).
    Descending: natural minor (MinorScale) — the classical convention.
    """
    asc_scale = music21.scale.MelodicMinorScale(key_obj.tonic)
    desc_scale = music21.scale.MinorScale(key_obj.tonic)  # natural minor
    start_octave = 4 if hand == "RH" else 2

    # Ascending with raised 6+7
    ascending: List[music21.pitch.Pitch] = []
    for i in range(1, 16):
        p = asc_scale.pitchFromDegree(i)
        octave_offset = (i - 1) // 7
        p.octave = start_octave + octave_offset
        if ascending and p.midi <= ascending[-1].midi:
            p.octave += 1
        ascending.append(p)

    # Descending with natural 6+7
    desc_full: List[music21.pitch.Pitch] = []
    for i in range(1, 16):
        p = desc_scale.pitchFromDegree(i)
        octave_offset = (i - 1) // 7
        p.octave = start_octave + octave_offset
        if desc_full and p.midi <= desc_full[-1].midi:
            p.octave += 1
        desc_full.append(p)

    descending = list(reversed(desc_full[:-1]))
    return ascending + descending


def _build_chromatic_pitches(key_obj: music21.key.Key, hand: str) -> List[music21.pitch.Pitch]:
    """Build a 2-octave ascending + descending chromatic scale.

    25 notes ascending (24 semitones), then 24 notes descending.
    """
    start_octave = 4 if hand == "RH" else 2
    start_pitch = music21.pitch.Pitch(key_obj.tonic.name)
    start_pitch.octave = start_octave

    ascending: List[music21.pitch.Pitch] = []
    for i in range(25):  # 2 octaves = 24 semitones + start
        p = start_pitch.transpose(i)
        ascending.append(p)

    descending = list(reversed(ascending[:-1]))
    return ascending + descending


def _get_fingerings_for_hand(fing_key: str, hand: str, fingering_dict: dict = None) -> List[int]:
    """Return the full ascending + descending fingering pattern for a hand.

    The dict stores 15 ascending fingers. Descending is the reverse
    (skip the top note since it was already played ascending).
    """
    if fingering_dict is None:
        fingering_dict = SCALE_FINGERINGS
    asc = fingering_dict[fing_key][hand]
    desc = list(reversed(asc[:-1]))
    return asc + desc


def _get_chromatic_fingerings(pitches: List[music21.pitch.Pitch], hand: str) -> List[int]:
    """Compute chromatic scale fingerings based on pitch class.

    Standard rule: thumb (1) on C and F (pitch classes 0 and 5),
    finger 3 on all other notes.
    """
    return [1 if p.pitchClass in (0, 5) else 3 for p in pitches]


def _choose_note_duration(bar_length: float) -> float:
    """Pick a random note duration that fits within a single bar."""
    candidates = [4.0, 2.0, 1.0, 0.5, 0.25]
    weights = [0.10, 0.15, 0.35, 0.25, 0.15]
    # Filter out durations that exceed the bar length
    valid = [(d, w) for d, w in zip(candidates, weights) if d <= bar_length]
    if not valid:
        return bar_length  # fallback: one note fills the whole bar
    durations, wts = zip(*valid)
    return random.choices(durations, weights=wts, k=1)[0]


def _inject_ornaments_on_note(note_obj: music21.note.Note, measure_idx: int):
    """Randomly attach failing-class ornaments, articulations, and dynamics."""
    # Ornaments (trill, mordent, turn) — keep rate moderate for scales
    if random.random() < 0.06:
        orn = random.choice(FAILING_ORNAMENTS)
        note_obj.expressions.append(orn())

    # Articulations (staccato, accent, etc.)
    if random.random() < 0.12:
        art = random.choice(FAILING_ARTICULATIONS)
        note_obj.articulations.append(art())


def _inject_dynamics(measure: music21.stream.Measure, offset: float):
    """Inject a dynamic marking at the given offset."""
    dyn_symbol = random.choice(FAILING_DYNAMICS)
    dyn = music21.dynamics.Dynamic(dyn_symbol)
    measure.insert(offset, dyn)


def _maybe_add_tie(
    prev_note: Optional[music21.note.Note],
    curr_note: music21.note.Note,
) -> bool:
    """Randomly tie two adjacent notes of the same pitch. Returns True if tied."""
    if prev_note is None:
        return False
    if prev_note.pitch.midi != curr_note.pitch.midi:
        return False
    if random.random() < 0.15:
        prev_note.tie = music21.tie.Tie("start")
        curr_note.tie = music21.tie.Tie("stop")
        return True
    return False


def _build_measures_from_pitches(
    pitches: List[music21.pitch.Pitch],
    fingerings: List[int],
    time_sig_str: str,
    key_obj: music21.key.Key,
    hand: str,
) -> List[music21.stream.Measure]:
    """Pack scale pitches into measures with fingering, ornaments, and dynamics."""
    # Parse time signature to know beats per measure
    ts = music21.meter.TimeSignature(time_sig_str)
    bar_length = ts.barDuration.quarterLength

    measures: List[music21.stream.Measure] = []
    note_duration = _choose_note_duration(bar_length)

    current_measure = music21.stream.Measure(number=1)
    current_measure.timeSignature = copy.deepcopy(ts)
    current_measure.keySignature = music21.key.KeySignature(key_obj.sharps)
    # Add clef
    if hand == "RH":
        current_measure.clef = music21.clef.TrebleClef()
    else:
        current_measure.clef = music21.clef.BassClef()

    current_offset = 0.0
    measure_count = 1
    prev_note = None
    first_note_in_measure = True

    for i, (pitch, finger) in enumerate(zip(pitches, fingerings)):
        n = music21.note.Note(pitch, quarterLength=note_duration)
        n.articulations.append(music21.articulations.Fingering(finger))

        # Ornaments
        _inject_ornaments_on_note(n, measure_count)

        # Ties (at turning point of scale where same pitch repeats)
        _maybe_add_tie(prev_note, n)

        # Dynamics at start of some measures
        if first_note_in_measure and random.random() < 0.20:
            _inject_dynamics(current_measure, current_offset)
            first_note_in_measure = False

        # Check if note fits in current measure
        if current_offset + note_duration > bar_length + 0.001:
            # Finalize current measure, start new one
            measures.append(current_measure)
            measure_count += 1
            current_measure = music21.stream.Measure(number=measure_count)
            current_offset = 0.0
            first_note_in_measure = True

            if first_note_in_measure and random.random() < 0.20:
                _inject_dynamics(current_measure, 0.0)
                first_note_in_measure = False

        current_measure.insert(current_offset, n)
        current_offset += note_duration
        prev_note = n

    # Append the last measure (pad with rest if needed)
    remaining = bar_length - current_offset
    if remaining > 0.001:
        rest = music21.note.Rest(quarterLength=remaining)
        current_measure.insert(current_offset, rest)
    measures.append(current_measure)

    return measures


def _add_slurs(part: music21.stream.Part, density: float = 0.15):
    """Add slur spanners over random groups of adjacent notes."""
    all_notes = [n for n in part.recurse().notes if not n.isRest]
    if len(all_notes) < 4:
        return

    i = 0
    while i < len(all_notes) - 2:
        if random.random() < density:
            span_len = random.randint(2, min(6, len(all_notes) - i))
            slur = music21.spanner.Slur()
            slur.addSpannedElements(all_notes[i], all_notes[i + span_len - 1])
            part.insert(0, slur)
            i += span_len  # skip past the slurred notes
        else:
            i += 1


# =============================================================================
# Chord Progression Generator (Circle of Fifths)
# =============================================================================

COF = CircleOfFifths()


def _chord_name_to_root_and_quality(chord_name: str) -> Tuple[str, str]:
    """Parse a CircleOfFifths chord name into (root, quality).

    Examples: 'C' -> ('C', 'major'), 'Am' -> ('A', 'minor'),
              'G°' -> ('G', 'dim'), 'F#m' -> ('F#', 'minor')
    """
    if chord_name.endswith('°'):
        return chord_name[:-1], 'dim'
    elif chord_name.endswith('m'):
        return chord_name[:-1], 'minor'
    else:
        return chord_name, 'major'


def _build_chord_voicing(
    root_name: str, quality: str, octave: int
) -> music21.chord.Chord:
    """Build a music21 Chord from root name + quality at given octave."""
    intervals = CHORD_INTERVALS.get(quality, [0, 4, 7])  # fallback to major triad
    root_pitch = music21.pitch.Pitch(root_name)
    root_pitch.octave = octave

    pitches = [root_pitch.transpose(iv) for iv in intervals]
    return music21.chord.Chord(pitches)


def _chord_fingering(chord_obj: music21.chord.Chord) -> List[int]:
    """Assign fingering to a chord based on root pitch class.

    Black-key root (pitch class in {1,3,6,8,10}) -> [2,3,5]
    White-key root -> random choice of [1,3,5] or [1,2,4]
    For 4-note chords, extend with finger 5 or 4.
    """
    root_pc = chord_obj.pitches[0].pitchClass
    n_notes = len(chord_obj.pitches)

    if root_pc in (1, 3, 6, 8, 10):  # black key
        base = [2, 3, 5]
    else:
        base = random.choice([[1, 3, 5], [1, 2, 4]])

    # Extend for 4+ note chords
    while len(base) < n_notes:
        base.append(min(base[-1] + 1, 5))
    return base[:n_notes]


def _build_chord_progression(key_str: str, num_chords: int) -> List[str]:
    """Generate a chord progression walking the circle of fifths.

    Starts from the given major key's I chord and navigates through
    diatonic neighbors, occasionally repeating chords for ties.
    """
    current = key_str  # start on the I chord (major)
    progression: List[str] = [current]

    for _ in range(num_chords - 1):
        # Sometimes repeat the current chord (for ties)
        if random.random() < 0.20:
            progression.append(current)
        else:
            try:
                current = COF.get_random_neighbor(current)
            except ValueError:
                current = key_str  # fallback to tonic
            progression.append(current)

    return progression


def _build_measures_from_chords(
    progression: List[str],
    time_sig_str: str,
    key_obj: music21.key.Key,
    hand: str,
) -> List[music21.stream.Measure]:
    """Pack a chord progression into measures with fingering, arpeggios, and ties."""
    ts = music21.meter.TimeSignature(time_sig_str)
    bar_length = ts.barDuration.quarterLength
    chord_duration = _choose_note_duration(bar_length)
    octave = 4 if hand == "RH" else 3

    measures: List[music21.stream.Measure] = []
    current_measure = music21.stream.Measure(number=1)
    current_measure.timeSignature = copy.deepcopy(ts)
    current_measure.keySignature = music21.key.KeySignature(key_obj.sharps)
    current_measure.clef = (
        music21.clef.TrebleClef() if hand == "RH" else music21.clef.BassClef()
    )

    current_offset = 0.0
    measure_count = 1
    prev_chord = None
    first_in_measure = True

    for chord_name in progression:
        root, quality = _chord_name_to_root_and_quality(chord_name)
        c = _build_chord_voicing(root, quality, octave)
        c.quarterLength = chord_duration

        # Fingering on each note of the chord
        fingers = _chord_fingering(c)
        for pitch_obj, finger in zip(c.pitches, fingers):
            c.articulations.append(music21.articulations.Fingering(finger))

        # Arpeggio mark (~25%)
        if random.random() < 0.25:
            c.expressions.append(music21.expressions.ArpeggioMark())

        # Ornaments / articulations on the chord
        if random.random() < 0.08:
            art = random.choice(FAILING_ARTICULATIONS)
            c.articulations.append(art())

        # New measure if needed — BEFORE tie logic to prevent cross-barline ties
        if current_offset + chord_duration > bar_length + 0.001:
            measures.append(current_measure)
            measure_count += 1
            current_measure = music21.stream.Measure(number=measure_count)
            current_offset = 0.0
            first_in_measure = True
            prev_chord = None  # prevent ties across barlines
            if first_in_measure and random.random() < 0.20:
                _inject_dynamics(current_measure, 0.0)
                first_in_measure = False

        # Tie some repeated chords (~30%) — only within same measure
        if prev_chord is not None and chord_name == getattr(prev_chord, '_cof_name', None):
            if random.random() < 0.30:
                prev_chord.tie = music21.tie.Tie('start')
                c.tie = music21.tie.Tie('stop')

        # Dynamics
        if first_in_measure and random.random() < 0.20:
            _inject_dynamics(current_measure, current_offset)
            first_in_measure = False

        current_measure.insert(current_offset, c)
        current_offset += chord_duration
        c._cof_name = chord_name  # tag for tie detection
        prev_chord = c

    # Pad last measure
    remaining = bar_length - current_offset
    if remaining > 0.001:
        current_measure.insert(current_offset, music21.note.Rest(quarterLength=remaining))
    measures.append(current_measure)

    return measures


def _equalize_measures(
    rh_measures: List[music21.stream.Measure],
    lh_measures: List[music21.stream.Measure],
) -> Tuple[List[music21.stream.Measure], List[music21.stream.Measure]]:
    """Repeat each hand's measures so both have the same total length.

    Uses LCM: if RH has 3 measures and LH has 2, RH is repeated 2×
    (6 total) and LH is repeated 3× (6 total).
    """
    import math

    n_rh, n_lh = len(rh_measures), len(lh_measures)
    if n_rh == n_lh:
        return rh_measures, lh_measures

    target = math.lcm(n_rh, n_lh)
    # Cap to avoid unreasonably long scores from coprime counts
    MAX_MEASURES = 64
    if target > MAX_MEASURES:
        target = max(n_rh, n_lh)

    rh_reps = max(1, target // n_rh)
    lh_reps = max(1, target // n_lh)

    def _repeat(measures: List[music21.stream.Measure], target_len: int):
        result = list(measures)
        while len(result) < target_len:
            for m in measures:
                if len(result) >= target_len:
                    break
                new_m = copy.deepcopy(m)
                new_m.number = len(result) + 1
                result.append(new_m)
        return result

    final_target = max(n_rh * rh_reps, n_lh * lh_reps)
    return _repeat(rh_measures, final_target), _repeat(lh_measures, final_target)


def generate_score(min_measures, max_measures):
    """Generate a piano score with scale exercises or chord progressions.

    Randomly picks one of: major, harmonic_minor, melodic_minor, chromatic,
    or chord_progression. Creates a 2-staff piano layout (Treble RH + Bass LH).
    Both parts are equalized to the same number of measures via LCM repetition.
    """
    score = music21.stream.Score()

    scale_type = random.choice([
        "major", "harmonic_minor", "melodic_minor", "chromatic",
        "chord_progression",
    ])

    # Pick key based on scale type
    if scale_type in ("harmonic_minor", "melodic_minor"):
        key_str = random.choice(MINOR_STARTING_KEYS)
        key_obj = music21.key.Key(key_str, "minor")
    else:
        key_str = random.choice(STARTING_KEYS)
        key_obj = music21.key.Key(key_str)

    time_sig_str = random.choice(TIME_SIGNATURES)

    # Build measures for both hands first
    hands_measures: Dict[str, List[music21.stream.Measure]] = {}

    if scale_type == "chord_progression":
        num_chords = random.randint(8, 16)
        progression = _build_chord_progression(key_str, num_chords)
        for hand in ["RH", "LH"]:
            hands_measures[hand] = _build_measures_from_chords(
                progression, time_sig_str, key_obj, hand
            )
    else:
        for hand in ["RH", "LH"]:
            if scale_type == "major":
                pitches = _build_scale_pitches(key_obj, hand)
                fing_key = _get_fingering_key(key_obj)
                fingerings = _get_fingerings_for_hand(fing_key, hand)
            elif scale_type == "harmonic_minor":
                pitches = _build_harmonic_minor_pitches(key_obj, hand)
                fing_key = _get_fingering_key(key_obj, HARMONIC_MINOR_FINGERINGS)
                fingerings = _get_fingerings_for_hand(fing_key, hand, HARMONIC_MINOR_FINGERINGS)
            elif scale_type == "melodic_minor":
                pitches = _build_melodic_minor_pitches(key_obj, hand)
                fing_key = _get_fingering_key(key_obj, HARMONIC_MINOR_FINGERINGS)
                fingerings = _get_fingerings_for_hand(fing_key, hand, HARMONIC_MINOR_FINGERINGS)
            else:  # chromatic
                pitches = _build_chromatic_pitches(key_obj, hand)
                fingerings = _get_chromatic_fingerings(pitches, hand)

            hands_measures[hand] = _build_measures_from_pitches(
                pitches, fingerings, time_sig_str, key_obj, hand
            )

    # Equalize lengths by repeating shorter part
    hands_measures["RH"], hands_measures["LH"] = _equalize_measures(
        hands_measures["RH"], hands_measures["LH"]
    )

    # Assemble parts
    for hand in ["RH", "LH"]:
        part = music21.stream.Part()
        inst = music21.instrument.Piano()
        part.insert(0, inst)

        for m in hands_measures[hand]:
            part.append(m)

        _add_slurs(part)

        try:
            part = part.makeBeams(inPlace=False)
        except Exception:
            pass

        score.insert(0, part)

    return score


def process_file(args_tuple) -> bool:
    file_idx, output_dir, min_m, max_m = args_tuple
    try:
        out_path = output_dir / f"synthetic_score_{file_idx:06d}.mxl"
        if out_path.exists():
            return True
        
        score = generate_score(min_m, max_m)
        # Collect intentional tie pairs BEFORE makeNotation mangles them.
        tie_pairs: list = []  # [(start_id, stop_id), ...]
        all_elements = list(score.recurse())
        for i, el in enumerate(all_elements):
            if hasattr(el, 'tie') and el.tie is not None and el.tie.type == 'start':
                # Find the matching 'stop' after it
                for j in range(i + 1, len(all_elements)):
                    el2 = all_elements[j]
                    if hasattr(el2, 'tie') and el2.tie is not None and el2.tie.type == 'stop':
                        tie_pairs.append((id(el), id(el2)))
                        break

        # 1. Run makeNotation to handle complex durations & beaming.
        score.makeNotation(inPlace=True)
        # 2. Split any remaining complex durations at the measure level.
        for measure in score.recurse().getElementsByClass('Measure'):
            measure.splitAtDurations()
        # 3. Strip ALL ties unconditionally — including notes inside chords.
        for el in score.recurse():
            if hasattr(el, 'tie') and el.tie is not None:
                el.tie = None
            # Also clear ties on individual notes inside Chord objects
            if isinstance(el, music21.chord.Chord):
                for n in el.notes:
                    if n.tie is not None:
                        n.tie = None
        # 4. Re-apply only our intentional tie pairs.
        id_to_el = {id(el): el for el in score.recurse()}
        for start_id, stop_id in tie_pairs:
            if start_id in id_to_el and stop_id in id_to_el:
                start_el = id_to_el[start_id]
                stop_el = id_to_el[stop_id]
                start_el.tie = music21.tie.Tie('start')
                stop_el.tie = music21.tie.Tie('stop')
                # Also set on inner notes for Chord objects
                if isinstance(start_el, music21.chord.Chord):
                    for n in start_el.notes:
                        n.tie = music21.tie.Tie('start')
                if isinstance(stop_el, music21.chord.Chord):
                    for n in stop_el.notes:
                        n.tie = music21.tie.Tie('stop')
        # 5. Write without a second makeNotation pass.
        score.write('musicxml', fp=out_path, makeNotation=False)
        return True
    except Exception as e:
        logger.error(f"Failed to generate score {file_idx}: {e}", exc_info=True)
        return False


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Generate synthetic dataset prioritizing failing classes.")
    parser.add_argument("output_dir", type=str, help="Output directory for generated MXL files")
    parser.add_argument("num_files", type=int, help="Number of files to generate")
    parser.add_argument("--min_measures", type=int, default=12, help="Min measures per file")
    parser.add_argument("--max_measures", type=int, default=32, help="Max measures per file")
    
    args = parser.parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {args.num_files} synthetic files in {output_path}...")
    
    success_count = 0
    error_count = 0
    
    tasks = [(i, output_path, args.min_measures, args.max_measures) for i in range(args.num_files)]

    with multiprocessing.Pool(
        processes=max(1, multiprocessing.cpu_count() - 1), maxtasksperchild=10
    ) as pool:
        results = pool.imap_unordered(process_file, tasks)
        with tqdm(total=args.num_files, desc="Generating files") as pbar:
            for is_success in results:
                if is_success:
                    success_count += 1
                else:
                    error_count += 1
                pbar.update(1)

    logger.info(f"Finished. Generated {success_count} files (Failed: {error_count}).")


if __name__ == "__main__":
    main()
