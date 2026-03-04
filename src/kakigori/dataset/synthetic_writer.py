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

# Third party imports
import numpy as np
from tqdm import tqdm

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
# Major scale intervals in semitones: W-W-H-W-W-W-H
# But in our grid, columns are white keys (C, D, E, F, G, A, B)
# and row 1 is for accidentals (black keys)
#
# For our 2x52 grid representation:
# - Column index = white key index (0=C, 1=D, 2=E, 3=F, 4=G, 5=A, 6=B per octave)
# - Row 0 = natural (white key), Row 1 = accidental (black key on that column)
#
# Major scales with their patterns:
# Each tuple is (column_offset_from_root, is_black)
# Root is always (0, 0) for white key roots

MAJOR_SCALES = {
    # C Major: C D E F G A B (all white)
    "C": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],
    # G Major: G A B C D E F# (F# is black on column 3 of next position)
    "G": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 1)],  # F# = col 6, black
    # D Major: D E F# G A B C#
    "D": [(0, 0), (1, 0), (2, 1), (3, 0), (4, 0), (5, 0), (6, 1)],  # F#, C#
    # A Major: A B C# D E F# G#
    "A": [(0, 0), (1, 0), (2, 1), (3, 0), (4, 0), (5, 1), (6, 1)],  # C#, F#, G#
    # E Major: E F# G# A B C# D#
    "E": [(0, 0), (1, 1), (2, 1), (3, 0), (4, 0), (5, 1), (6, 1)],  # F#, G#, C#, D#
    # B Major: B C# D# E F# G# A#
    "B": [(0, 0), (1, 1), (2, 1), (3, 0), (4, 1), (5, 1), (6, 1)],  # C#, D#, F#, G#, A#
    # F Major: F G A Bb C D E (Bb is black on column 6 of prev octave position)
    "F": [(0, 0), (1, 0), (2, 0), (3, 1), (4, 0), (5, 0), (6, 0)],  # Bb
    # Bb Major: Bb C D Eb F G A
    "Bb": [(0, 1), (1, 0), (2, 0), (3, 1), (4, 0), (5, 0), (6, 0)],  # Bb, Eb
    # Eb Major: Eb F G Ab Bb C D
    "Eb": [(0, 1), (1, 0), (2, 0), (3, 1), (4, 1), (5, 0), (6, 0)],  # Eb, Ab, Bb
    # Ab Major: Ab Bb C Db Eb F G
    "Ab": [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1), (5, 0), (6, 0)],  # Ab, Bb, Db, Eb
    # Db Major: Db Eb F Gb Ab Bb C
    "Db": [
        (0, 1),
        (1, 1),
        (2, 0),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 0),
    ],  # Db, Eb, Gb, Ab, Bb
    # Gb Major: Gb Ab Bb Cb Db Eb F
    "Gb": [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 0),
    ],  # Gb, Ab, Bb, Db, Eb, Cb
}

# Root positions for each scale (which white key column is the root)
# In our grid: 0=C, 1=D, 2=E, 3=F, 4=G, 5=A, 6=B (repeating every 7 columns)
SCALE_ROOTS = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
    "Bb": 6,
    "Eb": 2,
    "Ab": 5,
    "Db": 1,
    "Gb": 4,
}

# Classical 2-octave fingerings for major scales
# Fingers: 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
# RH = Right Hand (ascending), LH = Left Hand (ascending)
# These are the standard fingerings taught in classical piano
SCALE_FINGERINGS = {
    # White key scales - standard pattern
    "C": {
        "RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5],
        "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1],
    },
    "G": {
        "RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5],
        "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1],
    },
    "D": {
        "RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5],
        "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1],
    },
    "A": {
        "RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5],
        "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1],
    },
    "E": {
        "RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5],
        "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1],
    },
    # B major - different LH pattern
    "B": {
        "RH": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5],
        "LH": [4, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1],
    },
    # F# / Gb - starts on black key
    "Gb": {
        "RH": [2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2],
        "LH": [4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4],
    },
    # F major - thumb crosses to 4th
    "F": {
        "RH": [1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4],
        "LH": [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1],
    },
    # Bb major - starts with 2
    "Bb": {
        "RH": [2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4],
        "LH": [3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3],
    },
    # Eb major - starts with 3
    "Eb": {
        "RH": [3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3],
        "LH": [3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3],
    },
    # Ab major - starts with 3-4
    "Ab": {
        "RH": [3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3],
        "LH": [3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3],
    },
    # Db major - starts with 2-3
    "Db": {
        "RH": [2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2],
        "LH": [3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3],
    },
}

# =============================================================================
# Circle of Fifths
# =============================================================================

# Circle of fifths with sharp (clockwise) and flat (counter-clockwise) neighbors
# Also includes relative minor for each major key
CIRCLE_OF_FIFTHS = {
    "C": {"sharp": "G", "flat": "F", "relative_minor": "Am"},
    "G": {"sharp": "D", "flat": "C", "relative_minor": "Em"},
    "D": {"sharp": "A", "flat": "G", "relative_minor": "Bm"},
    "A": {"sharp": "E", "flat": "D", "relative_minor": "F#m"},
    "E": {"sharp": "B", "flat": "A", "relative_minor": "C#m"},
    "B": {"sharp": "F#", "flat": "E", "relative_minor": "G#m"},
    "F#": {"sharp": "C#", "flat": "B", "relative_minor": "D#m"},
    "C#": {"sharp": "G#", "flat": "F#", "relative_minor": "A#m"},
    "F": {"sharp": "C", "flat": "Bb", "relative_minor": "Dm"},
    "Bb": {"sharp": "F", "flat": "Eb", "relative_minor": "Gm"},
    "Eb": {"sharp": "Bb", "flat": "Ab", "relative_minor": "Cm"},
    "Ab": {"sharp": "Eb", "flat": "Db", "relative_minor": "Fm"},
    "Db": {"sharp": "Ab", "flat": "Gb", "relative_minor": "Bbm"},
    "Gb": {"sharp": "Db", "flat": "Cb", "relative_minor": "Ebm"},
}

# Chromatic note to semitone offset from C
NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
    "B#": 0,
}

# Semitone to (white_key_column_in_octave, is_black)
# Column layout: C=0, D=1, E=2, F=3, G=4, A=5, B=6
SEMITONE_TO_GRID = {
    0: (0, 0),  # C
    1: (0, 1),  # C#/Db
    2: (1, 0),  # D
    3: (1, 1),  # D#/Eb
    4: (2, 0),  # E
    5: (3, 0),  # F
    6: (3, 1),  # F#/Gb
    7: (4, 0),  # G
    8: (4, 1),  # G#/Ab
    9: (5, 0),  # A
    10: (5, 1),  # A#/Bb
    11: (6, 0),  # B
}

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
# Advanced Music Generation
# =============================================================================


class ArpeggioPattern(Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"
    BROKEN = "broken"
    ALBERTI = "alberti"


# User starting keys
STARTING_KEYS = ["C", "G", "D", "F", "Bb", "A", "E", "Eb", "Ab"]
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
    music21.articulations.Tenuto,
    music21.articulations.Accent,
    music21.articulations.Staccatissimo,
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


class MusicGenerator:
    """Generates a complete musical Part containing block chords or arpeggios."""

    def __init__(
        self,
        key_str: str,
        total_bars: int,
        beats_per_bar: int,
        is_bass: bool = False,
        gen_type: str = "block",
    ):
        self.key_str = key_str
        self.total_bars = total_bars
        self.beats_per_bar = beats_per_bar
        self.is_bass = is_bass
        self.gen_type = gen_type  # 'block', 'arpeggio', 'mixed'

        self.target_octave = 2 if is_bass else 4
        self.diatonic = DIATONIC_CHORDS_DEGREES

    def generate_part(self) -> music21.stream.Part:
        part = music21.stream.Part()

        # Decide initial and changing clefs strictly based on register
        if self.is_bass:
            current_clef = music21.clef.BassClef()
            clef_pool = [
                music21.clef.BassClef,
                music21.clef.BassClef,
                music21.clef.AltoClef,
                music21.clef.TenorClef,
            ]
        else:
            current_clef = music21.clef.TrebleClef()
            clef_pool = [
                music21.clef.TrebleClef,
                music21.clef.TrebleClef,
                music21.clef.AltoClef,
                music21.clef.Treble8vbClef,
            ]

        # Rules state chord every 2 beats
        beat = 0
        total_beats = self.total_bars * self.beats_per_bar
        prev_pitches = []

        m_idx = 1
        m = music21.stream.Measure(number=m_idx)
        m.append(current_clef)
        m.append(music21.key.Key(self.key_str))
        m.append(music21.meter.TimeSignature(f"{self.beats_per_bar}/4"))

        current_bar_beats = 0

        while beat < total_beats:
            chord_duration = 2

            # Select chord
            # Rule 3: Mix progressions, bias to I, IV, V
            deg_idx = random.choices(
                range(7), weights=[0.25, 0.12, 0.10, 0.18, 0.20, 0.12, 0.03], k=1
            )[0]

            if beat == 0 or beat >= total_beats - 2:
                deg_idx = 0  # Force I at start and end

            degree, quality = self.diatonic[deg_idx]

            # Rule 5: 7th on V
            if deg_idx == 4 and random.random() < 0.5:
                quality = "dom7"

            # Rule 6: Sus4 resolution
            if random.random() < 0.1 and quality == "major":
                # Inject sus4 then resolve
                sus_inv, sus_pitches = choose_best_inversion(
                    self.key_str, prev_pitches, degree, "sus4", self.target_octave
                )
                self._add_to_measure(m, sus_pitches, 1.0, self.gen_type)
                prev_pitches = sus_pitches
                current_bar_beats += 1
                beat += 1
                chord_duration = 1

            # Fetch voice led pitches
            inv, next_pitches = choose_best_inversion(
                self.key_str, prev_pitches, degree, quality, self.target_octave
            )

            # Add to measure
            self._add_to_measure(m, next_pitches, float(chord_duration), self.gen_type)
            prev_pitches = next_pitches

            current_bar_beats += chord_duration
            beat += chord_duration

            # Measure boundary handling
            while current_bar_beats >= self.beats_per_bar:
                part.append(m)
                m_idx += 1
                m = music21.stream.Measure(number=m_idx)
                current_bar_beats -= self.beats_per_bar

                # Active clef changes based on register and random injection
                if current_bar_beats == 0 and m_idx <= self.total_bars:
                    if random.random() < 0.15:
                        c_type = random.choice(clef_pool)
                        if not isinstance(current_clef, c_type):
                            current_clef = c_type()
                            m.append(current_clef)
                            # Adjust target octave to match the new clef naturally
                            if c_type == music21.clef.TrebleClef:
                                self.target_octave = 4
                            elif c_type == music21.clef.BassClef:
                                self.target_octave = 2
                            elif c_type == music21.clef.AltoClef:
                                self.target_octave = 3
                            elif c_type == music21.clef.TenorClef:
                                self.target_octave = 3
                            elif c_type == music21.clef.Treble8vbClef:
                                self.target_octave = 3

        # Final append if anything remains
        if len(m.notesAndRests) > 0:
            part.append(m)

        return part

    def _add_to_measure(
        self,
        m: music21.stream.Measure,
        pitches: List[music21.pitch.Pitch],
        duration: float,
        gen_type: str,
    ):
        actual_type = gen_type
        if gen_type == "mixed":
            actual_type = random.choice(["block", "arpeggio"])

        if actual_type == "block":
            c = music21.chord.Chord(pitches, quarterLength=duration)
            if random.random() < 0.3:
                c.expressions.append(music21.expressions.ArpeggioMark())
            self._inject_failing_classes(c)
            m.append(c)

        elif actual_type == "arpeggio":
            notes_per_beat = random.choice([2, 4])
            note_len = 1.0 / notes_per_beat
            num_notes = int(duration / note_len)

            pattern = random.choice(list(ArpeggioPattern))
            # Expand pitches across 2 octaves for broken/ascending patterns
            expanded = pitches + [p.transpose(12) for p in pitches]
            expanded = sorted(expanded)

            seq = []
            if pattern == ArpeggioPattern.ASCENDING:
                seq = expanded[:num_notes]
                while len(seq) < num_notes:
                    seq.extend(expanded[: num_notes - len(seq)])
            elif pattern == ArpeggioPattern.DESCENDING:
                seq = list(reversed(expanded))[:num_notes]
                while len(seq) < num_notes:
                    seq.extend(list(reversed(expanded))[: num_notes - len(seq)])
            elif pattern == ArpeggioPattern.BROKEN:
                orders = [0, 2, 1, 3] if len(expanded) > 3 else [0, 1, 0, 1]
                seq = [expanded[orders[i % len(orders)]] for i in range(num_notes)]
            else:
                orders = [0, 2, 1, 2] if len(pitches) >= 3 else [0, 1, 0, 1]
                seq = [pitches[orders[i % len(orders)]] for i in range(num_notes)]

            for p in seq[:num_notes]:
                n = music21.note.Note(p, quarterLength=note_len)
                self._inject_failing_classes(n)
                m.append(n)

    def _inject_failing_classes(self, obj):
        if random.random() < 0.15:
            obj.articulations.append(random.choice(FAILING_ARTICULATIONS)())
        if random.random() < 0.10:
            obj.expressions.append(random.choice(FAILING_ORNAMENTS)())
        if random.random() < 0.05:
            obj.articulations.append(
                music21.articulations.Fingering(random.randint(1, 5))
            )


def generate_score(min_measures, max_measures, min_parts, max_parts):
    score = music21.stream.Score()

    num_parts = random.randint(min_parts, max_parts)
    num_measures = random.randint(min_measures, max_measures)

    key_str = random.choice(STARTING_KEYS)
    # Use standard beats per bar for generator logic, but display complex signatures
    # We will pick a signature where numerator is beats_per_bar so the math maps easily
    beats_per_bar = random.choice([2, 3, 4, 6])

    # Generate parts using the advanced generator
    for p_idx in range(num_parts):
        # Top parts are usually Treble/Alto, bottom are Bass
        is_bass_register = p_idx >= (num_parts / 2.0)

        # Mix styles within parts
        gen_type = random.choice(["block", "arpeggio", "mixed"])

        gen = MusicGenerator(
            key_str=key_str,
            total_bars=num_measures,
            beats_per_bar=beats_per_bar,
            is_bass=is_bass_register,
            gen_type=gen_type,
        )
        part = gen.generate_part()

        # Override the time signature randomly with visually complex ones for the model
        valid_ts = [
            t
            for t in TIME_SIGNATURES
            if str(beats_per_bar) in t.split("/")[0]
            or (beats_per_bar == 4 and t in ["C", "cut"])
        ]
        if not valid_ts:
            valid_ts = TIME_SIGNATURES

        display_ts = music21.meter.TimeSignature(random.choice(valid_ts))
        for m in part.getElementsByClass("Measure"):
            # Swap out the TS
            tss = m.getElementsByClass("TimeSignature")
            if tss:
                m.replace(tss[0], display_ts)

            # Inject dynamics natively on measure boundaries
            if random.random() < 0.15:
                dyn = music21.dynamics.Dynamic(random.choice(FAILING_DYNAMICS))
                m.insert(0, dyn)

        # Inject Spanners (8va / Slurs)
        notes_and_chords = list(part.recurse().notes)
        if len(notes_and_chords) > 5 and random.random() < 0.3:
            s_idx = random.randint(0, len(notes_and_chords) - 4)
            e_idx = random.randint(s_idx + 2, len(notes_and_chords) - 1)
            ottava = music21.spanner.Ottava(type=random.choice(["8va", "8vb"]))
            ottava.addSpannedElements(notes_and_chords[s_idx], notes_and_chords[e_idx])
            part.insert(0, ottava)

        if len(notes_and_chords) > 3 and random.random() < 0.4:
            s_idx = random.randint(0, len(notes_and_chords) - 3)
            e_idx = random.randint(s_idx + 1, len(notes_and_chords) - 1)
            slur = music21.spanner.Slur()
            slur.addSpannedElements(notes_and_chords[s_idx], notes_and_chords[e_idx])
            part.insert(0, slur)

        score.insert(0, part)

    return score


def process_file(args_tuple) -> bool:
    file_idx, output_dir, min_m, max_m, min_p, max_p = args_tuple
    try:
        out_path = output_dir / f"synthetic_score_{file_idx:06d}.mxl"
        if out_path.exists():
            return True

        score = generate_score(min_m, max_m, min_p, max_p)
        score.write("musicxml", fp=out_path)
        return True
    except Exception as e:
        logger.error(f"Failed to generate score {file_idx}: {e}", exc_info=True)
        return False


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset prioritizing failing classes."
    )
    parser.add_argument(
        "output_dir", type=str, help="Output directory for generated MXL files"
    )
    parser.add_argument("num_files", type=int, help="Number of files to generate")
    parser.add_argument(
        "--min_measures", type=int, default=12, help="Min measures per file"
    )
    parser.add_argument(
        "--max_measures", type=int, default=32, help="Max measures per file"
    )
    parser.add_argument(
        "--min_parts", type=int, default=2, help="Min parts (instruments) per file"
    )
    parser.add_argument(
        "--max_parts", type=int, default=3, help="Max parts (instruments) per file"
    )

    args = parser.parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.num_files} synthetic files in {output_path}...")

    success_count = 0
    error_count = 0

    tasks = [
        (
            i,
            output_path,
            args.min_measures,
            args.max_measures,
            args.min_parts,
            args.max_parts,
        )
        for i in range(args.num_files)
    ]

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
