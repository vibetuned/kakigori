# Standard library imports
import random
import logging
import argparse
import multiprocessing
from typing import List, Callable
from pathlib import Path

# Third party imports
from tqdm import tqdm

try:
    import music21
    from music21 import environment

    environment.UserSettings()["warnings"] = 0
except ImportError:
    pass

# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# Articulation classes (SMuFL names)
# =============================================================================

ARTICULATION_CLASSES = [
    music21.articulations.Accent,          # articAccentAbove
    music21.articulations.StrongAccent,     # articMarcatoAbove
    music21.articulations.Staccatissimo,    # articStaccatissimoAbove
    music21.articulations.Staccato,         # articStaccatoAbove
    music21.articulations.Tenuto,           # articTenutoAbove
]

ORNAMENT_CLASSES = [
    music21.expressions.Mordent,
    music21.expressions.Trill,
    music21.expressions.Turn,
]


# =============================================================================
# Measure-level mutations
# =============================================================================


def _mutate_add_articulation(measure: music21.stream.Measure) -> music21.stream.Measure:
    """Pick one random articulation and apply it to every note/chord in the measure."""
    artic_cls = random.choice(ARTICULATION_CLASSES)

    for el in measure.recurse().notesAndRests:
        if el.isRest:
            continue
        el.articulations.append(artic_cls())

    return measure


def _mutate_add_ornament(measure: music21.stream.Measure) -> music21.stream.Measure:
    """Pick a random note/chord in the measure and add a random ornament to it."""
    notes = [el for el in measure.recurse().notesAndRests if not el.isRest]
    if not notes:
        return measure

    target = random.choice(notes)
    ornament_cls = random.choice(ORNAMENT_CLASSES)
    target.expressions.append(ornament_cls())

    # If the target is a chord, 50% chance to also add an arpeggio mark
    if isinstance(target, music21.chord.Chord) and random.random() < 0.5:
        target.expressions.append(music21.expressions.ArpeggioMark())

    return measure


DYNAMIC_SYMBOLS = [
    "mf",   # dynamicMezzo
    "p",    # dynamicPiano
    "sfz",  # dynamicSforzando
    "z",    # dynamicZ
]


def _mutate_add_dynamic(measure: music21.stream.Measure) -> music21.stream.Measure:
    """Pick a random note/chord in the measure and insert a dynamic at its offset."""
    notes = [el for el in measure.recurse().notesAndRests if not el.isRest]
    if not notes:
        return measure

    target = random.choice(notes)
    dyn = music21.dynamics.Dynamic(random.choice(DYNAMIC_SYMBOLS))
    measure.insert(target.offset, dyn)

    return measure


def _mutate_octave_shift(measure: music21.stream.Measure) -> music21.stream.Measure:
    """Add an 8va or 8vb ottava bracket and transpose notes to compensate."""
    # Only keep pitched elements (skip Unpitched / PercussionChord)
    notes = [
        el for el in measure.recurse().notesAndRests
        if not el.isRest and isinstance(el, (music21.note.Note, music21.chord.Chord))
    ]
    if len(notes) < 2:
        return measure

    direction = random.choice(['8va', '8vb'])
    # Transpose notes in the opposite direction so sounding pitch is preserved
    # 8va plays an octave up → write notes an octave down (-12)
    # 8vb plays an octave down → write notes an octave up (+12)
    shift = 12 if direction == '8va' else -12
    for el in notes:
        el.transpose(shift, inPlace=True)

    # '8va' = up, '8vb' = down
    ottava = music21.spanner.Ottava(type=direction, transposing=False)
    ottava.addSpannedElements(notes[0], notes[-1])
    measure.insert(0, ottava)


    return measure


# Building blocks for the duration-replacement mutation.
# Each entry: (label, quarterLength, is_rest)
# "16th_pair" is a 16th note followed by a 32nd rest (avoids beaming).
_DURATION_BLOCKS = [
    ("16th_pair", 0.25 + 0.125, False),  # 16th note + 32nd rest = 0.375
    ("half_note",  2.0, False),           # noteheadHalf
    ("whole_note", 4.0, False),           # noteheadWhole
    ("half_rest",  2.0, True),            # restHalf
    ("whole_rest", 4.0, True),            # restWhole
]


def _mutate_replace_durations(measure: music21.stream.Measure) -> music21.stream.Measure:
    """Replace notes with a mix of 16ths (+ 32nd-rest spacers), halves, and wholes.

    Preserves original pitches (cycled) and keeps exact measure duration.
    """
    # Collect original pitches (handle chords by taking the root)
    original_pitches: List[music21.pitch.Pitch] = []
    for el in measure.recurse().notesAndRests:
        if el.isRest:
            continue
        if isinstance(el, music21.chord.Chord):
            if el.pitches:
                original_pitches.append(el.pitches[0])
        elif hasattr(el, 'pitch'):
            original_pitches.append(el.pitch)

    if not original_pitches:
        return measure

    # Get total duration from time signature or existing content
    total_ql = measure.barDuration.quarterLength if measure.barDuration else 4.0

    # Preserve non-note elements (time sig, key sig, clef, etc.)
    keep = []
    for el in measure:
        if isinstance(el, (
            music21.meter.TimeSignature,
            music21.key.KeySignature,
            music21.key.Key,
            music21.clef.Clef,
        )):
            keep.append((el.offset, el))

    measure.clear()
    for offset, el in keep:
        measure.insert(offset, el)

    # Fill the measure with random blocks
    pitch_idx = 0
    current_offset = 0.0
    remaining = total_ql

    while remaining > 0.001:
        # Filter blocks that fit in the remaining space
        fitting = [(label, ql, is_rest) for label, ql, is_rest in _DURATION_BLOCKS
                    if ql <= remaining + 0.001]
        if not fitting:
            # Nothing fits — fill the leftover with a rest
            measure.insert(current_offset, music21.note.Rest(quarterLength=remaining))
            break

        label, ql, is_rest = random.choice(fitting)

        if label == "16th_pair":
            # 16th note
            p = original_pitches[pitch_idx % len(original_pitches)]
            pitch_idx += 1
            n = music21.note.Note(p, quarterLength=0.25)
            measure.insert(current_offset, n)
            current_offset += 0.25
            # 32nd rest spacer
            measure.insert(current_offset, music21.note.Rest(quarterLength=0.125))
            current_offset += 0.125
        elif is_rest:
            measure.insert(current_offset, music21.note.Rest(quarterLength=ql))
            current_offset += ql
        else:
            p = original_pitches[pitch_idx % len(original_pitches)]
            pitch_idx += 1
            n = music21.note.Note(p, quarterLength=ql)
            measure.insert(current_offset, n)
            current_offset += ql

        remaining = total_ql - current_offset

    return measure


def _mutate_add_tuplets(measure: music21.stream.Measure) -> music21.stream.Measure:
    """Replace notes with two triplet groups if the measure is full and even-length."""
    # Only apply if measure is complete (actual duration == bar duration)
    bar_ql = measure.barDuration.quarterLength if measure.barDuration else 4.0
    measure_ql = measure.duration.quarterLength

    if abs(measure_ql - bar_ql) > 0.001:
        return measure

    # Only apply if the bar duration is even (divisible by 2)
    half_ql = bar_ql / 2.0
    if abs(half_ql - round(half_ql)) > 0.001:
        return measure

    # Collect original pitches
    original_pitches: List[music21.pitch.Pitch] = []
    for el in measure.recurse().notesAndRests:
        if el.isRest:
            continue
        if isinstance(el, music21.chord.Chord):
            if el.pitches:
                original_pitches.append(el.pitches[0])
        elif hasattr(el, 'pitch'):
            original_pitches.append(el.pitch)

    if not original_pitches:
        return measure

    # Preserve metadata (time sig, key sig, clef)
    keep = []
    for el in measure:
        if isinstance(el, (
            music21.meter.TimeSignature,
            music21.key.KeySignature,
            music21.key.Key,
            music21.clef.Clef,
        )):
            keep.append((el.offset, el))

    measure.clear()
    for offset, el in keep:
        measure.insert(offset, el)

    # Each triplet: 3 notes spanning half the bar
    # note_ql = half_ql / 3  (actual sounding duration of each triplet note)
    note_ql = half_ql / 3.0

    # Determine the normal duration type for the tuplet
    # The "normal" note is half_ql / 2 (what 2 normal notes would fill)
    normal_ql = half_ql / 2.0
    normal_dur = music21.duration.Duration(quarterLength=normal_ql)

    current_offset = 0.0
    for _ in range(2):  # two triplet groups
        tup = music21.duration.Tuplet(
            numberNotesActual=3,
            numberNotesNormal=2,
            durationNormal=normal_dur,
        )
        for _ in range(3):
            p = random.choice(original_pitches)
            n = music21.note.Note(p, quarterLength=note_ql)
            n.duration.tuplets = (tup,)
            measure.insert(current_offset, n)
            current_offset += note_ql
    measure.makeNotation(inPlace=True)
    return measure


# Registry of all available mutations — chosen from at random per measure
MUTATIONS: List[Callable] = [
    _mutate_add_articulation,
    _mutate_add_ornament,
    _mutate_add_dynamic,
    _mutate_octave_shift,
    _mutate_replace_durations,
    _mutate_add_tuplets,
]


def mutate_measure(measure: music21.stream.Measure) -> music21.stream.Measure:
    """Pick a random mutation and apply it to the measure."""
    mutation = random.choice(MUTATIONS)
    return mutation(measure)


# =============================================================================
# File processing
# =============================================================================


def process_file(args_tuple) -> bool:
    """Load one MXL file, mutate each measure, and write the result."""
    input_path, output_dir = args_tuple
    try:
        out_path = output_dir / f"{input_path.stem}_arranged{input_path.suffix}"
        if out_path.exists():
            return True

        score = music21.converter.parse(str(input_path))

        # Walk every measure in every part and apply mutations
        for part in score.parts:
            for measure in part.getElementsByClass("Measure"):
                mutate_measure(measure)

        score.write("musicxml", fp=out_path, makeNotation=False)
        return True
    except Exception as e:
        logger.error(f"Failed to process {input_path.name}: {e}", exc_info=True)
        return False


# =============================================================================
# Main
# =============================================================================


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Mutate MXL scores measure by measure to create arranged variants."
    )
    parser.add_argument(
        "input_dir", type=str, help="Input directory containing MXL files"
    )
    parser.add_argument(
        "output_dir", type=str, help="Output directory for arranged MXL files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu_count - 1)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover all MXL files in the input directory (recursive)
    mxl_files: List[Path] = sorted(input_path.rglob("*.mxl"))
    if not mxl_files:
        logger.warning(f"No MXL files found in {input_path}")
        return

    logger.info(f"Found {len(mxl_files)} MXL files in {input_path}")
    logger.info(f"Output directory: {output_path}")

    num_workers = args.workers or max(1, multiprocessing.cpu_count() - 1)
    tasks = [(f, output_path) for f in mxl_files]

    success_count = 0
    error_count = 0

    with multiprocessing.Pool(
        processes=num_workers, maxtasksperchild=10
    ) as pool:
        results = pool.imap_unordered(process_file, tasks)
        with tqdm(total=len(mxl_files), desc="Arranging files") as pbar:
            for is_success in results:
                if is_success:
                    success_count += 1
                else:
                    error_count += 1
                pbar.update(1)

    logger.info(
        f"Finished. Arranged {success_count} files (Failed: {error_count})."
    )


if __name__ == "__main__":
    main()
