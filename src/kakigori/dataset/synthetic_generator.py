# Standard library imports
import copy
import json
import random
import logging
import zipfile
import argparse
import multiprocessing
import xml.etree.ElementTree as ET
from pathlib import Path
from fractions import Fraction

# Third party imports
from tqdm import tqdm

try:
    # Third party imports
    import music21
    from music21 import environment

    # Silencing some spammy warnings from music21
    environment.UserSettings()["warnings"] = 0
except ImportError:
    logging.warning(
        "music21 is not installed. Please add it to your environment if you want to run this script."
    )

# Standard library imports
import warnings

warnings.filterwarnings("ignore", message=".*we are out of midi channels.*")
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# SVG to Music21 Mapping definition (updated with your new classes in mind)
SVG_TO_MUSIC21_MAPPING = {
    "tupletNum": "music21.spanner.Tuplet",
    "rest": "music21.note.Rest",
    "mRest": "music21.note.Rest",
    "harm": "music21.harmony.ChordSymbol",
    "tie": "music21.tie.Tie",
    "slur": "music21.spanner.Slur",
    "barLine": "music21.bar.Barline",
    "voltaBracket": "music21.spanner.RepeatBracket",
    "arpeg": "music21.expressions.ArpeggioMark",
    "fing": "music21.articulations.Fingering",
    "trill": "music21.expressions.Trill",
    "mordent": "music21.expressions.Mordent",
    "octave": "music21.spanner.OctaveShift",
    "reh": "music21.expressions.RehearsalMark",
    "stem": "music21.note.Note.stemDirection",
    "chord": "music21.chord.Chord",
    "notehead": "music21.note.Note",
    "clef": "music21.clef.Clef",
    "keySig": "music21.key.KeySignature",
    "meterSig": "music21.meter.TimeSignature",
    "dynam": "music21.dynamics.Dynamic",
}


def test_zip_integrity(mxl_path: Path) -> bool:
    try:
        with zipfile.ZipFile(mxl_path, "r") as zf:
            xml_name = next(
                (
                    n
                    for n in zf.namelist()
                    if (n.endswith(".xml") or n.endswith(".musicxml"))
                    and n != "META-INF/container.xml"
                ),
                None,
            )
            if not xml_name:
                return False
            data = zf.read(xml_name)
            ET.fromstring(data)
            return True
    except zipfile.BadZipFile, ET.ParseError:
        return False
    except Exception:
        return False


def is_safe_to_mutate(n):
    """Returns False if node is part of a Tie or Spanner."""
    if hasattr(n, "tie") and n.tie is not None:
        return False
    if hasattr(n, "getSpannerSites") and n.getSpannerSites():
        return False
    return True


def apply_rhythmic_mutations(score):
    """Safely injects tuplets, 16ths, dotted rhythms, and 32nds by managing offsets."""
    for measure in score.recurse().getElementsByClass(music21.stream.Measure):
        elements = list(measure.notes)

        for n1 in elements:
            if not isinstance(n1, music21.note.Note) or not is_safe_to_mutate(n1):
                continue

            parent = n1.activeSite
            if not parent:
                continue

            # 1. Inject Dense 16ths
            if n1.duration.quarterLength >= 1.0 and random.random() < 0.10:
                qL = n1.duration.quarterLength
                num_16ths = int(qL / 0.25)
                orig_offset = n1.offset

                try:
                    parent.remove(n1)
                    for i in range(num_16ths):
                        if random.random() < 0.20:
                            new_elem = music21.note.Rest(quarterLength=0.25)
                        else:
                            new_elem = music21.note.Note(n1.pitch, quarterLength=0.25)
                        parent.insert(orig_offset + (i * 0.25), new_elem)
                except music21.exceptions21.StreamException:
                    pass
                continue

            # 2. Inject Dotted Rhythms (Dotted 8th + 16th)
            if n1.duration.quarterLength == 1.0 and random.random() < 0.15:
                orig_offset = n1.offset
                n_dotted = music21.note.Note(n1.pitch, quarterLength=0.75)
                n_short = music21.note.Note(n1.pitch, quarterLength=0.25)

                try:
                    parent.remove(n1)
                    parent.insert(orig_offset, n_dotted)
                    parent.insert(orig_offset + 0.75, n_short)
                except music21.exceptions21.StreamException:
                    pass
                continue

            # 3. Inject 32nd rests/notes
            if n1.duration.quarterLength == 0.25 and random.random() < 0.10:
                orig_offset = n1.offset
                try:
                    parent.remove(n1)
                    for i in range(2):
                        if random.random() < 0.3:
                            new_elem = music21.note.Rest(quarterLength=0.125)
                        else:
                            new_elem = music21.note.Note(n1.pitch, quarterLength=0.125)
                        parent.insert(orig_offset + (i * 0.125), new_elem)
                except music21.exceptions21.StreamException:
                    pass
                continue

            # 4. Tuplet generation
            if n1.duration.quarterLength >= 0.5 and random.random() < 0.15:
                written_qL = Fraction(n1.duration.quarterLength) / 2
                orig_offset = n1.offset

                new_n1 = music21.note.Note(n1.pitch, quarterLength=written_qL)
                new_n2 = music21.note.Note(n1.pitch, quarterLength=written_qL)
                new_n3 = music21.note.Note(n1.pitch, quarterLength=written_qL)

                tup = music21.duration.Tuplet(3, 2)
                tup_start = music21.duration.Tuplet(3, 2)
                tup_start.type = "start"
                tup_stop = music21.duration.Tuplet(3, 2)
                tup_stop.type = "stop"
                new_n1.duration.appendTuplet(tup_start)
                new_n2.duration.appendTuplet(tup)
                new_n3.duration.appendTuplet(tup_stop)

                offset_step = Fraction(n1.duration.quarterLength) / 3

                try:
                    parent.remove(n1)
                    parent.insert(orig_offset, new_n1)
                    parent.insert(orig_offset + offset_step, new_n2)
                    parent.insert(orig_offset + (offset_step * 2), new_n3)
                except music21.exceptions21.StreamException:
                    pass

    return score


def apply_ornaments(score):
    """Injects fingering, trills, mordents, articulations, dynamics, clefs, and builds chords."""
    for note in list(score.recurse().notes):
        if not isinstance(note, music21.note.Note):
            continue

        has_ornament = False

        # Original Ornaments
        if random.random() < 0.05:
            note.expressions.append(music21.expressions.Trill())
            has_ornament = True

        if not has_ornament and random.random() < 0.05:
            note.expressions.append(music21.expressions.Mordent())
            has_ornament = True

        if not has_ornament and random.random() < 0.2:
            note.articulations.append(
                music21.articulations.Fingering(random.randint(1, 5))
            )
            has_ornament = True

        if not has_ornament and random.random() < 0.05:
            turn = music21.expressions.Turn()
            note.expressions.append(turn)
            has_ornament = True

        # NEW: Standard Articulations
        if random.random() < 0.15:
            artic = random.choice(
                [
                    music21.articulations.Staccato(),
                    music21.articulations.Tenuto(),
                    music21.articulations.Accent(),
                    music21.articulations.Staccatissimo(),
                    music21.articulations.StrongAccent(),  # Maps to Marcato
                ]
            )
            note.articulations.append(artic)

        # NEW: Dynamics
        if random.random() < 0.05 and note.activeSite:
            dyn_symbol = random.choice(["p", "mp", "mf", "f", "sfz", "fz"])
            dyn = music21.dynamics.Dynamic(dyn_symbol)
            try:
                note.activeSite.insert(note.offset, dyn)
            except music21.exceptions21.StreamException:
                pass

        # Chords and Arpeggios
        if (
            random.random() < 0.10
            and is_safe_to_mutate(note)
            and note.duration.quarterLength >= 0.5
        ):
            pitches = [note.pitch, note.pitch.transpose(5), note.pitch.transpose(9)]
            if random.random() < 0.5:
                pitches.append(note.pitch.transpose(12))

            for p in pitches:
                if p.accidental:
                    p.accidental.displayStatus = False

            new_chord = music21.chord.Chord(pitches)
            new_chord.duration = copy.deepcopy(note.duration)

            # Transfer existing expressions
            new_chord.expressions = copy.deepcopy(note.expressions)
            new_chord.articulations = copy.deepcopy(note.articulations)

            has_accidental = any(p.accidental is not None for p in pitches)
            if not has_accidental and random.random() < 0.5:
                new_chord.expressions.append(music21.expressions.ArpeggioMark())

            parent = note.activeSite
            if parent:
                try:
                    parent.replace(note, new_chord)
                    note = new_chord
                except music21.exceptions21.StreamException:
                    pass

        # Text and Rehearsal marks
        if random.random() < 0.1:
            note.lyric = random.choice(["C", "G7", "Amin", "F#dim"])

        if random.random() < 0.02:
            note.expressions.append(
                music21.expressions.RehearsalMark(random.choice(["A", "B", "C", "D"]))
            )

        # Stems
        if random.random() < 0.05 and note.duration.quarterLength >= 1.0:
            note.stemDirection = "up" if random.random() < 0.5 else "down"

    return score


def apply_spanners(score):
    """Safely adds slurs at the Part level, avoiding Measure corruption."""
    for part in score.parts:
        notes_and_chords = list(part.recurse().notesAndRests)
        valid_targets = [n for n in notes_and_chords if not n.isRest]
        part_changed = False

        if not part_changed and len(valid_targets) >= 3 and random.random() < 0.3:
            start_idx = random.randint(0, len(valid_targets) - 3)
            end_idx = random.randint(start_idx + 2, len(valid_targets) - 1)

            slur = music21.spanner.Slur()
            slur.addSpannedElements(valid_targets[start_idx], valid_targets[end_idx])
            part.insert(0, slur)
            part_changed = True

        if not part_changed and len(valid_targets) >= 4 and random.random() < 0.15:
            start_idx = random.randint(0, len(valid_targets) - 4)
            end_idx = random.randint(start_idx + 3, len(valid_targets) - 1)

            shift_type = random.choice(["8va", "8vb"])
            ottava = music21.spanner.Ottava(type=shift_type)
            ottava.addSpannedElements(valid_targets[start_idx], valid_targets[end_idx])
            part.insert(0, ottava)
            part_changed = True

    return score


def mutate_score(score):
    """Master pipeline for orchestrating the mutations safely."""
    score = apply_rhythmic_mutations(score)
    score = apply_ornaments(score)
    score = apply_spanners(score)

    try:
        for part in score.parts:
            part.makeBeams(inPlace=True)
    except Exception as e:
        logger.debug(f"Failed to generate beams natively, clearing them. {e}")
        for n in score.recurse().notes:
            if hasattr(n, "beams") and n.beams:
                n.beams.beamsList = []

    return score


def process_file(mxl_path: Path, output_dir: Path) -> bool:
    if not test_zip_integrity(mxl_path):
        logger.debug(f"Skipping corrupted file: {mxl_path.name}")
        return False

    out_path = output_dir / f"synthetic_{mxl_path.name}"
    if out_path.exists():
        logger.debug(f"Output already exists: {out_path.name}")
        return True

    try:
        score = music21.converter.parse(mxl_path)
        mutated_score = mutate_score(score)
        mutated_score.write("musicxml", fp=out_path)
        return True
    except Exception as e:
        logger.debug(f"Failed to process {mxl_path.name} with music21: {e}")
        return False


def _process_file_worker(args):
    mxl_path, output_dir = args
    return process_file(mxl_path, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset by coercing musicXML files."
    )
    parser.add_argument("input_dir", type=str, help="Input directory of MXL files")
    parser.add_argument(
        "output_dir", type=str, help="Output directory for mutated MXL files"
    )
    parser.add_argument(
        "--num_files", type=int, default=1000, help="Number of files to generate"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input dir {input_path} is invalid.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    mxl_files = list(input_path.rglob("*.mxl"))
    random.shuffle(mxl_files)

    logger.info(
        f"Found {len(mxl_files)} files. Target: {args.num_files} successful generations."
    )

    success_count = 0
    error_count = 0

    with open(output_path / "svg_music21_mapping.json", "w") as f:
        json.dump(SVG_TO_MUSIC21_MAPPING, f, indent=2)

    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(), maxtasksperchild=10
    ) as pool:
        tasks = [(f, output_path) for f in mxl_files]
        results = pool.imap_unordered(_process_file_worker, tasks)

        with tqdm(total=args.num_files, desc="Generating files") as pbar:
            for is_success in results:
                if is_success:
                    success_count += 1
                    pbar.update(1)
                else:
                    error_count += 1

                if success_count >= args.num_files:
                    logger.info("Reached target number of generated files!")
                    pool.terminate()
                    break

    logger.info(f"Finished. Generated {success_count} files (Failed: {error_count}).")


if __name__ == "__main__":
    main()
