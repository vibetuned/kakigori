import argparse
import logging
import zipfile
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

try:
    import music21
    from music21 import environment
    # Silencing some spammy warnings from music21
    environment.UserSettings()['warnings'] = 0
except ImportError:
    logging.warning("music21 is not installed. Please add it to your environment if you want to run this script.")

import warnings
warnings.filterwarnings("ignore", message=".*we are out of midi channels.*")
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# SVG to Music21 Mapping definition
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
    "dynam": "music21.dynamics.Dynamic"
}


def test_zip_integrity(mxl_path: Path) -> bool:
    try:
        with zipfile.ZipFile(mxl_path, 'r') as zf:
            xml_name = next((n for n in zf.namelist() if (n.endswith('.xml') or n.endswith('.musicxml')) and n != 'META-INF/container.xml'), None)
            if not xml_name:
                return False
            # Check if it parses
            data = zf.read(xml_name)
            ET.fromstring(data)
            return True
    except (zipfile.BadZipFile, ET.ParseError):
        return False
    except Exception:
        return False

def is_safe_to_mutate(n):
    """Returns False if node is part of a Tie or Spanner (Slur/Hairpin)."""
    if hasattr(n, 'tie') and n.tie is not None:
        return False
    if hasattr(n, 'getSpannerSites') and n.getSpannerSites():
        return False
    return True

def mutate_score(score):
    """
    Applies synthetic mutations according to curriculum strategy,
    ensuring resulting score remains valid (durations add up).
    """
    # 1. Tuplets overlap and Dense 16ths

    for measure in score.recurse().getElementsByClass(music21.stream.Measure):
        notes = list(measure.notes)
        
        # Track (parent_stream, element) and (parent_stream, offset, element)
        elements_to_remove = []
        elements_to_add = []
        
        for n1 in notes:
            # Injecting Dense 16th Notes:
            if isinstance(n1, music21.note.Note) and n1.duration.quarterLength >= 1.0:
                if random.random() < 0.10: 
                    parent = n1.activeSite # Find the exact Voice this note belongs to
                    if not parent: continue
                    
                    orig_offset = n1.offset
                    qL = n1.duration.quarterLength
                    
                    elements_to_remove.append((parent, n1))
                    num_16ths = int(qL / 0.25)
                    
                    for j in range(num_16ths):
                        if random.random() < 0.20:
                            new_elem = music21.note.Rest()
                        else:
                            new_elem = music21.note.Note(n1.pitch)
                            
                        new_elem.duration.quarterLength = 0.25
                        elements_to_add.append((parent, orig_offset + (j * 0.25), new_elem))
                        
                    continue # Skip to next note so we don't double-mutate

            # Tuplet generation
            if isinstance(n1, music21.note.Note) and n1.duration.quarterLength >= 0.5:
                if random.random() < 0.15: 
                    parent = n1.activeSite # Find the exact Voice
                    if not parent: continue
                    
                    qL = n1.duration.quarterLength
                    orig_offset = n1.offset
                    
                    elements_to_remove.append((parent, n1))
                    
                    # Written duration (e.g., 0.25 for a 16th note)
                    written_qL = qL / 2.0 
                    # ACTUAL timeline space each note takes (e.g., 0.1666)
                    offset_step = qL / 3.0 
                    
                    # Tuplet generation
                    tup_start = music21.duration.Tuplet(3, 2)
                    tup_start.type = 'start'
                    
                    tup_mid = music21.duration.Tuplet(3, 2)
                    # mid type is implicitly None/continue
                    
                    tup_stop = music21.duration.Tuplet(3, 2)
                    tup_stop.type = 'stop'
                    
                    new_n1 = music21.note.Note(n1.pitch)
                    new_n1.duration.quarterLength = written_qL
                    new_n1.duration.appendTuplet(tup_start)
                    
                    new_n2 = music21.note.Note(n1.pitch)
                    new_n2.duration.quarterLength = written_qL
                    new_n2.duration.appendTuplet(tup_mid)
                    
                    new_n3 = music21.note.Note(n1.pitch)
                    new_n3.duration.quarterLength = written_qL
                    new_n3.duration.appendTuplet(tup_stop)
                    
                    # Use offset_step so they perfectly fit in the timeline!
                    elements_to_add.append((parent, orig_offset, new_n1))
                    elements_to_add.append((parent, orig_offset + offset_step, new_n2))
                    elements_to_add.append((parent, orig_offset + (offset_step * 2), new_n3))
                    
                    continue

        # --- SAFELY APPLY MUTATIONS TO THEIR SPECIFIC PARENT STREAMS ---
        for parent, elem in elements_to_remove:
            try:
                parent.remove(elem)
            except (ValueError, AttributeError):
                pass
                
        for parent, offset, elem in elements_to_add:
            try:
                parent.insert(offset, elem)
            except (ValueError, AttributeError):
                pass
        
    # 2. Add decorations (Zeroes)
    for note in list(score.recurse().notes):
        if isinstance(note, music21.note.Note):
            # Spam the Zeroes: fing, trill, mordent, arpeg, harm, reh
            has_ornament = False
            if random.random() < 0.05:
                # Add trill
                trill = music21.expressions.Trill()
                note.expressions.append(trill)
                has_ornament = True
            
            if random.random() < 0.05:
                # Add Turn
                mordent = music21.expressions.Mordent()
                note.expressions.append(mordent)
                has_ornament = True

            if not has_ornament and random.random() < 0.2:
                # Add random fingering 1-5
                fingering = music21.articulations.Fingering(random.randint(1, 5))
                note.articulations.append(fingering)

            # Create Chords and Arpeggios
            if random.random() < 0.10 and is_safe_to_mutate(note) and note.duration.quarterLength >= 0.5:
                # FIXED: Use 'note' instead of the leaked 'n1'
                pitches = [note.pitch, note.pitch.transpose(5), note.pitch.transpose(9)]
                if random.random() < 0.5:
                    pitches.append(note.pitch.transpose(12)) 
                    
                new_chord = music21.chord.Chord(pitches)
                new_chord.duration = note.duration
                
                has_accidental = any(p.accidental is not None for p in pitches)
                if not has_accidental and random.random() < 0.5: 
                    arpeg = music21.expressions.ArpeggioMark()
                    new_chord.expressions.append(arpeg)
                
                # FIXED: Swap the note for the chord dynamically
                try:
                    parent_stream = note.activeSite
                    if parent_stream:
                        parent_stream.replace(note, new_chord)
                        note = new_chord # Update reference
                except (ValueError, AttributeError):
                    pass

            if random.random() < 0.1:
                # Add arbitrary text to force "harm" or "verse" like elements
                note.lyric = str(random.choice(['C', 'G7', 'Amin', 'F#dim']))
                
            # Random string/word as a label or rehearsal mark
            if random.random() < 0.02:
                reh = music21.expressions.RehearsalMark(random.choice(['A', 'B', 'C', 'D']))
                note.expressions.append(reh)

            # Randomize stems to force tricky bounding boxes
            if random.random() < 0.05 and note.duration.quarterLength >= 1.0:
                note.stemDirection = 'up' if random.random() < 0.5 else 'down'

    # Add Slurs spanning multiple notes
    for measure in score.recurse().getElementsByClass(music21.stream.Measure):
        notes_and_chords = list(measure.notesAndRests)
        valid_targets = [n for n in notes_and_chords if not n.isRest]
        
        if len(valid_targets) >= 3 and random.random() < 0.3:
            # Pick a random start and end point
            start_idx = random.randint(0, len(valid_targets) - 3)
            end_idx = random.randint(start_idx + 2, len(valid_targets) - 1)
            
            slur = music21.spanner.Slur()
            slur.addSpannedElements(valid_targets[start_idx], valid_targets[end_idx])
            measure.insert(0, slur)

    for n in score.recurse().notes:
        if hasattr(n, 'beams') and n.beams:
            n.beams.clear()

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
        
        # Write back to musicxml (.mxl)
        mutated_score.write('musicxml', fp=out_path)
        return True
    except Exception as e:
        logger.debug(f"Failed to process {mxl_path.name} with music21: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset by coercing musicXML files.")
    parser.add_argument("input_dir", type=str, help="Input directory of MXL files")
    parser.add_argument("output_dir", type=str, help="Output directory for mutated MXL files")
    parser.add_argument("--num_files", type=int, default=1000, help="Number of files to successfully generate and then stop")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input dir {input_path} is invalid.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    
    mxl_files = list(input_path.rglob("*.mxl"))
    random.shuffle(mxl_files) # Shuffle to grab random subset

    logger.info(f"Found {len(mxl_files)} files to consider. Target: {args.num_files} successful generations.")
    
    success_count = 0
    error_count = 0
    
    # Dump the mapping as a JSON so users can look at it later
    with open(output_path / "svg_music21_mapping.json", "w") as f:
        import json
        json.dump(SVG_TO_MUSIC21_MAPPING, f, indent=2)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # submit tasks
        futures = {executor.submit(process_file, f, output_path): f for f in mxl_files}

        with tqdm(total=args.num_files, desc="Generating files") as pbar:
            for future in as_completed(futures):
                try:
                    if future.result():
                        success_count += 1
                        pbar.update(1)
                    else:
                        error_count += 1
                except concurrent.futures.process.BrokenProcessPool:
                    # FIXED: Catch the exact error that causes the cascading failure
                    logger.debug("FATAL: A worker process crashed abruptly (likely Out-of-Memory). The process pool is dead. Halting.")
                    break
                except Exception as e:
                    # CHANGED: Show the error so you know what failed
                    logger.debug(f"Worker task failed unexpectedly: {repr(e)}")
                    error_count += 1

                if success_count >= args.num_files:
                    logger.info("Reached target number of generated files!")
                    for f_unresolved in futures:
                        f_unresolved.cancel()
                    break

    logger.info(f"Finished. Successfully generated {success_count} files (Failed: {error_count}).")

if __name__ == "__main__":
    main()
