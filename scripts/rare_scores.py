"""Find real scores that carry the rare classes of a label set: scan a musicset
selection (selection.jsonl) for the MusicXML features that engrave as those
classes, skipping scores already in the given datasets, and write one line
per score that has any (feature counts as JSON).

The features are what the source encodes; whether the engraving shows them
as the class (and the score passes the build gates) is for the build to say.

Usage:
  uv run python scripts/rare_scores.py data/sources/lite2-pool/selection.jsonl OUT.jsonl \\
      --skip data/train-lite2/filter data/validation-lite2/filter
"""
# Standard library imports
import re
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from musicset.score import read_musicxml

# feature -> the classes it engraves as; regex on the MusicXML text
FEATURES = {
    "clefC": r"<sign>C</sign>",
    "barlineDashed": r"<bar-style>dashed</bar-style>",
    "accidentalDoubleFlat": r"<accidental[^>]*>flat-flat</accidental>",
    "accidentalDoubleSharp": r"<accidental[^>]*>(?:double-sharp|sharp-sharp)</accidental>",
    "keyAccidDouble": r"<key-alter>-?2(?:\.0)?</key-alter>",
    "timeSig0": r"<beats>[1-9]*0</beats>",
    "repeatMark": r"<segno\b|<coda\b",
    "dynamicFFF": r"<fff\s*/>",
    "dynamicForzando": r"<fz\s*/>",
    "dynamicSforzato": r"<sfz\s*/>",
    "dynamicOther": r"<other-dynamics>[^<]*</other-dynamics>",
    "turn": r"<(?:inverted-|delayed-|delayed-inverted-)?turn\b",
    "articStaccatissimo": r"<(?:staccatissimo|spiccato)\b",
    "articMarcato": r"<strong-accent\b",
    "clefG8va": r"<sign>G</sign>\s*<line>2</line>\s*<clef-octave-change>1</clef-octave-change>",
    "clefG8vb": r"<sign>G</sign>\s*<line>2</line>\s*<clef-octave-change>-1</clef-octave-change>",
}
PATTERNS = {name: re.compile(p) for name, p in FEATURES.items()}
MEASURE = re.compile(r"<measure\b")
KEY = re.compile(r"<fifths>(-?\d+)</fifths>")
CLEF_C = re.compile(r"<clef\b[^>]*>\s*<sign>C</sign>")


def features(path: str) -> tuple[str, dict]:
    try:
        xml = read_musicxml(Path(path)).text
    except Exception:
        return path, {}
    found = {name: len(p.findall(xml)) for name, p in PATTERNS.items()}
    # a C clef after the first measure: a clef change (cClefChange mid-system)
    first = [m.start() for m in MEASURE.finditer(xml)][1:2]
    if first:
        found["cClefChange"] = len(CLEF_C.findall(xml, first[0]))
    # key signatures that change mid-piece: cancellation naturals (keyAccidNatural)
    keys = [int(k) for k in KEY.findall(xml)]
    found["keyChange"] = sum(a != b for a, b in zip(keys, keys[1:], strict=False))
    return path, {k: v for k, v in found.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("selection", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--skip", type=Path, nargs="*", default=[], help="dirs of scores already used")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    skip = {p.name.split(".")[0] for d in args.skip for p in d.iterdir()}
    rows = [json.loads(line) for line in args.selection.open()]
    paths = [r["path"] for r in rows if Path(r["path"]).stem not in skip]
    staves = {r["path"]: r["max_staves"] for r in rows}
    hits = 0
    with ProcessPoolExecutor(args.workers) as ex, args.out.open("w") as f:
        for path, found in ex.map(features, paths, chunksize=64):
            if found:
                hits += 1
                f.write(json.dumps({"path": path, "staves": staves[path], **found}) + "\n")
    print(f"{len(paths)} scores scanned ({len(skip)} skipped as used), {hits} with a rare feature -> {args.out}")


if __name__ == "__main__":
    main()
