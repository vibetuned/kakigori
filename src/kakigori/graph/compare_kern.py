# Standard library imports
import re
import logging
import argparse
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MEI_NS = "{http://www.music-encoding.org/ns/mei}"
NS = {"mei": "http://www.music-encoding.org/ns/mei"}

SHARPS_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
FLATS_ORDER = ["B", "E", "A", "D", "G", "C", "F"]
ACCID_ALTER = {"s": 1, "f": -1, "n": 0, "ss": 2, "x": 2, "ff": -2}
KERN_ALTER = {"#": 1, "##": 2, "-": -1, "--": -2, "n": 0, "": 0}
KERN_NOTE = re.compile(r"([a-gA-G]+)(##|--|#|-|n)?")


def _keysig_alters(sig: str) -> dict:
    """'2s' -> {'F': 1, 'C': 1}; '1f' -> {'B': -1}; '0'/None -> {}."""
    m = re.match(r"(\d+)([sf])", sig or "")
    if not m:
        return {}
    n, kind = int(m.group(1)), m.group(2)
    order = SHARPS_ORDER if kind == "s" else FLATS_ORDER
    return {letter: (1 if kind == "s" else -1) for letter in order[:n]}


def _note_alter(note, parents: dict, keysig: dict, carry: dict) -> int:
    """Sounding alteration of an MEI note, applying keysig + measure carry-over."""
    accid = note.get("accid")
    if accid is None:
        ac_el = note.find("mei:accid", NS)
        if ac_el is not None:
            accid = ac_el.get("accid") or ac_el.get("accid.ges")
    letter = note.get("pname", "").upper()
    octave = int(note.get("oct"))
    if accid is not None:
        alter = ACCID_ALTER.get(accid, 0)
        carry[(letter, octave)] = alter
        return alter
    ges = note.get("accid.ges")
    if ges is not None:
        return ACCID_ALTER.get(ges, 0)
    if (letter, octave) in carry:
        return carry[(letter, octave)]
    return keysig.get(letter, 0)


def _note_value(note, parents: dict, ppq: float) -> float | None:
    """Duration of an MEI note in whole-note units, or None for grace notes.

    Prefers dur.ppq (which already includes dots and tuplet scaling); falls
    back to dur/dots arithmetic scaled by tuplet ancestors. Notes inside a
    chord inherit the chord's duration attributes.
    """
    holder = note
    while holder is not None and holder.get("dur") is None and holder.get("dur.ppq") is None:
        holder = parents.get(holder)
        if holder is None or holder.tag != f"{MEI_NS}chord":
            break
    if holder is None:
        return None
    if note.get("grace") is not None or (holder is not note and holder.get("grace") is not None):
        return None

    dur_ppq = holder.get("dur.ppq")
    if dur_ppq is not None and ppq:
        return float(dur_ppq) / (4.0 * ppq)

    dur = holder.get("dur")
    if dur is None or not dur.isdigit():
        return None
    value = 1.0 / int(dur)
    dots = int(holder.get("dots", 0))
    if dots:
        value *= 2.0 - 0.5 ** dots
    anc = parents.get(holder)
    while anc is not None:
        if anc.tag == f"{MEI_NS}tuplet":
            num, numbase = anc.get("num"), anc.get("numbase")
            if num and numbase:
                value *= float(numbase) / float(num)
        anc = parents.get(anc)
    return value


def mei_bags(path: Path) -> tuple[dict, dict]:
    """Per (staff_n, measure_idx): Counter of (letter, oct, alter) and of
    (letter, oct, alter, value) for notes with a resolvable duration."""
    root = ET.parse(path).getroot()
    parents = {child: parent for parent in root.iter() for child in parent}

    keysigs, ppqs = {}, {}
    for sd in root.iter(f"{MEI_NS}staffDef"):
        n = sd.get("n")
        if n in keysigs:
            continue  # keep the initial signature; mid-piece changes are rare
        ks = sd.find("mei:keySig", NS)
        sig = ks.get("sig") if ks is not None else sd.get("key.sig")
        keysigs[n] = _keysig_alters(sig)
        if sd.get("ppq"):
            ppqs[n] = float(sd.get("ppq"))

    pitch_bags, rhythm_bags = {}, {}
    for m_idx, measure in enumerate(root.iter(f"{MEI_NS}measure"), start=1):
        for staff in measure.findall("mei:staff", NS):
            n = staff.get("n")
            carry = {}
            p_bag, r_bag = Counter(), Counter()
            for note in staff.iter(f"{MEI_NS}note"):
                if note.get("pname") is None or note.get("oct") is None:
                    continue  # unpitched (percussion 'loc' notation)
                letter, octave = note.get("pname").upper(), int(note.get("oct"))
                alter = _note_alter(note, parents, keysigs.get(n, {}), carry)
                p_bag[(letter, octave, alter)] += 1
                value = _note_value(note, parents, ppqs.get(n, 0.0))
                if value is not None:
                    r_bag[(letter, octave, alter, round(value, 4))] += 1
            pitch_bags[(n, m_idx)] = p_bag
            rhythm_bags[(n, m_idx)] = r_bag
    return pitch_bags, rhythm_bags


def kern_bags(path: Path) -> tuple[dict, dict]:
    """Same two bag structures extracted from a generated **kern file."""
    lines = path.read_text(encoding="utf-8").splitlines()

    staff_of_col = {}
    for line in lines:
        if line.startswith("*") and "staff" in line:
            for col, tok in enumerate(line.split("\t")):
                m = re.match(r"\*staff(\d+)", tok)
                if m:
                    staff_of_col[col] = m.group(1)
            break

    pitch_bags, rhythm_bags = {}, {}
    m_idx = 0
    for line in lines:
        if line.startswith("="):
            m = re.match(r"=(\d+)", line)
            if m:
                m_idx = int(m.group(1))
            continue
        if not line or line.startswith(("*", "!")):
            continue
        for col, tok in enumerate(line.split("\t")):
            staff_n = staff_of_col.get(col)
            if staff_n is None:
                continue
            p_bag = pitch_bags.setdefault((staff_n, m_idx), Counter())
            r_bag = rhythm_bags.setdefault((staff_n, m_idx), Counter())
            for sub in tok.split(" "):
                if sub == "." or "r" in sub:
                    continue
                m = KERN_NOTE.search(re.sub(r"^\D*\d+", "", sub))
                if not m:
                    continue
                letters, accid = m.group(1), m.group(2) or ""
                if len(set(letters.upper())) != 1:
                    continue
                octave = len(letters) + 3 if letters.islower() else 4 - len(letters)
                key = (letters[0].upper(), octave, KERN_ALTER[accid])
                p_bag[key] += 1
                if "q" in sub:
                    continue  # grace notes carry no duration on either side
                dur_m = re.match(r"\D*(\d+)", sub)
                if not dur_m:
                    continue
                value = 1.0 / int(dur_m.group(1))
                dots = sub.count(".")
                if dots:
                    value *= 2.0 - 0.5 ** dots
                r_bag[key + (round(value, 4),)] += 1
    return pitch_bags, rhythm_bags


def _match(gt_bags: dict, out_bags: dict) -> tuple[int, int]:
    total = matched = 0
    for key, gt_bag in gt_bags.items():
        out_bag = out_bags.get(key, Counter())
        total += sum(gt_bag.values())
        matched += sum((gt_bag & out_bag).values())
    return total, matched


def compare_file(mei_path: Path, krn_path: Path) -> tuple[int, int, int, int]:
    mei_pitch, mei_rhythm = mei_bags(mei_path)
    krn_pitch, krn_rhythm = kern_bags(krn_path)
    p_total, p_matched = _match(mei_pitch, krn_pitch)
    r_total, r_matched = _match(mei_rhythm, krn_rhythm)
    return p_total, p_matched, r_total, r_matched


def main():
    parser = argparse.ArgumentParser(
        description="Compare generated **kern files against groundtruth MEI, "
        "matching per-staff/per-measure multisets of sounding pitches "
        "(letter, octave, alteration) and of pitch+duration."
    )
    parser.add_argument("--mei_dir", type=str, default="data/validation-test/mei")
    parser.add_argument("--krn_dir", type=str, default="data/validation-test/krn-me")
    args = parser.parse_args()

    mei_dir, krn_dir = Path(args.mei_dir), Path(args.krn_dir)
    if not mei_dir.exists() or not krn_dir.exists():
        logger.error(f"Missing directory: {mei_dir if not mei_dir.exists() else krn_dir}")
        return

    print(f"{'file':<55} {'notes':>6} {'pitch':>7} {'rhythm':>7}")
    gp_total = gp_matched = gr_total = gr_matched = 0
    for mei_path in sorted(mei_dir.glob("*.mei")):
        krn_path = krn_dir / f"{mei_path.stem}.krn"
        if not krn_path.exists():
            logger.warning(f"No kern output for {mei_path.stem}")
            continue
        try:
            p_total, p_matched, r_total, r_matched = compare_file(mei_path, krn_path)
        except Exception as e:
            logger.error(f"Failed on {mei_path.stem}: {e}")
            continue
        gp_total += p_total; gp_matched += p_matched
        gr_total += r_total; gr_matched += r_matched
        p_pct = 100.0 * p_matched / p_total if p_total else float("nan")
        r_pct = 100.0 * r_matched / r_total if r_total else float("nan")
        print(f"{mei_path.stem:<55} {p_total:>6} {p_pct:>6.1f}% {r_pct:>6.1f}%")

    if gp_total:
        print(f"{'TOTAL':<55} {gp_total:>6} {100.0 * gp_matched / gp_total:>6.1f}% "
              f"{100.0 * gr_matched / gr_total if gr_total else float('nan'):>6.1f}%")


if __name__ == "__main__":
    main()
