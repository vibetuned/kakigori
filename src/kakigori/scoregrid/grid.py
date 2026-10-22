"""A system of music as a fixed-shape geometric matrix of categorical fields.

The learned decoder of the detection path denoises this matrix instead of
generating **kern text: every matrix decodes to valid kern, so there is no
tokenizer and no constrained decoding. Columns are horizontal x-bins across
the width of the system's nodes rather than abstract onset ordinals, so
alignment is local and timing is recovered from durations at emission.

Per system (``SystemGrid``):
  barline[b]              0 no barline, 1 a measure starts at bin b
  event[b, s]             pad / note / rest / mrest
  dur[b, s]               kern reciprocal duration (DURATIONS), 0 = none
  dots[b, s]              0-3
  grace[b, s]             0/1
  pitch[b, s, k]          diatonic step (octave * 7 + step) + 1, 0 = none;
                          k < K_MAX chord members, lowest first
  alter[b, s, k]          sounding alteration + 3 (-2..+2), 0 = none
  clef[s], key[s]         at the system's start (CLEFS, KEYS), 0 = none
  staff[s]                0 pad, 1 the staff exists

Ground truth pairs the MEI (``systems_from_mei``) with the page annotations:
note, chord, rest, mRest and measure ids are MEI xml:ids, so each event's
horizontal position and its MEI attributes pair up directly. Sounding
alterations follow musicset.evaluate (the initial key signature and the
measure's carry-over), so a ground-truth round trip scores what compare-kern
scores.
"""

# Future imports
from __future__ import annotations

# Standard library imports
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from fractions import Fraction
from dataclasses import field, dataclass

# Third party imports
import numpy as np
from musicset.evaluate import _note_alter, _keysig_alters

MEI = "{http://www.music-encoding.org/ns/mei}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

# 192 x-bins (~8 px each across a system's nodes) x 2 staves, chords <= 4
B_MAX, S_MAX, K_MAX = 192, 2, 4

EVENTS = ("pad", "note", "rest", "mrest")
PAD, NOTE, REST, MREST = range(len(EVENTS))
DURATIONS = (
    "none", "1", "2", "4", "8", "16", "32", "64", "128", "0", "00",
    "3", "6", "12", "24", "48", "96", "5", "10", "20", "40", "7", "14", "28", "56",
    "9", "18", "36", "11", "22", "13", "26",
)
DUR_INDEX = {d: i for i, d in enumerate(DURATIONS)}
MAX_DOTS = 3
STEPS = "cdefgab"
N_PITCH = 10 * 7 + 1  # octaves 0-9, + none
N_ALTER = 6           # none, -2..+2
CLEFS = ("none", "G2", "F4", "C3", "C4", "C1", "C2", "C5", "G1", "F3", "F5", "G2v", "G2^", "F4v", "other")
KEYS = ("none", "0", *(f"{n}s" for n in range(1, 8)), *(f"{n}f" for n in range(1, 8)), "other")
SPAN_EXCLUDE = ("page-margin", "system", "system-staff")


@dataclass
class Event:
    onset: Fraction          # quarter notes from the measure start
    grace: int               # 0, or the grace note's rank before its main note (1, 2, ...)
    kind: int                # NOTE / REST / MREST
    dur: str = "none"
    dots: int = 0
    pitches: list = field(default_factory=list)  # [(diatonic, alter)], lowest first
    id: str = ""             # MEI xml:id (pairs with its page annotation box)


@dataclass
class Measure:
    id: str
    index: int               # 1-based position in the MEI (compare-kern's measure number)
    staves: dict             # staff n -> [Event]
    clef: dict               # staff n -> clef at the measure's start
    key: dict                # staff n -> key at the measure's start


@dataclass
class SystemGrid:
    barline: np.ndarray
    event: np.ndarray
    dur: np.ndarray
    dots: np.ndarray
    grace: np.ndarray
    pitch: np.ndarray
    alter: np.ndarray
    clef: np.ndarray
    key: np.ndarray
    staff: np.ndarray
    # not predicted: the measures' numbers in the score, for emission
    numbers: list = field(default_factory=list)
    overflow: list = field(default_factory=list)

    FIELDS = ("barline", "event", "dur", "dots", "grace", "pitch", "alter", "clef", "key", "staff")

    @classmethod
    def empty(cls) -> SystemGrid:
        cell = (B_MAX, S_MAX)
        return cls(
            barline=np.zeros(B_MAX, np.int64),
            event=np.zeros(cell, np.int64),
            dur=np.zeros(cell, np.int64),
            dots=np.zeros(cell, np.int64),
            grace=np.zeros(cell, np.int64),
            pitch=np.zeros((*cell, K_MAX), np.int64),
            alter=np.zeros((*cell, K_MAX), np.int64),
            clef=np.zeros(S_MAX, np.int64),
            key=np.zeros(S_MAX, np.int64),
            staff=np.zeros(S_MAX, np.int64),
        )

    def arrays(self) -> dict:
        return {name: getattr(self, name) for name in self.FIELDS}


# --- MEI -> measures ---------------------------------------------------------


def _clef_name(shape: str | None, line: str | None, dis: str | None = None, place: str | None = None) -> str:
    if not shape:
        return "none"
    name = f"{shape}{line or ''}"
    if dis == "8":
        name += "v" if place == "below" else "^"
    return name if name in CLEFS else "other"


def _key_name(sig: str | None) -> str:
    if sig is None:
        return "none"
    if sig in ("0", "0s", "0f"):
        return "0"
    return sig if sig in KEYS else "other"


def _quarters(el, parents: dict, ppq: int) -> Fraction:
    """An event's length in quarter notes: dur.ppq when given, else dur, dots
    and tuplet ancestors."""
    if el.get("dur.ppq") is not None and ppq:
        return Fraction(int(el.get("dur.ppq")), ppq)
    dur = el.get("dur")
    base = {"long": Fraction(16), "breve": Fraction(8)}.get(dur)
    if base is None:
        if not dur or not dur.isdigit():
            return Fraction(0)
        base = Fraction(4, int(dur))
    dots = int(el.get("dots", 0))
    value = base * (2 - Fraction(1, 2 ** dots))
    anc = parents.get(el)
    while anc is not None:
        if anc.tag == MEI + "tuplet" and anc.get("num") and anc.get("numbase"):
            value *= Fraction(int(anc.get("numbase")), int(anc.get("num")))
        anc = parents.get(anc)
    return value


def _kern_dur(el, parents: dict) -> str:
    """The kern reciprocal duration of an event (tuplets scale it: a triplet
    eighth is 12), or 'none' when it has no kern value."""
    dur = el.get("dur")
    if dur in ("breve", "long"):
        return "0" if dur == "breve" else "00"
    if not dur or not dur.isdigit():
        return "none"
    value = Fraction(int(dur))
    anc = parents.get(el)
    while anc is not None:
        if anc.tag == MEI + "tuplet" and anc.get("num") and anc.get("numbase"):
            value *= Fraction(int(anc.get("num")), int(anc.get("numbase")))
        anc = parents.get(anc)
    if value.denominator != 1:
        return "none"
    name = str(value.numerator)
    return name if name in DUR_INDEX else "none"


def mei_measures(mei_path: Path) -> tuple[list[Measure], dict]:
    """The MEI's measures in order, with their events per staff and the clef
    and key in effect at their start; and the score header (meter, staves)."""
    root = ET.parse(mei_path).getroot()
    parents = {child: parent for parent in root.iter() for child in parent}

    # sounding alterations exactly as compare-kern computes them
    initial_keys, ppqs = {}, {}
    for sd in root.iter(MEI + "staffDef"):
        n = sd.get("n")
        if n not in initial_keys:
            ks = sd.find(MEI + "keySig")
            initial_keys[n] = _keysig_alters(ks.get("sig") if ks is not None else sd.get("key.sig"))
        if sd.get("ppq"):
            ppqs.setdefault(n, int(sd.get("ppq")))

    clef, key, meter = {}, {}, None
    key_all = "none"  # a scoreDef key signature: every staff without its own
    measures = []
    for el in root.iter():
        tag = el.tag.replace(MEI, "")
        if tag == "staffDef":
            n = el.get("n")
            c = el.find(MEI + "clef")
            if c is not None:
                clef[n] = _clef_name(c.get("shape"), c.get("line"), c.get("dis"), c.get("dis.place"))
            elif el.get("clef.shape"):
                clef[n] = _clef_name(el.get("clef.shape"), el.get("clef.line"), el.get("clef.dis"), el.get("clef.dis.place"))
            ks = el.find(MEI + "keySig")
            if ks is not None or el.get("key.sig") is not None:
                key[n] = _key_name(ks.get("sig") if ks is not None else el.get("key.sig"))
            ms = el.find(MEI + "meterSig")
            if meter is None and ms is not None and ms.get("count"):
                meter = (ms.get("count"), ms.get("unit"))
        elif tag == "scoreDef":
            ks = el.find(MEI + "keySig")
            if ks is not None or el.get("key.sig") is not None:
                key_all = _key_name(ks.get("sig") if ks is not None else el.get("key.sig"))
                key = {}
            ms = el.find(MEI + "meterSig")
            if meter is None and ms is not None and ms.get("count"):
                meter = (ms.get("count"), ms.get("unit"))
        elif tag == "clef":
            staff = parents.get(el)
            while staff is not None and staff.tag != MEI + "staff":
                staff = parents.get(staff)
            if staff is not None and parents.get(el).tag != MEI + "staffDef":
                clef[staff.get("n")] = _clef_name(el.get("shape"), el.get("line"), el.get("dis"), el.get("dis.place"))
        elif tag == "measure":
            staff_ns = [s.get("n") for s in el.findall(MEI + "staff")]
            keys = {n: key.get(n, key_all) for n in staff_ns}
            measures.append(_parse_measure(el, len(measures) + 1, parents, initial_keys, ppqs, dict(clef), keys))
    n_staves = max((len(m.staves) for m in measures), default=1)
    return measures, {"meter": meter, "staves": n_staves}


def _parse_measure(measure, index: int, parents: dict, initial_keys: dict, ppqs: dict,
                   clef: dict, key: dict) -> Measure:
    staves = {}
    for staff in measure.findall(MEI + "staff"):
        n = staff.get("n")
        carry, alters = {}, {}
        for note in staff.iter(MEI + "note"):
            if note.get("pname") and note.get("oct"):
                alters[note] = _note_alter(note, parents, initial_keys.get(n, {}), carry)
        events = []
        for layer in staff.findall(MEI + "layer"):
            t = Fraction(0)
            pending_grace = []
            for el in layer.iter():
                tag = el.tag.replace(MEI, "")
                if tag == "note" and parents[el].tag == MEI + "chord":
                    continue
                if tag not in ("note", "chord", "rest", "mRest", "space"):
                    continue
                if tag == "space":
                    t += _quarters(el, parents, ppqs.get(n, 0))
                    continue
                if tag == "mRest":
                    events.append(Event(onset=t, grace=0, kind=MREST, id=el.get(XML_ID, "")))
                    continue
                notes = [el] if tag == "note" else el.findall(MEI + "note") if tag == "chord" else []
                pitches = sorted(
                    (int(x.get("oct")) * 7 + STEPS.index(x.get("pname")), alters[x])
                    for x in notes if x in alters
                )
                ev = Event(
                    onset=t, grace=0, kind=REST if tag == "rest" else NOTE,
                    dur=_kern_dur(el, parents), dots=min(int(el.get("dots", 0)), MAX_DOTS),
                    pitches=pitches, id=el.get(XML_ID, ""),
                )
                if el.get("grace") is not None:
                    pending_grace.append(ev)
                    continue
                for rank, g in enumerate(reversed(pending_grace), start=1):
                    g.onset, g.grace = t, rank
                    events.append(g)
                pending_grace = []
                events.append(ev)
                t += _quarters(el, parents, ppqs.get(n, 0))
            for rank, g in enumerate(reversed(pending_grace), start=1):  # graces ending a layer
                g.onset, g.grace = t, rank
                events.append(g)
        staves[n] = events
    return Measure(id=measure.get(XML_ID), index=index, staves=staves, clef=clef, key=key)


# --- systems -------------------------------------------------------------------


def score_systems(ann_dir: Path, score: str, page_files: list | None = None) -> tuple[list[tuple], dict]:
    """The score's systems in reading order as (page index, system box,
    horizontal node-centre span, measure ids left to right), plus the score's
    annotation boxes by MEI xml:id."""
    if page_files is None:
        page_files = sorted(
            (p.name for p in ann_dir.glob(f"{score}_page*.json")),
            key=lambda name: int(name.rsplit("_page", 1)[1].split(".")[0]),
        )
    systems, ann_by_id = [], {}
    for page, name in enumerate(page_files):
        anns = json.loads((ann_dir / name).read_text())["annotations"]
        for a in anns:
            if "id" in a:
                ann_by_id[a["id"]] = a["bbox"]
        boxes = sorted((a for a in anns if a["class"] == "system"), key=lambda a: a["bbox"][1])
        measures = [a for a in anns if a["class"] == "measure"]
        for box in boxes:
            _, y1, _, y2 = box["bbox"]
            inside = [m for m in measures if y1 <= (m["bbox"][1] + m["bbox"][3]) / 2 <= y2]
            in_nodes = [
                a for a in anns
                if y1 <= (a["bbox"][1] + a["bbox"][3]) / 2 <= y2 and a["class"] not in SPAN_EXCLUDE
            ]
            cxs = [0.5 * (a["bbox"][0] + a["bbox"][2]) for a in in_nodes]
            x_span = (min(cxs, default=box["bbox"][0]), max(cxs, default=box["bbox"][2]))
            systems.append((
                page, box["bbox"], x_span,
                [m["id"] for m in sorted(inside, key=lambda m: m["bbox"][0])],
            ))
    return systems, ann_by_id


def grid_from_measures(
    measures: list[Measure],
    n_staves: int,
    ann_by_id: dict | None = None,
    x_span: tuple[float, float] = (0.0, 1.0),
) -> SystemGrid:
    """One system's geometric x-bin matrix over ``x_span = (x1, x2)``; what
    does not fit is listed in ``overflow``."""
    g = SystemGrid.empty()
    g.staff[:min(n_staves, S_MAX)] = 1
    if n_staves > S_MAX:
        g.overflow.append(f"{n_staves} staves")
    staff_ns = [str(s + 1) for s in range(min(n_staves, S_MAX))]
    if measures:
        first = measures[0]
        for s, n in enumerate(staff_ns):
            g.clef[s] = CLEFS.index(first.clef.get(n, "none"))
            g.key[s] = KEYS.index(first.key.get(n, "none"))

    ann_by_id = ann_by_id or {}
    x1, x2 = x_span
    w = max(float(x2 - x1), 1.0)
    min_b = 0
    for mi, measure in enumerate(measures):
        mbbox = ann_by_id.get(measure.id, (x1, 0.0, x2, 0.0))
        b_bar = 0 if mi == 0 else max(min_b, int(np.clip((mbbox[0] - x1) / w * B_MAX, 0, B_MAX - 1)))
        if b_bar >= B_MAX:
            g.overflow.append(f"measure {measure.index}: barline at bin {b_bar}")
            break
        g.barline[b_bar] = 1
        g.numbers.append(measure.index)
        max_ev_b = b_bar
        for s, n in enumerate(staff_ns):
            evs = measure.staves.get(n, [])
            with_x = []
            for idx_e, ev in enumerate(evs):
                eb = ann_by_id.get(ev.id, mbbox)
                cx = 0.5 * (eb[0] + eb[2])
                with_x.append((ev.onset, -ev.grace, cx, idx_e, ev))
            with_x.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            for _, _, cx, _, ev in with_x:
                b = int(np.clip((cx - x1) / w * B_MAX, b_bar, B_MAX - 1))
                while b < B_MAX and g.event[b, s] != PAD:
                    b += 1
                if b >= B_MAX:
                    g.overflow.append(f"measure {measure.index}: bin overflow")
                    continue
                g.event[b, s] = ev.kind
                g.dur[b, s] = DUR_INDEX[ev.dur]
                g.dots[b, s] = ev.dots
                g.grace[b, s] = int(ev.grace > 0)
                if len(ev.pitches) > K_MAX:
                    g.overflow.append(f"measure {measure.index}: chord of {len(ev.pitches)}")
                in_range = [(d, a) for d, a in ev.pitches if 0 <= d < N_PITCH - 1 and -2 <= a <= 2]
                if len(in_range) < len(ev.pitches):
                    # e.g. a negative octave in a source MEI
                    g.overflow.append(f"measure {measure.index}: pitch out of range")
                for k, (diatonic, alter) in enumerate(in_range[:K_MAX]):
                    g.pitch[b, s, k] = diatonic + 1
                    g.alter[b, s, k] = alter + 3
                max_ev_b = max(max_ev_b, b)
        min_b = max(b_bar + 1, max_ev_b + 1)
    return g


def systems_from_mei(mei_path: Path, ann_dir: Path, page_files: list | None = None) -> tuple[list[SystemGrid], dict]:
    """The score's ground-truth system matrices and header: ``boxes`` holds
    each system's (page index, box), ``system_measures`` the MEI Measure
    lists per system, ``ann_by_id`` the annotation boxes by id, and
    ``unplaced`` the measures no system annotation holds."""
    measures, header = mei_measures(mei_path)
    by_id = {m.id: m for m in measures}
    sys_specs, ann_by_id = score_systems(ann_dir, mei_path.stem, page_files)
    grids, placed, header["boxes"], header["system_measures"], header["ann_by_id"] = [], set(), [], [], ann_by_id
    for page, box, x_span, ids in sys_specs:
        system = [by_id[i] for i in ids if i in by_id]
        placed.update(m.id for m in system)
        grids.append(grid_from_measures(system, header["staves"], ann_by_id, x_span))
        header["boxes"].append((page, box))
        header["system_measures"].append(system)
    header["unplaced"] = [m.index for m in measures if m.id not in placed]
    return grids, header


# --- matrix -> **kern ------------------------------------------------------------


def _kern_pitch(diatonic: int, alter: int) -> str:
    octave, step = divmod(diatonic, 7)
    letter = STEPS[step]
    name = letter * (octave - 3) if octave >= 4 else letter.upper() * (4 - octave)
    return name + {-2: "--", -1: "-", 0: "", 1: "#", 2: "##"}[alter]


def _dur_quarters(dur_idx: int, dots: int) -> Fraction:
    dur = DURATIONS[dur_idx]
    base = {"00": Fraction(16), "0": Fraction(8)}.get(dur)
    if base is None:
        base = Fraction(4, int(dur)) if dur.isdigit() else Fraction(1)
    return base * (2 - Fraction(1, 2 ** int(dots)))


def _kern_cell(g: SystemGrid, b: int, s: int) -> str:
    kind = g.event[b, s]
    if kind == PAD:
        return "."
    if kind == MREST:
        return "1r"
    dur = DURATIONS[g.dur[b, s]]
    dur = "4" if dur == "none" else dur
    rhythm = dur + "." * int(g.dots[b, s])
    if kind == REST:
        return rhythm + "r"
    grace = "q" if g.grace[b, s] else ""
    tokens = [
        # a decoded pitch may come without its alteration: unaltered
        rhythm + _kern_pitch(int(p) - 1, int(a) - 3 if a > 0 else 0) + grace
        for p, a in zip(g.pitch[b, s], g.alter[b, s], strict=True)
        if p > 0
    ]
    return " ".join(tokens) if tokens else rhythm + "r"


def _key_token(key: str) -> str:
    if key in ("none", "other", "0"):
        return "*k[]"
    n, kind = int(key[:-1]), key[-1]
    order = "fcgdaeb" if kind == "s" else "beadgcf"
    return "*k[" + "".join(c + ("#" if kind == "s" else "-") for c in order[:n]) + "]"


def grids_to_kern(grids: list[SystemGrid], meter=None, numbers: list | None = None) -> str:
    """A score's **kern from its system matrices in reading order. Within each
    measure (between consecutive barline bins), each staff's events are read
    in bin order, their durations accumulated into onsets (resetting at each
    barline), and onsets merged across staves into **kern lines."""
    n_staves = max((int(g.staff.sum()) for g in grids), default=1) or 1
    cols = list(range(n_staves - 1, -1, -1))  # staff index per spine, lowest first

    def line(cells) -> str:
        return "\t".join(cells)

    first = grids[0] if grids else SystemGrid.empty()
    out = [
        line("**kern" for _ in cols),
        line(f"*staff{s + 1}" for s in cols),
        line("*clef" + ({"G2v": "Gv2", "G2^": "G^2", "F4v": "Fv4"}.get(c, c) if c not in ("none", "other") else "G2")
             for c in (CLEFS[first.clef[s]] for s in cols)),
        line(_key_token(KEYS[first.key[s]]) for s in cols),
    ]
    if meter and meter[0]:
        out.append(line(f"*M{meter[0]}/{meter[1]}" for _ in cols))
    count = 0
    for gi, g in enumerate(grids):
        bars = [b for b in range(B_MAX) if g.barline[b]]
        if not bars:
            if (g.event[:, :n_staves] != PAD).any():
                bars = [0]
            else:
                continue
        for mi, b_start in enumerate(bars):
            b0 = 0 if mi == 0 else b_start
            b1 = bars[mi + 1] if mi + 1 < len(bars) else B_MAX
            count += 1
            number = numbers[gi][mi] if numbers is not None else (g.numbers[mi] if mi < len(g.numbers) else count)
            out.append(line(f"={number}" for _ in cols))
            per_staff = {}
            all_keys = set()
            for s in range(n_staves):
                t = Fraction(0)
                pending_grace = []
                s_map = {}
                for b in range(b0, b1):
                    kind = int(g.event[b, s])
                    if kind == PAD:
                        continue
                    tok = _kern_cell(g, b, s)
                    if kind == NOTE and g.grace[b, s]:
                        pending_grace.append(tok)
                        continue
                    for rank, gtok in enumerate(reversed(pending_grace), start=1):
                        k = (t, -rank, 0)
                        while k in s_map:
                            k = (k[0], k[1], k[2] + 1)
                        s_map[k] = gtok
                        all_keys.add(k)
                    pending_grace = []
                    k = (t, 0, 0)
                    while k in s_map:
                        k = (k[0], k[1], k[2] + 1)
                    s_map[k] = tok
                    all_keys.add(k)
                    t += Fraction(4) if kind == MREST else _dur_quarters(int(g.dur[b, s]), int(g.dots[b, s]))
                for rank, gtok in enumerate(reversed(pending_grace), start=1):
                    k = (t, -rank, 0)
                    while k in s_map:
                        k = (k[0], k[1], k[2] + 1)
                    s_map[k] = gtok
                    all_keys.add(k)
                per_staff[s] = s_map
            for k in sorted(all_keys):
                out.append(line(per_staff[s].get(k, ".") for s in cols))
    out.append(line("==" for _ in cols))
    out.append(line("*-" for _ in cols))
    return "\n".join(out) + "\n"
