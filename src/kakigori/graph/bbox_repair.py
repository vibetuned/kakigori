"""Detection-level box repairs for the detection-driven pipeline.

graph_repair.py restores EDGES the GNN missed; this module restores the
structural BOXES the detector missed, before the graph is ever built. The
serializer's layout logic runs entirely on structure nodes (system,
measure, staff cells, system-staff rows) whose detection AP is mediocre
(0.52-0.77) compared to the glyphs they contain — one missing staff cell
silently drops a measure of content, one missing system drops a whole band.

Every repair only SYNTHESIZES a box whose geometry is derivable from other
detections (unions of members, intersections of spans) and is guarded: it
never moves or removes a detector prediction. Synthesized nodes carry
score 0.0 so they remain distinguishable downstream.

Repairs (numbered, toggleable via `enabled`, measured on validation-small):
  1. system from uncovered y-bands of measures + system-staff rows
  2. measure from a stack of staff cells no measure x-covers
  3. staff cell at every (measure x-span × system-staff row) intersection
     that has no cell — the serializer needs a cell per part per measure
  4. measures from barlines: boundaries = x-clusters of barlines within a
     system band (barlines are the best-detected structural signal, ~0.86
     AP, vs 64%-missed measures at inference thresholds)
"""

STRUCTURE_CLASSES = ("system", "system-staff", "measure", "staff")
BARLINE_CLASSES = ("barlineSingle", "barlineDouble", "barlineFinal", "barlineHeavy")

# A fixed-line clef's bbox EXACTLY determines its 5-line staff row —
# verovio glyph geometry is deterministic (std 0.000 over 500 GT pairs):
#   row_y1 = clef_cy + A*clef_h,  row_y2 = clef_cy + B*clef_h
# clefC is excluded: its offset depends on which line the clef sits on,
# which is only knowable FROM the row.
CLEF_ROW_CONSTANTS = {
    "clefG":    (-0.314, 0.272),
    "clefG8vb": (-0.336, 0.180),
    "clefG8va": (-0.217, 0.299),
    "clefF":    (-0.510, 0.714),
}

ALL_BOX_REPAIRS = frozenset({1, 2, 3, 4, 5, 6})


def _y_overlap(a_bbox, b_bbox) -> float:
    return min(a_bbox[3], b_bbox[3]) - max(a_bbox[1], b_bbox[1])


def _x_overlap(a_bbox, b_bbox) -> float:
    return min(a_bbox[2], b_bbox[2]) - max(a_bbox[0], b_bbox[0])


def _union(bboxes) -> list:
    return [
        min(b[0] for b in bboxes), min(b[1] for b in bboxes),
        max(b[2] for b in bboxes), max(b[3] for b in bboxes),
    ]


def _bands(nodes: list) -> list:
    """Cluster nodes into horizontal bands by y-overlap (greedy sweep)."""
    bands = []
    for n in sorted(nodes, key=lambda n: n["bbox"][1]):
        for band in bands:
            if _y_overlap(band["bbox"], n["bbox"]) > 0:
                band["members"].append(n)
                band["bbox"] = _union([band["bbox"], n["bbox"]])
                break
        else:
            bands.append({"members": [n], "bbox": list(n["bbox"])})
    return bands


def repair_page_boxes(page_nodes: list, page_prefix: str = "p0",
                      enabled=ALL_BOX_REPAIRS) -> int:
    """Extend `page_nodes` in place with synthesized structure nodes.
    Returns the number of boxes added."""
    added = 0

    def add(cls: str, bbox: list):
        nonlocal added
        page_nodes.append({
            "id": f"{page_prefix}bbr{added}_{cls}",
            "class": cls,
            "score": 0.0,
            "bbox": [round(v, 1) for v in bbox],
            "cx": (bbox[0] + bbox[2]) / 2.0,
            "cy": (bbox[1] + bbox[3]) / 2.0,
        })
        added += 1

    def of(cls):
        return [n for n in page_nodes if n.get("class") == cls]

    # --- 6. structure FP filtering — runs FIRST. False-positive rows/cells
    # are as damaging as misses: one phantom system-staff row inflates the
    # global part count, turning EVERY real system into a "reduced" one
    # (spurious spine + rest-padding through the whole piece). Standard
    # class-aware post-NMS filtering: fragment rows (a real row spans its
    # page's full music width), near-duplicate structure boxes (keep the
    # higher score), and cells on no surviving row.
    if 6 in enabled:
        rows = of("system-staff")
        if rows:
            max_w = max(r["bbox"][2] - r["bbox"][0] for r in rows)
            frag = {
                r["id"] for r in rows
                if r["bbox"][2] - r["bbox"][0] < 0.5 * max_w
            }
            page_nodes[:] = [n for n in page_nodes if n["id"] not in frag]

        # Systems and rows tile the page vertically — they NEVER y-overlap.
        # Greedy band-NMS: keep by descending score, drop anything that
        # y-overlaps a kept box (>30% of the smaller height). This kills
        # both near-duplicates and low-score "blanket" boxes spanning two
        # real bands (a 0.30-score system covering two systems turned a
        # single-staff piece into a phantom two-part score).
        for cls in ("system", "system-staff"):
            kept, drop = [], set()
            for n in sorted(of(cls), key=lambda n: -n.get("score", 0.0)):
                h = n["bbox"][3] - n["bbox"][1]
                clash = any(
                    _y_overlap(n["bbox"], k["bbox"])
                    > 0.3 * min(h, k["bbox"][3] - k["bbox"][1])
                    for k in kept
                )
                if clash:
                    drop.add(n["id"])
                else:
                    kept.append(n)
            if drop:
                page_nodes[:] = [n for n in page_nodes if n["id"] not in drop]

        # Measures and cells can share bands but not areas — drop the lower
        # score of any pair overlapping >70% of the SMALLER box in both axes
        for cls in ("measure", "staff"):
            group = sorted(of(cls), key=lambda n: -n.get("score", 0.0))
            drop = set()
            for i, hi in enumerate(group):
                if hi["id"] in drop:
                    continue
                for lo in group[i + 1:]:
                    if lo["id"] in drop:
                        continue
                    yo = _y_overlap(hi["bbox"], lo["bbox"])
                    xo = _x_overlap(hi["bbox"], lo["bbox"])
                    min_h = min(hi["bbox"][3] - hi["bbox"][1],
                                lo["bbox"][3] - lo["bbox"][1])
                    min_w = min(hi["bbox"][2] - hi["bbox"][0],
                                lo["bbox"][2] - lo["bbox"][0])
                    if yo > 0.7 * min_h and xo > 0.7 * min_w:
                        drop.add(lo["id"])
            if drop:
                page_nodes[:] = [n for n in page_nodes if n["id"] not in drop]

        surviving_rows = of("system-staff")
        if surviving_rows:
            orphan = {
                c["id"] for c in of("staff")
                if not any(
                    r["bbox"][1] - 10 <= c["cy"] <= r["bbox"][3] + 10
                    for r in surviving_rows
                )
            }
            page_nodes[:] = [n for n in page_nodes if n["id"] not in orphan]

    # --- 5. system-staff row from a fixed-line clef with no row under it.
    # Runs FIRST: rows feed pitch geometry (staff_space) and everything
    # downstream. Clefs detect at ~1.0 AP and their bbox determines the row
    # exactly (CLEF_ROW_CONSTANTS); the x-extent comes from the page's
    # detected rows (uniform per page) or the clef itself as fallback.
    if 5 in enabled:
        rows = of("system-staff")
        row_x1 = min((r["bbox"][0] for r in rows), default=None)
        row_x2 = max((r["bbox"][2] for r in rows), default=None)
        for clef in [n for n in page_nodes if n.get("class") in CLEF_ROW_CONSTANTS]:
            a, b = CLEF_ROW_CONSTANTS[clef["class"]]
            h = clef["bbox"][3] - clef["bbox"][1]
            y1, y2 = clef["cy"] + a * h, clef["cy"] + b * h
            covered = any(
                _y_overlap([0, y1, 0, y2], r["bbox"]) > 0.5 * (y2 - y1)
                for r in of("system-staff")
            )
            if covered or y2 - y1 <= 0:
                continue
            x1 = row_x1 if row_x1 is not None else clef["bbox"][0]
            x2 = row_x2 if row_x2 is not None else clef["bbox"][0] + 12 * h
            add("system-staff", [x1, y1, x2, y2])

    # --- 1. system from uncovered bands of measures, rows, and staff cells.
    # A band of structure that no system vertically covers is a missed
    # system — its bbox is simply the union of the band's members.
    if 1 in enabled:
        systems = of("system")
        content = of("measure") + of("system-staff") + of("staff")
        uncovered = [
            n for n in content
            if not any(
                s["bbox"][1] <= (n["bbox"][1] + n["bbox"][3]) / 2.0 <= s["bbox"][3]
                for s in systems
            )
        ]
        for band in _bands(uncovered):
            # a lone stray box is more likely detector noise than a system
            if len(band["members"]) >= 2:
                add("system", band["bbox"])

    # --- 4. measures from barlines. Runs BEFORE 2/3 so downstream repairs
    # see the recovered measures. Within each system band, barline x-centers
    # cluster into measure boundaries (they repeat per row at the same x);
    # every span between consecutive boundaries — plus system-start to the
    # first boundary — that no detected measure covers becomes a measure.
    if 4 in enabled:
        measures = of("measure")
        for system in of("system"):
            s_bbox = system["bbox"]
            bars = [
                n for n in page_nodes
                if n.get("class") in BARLINE_CLASSES
                and s_bbox[1] <= n["cy"] <= s_bbox[3]
            ]
            if not bars:
                continue
            xs = sorted(b["cx"] for b in bars)
            boundaries = [s_bbox[0]]
            for x in xs:
                if x - boundaries[-1] > 15.0:
                    boundaries.append(x)
            sys_measures = [
                m for m in measures
                if _y_overlap(m["bbox"], s_bbox) > 0
            ]
            for b1, b2 in zip(boundaries, boundaries[1:]):
                if b2 - b1 < 25.0:
                    continue
                mid = (b1 + b2) / 2.0
                if any(m["bbox"][0] <= mid <= m["bbox"][2] for m in sys_measures):
                    continue
                add("measure", [b1, s_bbox[1], b2, s_bbox[3]])

    # --- 2. measure from a stack of staff cells that no measure x-covers
    # (within the same band). The measure bbox is the union of the stack.
    if 2 in enabled:
        measures = of("measure")
        cells = of("staff")
        orphan_cells = [
            c for c in cells
            if not any(
                m["bbox"][0] <= c["cx"] <= m["bbox"][2]
                and _y_overlap(m["bbox"], c["bbox"]) > 0
                for m in measures
            )
        ]
        # group orphans into x-overlapping stacks
        stacks = []
        for c in sorted(orphan_cells, key=lambda n: n["bbox"][0]):
            for stack in stacks:
                if _x_overlap(stack["bbox"], c["bbox"]) > 0.5 * (c["bbox"][2] - c["bbox"][0]):
                    stack["members"].append(c)
                    stack["bbox"] = _union([stack["bbox"], c["bbox"]])
                    break
            else:
                stacks.append({"members": [c], "bbox": list(c["bbox"])})
        for stack in stacks:
            add("measure", stack["bbox"])

    # --- 3. staff cell at every measure x-span × row y-span intersection
    # with no existing cell. Geometric certainty: a measure crosses every
    # visible row of its system; the y comes from the DETECTED row (tight
    # 5-line box), so no pitch-critical geometry is invented.
    if 3 in enabled:
        rows = of("system-staff")
        systems = of("system")
        cells = of("staff")
        for measure in of("measure"):
            m_bbox = measure["bbox"]
            m_cy = (m_bbox[1] + m_bbox[3]) / 2.0
            system = next(
                (s for s in systems
                 if s["bbox"][1] <= m_cy <= s["bbox"][3]), None,
            )
            for row in rows:
                r_bbox = row["bbox"]
                r_cy = (r_bbox[1] + r_bbox[3]) / 2.0
                # the row must belong to the measure's system band
                if not (m_bbox[1] <= r_cy <= m_bbox[3]):
                    continue
                if system is not None and not (
                        system["bbox"][1] <= r_cy <= system["bbox"][3]):
                    continue
                has_cell = any(
                    m_bbox[0] <= c["cx"] <= m_bbox[2]
                    and r_bbox[1] - 5 <= c["cy"] <= r_bbox[3] + 5
                    for c in cells
                )
                if not has_cell:
                    add("staff", [m_bbox[0], r_bbox[1], m_bbox[2], r_bbox[3]])

    return added
