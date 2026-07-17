"""Spatial repair heuristics for predicted score graphs.

The GNN owns the genuinely relational decisions (temporal order, sync,
which note a slur belongs to). This module owns the *geometric certainties*
that measured recall showed the prediction path losing — sub-glyph
containment (note→notehead→stem, mRest→restWhole, meterSig→digits,
keySig→accidentals), staff-level context attachment, and orphaned events.

Every repair is GUARDED: it only adds an edge when the child has no parent
of the appropriate kind, so model predictions are never overridden. All
geometry runs in the original annotation coordinate space.

This is the first of the planned serializer-hardening iterations; later
ones may recover missing systems from layer/measure evidence and missing
measures from note clusters.
"""

STEM_CLASSES = {"stem4", "stem8", "stem16", "stem32"}
FLAG_CLASSES = {"flag8thUp", "flag8thDown", "flag16thUp", "flag16thDown"}
NOTEHEAD_CLASSES = {"noteheadBlack", "noteheadHalf", "noteheadWhole"}
EVENT_CLASSES = {
    "note", "chord", "mRest", "restWhole", "restHalf", "restQuarter",
    "rest8th", "rest16th", "rest32nd",
}
MODIFIER_CLASSES = {
    "accidentalSharp", "accidentalFlat", "accidentalNatural",
    "accidentalDoubleSharp", "accidentalDoubleFlat", "dots",
}
STAFF_CONTEXT_PREFIXES = ("clef", "keySig", "meterSig")


def _center(node):
    b = node["bbox"]
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _contains(outer, inner_cx, inner_cy, pad=2.0):
    b = outer["bbox"]
    return (b[0] - pad <= inner_cx <= b[2] + pad
            and b[1] - pad <= inner_cy <= b[3] + pad)


def _x_overlap(a, b, pad=2.0):
    return a["bbox"][0] - pad <= b["bbox"][2] and b["bbox"][0] - pad <= a["bbox"][2]


def _y_gap(a, b):
    if a["bbox"][1] > b["bbox"][3]:
        return a["bbox"][1] - b["bbox"][3]
    if b["bbox"][1] > a["bbox"][3]:
        return b["bbox"][1] - a["bbox"][3]
    return 0.0


ALL_REPAIRS = frozenset(range(1, 10))


# Empirically tuned on validation-small (see docs/graph.md): repair 1 uses
# a kind-specific guard (heads under beams still need their note parent) and
# FIRST-MATCH owner assignment; all other repairs keep the conservative
# any-parent guard. Both nearest-center owners and kind-guards everywhere
# were tried and measured 6 points worse end-to-end.
DEFAULT_LOOSE = frozenset(range(2, 10))


def repair_page_edges(page_nodes: list, edges: list, enabled=ALL_REPAIRS,
                      loose=DEFAULT_LOOSE) -> int:
    """Extend `edges` (list of (u_id, v_id, cls)) in place. Returns the
    number of edges added. `enabled` selects which numbered repairs run;
    `loose` selects which use the conservative any-parent guard — both are
    exposed for ablation experiments."""
    by_class = {}
    for n in page_nodes:
        by_class.setdefault(n["class"], []).append(n)

    def of(*classes):
        out = []
        for c in classes:
            out.extend(by_class.get(c, []))
        return out

    def of_prefix(*prefixes):
        out = []
        for c, nodes in by_class.items():
            if c.startswith(prefixes):
                out.extend(nodes)
        return out

    id_to_class = {n["id"]: n["class"] for n in page_nodes}
    has_parent = {v for _, v, _ in edges}
    # Ownership means STRUCTURAL/MODIFIER parenthood; temporal (3) and sync
    # (4/5) neighbors must not satisfy kind guards
    struct_parents_of = {}
    for u, v, cls in edges:
        if cls in (1, 2):
            struct_parents_of.setdefault(v, set()).add(u)
    added = 0

    def guarded(repair_no, child_id, *kinds):
        """Guard for repair `repair_no`: kind-specific by default, or the
        conservative any-parent check when the repair is in `loose`."""
        if repair_no in loose:
            return child_id in has_parent
        return has_parent_kind(child_id, *kinds)

    def has_parent_kind(child_id, *kinds):
        """True if the child already has a structural parent of one of the
        EXACT classes. Exactness matters: 'note' must not match
        'noteheadBlack' (a prefix check silently disabled repair 1 for every
        head inside a temporal notehead chain)."""
        return any(
            id_to_class.get(p) in kinds
            for p in struct_parents_of.get(child_id, ())
        )

    def add(u, v, cls):
        nonlocal added
        edges.append((u["id"], v["id"], cls))
        has_parent.add(v["id"])
        if cls in (1, 2):
            struct_parents_of.setdefault(v["id"], set()).add(u["id"])
        added += 1

    # --- 1. note -> notehead (bbox containment; a note box covers its head)
    notes = of("note")
    for head in of(*NOTEHEAD_CLASSES):
        if 1 not in enabled:
            break
        if guarded(1, head["id"], "note"):
            continue
        hx, hy = _center(head)
        # First containing note in annotation order. Counterintuitively this
        # measured better than nearest-center: chord-member note boxes all
        # span the shared stem, so centers are poor ownership signals, while
        # annotation order tracks document order.
        owner = next((n for n in notes if _contains(n, hx, hy)), None)
        if owner is not None:
            add(owner, head, 1)

    # --- 2. notehead -> stem (x-adjacent, vertically touching)
    heads = of(*NOTEHEAD_CLASSES)
    for stem in of(*STEM_CLASSES):
        if 2 not in enabled:
            break
        if guarded(2, stem["id"], *NOTEHEAD_CLASSES, "note"):
            continue
        best, best_gap = None, float("inf")
        for head in heads:
            if not _x_overlap(stem, head, pad=4.0):
                continue
            gap = _y_gap(stem, head)
            if gap < best_gap:
                best_gap, best = gap, head
        if best is not None and best_gap < 12.0:
            add(best, stem, 1)

    # --- 3. flags attach like stems (nearest x-overlapping notehead)
    for flag in of(*FLAG_CLASSES):
        if 3 not in enabled:
            break
        if guarded(3, flag["id"], *NOTEHEAD_CLASSES, "note", *STEM_CLASSES):
            continue
        best, best_gap = None, float("inf")
        for head in heads:
            if not _x_overlap(flag, head, pad=6.0):
                continue
            gap = _y_gap(flag, head)
            if gap < best_gap:
                best_gap, best = gap, head
        if best is not None and best_gap < 40.0:
            add(best, flag, 1)

    # --- 4. mRest -> restWhole (co-located glyph pair)
    for rest in of("restWhole"):
        if 4 not in enabled:
            break
        if guarded(4, rest["id"], "mRest"):
            continue
        rx, ry = _center(rest)
        owner = next((m for m in of("mRest") if _contains(m, rx, ry, pad=6.0)), None)
        if owner is not None:
            add(owner, rest, 1)

    # --- 5. meterSig -> timeSig digits / common symbols (containment)
    metersigs = of("meterSig")
    for digit in of_prefix("timeSig"):
        if 5 not in enabled:
            break
        if guarded(5, digit["id"], "meterSig"):
            continue
        dx, dy = _center(digit)
        owner = next((m for m in metersigs if _contains(m, dx, dy, pad=4.0)), None)
        if owner is not None:
            add(owner, digit, 1)

    # --- 6. keySig -> keyAccid glyphs (containment)
    keysigs = of("keySig")
    for accid in of_prefix("keyAccid"):
        if 6 not in enabled:
            break
        if guarded(6, accid["id"], "keySig"):
            continue
        ax, ay = _center(accid)
        owner = next((k for k in keysigs if _contains(k, ax, ay, pad=4.0)), None)
        if owner is not None:
            add(owner, accid, 1)

    # --- 7. staff -> clef/keySig/meterSig context (vertical containment)
    staves = of("staff")
    for ctx in of_prefix(*STAFF_CONTEXT_PREFIXES):
        if 7 not in enabled:
            break
        if guarded(7, ctx["id"], "staff"):
            continue
        _, cy = _center(ctx)
        owner = next(
            (s for s in staves
             if s["bbox"][1] - 20 <= cy <= s["bbox"][3] + 20
             and s["bbox"][0] - 30 <= _center(ctx)[0] <= s["bbox"][2] + 30),
            None,
        )
        if owner is not None:
            add(owner, ctx, 1)

    # --- 8. orphan events -> containing staff (keeps them on the timeline).
    # A staff-reachability variant was tried instead of the parentless test
    # and measured no better; keep the simple form.
    for event in of(*EVENT_CLASSES):
        if 8 not in enabled:
            break
        if event["id"] in has_parent:
            continue
        ex, ey = _center(event)
        owner = next((s for s in staves if _contains(s, ex, ey, pad=10.0)), None)
        if owner is not None:
            add(owner, event, 1)

    # --- 9. orphan modifiers -> nearest event (parser-style nearest note)
    anchors = of("note", "chord", *NOTEHEAD_CLASSES)
    for mod in of(*MODIFIER_CLASSES):
        if 9 not in enabled:
            break
        if mod["id"] in has_parent:
            continue
        mx, my = _center(mod)
        best, best_d = None, float("inf")
        for a in anchors:
            ax, ay = _center(a)
            d = ((ax - mx) ** 2 + (ay - my) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, a
        if best is not None and best_d < 60.0:
            add(best, mod, 2)

    return added
