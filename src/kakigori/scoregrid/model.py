"""The scoregrid decoder: a system's detected nodes in, its SystemGrid out,
by masked discrete flow matching over the grid's categorical fields.

Every field value is a variable; training masks each one with probability
t ~ U(0.05, 1) and predicts the masked ones from the rest and the nodes.
Sampling starts fully masked and unmasks the most confident predictions over
K steps (K = 1: a one-shot classifier, the ablation that tells whether the
iterative refinement earns its keep).

Tokens: one per geometric cell (x-bin, staff) carrying its 12 event fields
and the bin's barline state, plus one per staff (``clef``, ``key``, ``staff``).
Nodes and grid cells share the x-bin positional embedding ``pos_b`` over the
system's horizontal node span. Each cell also starts from the encoded glyph
nodes drawn in it (``local_context``: mean-pooled by centre bin and row), so
the alignment of cells to nodes is given, not learned: with the positional
embedding alone, a model fit on the very systems it was scored on still put
its notes in the wrong bins (P(note) 0.26 on true note cells, 0.09 elsewhere).
"""

# Future imports
from __future__ import annotations

# Standard library imports
import math

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local folder imports
from .grid import (
    PAD,
    KEYS,
    NOTE,
    REST,
    B_MAX,
    CLEFS,
    K_MAX,
    MREST,
    S_MAX,
    EVENTS,
    N_ALTER,
    N_PITCH,
    MAX_DOTS,
    DURATIONS,
)

# cell columns: event, dur, dots, grace, pitch x K_MAX, alter x K_MAX
CELL_VOCAB = [len(EVENTS), len(DURATIONS), MAX_DOTS + 1, 2] + [N_PITCH] * K_MAX + [N_ALTER] * K_MAX
EVENT, DUR, DOTS, GRACE = 0, 1, 2, 3
PITCH0, ALTER0 = 4, 4 + K_MAX
N_CELL = len(CELL_VOCAB)
STAFF_VOCAB = [len(CLEFS), len(KEYS), 2]  # clef, key, staff
BARLINE_VOCAB = 2
MAX_DETECTED_MEASURES = 32
MAX_ROWS = 4
LOCAL_MAX_BINS = 4   # nodes pooled into their cell: glyphs, not measures/layers/beams
LOCAL_MAX_COUNT = 3


def grid_tensors(target: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A SystemGrid's arrays as (cells [B, S, 12], barline [B], staff [S, 3])."""
    cells = torch.cat([
        torch.as_tensor(target["event"])[..., None],
        torch.as_tensor(target["dur"])[..., None],
        torch.as_tensor(target["dots"])[..., None],
        torch.as_tensor(target["grace"])[..., None],
        torch.as_tensor(target["pitch"]),
        torch.as_tensor(target["alter"]),
    ], dim=-1).long()
    staff = torch.stack([torch.as_tensor(target[k]) for k in ("clef", "key", "staff")], dim=-1).long()
    return cells, torch.as_tensor(target["barline"]).long(), staff


def cell_loss_mask(cells: torch.Tensor) -> torch.Tensor:
    """Which cell fields carry information: the event always; duration and
    dots for notes and rests; grace and pitches for notes; alterations for
    present chord members (pitch > 0)."""
    event = cells[..., EVENT]
    sounding = (event == NOTE) | (event == REST)
    is_note = event == NOTE
    mask = torch.zeros_like(cells, dtype=torch.bool)
    mask[..., EVENT] = True
    mask[..., DUR] = sounding
    mask[..., DOTS] = sounding
    mask[..., GRACE] = is_note
    mask[..., PITCH0:ALTER0] = is_note.unsqueeze(-1)
    mask[..., ALTER0:] = is_note.unsqueeze(-1) & (cells[..., PITCH0:ALTER0] > 0)
    return mask


class ScoreGridDecoder(nn.Module):
    def __init__(self, num_classes: int, d: int = 256, heads: int = 8, enc_layers: int = 3,
                 dec_layers: int = 6, use_gnn: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_gnn = use_gnn
        # --- nodes
        self.cls_emb = nn.Embedding(num_classes, d)
        self.row_emb = nn.Embedding(MAX_ROWS + 1, d)
        self.meas_emb = nn.Embedding(MAX_DETECTED_MEASURES + 1, d)
        self.left_b = nn.Embedding(B_MAX, d)
        self.geom = nn.Sequential(nn.Linear(5 + 2, d), nn.GELU(), nn.Linear(d, d))
        self.roi = nn.Linear(256, d)
        self.gnn = nn.Linear(256, d) if use_gnn else None
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers)
        # --- grid tokens (each field embeds its value or MASK = its vocab size)
        self.cell_emb = nn.ModuleList([nn.Embedding(v + 1, d) for v in CELL_VOCAB])
        self.staff_emb = nn.ModuleList([nn.Embedding(v + 1, d) for v in STAFF_VOCAB])
        self.barline_emb = nn.Embedding(BARLINE_VOCAB + 1, d)
        self.pos_b = nn.Embedding(B_MAX, d)
        self.pos_s = nn.Embedding(S_MAX, d)
        self.type_emb = nn.Embedding(2, d)  # cell, staff
        self.local = nn.Linear(d, d)
        self.local_count = nn.Embedding(LOCAL_MAX_COUNT + 1, d)
        self.edge = nn.Linear(d, d)
        self.edge_count = nn.Embedding(LOCAL_MAX_COUNT + 1, d)
        dlayer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dlayer, dec_layers)
        self.norm = nn.LayerNorm(d)
        self.cell_heads = nn.ModuleList([nn.Linear(d, v) for v in CELL_VOCAB])
        self.staff_heads = nn.ModuleList([nn.Linear(d, v) for v in STAFF_VOCAB])
        self.barline_head = nn.Linear(d, BARLINE_VOCAB)

    def encode(self, nodes: dict) -> torch.Tensor:
        geom = torch.cat([nodes["geom"], nodes["conf"][..., None], nodes["synth"][..., None]], dim=-1)
        cx, w = nodes["geom"][..., 0], nodes["geom"][..., 2]
        node_bin = (cx * B_MAX).long().clamp(0, B_MAX - 1)
        left_bin = ((cx - 0.5 * w) * B_MAX).long().clamp(0, B_MAX - 1)
        h = (self.cls_emb(nodes["cls"]) + self.geom(geom) + self.roi(nodes["roi"])
             + self.pos_b(node_bin) + self.left_b(left_bin)
             + self.row_emb((nodes["row"] + 1).clamp(0, MAX_ROWS))
             + self.meas_emb((nodes["meas"] + 1).clamp(0, MAX_DETECTED_MEASURES)))
        if self.gnn is not None:
            h = h + self.gnn(nodes["gnn"])
        return self.encoder(h, src_key_padding_mask=~nodes["valid"])

    def local_context(self, memory: torch.Tensor, nodes: dict) -> torch.Tensor:
        """[B, B_MAX, S_MAX, d]: per cell, the mean encoded state of the glyph
        nodes (at most LOCAL_MAX_BINS wide) whose centre falls in its bin, on
        its staff (the node's row; no row: staff 0), plus how many there are;
        and per bin (both staves) the mean state of the wide boxes (measures,
        staff cells, layers, beams) whose left edge falls in it, so measure
        starts reach the barline decision locally too (attention alone missed
        6.6% of the barlines of a model fit on its own scoring systems, most of
        them next to a detected measure edge)."""
        b, n, d = memory.shape
        cx, w = nodes["geom"][..., 0].float(), nodes["geom"][..., 2].float()
        bins = (cx * B_MAX).long().clamp(0, B_MAX - 1)
        staff = nodes["row"].clamp(0, S_MAX - 1)
        glyph = nodes["valid"] & (w * B_MAX <= LOCAL_MAX_BINS)
        n_cells = B_MAX * S_MAX
        index = torch.where(glyph, bins * S_MAX + staff, torch.full_like(bins, n_cells))  # last: discarded
        pooled = memory.new_zeros(b, n_cells + 1, d).scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, d), memory)
        count = memory.new_zeros(b, n_cells + 1).scatter_add_(1, index, torch.ones_like(memory[..., 0]))
        pooled = pooled[:, :n_cells] / count[:, :n_cells].clamp(min=1).unsqueeze(-1)
        count = count[:, :n_cells].long().clamp(max=LOCAL_MAX_COUNT)
        local = (self.local(pooled) + self.local_count(count)).reshape(b, B_MAX, S_MAX, d)

        left = ((cx - 0.5 * w) * B_MAX).long().clamp(0, B_MAX - 1)
        wide = nodes["valid"] & ~glyph
        index = torch.where(wide, left, torch.full_like(left, B_MAX))
        edges = memory.new_zeros(b, B_MAX + 1, d).scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, d), memory)
        n_edges = memory.new_zeros(b, B_MAX + 1).scatter_add_(1, index, torch.ones_like(memory[..., 0]))
        edges = edges[:, :B_MAX] / n_edges[:, :B_MAX].clamp(min=1).unsqueeze(-1)
        n_edges = n_edges[:, :B_MAX].long().clamp(max=LOCAL_MAX_COUNT)
        return local + (self.edge(edges) + self.edge_count(n_edges)).unsqueeze(2)

    def forward(self, cells, barline, staff, memory, memory_valid, local=None):
        """cells [B, B_MAX, S_MAX, 12], barline [B, B_MAX], staff [B, S_MAX, 3]:
        field values, MASK where masked; ``local``: local_context(). Returns
        per-field logits."""
        b = cells.shape[0]
        cell_tok = sum(emb(cells[..., i]) for i, emb in enumerate(self.cell_emb))
        if local is not None:
            cell_tok = cell_tok + local
        cell_tok = (cell_tok + self.barline_emb(barline)[:, :, None]
                    + self.pos_b.weight[:, None] + self.pos_s.weight[None, :]
                    + self.type_emb.weight[0])
        staff_tok = (sum(emb(staff[..., i]) for i, emb in enumerate(self.staff_emb))
                     + self.pos_s.weight + self.type_emb.weight[1])
        tokens = torch.cat([cell_tok.reshape(b, -1, cell_tok.shape[-1]), staff_tok], dim=1)
        out = self.norm(self.decoder(tokens, memory, memory_key_padding_mask=~memory_valid))
        n_cells = B_MAX * S_MAX
        cell_out = out[:, :n_cells].reshape(b, B_MAX, S_MAX, -1)
        bar_out = cell_out.sum(dim=2)
        staff_out = out[:, n_cells:]
        return (
            [head(cell_out) for head in self.cell_heads],
            self.barline_head(bar_out),
            [head(staff_out) for head in self.staff_heads],
        )


# --- masked discrete flow: training and sampling -------------------------------


def _mask_like(x: torch.Tensor, vocab: list | int, keep: torch.Tensor) -> torch.Tensor:
    """x with every position where ``keep`` is False set to MASK (the vocab size)."""
    if isinstance(vocab, int):
        mask_value = torch.full_like(x, vocab)
    else:
        mask_value = torch.tensor(vocab, device=x.device).expand_as(x)
    return torch.where(keep, x, mask_value)


def _consistent_cell_keep(cells: torch.Tensor, keep_c: torch.Tensor) -> torch.Tensor:
    """Align the training mask with sampling's hierarchy:
    - when EVENT is masked, all attribute fields DUR: are masked;
    - when EVENT is unmasked as PAD or MREST, DUR: are unmasked (0);
    - when EVENT is unmasked as REST, GRACE: are unmasked (0);
    - when PITCH_k is masked, ALTER_k is masked (so ALTER_k == 0 never leaks
      that PITCH_k == 0); when PITCH_k is unmasked as 0, ALTER_k is unmasked (0)."""
    keep_c = keep_c.clone()
    ev_kept = keep_c[..., EVENT]
    event = cells[..., EVENT]
    keep_c[..., DUR:] = torch.where(ev_kept.unsqueeze(-1), keep_c[..., DUR:], False)
    silent = ev_kept & ((event == PAD) | (event == MREST))
    keep_c[..., DUR:] = torch.where(silent.unsqueeze(-1), True, keep_c[..., DUR:])
    rest = ev_kept & (event == REST)
    keep_c[..., GRACE:] = torch.where(rest.unsqueeze(-1), True, keep_c[..., GRACE:])
    p_kept = keep_c[..., PITCH0:ALTER0]
    p_zero = p_kept & (cells[..., PITCH0:ALTER0] == 0)
    keep_c[..., ALTER0:] = torch.where(p_kept, keep_c[..., ALTER0:], False)
    keep_c[..., ALTER0:] = torch.where(p_zero, True, keep_c[..., ALTER0:])
    return keep_c


def flow_loss(model: ScoreGridDecoder, cells, barline, staff, nodes, t_min: float = 0.05) -> tuple:
    """Mask every field with probability t ~ U(t_min, 1) per system;
    cross-entropy on the masked fields that carry information."""
    b = cells.shape[0]
    dev = cells.device
    t = torch.empty(b, device=dev).uniform_(t_min, 1.0)
    keep_c = _consistent_cell_keep(cells, torch.rand(cells.shape, device=dev) >= t.view(b, 1, 1, 1))
    keep_b = torch.rand(barline.shape, device=dev) >= t.view(b, 1)
    keep_s = torch.rand(staff.shape, device=dev) >= t.view(b, 1, 1)
    memory = model.encode(nodes)
    local = model.local_context(memory, nodes)
    cell_logits, bar_logits, staff_logits = model(
        _mask_like(cells, CELL_VOCAB, keep_c), _mask_like(barline, BARLINE_VOCAB, keep_b),
        _mask_like(staff, STAFF_VOCAB, keep_s), memory, nodes["valid"], local,
    )
    informative = cell_loss_mask(cells)
    losses, parts = [], {}
    ev_weight = torch.tensor([1.0, 2.0, 2.0, 2.0], device=dev)
    for i, logits in enumerate(cell_logits):
        sel = informative[..., i] & ~keep_c[..., i]
        if sel.any():
            w = ev_weight if i == EVENT else None
            loss = F.cross_entropy(logits[sel].float(), cells[..., i][sel], weight=w)
            losses.append(loss)
            parts[f"cell{i}"] = float(loss.detach())
    sel = ~keep_b
    if sel.any():
        bar_weight = torch.tensor([1.0, 4.0], device=dev)
        loss_b = F.cross_entropy(bar_logits[sel].float(), barline[sel], weight=bar_weight)
        losses.append(loss_b)
        parts["barline"] = float(loss_b.detach())
    for i, logits in enumerate(staff_logits):
        sel = ~keep_s[..., i]
        if sel.any():
            losses.append(F.cross_entropy(logits[sel].float(), staff[..., i][sel]))
    return torch.stack(losses).mean(), parts


def _suppress_adjacent_barlines(barline: torch.Tensor, probs_1: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """1-D non-maximum suppression on predicted barlines so two adjacent bins
    within ``radius`` never both fire and emit an empty 1-bin measure."""
    out = barline.clone()
    b_sz, n_bins = out.shape
    for bi in range(b_sz):
        active = torch.nonzero(out[bi] == 1, as_tuple=False).flatten().tolist()
        if len(active) <= 1:
            continue
        # sort candidate bins by confidence (always keep bin 0 if predicted)
        active.sort(key=lambda idx: (idx == 0, float(probs_1[bi, idx])), reverse=True)
        kept = []
        for idx in active:
            if all(abs(idx - k) > radius for k in kept):
                kept.append(idx)
        out[bi].zero_()
        out[bi, kept] = 1
    return out


def _commit_top(conf: torch.Tensor, masked: torch.Tensor, n_commit: torch.Tensor) -> torch.Tensor:
    conf = torch.where(masked, conf, torch.full_like(conf, -1.0))
    order = torch.argsort(conf, dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(order.shape[1], device=conf.device).expand_as(order))
    return masked & (rank < n_commit.unsqueeze(1))


@torch.no_grad()
def sample(model: ScoreGridDecoder, nodes: dict, steps: int = 8) -> tuple:
    """Decode grids for a batch of systems: start fully masked, and at each
    of ``steps`` steps commit a cosine schedule of primary structural fields
    (event, barline, staff) and of active note/rest attribute fields (DUR:)."""
    b = nodes["cls"].shape[0]
    dev = nodes["cls"].device
    cells = torch.tensor(CELL_VOCAB, device=dev).expand(b, B_MAX, S_MAX, N_CELL).clone()
    barline = torch.full((b, B_MAX), BARLINE_VOCAB, device=dev)
    staff = torch.tensor(STAFF_VOCAB, device=dev).expand(b, S_MAX, len(STAFF_VOCAB)).clone()
    memory = model.encode(nodes)
    local = model.local_context(memory, nodes)

    n_ev = B_MAX * S_MAX
    n_pri = n_ev + B_MAX + S_MAX * len(STAFF_VOCAB)
    last_bar_p1 = torch.zeros((b, B_MAX), device=dev)

    for step in range(steps):
        cell_logits, bar_logits, staff_logits = model(cells, barline, staff, memory, nodes["valid"], local)

        # 1. Primary schedule: EVENT [B, B_MAX*S_MAX], barline [B, B_MAX], staff [B, S_MAX*3]
        p_ev = cell_logits[EVENT].float().softmax(-1)
        conf_ev, val_ev = p_ev.max(-1)
        mask_ev = (cells[..., EVENT] == CELL_VOCAB[EVENT]).reshape(b, -1)

        p_bar = bar_logits.float().softmax(-1)
        last_bar_p1 = p_bar[..., 1]
        conf_bar, val_bar = p_bar.max(-1)
        mask_bar = barline == BARLINE_VOCAB

        conf_st, val_st, mask_st = [], [], []
        for i, logits in enumerate(staff_logits):
            c_i, v_i = logits.float().softmax(-1).max(-1)
            conf_st.append(c_i)
            val_st.append(v_i)
            mask_st.append(staff[..., i] == STAFF_VOCAB[i])

        pri_conf = torch.cat([conf_ev.reshape(b, -1), conf_bar, *conf_st], dim=1)
        pri_val = torch.cat([val_ev.reshape(b, -1), val_bar, *val_st], dim=1)
        pri_mask = torch.cat([mask_ev, mask_bar, *mask_st], dim=1)

        rem_pri = math.floor(n_pri * math.cos(math.pi / 2 * (step + 1) / steps)) if step < steps - 1 else 0
        n_commit_pri = (pri_mask.sum(1) - rem_pri).clamp(min=0)
        pri_commit = _commit_top(pri_conf, pri_mask, n_commit_pri)

        c_ev = pri_commit[:, :n_ev].reshape(b, B_MAX, S_MAX)
        v_ev = pri_val[:, :n_ev].reshape(b, B_MAX, S_MAX)
        cells[..., EVENT] = torch.where(c_ev, v_ev, cells[..., EVENT])

        c_bar = pri_commit[:, n_ev:n_ev + B_MAX]
        v_bar = pri_val[:, n_ev:n_ev + B_MAX]
        barline = torch.where(c_bar, v_bar, barline)

        offset = n_ev + B_MAX
        for i in range(len(STAFF_VOCAB)):
            c_s = pri_commit[:, offset:offset + S_MAX]
            v_s = pri_val[:, offset:offset + S_MAX]
            staff[..., i] = torch.where(c_s, v_s, staff[..., i])
            offset += S_MAX

        # Force deterministic 'none' (0) on silent/rest cells immediately
        event = cells[..., EVENT]
        silent = (event == PAD) | (event == MREST)
        cells[..., DUR:] = torch.where(silent.unsqueeze(-1), torch.zeros_like(cells[..., DUR:]), cells[..., DUR:])
        rest = event == REST
        cells[..., GRACE:] = torch.where(rest.unsqueeze(-1), torch.zeros_like(cells[..., GRACE:]), cells[..., GRACE:])

        # 2. Attribute schedule: DUR..ALTER on cells whose EVENT is already NOTE or REST
        sounding = (event == NOTE) | (event == REST)
        is_note = event == NOTE
        attr_conf, attr_val, attr_mask = [], [], []
        for i in range(DUR, N_CELL):
            c_i, v_i = cell_logits[i].float().softmax(-1).max(-1)
            elig = sounding if i in (DUR, DOTS) else is_note
            m_i = (cells[..., i] == CELL_VOCAB[i]) & elig
            attr_conf.append(c_i.reshape(b, -1))
            attr_val.append(v_i.reshape(b, -1))
            attr_mask.append(m_i.reshape(b, -1))

        a_conf = torch.cat(attr_conf, dim=1)
        a_val = torch.cat(attr_val, dim=1)
        a_mask = torch.cat(attr_mask, dim=1)
        n_elig = a_mask.sum(1)
        n_commit_attr = n_elig if step == steps - 1 else (n_elig.float() / (steps - step)).ceil().long()
        a_commit = _commit_top(a_conf, a_mask, n_commit_attr)

        for idx, i in enumerate(range(DUR, N_CELL)):
            c_a = a_commit[:, idx * n_ev:(idx + 1) * n_ev].reshape(b, B_MAX, S_MAX)
            v_a = a_val[:, idx * n_ev:(idx + 1) * n_ev].reshape(b, B_MAX, S_MAX)
            cells[..., i] = torch.where(c_a, v_a, cells[..., i])

        # Any pitch committed as 0 ('none') forces its alteration to 0
        p_zero = cells[..., PITCH0:ALTER0] == 0
        cells[..., ALTER0:] = torch.where(p_zero, torch.zeros_like(cells[..., ALTER0:]), cells[..., ALTER0:])

    barline = _suppress_adjacent_barlines(barline, last_bar_p1)
    return cells, barline, staff
