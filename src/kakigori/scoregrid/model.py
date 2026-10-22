"""The scoregrid decoder: a system's detected nodes in, its SystemGrid out,
by masked discrete flow matching over the grid's categorical fields.

Every field value is a variable; training masks each one independently with
probability t ~ U(0.05, 1) and predicts the masked ones from the rest and
the nodes. Sampling starts fully masked and unmasks the most confident
predictions over K steps (K = 1: a one-shot classifier, the ablation that
tells whether the iterative refinement earns its keep).

Tokens: one per cell (measure, slice, staff) carrying its 12 fields, one per
measure (``measure``), one per staff (``clef``, ``key``, ``staff``). They
self-attend and cross-attend to the encoded nodes (class, geometry, row,
detected measure, detector score, RoI feature and, optionally, the GNN's node
state).
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
    HOLD,
    KEYS,
    NOTE,
    REST,
    CLEFS,
    K_MAX,
    L_MAX,
    M_MAX,
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
MEASURE_VOCAB = 2
MAX_DETECTED_MEASURES = 32
MAX_ROWS = 4


def grid_tensors(target: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A SystemGrid's arrays as (cells [M, L, S, 12], measure [M], staff [S, 3])."""
    cells = torch.cat([
        torch.as_tensor(target["event"])[..., None],
        torch.as_tensor(target["dur"])[..., None],
        torch.as_tensor(target["dots"])[..., None],
        torch.as_tensor(target["grace"])[..., None],
        torch.as_tensor(target["pitch"]),
        torch.as_tensor(target["alter"]),
    ], dim=-1).long()
    staff = torch.stack([torch.as_tensor(target[k]) for k in ("clef", "key", "staff")], dim=-1).long()
    return cells, torch.as_tensor(target["measure"]).long(), staff


def cell_loss_mask(cells: torch.Tensor) -> torch.Tensor:
    """Which cell fields carry information: the event always; duration and
    dots for notes and rests; grace, pitches and alterations for notes. The
    others are 'none' by construction (and forced so when sampling)."""
    event = cells[..., EVENT]
    sounding = (event == NOTE) | (event == REST)
    mask = torch.zeros_like(cells, dtype=torch.bool)
    mask[..., EVENT] = True
    mask[..., DUR] = sounding
    mask[..., DOTS] = sounding
    mask[..., GRACE:] = (event == NOTE).unsqueeze(-1)
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
        self.geom = nn.Sequential(nn.Linear(5 + 2, d), nn.GELU(), nn.Linear(d, d))
        self.roi = nn.Linear(256, d)
        self.gnn = nn.Linear(256, d) if use_gnn else None
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers)
        # --- grid tokens (each field embeds its value or MASK = its vocab size)
        self.cell_emb = nn.ModuleList([nn.Embedding(v + 1, d) for v in CELL_VOCAB])
        self.staff_emb = nn.ModuleList([nn.Embedding(v + 1, d) for v in STAFF_VOCAB])
        self.measure_emb = nn.Embedding(MEASURE_VOCAB + 1, d)
        self.pos_m = nn.Embedding(M_MAX, d)
        self.pos_l = nn.Embedding(L_MAX, d)
        self.pos_s = nn.Embedding(S_MAX, d)
        self.type_emb = nn.Embedding(3, d)  # cell, measure, staff
        dlayer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dlayer, dec_layers)
        self.norm = nn.LayerNorm(d)
        self.cell_heads = nn.ModuleList([nn.Linear(d, v) for v in CELL_VOCAB])
        self.staff_heads = nn.ModuleList([nn.Linear(d, v) for v in STAFF_VOCAB])
        self.measure_head = nn.Linear(d, MEASURE_VOCAB)

    def encode(self, nodes: dict) -> torch.Tensor:
        geom = torch.cat([nodes["geom"], nodes["conf"][..., None], nodes["synth"][..., None]], dim=-1)
        h = (self.cls_emb(nodes["cls"]) + self.geom(geom) + self.roi(nodes["roi"])
             + self.row_emb((nodes["row"] + 1).clamp(0, MAX_ROWS))
             + self.meas_emb((nodes["meas"] + 1).clamp(0, MAX_DETECTED_MEASURES)))
        if self.gnn is not None:
            h = h + self.gnn(nodes["gnn"])
        return self.encoder(h, src_key_padding_mask=~nodes["valid"])

    def forward(self, cells, measure, staff, memory, memory_valid):
        """cells [B, M, L, S, 12], measure [B, M], staff [B, S, 3]: field
        values, MASK where masked. Returns per-field logits."""
        b = cells.shape[0]
        cell_tok = sum(emb(cells[..., i]) for i, emb in enumerate(self.cell_emb))
        cell_tok = (cell_tok + self.pos_m.weight[:, None, None] + self.pos_l.weight[None, :, None]
                    + self.pos_s.weight[None, None, :] + self.type_emb.weight[0])
        meas_tok = self.measure_emb(measure) + self.pos_m.weight + self.type_emb.weight[1]
        staff_tok = (sum(emb(staff[..., i]) for i, emb in enumerate(self.staff_emb))
                     + self.pos_s.weight + self.type_emb.weight[2])
        tokens = torch.cat([cell_tok.reshape(b, -1, cell_tok.shape[-1]), meas_tok, staff_tok], dim=1)
        out = self.norm(self.decoder(tokens, memory, memory_key_padding_mask=~memory_valid))
        n_cells = M_MAX * L_MAX * S_MAX
        cell_out = out[:, :n_cells].reshape(b, M_MAX, L_MAX, S_MAX, -1)
        meas_out = out[:, n_cells:n_cells + M_MAX]
        staff_out = out[:, n_cells + M_MAX:]
        return (
            [head(cell_out) for head in self.cell_heads],
            self.measure_head(meas_out),
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


def flow_loss(model: ScoreGridDecoder, cells, measure, staff, nodes, t_min: float = 0.05) -> tuple:
    """Mask every field independently with probability t ~ U(t_min, 1) per
    system; cross-entropy on the masked fields that carry information."""
    b = cells.shape[0]
    t = torch.empty(b, device=cells.device).uniform_(t_min, 1.0)
    keep_c = torch.rand(cells.shape, device=cells.device) >= t.view(b, 1, 1, 1, 1)
    keep_m = torch.rand(measure.shape, device=cells.device) >= t.view(b, 1)
    keep_s = torch.rand(staff.shape, device=cells.device) >= t.view(b, 1, 1)
    memory = model.encode(nodes)
    cell_logits, meas_logits, staff_logits = model(
        _mask_like(cells, CELL_VOCAB, keep_c), _mask_like(measure, MEASURE_VOCAB, keep_m),
        _mask_like(staff, STAFF_VOCAB, keep_s), memory, nodes["valid"],
    )
    informative = cell_loss_mask(cells)
    losses, parts = [], {}
    for i, logits in enumerate(cell_logits):
        sel = informative[..., i] & ~keep_c[..., i]
        if sel.any():
            loss = F.cross_entropy(logits[sel].float(), cells[..., i][sel])
            losses.append(loss)
            parts[f"cell{i}"] = float(loss.detach())
    sel = ~keep_m
    if sel.any():
        losses.append(F.cross_entropy(meas_logits[sel].float(), measure[sel]))
    for i, logits in enumerate(staff_logits):
        sel = ~keep_s[..., i]
        if sel.any():
            losses.append(F.cross_entropy(logits[sel].float(), staff[..., i][sel]))
    return torch.stack(losses).mean(), parts


@torch.no_grad()
def sample(model: ScoreGridDecoder, nodes: dict, steps: int = 8) -> tuple:
    """Decode grids for a batch of systems: start fully masked, and at each
    of ``steps`` steps commit the most confident of the still-masked fields
    (a cosine schedule; the last step commits the rest). A cell whose event
    is committed as pad or hold gets 'none' in its other fields at once."""
    b = nodes["cls"].shape[0]
    dev = nodes["cls"].device
    cells = torch.tensor(CELL_VOCAB, device=dev).expand(b, M_MAX, L_MAX, S_MAX, N_CELL).clone()
    measure = torch.full((b, M_MAX), MEASURE_VOCAB, device=dev)
    staff = torch.tensor(STAFF_VOCAB, device=dev).expand(b, S_MAX, len(STAFF_VOCAB)).clone()
    memory = model.encode(nodes)
    total = cells[0].numel() + measure[0].numel() + staff[0].numel()
    for step in range(steps):
        cell_logits, meas_logits, staff_logits = model(cells, measure, staff, memory, nodes["valid"])
        # candidates: (confidence, value) for every still-masked variable
        conf_parts, value_parts, masked_parts = [], [], []
        for i, logits in enumerate(cell_logits):
            p = logits.float().softmax(-1)
            conf, val = p.max(-1)
            conf_parts.append(conf.reshape(b, -1))
            value_parts.append(val.reshape(b, -1))
            masked_parts.append((cells[..., i] == CELL_VOCAB[i]).reshape(b, -1))
        p = meas_logits.float().softmax(-1)
        conf, val = p.max(-1)
        conf_parts.append(conf)
        value_parts.append(val)
        masked_parts.append(measure == MEASURE_VOCAB)
        for i, logits in enumerate(staff_logits):
            p = logits.float().softmax(-1)
            conf, val = p.max(-1)
            conf_parts.append(conf)
            value_parts.append(val)
            masked_parts.append(staff[..., i] == STAFF_VOCAB[i])
        conf = torch.cat(conf_parts, 1)
        value = torch.cat(value_parts, 1)
        masked = torch.cat(masked_parts, 1)
        # how many stay masked after this step (cosine schedule)
        remaining = math.floor(total * math.cos(math.pi / 2 * (step + 1) / steps)) if step < steps - 1 else 0
        conf = torch.where(masked, conf, torch.full_like(conf, -1.0))
        n_masked = masked.sum(1)
        n_commit = (n_masked - remaining).clamp(min=0)
        order = torch.argsort(conf, dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(order.shape[1], device=dev).expand_as(order))
        commit = masked & (rank < n_commit.unsqueeze(1))
        # write the committed values back field by field
        offset = 0
        for i in range(N_CELL):
            n = M_MAX * L_MAX * S_MAX
            c = commit[:, offset:offset + n].reshape(b, M_MAX, L_MAX, S_MAX)
            v = value[:, offset:offset + n].reshape(b, M_MAX, L_MAX, S_MAX)
            cells[..., i] = torch.where(c, v, cells[..., i])
            offset += n
        c, v = commit[:, offset:offset + M_MAX], value[:, offset:offset + M_MAX]
        measure = torch.where(c, v, measure)
        offset += M_MAX
        for i in range(len(STAFF_VOCAB)):
            c = commit[:, offset:offset + S_MAX]
            v = value[:, offset:offset + S_MAX]
            staff[..., i] = torch.where(c, v, staff[..., i])
            offset += S_MAX
        # fields that an event leaves meaningless are 'none' at once
        event = cells[..., EVENT]
        silent = (event == PAD) | (event == HOLD)
        cells[..., DUR:] = torch.where(silent.unsqueeze(-1), torch.zeros_like(cells[..., DUR:]), cells[..., DUR:])
        rest = event == REST
        cells[..., GRACE:] = torch.where(rest.unsqueeze(-1), torch.zeros_like(cells[..., GRACE:]), cells[..., GRACE:])
    return cells, measure, staff
