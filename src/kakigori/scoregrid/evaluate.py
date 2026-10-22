"""Score the scoregrid decoder on the detection path: decode every detected
system of a build-scoregrid-data set, join each score's systems into
**kern, and compare against the MEI (musicset.evaluate, as compare-kern).

  --oracle   no model: each matched system gets its ground-truth matrix and
             an unmatched one stays empty, the ceiling the detected systems
             allow (measures are numbered consecutively in both modes, as
             at inference)

Usage:
    eval-scoregrid --checkpoint checkpoints_scoregrid/gnn/best.pt \\
                   --data data/scoregrid/validation-lite --mei-dir data/validation-lite/mei \\
                   --out checkpoints_scoregrid/gnn/krn-val [--steps 8]
"""

# Standard library imports
import json
import logging
import argparse
import statistics
from pathlib import Path

# Third party imports
import numpy as np
import torch
from tqdm import tqdm
from musicset.evaluate import compare_file, output_notes

# Local folder imports
from .grid import SystemGrid, grids_to_kern
from .model import DUR, DOTS, EVENT, GRACE, ALTER0, PITCH0, ScoreGridDecoder, sample
from .train import collate, to_device, load_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def tensors_to_grid(cells: torch.Tensor, measure: torch.Tensor, staff: torch.Tensor) -> SystemGrid:
    g = SystemGrid.empty()
    c = cells.cpu().numpy()
    g.event, g.dur, g.dots, g.grace = c[..., EVENT], c[..., DUR], c[..., DOTS], c[..., GRACE]
    g.pitch, g.alter = c[..., PITCH0:ALTER0], c[..., ALTER0:]
    g.measure = measure.cpu().numpy()
    s = staff.cpu().numpy()
    g.clef, g.key, g.staff = s[:, 0], s[:, 1], s[:, 2]
    return g


def grid_from_target(target: dict) -> SystemGrid:
    g = SystemGrid.empty()
    for name, value in target.items():
        setattr(g, name, np.asarray(value))
    return g


def score_dir(krn_dir: Path, mei_dir: Path, scores: list) -> dict:
    """compare-kern's totals over ``scores`` (missing kern: excluded and counted)."""
    p_tot = p_hit = r_tot = r_hit = out_notes = 0
    per_file, missing = [], 0
    for score in scores:
        krn, mei = krn_dir / f"{score}.krn", mei_dir / f"{score}.mei"
        if not krn.exists():
            missing += 1
            continue
        p, pm, r, rm = compare_file(mei, krn)
        p_tot, p_hit, r_tot, r_hit = p_tot + p, p_hit + pm, r_tot + r, r_hit + rm
        out_notes += output_notes(krn)
        if p:
            per_file.append(100.0 * pm / p)
    recall = 100.0 * p_hit / max(1, p_tot)
    precision = 100.0 * p_hit / max(1, out_notes)
    return {
        "pitch_recall": recall,
        "rhythm_recall": 100.0 * r_hit / max(1, r_tot),
        "pitch_precision": precision,
        "pitch_f1": 2 * recall * precision / max(1e-9, recall + precision),
        "notes_out": out_notes / max(1, p_tot),
        "median_file": statistics.median(per_file) if per_file else 0.0,
        "missing": missing,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scoregrid decoder on the detection path")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--mei-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((data / "index.json").read_text())
    records = load_records(data, matched_only=False)
    by_score: dict = {}
    for r in records:
        by_score.setdefault(r["score"], []).append(r)

    grids = {}
    if args.oracle:
        for r in records:
            grids[id(r)] = grid_from_target(r["target"]) if r["target"] is not None else SystemGrid.empty()
    else:
        device = torch.device("cuda")
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = ScoreGridDecoder(**state["config"]).to(device).eval()
        model.load_state_dict(state["model"])
        for i in tqdm(range(0, len(records), args.batch_size), desc="Decoding", mininterval=30):
            chunk = records[i:i + args.batch_size]
            batch = to_device(collate([dict(r, target=None) for r in chunk]), device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cells, measure, staff = sample(model, batch, steps=args.steps)
            for j, r in enumerate(chunk):
                grids[id(r)] = tensors_to_grid(cells[j], measure[j], staff[j])

    for score, info in index["scores"].items():
        systems = sorted(by_score.get(score, []), key=lambda r: (r["page"], r["order"]))
        if not systems:
            continue
        meter = tuple(info["meter"]) if info.get("meter") else None
        (out / f"{score}.krn").write_text(grids_to_kern([grids[id(r)] for r in systems], meter))
    result = score_dir(out, Path(args.mei_dir), list(index["scores"]))
    (out / "result.json").write_text(json.dumps(result, indent=1))
    logger.info(" | ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in result.items()))


if __name__ == "__main__":
    main()
