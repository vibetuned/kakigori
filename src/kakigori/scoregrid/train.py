"""Train the scoregrid decoder on build-scoregrid-data output.

Usage:
    train-scoregrid --train data/scoregrid/train-lite --val data/scoregrid/validation-lite \\
                    --out checkpoints_scoregrid/gnn [--no-gnn] [--epochs 30]
"""

# Standard library imports
import json
import math
import logging
import argparse
from pathlib import Path

# Third party imports
import torch
from torch.utils.data import Dataset, DataLoader

# Local folder imports
from .model import ScoreGridDecoder, flow_loss, grid_tensors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

N_MAX = 512  # node tokens per system (detections: p99 ~ 400)


def load_records(data_dir: Path, matched_only: bool = True) -> list:
    records = []
    for shard in sorted(Path(data_dir).glob("shard_*.pt")):
        for r in torch.load(shard, weights_only=False):
            if matched_only and r["target"] is None:
                continue
            records.append(r)
    return records


class SystemDataset(Dataset):
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


def collate(records: list) -> dict:
    """Pad the node tokens; stack the targets (when every record has one)."""
    n = min(max(len(r["cls"]) for r in records), N_MAX)
    b = len(records)
    out = {
        "cls": torch.zeros(b, n, dtype=torch.long),
        "geom": torch.zeros(b, n, 5),
        "row": torch.full((b, n), -1, dtype=torch.long),
        "meas": torch.full((b, n), -1, dtype=torch.long),
        "conf": torch.zeros(b, n),
        "synth": torch.zeros(b, n),
        "roi": torch.zeros(b, n, 256),
        "gnn": torch.zeros(b, n, 256),
        "valid": torch.zeros(b, n, dtype=torch.bool),
    }
    for i, r in enumerate(records):
        k = min(len(r["cls"]), n)
        out["cls"][i, :k] = r["cls"][:k].long()
        out["geom"][i, :k] = r["geom"][:k].float()
        out["row"][i, :k] = r["row"][:k].long()
        out["meas"][i, :k] = r["meas"][:k].long()
        out["conf"][i, :k] = r["conf"][:k].float()
        out["synth"][i, :k] = r["synth"][:k].float()
        out["roi"][i, :k] = r["roi"][:k].float()
        out["gnn"][i, :k] = r["gnn"][:k].float()
        out["valid"][i, :k] = True
    if all(r["target"] is not None for r in records):
        cells, measure, staff = zip(*(grid_tensors(r["target"]) for r in records), strict=True)
        out["cells"], out["measure"], out["staff"] = torch.stack(cells), torch.stack(measure), torch.stack(staff)
    return out


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = flow_loss(model, batch["cells"], batch["measure"], batch["staff"], batch)
        total += float(loss) * batch["cls"].shape[0]
        n += batch["cls"].shape[0]
    model.train()
    return total / max(1, n)


def main():
    parser = argparse.ArgumentParser(description="Train the scoregrid decoder")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--config", type=str, default="conf-lite/config.json")
    parser.add_argument("--no-gnn", action="store_true", help="Ablation: no GNN node states")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (smoke tests)")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    num_classes = len(json.load(open(args.config))["target_classes"])
    train = load_records(Path(args.train))
    val = load_records(Path(args.val))
    logger.info(f"{len(train)} training systems, {len(val)} validation systems")
    # Workers fork (Python 3.14 defaults to forkserver, which pickles the
    # in-memory records to every worker: "too many fds"); they only collate
    workers = {"num_workers": args.num_workers}
    if args.num_workers:
        workers.update(multiprocessing_context="fork", persistent_workers=True)
    train_loader = DataLoader(SystemDataset(train), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True, **workers)
    val_loader = DataLoader(SystemDataset(val), batch_size=args.batch_size, collate_fn=collate, **workers)

    config = {"num_classes": num_classes, "use_gnn": not args.no_gnn}
    model = ScoreGridDecoder(**config).to(device)
    logger.info(f"ScoreGridDecoder: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
                f"GNN node states {'on' if config['use_gnn'] else 'off'}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.max_steps or args.epochs * len(train_loader)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    step, best = 0, float("inf")
    log = open(out / "train.log", "a")
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = flow_loss(model, batch["cells"], batch["measure"], batch["staff"], batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 100 == 0:
                msg = f"epoch {epoch} step {step}/{total_steps} loss {float(loss):.4f} lr {scheduler.get_last_lr()[0]:.2e}"
                logger.info(msg)
                log.write(msg + "\n")
                log.flush()
            if args.max_steps and step >= args.max_steps:
                break
        val_loss = eval_loss(model, val_loader, device)
        msg = f"epoch {epoch} done: val loss {val_loss:.4f}"
        logger.info(msg)
        log.write(msg + "\n")
        log.flush()
        state = {"model": model.state_dict(), "config": config, "epoch": epoch, "val_loss": val_loss}
        torch.save(state, out / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(state, out / "best.pt")
        if args.max_steps and step >= args.max_steps:
            break
    logger.info(f"done: best val loss {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
