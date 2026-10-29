"""Split the real scores carrying rare classes (scripts/rare_scores.py output)
between validation and training, deterministically.

Rarest feature first: validation takes the scores with the fewest occurrences
until it has ``--val-target`` occurrences of the feature, but never more than
a quarter of that feature's scores (at least one), so the scores rich in it
stay for training; training takes the rest, the richest first,
at most ``--train-cap`` scores per feature (repeat marks and key changes are
common enough to swamp the set otherwise). A score goes to one side only.

Usage:
  uv run python scripts/choose_rare.py data/sources/rare-pool.jsonl \\
      conf-lite/rare_validation_lite2.txt conf-lite/rare_train_lite2.txt
"""
# Standard library imports
import json
import argparse
from pathlib import Path
from collections import Counter

META = ("path", "staves")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pool", type=Path)
    parser.add_argument("val_out", type=Path)
    parser.add_argument("train_out", type=Path)
    parser.add_argument("--val-target", type=int, default=20)
    parser.add_argument("--train-cap", type=int, default=300)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.pool.open()]
    stem = {r["path"]: Path(r["path"]).stem for r in rows}
    by_feature: dict[str, list[dict]] = {}
    for r in rows:
        for k in r:
            if k not in META:
                by_feature.setdefault(k, []).append(r)
    order = sorted(by_feature, key=lambda k: (len(by_feature[k]), k))
    val, train = {}, {}
    for f in order:
        scores = sorted(by_feature[f], key=lambda r: (r[f], stem[r["path"]]))
        have = sum(r[f] for r in val.values() if f in r)
        budget = max(1, len(scores) // 4) - sum(1 for r in val.values() if f in r)
        for r in scores:
            if have >= args.val_target or budget <= 0:
                break
            if r["path"] in val or r["path"] in train:
                continue
            val[r["path"]] = r
            have += r[f]
            budget -= 1
    for f in order:
        taken = sum(1 for r in train.values() if f in r)
        for r in sorted(by_feature[f], key=lambda r: (-r[f], stem[r["path"]])):
            if taken >= args.train_cap:
                break
            if r["path"] in val or r["path"] in train:
                continue
            train[r["path"]] = r
            taken += 1
    for out, chosen in ((args.val_out, val), (args.train_out, train)):
        out.write_text(
            f"# {len(chosen)} PDMX scores carrying rare lite classes (scripts/choose_rare.py)\n"
            + "".join(f"{stem[p]}\n" for p in sorted(chosen, key=stem.get))
        )
    for name, chosen in (("validation", val), ("train", train)):
        feats = Counter()
        for r in chosen.values():
            for k, v in r.items():
                if k not in META:
                    feats[k] += v
        staves = Counter(r["staves"] for r in chosen.values())
        print(f"{name}: {len(chosen)} scores (by staff count {dict(staves)}); occurrences {dict(feats)}")


if __name__ == "__main__":
    main()
