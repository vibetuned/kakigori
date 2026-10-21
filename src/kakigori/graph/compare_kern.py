# Standard library imports
import logging
import argparse
import statistics
from pathlib import Path

# The comparison itself lives with the MEI and kern it reads, in musicset
from musicset.evaluate import compare_file, output_notes

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

    # pitch / rhythm are recall (matched / groundtruth notes); "out" is the
    # output's note count relative to the groundtruth, and the TOTAL line adds
    # pitch precision (matched / output notes) and F1: extra notes are only
    # visible there
    print(f"{'file':<55} {'notes':>6} {'pitch':>7} {'rhythm':>7} {'out':>6}")
    gp_total = gp_matched = gr_total = gr_matched = g_out = 0
    per_file, missing = [], 0
    for mei_path in sorted(mei_dir.glob("*.mei")):
        krn_path = krn_dir / f"{mei_path.stem}.krn"
        if not krn_path.exists():
            logger.warning(f"No kern output for {mei_path.stem}")
            missing += 1
            continue
        try:
            p_total, p_matched, r_total, r_matched = compare_file(mei_path, krn_path)
        except Exception as e:
            logger.error(f"Failed on {mei_path.stem}: {e}")
            continue
        out = output_notes(krn_path)
        gp_total += p_total; gp_matched += p_matched
        gr_total += r_total; gr_matched += r_matched
        g_out += out
        p_pct = 100.0 * p_matched / p_total if p_total else float("nan")
        r_pct = 100.0 * r_matched / r_total if r_total else float("nan")
        ratio = out / p_total if p_total else float("nan")
        print(f"{mei_path.stem:<55} {p_total:>6} {p_pct:>6.1f}% {r_pct:>6.1f}% {ratio:>5.2f}x")
        if p_total:
            per_file.append((p_pct, r_pct))

    if gp_total:
        print(f"{'TOTAL':<55} {gp_total:>6} {100.0 * gp_matched / gp_total:>6.1f}% "
              f"{100.0 * gr_matched / gr_total if gr_total else float('nan'):>6.1f}%")
        recall, precision = gp_matched / gp_total, gp_matched / g_out if g_out else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        print(f"{'PITCH PRECISION / F1 (' + f'{g_out / gp_total:.2f}x notes out)':<55} {'':>6} "
              f"{100.0 * precision:>6.1f}% {100.0 * f1:>6.1f}%")
        label = f"MEDIAN FILE ({len(per_file)} files)"
        print(f"{label:<55} {'':>6} {statistics.median(p for p, _ in per_file):>6.1f}% "
              f"{statistics.median(r for _, r in per_file):>6.1f}%")
    if missing:
        # not counted in TOTAL / MEDIAN: a score that fails to serialize is
        # excluded, not scored 0
        print(f"{missing} groundtruth files have no kern output (not counted above)")


if __name__ == "__main__":
    main()
