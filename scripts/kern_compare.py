"""Side-by-side engravings of ground-truth and predicted **kern: a static
HTML viewer to see what a pipeline gets wrong, score by score.

  OUT/index.html          every score, worst pitch recall first, with its
                          metrics (recall / precision vs the MEI, notes out)
  OUT/scores/<stem>.html  the input pages (the source MusicXML as engraved
                          for the dataset: what the detector reads) |
                          ground-truth kern | predicted kern, both kerns
                          engraved by Verovio at the same width on white;
                          both kern texts below. Arrow keys go to the
                          previous / next score.

Usage:
  uv run python scripts/kern_compare.py --pred-dir inference_out/validation_det/krn \\
      --out inference_out/kern_compare
then open OUT/index.html in a browser.
"""
import os
import sys
import json
import html
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from musicset.evaluate import compare_file, output_notes

# Verovio's C++ core can abort on an inconsistent **kern: one subprocess per
# file, so a crash only costs that file's engraving
RENDER = r"""
import re, sys, json, verovio
from pathlib import Path
krn, prefix = Path(sys.argv[1]), sys.argv[2]
tk = verovio.toolkit()
tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 35, "adjustPageHeight": True,
               "header": "none", "footer": "none"})
tk.loadData(krn.read_text())
files = []
for p in range(1, tk.getPageCount() + 1):
    svg = re.sub(r"(<svg\b[^>]*>)", r'\1<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
                 tk.renderToSVG(p), count=1)
    path = Path(f"{prefix}_{p}.svg")
    path.write_text(svg)
    files.append(path.name)
print(json.dumps(files))
"""

CSS = """
body { margin: 0; font: 14px system-ui, sans-serif; background: #e6e6e6; color: #111; }
header { position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #bbb;
         padding: 8px 16px; display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: baseline; }
header b { font-family: ui-monospace, monospace; }
.metric { white-space: nowrap; }
.bad { color: #b00020; font-weight: 600; }
a { color: #0645ad; }
.cols { display: flex; gap: 12px; padding: 12px 16px; align-items: flex-start; }
.col { flex: 1 1 0; min-width: 0; }
.col h3 { margin: 0 0 6px; font-size: 13px; }
.col img { display: block; width: 100%; background: #fff; border: 1px solid #aaa; margin-bottom: 8px; }
.missing { background: #fff3f3; border: 1px solid #e0a0a0; padding: 12px; }
pre { background: #fff; border: 1px solid #aaa; padding: 8px; font-size: 11px; overflow: auto; max-height: 70vh; }
details { padding: 0 16px 16px; }
body:not(.show-source) .v-source, body:not(.show-gt) .v-gt, body:not(.show-pred) .v-pred { display: none; }
.toggles label { margin-right: 10px; white-space: nowrap; }
table { border-collapse: collapse; margin: 12px 16px; background: #fff; }
th, td { border: 1px solid #ccc; padding: 3px 8px; text-align: right; }
td:first-child, th:first-child { text-align: left; font-family: ui-monospace, monospace; }
tr:hover { background: #f0f6ff; }
"""

KEYS = """
<script>
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const go = id => { const a = document.getElementById(id); if (a) location.href = a.href; };
  if (e.key === 'ArrowLeft') go('prev');
  if (e.key === 'ArrowRight') go('next');
});
// one checkbox per column (source pages, ground truth, predicted): shown by
// default, the choice remembered across score pages
document.querySelectorAll('input[data-view]').forEach(box => {
  const view = box.dataset.view, key = 'kc-show-' + view;
  try { box.checked = localStorage.getItem(key) !== '0'; } catch (err) { box.checked = true; }
  const apply = () => document.body.classList.toggle('show-' + view, box.checked);
  box.addEventListener('change', () => { apply(); try { localStorage.setItem(key, box.checked ? '1' : '0'); } catch (err) {} });
  apply();
});
</script>
"""


def render(krn: Path, prefix: Path, reuse: bool = False) -> list:
    if not krn.exists():
        return []
    if reuse:
        done = sorted(prefix.parent.glob(f"{prefix.name}_*.svg"),
                      key=lambda f: int(f.stem.rsplit("_", 1)[1]))
        if done:
            return [f.name for f in done]
    r = subprocess.run([sys.executable, "-c", RENDER, str(krn), str(prefix)],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        return []
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return []


def metrics(mei: Path, krn: Path) -> dict:
    if not krn.exists():
        return {"pitch": 0.0, "rhythm": 0.0, "precision": 0.0, "ratio": 0.0, "notes": 0}
    p_total, p_matched, r_total, r_matched = compare_file(mei, krn)
    out = output_notes(krn)
    return {
        "pitch": 100.0 * p_matched / p_total if p_total else 0.0,
        "rhythm": 100.0 * r_matched / r_total if r_total else 0.0,
        "precision": 100.0 * p_matched / out if out else 0.0,
        "ratio": out / p_total if p_total else 0.0,
        "notes": p_total,
    }


def fmt(value: float, bad: bool, unit: str = "%") -> str:
    text = f"{value:.1f}{unit}"
    return f'<span class="bad">{text}</span>' if bad else text


def score_page(i, stems, rows, args, out: Path) -> str:
    stem = stems[i]
    row = rows[stem]
    m, gt_m = row["pred"], row["gt"]
    links = [f'<a href="../index.html">index</a>']
    if i > 0:
        links.append(f'<a id="prev" href="{stems[i - 1]}.html">&larr; prev</a>')
    if i + 1 < len(stems):
        links.append(f'<a id="next" href="{stems[i + 1]}.html">next &rarr;</a>')

    def column(title, files, krn, view):
        if files:
            body = "".join(f'<img loading="lazy" src="../svg/{f}" alt="{html.escape(title)} page">' for f in files)
        elif krn.exists():
            body = '<div class="missing">Verovio could not engrave this kern (see the text below).</div>'
        else:
            body = '<div class="missing">No kern output for this score.</div>'
        return f'<div class="col v-{view}"><h3>{html.escape(title)}</h3>{body}</div>'

    imgs = sorted(Path(args.img_dir).glob(f"{stem}_page*.png"))
    pages = "".join(
        f'<img loading="lazy" src="{html.escape(os.path.relpath(p, out / "scores"))}" alt="page image">' for p in imgs
    )
    gt_krn, pred_krn = Path(args.gt_dir) / f"{stem}.krn", Path(args.pred_dir) / f"{stem}.krn"
    sources = sorted(Path(args.source_dir).glob(f"{stem}.*"))
    source = (
        f'<a href="{html.escape(os.path.relpath(sources[0], out / "scores"))}">source {sources[0].suffix}</a>'
        if sources else ""
    )
    text = lambda p: html.escape(p.read_text()) if p.exists() else "(missing)"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{stem[:12]} kern compare</title>
<style>{CSS}</style></head><body>
<header><b>{stem}</b>
<span class="metric">pitch recall {fmt(m["pitch"], m["pitch"] < 80)}</span>
<span class="metric">rhythm recall {fmt(m["rhythm"], m["rhythm"] < 80)}</span>
<span class="metric">pitch precision {fmt(m["precision"], m["precision"] < 70)}</span>
<span class="metric">notes out {fmt(m["ratio"], abs(m["ratio"] - 1) > 0.15, "x")} of {m["notes"]}</span>
<span class="metric">ground-truth kern vs MEI {fmt(gt_m["pitch"], gt_m["pitch"] < 95)}</span>
<span class="toggles"><label><input type="checkbox" data-view="source" checked> source pages</label><label><input type="checkbox" data-view="gt" checked> ground truth</label><label><input type="checkbox" data-view="pred" checked> predicted</label></span>
<span>{" · ".join(links + ([source] if source else []))}</span></header>
<div class="cols">
<div class="col v-source"><h3>Source pages (the MusicXML engraved for the dataset: the detector's input)</h3>{pages}</div>
{column("Ground truth (musicset kern)", row["gt_svg"], gt_krn, "gt")}
{column("Predicted (" + args.pred_label + ")", row["pred_svg"], pred_krn, "pred")}
</div>
<details><summary>kern text</summary><div class="cols">
<div class="col v-gt"><h3>Ground truth</h3><pre>{text(gt_krn)}</pre></div>
<div class="col v-pred"><h3>Predicted</h3><pre>{text(pred_krn)}</pre></div>
</div></details>
{KEYS}</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--pred-label", default="detection path")
    ap.add_argument("--gt-dir", default="data/validation-lite/kern")
    ap.add_argument("--mei-dir", default="data/validation-lite/mei")
    ap.add_argument("--img-dir", default="data/validation-lite/imgs")
    ap.add_argument("--source-dir", default="data/validation-lite/filter",
                    help="the source MusicXML (linked from each score page)")
    ap.add_argument("--reuse-svg", action="store_true",
                    help="keep engravings already in OUT/svg (rebuild the pages only)")
    ap.add_argument("--out", default="inference_out/kern_compare")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "svg").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)
    stems = sorted(p.stem for p in Path(args.mei_dir).glob("*.mei"))[: args.limit]

    def work(stem):
        mei = Path(args.mei_dir) / f"{stem}.mei"
        gt, pred = Path(args.gt_dir) / f"{stem}.krn", Path(args.pred_dir) / f"{stem}.krn"
        return stem, {
            "gt": metrics(mei, gt), "pred": metrics(mei, pred),
            "gt_svg": render(gt, out / "svg" / f"{stem}_gt", args.reuse_svg),
            "pred_svg": render(pred, out / "svg" / f"{stem}_pred", args.reuse_svg),
        }

    rows = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for n, (stem, row) in enumerate(ex.map(work, stems), 1):
            rows[stem] = row
            if n % 50 == 0:
                print(f"{n}/{len(stems)}", flush=True)

    order = sorted(stems, key=lambda s: (rows[s]["pred"]["pitch"], s))
    for i, stem in enumerate(order):
        (out / "scores" / f"{stem}.html").write_text(score_page(i, order, rows, args, out))

    body = "".join(
        f'<tr><td><a href="scores/{s}.html">{s}</a></td>'
        f'<td>{fmt(r["pred"]["pitch"], r["pred"]["pitch"] < 80)}</td><td>{fmt(r["pred"]["rhythm"], r["pred"]["rhythm"] < 80)}</td>'
        f'<td>{fmt(r["pred"]["precision"], r["pred"]["precision"] < 70)}</td>'
        f'<td>{fmt(r["pred"]["ratio"], abs(r["pred"]["ratio"] - 1) > 0.15, "x")}</td><td>{r["pred"]["notes"]}</td>'
        f'<td>{fmt(r["gt"]["pitch"], r["gt"]["pitch"] < 95)}</td>'
        f'<td>{"" if r["pred_svg"] else "no"}</td></tr>'
        for s in order for r in [rows[s]]
    )
    (out / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Kern compare</title><style>{CSS}</style></head><body>
<header><b>{html.escape(args.pred_label)}</b> vs ground truth, {len(order)} scores, worst pitch recall first.
Recall = matched / ground-truth notes per (staff, measure); precision = matched / output notes.</header>
<table><tr><th>score</th><th>pitch recall</th><th>rhythm recall</th><th>pitch precision</th><th>notes out</th><th>GT notes</th><th>GT kern vs MEI</th><th>engraved</th></tr>
{body}</table></body></html>""")
    failed = sum(1 for r in rows.values() if not r["pred_svg"])
    print(f"wrote {out / 'index.html'}: {len(order)} scores, {failed} predicted kern Verovio could not engrave")


if __name__ == "__main__":
    main()
