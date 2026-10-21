"""Visual QA for an `infer-omr --split` output directory (real scans, no
ground truth): for every page

  overlay/page_NNNN.png  the scan with the structure the serializer reads,
                         after bbox repair of the page's raw detections
                         (page_NNNN.json): rows red with the five staff lines
                         the serializer assumes in magenta, systems blue,
                         measures green, clefs orange
  render/page_NNNN*.svg  the page's predicted **kern engraved by Verovio, on
                         a white background (one SVG per Verovio page)
  render/FAILED.txt      pages whose kern Verovio could not engrave

Usage: uv run python scripts/qa_views.py OUT_DIR
"""
import sys
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from kakigori.graph.bbox_repair import repair_page_boxes

COLORS = {"system-staff": (220, 0, 0), "system": (0, 80, 255), "measure": (0, 170, 0)}

# Verovio's C++ core can abort on an inconsistent predicted **kern: one
# subprocess per page, so a crash only costs that page's render
RENDER = r"""
import re, sys, verovio
from pathlib import Path
krn, svg = Path(sys.argv[1]), Path(sys.argv[2])
tk = verovio.toolkit()
tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 40, "adjustPageHeight": True})
tk.loadData(krn.read_text())
for p in range(1, tk.getPageCount() + 1):
    text = tk.renderToSVG(p)
    # Verovio draws on a transparent canvas: paint it white
    text = re.sub(r"(<svg\b[^>]*>)", r'\1<rect x="0" y="0" width="100%" height="100%" fill="white"/>', text, count=1)
    (svg.parent / (svg.stem + ("" if p == 1 else f"_{p}") + ".svg")).write_text(text)
"""


def overlay(out: Path, page_json: Path):
    data = json.load(open(page_json))
    nodes = []
    for i, a in enumerate(data["annotations"]):
        x1, y1, x2, y2 = a["bbox"]
        nodes.append({"id": f"n{i}", "class": a["class"], "score": a.get("score", 1.0),
                      "bbox": a["bbox"], "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2})
    repair_page_boxes(nodes, page_prefix="p0")
    img = Image.open(out / data["image"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    for n in nodes:
        b = n["bbox"]
        if n["class"] in COLORS:
            draw.rectangle(b, outline=COLORS[n["class"]], width=3 if n["class"] == "system-staff" else 2)
            if n["class"] == "system-staff":
                for k in range(5):
                    y = b[1] + k * (b[3] - b[1]) / 4
                    draw.line([b[0], y, b[2], y], fill=(255, 0, 200), width=1)
        elif n["class"].startswith("clef"):
            draw.rectangle(b, outline=(255, 140, 0), width=3)
    img.save(out / "overlay" / f"{page_json.stem}.png")


def main():
    out = Path(sys.argv[1])
    (out / "overlay").mkdir(exist_ok=True)
    (out / "render").mkdir(exist_ok=True)
    for page_json in sorted(out.glob("page_*.json")):
        overlay(out, page_json)
    failed = []
    for krn in sorted(out.glob("page_*.krn")):
        svg = out / "render" / f"{krn.stem}.svg"
        r = subprocess.run([sys.executable, "-c", RENDER, str(krn), str(svg)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not svg.exists():
            failed.append(krn.stem)
    (out / "render" / "FAILED.txt").write_text("".join(f"{s}\n" for s in failed))
    print(f"{out}: {len(list((out / 'overlay').glob('*.png')))} overlays, "
          f"{len(list(out.glob('page_*.krn')))} kern pages, {len(failed)} not engravable by Verovio")


if __name__ == "__main__":
    main()
