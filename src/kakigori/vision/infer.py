"""PDF Inference Script for MusicDetector.

Renders every page of a PDF, runs the trained model on each, and outputs:
  - page_NNNN.png  — raw rendered page image
  - page_NNNN.json — detections in dataset annotation format

Usage:
    infer-model --checkpoint checkpoints/my_run --pdf score.pdf --output-dir out/
"""

# Standard library imports
import json
import logging
import argparse
from pathlib import Path

# Third party imports
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# Local folder imports
from .utils import load_detector, decode_model_outputs
from .dataset import canvas_to_page, letterbox_image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF → PIL pages
# ---------------------------------------------------------------------------


def pdf_to_pages(pdf_path: Path, dpi: int = 300) -> list[Image.Image]:
    """Render each page of a PDF to a PIL RGB image."""
    try:
        # Third party imports
        import pypdfium2 as pdfium  # type: ignore
    except ImportError:
        raise ImportError(
            "pypdfium2 is required for PDF rendering. Install it with:\n"
            "  uv add pypdfium2"
        )

    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    pages = []
    for i, page in tqdm(enumerate(doc), desc="Rendering PDF pages", total=len(doc)):
        bitmap = page.render(scale=scale, rotation=0)
        pages.append(bitmap.to_pil().convert("RGB"))
    return pages


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess(
    image: Image.Image, input_size, device: torch.device
) -> tuple[torch.Tensor, dict]:
    """Fit a page into the detector's canvas (``input_size``: int or
    (width, height), e.g. ``model.input_size``) as one grey channel. Returns
    (tensor, ``letterbox_geometry``); map boxes back with ``canvas_to_page``."""
    canvas, meta = letterbox_image(image.convert("L"), input_size)
    tensor = TF.to_tensor(canvas).unsqueeze(0).to(device)  # (1, 1, H, W)
    meta["input_size"] = (meta["canvas_w"], meta["canvas_h"])
    return tensor, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run MusicDetector on a PDF score.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory.",
    )
    parser.add_argument("--config", type=str, default="conf/config.json")
    parser.add_argument("--output-dir", type=str, default="inference_out")
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument(
        "--multi-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Every class over the threshold per cell (default), or only the argmax.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for PDF page rendering."
    )
    # The input canvas, detection levels and neck come from the checkpoint
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load config ---
    with open(args.config) as f:
        config = json.load(f)
    class_list = config["target_classes"]
    num_classes = len(class_list)

    # --- Load model ---
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_detector(args.checkpoint, num_classes, device)
    input_size = model.input_size

    # --- Render PDF ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rendering PDF: {args.pdf}")
    pages = pdf_to_pages(Path(args.pdf), dpi=args.dpi)

    with torch.inference_mode():
        pbar = tqdm(enumerate(pages), total=len(pages), unit="page", desc="Inferring")
        for page_idx, page_image in pbar:
            stem = f"page_{page_idx + 1:04d}"
            img_path = output_dir / f"{stem}.png"
            page_image.save(img_path)

            tensor, meta = preprocess(page_image, input_size, device)
            outputs = model(tensor)
            preds = decode_model_outputs(
                outputs, args.conf_thresh, args.iou_thresh, input_size, multi_label=args.multi_label
            )[0]
            boxes = canvas_to_page(preds["boxes"].float().cpu(), meta)

            detections = [
                {
                    "class": class_list[int(c)],
                    "score": round(float(s), 4),
                    "bbox": [round(float(v), 1) for v in box],
                }
                for box, s, c in zip(boxes, preds["scores"], preds["labels"])
            ]
            pbar.set_postfix(dets=len(detections))

            # One JSON per page — matches dataset annotation format
            ann_data = {
                "image": img_path.name,
                "width": page_image.width,
                "height": page_image.height,
                "annotations": [
                    {"class": d["class"], "score": d["score"], "bbox": d["bbox"]}
                    for d in detections
                ],
            }
            json_path = output_dir / f"{stem}.json"
            with open(json_path, "w") as f:
                json.dump(ann_data, f, indent=2)

    logger.info(f"Done. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
