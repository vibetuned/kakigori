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
from .model import MusicDetector
from .utils import load_checkpoint, decode_model_outputs
from .dataset import _letterbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

def preprocess(image: Image.Image, input_size: int, device: torch.device) -> tuple[torch.Tensor, dict]:
    """Letterbox + normalise a PIL image. Returns (tensor, padding metadata)."""
    orig_w, orig_h = image.size
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2

    img_lb, _ = _letterbox(image, input_size, torch.zeros(0, 4))
    tensor = TF.to_tensor(img_lb).unsqueeze(0).to(device)  # (1, 3, H, W)

    meta = {
        "orig_w": orig_w, "orig_h": orig_h,
        "scale": scale, "pad_x": pad_x, "pad_y": pad_y,
        "input_size": input_size,
    }
    return tensor, meta




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run MusicDetector on a PDF score.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory.")
    parser.add_argument("--config", type=str, default="conf/config.json")
    parser.add_argument("--output-dir", type=str, default="inference_out")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PDF page rendering.")
    parser.add_argument("--use-bottom-up", action="store_true")
    parser.add_argument(
        "--out-indices", type=int, nargs=3, default=[1, 2, 3],
        help="Three backbone feature map indices to extract (default: 1 2 3).",
    )
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
    model = MusicDetector(num_classes=num_classes, use_bottom_up=args.use_bottom_up, out_indices=args.out_indices)
    load_checkpoint(model, args.checkpoint, device)

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

            tensor, meta = preprocess(page_image, args.input_size, device)
            outputs = model(tensor)
            batch_preds = decode_model_outputs(outputs, args.conf_thresh, args.iou_thresh, args.input_size)

            detections_dict = batch_preds[0]

            detections = [
                {
                    "class": class_list[int(c)],
                    "score": round(float(s), 4),
                    
                    "bbox": [
                        round(float(((bx1 - meta["pad_x"]) / meta["scale"]).clamp(min=0.0)), 1),
                        round(float(((by1 - meta["pad_y"]) / meta["scale"]).clamp(min=0.0)), 1),
                        round(float(((bx2 - meta["pad_x"]) / meta["scale"]).clamp(max=float(meta["orig_w"]))), 1),
                        round(float(((by2 - meta["pad_y"]) / meta["scale"]).clamp(max=float(meta["orig_h"]))), 1),
                    ]
                }
                for bx1, by1, bx2, by2, s, c in zip(
                    detections_dict["boxes"][:, 0], detections_dict["boxes"][:, 1], 
                    detections_dict["boxes"][:, 2], detections_dict["boxes"][:, 3], 
                    detections_dict["scores"], detections_dict["labels"]
                )
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
