"""OMR Dataset — loads rendered score images and their bounding-box annotations."""

# Standard library imports
import json
import random
from pathlib import Path

# Third party imports
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OMRDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_dir: str,
        class_list: list[str],
        input_size: int | tuple[int, int] = 640,
        augment: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.input_size = canvas_size(input_size)
        self.augment = augment

        self.class_to_idx = {c: i for i, c in enumerate(class_list)}

        ann_stems = {p.stem for p in self.ann_dir.glob("*.json")}
        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(self.img_dir.glob("*.png")):
            if img_path.stem in ann_stems:
                ann_path = self.ann_dir / f"{img_path.stem}.json"
                self.samples.append((img_path, ann_path))

        # --- Initialize Albumentations conditionally ---
        if self.augment:
            # Third party imports
            import albumentations as A

            # Dynamically calculate hole sizes based on the canvas
            max_hole = max(10, min(self.input_size) // 20)
            min_hole = 5

            self.scanner_pipeline = A.Compose(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.15, p=0.5
                    ),
                    A.GaussNoise(std_range=(2 / 255.0, 8 / 255.0), p=0.5),
                    A.CoarseDropout(
                        num_holes_range=(5, 20),
                        hole_height_range=(min_hole, max_hole),
                        hole_width_range=(min_hole, max_hole),
                        fill=240,
                        p=0.5,
                    ),
                    A.ImageCompression(quality_range=(40, 75), p=0.3),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, ann_path = self.samples[idx]

        try:
            image = Image.open(img_path).convert("L")
        except (OSError, SyntaxError) as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        with open(ann_path) as f:
            ann_data = json.load(f)

        orig_w, orig_h = image.size

        boxes = []
        labels = []
        for ann in ann_data["annotations"]:
            cls_name = ann["class"]
            if cls_name not in self.class_to_idx:
                continue
            x1, y1, x2, y2 = ann["bbox"]
            cx = ((x1 + x2) / 2) / orig_w
            cy = ((y1 + y2) / 2) / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([cx, cy, bw, bh])
            labels.append(self.class_to_idx[cls_name])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4)
        labels = (
            torch.tensor(labels, dtype=torch.long)
            if labels
            else torch.zeros(0, dtype=torch.long)
        )

        image, boxes = _letterbox(image, self.input_size, boxes)

        # --- Augmentation Call ---
        if self.augment:
            image = self._scanned_document_augment(image)

        image = TF.to_tensor(image)

        targets = {"boxes": boxes, "labels": labels}
        return image, targets

    def _scanned_document_augment(self, image: Image.Image) -> Image.Image:
        """Apply augmentations using the pre-initialized pipeline."""
        img = np.array(image)
        h, w = img.shape[:2]
        grey = img.ndim == 2

        # 1. C++ Albumentations pass (on an explicit channel axis)
        img = self.scanner_pipeline(image=img[..., None] if grey else img)["image"]
        if grey:
            img = img.reshape(h, w)

        # 2. NumPy manual passes. No morphology here: at 640 px stems, barlines
        # and staff lines are under a pixel wide, and cv2.dilate (a max filter,
        # it grows the paper) erased most of them while their boxes stayed in
        # the targets (docs/experiments/vision/augmentation_dilate.ipynb)
        if np.random.rand() < 0.3:
            sp_amount = np.random.uniform(0.001, 0.005)
            mask = np.random.rand(h, w)
            img[mask < sp_amount / 2] = 0
            img[mask > 1 - sp_amount / 2] = 255

        if np.random.rand() < 0.4:
            direction = np.random.choice(["horizontal", "vertical", "diagonal"])
            if direction == "horizontal":
                grad = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
            elif direction == "vertical":
                grad = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
            else:
                gx = np.linspace(0, 1, w, dtype=np.float32)
                gy = np.linspace(0, 1, h, dtype=np.float32)
                grad = ((gx[None, :] + gy[:, None]) / 2)[:, :, None]

            if np.random.rand() < 0.5:
                grad = 1.0 - grad
            if grey:
                grad = grad[..., 0]

            intensity = np.random.uniform(10, 35)
            img = np.clip(
                img.astype(np.float32) + (grad - 0.5) * intensity, 0, 255
            ).astype(np.uint8)

        return Image.fromarray(img)


# ---------------------------------------------------------------------------
# Letterbox helpers: the page is fitted, undistorted, into a (width, height)
# canvas and the remainder padded grey. A square canvas wastes ~30% of the
# input on padding for portrait pages; a page-shaped one (512x736 for A4)
# spends it on resolution instead.
# ---------------------------------------------------------------------------


def canvas_size(input_size) -> tuple[int, int]:
    """(width, height) of the network input; an int is a square canvas."""
    if isinstance(input_size, int):
        return (input_size, input_size)
    width, height = input_size
    return (int(width), int(height))


def letterbox_geometry(orig_w: int, orig_h: int, input_size) -> dict:
    """Where a page lands on the canvas. Boxes map with the per-axis factors
    of the rounded resize (``scale_x``/``scale_y``), not the unrounded
    ``scale``, which drifts up to half a canvas pixel at the far edge."""
    canvas_w, canvas_h = canvas_size(input_size)
    scale = min(canvas_w / orig_w, canvas_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    return {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "new_w": new_w,
        "new_h": new_h,
        "pad_x": (canvas_w - new_w) // 2,
        "pad_y": (canvas_h - new_h) // 2,
        "scale": scale,
        "scale_x": new_w / orig_w,
        "scale_y": new_h / orig_h,
    }


def letterbox_image(image: Image.Image, input_size) -> tuple[Image.Image, dict]:
    """The page fitted into the canvas, and its ``letterbox_geometry``."""
    geo = letterbox_geometry(*image.size, input_size)
    image = image.resize((geo["new_w"], geo["new_h"]), Image.BILINEAR)
    fill = 114 if image.mode == "L" else (114, 114, 114)
    canvas = Image.new(image.mode, (geo["canvas_w"], geo["canvas_h"]), fill)
    canvas.paste(image, (geo["pad_x"], geo["pad_y"]))
    return canvas, geo


def page_to_canvas(boxes: torch.Tensor, geo: dict) -> torch.Tensor:
    """(x1, y1, x2, y2) page pixels -> canvas pixels."""
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * geo["scale_x"] + geo["pad_x"]
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * geo["scale_y"] + geo["pad_y"]
    return boxes


def canvas_to_page(boxes: torch.Tensor, geo: dict) -> torch.Tensor:
    """(x1, y1, x2, y2) canvas pixels -> page pixels, clamped to the page."""
    boxes = boxes.clone()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - geo["pad_x"]) / geo["scale_x"]).clamp(0.0, float(geo["orig_w"]))
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - geo["pad_y"]) / geo["scale_y"]).clamp(0.0, float(geo["orig_h"]))
    return boxes


def _letterbox(
    image: Image.Image,
    size,
    boxes: torch.Tensor,
) -> tuple[Image.Image, torch.Tensor]:
    """Fit the page into the canvas (``size``: int or (width, height)).

    Boxes are normalised (cx, cy, w, h) relative to the page, and come back
    normalised relative to the canvas (x by its width, y by its height).
    """
    canvas, geo = letterbox_image(image, size)
    if boxes.numel() > 0:
        canvas_w, canvas_h = geo["canvas_w"], geo["canvas_h"]
        boxes = torch.stack(
            [
                (boxes[:, 0] * geo["new_w"] + geo["pad_x"]) / canvas_w,
                (boxes[:, 1] * geo["new_h"] + geo["pad_y"]) / canvas_h,
                boxes[:, 2] * geo["new_w"] / canvas_w,
                boxes[:, 3] * geo["new_h"] / canvas_h,
            ],
            dim=1,
        )
    return canvas, boxes
