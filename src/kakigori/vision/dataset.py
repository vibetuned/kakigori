"""OMR Dataset — loads rendered score images and their bounding-box annotations."""
import json
import random
from pathlib import Path
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True

class OMRDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_dir: str,
        class_list: list[str],
        input_size: int = 640,
        augment: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.input_size = input_size
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
            import albumentations as A
            
            # Dynamically calculate hole sizes based on the padded input_size
            max_hole = max(10, self.input_size // 20)
            min_hole = 5
            
            self.scanner_pipeline = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
                A.GaussNoise(std_range=(2/255.0, 8/255.0), p=0.5),
                A.CoarseDropout(
                    num_holes_range=(5, 20),
                    hole_height_range=(min_hole, max_hole),
                    hole_width_range=(min_hole, max_hole),
                    fill=240, 
                    p=0.5
                ),
                A.ImageCompression(quality_range=(40, 75), p=0.3)
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, ann_path = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
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
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)

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

        # 1. C++ Albumentations pass
        img = self.scanner_pipeline(image=img)["image"]

        # 2. NumPy / OpenCV manual passes
        if np.random.rand() < 0.4:
            kernel_size = np.random.choice([2, 3])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            img = cv2.dilate(img, kernel, iterations=1)

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

            intensity = np.random.uniform(10, 35)
            img = np.clip(img.astype(np.float32) + (grad - 0.5) * intensity, 0, 255).astype(np.uint8)

        return Image.fromarray(img)


# ---------------------------------------------------------------------------
# Letterbox helper
# ---------------------------------------------------------------------------

def _letterbox(
    image: Image.Image,
    size: int,
    boxes: torch.Tensor,
) -> tuple[Image.Image, torch.Tensor]:
    """Resize image to (size x size) with grey padding, preserving aspect ratio.

    Boxes are assumed to be in normalised (cx, cy, w, h) format relative to the
    original image. They are remapped to stay valid relative to the padded canvas.
    """
    orig_w, orig_h = image.size
    scale = min(size / orig_w, size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Resize without distortion
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Paste onto a grey canvas of the target size
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(image, (pad_x, pad_y))

    # Remap box coordinates to the padded canvas
    if boxes.numel() > 0:
        # Convert cx/cy from original-normalised to pixel coords on canvas
        cx = boxes[:, 0] * new_w + pad_x
        cy = boxes[:, 1] * new_h + pad_y
        bw = boxes[:, 2] * new_w
        bh = boxes[:, 3] * new_h
        # Renormalise to canvas size
        boxes = torch.stack([
            cx / size,
            cy / size,
            bw / size,
            bh / size,
        ], dim=1)

    return canvas, boxes

