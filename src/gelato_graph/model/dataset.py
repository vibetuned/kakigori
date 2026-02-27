"""OMR Dataset — loads rendered score images and their bounding-box annotations."""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF

# Support loading of truncated images (common if rendering was interrupted)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class OMRDataset(Dataset):
    """Dataset for Optical Music Recognition object detection.

    Each sample is a full-page score image paired with bounding-box annotations.
    Annotations are stored as JSON with format:
        {"image": "...", "width": W, "height": H,
         "annotations": [{"class": "note", "bbox": [x1, y1, x2, y2]}, ...]}

    Boxes are returned in *normalised* (cx, cy, w, h) format relative to the
    **model input size**, after resizing.
    """

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

        # Build class → index mapping (0-indexed)
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}

        # Pair images ↔ annotations by stem
        ann_stems = {p.stem for p in self.ann_dir.glob("*.json")}
        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(self.img_dir.glob("*.png")):
            if img_path.stem in ann_stems:
                ann_path = self.ann_dir / f"{img_path.stem}.json"
                self.samples.append((img_path, ann_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, ann_path = self.samples[idx]

        # --- Load image & annotations ------------------------------------
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, SyntaxError) as e:
            # If image is broken, pick a random different sample to avoid crashing the loop
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        with open(ann_path) as f:
            ann_data = json.load(f)

        orig_w, orig_h = image.size

        # --- Parse annotations -------------------------------------------
        boxes = []  # will be (cx, cy, w, h) normalised
        labels = []
        for ann in ann_data["annotations"]:
            cls_name = ann["class"]
            if cls_name not in self.class_to_idx:
                continue
            x1, y1, x2, y2 = ann["bbox"]
            # Normalise to [0, 1] relative to original image size
            cx = ((x1 + x2) / 2) / orig_w
            cy = ((y1 + y2) / 2) / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            # Skip degenerate boxes
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([cx, cy, bw, bh])
            labels.append(self.class_to_idx[cls_name])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4)
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)

        # --- Letterbox resize (preserve aspect ratio, pad with grey) ------
        image, boxes = _letterbox(image, self.input_size, boxes)

        # --- Augmentation (train-time only) ------------------------------
        if self.augment:
            image = _scanned_document_augment(image)

        # --- To tensor & normalise [0, 1] --------------------------------
        image = TF.to_tensor(image)  # (3, H, W), float32, [0, 1]

        targets = {"boxes": boxes, "labels": labels}
        return image, targets


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


# ---------------------------------------------------------------------------
# Scanned-document style augmentations
# ---------------------------------------------------------------------------

def _scanned_document_augment(image: Image.Image) -> Image.Image:
    """Apply augmentations that simulate a scanned/photocopied document."""
    import numpy as np
    import cv2
    import random
    from io import BytesIO

    img = np.array(image, dtype=np.float32)  # (H, W, 3), [0, 255]

    # 1. Random brightness / contrast jitter
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)  # contrast
        beta = random.uniform(-15, 15)       # brightness
        img = np.clip(alpha * img + beta, 0, 255)

    # 2. Gaussian noise (simulates scanner sensor noise)
    if random.random() < 0.5:
        sigma = random.uniform(2, 8)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255)

    # 3. Faded Ink (Morphological Erosion)
    # Thins out black pixels, simulating degraded or lightly printed ink
    if random.random() < 0.4:
        # We use dilation because our text is dark on a light background 
        # (dilating the light background erodes the dark text)
        kernel_size = random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

    # 4. Whiteout / Missing Chunks (Random Erasing)
    # Drops random white blobs to simulate holes, severe fading, or whiteout
    if random.random() < 0.5:
        h, w = img.shape[:2]
        num_holes = random.randint(5, 20)
        for _ in range(num_holes):
            # Size of the missing chunk
            hole_w = random.randint(5, max(10, w // 20))
            hole_h = random.randint(5, max(10, h // 20))
            # Position
            x = random.randint(0, w - hole_w)
            y = random.randint(0, h - hole_h)
            # Fill with a light color (simulating paper)
            paper_color = random.uniform(220, 255)
            img[y:y+hole_h, x:x+hole_w] = paper_color

    # 5. Gray gradient overlay
    if random.random() < 0.4:
        h, w = img.shape[:2]
        direction = random.choice(["horizontal", "vertical", "diagonal"])
        if direction == "horizontal":
            grad = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
            grad = np.broadcast_to(grad, img.shape)
        elif direction == "vertical":
            grad = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
            grad = np.broadcast_to(grad, img.shape)
        else:  # diagonal
            gx = np.linspace(0, 1, w, dtype=np.float32)
            gy = np.linspace(0, 1, h, dtype=np.float32)
            grad = (gx[None, :] + gy[:, None]) / 2
            grad = grad[:, :, None]
            grad = np.broadcast_to(grad, img.shape)

        if random.random() < 0.5:
            grad = 1.0 - grad

        intensity = random.uniform(10, 35)
        img = np.clip(img + (grad - 0.5) * intensity, 0, 255)

    # 6. Salt-and-pepper noise
    if random.random() < 0.3:
        sp_amount = random.uniform(0.001, 0.005)
        mask = np.random.random(img.shape[:2])
        img[mask < sp_amount / 2] = 0        # pepper
        img[mask > 1 - sp_amount / 2] = 255  # salt

    # 7. Slight JPEG-style compression artifacts
    if random.random() < 0.3:
        pil_img = Image.fromarray(img.astype(np.uint8))
        buf = BytesIO()
        quality = random.randint(40, 75)
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGB")
        img = np.array(pil_img, dtype=np.float32)

    return Image.fromarray(img.astype(np.uint8))