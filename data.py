"""Synthetic object-detection dataset built with OpenCV.

Each image is 128x128 with 1-4 shapes (circle / square / triangle) drawn on a
noisy background. Ground truth is a list of (class_id, x1, y1, x2, y2) boxes.
Generating data on the fly means no downloads and unlimited training samples.
"""
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

CLASSES = ["circle", "square", "triangle"]
IMG = 128          # image size in pixels
S = 8              # grid size -> each cell covers 16x16 pixels
C = len(CLASSES)


def _iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def make_sample(rng):
    """Return (image HxWx3 uint8, boxes list[(cls,x1,y1,x2,y2)])."""
    bg = rng.integers(20, 90)
    img = np.full((IMG, IMG, 3), bg, np.uint8)
    img = img + rng.normal(0, 8, img.shape)           # gaussian sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)

    boxes, n_target = [], int(rng.integers(1, 5))
    tries = 0
    while len(boxes) < n_target and tries < 40:
        tries += 1
        cls = int(rng.integers(0, C))
        size = int(rng.integers(20, 40))
        cx = int(rng.integers(size // 2 + 2, IMG - size // 2 - 2))
        cy = int(rng.integers(size // 2 + 2, IMG - size // 2 - 2))
        box = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)
        # keep shapes mostly separate and ensure one object per grid cell centre
        cell = (cx * S // IMG, cy * S // IMG)
        if any(_iou_xyxy(box, b[1:]) > 0.05 for b in boxes):
            continue
        if any(((b[1] + b[3]) // 2 * S // IMG, (b[2] + b[4]) // 2 * S // IMG) == cell for b in boxes):
            continue
        color = tuple(int(v) for v in rng.integers(150, 255, 3))
        if cls == 0:
            cv2.circle(img, (cx, cy), size // 2, color, -1)
        elif cls == 1:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, -1)
        else:
            pts = np.array([[cx, box[1]], [box[0], box[3]], [box[2], box[3]]], np.int32)
            cv2.fillPoly(img, [pts], color)
        boxes.append((cls, *box))
    return img, boxes


def encode_target(boxes):
    """Boxes -> grid target of shape (S, S, 5 + C): [obj, x, y, w, h, class one-hot].
    x, y are the box centre offset *within* its cell (0-1); w, h relative to image."""
    t = np.zeros((S, S, 5 + C), np.float32)
    for cls, x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / IMG, (y1 + y2) / 2 / IMG
        gi, gj = min(int(cx * S), S - 1), min(int(cy * S), S - 1)
        t[gj, gi, 0] = 1
        t[gj, gi, 1] = cx * S - gi
        t[gj, gi, 2] = cy * S - gj
        t[gj, gi, 3] = (x2 - x1) / IMG
        t[gj, gi, 4] = (y2 - y1) / IMG
        t[gj, gi, 5 + cls] = 1
    return t


class ShapeDataset(Dataset):
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.samples = [make_sample(rng) for _ in range(n)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, boxes = self.samples[i]
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return x, torch.from_numpy(encode_target(boxes)), i
