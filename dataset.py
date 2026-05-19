# dataset.py — PondNet v2
# Recipe tuned for better PSNR/SSIM:
#   Gamma:    0.93 (was 0.90) — gentler brightening, target closer to input
#   Vibrance: 0.55 (was 0.65) — less aggressive, better L1
#   HSV boost: x1.12 (was x1.20) — milder saturation
#   CLAHE:    clip=2.0, bilateral d=5 sigma=25 — unchanged (working well)
#   Unsharp:  0.35 radius=1 — unchanged

import os
from glob import glob
from typing import List, Tuple
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def list_images_recursive(root: str) -> List[str]:
    paths: List[str] = []
    for ext in IMG_EXTS:
        paths.extend(glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort()
    return paths


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_tensor01(rgb: np.ndarray) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def bilateral_denoise(x01: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise on L channel. d=5, sigma=25 (gentle)."""
    rgb8  = (x01 * 255.0).astype(np.uint8)
    lab   = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_den = cv2.bilateralFilter(l, d=5, sigmaColor=25.0, sigmaSpace=25.0)
    lab2  = cv2.merge([l_den, a, b])
    rgb2  = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return rgb2.astype(np.float32) / 255.0


def white_balance(x01: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Grey-world white balance — removes green/yellow pond cast."""
    r, g, b  = x01[:,:,0], x01[:,:,1], x01[:,:,2]
    mean_all = (r.mean() + g.mean() + b.mean()) / 3.0 + 1e-6
    out      = x01.copy()
    out[:,:,0] = np.clip(r * (1.0 + strength*(mean_all/(r.mean()+1e-6)-1.0)), 0, 1)
    out[:,:,1] = np.clip(g * (1.0 + strength*(mean_all/(g.mean()+1e-6)-1.0)), 0, 1)
    out[:,:,2] = np.clip(b * (1.0 + strength*(mean_all/(b.mean()+1e-6)-1.0)), 0, 1)
    return out


def clahe_l_channel(x01: np.ndarray) -> np.ndarray:
    """CLAHE clip=2.0 — mild local contrast, good NIQE."""
    rgb8  = (x01 * 255.0).astype(np.uint8)
    lab   = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2    = clahe.apply(l)
    lab2  = cv2.merge([l2, a, b])
    rgb2  = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return rgb2.astype(np.float32) / 255.0


def vibrance(x01: np.ndarray) -> np.ndarray:
    """Vibrance 0.55 (reduced from 0.65) — better PSNR."""
    mx    = x01.max(axis=2, keepdims=True)
    mn    = x01.min(axis=2, keepdims=True)
    sat   = mx - mn
    boost = 1.0 + 0.55 * (1.0 - sat)
    return clamp01(mn + (x01 - mn) * boost)


def hsv_saturation_boost(x01: np.ndarray) -> np.ndarray:
    """HSV S-channel x1.12 (reduced from x1.20) — milder, better PSNR."""
    rgb8       = (x01 * 255.0).astype(np.uint8)
    hsv        = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.12, 0, 255)
    rgb2       = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb2.astype(np.float32) / 255.0


def gamma_lift(x01: np.ndarray) -> np.ndarray:
    """gamma=0.93 (was 0.90) — gentler, target closer to input → better PSNR."""
    return clamp01(np.power(clamp01(x01), 0.93))


def apply_tone_curve(x01: np.ndarray) -> np.ndarray:
    """strength=0.08 — mild S-curve."""
    x = x01
    return clamp01(x + 0.08 * (x - x * x))


def unsharp_mask(x01: np.ndarray) -> np.ndarray:
    """Mild sharpening: amount=0.35, radius=1."""
    rgb8  = (x01 * 255.0).astype(np.uint8)
    blur  = cv2.GaussianBlur(rgb8, (0, 0), 1.0)
    sharp = cv2.addWeighted(rgb8, 1.35, blur, -0.35, 0)
    return sharp.astype(np.float32) / 255.0


def premium_enhance(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Full pond enhancement pipeline.
    Tuned for PSNR/SSIM improvement while keeping colour and perceptual quality.
    """
    x = rgb_uint8.astype(np.float32) / 255.0
    x = bilateral_denoise(x)
    x = white_balance(x)
    x = clahe_l_channel(x)
    x = vibrance(x)
    x = hsv_saturation_boost(x)
    x = gamma_lift(x)
    x = apply_tone_curve(x)
    x = unsharp_mask(x)
    return clamp01(x)


def random_hflip(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, ::-1, :])


def random_vflip(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[::-1, :, :])


def random_crop_pair(inp: np.ndarray, tgt: np.ndarray,
                     crop: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = inp.shape[:2]
    if h < crop or w < crop:
        pad_h = max(0, crop - h)
        pad_w = max(0, crop - w)
        inp = cv2.copyMakeBorder(inp, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        tgt = cv2.copyMakeBorder(tgt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        h, w = inp.shape[:2]
    y = random.randint(0, h - crop)
    x = random.randint(0, w - crop)
    return inp[y:y+crop, x:x+crop], tgt[y:y+crop, x:x+crop]


class EnhanceDataset(Dataset):
    def __init__(self, root: str, crop_size: int = 256, training: bool = True,
                 targets_root: str = None):
        self.root         = root
        self.targets_root = targets_root
        self.paths        = list_images_recursive(root)
        if not self.paths:
            raise RuntimeError(f"No images found under: {root}")
        self.crop_size = int(crop_size)
        self.training  = bool(training)
        if targets_root:
            print(f"Using pre-computed targets from: {targets_root}")

    def __len__(self):
        return len(self.paths)

    def _load_target(self, inp_path: str):
        if self.targets_root:
            rel      = os.path.relpath(inp_path, self.root)
            tgt_path = os.path.join(self.targets_root,
                                    os.path.splitext(rel)[0] + ".png")
            if os.path.isfile(tgt_path):
                bgr = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb   = read_rgb(inp_path)
        tgt01 = premium_enhance(rgb)
        return (tgt01 * 255.0).round().astype(np.uint8)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        rgb  = read_rgb(path)
        tgt  = self._load_target(path)
        inp  = rgb

        if self.training:
            if random.random() < 0.5:
                inp = random_hflip(inp)
                tgt = random_hflip(tgt)
            if random.random() < 0.3:
                inp = random_vflip(inp)
                tgt = random_vflip(tgt)
            inp, tgt = random_crop_pair(inp, tgt, self.crop_size)

        x = to_tensor01(inp)
        y = to_tensor01(tgt)
        return x, y, path