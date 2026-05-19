# measure_psnr.py
# Correct PSNR measurement: model_output vs target (NOT vs raw input)
#
# Usage:
#   python measure_psnr.py \
#     --checkpoint checkpoints\enhancer_best.pt \
#     --data "C:\...\data\train" \
#     --targets "C:\...\data\targets"

import os
import math
import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import EnhanceDataset
from model import ResidualUNet


def psnr(pred, target):
    """PSNR between pred and target. Both [B,3,H,W] float32 in [0,1]."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def ssim(pred, target, window_size=11):
    """SSIM between pred and target."""
    C1, C2 = 0.01**2, 0.03**2
    mu_p  = F.avg_pool2d(pred,    window_size, 1, window_size//2)
    mu_t  = F.avg_pool2d(target,  window_size, 1, window_size//2)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
    sig_p  = F.avg_pool2d(pred**2,     window_size, 1, window_size//2) - mu_p2
    sig_t  = F.avg_pool2d(target**2,   window_size, 1, window_size//2) - mu_t2
    sig_pt = F.avg_pool2d(pred*target, window_size, 1, window_size//2) - mu_pt
    num = (2*mu_pt + C1)*(2*sig_pt + C2)
    den = (mu_p2 + mu_t2 + C1)*(sig_p + sig_t + C2) + 1e-8
    return (num/den).mean().item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data",       required=True, help="Raw input images folder")
    ap.add_argument("--targets",    default=None,  help="Pre-computed targets folder")
    ap.add_argument("--samples",    type=int, default=50)
    ap.add_argument("--crop",       type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model — supports both old {"model":...} and new {"G":...} checkpoints
    model = ResidualUNet(base=48).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("G", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    ds = EnhanceDataset(args.data, crop_size=args.crop,
                        training=False, targets_root=args.targets)

    n      = min(args.samples, len(ds))
    idxs   = random.sample(range(len(ds)), k=n)

    psnrs, ssims, l1s = [], [], []

    for idx in idxs:
        x, y, path = ds[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        pred = model(x)

        psnrs.append(psnr(pred, y))
        ssims.append(ssim(pred, y))
        l1s.append(F.l1_loss(pred, y).item())

    print(f"\n{'='*45}")
    print(f"  CORRECT METRICS (output vs TARGET)")
    print(f"  Samples evaluated: {n}")
    print(f"{'='*45}")
    print(f"  PSNR  (↑ higher=better): {sum(psnrs)/len(psnrs):.2f} dB")
    print(f"  SSIM  (↑ higher=better): {sum(ssims)/len(ssims):.4f}")
    print(f"  L1    (↓ lower=better):  {sum(l1s)/len(l1s):.5f}")
    print(f"{'='*45}")
    print(f"\n  NOTE: PSNR is measured against TARGET image")
    print(f"  (target = recipe-enhanced version of raw input)")
    print(f"  This is the correct way to evaluate this model.")


if __name__ == "__main__":
    main()