# metrics.py
import os
import argparse
import random
import math

import torch
import torch.nn.functional as F

from dataset import EnhanceDataset
from model import ResidualUNet


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse.item())


def save_image_tensor(path, x):
    import cv2
    import numpy as np
    x = x.detach().cpu().clamp(0, 1).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    x = x.transpose(1, 2, 0)  # HWC RGB
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgr)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--targets", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="metrics_out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    ds = EnhanceDataset(args.data, crop_size=256, training=False, targets_root=args.targets)

    model = ResidualUNet(base=48).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("G", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()

    idxs = random.sample(range(len(ds)), k=min(args.num_samples, len(ds)))

    l1s, psnrs = [], []

    for i, idx in enumerate(idxs):
        x, y, path = ds[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        pred = model(x)

        l1 = F.l1_loss(pred, y).item()
        p = psnr(pred, y)

        l1s.append(l1)
        psnrs.append(p)

        save_image_tensor(os.path.join(args.outdir, f"{i:03d}_inp.png"), x[0])
        save_image_tensor(os.path.join(args.outdir, f"{i:03d}_pred.png"), pred[0])
        save_image_tensor(os.path.join(args.outdir, f"{i:03d}_tgt.png"), y[0])

    print(f"Samples: {len(idxs)}")
    print(f"Mean L1:   {sum(l1s)/len(l1s):.5f}")
    print(f"Mean PSNR: {sum(psnrs)/len(psnrs):.2f} dB")
    print("Saved images to:", args.outdir)


if __name__ == "__main__":
    main()