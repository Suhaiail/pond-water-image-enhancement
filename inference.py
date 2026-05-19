# inference.py
import os
import argparse

import cv2
import numpy as np
import torch

from dataset import list_images_recursive, unsharp_mask
from model import ResidualUNet


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_tensor01(rgb: np.ndarray) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def save_rgb_uint8(path: str, rgb: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--max-size", type=int, default=2000, help="resize longest side if too large")
    ap.add_argument("--final-sharpen", type=float, default=0.25, help="extra tiny polish sharpening")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResidualUNet(base=48).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("G", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()

    paths = list_images_recursive(args.input)
    print("Found images:", len(paths))

    processed = 0
    skipped = 0

    for p in paths:
        try:
            rgb = read_rgb(p)
        except Exception as e:
            skipped += 1
            continue

        h, w = rgb.shape[:2]
        mx = max(h, w)
        scale = 1.0
        rgb_small = rgb
        if mx > args.max_size:
            scale = args.max_size / mx
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            rgb_small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        x = to_tensor01(rgb_small).unsqueeze(0).to(device)

        pred = model(x)[0].clamp(0, 1).cpu().numpy()  # [3,H,W]
        pred = (pred * 255.0).round().astype(np.uint8).transpose(1, 2, 0)  # HWC RGB

        if scale != 1.0:
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)

        # Final tiny polish sharpen (optional, helps "premium" look)
        if args.final_sharpen > 0:
            pred_f = pred.astype(np.float32) / 255.0
            pred_f = unsharp_mask(pred_f)
            pred = (pred_f * 255.0).round().astype(np.uint8)

        rel = os.path.relpath(p, args.input)
        out_path = os.path.join(args.output, rel)
        save_rgb_uint8(out_path, pred)

        processed += 1

    print(f"Done. Output: {args.output}")
    print(f"Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()