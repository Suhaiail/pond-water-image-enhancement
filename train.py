# train.py — PondNet GAN Training
#
# Loss breakdown:
#   Generator:     1.0×L1 + 0.15×Edge + 0.25×SSIM + 0.05×Perceptual + 0.10×GAN + 0.30×ColourCorrection
#   Discriminator: LSGAN with spectral norm + label smoothing (real=0.9)
#
# Stability measures:
#   - 10 epoch generator warmup before discriminator activates
#   - Spectral normalisation on discriminator
#   - Label smoothing (real=0.9, fake=0.0)
#   - Low GAN weight (0.10) — L1 still dominates
#   - Gradient clipping on both G and D
#   - isfinite check on all losses

import os
import argparse
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EnhanceDataset
from model import ResidualUNet, PatchDiscriminator


# ══════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════

def sobel_edges(x):
    gray = 0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                       device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                       device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-6)


def ssim_loss(pred, target, window_size=11):
    """1 - SSIM. Range [0,1], lower=better."""
    C1, C2 = 0.01**2, 0.03**2
    mu_p  = F.avg_pool2d(pred,    window_size, 1, window_size//2)
    mu_t  = F.avg_pool2d(target,  window_size, 1, window_size//2)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
    sig_p  = F.avg_pool2d(pred**2,     window_size, 1, window_size//2) - mu_p2
    sig_t  = F.avg_pool2d(target**2,   window_size, 1, window_size//2) - mu_t2
    sig_pt = F.avg_pool2d(pred*target, window_size, 1, window_size//2) - mu_pt
    num = (2*mu_pt + C1) * (2*sig_pt + C2)
    den = (mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2) + 1e-8
    return 1.0 - (num/den).mean()


def colour_correction_loss(pred):
    """
    Pond-specific loss: penalises green channel dominance.
    Green/yellow cast is the primary degradation in pond water.
    Encourages balanced RGB channels in the output.
    """
    r = pred[:,0].mean()
    g = pred[:,1].mean()
    b = pred[:,2].mean()
    # Penalise green exceeding red or blue
    green_dom = F.relu(g - r) + F.relu(g - b)
    # Penalise overall imbalance
    mean_all  = (r + g + b) / 3.0
    balance   = (r-mean_all).abs() + (g-mean_all).abs() + (b-mean_all).abs()
    return green_dom + 0.5 * balance


class PerceptualLoss(nn.Module):
    """VGG16 relu2_2 perceptual loss — preserves texture and structure."""
    def __init__(self, device):
        super().__init__()
        try:
            from torchvision import models
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.feat    = nn.Sequential(*list(vgg.features)[:10]).to(device)
            for p in self.feat.parameters():
                p.requires_grad = False
            self.enabled = True
            self.mean = torch.tensor([0.485,0.456,0.406],
                                      device=device).view(1,3,1,1)
            self.std  = torch.tensor([0.229,0.224,0.225],
                                      device=device).view(1,3,1,1)
            print("Perceptual loss: VGG16 relu2_2 ✓")
        except Exception as e:
            print(f"Perceptual loss disabled ({e}) — install torchvision")
            self.enabled = False

    def forward(self, pred, target):
        if not self.enabled:
            return torch.tensor(0.0, device=pred.device)
        p = (pred   - self.mean) / self.std
        t = (target - self.mean) / self.std
        return F.l1_loss(self.feat(p), self.feat(t))


# ══════════════════════════════════════════════════════════════
# LR SCHEDULE
# ══════════════════════════════════════════════════════════════

def get_lr(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * max(step, 1) / warmup_steps
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi*t))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr


# ══════════════════════════════════════════════════════════════
# SAVE UTIL
# ══════════════════════════════════════════════════════════════

def save_image_tensor(path, x):
    import cv2, numpy as np
    x = x.detach().cpu().clamp(0,1).numpy()
    x = (x*255).round().astype(np.uint8).transpose(1,2,0)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(x, cv2.COLOR_RGB2BGR))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",        type=str, required=True)
    ap.add_argument("--targets",     type=str, default=None)
    ap.add_argument("--outdir",      type=str, default="checkpoints")
    ap.add_argument("--epochs",      type=int, default=100)
    ap.add_argument("--batch",       type=int, default=4)
    ap.add_argument("--crop",        type=int, default=256)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--lr_d",        type=float, default=4e-5)
    ap.add_argument("--warmup",      type=int, default=300,
                    help="LR warmup steps")
    ap.add_argument("--gan_warmup",  type=int, default=10,
                    help="Epochs of generator-only training before GAN activates")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-split",   type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    # ── Dataset ──
    full_ds     = EnhanceDataset(args.data, crop_size=args.crop,
                                  training=True, targets_root=args.targets)
    all_paths   = full_ds.paths[:]
    random.seed(42)
    random.shuffle(all_paths)
    n_val       = max(1, int(len(all_paths) * args.val_split))
    val_paths   = all_paths[:n_val]
    train_paths = all_paths[n_val:]

    train_ds       = EnhanceDataset(args.data, crop_size=args.crop,
                                     training=True, targets_root=args.targets)
    train_ds.paths = train_paths
    val_ds         = EnhanceDataset(args.data, crop_size=args.crop,
                                     training=False, targets_root=args.targets)
    val_ds.paths   = val_paths

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Device: {device}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers,
                          pin_memory=(device=="cuda"), drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=0)

    # ── Models ──
    G = ResidualUNet(base=48).to(device)
    D = PatchDiscriminator().to(device)

    opt_G = torch.optim.AdamW(G.parameters(), lr=args.lr,  weight_decay=1e-4,
                               betas=(0.5, 0.999))
    opt_D = torch.optim.AdamW(D.parameters(), lr=args.lr_d, weight_decay=1e-4,
                               betas=(0.5, 0.999))

    perc_loss = PerceptualLoss(device)

    total_steps = len(train_dl) * args.epochs
    print(f"Total steps: {total_steps} | LR warmup: {args.warmup} | GAN warmup: {args.gan_warmup} epochs")

    best_val = float('inf')
    g_step   = 0

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        gan_active = (epoch > args.gan_warmup)

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}"
                    + (" [GAN ON]" if gan_active else " [warmup]"))

        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ── LR schedule (generator) ──
            lr = get_lr(g_step, total_steps, args.warmup, args.lr)
            set_lr(opt_G, lr)

            # ════════════════════════════════
            # TRAIN DISCRIMINATOR
            # ════════════════════════════════
            d_loss_val = 0.0
            if gan_active:
                D.requires_grad_(True)
                opt_D.zero_grad(set_to_none=True)

                with torch.no_grad():
                    fake = G(x)

                # Real: target paired with input
                pred_real  = D(x, y)
                label_real = torch.ones_like(pred_real) * 0.9  # label smoothing
                loss_real  = F.mse_loss(pred_real, label_real)

                # Fake: generator output paired with input
                pred_fake  = D(x, fake.detach())
                label_fake = torch.zeros_like(pred_fake)
                loss_fake  = F.mse_loss(pred_fake, label_fake)

                d_loss     = 0.5 * (loss_real + loss_fake)

                if torch.isfinite(d_loss):
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                    opt_D.step()
                    d_loss_val = d_loss.item()

            # ════════════════════════════════
            # TRAIN GENERATOR
            # ════════════════════════════════
            D.requires_grad_(False)
            opt_G.zero_grad(set_to_none=True)

            pred = G(x)

            # Pixel losses
            l1   = F.l1_loss(pred, y)
            edge = F.l1_loss(sobel_edges(pred), sobel_edges(y))
            ssim = ssim_loss(pred, y)
            perc = perc_loss(pred, y)
            cc   = colour_correction_loss(pred)

            # Adversarial loss
            g_adv = torch.tensor(0.0, device=device)
            if gan_active:
                pred_fake = D(x, pred)
                g_adv     = F.mse_loss(pred_fake, torch.ones_like(pred_fake))

            g_loss = (1.00 * l1   +
                      0.15 * edge +
                      0.25 * ssim +
                      0.05 * perc +
                      0.10 * g_adv +
                      0.30 * cc)

            if torch.isfinite(g_loss):
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                opt_G.step()

            g_step += 1
            pbar.set_postfix(
                G=f"{g_loss.item():.4f}",
                l1=f"{l1.item():.4f}",
                ssim=f"{ssim.item():.4f}",
                D=f"{d_loss_val:.4f}",
                lr=f"{lr:.2e}"
            )

        # ── Validation ──
        G.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for x, y, _ in val_dl:
                x = x.to(device)
                y = y.to(device)
                val_l1 += F.l1_loss(G(x), y).item()
        val_l1 /= len(val_dl)
        print(f"  → Epoch {epoch} val L1: {val_l1:.5f}"
              + (" [GAN active]" if gan_active else " [generator warmup]"))

        if val_l1 < best_val:
            best_val = val_l1
            torch.save({"G": G.state_dict(),
                        "D": D.state_dict(),
                        "epoch": epoch,
                        "val_l1": best_val},
                       os.path.join(args.outdir, "enhancer_best.pt"))
            print(f"  ✓ New best saved (val_L1={best_val:.5f})")

    print(f"\nDone. Best val L1: {best_val:.5f}")


if __name__ == "__main__":
    main()