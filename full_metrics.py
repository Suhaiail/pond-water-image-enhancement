# full_metrics.py — PondNet Comprehensive Evaluation
#
# Reference metrics: enhanced output vs TARGET (correct!)
# No-reference metrics: enhanced output only
#
# Usage:
#   python full_metrics.py \
#     --input    "C:\...\data\train" \
#     --targets  "C:\...\data\targets" \
#     --enhanced "C:\...\enhanced_out" \
#     --output   metrics_full.csv \
#     --limit    50

import os, argparse, glob, math, warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import torch, piq
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

IMG_EXTS = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff")


# ── Image loading ──────────────────────────────────────────────────────────

def load_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def list_images(folder, limit=999999):
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(folder,"**",ext), recursive=True)
    return sorted(paths)[:limit]


# ── Reference-based metrics (output vs TARGET) ─────────────────────────────

def compute_psnr(pred_bgr, tgt_bgr):
    """PSNR: pred vs target. Higher=better. >25dB is good."""
    if HAS_SKIMAGE:
        p = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        t = cv2.cvtColor(tgt_bgr,  cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        return float(sk_psnr(t, p, data_range=1.0))
    mse = np.mean((pred_bgr.astype(np.float32) - tgt_bgr.astype(np.float32))**2)
    if mse < 1e-10: return 99.0
    return float(10 * math.log10(255**2 / mse))


def compute_ssim(pred_bgr, tgt_bgr):
    """SSIM: pred vs target. Higher=better. >0.85 is good."""
    if HAS_SKIMAGE:
        p = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        t = cv2.cvtColor(tgt_bgr,  cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        return float(sk_ssim(t, p, channel_axis=2, data_range=1.0))
    # fallback
    p = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)/255.0
    t = cv2.cvtColor(tgt_bgr,  cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)/255.0
    C1, C2 = 0.01**2, 0.03**2
    mu1,mu2 = p.mean(), t.mean()
    s1,s2   = p.std(), t.std()
    s12     = np.mean((p-mu1)*(t-mu2))
    return float(((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1**2+s2**2+C2)+1e-8))


def compute_ms_ssim(pred_bgr, tgt_bgr, levels=3):
    """MS-SSIM across 3 scales."""
    p = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    t = cv2.cvtColor(tgt_bgr,  cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    scores = []
    for _ in range(levels):
        if p.shape[0] < 16 or p.shape[1] < 16: break
        if HAS_SKIMAGE:
            scores.append(sk_ssim(t, p, data_range=1.0))
        p = cv2.resize(p, (p.shape[1]//2, p.shape[0]//2))
        t = cv2.resize(t, (t.shape[1]//2, t.shape[0]//2))
    return float(np.mean(scores)) if scores else 0.0


def compute_lpips_proxy(pred_bgr, tgt_bgr):
    """Gradient-based perceptual proxy. Lower=better."""
    def grad(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        gx = cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
        gy = cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
        return np.stack([gx,gy],axis=-1)
    g1,g2 = grad(pred_bgr), grad(tgt_bgr)
    return float(np.mean(np.abs(g1-g2)))


# ── No-reference metrics (enhanced only) ──────────────────────────────────

def compute_niqe(img_bgr):
    """NIQE — lower=more natural. Uses piq if available."""
    if HAS_PIQ:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            t   = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0)
            return float(piq.niqe(t, data_range=1.0).item())
        except: pass
    # Fallback
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0
    from scipy.ndimage import uniform_filter
    mu    = uniform_filter(gray, 7)
    mu2   = uniform_filter(gray**2, 7)
    sigma = np.sqrt(np.maximum(mu2 - mu**2, 0))
    mscn  = (gray - mu) / (sigma + 1.0)
    vals  = []
    for y in range(0, gray.shape[0]-96, 48):
        for x in range(0, gray.shape[1]-96, 48):
            p = mscn[y:y+96,x:x+96]
            if np.var(p) > 0.005:
                mu_p = np.mean(np.abs(p))
                if mu_p > 1e-8:
                    vals.append(abs(np.var(p)/(mu_p**2+1e-10)-2.0))
    return float(np.clip(np.mean(vals)*3.0,0,15)) if vals else 5.0


def compute_brisque(img_bgr):
    """BRISQUE — lower=better. Uses piq if available."""
    if HAS_PIQ:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            t   = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0)
            return float(piq.brisque(t, data_range=1.0).item())
        except: pass
    return 50.0


def compute_sharpness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_brightness(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return float(lab[:,:,0].mean() / 255.0 * 100.0)

def compute_saturation(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:,:,1].mean() / 255.0 * 100.0)

def compute_contrast(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    return float(gray.std())

def compute_colorfulness(img_bgr):
    rgb  = img_bgr.astype(np.float32)
    R,G,B = rgb[:,:,2], rgb[:,:,1], rgb[:,:,0]
    rg   = R - G
    yb   = 0.5*(R+G) - B
    return float(np.sqrt(rg.std()**2 + yb.std()**2) + 0.3*np.sqrt(rg.mean()**2 + yb.mean()**2))

def compute_entropy(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    hist = hist[hist>0] / hist.sum()
    return float(-np.sum(hist * np.log2(hist)))

def compute_noise_estimate(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    blur  = cv2.GaussianBlur(gray,(0,0),1.0)
    return float(np.std(gray - blur))

def compute_uciqe(img_bgr):
    """UCIQE — underwater-specific. Higher=better."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L   = lab[:,:,0]/255.0
    a   = lab[:,:,1] - 128.0
    b   = lab[:,:,2] - 128.0
    chroma = np.sqrt(a**2 + b**2)
    sat    = chroma / (L * 255.0 + 1e-6)
    con_l  = L.std()
    avg_sat= sat.mean()
    avg_chr= chroma.mean() / 100.0
    return float(0.4680*con_l + 0.2745*avg_chr + 0.2576*avg_sat)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",    required=True,  help="Raw input images folder")
    ap.add_argument("--targets",  required=True,  help="Target images folder (pre-computed)")
    ap.add_argument("--enhanced", required=True,  help="Model output folder")
    ap.add_argument("--output",   default="metrics_full.csv")
    ap.add_argument("--limit",    type=int, default=50)
    args = ap.parse_args()

    # Build filename maps
    inp_map = {os.path.basename(p): p for p in list_images(args.input,   args.limit)}
    tgt_map = {os.path.splitext(os.path.basename(p))[0]+".png": p
               for p in list_images(args.targets, 999999)}
    # Also try matching without extension change
    for p in list_images(args.targets, 999999):
        tgt_map[os.path.basename(p)] = p
    enh_map = {os.path.basename(p): p for p in list_images(args.enhanced, 999999)}

    common = sorted(set(inp_map.keys()) & set(enh_map.keys()))
    if not common:
        print("No matching filenames between input and enhanced folders!")
        return

    print(f"Evaluating {min(len(common), args.limit)} image pairs...")
    print(f"Reference: enhanced output vs TARGET (correct PSNR)")
    print()

    rows = []
    metric_keys = ["PSNR","SSIM","MS-SSIM","LPIPS-proxy",
                   "Sharpness","Brightness","Saturation","Contrast",
                   "Colorfulness","Entropy","Noise","NIQE","BRISQUE","UCIQE"]
    accum = {k: {"inp":[], "enh":[], "tgt":[]} for k in metric_keys}

    for fname in common[:args.limit]:
        try:
            inp_bgr = load_bgr(inp_map[fname])
            enh_bgr = load_bgr(enh_map[fname])

            # Find target — try exact name first, then stem+.png
            stem    = os.path.splitext(fname)[0]
            tgt_path= tgt_map.get(fname) or tgt_map.get(stem+".png")
            if tgt_path:
                tgt_bgr = load_bgr(tgt_path)
                if tgt_bgr.shape != enh_bgr.shape:
                    tgt_bgr = cv2.resize(tgt_bgr,
                                         (enh_bgr.shape[1], enh_bgr.shape[0]))
            else:
                tgt_bgr = None

            if inp_bgr.shape != enh_bgr.shape:
                inp_bgr = cv2.resize(inp_bgr,
                                     (enh_bgr.shape[1], enh_bgr.shape[0]))

            row = {"filename": fname}

            # Reference metrics (vs target)
            if tgt_bgr is not None:
                row["PSNR"]       = compute_psnr(enh_bgr, tgt_bgr)
                row["SSIM"]       = compute_ssim(enh_bgr, tgt_bgr)
                row["MS-SSIM"]    = compute_ms_ssim(enh_bgr, tgt_bgr)
                row["LPIPS-proxy"]= compute_lpips_proxy(enh_bgr, tgt_bgr)
            else:
                row["PSNR"] = row["SSIM"] = row["MS-SSIM"] = row["LPIPS-proxy"] = float('nan')

            # No-reference metrics
            row["Sharpness"]   = compute_sharpness(enh_bgr)
            row["Brightness"]  = compute_brightness(enh_bgr)
            row["Saturation"]  = compute_saturation(enh_bgr)
            row["Contrast"]    = compute_contrast(enh_bgr)
            row["Colorfulness"]= compute_colorfulness(enh_bgr)
            row["Entropy"]     = compute_entropy(enh_bgr)
            row["Noise"]       = compute_noise_estimate(enh_bgr)
            row["NIQE"]        = compute_niqe(enh_bgr)
            row["BRISQUE"]     = compute_brisque(enh_bgr)
            row["UCIQE"]       = compute_uciqe(enh_bgr)

            # Also compute for input (for comparison)
            row["inp_Sharpness"]   = compute_sharpness(inp_bgr)
            row["inp_Brightness"]  = compute_brightness(inp_bgr)
            row["inp_Saturation"]  = compute_saturation(inp_bgr)
            row["inp_Colorfulness"]= compute_colorfulness(inp_bgr)
            row["inp_UCIQE"]       = compute_uciqe(inp_bgr)

            rows.append(row)

        except Exception as e:
            print(f"  Skipped {fname}: {e}")

    if not rows:
        print("No images evaluated!")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in rows if key in r and not math.isnan(r[key])]
        return sum(vals)/len(vals) if vals else float('nan')

    print("=" * 65)
    print("  SUMMARY — MEAN ACROSS ALL IMAGES")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Enhanced':>10}   {'Change vs Input':>18}")
    print("-" * 65)

    ref_metrics = [
        ("PSNR (dB) ↑",    "PSNR",        None,             True),
        ("SSIM ↑",         "SSIM",         None,             True),
        ("MS-SSIM ↑",      "MS-SSIM",      None,             True),
        ("LPIPS-proxy ↓",  "LPIPS-proxy",  None,             False),
    ]
    print("  [Reference-based: output vs TARGET]")
    for label, key, inp_key, higher_better in ref_metrics:
        val = avg(key)
        print(f"  {label:<22} {val:>10.4f}   {'(vs target)':>18}")

    print()
    print("  [No-reference: enhanced image quality]")
    nr_metrics = [
        ("Sharpness ↑",    "Sharpness",    "inp_Sharpness",   True),
        ("Brightness ↑",   "Brightness",   "inp_Brightness",  True),
        ("Saturation ↑",   "Saturation",   "inp_Saturation",  True),
        ("Contrast ↑",     "Contrast",     None,              True),
        ("Colorfulness ↑", "Colorfulness", "inp_Colorfulness",True),
        ("Entropy ↑",      "Entropy",      None,              True),
        ("Noise ↓",        "Noise",        None,              False),
        ("NIQE ↓",         "NIQE",         None,              False),
        ("BRISQUE ↓",      "BRISQUE",      None,              False),
        ("UCIQE ↑",        "UCIQE",        "inp_UCIQE",       True),
    ]
    for label, key, inp_key, higher_better in nr_metrics:
        val = avg(key)
        if inp_key:
            inp_val = avg(inp_key)
            diff    = val - inp_val
            pct     = (diff/inp_val*100) if inp_val != 0 else 0
            sign    = "+" if diff > 0 else ""
            ok      = "✓" if (diff > 0) == higher_better else "✗"
            change  = f"{sign}{diff:+.3f} ({sign}{pct:.1f}%) {ok}"
        else:
            change  = "(no-reference)"
        print(f"  {label:<22} {val:>10.4f}   {change:>18}")

    print("=" * 65)
    print(f"  Images evaluated: {len(rows)}")
    print(f"  NOTE: PSNR/SSIM measured vs TARGET image (not raw input)")
    print("=" * 65)

    # ── Save CSV ───────────────────────────────────────────────────────────
    if HAS_PANDAS:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\n  Per-image results saved to: {args.output}")
    else:
        import csv
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"\n  Per-image results saved to: {args.output}")


if __name__ == "__main__":
    main()