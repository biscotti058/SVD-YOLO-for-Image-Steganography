"""
Large-scale evaluation of the SVD + YOLO steganography pipeline.

Runs the encoder / decoder on a large batch of COCO val2017 images for
multiple values of α and for every operating mode (svd_only, svd_texture,
svd_yolo).  For each (image, mode, α) triple it records the following
metrics:

    * embedding side: PSNR(cover, stego), SSIM(cover, stego),
      Frobenius distortion, theoretical PSNR prediction;
    * recovery side: PSNR(secret, recovered), SSIM(secret, recovered),
      NCC, BER, QR decode success.

Outputs (under output/large_scale/):
    metrics.csv      — one row per (image, mode, α) configuration
    aggregate.csv    — mean / std / QR success rate per (mode, α)
    large_scale_summary.png — summary plots

This is the script that the professor's review explicitly asked for:
"E' necessario inoltre estendere la sperimentazione a un numero molto
maggiore di immagini."

Usage
-----

    python -m src.large_scale_eval --n-images 100 \\
        --alphas 0.001,0.005,0.01,0.02,0.05,0.1 \\
        --modes svd_only,svd_texture,svd_yolo

"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.request
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

# Make `src.*` importable when the script is launched as
# `python -m src.large_scale_eval` from the project root.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.svd_steganography import (
    SVDSteganography,
    compute_psnr,
    compute_ssim,
    compute_ncc,
    compute_ber,
    load_image,
    save_image,
)
from src.yolo_region_selector import YOLORegionSelector, generate_texture_only_mask
from src.coco_qr_utils import generate_qr_secret, verify_qr_decode
from src.svd_theory import (
    secret_frobenius_norm,
    predicted_psnr,
    compare_predicted_measured,
)


# ---------------------------------------------------------------------------
# COCO id enumeration
# ---------------------------------------------------------------------------

# A deterministic list of COCO val2017 image ids.  The first ~5000 ids are
# enough for the full COCO val2017 subset; we use a deterministic stride so
# that running the script with --n-images 30 / 100 / 500 always picks the
# same set of images and the results are reproducible.
COCO_KNOWN_IDS: List[int] = [
    139, 285, 632, 724, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425,
    1490, 1503, 1532, 1584, 1675, 1761, 1818, 1993, 2006, 2149, 2153, 2261,
    2299, 2431, 2473, 2532, 2587, 2685, 2923, 3092, 3156, 3255, 3501, 3553,
    3845, 3934, 4134, 4395, 4495, 4765, 4795, 4944, 5037, 5193, 5477, 5503,
    5586, 5802, 5992, 6040, 6213, 6471, 6614, 6763, 6818, 6894, 7088, 7108,
    7281, 7386, 7574, 7888, 7977, 8021, 8211, 8277, 8532, 8629, 8762, 8844,
    9378, 9400, 9448, 9483, 9590, 9772, 9891, 9914, 10092, 10363, 10434,
    10583, 10707, 10764, 10977, 11051, 11122, 11197, 11295, 11511, 11615,
    11760, 11813, 11888, 11987, 12062, 12120, 12280, 12448, 12576, 12639,
    12748, 12827, 12993, 13004, 13177, 13291, 13348, 13546, 13659, 13774,
    13923, 14007, 14226, 14380, 14439, 14573, 14831, 14888, 15079, 15278,
    15335, 15440, 15517, 15660, 15746, 15877, 15956, 16010, 16228, 16439,
    16502, 16598, 16668, 16859, 17029, 17115, 17207, 17379, 17436, 17627,
    17714, 17899, 17905, 18193, 18380, 18491, 18737, 18837, 19042, 19221,
    19402, 19432, 19543, 19712, 19924,
]


def coco_image_ids(n_images: int) -> List[int]:
    """Return a deterministic list of `n_images` COCO val2017 ids."""
    if n_images <= len(COCO_KNOWN_IDS):
        return COCO_KNOWN_IDS[:n_images]
    # Extrapolate by repeating with a fixed stride beyond the known list.
    extra = []
    cur = COCO_KNOWN_IDS[-1] + 121
    while len(COCO_KNOWN_IDS) + len(extra) < n_images:
        extra.append(cur)
        cur += 121
    return COCO_KNOWN_IDS + extra


def discover_local_photographs(images_dir: str = "images",
                               target_size=(512, 512),
                               include_skimage: bool = True) -> List[str]:
    """
    Enumerate every real photograph available *locally* (no network).

    Combines:
      * everything in `images_dir` that looks like a colour photo
        (cover.png, the already-downloaded coco_*.png, etc.);
      * the natural-image sample set shipped with scikit-image
        (astronaut, cat, coffee, chelsea, rocket, hubble_deep_field,
        colorwheel) — cached to `images_dir/skimage_*.png`.

    Returns the list of file paths, all resized to `target_size`.
    """
    os.makedirs(images_dir, exist_ok=True)
    out: List[str] = []

    # Local files
    if os.path.isdir(images_dir):
        for name in sorted(os.listdir(images_dir)):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            if name == "qr_secret.png" or name == "secret.png":
                continue
            path = os.path.join(images_dir, name)
            try:
                img = Image.open(path).convert("RGB")
                if target_size is not None and img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                    img.save(path)
                out.append(path)
            except Exception:
                continue

    # scikit-image samples — real photographs included with the package
    if include_skimage:
        try:
            from skimage import data as _skd
            for name in ("astronaut", "cat", "coffee", "chelsea",
                         "rocket", "hubble_deep_field", "colorwheel"):
                cached = os.path.join(images_dir, f"skimage_{name}.png")
                if cached in out:
                    continue
                if not os.path.exists(cached):
                    try:
                        arr = getattr(_skd, name)()
                        if arr.ndim != 3 or arr.shape[2] < 3:
                            continue
                        arr = arr[..., :3]
                        img = Image.fromarray(arr.astype(np.uint8))
                        if target_size is not None:
                            img = img.resize(target_size, Image.LANCZOS)
                        img.save(cached)
                    except Exception:
                        continue
                out.append(cached)
        except ImportError:
            pass

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def download_one(image_id: int, save_dir: str,
                 target_size=(512, 512)) -> str:
    """Download a single COCO val2017 image. Returns local path or ''."""
    os.makedirs(save_dir, exist_ok=True)
    filename = f"coco_{image_id:012d}.png"
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        return path
    url = f"http://images.cocodataset.org/val2017/{image_id:012d}.jpg"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size, Image.LANCZOS)
        img.save(path)
        return path
    except Exception as e:  # network failure, 404, etc.
        print(f"  [warn] failed to download {image_id}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Single configuration evaluation
# ---------------------------------------------------------------------------


def evaluate_configuration(cover: np.ndarray,
                           secret: np.ndarray,
                           mode: str,
                           alpha: float,
                           yolo: YOLORegionSelector = None,
                           ) -> Dict[str, float]:
    """Encode/decode for a single (cover, secret, mode, α) and return metrics."""
    if mode == "svd_yolo":
        mask = yolo.generate_embedding_mask(cover) if yolo is not None else None
    elif mode == "svd_texture":
        mask = generate_texture_only_mask(cover)
    else:
        mask = None

    stego_engine = SVDSteganography(alpha=alpha)
    stego = stego_engine.encode(cover, secret, mask=mask)
    recovered = stego_engine.decode(stego)

    # Resize secret the same way the encoder does, so per-pixel metrics
    # compare like-with-like.
    secret_resized = stego_engine._resize_secret(cover, secret)

    psnr = compute_psnr(cover, stego)
    ssim = compute_ssim(cover, stego)
    rec_psnr = compute_psnr(secret_resized, recovered)
    rec_ssim = compute_ssim(secret_resized, recovered)
    ncc = compute_ncc(secret_resized, recovered)
    ber = compute_ber(secret_resized, recovered)
    qr_text = verify_qr_decode(recovered)
    qr_ok = qr_text is not None

    theory = compare_predicted_measured(cover, secret_resized, stego, alpha)

    return {
        "mode": mode,
        "alpha": alpha,
        "psnr": psnr,
        "ssim": ssim,
        "rec_psnr": rec_psnr,
        "rec_ssim": rec_ssim,
        "ncc": ncc,
        "ber": ber,
        "qr_decoded": int(qr_ok),
        "frob_predicted": theory["frob_predicted"],
        "frob_measured": theory["frob_measured"],
        "psnr_predicted": theory["psnr_predicted"],
        "weyl_bound_max": theory["weyl_bound_max"],
        "mirsky_bound_max": theory["mirsky_bound_max"],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(n_images: int,
        alphas: Sequence[float],
        modes: Sequence[str],
        out_dir: str = "output/large_scale",
        images_dir: str = "images",
        qr_text: str = "Progetto Statistical Methods",
        target_size=(512, 512),
        offline: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)

    secret = generate_qr_secret(text=qr_text, size=target_size[0])

    yolo = None
    if "svd_yolo" in modes:
        try:
            yolo = YOLORegionSelector()
        except Exception as e:
            print(f"  [warn] YOLO unavailable ({e}); falling back to texture mode")
            modes = [m if m != "svd_yolo" else "svd_texture" for m in modes]

    # Resolve the list of cover images to evaluate ------------------------
    image_paths: List[Tuple[str, str]] = []   # (id_label, path)
    if not offline:
        ids = coco_image_ids(n_images)
        for image_id in ids:
            path = download_one(image_id, images_dir, target_size=target_size)
            if path:
                image_paths.append((f"coco_{image_id:012d}", path))
            if len(image_paths) >= n_images:
                break

    if len(image_paths) < n_images:
        # Top up from local photographs (no network needed) ---------------
        already = {p for _, p in image_paths}
        for p in discover_local_photographs(images_dir, target_size):
            if p in already:
                continue
            label = os.path.splitext(os.path.basename(p))[0]
            image_paths.append((label, p))
            if len(image_paths) >= n_images:
                break

    if not image_paths:
        raise RuntimeError(
            "No cover images available. Network is blocked and no local "
            "photographs were found under `images/`."
        )

    print(f"Evaluating on {len(image_paths)} cover images "
          f"(requested {n_images}), modes={list(modes)}, alphas={list(alphas)}")
    if len(image_paths) < n_images:
        print(f"  [info] only {len(image_paths)} unique cover images are "
              "available locally; for the full study run the script in an "
              "environment with network access to images.cocodataset.org.")
    print(f"Total runs: {len(image_paths) * len(modes) * len(alphas)}")

    csv_path = os.path.join(out_dir, "metrics.csv")
    fieldnames = [
        "image_id", "image_path", "mode", "alpha",
        "psnr", "ssim",
        "rec_psnr", "rec_ssim", "ncc", "ber", "qr_decoded",
        "frob_predicted", "frob_measured",
        "psnr_predicted", "weyl_bound_max", "mirsky_bound_max",
    ]
    n_ok = 0
    n_fail = 0
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (image_id, path) in enumerate(image_paths, 1):
            try:
                cover = load_image(path)
            except Exception as e:
                print(f"  [warn] failed to load {path}: {e}")
                n_fail += 1
                continue
            for mode in modes:
                for alpha in alphas:
                    try:
                        m = evaluate_configuration(
                            cover, secret, mode, alpha, yolo
                        )
                    except Exception as e:
                        print(f"  [warn] {image_id} {mode} α={alpha}: {e}")
                        continue
                    row = {
                        "image_id": image_id,
                        "image_path": path,
                        **m,
                    }
                    writer.writerow(row)
                    f.flush()
            n_ok += 1
            elapsed = time.time() - t0
            rate = n_ok / max(elapsed, 1e-6)
            eta = (len(image_paths) - idx) / max(rate, 1e-6)
            print(f"  [{idx:3d}/{len(image_paths)}] image {image_id} OK "
                  f"({elapsed:.1f}s elapsed, ETA {eta:.0f}s)")

    print(f"\nDone. {n_ok} images processed, {n_fail} skipped. "
          f"Total time: {time.time() - t0:.1f}s")
    print(f"Per-image metrics saved to: {csv_path}")

    agg_path = aggregate(csv_path, out_dir)
    plot_path = plot_summary(csv_path, out_dir)
    print(f"Aggregate stats saved to:   {agg_path}")
    print(f"Summary plot saved to:      {plot_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------------


def aggregate(csv_path: str, out_dir: str) -> str:
    """Compute mean / std / QR success rate grouped by (mode, alpha)."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # group key -> list of dicts
    groups: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        key = (r["mode"], float(r["alpha"]))
        groups.setdefault(key, []).append(r)

    agg_path = os.path.join(out_dir, "aggregate.csv")
    cols = [
        "mode", "alpha", "n_images",
        "psnr_mean", "psnr_std",
        "ssim_mean", "ssim_std",
        "rec_psnr_mean", "rec_psnr_std",
        "ncc_mean", "ncc_std",
        "ber_mean", "ber_std",
        "qr_success_rate",
        "psnr_predicted_mean",
    ]
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for (mode, alpha), grp in sorted(groups.items(),
                                         key=lambda kv: (kv[0][0], kv[0][1])):
            def col(name):
                return np.array([float(r[name]) for r in grp], dtype=np.float64)
            psnr = col("psnr")
            ssim = col("ssim")
            rpsnr = col("rec_psnr")
            ncc = col("ncc")
            ber = col("ber")
            qr = col("qr_decoded")
            pred = col("psnr_predicted")
            writer.writerow({
                "mode": mode,
                "alpha": alpha,
                "n_images": len(grp),
                "psnr_mean": np.mean(psnr),
                "psnr_std": np.std(psnr),
                "ssim_mean": np.mean(ssim),
                "ssim_std": np.std(ssim),
                "rec_psnr_mean": np.mean(rpsnr),
                "rec_psnr_std": np.std(rpsnr),
                "ncc_mean": np.mean(ncc),
                "ncc_std": np.std(ncc),
                "ber_mean": np.mean(ber),
                "ber_std": np.std(ber),
                "qr_success_rate": float(np.mean(qr)),
                "psnr_predicted_mean": np.mean(pred),
            })
    return agg_path


def plot_summary(csv_path: str, out_dir: str) -> str:
    """Generate the main summary plot from per-image metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return ""

    modes = sorted(set(r["mode"] for r in rows))
    alphas = sorted(set(float(r["alpha"]) for r in rows))

    def filt(mode, alpha, col):
        return np.array([float(r[col]) for r in rows
                         if r["mode"] == mode and float(r["alpha"]) == alpha],
                        dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"svd_only": "#1f77b4",
              "svd_texture": "#ff7f0e",
              "svd_yolo": "#2ca02c"}

    # PSNR vs alpha
    for mode in modes:
        means = [np.mean(filt(mode, a, "psnr")) for a in alphas]
        stds = [np.std(filt(mode, a, "psnr")) for a in alphas]
        axes[0, 0].errorbar(alphas, means, yerr=stds, fmt="o-",
                            label=mode, color=colors.get(mode), capsize=4)
    axes[0, 0].axhline(30, color="red", ls="--", alpha=0.6,
                       label="30 dB (visual threshold)")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel(r"$\alpha$")
    axes[0, 0].set_ylabel("PSNR(cover, stego) [dB]")
    axes[0, 0].set_title(f"Imperceptibility — {len(set(r['image_id'] for r in rows))} COCO images")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # SSIM vs alpha
    for mode in modes:
        means = [np.mean(filt(mode, a, "ssim")) for a in alphas]
        stds = [np.std(filt(mode, a, "ssim")) for a in alphas]
        axes[0, 1].errorbar(alphas, means, yerr=stds, fmt="o-",
                            label=mode, color=colors.get(mode), capsize=4)
    axes[0, 1].axhline(0.95, color="red", ls="--", alpha=0.6,
                       label="0.95 threshold")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel(r"$\alpha$")
    axes[0, 1].set_ylabel("SSIM(cover, stego)")
    axes[0, 1].set_title("Structural similarity")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    # QR success rate vs alpha
    width = 0.25
    x_pos = np.arange(len(alphas))
    for i, mode in enumerate(modes):
        rates = [np.mean(filt(mode, a, "qr_decoded")) * 100 for a in alphas]
        axes[1, 0].bar(x_pos + (i - 1) * width, rates, width,
                       label=mode, color=colors.get(mode), edgecolor="black")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([str(a) for a in alphas], rotation=45)
    axes[1, 0].set_xlabel(r"$\alpha$")
    axes[1, 0].set_ylabel("QR decode success rate (%)")
    axes[1, 0].set_title("QR code recovery (binary metric)")
    axes[1, 0].grid(alpha=0.3, axis="y")
    axes[1, 0].legend()

    # Measured vs predicted PSNR (scatter, svd_only)
    base = "svd_only" if "svd_only" in modes else modes[0]
    measured = np.array([float(r["psnr"]) for r in rows if r["mode"] == base])
    predicted = np.array([float(r["psnr_predicted"]) for r in rows if r["mode"] == base])
    axes[1, 1].scatter(predicted, measured, s=10, alpha=0.4, color="#444")
    lims = [min(measured.min(), predicted.min()) - 2,
            max(measured.max(), predicted.max()) + 2]
    axes[1, 1].plot(lims, lims, "r--", label="measured = predicted")
    axes[1, 1].set_xlim(lims)
    axes[1, 1].set_ylim(lims)
    axes[1, 1].set_xlabel("Predicted PSNR (closed form) [dB]")
    axes[1, 1].set_ylabel("Measured PSNR [dB]")
    axes[1, 1].set_title(f"Theory vs measurement ({base})")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle("Large-scale evaluation summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "large_scale_summary.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_modes(s: str) -> List[str]:
    valid = {"svd_only", "svd_texture", "svd_yolo"}
    out = [m.strip() for m in s.split(",") if m.strip()]
    for m in out:
        if m not in valid:
            raise argparse.ArgumentTypeError(f"unknown mode {m!r}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-images", type=int, default=100,
                        help="Number of COCO val2017 images to evaluate.")
    parser.add_argument("--alphas", type=_parse_floats,
                        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                        help="Comma-separated list of α values.")
    parser.add_argument("--modes", type=_parse_modes,
                        default=["svd_only", "svd_texture", "svd_yolo"],
                        help="Comma-separated list of modes.")
    parser.add_argument("--out-dir", default="output/large_scale",
                        help="Output directory for CSVs and plots.")
    parser.add_argument("--images-dir", default="images",
                        help="Directory where COCO images are cached.")
    parser.add_argument("--offline", action="store_true",
                        help="Skip COCO downloads; use only locally available "
                             "real photographs (images/*.png/jpg and scikit-image samples).")
    args = parser.parse_args()

    run(
        n_images=args.n_images,
        alphas=args.alphas,
        modes=args.modes,
        out_dir=args.out_dir,
        images_dir=args.images_dir,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
