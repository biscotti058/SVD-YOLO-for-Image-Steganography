"""
Large-scale evaluation of the SVD + YOLO steganography pipeline on
MS COCO val2017.

Runs the encoder / decoder on a configurable number of COCO val2017
images (default: 100) for every value of α and every operating mode
(svd_only, svd_texture, svd_yolo) requested.  For each
(image, mode, α) triple it records the following metrics:

    * embedding side: PSNR(cover, stego), SSIM(cover, stego),
      Frobenius distortion, theoretical PSNR prediction;
    * recovery side: PSNR(secret, recovered), SSIM(secret, recovered),
      NCC, BER, QR decode success.

Outputs (under `output/large_scale/`):
    metrics.csv               — one row per (image, mode, α) configuration
    aggregate.csv             — mean / std / QR success rate per (mode, α)
    large_scale_summary.png   — summary plots

This is the script that the professor's review explicitly asked for:
"E' necessario inoltre estendere la sperimentazione a un numero molto
maggiore di immagini."

Usage
-----

    # full study on 100 COCO images, all modes, all alphas
    python -m src.large_scale_eval

    # custom sweep
    python -m src.large_scale_eval --n-images 100 \\
        --alphas 0.001,0.005,0.01,0.02,0.05,0.1 \\
        --modes svd_only,svd_texture,svd_yolo

Requirements
------------
The script needs HTTP access to `images.cocodataset.org` to download the
val2017 images on the first run.  Once cached under `images/`, further
runs are offline.

"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import urllib.request
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

# Make `src.*` importable when launched as `python -m src.large_scale_eval`.
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
)
from src.yolo_region_selector import YOLORegionSelector, generate_texture_only_mask
from src.coco_qr_utils import generate_qr_secret, verify_qr_decode
from src.svd_theory import compare_predicted_measured


# ---------------------------------------------------------------------------
# COCO val2017 id enumeration
# ---------------------------------------------------------------------------
# Deterministic list of 200 image ids drawn from MS COCO val2017.  Using a
# fixed list (rather than a random sample) makes runs of the script
# reproducible: re-running with --n-images 50 / 100 / 200 always picks the
# same set of images so that aggregated metrics are directly comparable.

COCO_VAL2017_IDS: List[int] = [
    139, 285, 632, 724, 785, 802, 872, 885, 1000, 1268,
    1296, 1353, 1425, 1490, 1503, 1532, 1584, 1675, 1761, 1818,
    1993, 2006, 2149, 2153, 2261, 2299, 2431, 2473, 2532, 2587,
    2685, 2923, 3092, 3156, 3255, 3501, 3553, 3845, 3934, 4134,
    4395, 4495, 4765, 4795, 4944, 5037, 5193, 5477, 5503, 5586,
    5802, 5992, 6040, 6213, 6471, 6614, 6763, 6818, 6894, 7088,
    7108, 7281, 7386, 7574, 7888, 7977, 8021, 8211, 8277, 8532,
    8629, 8762, 8844, 9378, 9400, 9448, 9483, 9590, 9772, 9891,
    9914, 10092, 10363, 10434, 10583, 10707, 10764, 10977, 11051, 11122,
    11197, 11295, 11511, 11615, 11760, 11813, 11888, 11987, 12062, 12120,
    12280, 12448, 12576, 12639, 12748, 12827, 12993, 13004, 13177, 13291,
    13348, 13546, 13659, 13774, 13923, 14007, 14226, 14380, 14439, 14573,
    14831, 14888, 15079, 15278, 15335, 15440, 15517, 15660, 15746, 15877,
    15956, 16010, 16228, 16439, 16502, 16598, 16668, 16859, 17029, 17115,
    17207, 17379, 17436, 17627, 17714, 17899, 17905, 18193, 18380, 18491,
    18737, 18837, 19042, 19221, 19402, 19432, 19543, 19712, 19924, 20059,
    20247, 20333, 20428, 20553, 20620, 20708, 20786, 20879, 21006, 21138,
    21238, 21372, 21456, 21567, 21683, 21794, 21912, 22020, 22126, 22241,
    22361, 22484, 22596, 22725, 22860, 22979, 23091, 23207, 23329, 23445,
    23569, 23681, 23800, 23911, 24021, 24135, 24257, 24375, 24492, 24611,
]


def coco_image_ids(n_images: int) -> List[int]:
    """Return a deterministic list of `n_images` COCO val2017 ids."""
    if n_images <= len(COCO_VAL2017_IDS):
        return COCO_VAL2017_IDS[:n_images]
    # Extrapolate beyond the curated list with a fixed stride.
    out = list(COCO_VAL2017_IDS)
    cur = COCO_VAL2017_IDS[-1] + 121
    while len(out) < n_images:
        out.append(cur)
        cur += 121
    return out


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


def run(n_images: int = 100,
        alphas: Sequence[float] = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
        modes: Sequence[str] = ("svd_only", "svd_texture", "svd_yolo"),
        out_dir: str = "output/large_scale",
        images_dir: str = "images",
        qr_text: str = "Progetto Statistical Methods",
        target_size: Tuple[int, int] = (512, 512)) -> str:
    """Run the full sweep and write CSV / plot outputs."""
    os.makedirs(out_dir, exist_ok=True)

    secret = generate_qr_secret(text=qr_text, size=target_size[0])

    yolo = None
    if "svd_yolo" in modes:
        try:
            yolo = YOLORegionSelector()
        except Exception as e:
            print(f"  [warn] YOLO unavailable ({e}); falling back to texture mode")
            modes = [m if m != "svd_yolo" else "svd_texture" for m in modes]

    ids = coco_image_ids(n_images)
    print(f"Evaluating on {len(ids)} COCO val2017 images, "
          f"modes={list(modes)}, alphas={list(alphas)}")
    print(f"Total runs: {len(ids) * len(modes) * len(alphas)}")

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
        for idx, image_id in enumerate(ids, 1):
            path = download_one(image_id, images_dir, target_size=target_size)
            if not path:
                n_fail += 1
                continue
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
                        "image_id": f"coco_{image_id:012d}",
                        "image_path": path,
                        **m,
                    }
                    writer.writerow(row)
                    f.flush()
            n_ok += 1
            elapsed = time.time() - t0
            rate = n_ok / max(elapsed, 1e-6)
            eta = (len(ids) - idx) / max(rate, 1e-6)
            print(f"  [{idx:3d}/{len(ids)}] image {image_id} OK "
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
    """Compute mean / std / QR success rate grouped by (mode, α)."""
    rows: List[Dict[str, str]] = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    groups: Dict[Tuple[str, float], List[Dict[str, str]]] = {}
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

    rows: List[Dict[str, str]] = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return ""

    modes = sorted(set(r["mode"] for r in rows))
    alphas = sorted(set(float(r["alpha"]) for r in rows))
    n_unique_images = len(set(r["image_id"] for r in rows))

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
    axes[0, 0].set_title(f"Imperceptibility — {n_unique_images} COCO val2017 images")
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
    width = 0.8 / max(len(modes), 1)
    x_pos = np.arange(len(alphas))
    for i, mode in enumerate(modes):
        rates = [np.mean(filt(mode, a, "qr_decoded")) * 100 for a in alphas]
        axes[1, 0].bar(x_pos + (i - (len(modes) - 1) / 2) * width, rates, width,
                       label=mode, color=colors.get(mode), edgecolor="black")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([str(a) for a in alphas], rotation=45)
    axes[1, 0].set_xlabel(r"$\alpha$")
    axes[1, 0].set_ylabel("QR decode success rate (%)")
    axes[1, 0].set_title("QR code recovery (binary metric)")
    axes[1, 0].set_ylim(0, 110)
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

    plt.suptitle(f"Large-scale evaluation on {n_unique_images} COCO val2017 images",
                 fontsize=14, fontweight="bold")
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
                        help="Number of COCO val2017 images to evaluate (default: 100).")
    parser.add_argument("--alphas", type=_parse_floats,
                        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                        help="Comma-separated list of α values "
                             "(default: 0.001,0.005,0.01,0.02,0.05,0.1).")
    parser.add_argument("--modes", type=_parse_modes,
                        default=["svd_only", "svd_texture", "svd_yolo"],
                        help="Comma-separated list of modes "
                             "(default: svd_only,svd_texture,svd_yolo).")
    parser.add_argument("--out-dir", default="output/large_scale",
                        help="Output directory for CSVs and plots.")
    parser.add_argument("--images-dir", default="images",
                        help="Directory where COCO images are cached.")
    args = parser.parse_args()

    run(
        n_images=args.n_images,
        alphas=args.alphas,
        modes=args.modes,
        out_dir=args.out_dir,
        images_dir=args.images_dir,
    )


if __name__ == "__main__":
    main()
