"""
Detailed analysis plots from the large-scale evaluation CSVs.

Generates additional publication-style figures from
`output/large_scale/metrics.csv` (per-image, per-mode, per-α metrics),
beyond the single overview plot produced by `large_scale_eval`:

* `01_psnr_distribution.png`  — boxplots of PSNR(cover, stego) per (mode, α)
* `02_pareto_psnr_vs_qr.png`  — Pareto frontier imperceptibility vs robustness
* `03_theory_vs_measured.png` — scatter + R² of predicted vs measured PSNR
* `04_per_mode_curves.png`    — mean curves with confidence bands
* `05_ncc_distribution.png`   — NCC of recovered secret per (mode, α)
* `06_alpha_window.png`       — visual summary of the operating window

Run:
    python -m src.analyze_results
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _by(rows, mode=None, alpha=None):
    out = rows
    if mode is not None:
        out = [r for r in out if r["mode"] == mode]
    if alpha is not None:
        out = [r for r in out if abs(float(r["alpha"]) - alpha) < 1e-9]
    return out


def _arr(rows, col):
    return np.array([float(r[col]) for r in rows], dtype=np.float64)


def _ensure_alpha_eff(rows: List[Dict[str, str]]) -> None:
    """
    Backfill alpha_eff column for CSVs produced by an older version of the
    pipeline (which only stored the nominal alpha).  We recover it from the
    measured Frobenius distortion:

        ||C* - C||_F      = alpha_eff * ||S||_F                  (identity)
        frob_predicted    = alpha     * ||S||_F                  (definition)
        => alpha_eff      = alpha * frob_measured / frob_predicted.
    """
    if rows and "alpha_eff" in rows[0]:
        return
    for r in rows:
        alpha = float(r["alpha"])
        fm = float(r["frob_measured"])
        fp = float(r["frob_predicted"])
        r["alpha_eff"] = alpha * (fm / fp) if fp > 0 else alpha


def _psnr_from_frob(frob: float, n_pixels_total: float) -> float:
    """PSNR for an L2 distortion `frob` between two images of `n_pixels_total` cells."""
    if frob <= 0:
        return float("inf")
    mse = frob ** 2 / n_pixels_total
    return float(10.0 * np.log10(255.0 ** 2 / mse))


MODE_COLORS = {
    "svd_only": "#1f77b4",
    "svd_texture": "#ff7f0e",
    "svd_yolo": "#2ca02c",
}


def plot_psnr_boxplot(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted(set(r["mode"] for r in rows))
    alphas = sorted(set(float(r["alpha"]) for r in rows))

    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.8 / len(modes)
    positions = np.arange(len(alphas))
    for i, mode in enumerate(modes):
        data = [_arr(_by(rows, mode=mode, alpha=a), "psnr") for a in alphas]
        bp = ax.boxplot(
            data,
            positions=positions + (i - (len(modes) - 1) / 2) * width,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=4, alpha=0.4),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(MODE_COLORS.get(mode, "gray"))
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("black")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("PSNR(cover, stego) [dB]")
    ax.axhline(30, color="red", ls="--", alpha=0.5, label="30 dB (visual threshold)")
    ax.set_title(f"Distribution of PSNR across {len(set(r['image_id'] for r in rows))} COCO val2017 covers")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS.get(m), alpha=0.75) for m in modes]
    handles.append(plt.Line2D([0], [0], color="red", ls="--"))
    labels = modes + ["30 dB threshold"]
    ax.legend(handles, labels, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted(set(r["mode"] for r in rows))
    alphas = sorted(set(float(r["alpha"]) for r in rows))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for mode in modes:
        xs, ys, labels = [], [], []
        for a in alphas:
            grp = _by(rows, mode=mode, alpha=a)
            xs.append(np.mean(_arr(grp, "psnr")))
            ys.append(np.mean(_arr(grp, "qr_decoded")) * 100)
            labels.append(a)
        ax.plot(xs, ys, "o-", color=MODE_COLORS.get(mode), lw=2,
                markersize=9, label=mode)
        for x, y, a in zip(xs, ys, labels):
            ax.annotate(f"α={a}", (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color=MODE_COLORS.get(mode))
    ax.axvline(30, color="red", ls="--", alpha=0.5, label="30 dB visual threshold")
    ax.axhline(95, color="green", ls=":", alpha=0.5, label="95% QR success")
    ax.set_xlabel("Mean PSNR(cover, stego) [dB]  →  more imperceptible")
    ax.set_ylabel("QR decode success rate [%]  →  more robust")
    ax.set_title("Pareto frontier: imperceptibility ↔ robustness")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_theory_vs_measured(rows, out_path):
    """
    Two columns × three rows:
        left:  prediction made with the *nominal* alpha
        right: prediction made with the *effective* alpha = alpha · (0.1 + 0.9·mean_mask)

    Only the right-hand column is a fair test of the Frobenius identity
    for masked modes.  The left-hand column quantifies the attenuation of
    the embedding caused by the mask, expressed in dB.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ["svd_only", "svd_texture", "svd_yolo"]
    fig, axes = plt.subplots(len(modes), 2, figsize=(13, 14))

    for row_idx, mode in enumerate(modes):
        grp = _by(rows, mode=mode)
        pred_nominal = _arr(grp, "psnr_predicted")  # built with alpha nominal
        meas = _arr(grp, "psnr")
        alpha = _arr(grp, "alpha")
        alpha_eff = _arr(grp, "alpha_eff")
        # Closed-form prediction with alpha_eff: shift by 20*log10(alpha/alpha_eff).
        pred_eff = pred_nominal + 20.0 * np.log10(alpha / alpha_eff)

        for col_idx, (pred, label) in enumerate(
                [(pred_nominal, "nominal α"), (pred_eff, "effective α")]):
            ax = axes[row_idx, col_idx]
            delta = meas - pred
            ssr = np.sum((meas - pred) ** 2)
            sst = np.sum((meas - np.mean(meas)) ** 2)
            r2 = 1 - ssr / sst if sst > 0 else float("nan")
            mean_delta = float(np.mean(delta))

            sc = ax.scatter(pred, meas, c=np.log10(alpha), s=14, alpha=0.5,
                            cmap="viridis")
            lims = [min(meas.min(), pred.min()) - 2,
                    max(meas.max(), pred.max()) + 2]
            ax.plot(lims, lims, "r--", label="measured = predicted")
            ax.set_title(
                f"{mode} — prediction with {label}\n"
                f"R² = {r2:.3f},  mean(Δ) = {mean_delta:+.2f} dB"
            )
            ax.set_xlabel(f"Predicted PSNR ({label}) [dB]")
            ax.set_ylabel("Measured PSNR [dB]")
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.grid(alpha=0.3); ax.legend(loc="upper left")
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(r"$\log_{10} \alpha$")

    plt.suptitle("Closed-form PSNR prediction vs measurement\n"
                 "Left column: nominal α (over-predicts distortion for masked modes).\n"
                 "Right column: effective α — the Frobenius identity is exact.",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_mode_curves(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted(set(r["mode"] for r in rows))
    alphas = sorted(set(float(r["alpha"]) for r in rows))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = [
        ("psnr", "PSNR(cover, stego) [dB]", "Imperceptibility", 30, "30 dB"),
        ("rec_psnr", "PSNR(secret, recovered) [dB]", "Recovery fidelity", None, None),
        ("ncc", "NCC(secret, recovered)", "Normalized cross-correlation", 0.95, "0.95"),
    ]
    for ax, (col, ylabel, title, thresh, thlabel) in zip(axes, metrics):
        for mode in modes:
            mean = np.array([np.mean(_arr(_by(rows, mode=mode, alpha=a), col)) for a in alphas])
            std = np.array([np.std(_arr(_by(rows, mode=mode, alpha=a), col)) for a in alphas])
            ax.plot(alphas, mean, "o-", lw=2, color=MODE_COLORS.get(mode),
                    label=mode)
            ax.fill_between(alphas, mean - std, mean + std,
                            color=MODE_COLORS.get(mode), alpha=0.15)
        if thresh is not None:
            ax.axhline(thresh, color="red", ls="--", alpha=0.6, label=thlabel)
        ax.set_xscale("log"); ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend()
    plt.suptitle("Per-mode curves with ±1σ band", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_ncc_boxplot(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted(set(r["mode"] for r in rows))
    alphas = sorted(set(float(r["alpha"]) for r in rows))

    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.8 / len(modes)
    positions = np.arange(len(alphas))
    for i, mode in enumerate(modes):
        data = [_arr(_by(rows, mode=mode, alpha=a), "ncc") for a in alphas]
        bp = ax.boxplot(
            data,
            positions=positions + (i - (len(modes) - 1) / 2) * width,
            widths=width * 0.85, patch_artist=True, showfliers=True,
            flierprops=dict(marker=".", markersize=4, alpha=0.4),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(MODE_COLORS.get(mode, "gray"))
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("black")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("NCC(secret, recovered)")
    ax.set_title(f"Distribution of NCC across {len(set(r['image_id'] for r in rows))} COCO val2017 covers")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS.get(m), alpha=0.75) for m in modes]
    ax.legend(handles, modes, loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_window(rows, out_path):
    """Show the operating window: where PSNR>=30 AND QR>=95% simultaneously."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted(set(r["mode"] for r in rows))
    alphas = sorted(set(float(r["alpha"]) for r in rows))

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / len(modes)
    positions = np.arange(len(alphas))
    for i, mode in enumerate(modes):
        psnr_means = [np.mean(_arr(_by(rows, mode=mode, alpha=a), "psnr"))
                      for a in alphas]
        qr_rates = [np.mean(_arr(_by(rows, mode=mode, alpha=a), "qr_decoded")) * 100
                    for a in alphas]
        for j, (p, q) in enumerate(zip(psnr_means, qr_rates)):
            color = MODE_COLORS.get(mode)
            # Box: imperceptible AND robust = green border
            ok_imperc = p >= 30
            ok_robust = q >= 95
            edge = "darkgreen" if (ok_imperc and ok_robust) else "black"
            lw = 2.5 if (ok_imperc and ok_robust) else 0.8
            xpos = positions[j] + (i - (len(modes) - 1) / 2) * width
            ax.bar(xpos, p, width=width * 0.85,
                   color=color, alpha=0.55, edgecolor=edge, linewidth=lw)
            ax.text(xpos, p + 1, f"{q:.0f}%", ha="center", fontsize=7,
                    color="black", fontweight="bold")
    ax.axhline(30, color="red", ls="--", alpha=0.7, label="PSNR=30 dB")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Mean PSNR(cover, stego) [dB]")
    ax.set_title("Operating window — green border = PSNR≥30 AND QR≥95% (labels: QR success rate)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS.get(m), alpha=0.55) for m in modes]
    handles.append(plt.Line2D([0], [0], color="red", ls="--"))
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="none", ec="darkgreen", lw=2.5))
    labels = modes + ["30 dB threshold", "operating point"]
    ax.legend(handles, labels, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows):
    modes = sorted(set(r["mode"] for r in rows))
    alphas = sorted(set(float(r["alpha"]) for r in rows))
    n_images = len(set(r["image_id"] for r in rows))
    print(f"\nLarge-scale evaluation: {n_images} unique COCO covers, "
          f"{len(modes)} modes × {len(alphas)} α values = {len(rows)} runs.\n")

    # Operating points: best (mode, α) by simple criterion
    # imperceptibility >= 30 dB AND qr_success_rate >= 0.95
    print("Operating points (mean PSNR ≥ 30 dB and QR success ≥ 95%):")
    best = []
    for mode in modes:
        for a in alphas:
            grp = _by(rows, mode=mode, alpha=a)
            psnr_mean = float(np.mean(_arr(grp, "psnr")))
            qr_rate = float(np.mean(_arr(grp, "qr_decoded")))
            if psnr_mean >= 30 and qr_rate >= 0.95:
                best.append((mode, a, psnr_mean, qr_rate * 100))
    if best:
        print(f"  {'mode':<14} {'alpha':>7}  {'PSNR':>7} dB   {'QR%':>5}")
        for mode, a, p, q in sorted(best, key=lambda t: (-t[2])):
            print(f"  {mode:<14} {a:>7.4f}  {p:>7.2f}    {q:>5.1f}")
    else:
        print("  none found.")
    print()

    # Theory verification (with nominal vs effective alpha)
    print("Theory vs measurement, prediction made with NOMINAL α:")
    for mode in modes:
        grp = _by(rows, mode=mode)
        delta = _arr(grp, "psnr") - _arr(grp, "psnr_predicted")
        print(f"  {mode:<14} mean Δ = {np.mean(delta):+.2f} dB   "
              f"max |Δ| = {np.max(np.abs(delta)):.2f} dB   "
              f"% Δ≥0 = {100 * np.mean(delta >= 0):.0f}%")

    print("\nTheory vs measurement, prediction made with EFFECTIVE α "
          "(Frobenius identity directly):")
    for mode in modes:
        grp = _by(rows, mode=mode)
        alpha = _arr(grp, "alpha")
        alpha_eff = _arr(grp, "alpha_eff")
        pred_eff = _arr(grp, "psnr_predicted") + 20.0 * np.log10(alpha / alpha_eff)
        delta = _arr(grp, "psnr") - pred_eff
        print(f"  {mode:<14} mean Δ = {np.mean(delta):+.2f} dB   "
              f"max |Δ| = {np.max(np.abs(delta)):.2f} dB   "
              f"% Δ≥0 = {100 * np.mean(delta >= 0):.0f}%")
    print("\n(Δ near 0 with effective α confirms ||C* - C||_F = α_eff · ||S||_F.)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", default="output/large_scale/metrics.csv",
                        help="Path to the per-image metrics CSV.")
    parser.add_argument("--out-dir", default="output/large_scale",
                        help="Directory where the analysis plots are written.")
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        raise SystemExit(
            f"Metrics CSV not found: {args.metrics}. "
            "Run `python -m src.large_scale_eval` first."
        )

    rows = _load(args.metrics)
    _ensure_alpha_eff(rows)
    os.makedirs(args.out_dir, exist_ok=True)

    print_summary(rows)

    print("Generating analysis plots...")
    plot_psnr_boxplot(rows,        os.path.join(args.out_dir, "01_psnr_distribution.png"))
    plot_pareto(rows,              os.path.join(args.out_dir, "02_pareto_psnr_vs_qr.png"))
    plot_theory_vs_measured(rows,  os.path.join(args.out_dir, "03_theory_vs_measured.png"))
    plot_per_mode_curves(rows,     os.path.join(args.out_dir, "04_per_mode_curves.png"))
    plot_ncc_boxplot(rows,         os.path.join(args.out_dir, "05_ncc_distribution.png"))
    plot_alpha_window(rows,        os.path.join(args.out_dir, "06_alpha_window.png"))
    print(f"\nAll plots written to {args.out_dir}/")


if __name__ == "__main__":
    main()
