# SVD & YOLO for Image Steganography

**Statistical and Mathematical Methods for Artificial Intelligence — Project**

Authors: Nicola Biscotti, Giovanni Michele Iuliani — Politecnico di Bari, A.A. 2024/2025

## Overview

This project implements an image steganography system that combines **Singular Value Decomposition (SVD)** with **YOLO object detection** to hide a QR code secret inside real-world photographs from the **MS COCO dataset**.

### How It Works

1. **COCO Cover Images**: Real-world photographs from MS COCO provide complex, textured scenes ideal for steganography.
2. **QR Code Secret**: A QR code encoding a text message serves as the secret payload — providing a binary, verifiable recovery metric.
3. **SVD Decomposition**: Both cover and QR images are decomposed using SVD: `A = U · Σ · Vᵀ`.
4. **YOLO Detection**: YOLO identifies objects/complex regions in the cover image.
5. **Smart Embedding**: Secret data is embedded by modifying the singular values, guided by a mask that prioritizes complex/textured regions.
6. **Extraction & Verification**: The secret is recovered using stored SVD keys, and QR decodability is verified.

## Mathematical Foundation

Given a cover image matrix `C` and secret image `S` (per-channel real matrices):

```
C = Uc · Σc · Vcᵀ
S = Us · Σs · Vsᵀ

Σ_stego = Σc + α · Σs
I_stego = Uc · Σ_stego · Vcᵀ
```

The parameter `α` (alpha) controls the trade-off between:
- **Imperceptibility** (low α → less visible changes)
- **Robustness** (high α → better QR recovery after attacks)

The choice of α and the stability of the recovery are formally justified through:

- **Eckart–Young–Mirsky theorem**: SVD provides the optimal low-rank approximation in the Frobenius and spectral norms.
- **Weyl's inequality**: `|σᵢ(C + αS) − σᵢ(C)| ≤ α · σ₁(S)`, which bounds the perturbation of cover singular values by `α · σ₁(S)`.
- **Mirsky's theorem**: the total perturbation in Frobenius norm is bounded by `α · ‖Σs‖_F = α · ‖S‖_F`.
- **PSNR bound**: a closed-form lower bound on PSNR in terms of `α` and `‖S‖_F` (see `src/svd_theory.py` and the report).

A full derivation, with proofs and parameter selection criteria, is available in `report_finale.pdf` and in `src/svd_theory.py`.

### Why COCO + QR Code?

- **COCO**: real-world photographs with rich, diverse scenes and multiple objects — ideal cover images because they contain complex textures that mask embedding artifacts, and YOLO can detect many objects per image.
- **QR code**: provides a **binary, verifiable payload** — we can objectively measure whether the hidden message was recovered successfully by checking if the QR is still decodable after extraction. The QR's built-in error correction (level H = 30% redundancy) adds extra resilience.

## Project Structure

```
SVD-YOLO-for-Image-Steganography/
├── README.md
├── requirements.txt
├── demo.ipynb                    # Main demo notebook
├── report_finale.pdf             # Final report (with mathematical analysis)
├── src/
│   ├── __init__.py
│   ├── svd_steganography.py      # Core SVD encode/decode
│   ├── svd_theory.py             # Theoretical bounds, perturbation analysis
│   ├── yolo_region_selector.py   # YOLO-based region selection
│   ├── coco_qr_utils.py          # COCO download & QR code generation
│   ├── pipeline.py               # Combined pipeline
│   └── large_scale_eval.py       # Large-scale evaluation on many COCO images
├── images/                       # Downloaded COCO images + generated QR
└── output/                       # Generated results, figures, CSV tables
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
jupyter notebook demo.ipynb
```

Or in VS Code / Cursor, just open `demo.ipynb` and run all cells.

COCO images are downloaded automatically on first run (requires internet). If network is unavailable, a synthetic fallback cover image is generated.

### 3. Run the large-scale evaluation (optional, may take 30–60 minutes)

```bash
python -m src.large_scale_eval --n-images 100 --alphas 0.001,0.005,0.01,0.02,0.05,0.1
```

Results are written to `output/large_scale/`:
- `metrics.csv` — per-image metrics for every (mode, α) configuration
- `aggregate.csv` — mean / std across images for every (mode, α)
- `large_scale_summary.png` — summary plots

## Features

- **COCO dataset** as cover images (auto-download from MS COCO val2017)
- **QR code generation** as secret payload with configurable text and error correction level
- **QR decode verification** after recovery — binary success metric
- **3 operating modes**:
  - `svd_only`: Basic SVD steganography (baseline)
  - `svd_texture`: SVD with texture-based region selection
  - `svd_yolo`: Full SVD + YOLO pipeline
- **Quality metrics**: PSNR, SSIM, NCC, BER
- **Theoretical analysis**: Eckart–Young–Mirsky error, Weyl perturbation bounds, theoretical PSNR vs measured PSNR
- **QR recovery heatmap** across multiple alpha values and COCO images
- **Robustness analysis**: Tests against JPEG compression, noise, blur, scaling
- **Comparative analysis**: Tests multiple alpha values with QR decode annotations
- **SVD visualization**: Singular value spectrum, low-rank approximations, cover vs QR spectrum comparison
- **Large-scale statistical evaluation**: aggregate PSNR/SSIM/NCC/BER/QR-rate with error bars across 100+ COCO images

## Key Results (orders of magnitude — full per-image statistics in `output/large_scale/`)

| Alpha | PSNR (dB) | SSIM   | Imperceptibility | QR Recovery |
|-------|-----------|--------|------------------|-------------|
| 0.001 | >50       | >0.999 | Excellent        | Unreliable  |
| 0.005 | ~40-45    | >0.995 | Very Good        | Usually OK  |
| 0.01  | ~35-40    | >0.99  | Good             | Reliable    |
| 0.05  | ~25-30    | >0.95  | Moderate         | Always OK   |
| 0.1   | ~20-25    | >0.90  | Poor             | Always OK   |

## Limitations

- **Decoding keys**: Extracting the secret requires the `keys.npz` file (U, V matrices from SVD). The recipient must receive the keys; this is not "blind" steganography (password/seed only).
- **Robustness**: Aggressive JPEG compression, heavy resizing, and strong noise can degrade or destroy the QR payload.
- **Hiding only**: Data is not encrypted; anyone who extracts the payload from the SVD sees the QR code in the clear.
- **QR binary nature**: The QR either decodes or it doesn't — there is no partial recovery.

## Future Developments

- **Blind steganography**: Recover the message with only a password/seed, without transmitting SVD matrices.
- **Encryption** (e.g. AES) of the QR payload before embedding, for confidentiality even if the payload is extracted.
- Explicit **YOLO model** choice (n/s/m/l/x) in the pipeline to balance speed (CPU) and accuracy (GPU).
- **Adaptive α**: Local adaptation of α based on region complexity and QR module density.
- Testing on the full COCO validation set (5000 images) for statistically robust results.
- Exploring QR **error correction levels** (L/M/Q/H) as an additional tuneable parameter.

## Technologies

- **Python 3.10+**
- **NumPy/SciPy** — SVD computation, linear algebra, perturbation bounds
- **OpenCV** — Image processing & QR decoding
- **Ultralytics YOLOv8** — Object detection
- **qrcode** — QR code generation
- **MS COCO** — Cover image dataset
- **scikit-image** — SSIM metric
- **Matplotlib** — Visualization
