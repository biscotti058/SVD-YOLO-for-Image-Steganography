# SVD & YOLO for Image Steganography

**Statistical Methods Project**

## Overview

This project implements an image steganography system that combines **Singular Value Decomposition (SVD)** with **YOLO object detection** to hide secret images within cover images.

### How It Works

1. **SVD Decomposition**: Both cover and secret images are decomposed using SVD: `A = U · Σ · Vᵀ`
2. **YOLO Detection**: YOLO identifies objects/complex regions in the cover image
3. **Smart Embedding**: Secret image data is embedded by modifying the singular values of the cover image, guided by a mask that prioritizes complex/textured regions
4. **Extraction**: The secret is recovered using stored SVD keys

### Mathematical Foundation

Given a cover image matrix `C` and secret image `S`:

```
C = Uc · Σc · Vc^T
S = Us · Σs · Vs^T

Σ_stego = Σc + α · Σs
I_stego = Uc · Σ_stego · Vc^T
```

The parameter `α` (alpha) controls the trade-off between:
- **Imperceptibility** (low α → less visible changes)
- **Robustness** (high α → better recovery after attacks)

## Project Structure

```
Statistical Meto/
├── README.md
├── requirements.txt
├── demo.ipynb              # Main demo notebook (run this!)
├── src/
│   ├── __init__.py
│   ├── svd_steganography.py    # Core SVD encode/decode
│   ├── yolo_region_selector.py # YOLO-based region selection
│   └── pipeline.py             # Combined pipeline
├── images/                 # Input images (auto-generated if empty)
└── output/                 # Generated results
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

## Features

- **3 operating modes**:
  - `svd_only`: Basic SVD steganography (baseline)
  - `svd_texture`: SVD with texture-based region selection
  - `svd_yolo`: Full SVD + YOLO pipeline

- **Quality metrics**: PSNR, SSIM
- **Robustness analysis**: Tests against JPEG compression, noise, blur, scaling
- **Comparative analysis**: Tests multiple alpha values
- **SVD visualization**: Singular value spectrum, low-rank approximations

## Key Results

| Alpha | PSNR (dB) | SSIM   | Imperceptibility |
|-------|-----------|--------|------------------|
| 0.001 | >50       | >0.999 | Excellent        |
| 0.01  | ~35-40    | >0.99  | Good             |
| 0.05  | ~25-30    | >0.95  | Moderate         |
| 0.1   | ~20-25    | >0.90  | Poor             |

## Limitations

- **Decoding keys**: Extracting the secret requires the `keys.npz` file (U, V matrices from SVD). The recipient must receive the keys; this is not "blind" steganography (password/seed only).
- **Robustness**: Aggressive JPEG compression, heavy resizing, and strong noise can degrade or destroy the message.
- **Hiding only**: Data is not encrypted; anyone who extracts the payload from the SVD sees the secret in the clear.

## Future developments

- **Blind steganography**: recover the message with only a password/seed, without transmitting SVD matrices.
- **Encryption** (e.g. AES) of the payload before embedding, for confidentiality even if the payload is extracted.
- Explicit **YOLO model** choice (n/s/m/l/x) in the pipeline to balance speed (CPU) and accuracy (GPU).
- Local adaptation of **α** based on region complexity.

## Technologies

- **Python 3.10+**
- **NumPy/SciPy** — SVD computation
- **OpenCV** — Image processing
- **Ultralytics YOLOv8** — Object detection
- **scikit-image** — SSIM metric
- **Matplotlib** — Visualization
