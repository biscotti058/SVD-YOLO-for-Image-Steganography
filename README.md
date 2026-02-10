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

## Limitazioni

- **Chiavi di decodifica**: Per estrarre il segreto serve il file `keys.npz` (matrici U, V dell'SVD). Il destinatario deve ricevere anche le chiavi; non è steganografia "blind" (solo password/seme).
- **Robustezza**: Compressione JPEG aggressiva, ridimensionamento forte e rumore elevato possono degradare o perdere il messaggio.
- **Solo nascondimento**: I dati non sono cifrati; chi estrae il payload dall'SVD vede il segreto in chiaro.

## Sviluppi futuri

- Steganografia **blind**: recupero del messaggio con sola password/seme, senza trasmettere le matrici SVD.
- **Crittografia** (es. AES) del payload prima dell'embedding, per confidenzialità anche in caso di estrazione.
- Scelta esplicita del **modello YOLO** (n/s/m/l/x) dalla pipeline per bilanciare velocità (CPU) e precisione (GPU).
- Adattamento locale di **α** in base alla complessità della regione.

## Technologies

- **Python 3.10+**
- **NumPy/SciPy** — SVD computation
- **OpenCV** — Image processing
- **Ultralytics YOLOv8** — Object detection
- **scikit-image** — SSIM metric
- **Matplotlib** — Visualization
