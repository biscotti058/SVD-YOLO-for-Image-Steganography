"""
SVD-Based Image Steganography Module
=====================================
Uses Singular Value Decomposition (SVD) to embed secret data into cover images.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import os


class SVDSteganography:
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self._u_cover = None
        self._vt_cover = None
        self._s_cover = None
        self._secret_shape = None
        self._u_secret = None
        self._vt_secret = None

    def _resize_secret(self, cover, secret):
        from PIL import Image as PILImage
        secret_pil = PILImage.fromarray(secret.astype(np.uint8))
        secret_pil = secret_pil.resize((cover.shape[1], cover.shape[0]), PILImage.LANCZOS)
        return np.array(secret_pil)

    def _ensure_3channel(self, img):
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def encode(self, cover_img, secret_img, mask=None):
        cover = self._ensure_3channel(cover_img).astype(np.float64)
        secret = self._ensure_3channel(secret_img).astype(np.float64)
        secret = self._resize_secret(cover, secret)
        self._secret_shape = secret.shape
        h, w, c = cover.shape
        stego = np.zeros_like(cover)
        self._u_cover = []
        self._vt_cover = []
        self._s_cover = []
        self._u_secret = []
        self._vt_secret = []
        for ch in range(c):
            U_c, S_c, Vt_c = np.linalg.svd(cover[:, :, ch], full_matrices=False)
            U_s, S_s, Vt_s = np.linalg.svd(secret[:, :, ch], full_matrices=False)
            self._u_cover.append(U_c)
            self._vt_cover.append(Vt_c)
            self._s_cover.append(S_c.copy())
            self._u_secret.append(U_s)
            self._vt_secret.append(Vt_s)
            if mask is not None:
                mask_ratio = np.mean(mask)
                effective_alpha = self.alpha * (0.1 + 0.9 * mask_ratio)
            else:
                effective_alpha = self.alpha
            S_stego = S_c + effective_alpha * S_s
            stego[:, :, ch] = U_c @ np.diag(S_stego) @ Vt_c
        stego = np.clip(stego, 0, 255).astype(np.uint8)
        return stego

    def decode(self, stego_img):
        if self._u_cover is None:
            raise ValueError("Must call encode() before decode(), or load keys.")
        stego = self._ensure_3channel(stego_img).astype(np.float64)
        h, w, c = stego.shape
        secret_recovered = np.zeros_like(stego)
        for ch in range(c):
            U_st, S_st, Vt_st = np.linalg.svd(stego[:, :, ch], full_matrices=False)
            S_secret_recovered = (S_st - self._s_cover[ch]) / self.alpha
            secret_recovered[:, :, ch] = (
                self._u_secret[ch] @ np.diag(S_secret_recovered) @ self._vt_secret[ch]
            )
        secret_recovered = np.clip(secret_recovered, 0, 255).astype(np.uint8)
        return secret_recovered

    def save_keys(self, filepath):
        if self._u_cover is None:
            raise ValueError("No keys to save. Run encode() first.")
        np.savez_compressed(
            filepath,
            alpha=self.alpha,
            secret_shape=self._secret_shape,
            u_cover_0=self._u_cover[0], u_cover_1=self._u_cover[1], u_cover_2=self._u_cover[2],
            vt_cover_0=self._vt_cover[0], vt_cover_1=self._vt_cover[1], vt_cover_2=self._vt_cover[2],
            s_cover_0=self._s_cover[0], s_cover_1=self._s_cover[1], s_cover_2=self._s_cover[2],
            u_secret_0=self._u_secret[0], u_secret_1=self._u_secret[1], u_secret_2=self._u_secret[2],
            vt_secret_0=self._vt_secret[0], vt_secret_1=self._vt_secret[1], vt_secret_2=self._vt_secret[2],
        )

    def load_keys(self, filepath):
        data = np.load(filepath)
        self.alpha = float(data['alpha'])
        self._secret_shape = tuple(data['secret_shape'])
        self._u_cover = [data['u_cover_0'], data['u_cover_1'], data['u_cover_2']]
        self._vt_cover = [data['vt_cover_0'], data['vt_cover_1'], data['vt_cover_2']]
        self._s_cover = [data['s_cover_0'], data['s_cover_1'], data['s_cover_2']]
        self._u_secret = [data['u_secret_0'], data['u_secret_1'], data['u_secret_2']]
        self._vt_secret = [data['vt_secret_0'], data['vt_secret_1'], data['vt_secret_2']]


def compute_psnr(original, modified):
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(original, modified):
    from skimage.metrics import structural_similarity as ssim
    if len(original.shape) == 3:
        return ssim(original, modified, channel_axis=2, data_range=255)
    return ssim(original, modified, data_range=255)


def compute_ncc(original, recovered):
    orig_flat = original.astype(np.float64).flatten()
    rec_flat = recovered.astype(np.float64).flatten()
    mean_orig = np.mean(orig_flat)
    mean_rec = np.mean(rec_flat)
    numerator = np.sum((orig_flat - mean_orig) * (rec_flat - mean_rec))
    denominator = np.sqrt(np.sum((orig_flat - mean_orig)**2) * np.sum((rec_flat - mean_rec)**2))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_ber(original, recovered, threshold=128):
    import cv2
    if len(original.shape) == 3:
        original = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        recovered = cv2.cvtColor(recovered.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    orig_bin = (original > threshold).astype(int)
    rec_bin = (recovered > threshold).astype(int)
    errors = np.sum(orig_bin != rec_bin)
    total_bits = orig_bin.size
    return float(errors / total_bits)


def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)


def save_image(img, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(path)
