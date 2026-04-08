"""
SVD-Based Image Steganography Module
=====================================
Uses Singular Value Decomposition (SVD) to embed secret data into cover images.

Theory:
    Given a cover image matrix A, SVD decomposes it as:
        A = U * S * V^T
    where S is a diagonal matrix of singular values.
    
    The secret is embedded by modifying the singular values of the cover image
    using the singular values of the secret image. The modification is controlled
    by a scaling factor (alpha) that balances imperceptibility vs robustness.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import os


class SVDSteganography:
    """
    SVD-based steganography encoder/decoder.
    
    Embeds a secret image into a cover image by modifying singular values
    of each color channel independently.
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Parameters
        ----------
        alpha : float
            Embedding strength. Higher = more robust but more visible.
            Typical range: 0.001 - 0.1
        """
        self.alpha = alpha
<<<<<<< HEAD
=======
        # These keys are needed for extraction
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
        self._u_cover = None
        self._vt_cover = None
        self._s_cover = None
        self._secret_shape = None
        self._u_secret = None
        self._vt_secret = None
    
    def _resize_secret(self, cover: np.ndarray, secret: np.ndarray) -> np.ndarray:
        """Resize secret image to match cover image dimensions."""
        from PIL import Image as PILImage
        secret_pil = PILImage.fromarray(secret.astype(np.uint8))
        secret_pil = secret_pil.resize((cover.shape[1], cover.shape[0]), PILImage.LANCZOS)
        return np.array(secret_pil)
    
    def _ensure_3channel(self, img: np.ndarray) -> np.ndarray:
        """Ensure image has 3 channels."""
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    
    def encode(self, cover_img: np.ndarray, secret_img: np.ndarray,
               mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Embed secret image into cover image using SVD.
<<<<<<< HEAD
=======
        
        Parameters
        ----------
        cover_img : np.ndarray
            Cover image (H, W, 3) in uint8
        secret_img : np.ndarray
            Secret image to hide, will be resized to match cover
        mask : np.ndarray, optional
            Binary mask (H, W) where 1 = embed here, 0 = don't embed.
            Used for YOLO-guided region selection.
            
        Returns
        -------
        stego_img : np.ndarray
            Stego image with hidden data (H, W, 3) in uint8
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
        """
        cover = self._ensure_3channel(cover_img).astype(np.float64)
        secret = self._ensure_3channel(secret_img).astype(np.float64)
        secret = self._resize_secret(cover, secret)
        
        self._secret_shape = secret.shape
        
        h, w, c = cover.shape
        stego = np.zeros_like(cover)
        
<<<<<<< HEAD
=======
        # Store SVD components for each channel (needed for decoding)
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
        self._u_cover = []
        self._vt_cover = []
        self._s_cover = []
        self._u_secret = []
        self._vt_secret = []
        
        for ch in range(c):
<<<<<<< HEAD
            U_c, S_c, Vt_c = np.linalg.svd(cover[:, :, ch], full_matrices=False)
            U_s, S_s, Vt_s = np.linalg.svd(secret[:, :, ch], full_matrices=False)
            
=======
            # SVD of cover channel
            U_c, S_c, Vt_c = np.linalg.svd(cover[:, :, ch], full_matrices=False)
            
            # SVD of secret channel
            U_s, S_s, Vt_s = np.linalg.svd(secret[:, :, ch], full_matrices=False)
            
            # Store for decoding
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
            self._u_cover.append(U_c)
            self._vt_cover.append(Vt_c)
            self._s_cover.append(S_c.copy())
            self._u_secret.append(U_s)
            self._vt_secret.append(Vt_s)
            
            if mask is not None:
<<<<<<< HEAD
=======
                # YOLO-guided: scale alpha based on mask
                # In masked regions, use full alpha; outside, use reduced alpha
                # We apply embedding to singular values globally but scale
                # the contribution based on the mask coverage
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
                mask_ratio = np.mean(mask)
                effective_alpha = self.alpha * (0.1 + 0.9 * mask_ratio)
            else:
                effective_alpha = self.alpha
            
<<<<<<< HEAD
            S_stego = S_c + effective_alpha * S_s
            stego[:, :, ch] = U_c @ np.diag(S_stego) @ Vt_c
        
=======
            # Embed: modify singular values
            S_stego = S_c + effective_alpha * S_s
            
            # Reconstruct channel with modified singular values
            stego[:, :, ch] = U_c @ np.diag(S_stego) @ Vt_c
        
        # Clip to valid range
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
        stego = np.clip(stego, 0, 255).astype(np.uint8)
        return stego
    
    def decode(self, stego_img: np.ndarray) -> np.ndarray:
        """
        Extract secret image from stego image.
<<<<<<< HEAD
=======
        
        Must be called after encode() as it uses stored SVD components.
        
        Parameters
        ----------
        stego_img : np.ndarray
            Stego image (H, W, 3) in uint8
            
        Returns
        -------
        secret_recovered : np.ndarray
            Recovered secret image (H, W, 3) in uint8
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
        """
        if self._u_cover is None:
            raise ValueError("Must call encode() before decode(), or load keys.")
        
        stego = self._ensure_3channel(stego_img).astype(np.float64)
        h, w, c = stego.shape
        secret_recovered = np.zeros_like(stego)
        
        for ch in range(c):
<<<<<<< HEAD
            U_st, S_st, Vt_st = np.linalg.svd(stego[:, :, ch], full_matrices=False)
            S_secret_recovered = (S_st - self._s_cover[ch]) / self.alpha
=======
            # SVD of stego channel
            U_st, S_st, Vt_st = np.linalg.svd(stego[:, :, ch], full_matrices=False)
            
            # Extract secret singular values
            S_secret_recovered = (S_st - self._s_cover[ch]) / self.alpha
            
            # Reconstruct secret channel using original secret's U and Vt
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
            secret_recovered[:, :, ch] = (
                self._u_secret[ch] @ np.diag(S_secret_recovered) @ self._vt_secret[ch]
            )
        
        secret_recovered = np.clip(secret_recovered, 0, 255).astype(np.uint8)
        return secret_recovered
    
    def save_keys(self, filepath: str):
<<<<<<< HEAD
        """Save the decoding keys (SVD components) to a file."""
=======
        """
        Save the decoding keys (SVD components) to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the keys (.npz file)
        """
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
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
    
    def load_keys(self, filepath: str):
<<<<<<< HEAD
        """Load decoding keys from a file."""
=======
        """
        Load decoding keys from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the keys file (.npz)
        """
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
        data = np.load(filepath)
        self.alpha = float(data['alpha'])
        self._secret_shape = tuple(data['secret_shape'])
        self._u_cover = [data['u_cover_0'], data['u_cover_1'], data['u_cover_2']]
        self._vt_cover = [data['vt_cover_0'], data['vt_cover_1'], data['vt_cover_2']]
        self._s_cover = [data['s_cover_0'], data['s_cover_1'], data['s_cover_2']]
        self._u_secret = [data['u_secret_0'], data['u_secret_1'], data['u_secret_2']]
        self._vt_secret = [data['vt_secret_0'], data['vt_secret_1'], data['vt_secret_2']]


<<<<<<< HEAD
# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio. Higher = less distortion."""
=======
def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Higher PSNR = less distortion. Typically > 30dB is considered good.
    """
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(original: np.ndarray, modified: np.ndarray) -> float:
<<<<<<< HEAD
    """Compute Structural Similarity Index. Closer to 1.0 = better."""
    from skimage.metrics import structural_similarity as ssim
=======
    """
    Compute Structural Similarity Index between two images.
    
    SSIM closer to 1.0 = better quality preservation.
    """
    from skimage.metrics import structural_similarity as ssim
    # Convert to grayscale for SSIM if needed
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
    if len(original.shape) == 3:
        return ssim(original, modified, channel_axis=2, data_range=255)
    return ssim(original, modified, data_range=255)


<<<<<<< HEAD
def compute_ncc(original: np.ndarray, recovered: np.ndarray) -> float:
    """Normalized Cross-Correlation tra originale e recuperato."""
    orig_flat = original.astype(np.float64).flatten()
    rec_flat = recovered.astype(np.float64).flatten()
    
    mean_orig = np.mean(orig_flat)
    mean_rec = np.mean(rec_flat)
    
    numerator = np.sum((orig_flat - mean_orig) * (rec_flat - mean_rec))
    denominator = np.sqrt(np.sum((orig_flat - mean_orig)**2) * np.sum((rec_flat - mean_rec)**2))
    
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_ber(original: np.ndarray, recovered: np.ndarray, threshold: int = 128) -> float:
    """Bit Error Rate per il QR code."""
    import cv2
    if len(original.shape) == 3:
        original = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        recovered = cv2.cvtColor(recovered.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
    orig_bin = (original > threshold).astype(int)
    rec_bin = (recovered > threshold).astype(int)
    
    errors = np.sum(orig_bin != rec_bin)
    total_bits = orig_bin.size
    return float(errors / total_bits)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

=======
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
def load_image(path: str) -> np.ndarray:
    """Load an image as a numpy array (RGB)."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def save_image(img: np.ndarray, path: str):
    """Save a numpy array as an image."""
<<<<<<< HEAD
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
=======
>>>>>>> 1a18a606ff5809894d00e4d88aa295400c16bd36
    Image.fromarray(img.astype(np.uint8)).save(path)
