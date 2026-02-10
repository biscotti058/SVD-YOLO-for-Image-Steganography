"""
SVD + YOLO Steganography Pipeline
===================================
Combines SVD steganography with YOLO-guided region selection
into a complete encoding/decoding pipeline.

Pipeline:
    1. Load cover image and secret image
    2. Run YOLO on cover image to find complex regions
    3. Generate embedding mask from detections + texture analysis
    4. Apply SVD steganography with mask-guided embedding
    5. Evaluate quality (PSNR, SSIM)
    6. Decode and recover secret image
"""

import numpy as np
import os
from typing import Optional, Tuple

from src.svd_steganography import (
    SVDSteganography,
    compute_psnr,
    compute_ssim,
    load_image,
    save_image,
)
from src.yolo_region_selector import YOLORegionSelector, generate_texture_only_mask


class StegoPipeline:
    """
    Complete SVD + YOLO steganography pipeline.
    
    Supports three modes:
        1. 'svd_only': Basic SVD steganography (baseline)
        2. 'svd_texture': SVD with texture-based mask
        3. 'svd_yolo': Full SVD + YOLO pipeline
    """
    
    def __init__(self, alpha: float = 0.01, mode: str = 'svd_yolo',
                 yolo_model: str = 'yolov8n.pt', yolo_confidence: float = 0.25):
        """
        Parameters
        ----------
        alpha : float
            SVD embedding strength (0.001 - 0.1)
        mode : str
            One of 'svd_only', 'svd_texture', 'svd_yolo'
        yolo_model : str
            YOLO model: 'yolov8n.pt' (nano, default, fast), 'yolov8s.pt', 'yolov8m.pt',
            'yolov8l.pt', 'yolov8x.pt' (larger = more accurate, GPU recommended).
        yolo_confidence : float
            YOLO detection confidence threshold
        """
        self.alpha = alpha
        self.mode = mode
        self.stego = SVDSteganography(alpha=alpha)
        
        if mode == 'svd_yolo':
            self.yolo = YOLORegionSelector(
                model_name=yolo_model,
                confidence=yolo_confidence
            )
        else:
            self.yolo = None
        
        # Results storage
        self.mask = None
        self.detections = None
        self.stego_image = None
        self.recovered_secret = None
        self.metrics = {}
    
    def encode(self, cover_path: str, secret_path: str,
               output_dir: str = 'output') -> dict:
        """
        Run the full encoding pipeline.
        
        Parameters
        ----------
        cover_path : str
            Path to cover image
        secret_path : str
            Path to secret image
        output_dir : str
            Directory to save outputs
            
        Returns
        -------
        results : dict
            Dictionary with metrics and file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        cover = load_image(cover_path)
        secret = load_image(secret_path)
        
        print(f"Cover image: {cover.shape}")
        print(f"Secret image: {secret.shape}")
        print(f"Mode: {self.mode}")
        print(f"Alpha: {self.alpha}")
        print()
        
        # Generate mask based on mode
        if self.mode == 'svd_yolo':
            print("Running YOLO object detection...")
            self.detections = self.yolo.detect_objects(cover)
            print(f"  Found {len(self.detections)} objects:")
            for det in self.detections:
                print(f"    - {det['class']} (conf: {det['confidence']:.2f})")
            
            print("Generating YOLO-guided embedding mask...")
            self.mask = self.yolo.generate_embedding_mask(cover)
            
            # Save mask visualization
            mask_vis = self.yolo.visualize_mask(cover, self.mask, self.detections)
            save_image(mask_vis, os.path.join(output_dir, 'mask_visualization.png'))
            
        elif self.mode == 'svd_texture':
            print("Generating texture-based embedding mask...")
            self.mask = generate_texture_only_mask(cover)
        else:
            print("No mask (basic SVD mode)")
            self.mask = None
        
        # Encode
        print("\nEncoding secret image using SVD...")
        self.stego_image = self.stego.encode(cover, secret, mask=self.mask)
        
        # Save stego image
        stego_path = os.path.join(output_dir, 'stego_image.png')
        save_image(self.stego_image, stego_path)
        
        # Save keys
        keys_path = os.path.join(output_dir, 'keys.npz')
        self.stego.save_keys(keys_path)
        
        # Compute quality metrics
        psnr = compute_psnr(cover, self.stego_image)
        ssim = compute_ssim(cover, self.stego_image)
        
        self.metrics = {
            'psnr': psnr,
            'ssim': ssim,
            'alpha': self.alpha,
            'mode': self.mode,
            'cover_shape': cover.shape,
            'secret_shape': secret.shape,
            'num_detections': len(self.detections) if self.detections else 0,
        }
        
        print(f"\n{'='*50}")
        print(f"ENCODING RESULTS")
        print(f"{'='*50}")
        print(f"  PSNR: {psnr:.2f} dB (higher is better, >30dB is good)")
        print(f"  SSIM: {ssim:.4f} (closer to 1.0 is better)")
        print(f"  Stego image saved: {stego_path}")
        print(f"  Keys saved: {keys_path}")
        
        return {
            'stego_path': stego_path,
            'keys_path': keys_path,
            'metrics': self.metrics,
        }
    
    def decode(self, stego_path: str, keys_path: str,
               output_dir: str = 'output') -> np.ndarray:
        """
        Run the decoding pipeline.
        
        Parameters
        ----------
        stego_path : str
            Path to stego image
        keys_path : str
            Path to keys file
        output_dir : str
            Directory to save outputs
            
        Returns
        -------
        recovered : np.ndarray
            Recovered secret image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load stego image
        stego = load_image(stego_path)
        
        # Load keys
        decoder = SVDSteganography()
        decoder.load_keys(keys_path)
        
        # Decode
        print("Decoding secret image...")
        self.recovered_secret = decoder.decode(stego)
        
        # Save recovered secret
        recovered_path = os.path.join(output_dir, 'recovered_secret.png')
        save_image(self.recovered_secret, recovered_path)
        
        print(f"Recovered secret saved: {recovered_path}")
        
        return self.recovered_secret
    
    def compare_modes(self, cover_path: str, secret_path: str,
                      alphas: list = None, output_dir: str = 'output') -> list:
        """
        Compare different steganography modes and alpha values.
        
        Parameters
        ----------
        cover_path : str
            Path to cover image
        secret_path : str
            Path to secret image
        alphas : list of float
            Alpha values to test
        output_dir : str
            Output directory
            
        Returns
        -------
        all_results : list of dict
            Results for each configuration
        """
        if alphas is None:
            alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        modes = ['svd_only', 'svd_texture', 'svd_yolo']
        all_results = []
        
        cover = load_image(cover_path)
        secret = load_image(secret_path)
        
        for mode in modes:
            for alpha in alphas:
                print(f"\nTesting: mode={mode}, alpha={alpha}")
                
                pipeline = StegoPipeline(alpha=alpha, mode=mode)
                
                try:
                    sub_dir = os.path.join(output_dir, f'{mode}_alpha{alpha}')
                    result = pipeline.encode(cover_path, secret_path, sub_dir)
                    
                    # Also decode and measure recovery quality
                    recovered = pipeline.decode(
                        result['stego_path'], result['keys_path'], sub_dir
                    )
                    
                    # Secret recovery quality
                    secret_resized = pipeline.stego._resize_secret(cover, secret)
                    recovery_psnr = compute_psnr(secret_resized, recovered)
                    recovery_ssim = compute_ssim(secret_resized, recovered)
                    
                    result['metrics']['recovery_psnr'] = recovery_psnr
                    result['metrics']['recovery_ssim'] = recovery_ssim
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    all_results.append({
                        'metrics': {
                            'mode': mode, 'alpha': alpha, 'error': str(e)
                        }
                    })
        
        return all_results
