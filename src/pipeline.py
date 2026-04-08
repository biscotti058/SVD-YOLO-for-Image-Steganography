"""
SVD + YOLO Steganography Pipeline
===================================
Combines SVD steganography with YOLO-guided region selection
into a complete encoding/decoding pipeline.
"""

import numpy as np
import os
import glob
import sys
from typing import Optional, Tuple

from src.svd_steganography import (
    SVDSteganography, compute_psnr, compute_ssim,
    compute_ncc, compute_ber, load_image, save_image
)
from src.yolo_region_selector import YOLORegionSelector, generate_texture_only_mask
from src.coco_qr_utils import verify_qr_decode

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


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
        
        self.mask = None
        self.detections = None
        self.stego_image = None
        self.recovered_secret = None
        self.metrics = {}
    
    def encode(self, cover_path: str, secret_path: str,
               output_dir: str = 'output') -> dict:
        os.makedirs(output_dir, exist_ok=True)
        
        cover = load_image(cover_path)
        secret = load_image(secret_path)
        
        print(f"Cover image: {cover.shape}")
        print(f"Secret image: {secret.shape}")
        print(f"Mode: {self.mode}")
        print(f"Alpha: {self.alpha}")
        print()
        
        if self.mode == 'svd_yolo':
            print("Running YOLO object detection...")
            self.detections = self.yolo.detect_objects(cover)
            print(f"  Found {len(self.detections)} objects:")
            for det in self.detections:
                print(f"    - {det['class']} (conf: {det['confidence']:.2f})")
            
            print("Generating YOLO-guided embedding mask...")
            self.mask = self.yolo.generate_embedding_mask(cover)
            
            mask_vis = self.yolo.visualize_mask(cover, self.mask, self.detections)
            save_image(mask_vis, os.path.join(output_dir, 'mask_visualization.png'))
            
        elif self.mode == 'svd_texture':
            print("Generating texture-based embedding mask...")
            self.mask = generate_texture_only_mask(cover)
        else:
            print("No mask (basic SVD mode)")
            self.mask = None
        
        print("\nEncoding secret image using SVD...")
        self.stego_image = self.stego.encode(cover, secret, mask=self.mask)
        
        stego_path = os.path.join(output_dir, 'stego_image.png')
        save_image(self.stego_image, stego_path)
        
        keys_path = os.path.join(output_dir, 'keys.npz')
        self.stego.save_keys(keys_path)
        
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
        os.makedirs(output_dir, exist_ok=True)
        
        stego = load_image(stego_path)
        
        decoder = SVDSteganography()
        decoder.load_keys(keys_path)
        
        print("Decoding secret image...")
        self.recovered_secret = decoder.decode(stego)
        
        recovered_path = os.path.join(output_dir, 'recovered_secret.png')
        save_image(self.recovered_secret, recovered_path)
        print(f"Recovered secret saved: {recovered_path}")
        
        # Try QR decode verification
        qr_text = verify_qr_decode(self.recovered_secret)
        if qr_text:
            print(f"  QR Code decoded successfully: \"{qr_text}\"")
        else:
            print(f"  QR Code decoding failed (too much distortion)")
        
        return self.recovered_secret
    
    def compare_modes(self, cover_path: str, secret_path: str,
                      alphas: list = None, output_dir: str = 'output') -> list:
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
                    
                    recovered = pipeline.decode(
                        result['stego_path'], result['keys_path'], sub_dir
                    )
                    
                    secret_resized = pipeline.stego._resize_secret(cover, secret)
                    recovery_psnr = compute_psnr(secret_resized, recovered)
                    recovery_ssim = compute_ssim(secret_resized, recovered)
                    
                    # QR decode check
                    qr_text = verify_qr_decode(recovered)
                    
                    result['metrics']['recovery_psnr'] = recovery_psnr
                    result['metrics']['recovery_ssim'] = recovery_ssim
                    result['metrics']['qr_decoded'] = qr_text is not None
                    result['metrics']['qr_text'] = qr_text or ""
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    all_results.append({
                        'metrics': {
                            'mode': mode, 'alpha': alpha, 'error': str(e)
                        }
                    })
        
        return all_results

    def evaluate_dataset(self, cover_dir: str, secret_path: str,
                         output_dir: str = 'output/dataset_results') -> list:
        """Evaluate steganography across an entire directory of cover images."""
        os.makedirs(output_dir, exist_ok=True)
        cover_paths = glob.glob(os.path.join(cover_dir, '*.jpg'))
        
        results = []
        secret = load_image(secret_path)
        
        for path in tqdm(cover_paths, desc=f"Valutazione {self.mode}"):
            filename = os.path.basename(path)
            try:
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                
                res = self.encode(path, secret_path,
                                  os.path.join(output_dir, f"{self.mode}_{filename}_temp"))
                recovered = self.decode(res['stego_path'], res['keys_path'],
                                        os.path.join(output_dir, f"{self.mode}_{filename}_temp"))
                sys.stdout = original_stdout
                
                cover = load_image(path)
                secret_resized = self.stego._resize_secret(cover, secret)
                
                metrics = res['metrics']
                metrics['filename'] = filename
                metrics['ncc'] = compute_ncc(secret_resized, recovered)
                metrics['ber'] = compute_ber(secret_resized, recovered)
                
                # QR verification
                qr_text = verify_qr_decode(recovered)
                metrics['qr_decoded'] = qr_text is not None
                
                results.append(metrics)
                
            except Exception as e:
                sys.stdout = original_stdout
                print(f"Errore su {filename}: {e}")
                
        return results
