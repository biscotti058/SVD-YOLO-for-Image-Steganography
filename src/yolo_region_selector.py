"""
YOLO-Based Region Selector for Steganography
==============================================
Uses YOLO object detection to identify regions in cover images where
hidden data will be less perceptible to the human eye.

Strategy:
    - YOLO detects objects in the cover image
    - Regions WITH objects (textured, complex areas) are preferred for embedding
      because modifications are harder to notice in complex regions
    - Smooth/uniform regions (background, sky) are avoided because even small
      changes are visible there
    - A binary mask is generated: 1 = good for embedding, 0 = avoid
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


class YOLORegionSelector:
    """
    Uses YOLO to create embedding masks for steganography.
    
    Detected object regions are marked as suitable for embedding
    because they contain complex textures where modifications are
    harder to perceive visually.
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25):
        """
        Parameters
        ----------
        model_name : str
            YOLO model: 'yolov8n.pt' (nano, default, fast), 'yolov8s.pt', 'yolov8m.pt',
            'yolov8l.pt', 'yolov8x.pt' (larger = more accurate, GPU recommended).
        confidence : float
            Minimum detection confidence threshold.
        """
        self.model_name = model_name
        self.confidence = confidence
        self._model = None
    
    def _load_model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
    
    def detect_objects(self, image: np.ndarray) -> List[dict]:
        """
        Run YOLO detection on an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (H, W, 3) RGB
            
        Returns
        -------
        detections : list of dict
            Each dict has: 'bbox' (x1,y1,x2,y2), 'class', 'confidence'
        """
        self._load_model()
        
        # YOLO expects BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self._model(img_bgr, conf=self.confidence, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                cls = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                class_name = result.names[cls]
                
                detections.append({
                    'bbox': tuple(bbox),
                    'class': class_name,
                    'confidence': conf
                })
        
        return detections
    
    def generate_embedding_mask(self, image: np.ndarray,
                                 expansion_factor: float = 0.2,
                                 use_texture: bool = True) -> np.ndarray:
        """
        Generate a binary mask indicating where to embed secret data.
        
        Parameters
        ----------
        image : np.ndarray
            Cover image (H, W, 3) RGB
        expansion_factor : float
            How much to expand bounding boxes (0.2 = 20% larger)
        use_texture : bool
            If True, also consider texture complexity (high-frequency regions)
            
        Returns
        -------
        mask : np.ndarray
            Binary mask (H, W) with values in [0, 1].
            Higher values = better embedding locations.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float64)
        
        # Step 1: YOLO object detection
        detections = self.detect_objects(image)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Expand bounding box
            bw = x2 - x1
            bh = y2 - y1
            x1 = max(0, int(x1 - bw * expansion_factor))
            y1 = max(0, int(y1 - bh * expansion_factor))
            x2 = min(w, int(x2 + bw * expansion_factor))
            y2 = min(h, int(y2 + bh * expansion_factor))
            
            # Weight by confidence
            mask[y1:y2, x1:x2] = max(mask[y1:y2, x1:x2].max(), det['confidence'])
        
        if use_texture:
            # Step 2: Texture complexity analysis
            texture_mask = self._compute_texture_map(image)
            
            # Combine: use texture in non-detected regions too
            # Normalize texture to [0, 1]
            if texture_mask.max() > 0:
                texture_mask = texture_mask / texture_mask.max()
            
            # Final mask: YOLO regions get high weight, textured regions get moderate weight
            mask = np.clip(mask + 0.5 * texture_mask, 0, 1)
        
        # If no detections and no texture, fall back to uniform mask
        if mask.max() == 0:
            mask = np.ones((h, w), dtype=np.float64)
        else:
            # Normalize
            mask = mask / mask.max()
        
        return mask
    
    def _compute_texture_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute a texture complexity map using edge detection.
        
        High-frequency regions (edges, textures) are better for embedding.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (H, W, 3) RGB
            
        Returns
        -------
        texture_map : np.ndarray
            Texture complexity map (H, W), higher = more complex
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture = np.abs(laplacian)
        
        # Smooth the texture map
        texture = cv2.GaussianBlur(texture, (21, 21), 0)
        
        return texture
    
    def visualize_mask(self, image: np.ndarray, mask: np.ndarray,
                       detections: Optional[List[dict]] = None) -> np.ndarray:
        """
        Create a visualization overlay showing the embedding mask on the image.
        
        Parameters
        ----------
        image : np.ndarray
            Original image (H, W, 3) RGB
        mask : np.ndarray
            Embedding mask (H, W)
        detections : list, optional
            YOLO detections to draw bounding boxes
            
        Returns
        -------
        vis : np.ndarray
            Visualization image (H, W, 3) RGB
        """
        vis = image.copy()
        
        # Create green overlay for embedding regions
        overlay = np.zeros_like(image)
        overlay[:, :, 1] = 255  # Green channel
        
        # Apply mask as alpha blend
        mask_3ch = np.stack([mask] * 3, axis=-1)
        vis = (vis * (1 - 0.3 * mask_3ch) + overlay * 0.3 * mask_3ch).astype(np.uint8)
        
        # Draw bounding boxes
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(vis, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis


def generate_texture_only_mask(image: np.ndarray) -> np.ndarray:
    """
    Generate embedding mask based only on texture (no YOLO needed).
    Useful as a baseline comparison.
    
    Parameters
    ----------
    image : np.ndarray
        Cover image (H, W, 3) RGB
        
    Returns
    -------
    mask : np.ndarray
        Texture-based mask (H, W) normalized to [0, 1]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Combine multiple edge detectors
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    
    texture = laplacian + sobel_x + sobel_y
    texture = cv2.GaussianBlur(texture, (21, 21), 0)
    
    if texture.max() > 0:
        texture = texture / texture.max()
    
    return texture
