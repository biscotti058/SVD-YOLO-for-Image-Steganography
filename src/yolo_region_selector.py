"""
YOLO-Based Region Selector for Steganography
"""

import numpy as np
import cv2
from typing import List, Optional


class YOLORegionSelector:
    def __init__(self, model_name="yolov8n.pt", confidence=0.25):
        self.model_name = model_name
        self.confidence = confidence
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)

    def detect_objects(self, image):
        self._load_model()
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

    def generate_embedding_mask(self, image, expansion_factor=0.2, use_texture=True):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float64)
        detections = self.detect_objects(image)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            bw = x2 - x1
            bh = y2 - y1
            x1 = max(0, int(x1 - bw * expansion_factor))
            y1 = max(0, int(y1 - bh * expansion_factor))
            x2 = min(w, int(x2 + bw * expansion_factor))
            y2 = min(h, int(y2 + bh * expansion_factor))
            mask[y1:y2, x1:x2] = max(mask[y1:y2, x1:x2].max(), det['confidence'])
        if use_texture:
            texture_mask = self._compute_texture_map(image)
            if texture_mask.max() > 0:
                texture_mask = texture_mask / texture_mask.max()
            mask = np.clip(mask + 0.5 * texture_mask, 0, 1)
        if mask.max() == 0:
            mask = np.ones((h, w), dtype=np.float64)
        else:
            mask = mask / mask.max()
        return mask

    def _compute_texture_map(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture = np.abs(laplacian)
        texture = cv2.GaussianBlur(texture, (21, 21), 0)
        return texture

    def visualize_mask(self, image, mask, detections=None):
        vis = image.copy()
        overlay = np.zeros_like(image)
        overlay[:, :, 1] = 255
        mask_3ch = np.stack([mask] * 3, axis=-1)
        vis = (vis * (1 - 0.3 * mask_3ch) + overlay * 0.3 * mask_3ch).astype(np.uint8)
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(vis, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return vis


def generate_texture_only_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    texture = laplacian + sobel_x + sobel_y
    texture = cv2.GaussianBlur(texture, (21, 21), 0)
    if texture.max() > 0:
        texture = texture / texture.max()
    return texture
