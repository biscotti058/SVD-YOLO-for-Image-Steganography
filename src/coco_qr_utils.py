"""
COCO Dataset & QR Code Utilities
"""

import numpy as np
import os
import io
from PIL import Image
from typing import Optional, Tuple, List


def generate_qr_secret(text="Progetto Statistical Methods", size=512, border=4, error_correction='H'):
    import qrcode
    ec_map = {
        'L': qrcode.constants.ERROR_CORRECT_L,
        'M': qrcode.constants.ERROR_CORRECT_M,
        'Q': qrcode.constants.ERROR_CORRECT_Q,
        'H': qrcode.constants.ERROR_CORRECT_H,
    }
    qr = qrcode.QRCode(
        version=None,
        error_correction=ec_map.get(error_correction, qrcode.constants.ERROR_CORRECT_H),
        box_size=10,
        border=border,
    )
    qr.add_data(text)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    pil_img = pil_img.resize((size, size), Image.NEAREST)
    return np.array(pil_img)


def verify_qr_decode(image):
    import cv2
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    detector = cv2.QRCodeDetector()
    data, vertices, _ = detector.detectAndDecode(gray)
    if data:
        return data
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    data, vertices, _ = detector.detectAndDecode(enhanced)
    if data:
        return data
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data, vertices, _ = detector.detectAndDecode(binary)
    return data if data else None


COCO_SAMPLE_IDS = [139, 785, 872, 1268, 1503, 2532, 3553, 5477, 7386, 9448]


def coco_url_from_id(image_id, split="val2017"):
    return f"http://images.cocodataset.org/{split}/{image_id:012d}.jpg"


def download_coco_image(image_id, save_dir="images", split="val2017", target_size=(512, 512)):
    import urllib.request
    os.makedirs(save_dir, exist_ok=True)
    filename = f"coco_{image_id:012d}.png"
    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        return filepath
    url = coco_url_from_id(image_id, split)
    print(f"  Downloading COCO image {image_id} ...")
    try:
        response = urllib.request.urlopen(url, timeout=30)
        img_data = response.read()
        pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        if target_size is not None:
            pil_img = pil_img.resize(target_size, Image.LANCZOS)
        pil_img.save(filepath)
        print(f"  Saved: {filepath} ({pil_img.size[0]}x{pil_img.size[1]})")
        return filepath
    except Exception as e:
        print(f"  Failed to download COCO {image_id}: {e}")
        return ""


def download_coco_samples(n=5, save_dir="images", target_size=(512, 512)):
    ids = COCO_SAMPLE_IDS[:n]
    paths = []
    print(f"Downloading {len(ids)} COCO images...")
    for img_id in ids:
        path = download_coco_image(img_id, save_dir, target_size=target_size)
        if path:
            paths.append(path)
    print(f"Downloaded {len(paths)}/{len(ids)} images successfully.")
    return paths


def load_coco_or_fallback(save_dir="images", target_size=(512, 512)):
    import cv2
    if os.path.exists(save_dir):
        existing = [f for f in os.listdir(save_dir) if f.startswith("coco_") and f.endswith(".png")]
        if existing:
            path = os.path.join(save_dir, existing[0])
            img = np.array(Image.open(path).convert("RGB"))
            return img, "coco"
    paths = download_coco_samples(n=1, save_dir=save_dir, target_size=target_size)
    if paths:
        img = np.array(Image.open(paths[0]).convert("RGB"))
        return img, "coco"
    print("COCO download failed. Generating synthetic cover image...")
    h, w = target_size[1], target_size[0]
    img = _generate_synthetic_cover((h, w))
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "synthetic_cover.png")
    Image.fromarray(img).save(path)
    return img, "synthetic"


def _generate_synthetic_cover(size=(512, 512)):
    import cv2
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h // 3):
        ratio = y / (h // 3)
        img[y, :] = [int(135 + 50 * ratio), int(206 + 30 * ratio), int(235 + 20 * ratio)]
    for y in range(h // 3, 2 * h // 3):
        ratio = (y - h // 3) / (h // 3)
        base_g = int(160 - 40 * ratio)
        for x in range(w):
            noise = np.random.randint(-15, 15)
            img[y, x] = [34 + noise, base_g + noise, 34 + noise]
    for y in range(2 * h // 3, h):
        for x in range(w):
            by = (y - 2 * h // 3) % 20
            bx = x % 40
            if by < 2 or (bx < 2 and by > 10):
                img[y, x] = [100, 100, 100]
            else:
                noise = np.random.randint(-10, 10)
                img[y, x] = [180 + noise, 100 + noise, 60 + noise]
    cv2.circle(img, (w // 4, h // 4), 40, (255, 220, 50), -1)
    cv2.rectangle(img, (w // 2, h // 3), (w // 2 + 80, 2 * h // 3), (100, 80, 60), -1)
    cv2.circle(img, (w // 2 + 40, h // 4 + 20), 60, (34, 139, 34), -1)
    return img
