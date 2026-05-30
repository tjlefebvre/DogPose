import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU — tensorflow-metal segfaults on M-series

import math
import numpy as np
from PIL import Image, ImageOps
from dlclive import DLCLive, Processor
from ultralytics import YOLO

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = SERVER_DIR   # TF snapshot-700000 checkpoint

# Downscale long side to this before YOLO detection.
INFER_SIZE = 640

# DLC was trained on ~400px crops. Letterbox the YOLO crop to this square
# before feeding DLC, so the input shape matches the training distribution
# regardless of crop aspect ratio.
DLC_INPUT_SIZE = 400

# A keypoint-spread bounding box smaller than this fraction of the YOLO bbox
# diagonal is treated as a model-collapse failure.
FAILURE_SPREAD_RATIO = 0.20

# Confidence floor for spread computation (matches client-side CONFIDENCE_DRAW).
SPREAD_CONFIDENCE_FLOOR = 0.30

# COCO animal classes (dog=16, cat=15, horse=17, cow=19, sheep=18, bear=21, etc.)
ANIMAL_CLASSES = [16]

dlc  = DLCLive(MODEL_PATH, processor=Processor())
yolo = YOLO("yolov8n.pt")
_initialized = False


def _resize(frame):
    """Downscale so the longest side == INFER_SIZE. Returns (small_frame, scale)."""
    h, w = frame.shape[:2]
    scale = INFER_SIZE / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return frame, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    resized = np.array(Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS))
    return resized, scale


def _detect_crop(small_frame, margin=0.12):
    """Locate the largest animal bbox in small_frame.

    Returns (crop, ox, oy, bbox_small, yolo_found) where:
      - crop:        cropped pixel array (with margin)
      - ox, oy:      offset of crop inside small_frame
      - bbox_small:  YOLO bbox before margin, in small_frame coords (x1, y1, x2, y2).
                     If YOLO finds nothing, this is the full small_frame bounds.
      - yolo_found:  True iff YOLO returned at least one animal box.
    """
    h, w = small_frame.shape[:2]
    results = yolo(small_frame, classes=ANIMAL_CLASSES, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print(f"  YOLO: no animal — using full {w}x{h} frame")
        return small_frame, 0, 0, (0.0, 0.0, float(w), float(h)), False

    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
    best  = int(areas.argmax())
    bx1, by1, bx2, by2 = [float(v) for v in boxes.xyxy[best].cpu().numpy()]
    conf  = float(boxes.conf[best].cpu())
    label = results[0].names[int(boxes.cls[best].cpu())]
    print(f"  YOLO: {label} ({conf:.2f})  bbox ({int(bx1)},{int(by1)})-({int(bx2)},{int(by2)})")

    bw, bh = bx2 - bx1, by2 - by1
    x1 = int(max(0, bx1 - bw * margin))
    y1 = int(max(0, by1 - bh * margin))
    x2 = int(min(w, bx2 + bw * margin))
    y2 = int(min(h, by2 + bh * margin))

    crop = small_frame[y1:y2, x1:x2]
    print(f"  crop: {crop.shape[1]}x{crop.shape[0]}  offset ({x1},{y1})")
    return crop, x1, y1, (bx1, by1, bx2, by2), True


def _letterbox_to_dlc(crop):
    """Resize crop to DLC_INPUT_SIZE x DLC_INPUT_SIZE square, preserving aspect ratio
    by centred zero-padding. Returns (padded, scale, pad_x, pad_y) so keypoints
    predicted on `padded` can be mapped back via:
        x_crop = (x_padded - pad_x) / scale
        y_crop = (y_padded - pad_y) / scale
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        # Degenerate crop — return a black square; DLC will produce garbage,
        # which the spread check will catch.
        padded = np.zeros((DLC_INPUT_SIZE, DLC_INPUT_SIZE, 3), dtype=np.uint8)
        return padded, 1.0, 0, 0

    scale = DLC_INPUT_SIZE / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = np.array(Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS))

    pad_x = (DLC_INPUT_SIZE - new_w) // 2
    pad_y = (DLC_INPUT_SIZE - new_h) // 2
    padded = np.zeros((DLC_INPUT_SIZE, DLC_INPUT_SIZE, 3), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    print(f"  letterbox: {w}x{h} -> {new_w}x{new_h} pad ({pad_x},{pad_y}) "
          f"into {DLC_INPUT_SIZE}x{DLC_INPUT_SIZE}")
    return padded, scale, pad_x, pad_y


def _spread_failed(keypoints, bbox_orig):
    """Diagonal of the high-confidence keypoint bbox vs the YOLO bbox diagonal.
    Returns True if the keypoints have collapsed to a small region relative
    to the detected animal."""
    hi = [k for k in keypoints if k["confidence"] >= SPREAD_CONFIDENCE_FLOOR]
    if len(hi) < 2:
        return True
    xs = [k["x"] for k in hi]
    ys = [k["y"] for k in hi]
    kp_diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    bbox_diag = math.hypot(bbox_orig[2] - bbox_orig[0], bbox_orig[3] - bbox_orig[1])
    if bbox_diag <= 0:
        return True
    ratio = kp_diag / bbox_diag
    print(f"  spread: kp_diag={kp_diag:.0f}  bbox_diag={bbox_diag:.0f}  ratio={ratio:.2f}")
    return ratio < FAILURE_SPREAD_RATIO


def run_inference(image_file):
    global _initialized

    # 1. Load + correct EXIF orientation
    img   = ImageOps.exif_transpose(Image.open(image_file)).convert("RGB")
    frame = np.array(img)
    H, W = frame.shape[:2]
    print(f"Image: {W}x{H}")

    # 2. Downscale for detection
    small, scale = _resize(frame)
    print(f"  scaled to {small.shape[1]}x{small.shape[0]}  (scale={scale:.4f})")

    # 3. Detect + crop within the downscaled frame
    crop, ox, oy, bbox_small, yolo_found = _detect_crop(small)

    # 4. Letterbox the crop to a square matching the DLC training input shape
    padded, l_scale, pad_x, pad_y = _letterbox_to_dlc(crop)

    # 5. Pose estimation on the padded square
    if not _initialized:
        dlc.init_inference(padded)
        _initialized = True
    raw = dlc.get_pose(padded)

    # 6. Unmap: padded -> crop -> small_frame -> original
    keypoints = []
    for i, (x, y, conf) in enumerate(raw):
        x_crop = (float(x) - pad_x) / l_scale
        y_crop = (float(y) - pad_y) / l_scale
        x_small = x_crop + ox
        y_small = y_crop + oy
        keypoints.append({
            "id":         i,
            "x":          x_small / scale,
            "y":          y_small / scale,
            "confidence": float(conf),
        })

    # 7. Map the YOLO bbox to original-image coords
    bbox_orig = [
        bbox_small[0] / scale,
        bbox_small[1] / scale,
        bbox_small[2] / scale,
        bbox_small[3] / scale,
    ]

    # 8. Failure detection: collapsed keypoints, OR YOLO found nothing.
    failed = (not yolo_found) or _spread_failed(keypoints, bbox_orig)

    return {
        "keypoints": keypoints,
        "bbox": bbox_orig,
        "failed": failed,
        "image_size": [W, H],
    }
