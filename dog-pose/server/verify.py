"""
Verification harness for the model.py pipeline.

Runs run_inference() on the test PNGs at the repo root and reports:
  - count of keypoints above 0.05 / 0.30 / 0.60
  - bounding box of high-confidence (>=0.30) keypoints
  - spread (diagonal of that bbox) as a fraction of the image's full extent
  - per-keypoint dump (id, name, x, y, conf)

Used to compare BEFORE/AFTER the A1 (letterbox) + A3 (failure detect) fixes.

Run from the server dir:
    .venv/bin/python verify.py
"""
import os
import sys
import math

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SERVER_DIR, "..", ".."))

TEST_IMAGES = [
    os.path.join(REPO_ROOT, "Bounding box efforts resolution too large.png"),
    os.path.join(REPO_ROOT, "dog_pose_composite first working.png"),
]

KEYPOINT_NAMES = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
    "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
    "neck_base", "neck_end", "throat_base", "throat_end",
    "back_base", "back_end", "back_middle", "tail_base", "tail_end",
    "front_left_thai", "front_left_knee", "front_left_paw",
    "front_right_thai", "front_right_knee", "front_right_paw",
    "back_left_paw", "back_left_thai", "back_right_thai",
    "back_left_knee", "back_right_knee", "back_right_paw",
    "belly_bottom", "body_middle_right", "body_middle_left",
]


def summarise(result, image_w, image_h):
    """result may be either a list of keypoints (old shape) or a dict (new shape)."""
    if isinstance(result, dict):
        kps = result.get("keypoints", [])
        bbox = result.get("bbox")
        failed = result.get("failed", False)
        silhouette = result.get("silhouette")
    else:
        kps = result
        bbox = None
        failed = False
        silhouette = None

    n_05 = sum(1 for k in kps if k["confidence"] >= 0.05)
    n_30 = sum(1 for k in kps if k["confidence"] >= 0.30)
    n_60 = sum(1 for k in kps if k["confidence"] >= 0.60)

    hi = [k for k in kps if k["confidence"] >= 0.30]
    if hi:
        xs = [k["x"] for k in hi]
        ys = [k["y"] for k in hi]
        kp_diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    else:
        kp_diag = 0.0

    img_diag = math.hypot(image_w, image_h)
    if bbox:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        bbox_diag = math.hypot(bw, bh)
    else:
        bbox_diag = img_diag

    print(f"  image size   : {image_w} x {image_h}  (diag {img_diag:.0f})")
    if bbox:
        print(f"  YOLO bbox    : ({bbox[0]:.0f},{bbox[1]:.0f})-({bbox[2]:.0f},{bbox[3]:.0f})  diag {bbox_diag:.0f}")
    print(f"  conf>=0.05   : {n_05} / {len(kps)}")
    print(f"  conf>=0.30   : {n_30} / {len(kps)}")
    print(f"  conf>=0.60   : {n_60} / {len(kps)}")
    print(f"  kp spread    : {kp_diag:.0f}  ({100*kp_diag/bbox_diag:.1f}% of bbox diag, "
          f"{100*kp_diag/img_diag:.1f}% of image diag)")
    print(f"  failed flag  : {failed}")
    if silhouette is not None:
        sxs = [p[0] for p in silhouette]
        sys = [p[1] for p in silhouette]
        sbbox = (min(sxs), min(sys), max(sxs), max(sys))
        print(f"  silhouette   : {len(silhouette)} pts  bbox "
              f"({sbbox[0]:.0f},{sbbox[1]:.0f})-({sbbox[2]:.0f},{sbbox[3]:.0f})")
    else:
        print(f"  silhouette   : (none)")
    print()
    print(f"  {'id':>3} {'name':<22} {'x':>8} {'y':>8} {'conf':>6}")
    for k in kps:
        if k["confidence"] < 0.05:
            continue
        name = KEYPOINT_NAMES[k["id"]] if k["id"] < len(KEYPOINT_NAMES) else "?"
        print(f"  {k['id']:>3} {name:<22} {k['x']:>8.1f} {k['y']:>8.1f} {k['confidence']:>6.3f}")


def main():
    # Import after configuring so tensorflow CUDA-disable env var sticks
    from PIL import Image, ImageOps
    from model import run_inference

    for path in TEST_IMAGES:
        if not os.path.exists(path):
            print(f"!! missing: {path}")
            continue
        print("=" * 72)
        print(f"IMAGE: {os.path.basename(path)}")
        print("=" * 72)
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        w, h = img.size
        with open(path, "rb") as f:
            result = run_inference(f)
        summarise(result, w, h)
        print()


if __name__ == "__main__":
    main()
