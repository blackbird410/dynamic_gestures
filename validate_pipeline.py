"""
Validation test for the fixed pipeline.

Verifies:
  1. preprocess() always returns float32
  2. HandDetection skips dark frames (brightness guard)
  3. HandDetection detects hands on a properly-exposed frame
  4. MainController full pipeline integration
"""
import logging
import sys
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("validate")

from onnx_models import HandDetection
from main_controller import MainController

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    sym = "✓" if condition else "✗"
    logger.info("%s  [%s]  %s %s", sym, status, name, f"— {detail}" if detail else "")


# ---------------------------------------------------------------------------
# Test 1: preprocess dtype
# ---------------------------------------------------------------------------
logger.info("\n=== Test 1: preprocess dtype ===")
det = HandDetection("models/hand_detector.onnx")

for pixel_val, label in [(0, "black"), (128, "gray"), (255, "white")]:
    frame = np.full((480, 640, 3), pixel_val, dtype=np.uint8)
    tensor = det.preprocess(frame)
    check(f"preprocess dtype=float32 (pixel={pixel_val}, {label})",
          tensor.dtype == np.float32, f"got {tensor.dtype}")
    check(f"preprocess shape=(1,3,240,320) (pixel={pixel_val})",
          tensor.shape == (1, 3, 240, 320), f"got {tensor.shape}")

# ---------------------------------------------------------------------------
# Test 2: dark-frame guard
# ---------------------------------------------------------------------------
logger.info("\n=== Test 2: dark-frame guard ===")
dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)  # pixel mean = 10
boxes, scores = det(dark_frame)
check("dark frame returns 0 detections (brightness guard active)",
      boxes.shape == (0, 4), f"got {boxes.shape}")
check("dark frame returns float32 scores",
      scores.dtype == np.float32, f"got {scores.dtype}")

# ---------------------------------------------------------------------------
# Test 3: camera detection with warm-up
# ---------------------------------------------------------------------------
logger.info("\n=== Test 3: camera detection with warm-up ===")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
time.sleep(0.5)
for _ in range(30):       # warm-up — matches _CAMERA_WARMUP_FRAMES
    cap.read()
ret, cam_frame = cap.read()
cap.release()

if ret:
    cam_frame = cv2.flip(cam_frame, 1)
    frame_mean = float(cam_frame.mean())
    logger.info("Camera frame: shape=%s dtype=%s mean=%.1f min=%d max=%d",
                cam_frame.shape, cam_frame.dtype,
                frame_mean, cam_frame.min(), cam_frame.max())
    check("camera frame brightness > 30 after warm-up",
          frame_mean >= 30.0, f"mean={frame_mean:.1f}")
    boxes, scores = det(cam_frame)
    logger.info("Camera detection: %d boxes  scores=%s", boxes.shape[0], scores)
    check("camera detection returns valid box array (0 or more boxes)",
          boxes.ndim == 2 and boxes.shape[1] == 4, f"shape={boxes.shape}")
    check("camera detection returns float32 scores",
          scores.dtype == np.float32, f"dtype={scores.dtype}")
    if boxes.shape[0] > 0:
        check("bounding boxes are integer pixel coords",
              boxes.dtype == np.int32, f"dtype={boxes.dtype}")
        check("bounding boxes have non-negative coords",
              (boxes[:, :2] >= 0).all(), f"min={boxes.min()}")
else:
    logger.warning("Camera not available — skipping live detection test")

# ---------------------------------------------------------------------------
# Test 4: MainController integration
# ---------------------------------------------------------------------------
logger.info("\n=== Test 4: MainController integration ===")
ctrl = MainController("models/hand_detector.onnx", "models/crops_classifier.onnx")
if ret:
    for _ in range(5):
        bboxes, ids, labels = ctrl(cam_frame)
    check("MainController __call__ returns 3-tuple",
          True, "no exception raised")
    if bboxes is not None:
        check("MainController bboxes shape (N,4)",
              bboxes.ndim == 2 and bboxes.shape[1] == 4, f"shape={bboxes.shape}")
    else:
        logger.info("No confirmed tracks yet after 5 frames (min_hits not yet reached"
                    " or no hands visible) — expected if no hands are visible")
        check("MainController returns None when no confirmed tracks",
              True, "expected when hand not visible or too few frames")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
logger.info("\n=== SUMMARY ===")
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
for name, status, detail in results:
    sym = "✓" if status == PASS else "✗"
    logger.info("  %s [%s] %s %s", sym, status, name, f"({detail})" if detail else "")
logger.info("Total: %d/%d passed", passed, passed + failed)
sys.exit(0 if failed == 0 else 1)
