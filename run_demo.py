import argparse
import logging
import time

import cv2
import numpy as np

from main_controller import MainController
from utils import Drawer, Event, targets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Visual styling ────────────────────────────────────────────────────────────
# Box and label colours are intentionally distinct from the Drawer overlays
# (which use blue/red) so they remain legible over any background.
_BOX_COLOUR      = (0, 255, 0)      # bright green
_TEXT_COLOUR     = (0, 255, 0)      # bright green
_SHADOW_COLOUR   = (0, 0, 0)        # black drop-shadow for legibility
_FONT            = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE      = 0.65
_FONT_THICKNESS  = 2
_BOX_THICKNESS   = 3

# ── Camera warm-up ───────────────────────────────────────────────────────────
# Minimum number of frames to capture and discard before starting inference.
# Camera sensors need time to complete their initial auto-exposure and
# white-balance passes; frames captured before this settling period are
# often nearly black (mean < 30 DN), which causes the hand detector to
# return 0 detections and makes the pipeline appear broken.
_CAMERA_WARMUP_FRAMES = 30


def _warmup_camera(cap: cv2.VideoCapture) -> None:
    """Discard the first ``_CAMERA_WARMUP_FRAMES`` frames so the sensor has
    time to finish auto-exposure before we start running inference."""
    logger.info("Warming up camera (discarding %d frames)…", _CAMERA_WARMUP_FRAMES)
    for _ in range(_CAMERA_WARMUP_FRAMES):
        cap.read()
    logger.info("Camera warm-up complete.")


# ── Visualization helpers ─────────────────────────────────────────────────────

def _put_text_with_shadow(frame: np.ndarray, text: str, origin, scale: float,
                          colour, thickness: int) -> None:
    """Render *text* with a 1-pixel black drop-shadow for contrast."""
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), _FONT, scale,
                _SHADOW_COLOUR, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), _FONT, scale,
                colour, thickness, cv2.LINE_AA)


def _draw_hand_detections(frame: np.ndarray, bboxes: np.ndarray,
                          ids: np.ndarray, labels: list) -> None:
    """
    Draw a bounding box and a three-line overlay for every confirmed hand.

    This is always called regardless of ``debug_mode`` so the user can
    see that detection is working even when ``--debug`` is not passed.

    Overlay format (above each box):
        HAND DETECTED
        ID <tracker_id>
        <gesture_label>
    """
    bboxes = bboxes.astype(np.int32)
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i]
        gesture = targets[labels[i]] if labels[i] is not None else "—"
        track_id = int(ids[i])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOUR, _BOX_THICKNESS)

        # Three text lines stacked above the top edge of the box.
        # Line height is derived from font metrics for robustness across frames.
        (_, line_h), baseline = cv2.getTextSize(
            "HAND DETECTED", _FONT, _FONT_SCALE, _FONT_THICKNESS
        )
        line_step = line_h + baseline + 4
        lines = ["HAND DETECTED", f"ID {track_id}", gesture]
        for j, text in enumerate(lines):
            # Stack from bottom up: line 0 is closest to the box.
            top_y = y1 - (len(lines) - j) * line_step
            _put_text_with_shadow(frame, text, (x1, top_y),
                                  _FONT_SCALE, _TEXT_COLOUR, _FONT_THICKNESS)


def _draw_debug_overlay(frame: np.ndarray, fps: float,
                        n_detections: int, n_tracks: int) -> None:
    """
    Render the debug status bar in the top-left corner.

    Shows: FPS · detected boxes · confirmed tracks
    """
    lines = [
        f"FPS: {fps:.1f}",
        f"Detections: {n_detections}",
        f"Tracks: {n_tracks}",
    ]
    y = 30
    for text in lines:
        _put_text_with_shadow(frame, text, (10, y),
                              0.75, (0, 200, 255), 2)
        y += 32


# ── Per-frame structured logging ──────────────────────────────────────────────

def _log_detections(bboxes: np.ndarray, ids: np.ndarray,
                    labels: list, prev_count: int) -> int:
    """
    Emit structured log entries for the current frame's detections.

    * INFO  – logged only when the number of detected hands changes
              (hands appear / disappear).  This avoids flooding the terminal
              during steady-state operation.
    * DEBUG – full per-hand details (bbox, id, gesture) logged every frame
              so that ``--log-level DEBUG`` gives a complete trace.

    Returns
    -------
    int  The current detection count (to be stored as ``prev_count`` by the
         caller for the next frame).
    """
    n = bboxes.shape[0] if bboxes is not None else 0

    if n != prev_count:
        if n == 0:
            logger.info("Hands lost — no hands detected in frame")
        else:
            logger.info("Detected hands: %d", n)

    if n > 0 and logger.isEnabledFor(logging.DEBUG):
        for i in range(n):
            box = bboxes[i].tolist()
            gesture = targets[labels[i]] if labels[i] is not None else "None"
            logger.debug(
                "  Hand %d | id=%-4s bbox=[%d, %d, %d, %d] gesture=%s",
                i, ids[i], box[0], box[1], box[2], box[3], gesture,
            )

    return n


def run(args):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Allow the camera sensor to settle its auto-exposure before inference.
    _warmup_camera(cap)

    controller = MainController(args.detector, args.classifier)
    drawer = Drawer()
    debug_mode = args.debug
    prev_hand_count: int = 0   # tracks last-known detection count for state-change logging

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            start_time = time.time()
            bboxes, ids, labels = controller(frame)

            # ── Always-visible detection overlay ────────────────────────
            # Draw bounding boxes and labels for every confirmed hand track.
            # This is NOT gated on debug_mode so users can always verify
            # that the model is working.
            if bboxes is not None and bboxes.shape[0] > 0:
                _draw_hand_detections(frame, bboxes, ids, labels)

            # ── Terminal logging ─────────────────────────────────────────
            n_det = bboxes.shape[0] if bboxes is not None else 0
            prev_hand_count = _log_detections(bboxes, ids, labels, prev_hand_count)

            # ── Debug overlay (FPS + counters) ───────────────────────────
            if debug_mode:
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                _draw_debug_overlay(frame, fps, n_det, len(controller.tracks))

            if len(controller.tracks) > 0:
                count_of_zoom = 0
                thumb_boxes = []
                for trk in controller.tracks:
                    if trk["tracker"].time_since_update < 1:
                        if len(trk['hands']):
                            count_of_zoom += (trk['hands'][-1].gesture == 3)

                            thumb_boxes.append(trk['hands'][-1].bbox)
                            if len(trk['hands']) > 3 and [trk['hands'][-1].gesture, trk['hands'][-2].gesture, trk['hands'][-3].gesture] == [23, 23, 23]:
                                x, y, x2, y2 = map(int, trk['hands'][-1].bbox)
                                x, y, x2, y2 = max(x, 0), max(y, 0), max(x2, 0), max(y2, 0)
                                bbox_area = frame[y:y2, x:x2]
                                blurred_bbox = cv2.GaussianBlur(bbox_area, (51, 51), 10)
                                frame[y:y2, x:x2] = blurred_bbox

                        if trk["hands"].action is not None:
                            if Event.SWIPE_LEFT == trk["hands"].action or  Event.SWIPE_LEFT2 == trk["hands"].action or  Event.SWIPE_LEFT3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.SWIPE_RIGHT == trk["hands"].action or Event.SWIPE_RIGHT2 == trk["hands"].action or Event.SWIPE_RIGHT3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.SWIPE_UP == trk["hands"].action or Event.SWIPE_UP2 == trk["hands"].action or Event.SWIPE_UP3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.SWIPE_DOWN == trk["hands"].action or Event.SWIPE_DOWN2 == trk["hands"].action or Event.SWIPE_DOWN3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.DRAG == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                ...
                            elif Event.DROP == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.FAST_SWIPE_DOWN == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.FAST_SWIPE_UP == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.ZOOM_IN == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.ZOOM_OUT == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.DOUBLE_TAP == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.DRAG2 == trk["hands"].action or Event.DRAG3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                ...
                            elif Event.DROP2 == trk["hands"].action or Event.DROP3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.TAP == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.COUNTERCLOCK == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.CLOCKWISE == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                                
                if count_of_zoom == 2:
                    drawer.draw_two_hands(frame, thumb_boxes)

            # ── Gesture event animation ──────────────────────────────────
            # drawer.draw() renders swipe/tap/zoom event animations on top
            # of the frame.  This is the PRIMARY user-facing feedback for
            # recognised gestures and must NOT be gated on debug_mode.
            frame = drawer.draw(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run demo")
    parser.add_argument(
        "--detector",
        default='models/hand_detector.onnx',
        type=str,
        help="Path to detector onnx model"
    )

    parser.add_argument(
        "--classifier",
        default='models/crops_classifier.onnx',
        type=str,
        help="Path to classifier onnx model",
    )

    parser.add_argument("--debug", required=False, action="store_true", help="Debug mode")
    args = parser.parse_args()
    run(args)
