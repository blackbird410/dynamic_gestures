# Work Log — 2026-03-10

## Pipeline Stabilisation: ONNX Inference Reliability, Always-On Visual Feedback, and Structured Observability

**Branch:** `refactor/stabilize-pipeline`  
**Commits:** `fbbc0e2` → `7750362` (6 commits)  
**Related branch merged to main:** `feat/dynamic-ONNX-execution` (`d7ff7a4`)

---

## Context

Following the merge of `feat/dynamic-ONNX-execution`, which introduced runtime-selected ONNX execution providers (CoreML + CPU on Apple Silicon), the pipeline exhibited intermittent and sometimes total absence of hand detections during live demo runs. Initial observation suggested the model had regressed, but the failure was silent — no exceptions, no logged errors, and correct-looking inference paths in code review.

A diagnostic session was opened to isolate the root cause before attributing the failure to the provider-selection refactoring.

---

## Investigation

### Tool: `debug_detector.py` (ephemeral script)

A minimal diagnostic script was written to capture the first several frames from the live camera, run `HandDetection` inference on each, and log frame statistics (mean, max, shape) and detection results.

**Key finding:** The very first real frame captured by `cv2.VideoCapture` had `max=25` and `mean≈12`. Camera auto-exposure had not settled; the sensor was producing near-black frames. The hand-detection model — correctly — returned 0 detections with 0 confidence scores on those frames. With no warmup in the pipeline, all startup frames triggered this condition.

### Tool: `debug_coreml.py` (ephemeral script)

A second diagnostic forced inference through CoreML EP to rule out a provider-selection bug. This confirmed a separate but pre-existing issue: CoreML EP crashes with a runtime shape error when the NMS graph output tensor contains zero elements (`{0,3}`). The `-cpu_only = True` flag on `HandDetection` (introduced in `feat/dynamic-ONNX-execution`) is the correct mitigation and was retained.

### NumPy dtype inspection

Code review of `preprocess()` revealed that `(uint8_image − float32_mean) / float32_std` will produce a `float64` array under NumPy 2.x type-promotion semantics. ONNX Runtime accepted the `float64` input without raising an error but produced degraded output tensors. This was a latent defect activated by the NumPy 2.x dependency.

### Visualization audit

The `run_demo.py` main loop was audited. Two visualization issues were found:

1. `cv2.rectangle` / `cv2.putText` bounding-box drawing was inside `if debug_mode:` — users never saw detected hands unless they passed `--debug`.
2. `drawer.draw(frame)` (gesture event animation from the `Drawer` class) was also gated on `if debug_mode:` — the primary user-facing gesture feedback was architecturally unreachable in normal operation.

---

## Scope of Work

1. Fix the NumPy float64 type-promotion defect in `preprocess()`
2. Add a brightness guard to `HandDetection` to handle AE-settling frames gracefully
3. Add configurable confidence filtering to `HandDetection`
4. Add a camera warmup phase to `run_demo.py`
5. Decouple all user-facing visualization from `debug_mode`
6. Add structured logging throughout the inference pipeline
7. Write and pass a regression test suite covering the above

---

## Technical Changes

### `onnx_models.py`

**`preprocess()` — dtype fix**

Added `image.astype(np.float32)` before the normalization arithmetic. Prevents NumPy 2.x from widening the intermediate result to `float64`, which ONNX Runtime would silently accept but produce broken outputs for.

```python
# Before
image = (image - self.mean) / self.std

# After
image = (image.astype(np.float32) - self.mean) / self.std
```

**`HandDetection` — brightness guard**

Added `_MIN_FRAME_MEAN = 30.0` class attribute. `__call__` computes `frame.mean()` before inference; frames below threshold return empty arrays immediately with a `DEBUG`-level log message. This eliminates spurious "no detections" conditions during camera startup without modifying the model.

**`HandDetection.__init__` — score threshold**

Added `score_threshold: float = 0.5` parameter. Backward compatible (default matches previous implicit behaviour). Logged at `INFO` on initialization along with model path.

**`HandDetection.__call__` — confidence filtering and safe copy**

Post-NMS scores are filtered by `self.score_threshold`. Debug logs record the raw detection count + score range and the post-filter kept count. The bounding-box array is now `.copy().astype(np.float32)` before in-place pixel-coordinate scaling to avoid mutating the raw model output buffer.

**Docstrings**

`preprocess()`, `HandDetection`, and `HandDetection.__init__` updated to document: SSD+NMS graph topology, CoreML EP zero-element NMS crash rationale for `_cpu_only = True`, `score_threshold` semantics, and the `_MIN_FRAME_MEAN` guard.

---

### `run_demo.py`

**Camera warmup**

`_CAMERA_WARMUP_FRAMES = 30` module constant and `_warmup_camera(cap: cv2.VideoCapture) -> None` function. Called once before the main inference loop begins. Discards 30 frames to allow camera AE to stabilise.

**Visual styling constants**

`_BOX_COLOUR`, `_TEXT_COLOUR`, `_SHADOW_COLOUR`, `_FONT`, `_FONT_SCALE`, `_FONT_THICKNESS`, `_BOX_THICKNESS` defined at module level. Centralises all rendering parameters.

**`_put_text_with_shadow()`**

Helper that renders a text string with a 1 px black drop-shadow before the coloured foreground pass. Provides legibility on arbitrary scene backgrounds without a filled rectangle background.

**`_draw_hand_detections()` — always-on visualization**

`_draw_hand_detections(frame, bboxes, ids, labels)` replaces the previous inline bounding-box drawing. Called unconditionally in the main loop (not gated on `debug_mode`). Draws: green bounding box at pixel coordinates; 3-line text overlay above the box ("HAND DETECTED" / "ID N" / gesture label). Line spacing derived from `cv2.getTextSize` for font-metric accuracy.

**`_draw_debug_overlay()` — gated stats panel**

`_draw_debug_overlay(frame, fps, n_detections, n_tracks)` renders FPS, raw detection count, and confirmed track count in the top-left corner. Only called when `--debug` is active.

**`_log_detections()` — state-change logging**

`_log_detections(bboxes, ids, labels, prev_count)` returns the new hand count. Fires an `INFO` log only when the count changes (hand appears or disappears). Emits per-hand `DEBUG` logs with bbox coordinates and gesture label every frame for full trace capability.

**`drawer.draw(frame)` bugfix**

Moved `drawer.draw(frame)` outside the `if debug_mode:` block. Gesture event animations (the primary user-facing feedback for recognised gestures) now always render. This was the most significant user-impact issue.

**Loop structure (corrected):**

```
inference → _draw_hand_detections (unconditional)
          → _log_detections (unconditional)
          → _draw_debug_overlay (debug_mode only)
          → track event processing
          → drawer.draw(frame) (unconditional)  ← moved here
          → cv2.imshow
```

---

### `main_controller.py`

Added `import logging` and `logger = logging.getLogger(__name__)`. Added two `logger.debug()` calls in `__call__`: one after detection (reports raw count, frame count, active tracks) and one after `self.update()` (reports confirmed track count). No changes to public interface or algorithm.

---

## Testing and Validation

`validate_pipeline.py` was written and committed as a permanent regression suite.

**Test coverage:**

| Test                                        | What it Checks                                                                   |
| ------------------------------------------- | -------------------------------------------------------------------------------- |
| `test_preprocess_dtype_black`               | `preprocess()` output is `float32` for a zeroed frame                            |
| `test_preprocess_dtype_gray`                | `preprocess()` output is `float32` for a mid-tone frame                          |
| `test_preprocess_dtype_white`               | `preprocess()` output is `float32` for a saturated frame                         |
| `test_preprocess_shape_black`               | Output shape is `(1, 3, H, W)`                                                   |
| `test_preprocess_shape_gray`                | Output shape is `(1, 3, H, W)`                                                   |
| `test_preprocess_shape_white`               | Output shape is `(1, 3, H, W)`                                                   |
| `test_brightness_guard_skips_dark_frame`    | `HandDetection` returns empty arrays for `mean < 30` frames                      |
| `test_brightness_guard_passes_bright_frame` | No guard skip on a normally-exposed frame                                        |
| `test_camera_brightness_after_warmup`       | Live camera frame `mean ≥ 30` after warmup                                       |
| `test_camera_detection_returns_arrays`      | `HandDetection` output arrays are non-None and have matching first dimensions    |
| `test_camera_detection_scores_in_range`     | All returned scores are in `[0.0, 1.0]`                                          |
| `test_main_controller_single_frame`         | `MainController.__call__` returns `(bboxes, ids, labels)` with consistent shapes |
| `test_main_controller_multiple_frames`      | Tracker remains stable across 5 sequential frames                                |

**Result:** 13 / 13 PASS

Syntax check:

```
python -m py_compile run_demo.py main_controller.py onnx_models.py
# → clean
```

---

## Outcome

- All three root causes resolved with no regressions introduced.
- Hand detections are now always visible in the video stream without requiring `--debug`.
- Gesture animations (`Drawer`) now always render.
- Pipeline is robust to camera AE-settling at startup.
- NumPy 2.x float64 type-promotion defect eliminated.
- Confidence threshold is now an explicit, logged, tunable parameter.
- Structured logging is available at `DEBUG` level for full trace and at `INFO` level for state-change events.
- No breaking changes to any public API.
- No model weights modified.
