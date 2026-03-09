import logging
import platform
from abc import ABC

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_execution_providers(cpu_only: bool = False) -> tuple[list[str], list[dict]]:
    """
    Dynamically select ONNX Runtime execution providers based on the current
    host platform and what is available in the installed ORT build.

    Priority order:
      - Apple Silicon (macOS arm64) : CoreMLExecutionProvider → CPU
      - NVIDIA GPU (CUDA available)  : CUDAExecutionProvider  → CPU
      - DirectML (Windows/AMD/Intel) : DmlExecutionProvider   → CPU
      - Fallback                     : CPUExecutionProvider

    Parameters
    ----------
    cpu_only : bool
        When True, skip all accelerated EPs and use CPUExecutionProvider only.
        Use this for models whose graphs contain ops (e.g. NonMaxSuppression)
        that produce zero-element tensors at runtime — a case that CoreML EP
        does not support.

    Returns
    -------
    providers : list[str]
        Ordered list of provider names accepted by ``ort.InferenceSession``.
    provider_options : list[dict]
        Matching list of per-provider option dicts (empty dict where unused).
    """
    available = ort.get_available_providers()
    logger.info("Detected ONNX Runtime providers: %s", available)

    providers: list[str] = []
    provider_options: list[dict] = []

    if cpu_only:
        logger.info("cpu_only=True: skipping accelerated EPs, using CPUExecutionProvider.")
        return ["CPUExecutionProvider"], [{}]

    # --- Apple Silicon (macOS arm64) ---
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
            # Empty options → CoreML defaults (All compute units: ANE + GPU + CPU).
            # Unsupported ops automatically delegate to CPUExecutionProvider.
            provider_options.append({})
            logger.info("Using CoreMLExecutionProvider for inference.")
        else:
            logger.warning(
                "CoreMLExecutionProvider not found; falling back to CPU. "
                "Consider upgrading to onnxruntime >= 1.16.0."
            )

    # --- NVIDIA CUDA ---
    elif "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        provider_options.append(
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
        )
        logger.info("Using CUDAExecutionProvider for inference.")

    # --- DirectML (Windows, AMD/Intel GPU) ---
    elif "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
        provider_options.append({"device_id": 0})
        logger.info("Using DmlExecutionProvider for inference.")

    # --- CPU fallback (always appended last) ---
    providers.append("CPUExecutionProvider")
    provider_options.append({})

    if len(providers) == 1:
        logger.info("Using CPUExecutionProvider for inference.")

    return providers, provider_options


class OnnxModel(ABC):
    def __init__(self, model_path, image_size):
        self.model_path = model_path
        self.image_size = image_size
        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = np.array([128, 128, 128], dtype=np.float32)
        options, providers, prov_opts = self._build_session_options(cpu_only=getattr(self, "_cpu_only", False))
        self.sess = ort.InferenceSession(
            model_path, sess_options=options, providers=providers, provider_options=prov_opts
        )
        logger.info("Session loaded with providers: %s", self.sess.get_providers())
        self._get_input_output()

    def preprocess(self, frame):
        """
        Preprocess frame
        Parameters
        ----------
        frame : np.ndarray
            Frame to preprocess
        Returns
        -------
        np.ndarray
            Preprocessed frame
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = (image - self.mean) / self.std
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        return image

    def _get_input_output(self):
        inputs = self.sess.get_inputs()
        self.inputs = "".join(
            [
                f"\n {i}: {input.name}" f" Shape: ({','.join(map(str, input.shape))})" f" Dtype: {input.type}"
                for i, input in enumerate(inputs)
            ]
        )

        outputs = self.sess.get_outputs()
        self.outputs = "".join(
            [
                f"\n {i}: {output.name}" f" Shape: ({','.join(map(str, output.shape))})" f" Dtype: {output.type}"
                for i, output in enumerate(outputs)
            ]
        )

    @staticmethod
    def _build_session_options(cpu_only: bool = False) -> tuple:
        """
        Build ``ort.SessionOptions`` and resolve execution providers.

        Parameters
        ----------
        cpu_only : bool
            Passed through to ``get_execution_providers``.

        Returns
        -------
        options : ort.SessionOptions
        providers : list[str]
        provider_options : list[dict]
        """
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers, provider_options = get_execution_providers(cpu_only=cpu_only)
        return options, providers, provider_options

    def __repr__(self):
        return (
            f"Providers: {self.sess.get_providers()}\n"
            f"Model: {self.sess.get_modelmeta().description}\n"
            f"Version: {self.sess.get_modelmeta().version}\n"
            f"Inputs: {self.inputs}\n"
            f"Outputs: {self.outputs}"
        )

class HandDetection(OnnxModel):
    # The hand detector uses NonMaxSuppression, whose output becomes a
    # zero-element tensor when no hands are present.  CoreML EP cannot
    # process zero-element tensors, so we force CPU-only for this model.
    _cpu_only = True

    def __init__(self, model_path, image_size=(320, 240)):
        super().__init__(model_path, image_size)
        # Reuse self.sess created by OnnxModel.__init__ (provider-aware).
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [output.name for output in self.sess.get_outputs()]
        
    def __call__(self, frame):
        input_tensor = self.preprocess(frame)
        boxes, _, probs = self.sess.run(self.output_names, {self.input_name: input_tensor})
        width, height = frame.shape[1], frame.shape[0]
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
        return boxes.astype(np.int32), probs


class HandClassification(OnnxModel):
    def __init__(self, model_path, image_size=(128, 128)):
        super().__init__(model_path, image_size)

    @staticmethod
    def get_square(box, image):
        """
        Get square box
        Parameters
        ----------
        box : np.ndarray
            Box coordinates (x1, y1, x2, y2)
        image : np.ndarray
            Image for shape
        """
        height, width, _ = image.shape
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        if h < w:
            y0 = y0 - int((w - h) / 2)
            y1 = y0 + w
        if h > w:
            x0 = x0 - int((h - w) / 2)
            x1 = x0 + h
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width - 1, x1)
        y1 = min(height - 1, y1)
        return x0, y0, x1, y1

    def get_crops(self, frame, bboxes):
        """
        Get crops from frame
        Parameters
        ----------
        frame : np.ndarray
            Frame to crop from bboxes
        bboxes : np.ndarray
            Bounding boxes

        Returns
        -------
        crops : np.ndarray
            Crops from frame
        """
        crops = []
        for bbox in bboxes:
            bbox = self.get_square(bbox, frame)
            crop = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            crops.append(crop)
        return crops

    def __call__(self, image, bboxes):
        """
        Get predictions from model
        Parameters
        ----------
        image : np.ndarray
            Image to predict
        bboxes : np.ndarray
            Bounding boxes

        Returns
        -------
        predictions : np.ndarray
            Predictions from model
        """
        crops = self.get_crops(image, bboxes)
        crops = [self.preprocess(crop) for crop in crops]
        input_name = self.sess.get_inputs()[0].name
        outputs = self.sess.run(None, {input_name: np.concatenate(crops, axis=0)})[0]
        labels = np.argmax(outputs, axis=1)
        return labels
