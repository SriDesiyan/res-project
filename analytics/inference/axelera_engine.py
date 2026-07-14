"""
analytics/inference/axelera_engine.py
=======================================
AxeleraInferenceEngine — targets the Axelera Metis Compute Board (RK3588 + AIPU)
via the Voyager SDK.

Architecture
------------
The Axelera Metis board has two compute units:
  1. RK3588 ARM CPU     — handles preprocessing, postprocessing, tracking,
                          MediaPipe (CPU), and all business logic.
  2. Metis AIPU         — accelerates the neural network forward passes
                          (YOLO detection, OSNet ReID, ResNet50 classifiers).

Voyager SDK integration points
-------------------------------
The Voyager SDK provides:
  - ``axelera.runtime`` — Python bindings to load compiled AIPU models (.axm files)
  - ``axelera.compiler`` — offline compilation tool (run separately via compile_axelera.py)

This engine loads pre-compiled .axm files from ``weights/axelera/``.
If an .axm file is not found, it falls back to ONNX Runtime (CPU).
If the Voyager SDK is not installed, it falls back to ONNX Runtime (CPU).

Fallback chain
--------------
  Axelera AIPU (.axm)
      ↓ (not found / SDK missing)
  ONNX Runtime (CPU)
      ↓ (onnxruntime missing / .onnx not found)
  PyTorch CPU

Usage on development machine (no Metis board)
----------------------------------------------
Set backend="axelera" — the engine will silently fall back to ONNX/CPU
and print a warning.  All output schemas are identical regardless of
the active fallback tier.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from analytics.inference.base_engine import BaseInferenceEngine
from analytics.inference.onnx_engine import OnnxInferenceEngine

# ---------------------------------------------------------------------------
# Attempt to import Voyager SDK
# ---------------------------------------------------------------------------
try:
    import axelera.runtime as _axrt  # type: ignore[import]
    _AXELERA_SDK_AVAILABLE = True
    print("[AxeleraEngine] Voyager SDK detected")
except ImportError:
    _axrt = None
    _AXELERA_SDK_AVAILABLE = False
    print("[AxeleraEngine] Voyager SDK NOT installed — will use ONNX/CPU fallback")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DISHWARE_INDICES = {504, 923, 968, 868, 809, 659, 440, 737, 898, 907}
_DEFAULT_FOOD_CLASSES = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                          49, 50, 51, 52, 53, 54, 55]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(bgr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """BGR → NCHW float32 ImageNet-normalised."""
    h, w = hw
    rgb = cv2.cvtColor(cv2.resize(bgr, (w, h)), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return x.transpose(2, 0, 1)[np.newaxis]


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


class _AxeleraModel:
    """
    Thin wrapper around an Axelera .axm compiled model.

    Handles both the Voyager SDK (when available) and a numpy stub
    (for testing without hardware).
    """

    def __init__(self, axm_path: Path, fallback_onnx_session=None) -> None:
        self._axm_path = axm_path
        self._session = fallback_onnx_session
        self._axm_model = None

        if _AXELERA_SDK_AVAILABLE and axm_path.exists():
            try:
                self._axm_model = _axrt.load_model(str(axm_path))
                print(f"[AxeleraModel] Loaded AIPU model: {axm_path.name}")
            except Exception as exc:
                print(f"[AxeleraModel] WARNING: Failed to load {axm_path.name}: {exc}")
        elif not axm_path.exists():
            print(f"[AxeleraModel] .axm not found: {axm_path} — using ONNX fallback")

    @property
    def available(self) -> bool:
        return self._axm_model is not None

    def run(self, inputs: np.ndarray, input_name: str = "input") -> np.ndarray:
        """Run inference — AIPU or ONNX fallback."""
        if self._axm_model is not None:
            # Voyager SDK inference call
            return _axrt.run(self._axm_model, {input_name: inputs})[0]
        if self._session is not None:
            in_name = self._session.get_inputs()[0].name
            return self._session.run(None, {in_name: inputs})[0]
        # Ultimate fallback — return zeros
        return np.zeros((1, 512), dtype=np.float32)


class AxeleraInferenceEngine(BaseInferenceEngine):
    """
    Axelera Metis AIPU inference engine.

    For each model it tries (in order):
      1. Load the compiled .axm from ``weights/axelera/``
      2. Fall back to ONNX Runtime session from ``weights/onnx/``
      3. Fall back to PyTorch CPU (via OnnxInferenceEngine's fallback path)

    YOLO detection + BoT-SORT tracking still uses the Ultralytics runtime
    (which can target the ONNX backend or the AIPU via a Voyager YOLO plugin
    when available).

    Parameters
    ----------
    yolo_model_path : Path
        Path to yolov8n.pt or yolov8n.onnx.
    tracker_cfg : str
        BoT-SORT config path.
    axelera_dir : Path
        Directory with .axm compiled models (``weights/axelera/``).
    onnx_dir : Path
        Directory with .onnx export models (``weights/onnx/``).
    embedding_dir : Path
        MediaPipe .task files + gallery .pt files.
    conf : float
        YOLO confidence threshold.
    """

    def __init__(
        self,
        yolo_model_path: Path,
        tracker_cfg: str,
        axelera_dir: Path,
        onnx_dir: Path,
        embedding_dir: Path,
        conf: float = 0.25,
    ) -> None:
        self._axelera_dir = Path(axelera_dir)
        self._onnx_dir = Path(onnx_dir)
        self._embedding_dir = Path(embedding_dir)
        self._conf = conf
        self._axelera_available = _AXELERA_SDK_AVAILABLE

        # ── Build ONNX fallback engine for YOLO + MediaPipe + gallery embs ─
        # The ONNX engine handles everything gracefully even without .onnx files.
        self._fallback = OnnxInferenceEngine(
            yolo_model_path=yolo_model_path,
            tracker_cfg=tracker_cfg,
            onnx_dir=onnx_dir,
            embedding_dir=embedding_dir,
            conf=conf,
        )

        # ── Build ONNX Runtime sessions for embedding models ───────────────
        # These sessions are wrapped by _AxeleraModel which will prefer .axm.
        ort_providers = ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort
            _reid_onnx = onnx_dir / "osnet_x1_0.onnx"
            _waiter_onnx = onnx_dir / "resnet50_waiter.onnx"
            _plate_onnx = onnx_dir / "resnet50_plate.onnx"

            _reid_sess = ort.InferenceSession(str(_reid_onnx), providers=ort_providers) if _reid_onnx.exists() else None
            _waiter_sess = ort.InferenceSession(str(_waiter_onnx), providers=ort_providers) if _waiter_onnx.exists() else None
            _plate_sess = ort.InferenceSession(str(_plate_onnx), providers=ort_providers) if _plate_onnx.exists() else None
        except Exception:
            _reid_sess = _waiter_sess = _plate_sess = None

        # ── Wrap ONNX sessions with Axelera model loader ───────────────────
        self._reid_model = _AxeleraModel(
            self._axelera_dir / "osnet_x1_0.axm",
            fallback_onnx_session=_reid_sess,
        )
        self._waiter_model = _AxeleraModel(
            self._axelera_dir / "resnet50_waiter.axm",
            fallback_onnx_session=_waiter_sess,
        )
        self._plate_model = _AxeleraModel(
            self._axelera_dir / "resnet50_plate.axm",
            fallback_onnx_session=_plate_sess,
        )

        # Print status summary
        _status = {
            "YOLO": "AIPU" if (_AXELERA_SDK_AVAILABLE and (axelera_dir / "yolov8n.axm").exists()) else "ONNX/PT",
            "OSNet": "AIPU" if self._reid_model.available else "ONNX/CPU",
            "ResNet50 Waiter": "AIPU" if self._waiter_model.available else "ONNX/CPU",
            "ResNet50 Plate": "AIPU" if self._plate_model.available else "ONNX/CPU",
            "MediaPipe Pose": "CPU",
            "MediaPipe Hand": "CPU",
        }
        print("[AxeleraEngine] Model routing:")
        for k, v in _status.items():
            print(f"  {k:25s} -> {v}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warmup(self, n_frames: int = 10) -> None:
        self._fallback.warmup(n_frames)
        # Warm up AIPU models with dummy inputs
        dummy_reid = np.zeros((1, 3, 256, 128), dtype=np.float32)
        dummy_cls = np.zeros((1, 3, 224, 224), dtype=np.float32)
        for _ in range(min(3, n_frames)):
            self._reid_model.run(dummy_reid)
            self._waiter_model.run(dummy_cls)
            self._plate_model.run(dummy_cls)
        print(f"[AxeleraEngine] Warm-up complete ({n_frames} frames)")

    def release(self) -> None:
        self._fallback.release()
        if _AXELERA_SDK_AVAILABLE:
            try:
                _axrt.cleanup()
            except Exception:
                pass
        print("[AxeleraEngine] Released")

    # ------------------------------------------------------------------
    # Detection — delegate to fallback (YOLO + BoT-SORT via Ultralytics)
    # ------------------------------------------------------------------

    def detect_persons_raw(self, frame: np.ndarray, conf: float = 0.25) -> List[Dict]:
        return self._fallback.detect_persons_raw(frame, conf)

    def track_persons(self, frame: np.ndarray, conf: float = 0.25) -> Tuple[Any, List[float], List[float]]:
        return self._fallback.track_persons(frame, conf)

    def detect_food(self, frame: np.ndarray, food_classes=None, conf: float = 0.20) -> List[Dict]:
        return self._fallback.detect_food(frame, food_classes, conf)

    # ------------------------------------------------------------------
    # Re-ID / Embedding — AIPU preferred, ONNX fallback
    # ------------------------------------------------------------------

    def extract_reid_embedding(self, person_crop_bgr: np.ndarray) -> np.ndarray:
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return np.zeros((1, 512), dtype=np.float32)
        x = _preprocess(person_crop_bgr, (256, 128))
        emb = self._reid_model.run(x)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norm + 1e-8)

    def extract_waiter_embedding(self, person_crop_bgr: np.ndarray) -> np.ndarray:
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return np.zeros((1, 2048), dtype=np.float32)
        h = person_crop_bgr.shape[0]
        top_crop = person_crop_bgr[:int(h * 0.4)]
        x = _preprocess(top_crop, (224, 224))
        emb = self._waiter_model.run(x)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norm + 1e-8)

    def get_waiter_gallery_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._fallback.get_waiter_gallery_embeddings()

    # ------------------------------------------------------------------
    # Pose / hands — CPU (MediaPipe on RK3588)
    # ------------------------------------------------------------------

    def detect_pose(self, crop_bgr: np.ndarray) -> Dict:
        return self._fallback.detect_pose(crop_bgr)

    def detect_hands(self, crop_bgr: np.ndarray) -> Dict:
        return self._fallback.detect_hands(crop_bgr)

    # ------------------------------------------------------------------
    # Plate classifier — AIPU preferred
    # ------------------------------------------------------------------

    def classify_plate(self, roi_crop_bgr: np.ndarray) -> int:
        if roi_crop_bgr is None or roi_crop_bgr.size == 0:
            return 0
        if roi_crop_bgr.shape[0] < 20 or roi_crop_bgr.shape[1] < 20:
            return 0
        x = _preprocess(roi_crop_bgr, (224, 224))
        logits = self._plate_model.run(x)[0]
        probs = _softmax(logits)
        top5 = set(np.argsort(probs)[-5:].tolist())
        return len(_DISHWARE_INDICES.intersection(top5))

    # ------------------------------------------------------------------
    # Availability / metadata
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return "axelera"

    def is_cuda_available(self) -> bool:
        return False  # Metis board has no NVIDIA GPU

    def is_axelera_available(self) -> bool:
        return _AXELERA_SDK_AVAILABLE and any(
            (self._axelera_dir / f).exists()
            for f in ["yolov8n.axm", "osnet_x1_0.axm"]
        )
