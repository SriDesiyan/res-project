"""
analytics/inference/onnx_engine.py
=====================================
OnnxInferenceEngine — runs YOLO, OSNet, and ResNet50 via ONNX Runtime.

This engine is the intermediate step between pure PyTorch and the Axelera
AIPU.  It allows validation on a development machine (Windows/Linux without
the Voyager SDK) before burning models to the AIPU.

Prerequisites
-------------
  pip install onnxruntime          # CPU
  pip install onnxruntime-gpu      # GPU (if CUDA available)

ONNX model paths (produced by scripts/export_onnx.py):
  weights/onnx/yolov8n.onnx
  weights/onnx/osnet_x1_0.onnx
  weights/onnx/resnet50_waiter.onnx
  weights/onnx/resnet50_plate.onnx

MediaPipe is still used for pose/hand detection in this engine because
MediaPipe .task files run on CPU natively (no CUDA needed).  Phase 5 will
replace MediaPipe with a YOLOv8-Pose ONNX model when Axelera support is ready.

Pose / hand detection falls back to the CudaEngine's MediaPipe implementation
so no business logic changes are needed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from analytics.inference.base_engine import BaseInferenceEngine

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    print("[OnnxEngine] WARNING: onnxruntime not installed — OnnxInferenceEngine unavailable")

try:
    from ultralytics import YOLO as _UltralyticsYOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _UltralyticsYOLO = None
    _ULTRALYTICS_AVAILABLE = False

try:
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

# ---------------------------------------------------------------------------
# DISHWARE / food constants (same as cuda_engine)
# ---------------------------------------------------------------------------
_DISHWARE_INDICES = {504, 923, 968, 868, 809, 659, 440, 737, 898, 907}
_DEFAULT_FOOD_CLASSES = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                          49, 50, 51, 52, 53, 54, 55]

# ImageNet normalisation (same as torchvision ResNet defaults)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_imagenet(bgr_crop: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize → BGR→RGB → float32 / 255 → normalise → NCHW.
    Returns (1, 3, H, W) float32 array.
    """
    h, w = size
    rgb = cv2.cvtColor(cv2.resize(bgr_crop, (w, h)), cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return x.transpose(2, 0, 1)[np.newaxis]   # NCHW


class OnnxInferenceEngine(BaseInferenceEngine):
    """
    ONNX Runtime engine.  YOLOv8 tracking still uses Ultralytics (which
    internally calls its own ONNX/TensorRT logic).  Embedding models
    (OSNet, ResNet50) are run via onnxruntime for maximum portability.

    Parameters
    ----------
    yolo_model_path : Path
        Path to yolov8n.pt (or .onnx — Ultralytics handles both).
    tracker_cfg : str
        Path to BoT-SORT YAML.
    onnx_dir : Path
        Directory containing pre-exported ONNX models.
    embedding_dir : Path
        Directory containing MediaPipe .task files and waiter embeddings.
    conf : float
        Default YOLO confidence threshold.
    providers : list[str]
        ONNX Runtime execution providers, e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"].
    """

    def __init__(
        self,
        yolo_model_path: Path,
        tracker_cfg: str,
        onnx_dir: Path,
        embedding_dir: Path,
        conf: float = 0.25,
        providers: Optional[List[str]] = None,
    ) -> None:
        if not _ORT_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed. "
                "Run: pip install onnxruntime  (or onnxruntime-gpu)"
            )

        self._tracker_cfg = tracker_cfg
        self._conf = conf
        self._onnx_dir = Path(onnx_dir)
        self._embedding_dir = Path(embedding_dir)

        # Default providers: GPU first, then CPU
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # ── YOLO (Ultralytics handles ONNX internally) ─────────────────────
        if _ULTRALYTICS_AVAILABLE:
            _onnx_yolo = self._onnx_dir / "yolov8n.onnx"
            if _onnx_yolo.exists():
                self._yolo = _UltralyticsYOLO(str(_onnx_yolo))
                print(f"[OnnxEngine] Loaded YOLO from ONNX: {_onnx_yolo}")
            else:
                self._yolo = _UltralyticsYOLO(str(yolo_model_path))
                print(f"[OnnxEngine] Loaded YOLO from PT (ONNX not found): {yolo_model_path}")
        else:
            self._yolo = None

        # ── OSNet ONNX session ─────────────────────────────────────────────
        _osnet_path = self._onnx_dir / "osnet_x1_0.onnx"
        if _osnet_path.exists():
            self._reid_session = ort.InferenceSession(
                str(_osnet_path), providers=providers
            )
            self._reid_input_name = self._reid_session.get_inputs()[0].name
            print(f"[OnnxEngine] OSNet ONNX session: {_osnet_path}")
        else:
            self._reid_session = None
            print(f"[OnnxEngine] WARNING: OSNet ONNX not found at {_osnet_path}")

        # ── ResNet50 waiter ONNX session ───────────────────────────────────
        _waiter_path = self._onnx_dir / "resnet50_waiter.onnx"
        if _waiter_path.exists():
            self._waiter_session = ort.InferenceSession(
                str(_waiter_path), providers=providers
            )
            self._waiter_input_name = self._waiter_session.get_inputs()[0].name
            print(f"[OnnxEngine] ResNet50 waiter ONNX session: {_waiter_path}")
        else:
            self._waiter_session = None
            print(f"[OnnxEngine] WARNING: ResNet50 waiter ONNX not found at {_waiter_path}")

        # ── ResNet50 plate ONNX session ────────────────────────────────────
        _plate_path = self._onnx_dir / "resnet50_plate.onnx"
        if _plate_path.exists():
            self._plate_session = ort.InferenceSession(
                str(_plate_path), providers=providers
            )
            self._plate_input_name = self._plate_session.get_inputs()[0].name
            print(f"[OnnxEngine] ResNet50 plate ONNX session: {_plate_path}")
        else:
            self._plate_session = None
            print(f"[OnnxEngine] WARNING: ResNet50 plate ONNX not found at {_plate_path}")

        # ── Waiter gallery embeddings (numpy) ─────────────────────────────
        self._waiter_emb: Optional[np.ndarray] = None
        self._server_emb: Optional[np.ndarray] = None
        try:
            import torch as _torch
            _wp = self._embedding_dir / "waiter_average_embedding.pt"
            _sp = self._embedding_dir / "server_average_embedding.pt"
            if _wp.exists():
                _t = _torch.load(str(_wp), map_location="cpu")
                self._waiter_emb = _torch.nn.functional.normalize(_t, p=2, dim=1).numpy()
            if _sp.exists():
                _t = _torch.load(str(_sp), map_location="cpu")
                self._server_emb = _torch.nn.functional.normalize(_t, p=2, dim=1).numpy()
        except Exception as exc:
            print(f"[OnnxEngine] WARNING: Could not load gallery embeddings: {exc}")

        # ── MediaPipe (CPU, used for pose/hands) ──────────────────────────
        self._mp_hands = None
        self._mp_pose = None
        if _MEDIAPIPE_AVAILABLE:
            try:
                _hand_path = str(self._embedding_dir / "hand_landmarker.task")
                _pose_path = str(self._embedding_dir / "pose_landmarker.task")
                _ho = _mp_vision.HandLandmarkerOptions(
                    base_options=_mp_python.BaseOptions(model_asset_path=_hand_path),
                    running_mode=_mp_vision.RunningMode.IMAGE,
                    num_hands=2,
                )
                self._mp_hands = _mp_vision.HandLandmarker.create_from_options(_ho)
                _po = _mp_vision.PoseLandmarkerOptions(
                    base_options=_mp_python.BaseOptions(model_asset_path=_pose_path),
                    running_mode=_mp_vision.RunningMode.IMAGE,
                    output_segmentation_masks=False,
                )
                self._mp_pose = _mp_vision.PoseLandmarker.create_from_options(_po)
                print("[OnnxEngine] MediaPipe Hand + Pose initialised")
            except Exception as exc:
                print(f"[OnnxEngine] WARNING: MediaPipe init failed: {exc}")

        print("[OnnxEngine] Initialised")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warmup(self, n_frames: int = 10) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        if self._yolo:
            for _ in range(n_frames):
                self._yolo(dummy, verbose=False)
        # Warm up ONNX sessions
        dummy_reid = np.zeros((1, 3, 256, 128), dtype=np.float32)
        dummy_cls = np.zeros((1, 3, 224, 224), dtype=np.float32)
        if self._reid_session:
            self._reid_session.run(None, {self._reid_input_name: dummy_reid})
        if self._waiter_session:
            self._waiter_session.run(None, {self._waiter_input_name: dummy_cls})
        if self._plate_session:
            self._plate_session.run(None, {self._plate_input_name: dummy_cls})
        print(f"[OnnxEngine] Warm-up complete ({n_frames} frames)")

    def release(self) -> None:
        self._reid_session = None
        self._waiter_session = None
        self._plate_session = None
        print("[OnnxEngine] Released")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_persons_raw(self, frame: np.ndarray, conf: float = 0.25) -> List[Dict]:
        if not self._yolo:
            return []
        results = self._yolo(frame, classes=[0], conf=conf, verbose=False)
        out = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, c in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                out.append({"track_id": -1, "bbox": (x1, y1, x2, y2),
                             "conf": float(c), "class_id": 0})
        return out

    def track_persons(self, frame: np.ndarray, conf: float = 0.25) -> Tuple[Any, List[float], List[float]]:
        if not self._yolo:
            return None, [0.0], [0.0]
        t0 = time.time()
        results = self._yolo.track(
            frame, classes=[0], conf=conf, persist=True,
            tracker=self._tracker_cfg, verbose=False,
        )
        t1 = time.time()
        yolo_inf = 0.0
        if results and hasattr(results[0], "speed"):
            yolo_inf = results[0].speed.get("inference", 0.0) / 1000.0
        return results, [yolo_inf], [max(0.0, (t1 - t0) - yolo_inf)]

    def detect_food(self, frame: np.ndarray, food_classes=None, conf: float = 0.20) -> List[Dict]:
        if not self._yolo:
            return []
        classes = food_classes or _DEFAULT_FOOD_CLASSES
        results = self._yolo(frame, classes=classes, conf=conf, verbose=False)
        out = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cids = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, cid, c in zip(boxes, cids, confs):
                out.append({"class": self._yolo.names[int(cid)],
                             "bbox": tuple(map(float, box)), "confidence": float(c)})
        return out

    # ------------------------------------------------------------------
    # Re-ID / Embedding
    # ------------------------------------------------------------------

    def extract_reid_embedding(self, person_crop_bgr: np.ndarray) -> np.ndarray:
        """OSNet via ONNX Runtime — L2-normalised (1, 512)."""
        if self._reid_session is None or person_crop_bgr is None or person_crop_bgr.size == 0:
            return np.zeros((1, 512), dtype=np.float32)
        x = _preprocess_imagenet(person_crop_bgr, (256, 128))
        emb = self._reid_session.run(None, {self._reid_input_name: x})[0]
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norm + 1e-8)

    def extract_waiter_embedding(self, person_crop_bgr: np.ndarray) -> np.ndarray:
        """ResNet50 waiter via ONNX Runtime — L2-normalised (1, 2048)."""
        if self._waiter_session is None or person_crop_bgr is None or person_crop_bgr.size == 0:
            return np.zeros((1, 2048), dtype=np.float32)
        h = person_crop_bgr.shape[0]
        top_crop = person_crop_bgr[:int(h * 0.4)]
        x = _preprocess_imagenet(top_crop, (224, 224))
        emb = self._waiter_session.run(None, {self._waiter_input_name: x})[0]
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norm + 1e-8)

    def get_waiter_gallery_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._waiter_emb, self._server_emb

    # ------------------------------------------------------------------
    # Pose / hands (MediaPipe — CPU)
    # ------------------------------------------------------------------

    def _mp_landmarks_to_list(self, landmarks) -> List[Dict]:
        return [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z),
                  "visibility": float(getattr(lm, "visibility", 1.0))}
                for lm in landmarks]

    def detect_pose(self, crop_bgr: np.ndarray) -> Dict:
        if self._mp_pose is None or crop_bgr is None or crop_bgr.size == 0:
            return {"landmarks": []}
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)  # type: ignore[name-defined]
            res = self._mp_pose.detect(mp_img)
            if res.pose_landmarks:
                return {"landmarks": self._mp_landmarks_to_list(res.pose_landmarks[0])}
        except Exception as exc:
            print(f"[OnnxEngine] detect_pose error: {exc}")
        return {"landmarks": []}

    def detect_hands(self, crop_bgr: np.ndarray) -> Dict:
        if self._mp_hands is None or crop_bgr is None or crop_bgr.size == 0:
            return {"landmarks": []}
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)  # type: ignore[name-defined]
            res = self._mp_hands.detect(mp_img)
            if res.hand_landmarks:
                return {"landmarks": self._mp_landmarks_to_list(res.hand_landmarks[0])}
        except Exception as exc:
            print(f"[OnnxEngine] detect_hands error: {exc}")
        return {"landmarks": []}

    # ------------------------------------------------------------------
    # Plate classifier
    # ------------------------------------------------------------------

    def classify_plate(self, roi_crop_bgr: np.ndarray) -> int:
        if self._plate_session is None or roi_crop_bgr is None or roi_crop_bgr.size == 0:
            return 0
        if roi_crop_bgr.shape[0] < 20 or roi_crop_bgr.shape[1] < 20:
            return 0
        x = _preprocess_imagenet(roi_crop_bgr, (224, 224))
        logits = self._plate_session.run(None, {self._plate_input_name: x})[0][0]
        # Softmax
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        top5 = set(np.argsort(probs)[-5:].tolist())
        return len(_DISHWARE_INDICES.intersection(top5))

    # ------------------------------------------------------------------
    # Availability / metadata
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return "onnx"

    def is_cuda_available(self) -> bool:
        if not _ORT_AVAILABLE:
            return False
        return "CUDAExecutionProvider" in ort.get_available_providers()

    def is_axelera_available(self) -> bool:
        return False
