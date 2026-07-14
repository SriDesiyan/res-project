"""
analytics/inference/cuda_engine.py
=====================================
CudaInferenceEngine — wraps the original YOLO/BoT-SORT/OSNet/ResNet50/MediaPipe
pipeline running on NVIDIA CUDA exactly as it did before the migration.

This engine is the **reference implementation**.  All other engines must
produce outputs compatible with this engine's outputs.

It is also the safest fallback: if Axelera or CPU-ONNX fail for any reason,
the pipeline can switch back to this engine without code changes.

No business logic lives here — only model loading and inference calls.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from ultralytics import YOLO

from analytics.inference.base_engine import BaseInferenceEngine

# ---------------------------------------------------------------------------
# MediaPipe imports — optional; gracefully degrade if not installed.
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

# ---------------------------------------------------------------------------
# OSNet import — local module
# ---------------------------------------------------------------------------
from analytics.tracking.osnet import osnet_x1_0


# ---------------------------------------------------------------------------
# ResNet50 feature extractor (top-40% crop, 2048-dim)
# ---------------------------------------------------------------------------

class _FeatureExtractor(nn.Module):
    """ResNet50 feature extractor — produces 2048-dim embedding."""

    def __init__(self) -> None:
        super().__init__()
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        return x.view(x.size(0), -1)


# ---------------------------------------------------------------------------
# DISHWARE ImageNet indices (same as original PlateDetector)
# ---------------------------------------------------------------------------
_DISHWARE_INDICES = {504, 923, 968, 868, 809, 659, 440, 737, 898, 907}

# ---------------------------------------------------------------------------
# COCO food class IDs (same as original serve detector)
# ---------------------------------------------------------------------------
_DEFAULT_FOOD_CLASSES = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                          49, 50, 51, 52, 53, 54, 55]


class CudaInferenceEngine(BaseInferenceEngine):
    """
    Reference inference engine using the original CUDA/PyTorch stack.

    Parameters
    ----------
    device : torch.device
        The target compute device (cuda / mps / cpu).
    yolo_model_path : str | Path
        Path to the YOLOv8n .pt weights file.
    tracker_cfg : str
        Path to the BoT-SORT tracker YAML.
    embedding_dir : str | Path
        Directory containing OSNet weights and MediaPipe task files.
    conf : float
        Default YOLO detection confidence threshold.
    """

    def __init__(
        self,
        device: torch.device,
        yolo_model_path: str | Path,
        tracker_cfg: str,
        embedding_dir: str | Path,
        conf: float = 0.25,
    ) -> None:
        self.device = device
        self.conf = conf
        self._embedding_dir = Path(embedding_dir)
        self._yolo_model_path = Path(yolo_model_path)
        self._tracker_cfg = tracker_cfg

        # ── YOLO + BoT-SORT ────────────────────────────────────────────────
        self.yolo = YOLO(str(self._yolo_model_path))
        self.yolo.to(device)

        # ── ResNet50 waiter classifier ─────────────────────────────────────
        self._extractor = _FeatureExtractor().to(device)
        self._extractor.eval()
        self._resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        # ── ResNet50 plate classifier ──────────────────────────────────────
        # NOTE: Same architecture; separate instance to allow different devices later.
        _plate_weights = ResNet50_Weights.DEFAULT
        self._plate_model = resnet50(weights=_plate_weights).to(device)
        self._plate_model.eval()
        self._plate_preprocess = _plate_weights.transforms()

        # ── OSNet Re-ID ────────────────────────────────────────────────────
        self._reid_extractor = osnet_x1_0(num_classes=1000, pretrained=False).to(device)
        _reid_path = self._embedding_dir / "osnet_x1_0_msmt17.pth"
        if _reid_path.exists():
            state_dict = torch.load(str(_reid_path), map_location=device)
            model_dict = self._reid_extractor.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k[7:] if k.startswith("module.") else k
                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
            model_dict.update(new_state_dict)
            self._reid_extractor.load_state_dict(model_dict)
            print(f"[CudaEngine] Loaded OSNet Re-ID weights from {_reid_path}")
        else:
            print(f"[CudaEngine] WARNING: OSNet weights not found at {_reid_path}")
        self._reid_extractor.eval()

        self._reid_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        # ── Waiter gallery embeddings ──────────────────────────────────────
        self._waiter_emb: Optional[torch.Tensor] = None
        self._server_emb: Optional[torch.Tensor] = None
        _waiter_path = self._embedding_dir / "waiter_average_embedding.pt"
        _server_path = self._embedding_dir / "server_average_embedding.pt"
        if _waiter_path.exists():
            self._waiter_emb = torch.load(str(_waiter_path), map_location=device)
            self._waiter_emb = torch.nn.functional.normalize(self._waiter_emb, p=2, dim=1)
        if _server_path.exists():
            self._server_emb = torch.load(str(_server_path), map_location=device)
            self._server_emb = torch.nn.functional.normalize(self._server_emb, p=2, dim=1)

        # ── MediaPipe ─────────────────────────────────────────────────────
        self._mp_hands = None
        self._mp_pose = None
        if _MEDIAPIPE_AVAILABLE:
            _hand_path = str(self._embedding_dir / "hand_landmarker.task")
            _pose_path = str(self._embedding_dir / "pose_landmarker.task")
            try:
                _hand_opts = mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=_hand_path),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=2,
                )
                self._mp_hands = mp_vision.HandLandmarker.create_from_options(_hand_opts)

                _pose_opts = mp_vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=_pose_path),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    output_segmentation_masks=False,
                )
                self._mp_pose = mp_vision.PoseLandmarker.create_from_options(_pose_opts)
                print("[CudaEngine] MediaPipe Hand + Pose landmarkers initialised")
            except Exception as exc:
                print(f"[CudaEngine] WARNING: MediaPipe init failed: {exc}")
        else:
            print("[CudaEngine] WARNING: MediaPipe not installed — pose/hand detection unavailable")

        print(f"[CudaEngine] Initialised on device={device}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warmup(self, n_frames: int = 10) -> None:
        """Run dummy YOLO passes to prime CUDA kernel compilation."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n_frames):
            self.yolo(dummy, verbose=False)
        print(f"[CudaEngine] Warm-up complete ({n_frames} frames)")

    def release(self) -> None:
        """Release GPU resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[CudaEngine] Released")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_persons_raw(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Raw YOLO detect without tracking (returns no track IDs)."""
        results = self.yolo(frame, classes=[0], conf=conf, verbose=False)
        detections: List[Dict[str, Any]] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, c in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "track_id": -1,
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(c),
                    "class_id": 0,
                })
        return detections

    def track_persons(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
    ) -> Tuple[Any, List[float], List[float]]:
        """YOLO + BoT-SORT tracking.  Returns raw Results for PersonTracker."""
        t0 = time.time()
        results = self.yolo.track(
            frame,
            classes=[0],
            conf=conf,
            persist=True,
            tracker=self._tracker_cfg,
            verbose=False,
            device=self.device,
        )
        t1 = time.time()

        yolo_inf_sec = 0.0
        if results and len(results) > 0 and hasattr(results[0], "speed"):
            yolo_inf_sec = results[0].speed.get("inference", 0.0) / 1000.0

        tracking_overhead = max(0.0, (t1 - t0) - yolo_inf_sec)
        return results, [yolo_inf_sec], [tracking_overhead]

    def detect_food(
        self,
        frame: np.ndarray,
        food_classes: Optional[List[int]] = None,
        conf: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """Detect food/dining items using YOLOv8 on COCO classes."""
        classes = food_classes if food_classes is not None else _DEFAULT_FOOD_CLASSES
        results = self.yolo(frame, classes=classes, conf=conf, verbose=False)
        detections: List[Dict[str, Any]] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, cid, c in zip(boxes, cls_ids, confs):
                detections.append({
                    "class": self.yolo.names[int(cid)],
                    "bbox": tuple(map(float, box)),
                    "confidence": float(c),
                })
        return detections

    # ------------------------------------------------------------------
    # Re-ID / Embedding
    # ------------------------------------------------------------------

    def extract_reid_embedding(self, person_crop_bgr: np.ndarray) -> np.ndarray:
        """OSNet x1.0 Re-ID embedding — shape (1, 512)."""
        img_pil = Image.fromarray(cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self._reid_transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._reid_extractor(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

    def extract_waiter_embedding(self, person_crop_bgr: np.ndarray) -> np.ndarray:
        """ResNet50 2048-dim embedding on top-40% crop."""
        img_pil = Image.fromarray(cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB))
        h_total = img_pil.height
        top_img = img_pil.crop((0, 0, img_pil.width, int(h_total * 0.4)))
        tensor = self._resnet_transform(top_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._extractor(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

    def get_waiter_gallery_embeddings(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (waiter_emb_np, server_emb_np) or (None, None)."""
        w = self._waiter_emb.cpu().numpy() if self._waiter_emb is not None else None
        s = self._server_emb.cpu().numpy() if self._server_emb is not None else None
        return w, s

    # ------------------------------------------------------------------
    # Pose / hands
    # ------------------------------------------------------------------

    def _mp_landmarks_to_list(self, landmarks) -> List[Dict[str, float]]:
        """Convert MediaPipe NormalizedLandmark list to standard dicts."""
        result = []
        for lm in landmarks:
            result.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(getattr(lm, "visibility", 1.0)),
            })
        return result

    def detect_pose(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """MediaPipe PoseLandmarker on a BGR crop."""
        if self._mp_pose is None or crop_bgr is None or crop_bgr.size == 0:
            return {"landmarks": []}
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # type: ignore[name-defined]
            result = self._mp_pose.detect(mp_img)
            if result.pose_landmarks:
                return {"landmarks": self._mp_landmarks_to_list(result.pose_landmarks[0])}
        except Exception as exc:
            print(f"[CudaEngine] detect_pose error: {exc}")
        return {"landmarks": []}

    def detect_hands(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """MediaPipe HandLandmarker on a BGR crop."""
        if self._mp_hands is None or crop_bgr is None or crop_bgr.size == 0:
            return {"landmarks": []}
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # type: ignore[name-defined]
            result = self._mp_hands.detect(mp_img)
            if result.hand_landmarks:
                return {"landmarks": self._mp_landmarks_to_list(result.hand_landmarks[0])}
        except Exception as exc:
            print(f"[CudaEngine] detect_hands error: {exc}")
        return {"landmarks": []}

    # ------------------------------------------------------------------
    # Plate classifier
    # ------------------------------------------------------------------

    def classify_plate(self, roi_crop_bgr: np.ndarray) -> int:
        """ResNet50 dirty-dishware count (same logic as original PlateDetector)."""
        if roi_crop_bgr is None or roi_crop_bgr.size == 0:
            return 0
        if roi_crop_bgr.shape[0] < 20 or roi_crop_bgr.shape[1] < 20:
            return 0
        try:
            rgb = cv2.cvtColor(roi_crop_bgr, cv2.COLOR_BGR2RGB)
            img_t = self._plate_preprocess(
                torch.from_numpy(rgb).permute(2, 0, 1)
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self._plate_model(img_t)
            probs = torch.nn.functional.softmax(out[0], dim=0)
            _, top5_idx = torch.topk(probs, 5)
            matches = _DISHWARE_INDICES.intersection(set(top5_idx.cpu().numpy()))
            return len(matches)
        except RuntimeError as exc:
            if "cuda" in str(exc).lower() or "device" in str(exc).lower():
                print(f"[CudaEngine] classify_plate CUDA error: {exc}. Falling back to CPU.")
                self._plate_model = self._plate_model.cpu()
                self.device = torch.device("cpu")
                return self.classify_plate(roi_crop_bgr)
            raise

    # ------------------------------------------------------------------
    # Availability / metadata
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return "cuda" if torch.cuda.is_available() and str(self.device) != "cpu" else "cpu"

    def is_cuda_available(self) -> bool:
        return torch.cuda.is_available()

    def is_axelera_available(self) -> bool:
        return False
