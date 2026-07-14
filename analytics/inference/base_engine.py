"""
analytics/inference/base_engine.py
====================================
Abstract base class that defines the hardware-agnostic inference API.

Every concrete backend (CUDA, CPU, ONNX, Axelera) must implement ALL
methods defined here.  The rest of the project (pipeline.py, PersonTracker,
ServingDetector, PlateDetector) must ONLY call methods on this interface —
never import torch, YOLO, MediaPipe, or OSNet directly.

Detection result schemas
------------------------
``detect_persons`` returns a list of dicts:
    {
        "track_id":  int,           # BoT-SORT track id (0 on raw detection, set by tracker)
        "bbox":      (x1,y1,x2,y2),# pixel coordinates (ints)
        "conf":      float,         # detection confidence 0–1
        "class_id":  int,           # 0 = person
    }

``detect_food`` returns a list of dicts:
    {
        "class":     str,           # COCO class name
        "bbox":      (x1,y1,x2,y2),
        "confidence": float,
    }

``extract_reid_embedding`` returns:
    numpy.ndarray of shape (1, D) — L2-normalised float32 embedding.
    D is model-dependent (512 for OSNet x1.0, 2048 for ResNet50).

``extract_waiter_embedding`` returns:
    numpy.ndarray of shape (1, 2048) — L2-normalised ResNet50 embedding.

``classify_waiter_similarity`` returns:
    float — cosine similarity to the waiter gallery embedding.

``detect_pose`` returns:
    dict with keys:
        "landmarks": list[dict]  — each landmark has keys:
            "x": float (normalised 0–1 within crop),
            "y": float (normalised 0–1 within crop),
            "z": float,
            "visibility": float
        Index mapping matches MediaPipe PoseLandmarker:
            11: left_shoulder,  12: right_shoulder
            13: left_elbow,     14: right_elbow
            15: left_wrist,     16: right_wrist
        Returns {"landmarks": []} when no person detected.

``detect_hands`` returns:
    dict with key "landmarks" analogous to ``detect_pose``.
    MediaPipe Hand index 0 = wrist.
    Returns {"landmarks": []} when no hands detected.

``classify_plate`` returns:
    int — count of top-5 ImageNet predictions that fall inside the
    DISHWARE_INDICES set (same logic as original PlateDetector).

Availability queries
--------------------
``is_cuda_available``   → bool
``is_axelera_available`` → bool
``backend_name``        → str  ("cuda" | "cpu" | "onnx" | "axelera")
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


class BaseInferenceEngine(ABC):
    """Abstract inference engine.  All hardware backends extend this class."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def warmup(self, n_frames: int = 10) -> None:
        """
        Run N dummy inference passes to prime the runtime / JIT / AIPU.
        Call once after initialisation, before the main processing loop.
        """

    @abstractmethod
    def release(self) -> None:
        """Release any GPU / accelerator resources (called on shutdown)."""

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @abstractmethod
    def detect_persons_raw(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Run person detection on a full BGR frame.

        Returns a list of raw detection dicts (no tracking IDs yet).
        BoT-SORT tracking is handled separately by PersonTracker.
        """

    @abstractmethod
    def track_persons(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
    ) -> Tuple[Any, List[float], List[float]]:
        """
        Run YOLO + BoT-SORT tracking in one call.

        Returns:
            results  — raw ultralytics Results object (or equivalent)
            yolo_latencies_sec — list of per-call inference times in seconds
            tracking_latencies_sec — list of tracking overhead times
        The PersonTracker consumes the raw results object.
        """

    @abstractmethod
    def detect_food(
        self,
        frame: np.ndarray,
        food_classes: Optional[List[int]] = None,
        conf: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        Detect food / dining items in a full BGR frame.

        ``food_classes`` is the list of COCO class IDs for food/dishware.
        If None, the engine uses the default set from serving_detector.py.
        """

    # ------------------------------------------------------------------
    # Re-ID / Embedding
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_reid_embedding(
        self,
        person_crop_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Extract an L2-normalised Re-ID embedding from a person crop.

        Uses OSNet x1.0 (or ONNX equivalent).
        Input: BGR crop of any size (will be resized internally).
        Output: numpy float32 array, shape (1, 512).
        """

    @abstractmethod
    def extract_waiter_embedding(
        self,
        person_crop_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Extract an L2-normalised embedding for waiter classification.

        Uses ResNet50 top-40% crop (same as original FeatureExtractor).
        Output: numpy float32 array, shape (1, 2048).
        """

    @abstractmethod
    def get_waiter_gallery_embeddings(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (waiter_emb, server_emb) numpy arrays loaded from disk,
        or (None, None) if the gallery files do not exist.
        """

    # ------------------------------------------------------------------
    # Pose / hands
    # ------------------------------------------------------------------

    @abstractmethod
    def detect_pose(
        self,
        crop_bgr: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run pose landmark detection on a person crop.

        Returns dict:
            {
                "landmarks": [{"x": float, "y": float, "z": float,
                               "visibility": float}, ...],  # 33 landmarks
            }
        Empty "landmarks" list if no person detected.
        """

    @abstractmethod
    def detect_hands(
        self,
        crop_bgr: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run hand landmark detection on a person crop.

        Returns dict:
            {
                "landmarks": [{"x": float, "y": float, "z": float}, ...],  # 21 landmarks
            }
        Empty list if no hands detected.
        """

    # ------------------------------------------------------------------
    # Plate / cleanliness
    # ------------------------------------------------------------------

    @abstractmethod
    def classify_plate(
        self,
        roi_crop_bgr: np.ndarray,
    ) -> int:
        """
        Detect dirty dishware in a table ROI crop.

        Returns the count of top-5 ResNet50 predictions that match the
        DISHWARE_INDICES set (same as original PlateDetector logic).
        Returns 0 if no dishware detected.
        """

    # ------------------------------------------------------------------
    # Availability / metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier: "cuda", "cpu", "onnx", "axelera"."""

    @abstractmethod
    def is_cuda_available(self) -> bool:
        """Return True if a CUDA-capable GPU is present and initialised."""

    @abstractmethod
    def is_axelera_available(self) -> bool:
        """Return True if the Axelera Metis AIPU is present and accessible."""

    # ------------------------------------------------------------------
    # Convenience helper (shared implementation)
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable engine summary string."""
        return (
            f"InferenceEngine[{self.backend_name}] "
            f"cuda={self.is_cuda_available()} "
            f"axelera={self.is_axelera_available()}"
        )
