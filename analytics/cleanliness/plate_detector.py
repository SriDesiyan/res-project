"""
analytics/cleanliness/plate_detector.py
=========================================
Plate / dirty-dishware detector.

MIGRATION NOTE (Edge AI):
    No longer imports torch or ResNet50 directly.
    All model inference is delegated to the ``BaseInferenceEngine`` interface
    via ``engine.classify_plate(roi_crop_bgr)``.
    The dirty-object business logic (DISHWARE_INDICES set, top-5 matching,
    count threshold) is UNCHANGED and still lives in the engine implementation.
"""
from __future__ import annotations

import numpy as np

from analytics.inference.base_engine import BaseInferenceEngine


class PlateDetector:
    """
    Detects dirty dishware in a table ROI crop.

    Parameters
    ----------
    engine : BaseInferenceEngine
        The active inference backend.  All ResNet50 calls go through this.
    device : str | None
        Deprecated parameter — ignored (kept for backward-compatibility with
        call sites that pass ``device=device``).
    """

    # ImageNet indices for dishware (kept here for documentation; logic lives in engine)
    # 504: coffee mug, 923: plate, 968: cup, 868: tray, 809: soup bowl,
    # 659: mixing bowl, 440: beer bottle, 737: pop bottle,
    # 898: water bottle, 907: wine bottle
    DISHWARE_INDICES = {504, 923, 968, 868, 809, 659, 440, 737, 898, 907}

    def __init__(self, engine: BaseInferenceEngine, device=None) -> None:
        self.engine = engine
        # ``device`` ignored — present for drop-in backward compat.

    def detect_dirty_objects(self, frame: np.ndarray, polygon_pts: np.ndarray) -> int:
        """
        Return count of dishware-class matches in the table ROI crop.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR video frame.
        polygon_pts : np.ndarray
            Nx2 array of polygon vertices (dtype int32).

        Returns
        -------
        int
            Number of top-5 ResNet50 predictions inside DISHWARE_INDICES.
            Returns 0 if the crop is invalid.
        """
        if polygon_pts is None or len(polygon_pts) == 0:
            return 0

        h, w = frame.shape[:2]
        x1 = int(max(0, np.min(polygon_pts[:, 0])))
        y1 = int(max(0, np.min(polygon_pts[:, 1])))
        x2 = int(min(w, np.max(polygon_pts[:, 0])))
        y2 = int(min(h, np.max(polygon_pts[:, 1])))

        if x2 - x1 < 20 or y2 - y1 < 20:
            return 0

        crop = frame[y1:y2, x1:x2]
        return self.engine.classify_plate(crop)
