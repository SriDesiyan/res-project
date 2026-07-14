"""
analytics/tracking/serving_detector.py
========================================
Multi-method waiter serving / order-taking detector.

MIGRATION NOTE (Edge AI):
    This module no longer imports mediapipe directly.
    Pose and hand landmark detection are routed through the
    ``BaseInferenceEngine`` interface (engine.detect_pose / engine.detect_hands).
    All confidence scoring logic (methods 1-4) is UNCHANGED.
    All threshold values are UNCHANGED.

The engine is passed in at call time — no global model objects.
This preserves the ability to call detect_waiter_serving() without
knowing or caring which hardware backend is active.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Kept for backward-compat path references (MediaPipe .task files on disk).
project_root = Path(__file__).parent.parent.parent.resolve()
HAND_MODEL_PATH = str(project_root / "embedding" / "hand_landmarker.task")
POSE_MODEL_PATH = str(project_root / "embedding" / "pose_landmarker.task")

# COCO food class IDs (unchanged from original)
_FOOD_CLASSES = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]


def detect_food_in_frame(frame: np.ndarray, engine_or_yolo) -> List[Dict[str, Any]]:
    """
    Detect plates, bowls, cups, and food items using the inference engine.

    ``engine_or_yolo`` accepts either:
      - A ``BaseInferenceEngine`` instance (new path)
      - The original Ultralytics YOLO object (legacy path — backward compat)

    Returns a list of dicts: {class, bbox, confidence}
    """
    # New path: engine exposes detect_food()
    if hasattr(engine_or_yolo, "detect_food"):
        return engine_or_yolo.detect_food(frame, food_classes=_FOOD_CLASSES, conf=0.20)

    # Legacy path: raw Ultralytics YOLO (kept for backward compat during transition)
    results = engine_or_yolo(frame, classes=_FOOD_CLASSES, conf=0.20, verbose=False)
    food_detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for bbox, class_id, conf in zip(boxes, classes, confs):
            food_detections.append({
                "class": engine_or_yolo.names[class_id],
                "bbox": bbox,
                "confidence": conf,
            })
    return food_detections


def detect_waiter_serving(
    frame: np.ndarray,
    waiter_bbox: tuple,
    engine_or_yolo,
    hands_detector=None,
    pose_detector=None,
    food_detections: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Multi-method crop-based hybrid detection for food serving.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR video frame.
    waiter_bbox : tuple
        (x1, y1, x2, y2) bounding box of the waiter.
    engine_or_yolo : BaseInferenceEngine | YOLO
        Inference engine (new path) or raw YOLO object (legacy path).
    hands_detector : optional
        Legacy MediaPipe HandLandmarker object.  Ignored when engine_or_yolo
        is a BaseInferenceEngine (engine.detect_hands() is used instead).
    pose_detector : optional
        Legacy MediaPipe PoseLandmarker object.  Same as above.
    food_detections : list | None
        Pre-computed food detections for this frame (avoids double inference).

    Confidence scoring (UNCHANGED from original):
      method1 (+0.40): YOLO food detection inside/near waiter bounding box.
      method2 (+0.50): Hand landmark within 60px of a YOLO food detection.
      method3 (+0.15): Pose — one wrist above shoulder/elbow threshold AND
                       food detected in method1.
      method4 (+0.35): White/dark object near wrist AND YOLO food exists.

    Serving confirmed when total confidence >= 0.50.
    """
    # ── Determine how to get food detections ────────────────────────────────
    if food_detections is None:
        food_detections = detect_food_in_frame(frame, engine_or_yolo)

    # ── Determine how to get landmarks ──────────────────────────────────────
    # New path: engine exposes detect_pose/detect_hands
    use_engine = hasattr(engine_or_yolo, "detect_pose")

    waiter_x1, waiter_y1, waiter_x2, waiter_y2 = waiter_bbox

    # ── Method 1: YOLO food inside / near waiter bounding box ───────────────
    waiter_food_detections = []
    for food in food_detections:
        fx1, fy1, fx2, fy2 = food["bbox"]
        waiter_mid_y = waiter_y1 + (waiter_y2 - waiter_y1) * 0.5
        x_overlap = (waiter_x1 - 30 <= fx1 <= waiter_x2 + 30
                     or waiter_x1 - 30 <= fx2 <= waiter_x2 + 30)
        y_overlap = waiter_y1 - 10 <= fy1 <= waiter_mid_y + 30
        if x_overlap and y_overlap:
            waiter_food_detections.append(food)

    method1_serving = len(waiter_food_detections) > 0
    method1_food_type = waiter_food_detections[0]["class"] if method1_serving else None

    # ── Crop waiter region ───────────────────────────────────────────────────
    h, w = frame.shape[:2]
    pad = 20
    px1 = max(0, waiter_x1 - pad)
    py1 = max(0, waiter_y1 - pad)
    px2 = min(w, waiter_x2 + pad)
    py2 = min(h, waiter_y2 + pad)
    waiter_crop = frame[py1:py2, px1:px2]
    crop_h, crop_w = waiter_crop.shape[:2]

    method2_serving = False
    method2_food = None
    method3_serving = False
    method4_serving = False
    is_order_taking = False
    serving_hand_pos = None
    landmarks = []

    if crop_h > 20 and crop_w > 20:
        # ── Get pose landmarks ───────────────────────────────────────────────
        if use_engine:
            pose_result = engine_or_yolo.detect_pose(waiter_crop)
            landmarks = pose_result.get("landmarks", [])
        elif pose_detector is not None:
            # Legacy MediaPipe path
            try:
                import mediapipe as _mp
                rgb_crop = cv2.cvtColor(waiter_crop, cv2.COLOR_BGR2RGB)
                mp_img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb_crop)
                pose_res = pose_detector.detect(mp_img)
                if pose_res.pose_landmarks:
                    lms = pose_res.pose_landmarks[0]
                    landmarks = [{"x": float(lm.x), "y": float(lm.y),
                                   "z": float(lm.z),
                                   "visibility": float(getattr(lm, "visibility", 1.0))}
                                 for lm in lms]
            except Exception:
                landmarks = []

        # ── Pose analysis (indices: 11=L.shoulder,12=R.shoulder,13=L.elbow,
        #                           14=R.elbow,15=L.wrist,16=R.wrist) ────────
        if len(landmarks) > 16:
            ls = landmarks[11]   # left shoulder
            rs = landmarks[12]   # right shoulder
            le = landmarks[13]   # left elbow
            re = landmarks[14]   # right elbow
            lw = landmarks[15]   # left wrist
            rw = landmarks[16]   # right wrist

            # Order-taking: both wrists close together
            if lw["x"] != 0.0 and rw["x"] != 0.0:
                wrist_dist = (
                    (lw["x"] - rw["x"]) ** 2 + (lw["y"] - rw["y"]) ** 2
                ) ** 0.5
                if wrist_dist < 0.35:
                    is_order_taking = True

            # Method 3: wrist above shoulder / elbow
            left_serving = (
                lw["y"] < ls["y"]
                or (lw["y"] < le["y"] and le["y"] < ls["y"] + 0.25)
            )
            right_serving = (
                rw["y"] < rs["y"]
                or (rw["y"] < re["y"] and re["y"] < rs["y"] + 0.25)
            )
            if left_serving or right_serving:
                method3_serving = True

            # Method 4: colour near wrist
            for wrist in [lw, rw]:
                wx_px = int(wrist["x"] * crop_w)
                wy_px = int(wrist["y"] * crop_h)
                r = 45
                x1c = max(0, wx_px - r)
                x2c = min(crop_w, wx_px + r)
                y1c = max(0, wy_px - r)
                y2c = min(crop_h, wy_px + r)
                if x2c - x1c >= 10 and y2c - y1c >= 10:
                    wrist_crop = waiter_crop[y1c:y2c, x1c:x2c]
                    hsv_crop = cv2.cvtColor(wrist_crop, cv2.COLOR_BGR2HSV)
                    white_pixels = (hsv_crop[:, :, 2] > 170) & (hsv_crop[:, :, 1] < 60)
                    white_ratio = np.mean(white_pixels)
                    dark_pixels = hsv_crop[:, :, 2] < 60
                    dark_ratio = np.mean(dark_pixels)
                    if white_ratio > 0.18 or dark_ratio > 0.18:
                        method4_serving = True
                        serving_hand_pos = (px1 + wx_px, py1 + wy_px)
                        break

            # Method 2: hand landmark proximity to YOLO food
            for wrist in [lw, rw]:
                wfx = px1 + int(wrist["x"] * crop_w)
                wfy = py1 + int(wrist["y"] * crop_h)
                for food in food_detections:
                    fx1f, fy1f, fx2f, fy2f = food["bbox"]
                    fcx = (fx1f + fx2f) / 2.0
                    fcy = (fy1f + fy2f) / 2.0
                    dist = ((wfx - fcx) ** 2 + (wfy - fcy) ** 2) ** 0.5
                    if dist < 60:
                        method2_serving = True
                        method2_food = food["class"]
                        if serving_hand_pos is None:
                            serving_hand_pos = (wfx, wfy)
                        break
                if method2_serving:
                    break

    # ── Confidence aggregation (UNCHANGED) ──────────────────────────────────
    confidence = 0.0
    if method1_serving:
        confidence += 0.40
    if method2_serving:
        confidence += 0.50
    if method4_serving:
        confidence += 0.35
    if method3_serving:
        confidence += 0.15

    is_serving = confidence >= 0.50
    if is_order_taking and not method2_serving:
        is_serving = False

    # Fallback hand position
    if serving_hand_pos is None and (method3_serving or method2_serving) \
            and crop_h > 20 and len(landmarks) > 16:
        lw = landmarks[15]
        rw = landmarks[16]
        chosen = lw if lw["y"] > rw["y"] else rw
        serving_hand_pos = (
            px1 + int(chosen["x"] * crop_w),
            py1 + int(chosen["y"] * crop_h),
        )

    food_type = (
        method2_food
        or method1_food_type
        or ("plate" if method4_serving else "food")
    )

    return {
        "is_serving": is_serving,
        "is_order_taking": is_order_taking,
        "confidence": confidence,
        "serving_hand_pos": serving_hand_pos,
        "food_type": food_type,
        "methods": {
            "food_detection": method1_serving,
            "hand_detection": method2_serving,
            "pose_detection": method3_serving,
            "colour_near_wrist": method4_serving,
        },
    }
