import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# Paths to models
project_root = Path(__file__).parent.parent.parent.resolve()
HAND_MODEL_PATH = str(project_root / "embedding" / "hand_landmarker.task")
POSE_MODEL_PATH = str(project_root / "embedding" / "pose_landmarker.task")


def detect_food_in_frame(frame, yolo_model):
    """
    Detect plates, bowls, cups, and food items in the frame using COCO pre-trained YOLOv8.
    """
    # COCO classes for food and dining accessories:
    # 39: bottle, 40: wine glass, 41: cup, 42: fork, 43: knife, 44: spoon, 45: bowl
    # 46-55: various food classes (banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake)
    food_classes = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

    results = yolo_model(frame, classes=food_classes, conf=0.20, verbose=False)

    food_detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for bbox, class_id, conf in zip(boxes, classes, confs):
            class_name = yolo_model.names[class_id]
            food_detections.append({
                'class': class_name,
                'bbox': bbox,  # [x1, y1, x2, y2]
                'confidence': conf
            })
    return food_detections


def detect_waiter_serving(frame, waiter_bbox, yolo_model, hands_detector,
                          pose_detector, food_detections=None):
    """
    Multi-method crop-based hybrid detection for food serving.

    Confidence scoring:
      - method1  (+0.40): YOLO food detection inside/near the waiter bounding box.
      - method2  (+0.50): Hand landmark within 60px of a YOLO food detection
                          (strongest single indicator — hand touching food).
      - method3  (+0.15): Pose indicates one wrist above elbow+shoulder threshold
                          AND food was detected in method1 (pose alone is insufficient).
      - method4  (+0.35): White/dark object near wrist AND a YOLO food detection
                          exists somewhere in the frame (colour alone insufficient).

    Serving is confirmed when total confidence >= 0.60.

    NOTE: The black-plate unconditional override has been removed.
    NOTE: Pose (arm-raise) alone can no longer trigger is_serving.
    NOTE: Colour-near-wrist alone can no longer trigger is_serving.
    """
    if food_detections is None:
        food_detections = detect_food_in_frame(frame, yolo_model)

    waiter_x1, waiter_y1, waiter_x2, waiter_y2 = waiter_bbox

    # ── Method 1: YOLO food inside / near waiter bounding box ─────────
    waiter_food_detections = []
    for food in food_detections:
        fx1, fy1, fx2, fy2 = food['bbox']
        waiter_mid_y = waiter_y1 + (waiter_y2 - waiter_y1) * 0.5
        x_overlap = (waiter_x1 - 30 <= fx1 <= waiter_x2 + 30
                     or waiter_x1 - 30 <= fx2 <= waiter_x2 + 30)
        y_overlap = waiter_y1 - 10 <= fy1 <= waiter_mid_y + 30
        if x_overlap and y_overlap:
            waiter_food_detections.append(food)

    method1_serving = len(waiter_food_detections) > 0
    method1_food_type = waiter_food_detections[0]['class'] if method1_serving else None

    # ── Crop waiter bounding box (20px padding) to isolate from neighbours ──
    h, w = frame.shape[:2]
    pad = 20
    px1, py1 = max(0, waiter_x1 - pad), max(0, waiter_y1 - pad)
    px2, py2 = min(w, waiter_x2 + pad), min(h, waiter_y2 + pad)

    waiter_crop = frame[py1:py2, px1:px2]
    crop_h, crop_w = waiter_crop.shape[:2]

    method2_serving = False
    method2_food = None
    method3_serving = False
    method4_serving = False
    is_order_taking = False
    serving_hand_pos = None
    pose_results = None

    if crop_h > 20 and crop_w > 20:
        rgb_crop = cv2.cvtColor(waiter_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

        # ── Run Pose Landmarker on isolated waiter crop ────────────────
        pose_results = pose_detector.detect(mp_image)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]

            # Order-taking detection: both wrists close together
            if left_wrist.x != 0.0 and right_wrist.x != 0.0:
                wrist_dist = (
                    (left_wrist.x - right_wrist.x) ** 2
                    + (left_wrist.y - right_wrist.y) ** 2
                ) ** 0.5
                if wrist_dist < 0.35:
                    is_order_taking = True

            # Pose serving check (method3):
            # Wrist above shoulder OR wrist above elbow when elbow is near shoulder.
            left_serving = (
                left_wrist.y < left_shoulder.y
                or (left_wrist.y < left_elbow.y
                    and left_elbow.y < left_shoulder.y + 0.25)
            )
            right_serving = (
                right_wrist.y < right_shoulder.y
                or (right_wrist.y < right_elbow.y
                    and right_elbow.y < right_shoulder.y + 0.25)
            )
            if left_serving or right_serving:
                method3_serving = True

            # ── Method 4: Colour near wrist ──
            for wrist in [left_wrist, right_wrist]:
                wx = int(wrist.x * crop_w)
                wy = int(wrist.y * crop_h)

                r = 45
                x1_crop = max(0, wx - r)
                x2_crop = min(crop_w, wx + r)
                y1_crop = max(0, wy - r)
                y2_crop = min(crop_h, wy + r)

                if x2_crop - x1_crop >= 10 and y2_crop - y1_crop >= 10:
                    wrist_crop = waiter_crop[y1_crop:y2_crop, x1_crop:x2_crop]
                    hsv_crop = cv2.cvtColor(wrist_crop, cv2.COLOR_BGR2HSV)

                    # White: Value > 170, Saturation < 60
                    white_pixels = (
                        (hsv_crop[:, :, 2] > 170) & (hsv_crop[:, :, 1] < 60)
                    )
                    white_ratio = np.mean(white_pixels)

                    # Dark (including black plates): Value < 60
                    dark_pixels = (hsv_crop[:, :, 2] < 60)
                    dark_ratio = np.mean(dark_pixels)

                    if white_ratio > 0.18 or dark_ratio > 0.18:
                        method4_serving = True
                        serving_hand_pos = (px1 + wx, py1 + wy)
                        break

            # ── Method 2: Hand landmark proximity to YOLO food ─────────
            for wrist in [left_wrist, right_wrist]:
                wrist_frame_x = px1 + int(wrist.x * crop_w)
                wrist_frame_y = py1 + int(wrist.y * crop_h)

                for food in food_detections:
                    fx1, fy1, fx2, fy2 = food['bbox']
                    fcx = (fx1 + fx2) / 2.0
                    fcy = (fy1 + fy2) / 2.0
                    distance = (
                        (wrist_frame_x - fcx) ** 2
                        + (wrist_frame_y - fcy) ** 2
                    ) ** 0.5
                    if distance < 60:
                        method2_serving = True
                        method2_food = food['class']
                        if serving_hand_pos is None:
                            serving_hand_pos = (wrist_frame_x, wrist_frame_y)
                        break
                if method2_serving:
                    break

    # ── Confidence aggregation ─────────────────────────────────────────
    confidence = 0.0
    if method1_serving:
        confidence += 0.40
    if method2_serving:
        confidence += 0.50
    if method4_serving:
        confidence += 0.35
    if method3_serving:
        confidence += 0.15

    # Order-taking suppresses serving UNLESS hand-to-food proximity was detected
    # (a waiter carrying a plate with both hands has close wrists but IS serving).
    is_serving = confidence >= 0.50
    if is_order_taking and not method2_serving:
        is_serving = False

    # Fallback serving hand position
    if serving_hand_pos is None and (method3_serving or method2_serving) \
            and crop_h > 20 and crop_w > 20:
        if pose_results and pose_results.pose_landmarks:
            lw = pose_results.pose_landmarks[0][15]
            rw = pose_results.pose_landmarks[0][16]
            chosen = lw if lw.y > rw.y else rw
            serving_hand_pos = (
                px1 + int(chosen.x * crop_w),
                py1 + int(chosen.y * crop_h)
            )

    food_type = (
        method2_food
        or method1_food_type
        or ("plate" if method4_serving else "food")
    )

    return {
        'is_serving': is_serving,
        'is_order_taking': is_order_taking,
        'confidence': confidence,
        'serving_hand_pos': serving_hand_pos,
        'food_type': food_type,
        'methods': {
            'food_detection': method1_serving,
            'hand_detection': method2_serving,
            'pose_detection': method3_serving,
            'colour_near_wrist': method4_serving,
        }
    }
