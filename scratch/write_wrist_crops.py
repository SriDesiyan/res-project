import cv2
import numpy as np
import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from analytics.tracking.serving_detector import HAND_MODEL_PATH, POSE_MODEL_PATH

def main():
    cap = cv2.VideoCapture(str(project_root / "new.mp4"))
    
    start_frame = 7950
    target_frame = 8245
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for f in range(start_frame, target_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at {f}")
            cap.release()
            return
        
    print(f"Successfully loaded frame {target_frame}")
    
    # Waiter bounding box: [323, 87, 559, 678]
    # Let's add 20px padding
    h, w = frame.shape[:2]
    wx1, wy1, wx2, wy2 = 323, 87, 559, 678
    px1, py1 = max(0, wx1 - 20), max(0, wy1 - 20)
    px2, py2 = min(w, wx2 + 20), min(h, wy2 + 20)
    
    waiter_crop = frame[py1:py2, px1:px2]
    crop_h, crop_w = waiter_crop.shape[:2]
    print(f"Waiter crop shape: {waiter_crop.shape}")
    
    # Initialize Pose detector
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
    
    rgb_crop = cv2.cvtColor(waiter_crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
    results = pose_detector.detect(mp_image)
    
    if not results.pose_landmarks:
        print("No pose detected in cropped waiter image!")
        cap.release()
        return
        
    print("Pose detected in cropped waiter image!")
    landmarks = results.pose_landmarks[0]
    
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    
    out_dir = project_root / "inference_output" / "wrist_debug"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for name, wrist in [("left", left_wrist), ("right", right_wrist)]:
        # coordinates are normalized to the crop size!
        cx = int(wrist.x * crop_w)
        cy = int(wrist.y * crop_h)
        print(f"\nCropped {name.upper()} WRIST at ({cx}, {cy})")
        
        # Crop 45px radius around wrist inside the waiter crop
        r = 45
        x1, x2 = max(0, cx - r), min(crop_w, cx + r)
        y1, y2 = max(0, cy - r), min(crop_h, cy + r)
        
        crop = waiter_crop[y1:y2, x1:x2]
        cv2.imwrite(str(out_dir / f"crop_{name}_wrist.jpg"), crop)
        print(f"  Saved wrist crop to {out_dir / f'crop_{name}_wrist.jpg'}")
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Test white ranges: V > 170, S < 60
        white_pixels = (hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 60)
        white_pct = np.mean(white_pixels) * 100.0
        print(f"  White pixel percentage (V>170, S<60): {white_pct:.1f}%")

    cap.release()

if __name__ == "__main__":
    main()
