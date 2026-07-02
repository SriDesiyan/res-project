import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from analytics.tracking.serving_detector import POSE_MODEL_PATH

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
    h, w = frame.shape[:2]
    
    # Waiter bounding box: [323, 87, 559, 678]
    wx1, wy1, wx2, wy2 = 323, 87, 559, 678
    pad = 20
    px1, py1 = max(0, wx1 - pad), max(0, wy1 - pad)
    px2, py2 = min(w, wx2 + pad), min(h, wy2 + pad)
    
    waiter_crop = frame[py1:py2, px1:px2]
    crop_h, crop_w = waiter_crop.shape[:2]
    
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
        
    landmarks = results.pose_landmarks[0]
    
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    
    print("\nLandmarks (y increases downwards):")
    print(f"Left Shoulder:  x={left_shoulder.x:.4f}, y={left_shoulder.y:.4f}")
    print(f"Right Shoulder: x={right_shoulder.x:.4f}, y={right_shoulder.y:.4f}")
    print(f"Left Elbow:     x={left_elbow.x:.4f}, y={left_elbow.y:.4f}")
    print(f"Right Elbow:    x={right_elbow.x:.4f}, y={right_elbow.y:.4f}")
    print(f"Left Wrist:     x={left_wrist.x:.4f}, y={left_wrist.y:.4f}")
    print(f"Right Wrist:    x={right_wrist.x:.4f}, y={right_wrist.y:.4f}")
    
    # Check conditions
    left_serving_cond1 = left_wrist.y < left_shoulder.y
    left_serving_cond2 = (left_wrist.y < left_elbow.y and left_elbow.y < left_shoulder.y + 0.15)
    
    right_serving_cond1 = right_wrist.y < right_shoulder.y
    right_serving_cond2 = (right_wrist.y < right_elbow.y and right_elbow.y < right_shoulder.y + 0.15)
    
    print(f"\nLeft Serving Pose Conditions:")
    print(f"  wrist.y < shoulder.y: {left_serving_cond1} ({left_wrist.y:.4f} < {left_shoulder.y:.4f})")
    print(f"  wrist.y < elbow.y:    {left_wrist.y < left_elbow.y} ({left_wrist.y:.4f} < {left_elbow.y:.4f})")
    print(f"  elbow.y < shoulder.y + 0.15: {left_elbow.y < left_shoulder.y + 0.15} ({left_elbow.y:.4f} < {left_shoulder.y + 0.15:.4f})")
    print(f"  Combined (Cond2):     {left_serving_cond2}")
    
    print(f"\nRight Serving Pose Conditions:")
    print(f"  wrist.y < shoulder.y: {right_serving_cond1} ({right_wrist.y:.4f} < {right_shoulder.y:.4f})")
    print(f"  wrist.y < elbow.y:    {right_wrist.y < right_elbow.y} ({right_wrist.y:.4f} < {right_elbow.y:.4f})")
    print(f"  elbow.y < shoulder.y + 0.15: {right_elbow.y < right_shoulder.y + 0.15} ({right_elbow.y:.4f} < {right_shoulder.y + 0.15:.4f})")
    print(f"  Combined (Cond2):     {right_serving_cond2}")
    
    cap.release()

if __name__ == "__main__":
    main()
