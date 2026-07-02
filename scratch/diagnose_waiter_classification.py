import cv2
import numpy as np
import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.tracking.person_tracker import PersonTracker
from analytics.tracking.serving_detector import detect_food_in_frame, detect_waiter_serving
from analytics.tracking.serving_detector import HAND_MODEL_PATH, POSE_MODEL_PATH
from analytics.roi.roi_config import load_tables
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    cap = cv2.VideoCapture(str(project_root / "new.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    # Initialize MediaPipe models
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2
    )
    mp_hands = vision.HandLandmarker.create_from_options(hand_options)

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False
    )
    mp_pose = vision.PoseLandmarker.create_from_options(pose_options)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    tracker = PersonTracker(device, conf=0.20)
    
    # Start earlier to lock waiter role first, then monitor serving
    start_frame = 38700
    end_frame = 38850
    step = 1
    
    print(f"Running sequential verification from frame {start_frame} to {end_frame}...")
    
    for frame_num in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_time = frame_num / fps
        persons = tracker.process_frame(frame, frame_time)
        
        for p in persons:
            # Print status if they are waiter role, or if they are the primary waiter candidate in these frames
            if p.role == "waiter" or p.track_id == 2:
                # Run serving check (this mimics pipeline.py)
                food_detections = detect_food_in_frame(frame, tracker.yolo)
                res = detect_waiter_serving(frame, p.bbox, tracker.yolo, mp_hands, mp_pose, food_detections)
                
                if res['is_serving']:
                    print(f"F:{frame_num} | ID:{p.track_id} | role:{p.role} | hits:{tracker.waiter_hits[p.track_id]}")
                    print(f"  -> DETECTED SERVING EVENT FOR WAITER! conf={res['confidence']:.2f} | food={res['food_type']}")
                    print(f"     Methods: {res['methods']}")

    cap.release()

if __name__ == "__main__":
    main()
