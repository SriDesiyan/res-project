import cv2
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.tracking.person_tracker import PersonTracker
from analytics.tracking.serving_detector import detect_food_in_frame, detect_waiter_serving
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from analytics.tracking.serving_detector import HAND_MODEL_PATH, POSE_MODEL_PATH

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
    tracker = PersonTracker(device, conf=0.20)
    
    # We analyze frames from 7950 to 8310
    start_frame = 7950
    end_frame = 8310
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Debugging serving detection from frame {start_frame} to {end_frame}...")
    
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_time = frame_num / fps
        persons = tracker.process_frame(frame, frame_time)
        
        # Check if there is any waiter
        waiters = [p for p in persons if p.role == "waiter"]
        if waiters and frame_num >= 8240:
            # Detect food
            food_detections = detect_food_in_frame(frame, tracker.yolo)
            
            print(f"\n--- Frame {frame_num} ({frame_time:.2f}s) ---")
            print(f"Found {len(waiters)} waiter(s) and {len(food_detections)} food items.")
            
            # Print food items found
            for f_det in food_detections:
                print(f"  Food: {f_det['class']} at {f_det['bbox']} conf={f_det['confidence']:.2f}")
                
            for w in waiters:
                res = detect_waiter_serving(frame, w.bbox, tracker.yolo, mp_hands, mp_pose, food_detections)
                print(f"  Waiter ID {w.track_id}: bbox={w.bbox}")
                print(f"    is_serving: {res['is_serving']} (confidence={res['confidence']:.2f})")
                print(f"    food_type: {res['food_type']}")
                print(f"    methods: {res['methods']}")

    cap.release()

if __name__ == "__main__":
    main()
