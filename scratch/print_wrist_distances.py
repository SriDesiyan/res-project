import cv2
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from analytics.tracking.person_tracker import PersonTracker
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from analytics.tracking.serving_detector import POSE_MODEL_PATH

def main():
    cap = cv2.VideoCapture(str(project_root / "new.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False
    )
    mp_pose = vision.PoseLandmarker.create_from_options(pose_options)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracker = PersonTracker(device, conf=0.20)
    
    start_frame = 8280
    end_frame = 8310
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Tracking wrist distances from frame {start_frame} to {end_frame}...")
    
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_time = frame_num / fps
        persons = tracker.process_frame(frame, frame_time)
        
        for p in persons:
            if p.track_id == 1:
                # Crop waiter
                h, w = frame.shape[:2]
                pad = 20
                waiter_x1, waiter_y1, waiter_x2, waiter_y2 = p.bbox
                px1, py1 = max(0, waiter_x1 - pad), max(0, waiter_y1 - pad)
                px2, py2 = min(w, waiter_x2 + pad), min(h, waiter_y2 + pad)
                
                waiter_crop = frame[py1:py2, px1:px2]
                crop_h, crop_w = waiter_crop.shape[:2]
                
                if crop_h > 20 and crop_w > 20:
                    rgb_crop = cv2.cvtColor(waiter_crop, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                    pose_results = mp_pose.detect(mp_image)
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks[0]
                        left_wrist = landmarks[15]
                        right_wrist = landmarks[16]
                        
                        if left_wrist.x != 0.0 and right_wrist.x != 0.0:
                            wrist_dist = ((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)**0.5
                            print(f"Frame {frame_num}: wrist_dist={wrist_dist:.4f} | left_wrist=({left_wrist.x:.3f}, {left_wrist.y:.3f}), right_wrist=({right_wrist.x:.3f}, {right_wrist.y:.3f})")
                        else:
                            print(f"Frame {frame_num}: Wrists not detected properly")
                    else:
                        print(f"Frame {frame_num}: No pose landmarks")
                        
    cap.release()

if __name__ == "__main__":
    main()
