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
from analytics.roi.roi_config import load_tables

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
    
    tables = load_tables(project_root / "analytics" / "config" / "tables.json")
    
    start_frame = 1246
    end_frame = 1496
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Debugging serving detection from frame {start_frame} to {end_frame}...")
    
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_time = frame_num / fps
        persons = tracker.process_frame(frame, frame_time)
        
        # Check if there is any person
        for p in persons:
            best_table = None
            best_dist = -float('inf')
            for tid, t_info in tables.items():
                poly = np.array(t_info["polygon"], dtype=np.int32)
                dist = cv2.pointPolygonTest(poly, p.bottom_center, measureDist=True)
                if dist > best_dist:
                    best_dist = dist
                    best_table = tid
            if best_table and best_dist >= -180.0:
                p.assigned_table = best_table
            else:
                p.assigned_table = None
            
            # Print state
            if frame_num % 5 == 0 or frame_num >= 8280:
                print(f"Frame {frame_num} ({frame_time:.2f}s) | Person ID {p.track_id}: role={p.role}, hits={tracker.waiter_hits[p.track_id]}, bbox={p.bbox}, bottom_center={p.bottom_center}")
                print(f"  Assigned Table: {p.assigned_table} (dist={best_dist:.1f} to {best_table})")
                
                # Check serving
                if p.role == "waiter" or p.track_id == 1:
                    food_detections = detect_food_in_frame(frame, tracker.yolo)
                    res = detect_waiter_serving(frame, p.bbox, tracker.yolo, mp_hands, mp_pose, food_detections)
                    
                    table_has_plates = False
                    if p.assigned_table and food_detections:
                        table_info = tables.get(p.assigned_table)
                        if table_info:
                            poly = np.array(table_info["polygon"], dtype=np.int32)
                            for food in food_detections:
                                fx1, fy1, fx2, fy2 = food['bbox']
                                fcx = (fx1 + fx2) / 2.0
                                fcy = (fy1 + fy2) / 2.0
                                dist = cv2.pointPolygonTest(poly, (fcx, fcy), measureDist=True)
                                if dist >= -30.0:
                                    table_has_plates = True
                                    break
                    
                    is_serving_detected = res['is_serving']
                    is_order_taking_detected = res.get('is_order_taking', False)
                    
                    if is_serving_detected:
                        if table_has_plates:
                            is_serving = True
                            is_order_taking = False
                        else:
                            is_serving = False
                            is_order_taking = True
                    else:
                        is_serving = False
                        is_order_taking = is_order_taking_detected
                        
                    print(f"  is_serving: {is_serving} (is_order_taking={is_order_taking}, confidence={res['confidence']:.2f}, table_has_plates={table_has_plates})")
                    print(f"  food_type: {res['food_type']}")
                    print(f"  methods: {res['methods']}")

    cap.release()

if __name__ == "__main__":
    main()
