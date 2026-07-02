import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import json

project_root = Path("/Users/gaurisudharsinip/Desktop/wgtech")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.tracking.person_tracker import PersonTracker

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracker = PersonTracker(device, conf=0.20)
    
    cap = cv2.VideoCapture(str(project_root / "table_wghotel.mp4"))
    frame_num = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    while frame_num < 150:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_time = frame_num / fps
        h, w = frame.shape[:2]
        
        # We manually run the detection/classification part to trace similarities
        results = tracker.yolo.track(
            frame, classes=[0], conf=tracker.conf,
            persist=True, tracker=tracker.tracker_cfg, verbose=False
        )
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                    
                # Get similarity and uniform match
                emb, max_sim, is_uniform = tracker._get_embedding_and_classify(frame, x1, y1, x2, y2)
                
                # Print details for track_id 13 (waiter) and any other active tracks
                if track_id in [13, 7]:
                    print(f"Frame {frame_num} ({frame_time:.2f}s): Track {track_id} | max_sim={max_sim:.3f} | is_uniform={is_uniform}")

        frame_num += 1

    cap.release()

if __name__ == "__main__":
    main()
