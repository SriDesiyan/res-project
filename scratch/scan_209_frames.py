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
    
    found_ranges = []
    current_range = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_time = frame_num / fps
        
        # Run YOLO detection and tracking
        results = tracker.yolo.track(
            frame, classes=[0], conf=tracker.conf,
            persist=True, tracker=tracker.tracker_cfg, verbose=False
        )
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().numpy()
            if 209 in track_ids:
                if current_range is None:
                    current_range = [frame_num, frame_num]
                else:
                    current_range[1] = frame_num
            else:
                if current_range is not None:
                    found_ranges.append(current_range)
                    current_range = None

        frame_num += 1
        if frame_num % 2000 == 0:
            print(f"Scanned {frame_num} frames... found ranges: {found_ranges}")

    if current_range is not None:
        found_ranges.append(current_range)
        
    print(f"Scanning complete. Track 209 active during frames:")
    for r in found_ranges:
        start_sec = r[0] / fps
        end_sec = r[1] / fps
        print(f"  Frames {r[0]} - {r[1]} ({start_sec:.2f}s - {end_sec:.2f}s)")

    cap.release()

if __name__ == "__main__":
    main()
