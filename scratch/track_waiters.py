import sys
import cv2
import torch
import numpy as np
from pathlib import Path

project_root = Path("/Users/gaurisudharsinip/Desktop/wgtech")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.tracking.person_tracker import PersonTracker
from analytics.roi.roi_config import load_tables

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracker = PersonTracker(device, conf=0.20)
    
    # Load tables
    tables_path = project_root / "analytics" / "config" / "tables.json"
    with open(tables_path) as f:
        import json
        tables = json.load(f)["tables"]

    cap = cv2.VideoCapture(str(project_root / "table_wghotel.mp4"))
    frame_num = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    while frame_num < 500:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_time = frame_num / fps
        persons = tracker.process_frame(frame, frame_time)
        
        # Check if there is any person classified as a waiter, or check all persons
        waiters = [p for p in persons if p.role == "waiter"]
        if waiters:
            print(f"\nFrame {frame_num} ({frame_time:.2f}s): Found {len(waiters)} waiter(s)!")
            for w in waiters:
                print(f"  Waiter ID {w.track_id}: bbox={w.bbox}, bottom_center={w.bottom_center}")
                # Print distance to tables
                for tid, t_info in tables.items():
                    poly = np.array(t_info["polygon"], dtype=np.int32)
                    dist = cv2.pointPolygonTest(poly, w.bottom_center, measureDist=True)
                    print(f"    Distance to {tid}: {dist:.1f} px")
                    
        # Let's also print if a person has track_id 209
        for p in persons:
            if p.track_id == 209:
                print(f"  Frame {frame_num}: Person 209 detected! role={p.role}, bbox={p.bbox}, bottom_center={p.bottom_center}")
                for tid, t_info in tables.items():
                    poly = np.array(t_info["polygon"], dtype=np.int32)
                    dist = cv2.pointPolygonTest(poly, p.bottom_center, measureDist=True)
                    print(f"    Distance to {tid}: {dist:.1f} px")

        frame_num += 1

    cap.release()

if __name__ == "__main__":
    main()
