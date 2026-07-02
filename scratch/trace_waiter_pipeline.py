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
from analytics.tracking.session_manager import SessionManager
from analytics.occupancy.occupancy_engine import OccupancyEngine
from analytics.roi.roi_config import assign_to_table

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracker = PersonTracker(device, conf=0.20)
    session_manager = SessionManager(similarity_threshold=0.85, timeout_sec=900)
    
    # Load tables
    tables_path = project_root / "analytics" / "config" / "tables.json"
    with open(tables_path) as f:
        tables = json.load(f)["tables"]
    table_ids = list(tables.keys())
    
    occupancy = OccupancyEngine(table_ids, tables)

    cap = cv2.VideoCapture(str(project_root / "table_wghotel.mp4"))
    frame_num = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    while frame_num < 150:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_time = frame_num / fps
        persons = tracker.process_frame(frame, frame_time)
        
        # ── Step 2: Assign persons to tables ───────────────
        for person in persons:
            if person.role == "waiter":
                best_table = None
                best_dist = -float('inf')
                for tid, t_info in tables.items():
                    poly = np.array(t_info["polygon"], dtype=np.int32)
                    dist = cv2.pointPolygonTest(poly, person.bottom_center, measureDist=True)
                    if dist > best_dist:
                        best_dist = dist
                        best_table = tid
                if best_table and best_dist >= -180.0:
                    person.assigned_table = best_table
                else:
                    person.assigned_table = None
            else:
                person.assigned_table = assign_to_table(person.bottom_center, tables)

        # ── Step 3: Session Management (Re-ID) ─────────────
        resumed_sessions = session_manager.update(persons, frame_time)

        # Print state of track 13
        p13 = None
        for p in persons:
            if p.track_id == 13:
                p13 = p
                break
                
        # ── Step 4: Update occupancy ───────────────────────
        occupancy.update(persons, frame_time)
        occ_data = occupancy.get_all_status(frame_time)
        
        if p13 is not None:
            t1_occ = [occ for occ in occ_data if occ["table_id"] == "table_1"][0]
            t2_occ = [occ for occ in occ_data if occ["table_id"] == "table_2"][0]
            print(f"Frame {frame_num} ({frame_time:.2f}s): ID 13 role={p13.role}, session_id={p13.session_id}, assigned_table={p13.assigned_table}")
            print(f"  table_1: current_waiters={occupancy.tables['table_1'].current_waiters}, visits={t1_occ['waiter_visits']}, present={t1_occ['waiter_present']}")
            print(f"  table_2: current_waiters={occupancy.tables['table_2'].current_waiters}, visits={t2_occ['waiter_visits']}, present={t2_occ['waiter_present']}")

        frame_num += 1

    cap.release()

if __name__ == "__main__":
    main()
