import sys
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.roi.roi_config import load_tables, assign_to_table
from analytics.tracking.person_tracker import PersonTracker
from analytics.tracking.session_manager import SessionManager
from analytics.occupancy.occupancy_engine import OccupancyEngine
from analytics.fsm.table_fsm import TableStateFSM

# Configure output encoding
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

video_path = project_root / "test video 1.mp4"
tables_path = project_root / "analytics" / "config" / "tables.json"
tables = load_tables(tables_path)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

# Start around 2985s (frame 74435) to 3010s (frame 75057)
start_frame = int(2985 * fps)
end_frame = int(3010 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

tracker = PersonTracker(device="cuda", conf=0.20)
session_manager = SessionManager(similarity_threshold=0.85, timeout_sec=900)
occupancy = OccupancyEngine(["table_2"], tables)

# We initialize FSM directly to DINING to simulate the state before departure
fsm = TableStateFSM("table_2")
fsm.state = "DINING"
fsm.warm_up_complete = True

frame_num = start_frame
print("Frame | Track ID | Centroid | Distance to Table 2 Center | Assigned | Confirmed | Role")
print("-" * 80)

table_2_center = tables["table_2"]["center"]

while frame_num < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    frame_time = frame_num / fps
    
    persons = tracker.process_frame(frame, frame_time)
    
    # Assign persons to tables
    for person in persons:
        x1, y1, x2, y2 = person.bbox
        anchor = ((x1 + x2) / 2.0, float(y2))
        person.assigned_table = assign_to_table(anchor, tables)
    
    resumed = session_manager.update(persons, frame_time, frame_shape=frame.shape, tables=tables)
    
    # Print tracking details of all customers at table_2
    for p in persons:
        x1, y1, x2, y2 = p.bbox
        anchor = ((x1 + x2) / 2.0, float(y2))
        dist = ((anchor[0] - table_2_center[0])**2 + (anchor[1] - table_2_center[1])**2)**0.5
        pid = getattr(p, "session_id", f"T{p.track_id}")
        
        # If the person is assigned to table_2, or close to it, print their info
        if p.assigned_table == "table_2" or dist < 250:
            print(f"F:{frame_num} | {pid} | ({int(anchor[0])},{int(anchor[1])}) | {dist:.1f}px | Assigned: {p.assigned_table} | Confirmed: {p.confirmed} | Role: {p.role}")
            
    # Update occupancy
    occupancy.update(persons, frame_time)
    table_obj = occupancy.tables["table_2"]
    raw_customer_present = getattr(table_obj, "raw_customer_present", False)
    
    # Update FSM
    fsm.update(
        customer_present=raw_customer_present,
        waiter_present=False,
        is_writing=False,
        is_serving=False,
        dirty_object_count=0,
        frame_time=frame_time
    )
    
    if fsm.state != "DINING":
        print(f">>> FSM Transitioned to {fsm.state} at frame {frame_num} ({frame_time:.2f}s)!")
        break
        
    frame_num += 1

cap.release()
print("Finished analysis.")
