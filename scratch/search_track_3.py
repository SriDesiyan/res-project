import sys
import cv2
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.roi.roi_config import load_tables, assign_to_table
from analytics.tracking.person_tracker import PersonTracker

video_path = project_root / "test video 1.mp4"
tables_path = project_root / "analytics" / "config" / "tables.json"
tables = load_tables(tables_path)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

start_frame = int(2990 * fps)
end_frame = int(3010 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

tracker = PersonTracker(device="cuda", conf=0.20)

frame_num = start_frame
while frame_num < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    frame_time = frame_num / fps
    persons = tracker.process_frame(frame, frame_time)
    
    for p in persons:
        if p.track_id == 3:
            assigned = assign_to_table(p.bottom_center, tables)
            print(f"F:{frame_num} | Track:3 | Centroid:({int(p.centroid[0])},{int(p.centroid[1])}) | Assigned:{assigned} | Role:{p.role} | Confirmed:{p.confirmed}")
            
    frame_num += 1
cap.release()
