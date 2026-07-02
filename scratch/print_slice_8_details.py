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
        assigned_table = assign_to_table(p.bottom_center, tables)
        if assigned_table == "table_2":
            x1, y1, x2, y2 = p.bbox
            w = x2 - x1
            h = y2 - y1
            area = w * h
            pid = getattr(p, "session_id", None) or f"T{p.track_id}"
            print(f"F:{frame_num} | Track:{p.track_id} | Session:{pid} | Box:({int(x1)},{int(y1)},{int(x2)},{int(y2)}) | Centroid:({int(p.centroid[0])},{int(p.centroid[1])}) | Area:{area:.0f} | Vel:{p.velocity:.2f} | Confirmed:{p.confirmed} | Role:{p.role}")
            
    frame_num += 5
    for _ in range(4):
        cap.grab()

cap.release()
