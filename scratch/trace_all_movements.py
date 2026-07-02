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

start_frame = int(2950 * fps)
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
        x1, y1, x2, y2 = p.bbox
        cx, cy = p.centroid
        assigned = assign_to_table(p.bottom_center, tables)
        # Print tracks that are not waiters and have centroid y > 600 (bottom half of screen)
        if p.role == "customer" and cy > 600:
            print(f"F:{frame_num} | Track:{p.track_id} | Centroid:({int(cx)},{int(cy)}) | Assigned:{assigned} | Vel:{p.velocity:.1f}")
            
    frame_num += 10
    for _ in range(9):
        cap.grab()

cap.release()
