import sys
import cv2
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.roi.roi_config import load_tables, assign_to_table
from analytics.tracking.person_tracker import PersonTracker
from analytics.tracking.session_manager import SessionManager

video_path = project_root / "test video 1.mp4"
tables_path = project_root / "analytics" / "config" / "tables.json"
tables = load_tables(tables_path)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

# 2950 to 3010
start_frame = int(2950 * fps)
end_frame = int(3010 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

tracker = PersonTracker(device="cuda", conf=0.20)
session_manager = SessionManager(similarity_threshold=0.85, timeout_sec=900)

frame_num = start_frame
track_history = {} # track_id -> list of centroids

while frame_num < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    frame_time = frame_num / fps
    
    persons = tracker.process_frame(frame, frame_time)
    resumed = session_manager.update(persons, frame_time, frame_shape=frame.shape, tables=tables)
    for p in persons:
        x1, y1, x2, y2 = p.bbox
        anchor = ((x1 + x2) / 2.0, float(y2))
        pid = getattr(p, "session_id", None) or f"T{p.track_id}"
        
        assigned_table = assign_to_table(anchor, tables)
        if assigned_table == "table_2":
            track_history.setdefault(pid, []).append((frame_num, anchor))
            
    frame_num += 5
    # Skip
    for _ in range(4):
        cap.grab()

cap.release()

print("Track ID | Frames Visible | Centroid Range X | Centroid Range Y | Start Frame | End Frame")
print("-" * 90)
for pid, pts in track_history.items():
    xs = [pt[0] for fn, pt in pts]
    ys = [pt[1] for fn, pt in pts]
    fns = [fn for fn, pt in pts]
    print(f"{str(pid):8s} | {len(pts):14d} | {min(xs):.1f}-{max(xs):.1f} | {min(ys):.1f}-{max(ys):.1f} | {min(fns)} | {max(fns)}")
