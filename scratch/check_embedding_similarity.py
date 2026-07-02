import sys
import cv2
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.tracking.person_tracker import PersonTracker

video_path = project_root / "test video 1.mp4"
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

# Frame 74800 (seated) and 75050 (chair)
tracker = PersonTracker(device="cuda", conf=0.20)

cap.set(cv2.CAP_PROP_POS_FRAMES, int(2992 * fps))
ret, frame_seated = cap.read()

cap.set(cv2.CAP_PROP_POS_FRAMES, int(3009 * fps))
ret, frame_chair = cap.read()

cap.release()

# Let's run tracker on both frames
persons_seated = tracker.process_frame(frame_seated, 2992.0)
p_seated = next((p for p in persons_seated if p.track_id == 2), None)
emb_seated = p_seated.visual_embedding.clone() if p_seated and p_seated.visual_embedding is not None else None

persons_chair = tracker.process_frame(frame_chair, 3009.0)
p_chair = next((p for p in persons_chair if p.track_id == 2), None)
emb_chair = p_chair.visual_embedding.clone() if p_chair and p_chair.visual_embedding is not None else None

if emb_seated is not None and emb_chair is not None:
    sim = torch.mm(emb_seated, emb_chair.t()).item()
    print(f"Similarity: {sim:.4f}")
else:
    print(f"Track 2 not found or embedding is None. Seated found: {p_seated is not None}, Chair found: {p_chair is not None}")
