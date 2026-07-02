import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

project_root = Path("/Users/gaurisudharsinip/Desktop/wgtech")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.tracking.person_tracker import PersonTracker

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracker = PersonTracker(device, conf=0.20)
    
    # Run the tracker sequentially up to frame 93 to preserve state
    cap = cv2.VideoCapture(str(project_root / "table_wghotel.mp4"))
    frame_num = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    target_emb = None
    while frame_num <= 93:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_time = frame_num / fps
        results = tracker.yolo.track(
            frame, classes=[0], conf=tracker.conf,
            persist=True, tracker=tracker.tracker_cfg, verbose=False
        )
        
        if frame_num == 93:
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                for box, track_id in zip(boxes, track_ids):
                    if track_id == 13:
                        x1, y1, x2, y2 = map(int, box)
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        person_crop = frame[y1:y2, x1:x2]
                        img_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                        top_img = tracker._extract_top_40(img_pil)
                        tensor = tracker.transform(top_img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            target_emb = tracker.extractor(tensor)
                            target_emb = torch.nn.functional.normalize(target_emb, p=2, dim=1)
                        break
        frame_num += 1
    cap.release()
                
    if target_emb is None:
        print("Could not find track 13 in frame 93")
        return
        
    print("Comparing Track 13 in Frame 93 to each image in labelled/waiter:")
    gallery_path = Path("/Users/gaurisudharsinip/Desktop/wgtech/labelled/waiter")
    for img_path in gallery_path.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                img = Image.open(img_path).convert("RGB")
                img = tracker._extract_top_40(img)
                tensor = tracker.transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = tracker.extractor(tensor)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    sim = torch.mm(target_emb, emb.t()).item()
                print(f"  {img_path.name}: similarity = {sim:.4f}")
            except Exception as e:
                print(f"Failed to process {img_path.name}: {e}")

if __name__ == "__main__":
    main()
