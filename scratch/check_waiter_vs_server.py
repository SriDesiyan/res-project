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
    
    while frame_num < 150:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_time = frame_num / fps
        h, w = frame.shape[:2]
        
        results = tracker.yolo.track(
            frame, classes=[0], conf=tracker.conf,
            persist=True, tracker=tracker.tracker_cfg, verbose=False
        )
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                if track_id == 13:
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    person_crop = frame[y1:y2, x1:x2]
                    is_uniform = tracker._has_waiter_uniform(person_crop)
                    
                    from PIL import Image
                    img_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                    top_img = tracker._extract_top_40(img_pil)
                    tensor = tracker.transform(top_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        emb = tracker.extractor(tensor)
                        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                        
                        waiter_sim = torch.mm(emb, tracker.waiter_emb.t()).item() if tracker.waiter_emb is not None else 0.0
                        server_sim = torch.mm(emb, tracker.server_emb.t()).item() if tracker.server_emb is not None else 0.0
                        
                    print(f"Frame {frame_num} ({frame_time:.2f}s): Track 13 | waiter_sim={waiter_sim:.3f} | server_sim={server_sim:.3f} | is_uniform={is_uniform}")
                    
        frame_num += 1
        
    cap.release()

if __name__ == "__main__":
    main()
