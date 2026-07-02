import cv2
import numpy as np
import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.tracking.person_tracker import PersonTracker
from analytics.tracking.serving_detector import detect_food_in_frame, detect_waiter_serving

def main():
    img_path = project_root / "test_server.png"
    img = cv2.imread(str(img_path))
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracker = PersonTracker(device, conf=0.20)
    
    results = tracker.yolo(img, classes=[0], verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_crop, w_crop = hsv.shape[:2]
        
        aspect_ratio = float(h_crop) / float(w_crop) if w_crop > 0 else 0.0
        
        x_slice_start = int(w_crop * 0.20)
        x_slice_end = int(w_crop * 0.80)
        upper_body = hsv[int(h_crop * 0.15):int(h_crop * 0.45), x_slice_start:x_slice_end]
        lower_body = hsv[int(h_crop * 0.55):int(h_crop * 0.85), x_slice_start:x_slice_end]
        
        if upper_body.size == 0 or lower_body.size == 0:
            continue
            
        mean_upper = cv2.mean(upper_body)[:3]
        mean_lower = cv2.mean(lower_body)[:3]
        upper_s, upper_v = mean_upper[1], mean_upper[2]
        lower_s, lower_v = mean_lower[1], mean_lower[2]
        
        white_pixels = (upper_body[:, :, 2] > 165) & (upper_body[:, :, 1] < 70)
        white_pct = np.mean(white_pixels) * 100.0
        
        black_pixels = (lower_body[:, :, 2] < 115)
        black_pct = np.mean(black_pixels) * 100.0
        
        shirt_ok = (upper_v >= 150) and (((upper_v > 165) and (upper_s < 70)) or ((white_pct >= 20.0) and (upper_s < 85)))
        pants_ok = (black_pct >= 50.0) or (lower_v < 115)
        
        print(f"Person {i}: bbox=[{x1},{y1},{x2},{y2}] aspect={aspect_ratio:.2f}")
        print(f"  Upper V={upper_v:.1f} S={upper_s:.1f} white_pct={white_pct:.1f}% -> shirt_ok={shirt_ok}")
        print(f"  Lower V={lower_v:.1f} S={lower_s:.1f} black_pct={black_pct:.1f}% -> pants_ok={pants_ok}")
        print(f"  Is Waiter: {aspect_ratio >= 1.5 and shirt_ok and pants_ok}")

if __name__ == "__main__":
    main()
