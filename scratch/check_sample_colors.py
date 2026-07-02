import cv2
import numpy as np
from pathlib import Path

for folder in ["waiter-sample", "server-sample", "labelled/waiter"]:
    path = Path(f"/Users/gaurisudharsinip/Desktop/wgtech/{folder}")
    if not path.exists():
        continue
    print(f"\nFolder: {folder}")
    for img_path in path.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Compute average color in BGR
                mean_color = cv2.mean(img)[:3]
                # Convert to HSV to check brightness
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mean_hsv = cv2.mean(hsv)[:3]
                print(f"  {img_path.name}: BGR={mean_color} | HSV={mean_hsv}")
