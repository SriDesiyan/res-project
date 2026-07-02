import cv2
import numpy as np
from pathlib import Path

def analyze_crop(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_crop, w_crop = hsv.shape[:2]
    aspect_ratio = float(h_crop) / float(w_crop) if w_crop > 0 else 0.0
    
    x_slice_start = int(w_crop * 0.20)
    x_slice_end = int(w_crop * 0.80)
    upper_body = hsv[int(h_crop * 0.15):int(h_crop * 0.45), x_slice_start:x_slice_end]
    lower_body = hsv[int(h_crop * 0.55):int(h_crop * 0.85), x_slice_start:x_slice_end]
    
    if upper_body.size == 0 or lower_body.size == 0:
        print(f"  {img_path.name}: Empty regions")
        return
        
    mean_upper = cv2.mean(upper_body)[:3]
    mean_lower = cv2.mean(lower_body)[:3]
    
    upper_s, upper_v = mean_upper[1], mean_upper[2]
    lower_s, lower_v = mean_lower[1], mean_lower[2]
    
    # White pixels: V > 140, S < 75 (slightly relaxed for shadows)
    white_mask = (upper_body[:, :, 2] > 140) & (upper_body[:, :, 1] < 75)
    upper_white_pct = np.mean(white_mask) * 100.0
    
    # Black pixels: V < 125
    black_mask = lower_body[:, :, 2] < 125
    lower_black_pct = np.mean(black_mask) * 100.0
    
    # Rule checks
    diff_v = upper_v - lower_v
    
    # A waiter uniform:
    # 1. Aspect ratio >= 1.5
    aspect_ok = aspect_ratio >= 1.5
    # 2. Upper body is not highly saturated (shirt is white, not colored)
    sat_ok = upper_s < 80
    # 3. Upper body has white-shirt properties
    shirt_ok = (upper_v >= 135) or (upper_white_pct >= 20.0)
    # 4. Lower body has dark-pants properties
    pants_ok = (lower_v < 125) or (lower_black_pct >= 50.0)
    # 5. Contrast check: upper body is brighter than lower body
    contrast_ok = diff_v > 20
    
    is_waiter = aspect_ok and sat_ok and shirt_ok and pants_ok and contrast_ok
    
    print(f"  {img_path.name}:")
    print(f"    Aspect: {aspect_ratio:.2f} (ok={aspect_ok}) | Upper S: {upper_s:.1f} (ok={sat_ok})")
    print(f"    Upper V: {upper_v:.1f}, White%: {upper_white_pct:.1f}% (ok={shirt_ok})")
    print(f"    Lower V: {lower_v:.1f}, Black%: {lower_black_pct:.1f}% (ok={pants_ok})")
    print(f"    Diff V: {diff_v:.1f} (ok={contrast_ok})")
    print(f"    => IS WAITER: {is_waiter}")

print("--- WAITER SAMPLES ---")
waiter_path = Path("/Users/gaurisudharsinip/Desktop/wgtech/waiter-sample")
for p in waiter_path.iterdir():
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        analyze_crop(p)

print("\n--- SERVER SAMPLES ---")
server_path = Path("/Users/gaurisudharsinip/Desktop/wgtech/server-sample")
for p in server_path.iterdir():
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        analyze_crop(p)
