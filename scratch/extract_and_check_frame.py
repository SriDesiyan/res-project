import cv2
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.tracking.person_tracker import PersonTracker

def main():
    cap = cv2.VideoCapture(str(project_root / "new.mp4"))
    # We want to check around frame 7125 (285 seconds)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_frame = int(285.0 * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        cap.release()
        return

    print(f"Read frame {target_frame} from new.mp4")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracker = PersonTracker(device, conf=0.20)
    
    # Run YOLO detection
    results = tracker.yolo(frame, classes=[0], conf=0.20, verbose=False)
    
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # The waiter is on the left side of the screen (e.g. x1 < 600)
            if x1 < 600:
                crop = frame[y1:y2, x1:x2]
                h_crop, w_crop = crop.shape[:2]
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                y1_up, y2_up = int(h_crop * 0.15), int(h_crop * 0.45)
                y1_low, y2_low = int(h_crop * 0.55), int(h_crop * 0.85)
                
                x_slice_start = int(w_crop * 0.20)
                x_slice_end = int(w_crop * 0.80)
                upper_body = hsv[y1_up:y2_up, x_slice_start:x_slice_end]
                lower_body = hsv[y1_low:y2_low, x_slice_start:x_slice_end]
                
                mean_upper = cv2.mean(upper_body)[:3] if upper_body.size > 0 else (0,0,0)
                mean_lower = cv2.mean(lower_body)[:3] if lower_body.size > 0 else (0,0,0)
                upper_s = mean_upper[1]
                upper_v = mean_upper[2]
                lower_v = mean_lower[2]
                
                # Check pixel percentage for white shirt: V > 165 and S < 70
                white_pixels = (upper_body[:, :, 2] > 165) & (upper_body[:, :, 1] < 70)
                white_pct = np.mean(white_pixels) * 100.0 if upper_body.size > 0 else 0.0
                
                # Check pixel percentage for black pants: V < 115
                black_pixels = (lower_body[:, :, 2] < 115)
                black_pct = np.mean(black_pixels) * 100.0 if lower_body.size > 0 else 0.0
                
                # Slicing logic applied to tracker's method to test
                aspect_ratio = float(h_crop) / float(w_crop)
                shirt_ok = (upper_v >= 150) and (((upper_v > 165) and (upper_s < 70)) or ((white_pct >= 20.0) and (upper_s < 85)))
                pants_ok = (black_pct >= 50.0) or (lower_v < 115)
                is_uniform_test = (aspect_ratio >= 1.5) and shirt_ok and pants_ok
                
                print(f"Candidate {i} on Left: bbox=[{x1},{y1},{x2},{y2}] size={w_crop}x{h_crop}")
                print(f"  Aspect ratio: {aspect_ratio:.2f}")
                print(f"  Mean Upper V={upper_v:.1f}, S={upper_s:.1f} | Lower V={lower_v:.1f}")
                print(f"  White shirt pixel fraction (V>165, S<70): {white_pct:.1f}%")
                print(f"  Black pants pixel fraction (V<115): {black_pct:.1f}%")
                print(f"  Passed 60% Width Test? {is_uniform_test}")
                print(f"  Passed tracker._has_waiter_uniform? {tracker._has_waiter_uniform(crop)}")
                
                # Let's write out this candidate crop and the annotated frame
                out_dir = project_root / "inference_output" / "waiter_debug"
                out_dir.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(out_dir / f"candidate_{i}_crop.jpg"), crop)
                
                # Draw box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.rectangle(frame, (x1, y1 + y1_up), (x1 + w_crop, y1 + y2_up), (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1 + y1_low), (x1 + w_crop, y1 + y2_low), (0, 0, 0), 2)
                cv2.imwrite(str(out_dir / "debug_frame.jpg"), frame)
                print(f"Saved debug images to {out_dir}")

    cap.release()

if __name__ == "__main__":
    main()
