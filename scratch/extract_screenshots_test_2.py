import cv2
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
video_path = project_root / "output_example_test_2.mp4"
artifacts_dir = Path("C:/Users/desiy/.gemini/antigravity-ide/brain/7d2a584d-a1bd-4d84-b5f9-c7c54e7a861d")
artifacts_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("Cannot open output video!")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)

# State timestamps to capture (in seconds)
captures = {
    "occupancy.png": 65.0,        # Occupied
    "food_served.png": 144.5,     # Dining + Blinking Food Served Banner
    "dirty.png": 1150.0,          # Dirty state
    "clean.png": 1160.0,          # Clean state
}

for filename, sec in captures.items():
    frame_idx = int(sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        out_path = artifacts_dir / filename
        cv2.imwrite(str(out_path), frame)
        print(f"[OK] Saved {filename} at {sec}s (frame {frame_idx})")
    else:
        print(f"[FAIL] Could not capture frame at {sec}s")

cap.release()
