import cv2
from pathlib import Path

video_path = Path("c:/Users/desiy/Downloads/coe-intern-main (1)/coe-intern-main/test video 1.mp4")
cap = cv2.VideoCapture(str(video_path))
if cap.isOpened():
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    print(f"Video: {video_path.name}")
    print(f"Resolution: {w}x{h}")
    print(f"FPS: {fps}")
    print(f"Frames: {frames}")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
else:
    print("Cannot open test video 1.mp4")
cap.release()
