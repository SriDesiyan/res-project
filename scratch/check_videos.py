import cv2

for video in ["table_wghotel.mp4", "diner.mp4", "new.mp4"]:
    cap = cv2.VideoCapture(video)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{video}: {w}x{h}, {fps} FPS, {frames} frames")
    else:
        print(f"Cannot open {video}")
    cap.release()
