import cv2

video_path = "/Users/gaurisudharsinip/Desktop/wgtech/inference_output/analytics_out.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Cannot open output video!")
    exit(1)

# Go to frame 93
cap.set(cv2.CAP_PROP_POS_FRAMES, 93)
ret, frame = cap.read()
if ret:
    cv2.imwrite("/Users/gaurisudharsinip/Desktop/wgtech/scratch/frame_93.jpg", frame)
    print("Successfully saved frame 93 to scratch/frame_93.jpg")
else:
    print("Failed to read frame 93")
cap.release()
