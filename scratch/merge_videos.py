import cv2
import sys

def merge_videos(parts, output_path):
    print(f"Merging {parts} into {output_path}...")
    
    # Read the first video to get the properties
    cap = cv2.VideoCapture(parts[0])
    if not cap.isOpened():
        print(f"Error opening {parts[0]}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    frame_count = 0
    for part in parts:
        print(f"Reading {part}...")
        cap = cv2.VideoCapture(part)
        if not cap.isOpened():
            print(f"Error opening {part}")
            continue
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_count += 1
            if frame_count % 1000 == 0:
                print(f"Written {frame_count} frames...")
        cap.release()
        
    writer.release()
    print(f"Finished! Total frames: {frame_count}. Saved to {output_path}")

if __name__ == "__main__":
    parts = ["output_analytics_part1.mp4", "output_analytics_part2.mp4", "output_analytics_part3.mp4"]
    merge_videos(parts, "output_analytics.mp4")
