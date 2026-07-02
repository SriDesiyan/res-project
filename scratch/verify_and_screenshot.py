import subprocess
import re
import cv2
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
video_path = project_root / "test video 1.mp4"
artifacts_dir = Path("C:/Users/desiy/.gemini/antigravity-ide/brain/10e189fb-80c6-4658-99ec-23929f658b3d")

slices = [
    {"num": 1, "start": 393, "end": 399},   # frame 9870
    {"num": 2, "start": 821, "end": 827},   # frame 20550
    {"num": 3, "start": 877, "end": 883},   # frame 21945
    {"num": 4, "start": 893, "end": 899},   # frame 22320
    {"num": 5, "start": 910, "end": 916},   # frame 22765
    {"num": 6, "start": 1506, "end": 1512}  # frame 37645
]

# We need the FPS to map frame_num to output video frames
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"Detected video FPS: {fps:.3f}")

for s in slices:
    print(f"\n==========================================")
    print(f"RUNNING SLICE {s['num']}: {s['start']}s to {s['end']}s")
    print(f"==========================================")
    
    out_video = project_root / f"output_slice_{s['num']}.mp4"
    cmd = [
        "python", 
        str(project_root / "analytics" / "pipeline.py"), 
        "--video", str(video_path),
        "--start", str(s["start"]),
        "--end", str(s["end"]),
        "--out", str(out_video)
    ]
    
    # Run the pipeline and capture output
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    stdout_lines = res.stdout.splitlines()
    
    # Save the pipeline run log for debug
    log_path = project_root / "scratch" / f"run_slice_{s['num']}.log"
    log_path.write_text(res.stdout, encoding="utf-8")
    print(f"Pipeline stdout/stderr saved to {log_path.name}")
    
    # Look for the verification output in stdout
    # Target format:
    # Frame Number: <num>
    # Serving Wrist Coordinate: (<x>, <y>)
    # Selected Table ID: <tid>
    # Table Centroid: (<x>, <y>)
    # Distance: <dist>
    
    events = []
    current_event = {}
    for line in stdout_lines:
        m_fn = re.match(r"^Frame Number:\s*(\d+)", line)
        if m_fn:
            current_event = {"frame_num": int(m_fn.group(1))}
            continue
        m_wrist = re.match(r"^Serving Wrist Coordinate:\s*\(([^)]+)\)", line)
        if m_wrist and current_event:
            current_event["wrist"] = m_wrist.group(1)
            continue
        m_tid = re.match(r"^Selected Table ID:\s*(\w+)", line)
        if m_tid and current_event:
            current_event["table_id"] = m_tid.group(1)
            continue
        m_cent = re.match(r"^Table Centroid:\s*\(([^)]+)\)", line)
        if m_cent and current_event:
            current_event["centroid"] = m_cent.group(1)
            continue
        m_dist = re.match(r"^Distance:\s*([\d.]+)", line)
        if m_dist and current_event:
            current_event["distance"] = float(m_dist.group(1))
            events.append(current_event)
            current_event = {}
            
    print(f"Detected {len(events)} event frames in slice {s['num']}")
    
    # If events were detected, print them and extract a screenshot
    if events:
        # Print the events as requested
        print("\n--- Event Details ---")
        for ev in events[:5]:  # print first few
            print(f"Frame Number: {ev['frame_num']}")
            print(f"Serving Wrist Coordinate: ({ev['wrist']})")
            print(f"Selected Table ID: {ev['table_id']}")
            print(f"Table Centroid: ({ev['centroid']})")
            print(f"Distance: {ev['distance']:.2f}")
            print("-" * 30)
            
        # Select the middle event frame to extract a screenshot
        target_ev = events[len(events) // 2]
        target_frame_num = target_ev["frame_num"]
        
        start_frame = int(s["start"] * fps)
        out_frame_idx = target_frame_num - start_frame
        
        print(f"Extracting frame {out_frame_idx} (global frame {target_frame_num}) from {out_video.name}")
        
        cap_out = cv2.VideoCapture(str(out_video))
        if cap_out.isOpened():
            cap_out.set(cv2.CAP_PROP_POS_FRAMES, out_frame_idx)
            ret, frame = cap_out.read()
            if ret:
                screenshot_path = artifacts_dir / f"screenshot_event_{s['num']}.png"
                cv2.imwrite(str(screenshot_path), frame)
                print(f"[OK] Saved screenshot to {screenshot_path.name}")
            else:
                print("[FAIL] Failed to read frame from output video")
            cap_out.release()
        else:
            print("[FAIL] Failed to open output video for screenshot extraction")
            
print("\nAll slices finished!")
