import argparse
import time
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

def parse_args():
    script_dir = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser(description="Real-time Waiter Analytics with Precomputed Embeddings")
    parser.add_argument("--video", type=Path, default=script_dir.parent / "table_wghotel.mp4", help="Input video stream")
    parser.add_argument("--out", type=Path, default=script_dir.parent / "inference_output" / "infer_out.mp4", help="Output video path")
    parser.add_argument("--waiter-emb", type=Path, default=script_dir / "waiter_average_embedding.pt", help="Precomputed waiter average embedding")
    parser.add_argument("--server-emb", type=Path, default=script_dir / "server_average_embedding.pt", help="Precomputed server average embedding")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO person detection confidence threshold")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold")
    parser.add_argument("--history", type=int, default=15, help="Frames for temporal smoothing")
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the classification head
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_top_40(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((0, 0, w, int(h * 0.4)))

def box_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def distance(c1, c2):
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5

def main():
    args = parse_args()
    device = get_device()
    print(f"Hardware Device: {device}")

    # 1. Load Precomputed Embeddings
    print("Loading Precomputed Embeddings...")
    if not args.waiter_emb.exists():
        print(f"[ERROR] Waiter embedding missing at {args.waiter_emb}. Please run waiter-embeding.py first.")
        return
    if not args.server_emb.exists():
        print(f"[ERROR] Server embedding missing at {args.server_emb}. Please run server-embeddings.py first.")
        return
        
    waiter_avg_emb = torch.load(args.waiter_emb, map_location=device)
    server_avg_emb = torch.load(args.server_emb, map_location=device)
    
    # Ensure they are normalized
    waiter_avg_emb = torch.nn.functional.normalize(waiter_avg_emb, p=2, dim=1)
    server_avg_emb = torch.nn.functional.normalize(server_avg_emb, p=2, dim=1)

    # 2. Initialize ResNet50 Embedding Extractor
    print("Initializing ResNet50 Extractor...")
    extractor = FeatureExtractor().to(device)
    extractor.eval()
    transform = get_transforms()

    # 3. Initialize YOLOv8 Person Detector
    print("Loading YOLOv8n Person Detector...")
    script_dir = Path(__file__).parent.resolve()
    yolo_model = YOLO(str(script_dir.parent / "yolov8n.pt")) # Using the model file from root directory

    # 4. Setup Video Stream
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video {args.video}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))

    # State tracking: Track ID -> state info
    track_states = defaultdict(lambda: {"consecutive_hits": 0})
    locked_waiters = set()
    # Spatial handover: remember where locked waiters were last seen
    # key = track_id, value = {"center": (cx, cy), "frame": int}
    waiter_last_seen = {}
    HANDOVER_MAX_FRAMES = 15   # look back ~0.5 seconds only
    HANDOVER_MAX_DIST = 50     # pixels — tight to prevent spreading to nearby customers
    frame_count = 0

    print("\n--- Starting Real-Time Video Inference ---")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO Detection & Tracking with ReID-enabled BoT-SORT (DeepSORT-like)
        tracker_cfg = str(Path(__file__).parent.resolve() / "tracker_config.yaml")
        results = yolo_model.track(frame, classes=[0], conf=args.conf, persist=True, tracker=tracker_cfg, verbose=False)
        annotated_frame = frame.copy()

        active_track_ids = set()
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                active_track_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                center = box_center(x1, y1, x2, y2)

                # Spatial Handover: if this is a NEW track near where a locked waiter was last seen, inherit the lock
                if track_id not in locked_waiters:
                    for old_id, info in list(waiter_last_seen.items()):
                        if old_id not in active_track_ids and (frame_count - info["frame"]) < HANDOVER_MAX_FRAMES:
                            if distance(center, info["center"]) < HANDOVER_MAX_DIST:
                                locked_waiters.add(track_id)
                                print(f"[HANDOVER] Transferred waiter lock from ID:{old_id} -> ID:{track_id} (dist={distance(center, info['center']):.0f}px)")
                                del waiter_last_seen[old_id]
                                break

                # Check locked fast path
                if track_id in locked_waiters:
                    # Update last seen position for this locked waiter
                    waiter_last_seen[track_id] = {"center": center, "frame": frame_count}
                    # Fast path visualization — skip all embedding computation
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    color = (0, 0, 255)
                    text = f"ID:{track_id} WAITER (LOCKED)"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                    cv2.putText(annotated_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    continue
                
                # Prevent out of bounds errors
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                # Crop person from frame
                person_crop = frame[y1:y2, x1:x2]
                img_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                
                # Extract Top 40% and compute embedding
                top_img = extract_top_40(img_pil)
                tensor = transform(top_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    emb = extractor(tensor)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    
                    # Compute Cosine Similarity against both average embeddings
                    sim_waiter = torch.mm(emb, waiter_avg_emb.t()).item()
                    sim_server = torch.mm(emb, server_avg_emb.t()).item()
                    max_sim = max(sim_waiter, sim_server)

                # Threshold-based decision with hysteresis
                if max_sim > args.threshold:
                    track_states[track_id]["consecutive_hits"] += 1
                else:
                    track_states[track_id]["consecutive_hits"] = max(0, track_states[track_id]["consecutive_hits"] - 1)

                if track_states[track_id]["consecutive_hits"] >= 2:
                    locked_waiters.add(track_id)
                    waiter_last_seen[track_id] = {"center": center, "frame": frame_count}
                    smoothed_label = "waiter"
                else:
                    smoothed_label = "customer"

                # Visualization: Red for Waiter, Green for Customer (BGR format)
                color = (0, 0, 255) if smoothed_label == "waiter" else (0, 255, 0)
                text = f"ID:{track_id} {smoothed_label.upper()} ({max_sim:.2f})"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(annotated_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Cleanup intentionally removed to allow YOLO/BoT-SORT to persist IDs across occlusions
        # FPS Calculation
        end_time = time.time()
        process_fps = 1.0 / (end_time - start_time)
        cv2.putText(annotated_frame, f"FPS: {process_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        writer.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    writer.release()
    print(f"\nPipeline Complete. Output saved to: {args.out}")

if __name__ == "__main__":
    main()
