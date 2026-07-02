#!/usr/bin/env python3
"""Run the bow-server embedding pipeline on a video and save results.

This script:
1. Loads the bow-server reference from `bow-server-sample/`
2. Scans frames from `diner.mp4`
3. Scores each frame with ResNet50 cosine similarity
4. Saves an annotated output video
5. Copies matched frames into `labelled/bow-server/frames/`
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Bow-server video pipeline for diner.mp4")
    parser.add_argument("--video", type=Path, default=project_root / "diner.mp4")
    parser.add_argument("--sample-dir", type=Path, default=project_root / "bow-server-sample")
    parser.add_argument("--output-video", type=Path, default=project_root / "inference_output" / "diner_bow_server.mp4")
    parser.add_argument("--frames-dir", type=Path, default=project_root / "labelled" / "bow-server" / "frames")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Process only the first N seconds (0 = full video)")
    parser.add_argument("--frame-step", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--clear-frames", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def build_model(device: str) -> torch.nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    model.eval()
    return model


TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def extract_embedding(model: torch.nn.Module, img: Image.Image, device: str) -> torch.Tensor:
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        feat = model(tensor).flatten(start_dim=1)
        feat = F.normalize(feat, p=2, dim=1)
    return feat.squeeze(0)


def build_reference(model: torch.nn.Module, device: str, sample_dir: Path) -> torch.Tensor:
    sample_files = list_images(sample_dir)
    if not sample_files:
        raise FileNotFoundError(f"No sample images found in '{sample_dir}'")

    embeddings = []
    print(f"[INFO] Using {len(sample_files)} bow-server sample image(s):")
    for sample_path in sample_files:
        print(f"       • {sample_path.name}")
        embeddings.append(extract_embedding(model, Image.open(sample_path), device))

    reference = torch.stack(embeddings).mean(dim=0)
    reference = F.normalize(reference, p=2, dim=0)
    print(f"[INFO] Reference embedding built (dim={reference.shape[0]})\n")
    return reference


def annotate_frame(frame: np.ndarray, score: float, threshold: float, label: str) -> np.ndarray:
    annotated = frame.copy()
    color = (0, 200, 0) if label == "BOW-SERVER" else (0, 0, 200)
    text = f"{label} | sim={score:.3f} | thr={threshold:.2f}"
    cv2.rectangle(annotated, (12, 12), (12 + 20 + len(text) * 10, 58), (0, 0, 0), -1)
    cv2.putText(annotated, text, (22, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return annotated


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    video_path = args.video.resolve()
    sample_dir = args.sample_dir.resolve()
    output_video = args.output_video.resolve()
    frames_dir = args.frames_dir.resolve()

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_frames:
        for file in frames_dir.iterdir():
            if file.is_file() and file.suffix.lower() in IMG_EXTENSIONS:
                file.unlink()

    print(f"Using device: {device}")
    print(f"Video       : {video_path}")
    print(f"Sample dir  : {sample_dir}")
    print(f"Frames dir  : {frames_dir}")
    print(f"Output video: {output_video}")
    print(f"Threshold   : {args.threshold:.3f}")
    print(f"Frame step  : {args.frame_step}")

    model = build_model(device)
    reference = build_reference(model, device, sample_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.max_seconds > 0:
        max_frames = min(total_frames, int(args.max_seconds * fps))
    else:
        max_frames = total_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    matched_count = 0
    processed_count = 0
    frame_idx = 0

    print(f"[INFO] Processing up to {max_frames} frames ...\n")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_step != 0:
            writer.write(frame)
            frame_idx += 1
            continue

        processed_count += 1

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        embedding = extract_embedding(model, img, device)
        score = float(torch.dot(reference, embedding).item())
        is_match = score >= args.threshold
        label = "BOW-SERVER" if is_match else "other"

        if is_match:
            matched_count += 1
            frame_name = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frames_dir / frame_name), frame)

        annotated = annotate_frame(frame, score, args.threshold, label)
        writer.write(annotated)

        if processed_count % 250 == 0:
            print(f"Processed {processed_count} scored frames | matched={matched_count}")

        frame_idx += 1

    cap.release()
    writer.release()

    print("\nDone.")
    print(f"Processed frames : {processed_count}")
    print(f"Matched frames   : {matched_count}")
    print(f"Saved video      : {output_video}")
    print(f"Saved frames dir : {frames_dir}")


if __name__ == "__main__":
    main()
