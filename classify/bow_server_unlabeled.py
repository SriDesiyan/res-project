#!/usr/bin/env python3
"""Extract bow-server frames from `unlabeled/` using visual cues only.

This version saves a frame only when a detected person crop shows both:
- a white shirt in the upper torso region
- a black plate-like object in the lower hand region

Matched frames are copied into `labelled/bow-server/` as JPEGs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Extract bow-server-like frames from unlabeled video frames")
    parser.add_argument("--source-dir", type=Path, default=base / "unlabeled")
    parser.add_argument("--output-dir", type=Path, default=base / "labelled" / "bow-server")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--move-source", action="store_true", help="Move matched frames instead of copying them")
    parser.add_argument("--person-confidence", type=float, default=0.70)
    parser.add_argument("--person-area", type=float, default=0.12)
    parser.add_argument("--shirt-white-threshold", type=float, default=0.08)
    parser.add_argument("--plate-black-threshold", type=float, default=0.10)
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def unique_path(folder: Path, filename: str) -> Path:
    target = folder / filename
    if not target.exists():
        return target
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def jpeg_path(folder: Path, source_path: Path) -> Path:
    return unique_path(folder, f"{source_path.stem}.jpg")


def dark_pixel_ratio(img: Image.Image, threshold: int = 70) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    dark = (arr[:, :, 0] < threshold) & (arr[:, :, 1] < threshold) & (arr[:, :, 2] < threshold)
    return float(dark.mean())


def white_pixel_ratio(img: Image.Image, brightness: int = 190, chroma: int = 35) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    bright = arr.mean(axis=2) >= brightness
    neutral = (arr.max(axis=2) - arr.min(axis=2)) <= chroma
    return float((bright & neutral).mean())


def load_person_detector(device: str) -> YOLO:
    detector = YOLO("yolov8n.pt")
    if device in {"cpu", "cuda", "mps"}:
        detector.to(device)
    return detector


def person_view(img: Image.Image, detector: YOLO) -> Tuple[Image.Image, float, float]:
    results = detector.predict(img, verbose=False)
    if not results:
        return img, 0.0, 0.0

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return img, 0.0, 0.0

    best_box = None
    best_conf = 0.0
    for box in boxes:
        if int(box.cls.item()) != 0:
            continue
        conf = float(box.conf.item())
        if conf > best_conf:
            best_box = box
            best_conf = conf

    if best_box is None:
        return img, 0.0, 0.0

    left, top, right, bottom = map(int, best_box.xyxy[0].tolist())
    left = max(0, left)
    top = max(0, top)
    right = min(img.width, right)
    bottom = min(img.height, bottom)
    if right <= left or bottom <= top:
        return img, 0.0, 0.0

    area = ((right - left) * (bottom - top)) / float(img.width * img.height)
    return img.crop((left, top, right, bottom)), best_conf, area


def save_image(src: Path, output_dir: Path, move_source: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = jpeg_path(output_dir, src)
    image = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    image.save(destination, format="JPEG", quality=95, optimize=True)
    if move_source:
        src.unlink()
    return destination


def analyze_person_crop(person_img: Image.Image) -> Tuple[float, float]:
    shirt_region = person_img.crop(
        (
            int(person_img.width * 0.22),
            int(person_img.height * 0.18),
            int(person_img.width * 0.78),
            int(person_img.height * 0.58),
        )
    )
    plate_region = person_img.crop(
        (
            int(person_img.width * 0.18),
            int(person_img.height * 0.48),
            int(person_img.width * 0.82),
            int(person_img.height * 0.92),
        )
    )
    return white_pixel_ratio(shirt_region), dark_pixel_ratio(plate_region)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_images = list_images(source_dir)

    if not source_images:
        raise SystemExit(f"No source frames found in: {source_dir}")

    if args.max_images > 0:
        source_images = source_images[: args.max_images]

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_output:
        for file in output_dir.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                file.unlink()

    print(f"Using device: {device}")
    print(f"Source frames  : {len(source_images)} from {source_dir}")
    print(f"Output dir    : {output_dir}")
    print(
        "Rule          : person_conf>="
        f"{args.person_confidence:.2f} and area>={args.person_area:.2f} and "
        f"shirt_white>={args.shirt_white_threshold:.2f} and plate_black>={args.plate_black_threshold:.2f}"
    )
    print(f"Action        : {'move' if args.move_source else 'copy'} matched frames")

    detector = load_person_detector(device)

    csv_path = output_dir / "bow_server_unlabeled_similarity_results.csv"
    matched_count = 0
    skipped_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "person_confidence",
            "person_area",
            "shirt_white_score",
            "plate_black_score",
            "reason",
            "label",
            "action",
            "saved_path",
        ])

        for idx, img_path in enumerate(source_images, start=1):
            try:
                full_img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError) as exc:
                print(f"[SKIP] Cannot read image: {img_path} ({exc})")
                skipped_count += 1
                continue

            person_img, person_conf, person_area = person_view(full_img, detector)
            shirt_white_score, plate_black_score = analyze_person_crop(person_img)

            shirt_match = shirt_white_score >= args.shirt_white_threshold
            plate_match = plate_black_score >= args.plate_black_threshold
            person_match = person_conf >= args.person_confidence and person_area >= args.person_area
            is_match = person_match and shirt_match and plate_match

            reasons = []
            if person_match:
                reasons.append("person")
            if shirt_match:
                reasons.append("white-shirt")
            if plate_match:
                reasons.append("black-plate")
            reason_text = "+".join(reasons) if reasons else "none"

            action = "skipped"
            saved_path = ""

            if is_match:
                destination = save_image(img_path, output_dir, move_source=args.move_source)
                matched_count += 1
                action = "moved" if args.move_source else "copied"
                saved_path = str(destination)
                label = "bow-server"
            else:
                label = "unknown"
                skipped_count += 1

            writer.writerow([
                img_path.name,
                f"{person_conf:.6f}",
                f"{person_area:.6f}",
                f"{shirt_white_score:.6f}",
                f"{plate_black_score:.6f}",
                reason_text,
                label,
                action,
                saved_path,
            ])

            if idx % 250 == 0:
                print(f"Processed {idx}/{len(source_images)} | matched={matched_count} skipped={skipped_count} | reason={reason_text}")

    print("Done.")
    print(f"Matched frames : {matched_count} -> {output_dir}")
    print(f"Skipped frames  : {skipped_count}")
    print(f"CSV report      : {csv_path}")


if __name__ == "__main__":
    main()