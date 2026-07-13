#!/usr/bin/env python3
"""Extract bow-server-like images using ResNet50 embeddings.

The script builds an embedding prototype from the images in `server-sample/` and
compares candidate images against that prototype. Images whose similarity is at
or above the threshold are copied into `labelled/bow-server/`.

Default source folder:
- `labelled/customer/`

Default output folder:
- `labelled/bow-server/`
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import ssl
from pathlib import Path
from typing import List

import certifi
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models import ResNet50_Weights, resnet50


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Extract bow-server-like images using embedding similarity")
    parser.add_argument("--sample-dir", type=Path, default=base / "server-sample")
    parser.add_argument("--source-dir", type=Path, default=base / "labelled" / "customer")
    parser.add_argument("--output-dir", type=Path, default=base / "labelled" / "bow-server")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--move-source", action="store_true", help="Move matched images instead of copying them")
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
    """Return a unique .jpg path for the given source image."""
    return unique_path(folder, f"{source_path.stem}.jpg")


def extract_top_40(img: Image.Image) -> Image.Image:
    """Focus on the upper body region where the bow/server uniform is most visible."""
    width, height = img.size
    return img.crop((0, 0, width, int(height * 0.4)))


class ResnetEmbedder:
    def __init__(self, device: str):
        self.device = device
        self.weights = ResNet50_Weights.DEFAULT

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        ssl._create_default_https_context = lambda: ssl.create_default_context(  # type: ignore[assignment]
            cafile=certifi.where()
        )

        model = resnet50(weights=self.weights)
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
        self.backbone.eval()
        self.transforms = self.weights.transforms()

    def embed_image(self, image_path: Path) -> np.ndarray:
        try:
            img = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Cannot read image: {image_path}") from exc

        img = extract_top_40(img)
        x = self.transforms(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.backbone(x).flatten(start_dim=1)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat.squeeze(0).cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def save_image(src: Path, output_dir: Path, move_source: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = jpeg_path(output_dir, src)
    image = Image.open(src).convert("RGB")
    image.save(destination, format="JPEG", quality=95, optimize=True)
    if move_source:
        src.unlink()
    return destination


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    sample_dir = args.sample_dir.resolve()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    sample_images = list_images(sample_dir)
    source_images = list_images(source_dir)

    if not sample_images:
        raise SystemExit(f"No bow server sample images found in: {sample_dir}")
    if not source_images:
        raise SystemExit(f"No source images found in: {source_dir}")

    if args.max_images > 0:
        source_images = source_images[: args.max_images]

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_output:
        for file in output_dir.iterdir():
            if file.is_file():
                file.unlink()

    print(f"Using device: {device}")
    print(f"Sample images : {len(sample_images)} from {sample_dir}")
    print(f"Source images : {len(source_images)} from {source_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Threshold     : {args.threshold:.3f}")
    print(f"Action        : {'move' if args.move_source else 'copy'} matched images")

    embedder = ResnetEmbedder(device=device)

    sample_embs = [embedder.embed_image(path) for path in sample_images]
    bow_proto = np.mean(sample_embs, axis=0)
    bow_proto = bow_proto / np.linalg.norm(bow_proto)

    csv_path = output_dir / "bow_server_similarity_results.csv"
    matched_count = 0
    skipped_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "similarity", "label", "action", "saved_path"])

        for idx, img_path in enumerate(source_images, start=1):
            try:
                emb = embedder.embed_image(img_path)
            except RuntimeError as exc:
                print(f"[SKIP] {exc}")
                skipped_count += 1
                continue

            score = float(np.dot(emb, bow_proto))
            is_match = score >= args.threshold
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

            writer.writerow([img_path.name, f"{score:.6f}", label, action, saved_path])

            if idx % 250 == 0:
                print(f"Processed {idx}/{len(source_images)} | matched={matched_count} skipped={skipped_count}")

    print("Done.")
    print(f"Matched images : {matched_count} -> {output_dir}")
    print(f"Skipped images  : {skipped_count}")
    print(f"CSV report      : {csv_path}")


if __name__ == "__main__":
    main()