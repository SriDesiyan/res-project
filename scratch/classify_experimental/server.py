#!/usr/bin/env python3
"""Use `server-sample/` embeddings to clean and populate the labelled folders.

The script builds a ResNet50 embedding prototype from the images in
`server-sample/`. Any image in `labelled/customer` whose similarity to that
prototype is >= threshold is moved out of customer and saved into
`labelled/waiter`. The server-sample images themselves are also saved into
`labelled/waiter` when they meet the same threshold.
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
    parser = argparse.ArgumentParser(description="Clean labelled/customer using server-sample similarity")
    parser.add_argument("--sample-dir", type=Path, default=base / "server-sample")
    parser.add_argument("--waiter-dir", type=Path, default=base / "labelled" / "waiter")
    parser.add_argument("--customer-dir", type=Path, default=base / "labelled" / "customer")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--clear-output", action="store_true")
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


def best_similarity(emb: np.ndarray, refs: List[np.ndarray]) -> float:
    if not refs:
        return 0.0
    return max(cosine_similarity(emb, ref) for ref in refs)


def save_or_move_to_waiter(src: Path, waiter_dir: Path) -> Path:
    waiter_dir.mkdir(parents=True, exist_ok=True)
    destination = unique_path(waiter_dir, src.name)
    shutil.copy2(src, destination)
    return destination


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    sample_dir = args.sample_dir.resolve()
    waiter_dir = args.waiter_dir.resolve()
    customer_dir = args.customer_dir.resolve()

    sample_images = list_images(sample_dir)
    customer_images = list_images(customer_dir)

    if not sample_images:
        raise SystemExit(f"No server sample images found in: {sample_dir}")
    if not customer_images:
        raise SystemExit(f"No customer images found in: {customer_dir}")

    if args.max_images > 0:
        sample_images = sample_images[: args.max_images]

    waiter_dir.mkdir(parents=True, exist_ok=True)
    customer_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_output:
        for folder in (waiter_dir,):
            for file in folder.iterdir():
                if file.is_file():
                    file.unlink()

    print(f"Using device: {device}")
    print(f"Server samples: {len(sample_images)} from {sample_dir}")
    print(f"Customer images: {len(customer_images)} from {customer_dir}")
    print(f"Threshold: {args.threshold:.3f}")

    embedder = ResnetEmbedder(device=device)

    sample_embs = [embedder.embed_image(path) for path in sample_images]
    waiter_proto = np.mean(sample_embs, axis=0)
    waiter_proto = waiter_proto / np.linalg.norm(waiter_proto)

    sample_csv = waiter_dir.parent / "similarity_results.csv"
    waiter_count = 0
    removed_customer_count = 0

    with sample_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "similarity", "label", "action"])

        for idx, img_path in enumerate(sample_images, start=1):
            emb = embedder.embed_image(img_path)
            score = float(np.dot(emb, waiter_proto))
            if score >= args.threshold:
                save_or_move_to_waiter(img_path, waiter_dir)
                waiter_count += 1
                action = "saved_to_waiter"
                label = "waiter"
            else:
                action = "skipped"
                label = "unknown"
            writer.writerow([img_path.name, f"{score:.6f}", label, action])

            if idx % 100 == 0:
                print(f"Sample check {idx}/{len(sample_images)} | waiter_saved={waiter_count}")

    customer_csv = customer_dir.parent / "customer_similarity_results.csv"
    kept_customer = []
    with customer_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "similarity", "action"])

        for idx, img_path in enumerate(customer_images, start=1):
            emb = embedder.embed_image(img_path)
            score = best_similarity(emb, sample_embs)
            if score >= args.threshold:
                destination = unique_path(waiter_dir, img_path.name)
                shutil.copy2(img_path, destination)
                img_path.unlink()
                waiter_count += 1
                removed_customer_count += 1
                action = "moved_to_waiter_and_removed"
            else:
                kept_customer.append(img_path.name)
                action = "kept_in_customer"
            writer.writerow([img_path.name, f"{score:.6f}", action])

            if idx % 500 == 0:
                print(
                    f"Customer check {idx}/{len(customer_images)} | "
                    f"moved={removed_customer_count} kept={len(kept_customer)}"
                )

    print("Done.")
    print(f"Waiter saved from samples: {waiter_count} -> {waiter_dir}")
    print(f"Customer images removed   : {removed_customer_count} -> {customer_dir}")
    print(f"Sample CSV : {sample_csv}")
    print(f"Customer CSV: {customer_csv}")


if __name__ == "__main__":
    main()