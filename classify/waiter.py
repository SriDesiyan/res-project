#!/usr/bin/env python3
"""
Similarity-based waiter segregation.
Compares each image in `unlabeled/` with reference images in `waiter-sample/`
using embeddings from a pretrained torchvision ResNet50 model.
Images with cosine similarity >= threshold are copied to waiter output,
others are copied to customer output.
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
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Segregate unlabeled images into waiter/customer using similarity"
    )
    parser.add_argument(
        "--waiter-dir",
        type=Path,
        default=base / "waiter-sample",
        help="Directory with sample waiter images",
    )
    parser.add_argument(
        "--unlabeled-dir",
        type=Path,
        default=base / "unlabeled",
        help="Directory with candidate images to classify",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "similarity-output2",
        help="Directory where segregated outputs are written",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for waiter classification",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional limit for quick tests (0 = process all)",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear waiter/customer output folders before writing",
    )
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
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


class ResnetEmbedder:
    def __init__(self, device: str):
        self.device = device
        self.weights = ResNet50_Weights.DEFAULT

        # Ensure urllib uses certifi CA bundle to avoid SSL trust issues.
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    waiter_dir = args.waiter_dir.resolve()
    unlabeled_dir = args.unlabeled_dir.resolve()
    output_dir = args.output_dir.resolve()

    waiter_images = list_images(waiter_dir)
    candidate_images = list_images(unlabeled_dir)

    if not waiter_images:
        raise SystemExit(f"No waiter sample images found in: {waiter_dir}")
    if not candidate_images:
        raise SystemExit(f"No unlabeled images found in: {unlabeled_dir}")

    if args.max_images > 0:
        candidate_images = candidate_images[: args.max_images]

    waiter_out = output_dir / "waiter"
    customer_out = output_dir / "customer"
    waiter_out.mkdir(parents=True, exist_ok=True)
    customer_out.mkdir(parents=True, exist_ok=True)

    if args.clear_output:
        for folder in (waiter_out, customer_out):
            for file in folder.iterdir():
                if file.is_file():
                    file.unlink()

    print(f"Using device: {device}")
    print(f"Waiter samples: {len(waiter_images)} from {waiter_dir}")
    print(f"Candidates: {len(candidate_images)} from {unlabeled_dir}")
    print(f"Threshold: {args.threshold:.3f}")

    embedder = ResnetEmbedder(device=device)

    sample_embs = [embedder.embed_image(p) for p in waiter_images]
    waiter_proto = np.mean(sample_embs, axis=0)
    waiter_proto = waiter_proto / np.linalg.norm(waiter_proto)

    csv_path = output_dir / "similarity_results.csv"
    waiter_count = 0
    customer_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "similarity", "label"])

        for idx, img_path in enumerate(candidate_images, start=1):
            emb = embedder.embed_image(img_path)
            score = float(np.dot(emb, waiter_proto))
            is_waiter = score >= args.threshold
            label = "waiter" if is_waiter else "customer"

            out_dir = waiter_out if is_waiter else customer_out
            shutil.copy2(img_path, out_dir / img_path.name)

            if is_waiter:
                waiter_count += 1
            else:
                customer_count += 1

            writer.writerow([img_path.name, f"{score:.6f}", label])

            if idx % 500 == 0:
                print(
                    f"Processed {idx}/{len(candidate_images)} | "
                    f"waiter={waiter_count} customer={customer_count}"
                )

    print("Done.")
    print(f"Waiter matches: {waiter_count} -> {waiter_out}")
    print(f"Customer matches: {customer_count} -> {customer_out}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()