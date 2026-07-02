import sys
from pathlib import Path
import cv2
import numpy as np

# Adjust python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))
sys.path.insert(0, str(project_root / "analytics" / "roi"))

from auto_roi import AutoTableDetector
from roi_config import save_tables

import argparse

def main():
    parser = argparse.ArgumentParser(description="Test ROI Table detection on a single image")
    parser.add_argument("--image", type=str, default="sddefault.jpg", help="Path to input image (e.g. sddefault.jpg, 4.webp)")
    parser.add_argument("--model", type=str, default="yolov8m-seg.pt", help="YOLOv8 segmentation model (default: yolov8m-seg.pt)")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default: 0.15)")
    parser.add_argument("--method", type=str, default="aruco", choices=["aruco", "yolo"], help="Detection method: 'aruco' (default) or 'yolo'")
    args = parser.parse_args()

    input_path = Path(args.image)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    print(f"Loading {input_path}...")
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    # Load image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Error: OpenCV could not read {input_path}")
        return

    # If it is a webp, let's also write a jpg copy to help verify conversion if needed
    if input_path.suffix.lower() == ".webp":
        jpg_path = input_path.with_suffix(".jpg")
        cv2.imwrite(str(jpg_path), img)
        print(f"Saved a JPG copy to {jpg_path}")

    # Set output paths dynamically based on image name
    base_name = input_path.stem
    out_json = project_root / "analytics" / "config" / f"tables_{base_name}.json"
    out_preview = project_root / f"{base_name}_detected.jpg"

    # Run auto-roi detection
    # We will use the model, method, and confidence threshold requested
    detector = AutoTableDetector(model_name=args.model, conf=args.conf)
    print(f"Running table detection using method={args.method}, model={args.model} at conf={args.conf}...")
    tables = detector.detect(img, method=args.method)

    if not tables:
        print("No tables detected! Try adjusting the model or lowering confidence.")
        # Save original anyway
        cv2.imwrite(str(out_preview), img)
        return

    print(f"Detected {len(tables)} table(s).")
    save_tables(tables, out_json)
    print(f"Saved table definitions to {out_json}")

    # Draw preview overlay
    overlay = img.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
        (0, 165, 255), (255, 255, 0), (128, 0, 128), (0, 128, 255),
    ]

    for i, (table_id, info) in enumerate(tables.items()):
        color = colors[i % len(colors)]
        poly = np.array(info["polygon"], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(img, [poly], True, color, 2)

        cx, cy = int(info["center"][0]), int(info["center"][1])
        cv2.putText(
            img, table_id, (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.imwrite(str(out_preview), img)
    print(f"Saved visual detection preview to {out_preview}")

if __name__ == "__main__":
    main()
