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

def main():
    # Paths
    img_path = project_root / "sddefault.jpg"
    overlaid_path = project_root / "sddefault_with_aruco.jpg"
    out_json = project_root / "analytics" / "config" / "tables_aruco_test.json"
    out_preview = project_root / "sddefault_with_aruco_detected.jpg"

    print(f"Loading base image {img_path}...")
    img = cv2.imread(str(img_path))
    if img is None:
        print("Error: Could not read base image.")
        return

    # Generate an ArUco marker
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 35
    border = 6
    total_size = marker_size + 2 * border
    
    # Tag 3
    marker_3 = cv2.aruco.generateImageMarker(dictionary, 3, marker_size)
    # Add white border
    marker_3 = cv2.copyMakeBorder(marker_3, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    marker_3_bgr = cv2.cvtColor(marker_3, cv2.COLOR_GRAY2BGR)
    
    # Location 1: Center Table area (x=330, y=210)
    x1, y1 = 330, 210
    img[y1:y1+total_size, x1:x1+total_size] = marker_3_bgr

    # Tag 7
    marker_7 = cv2.aruco.generateImageMarker(dictionary, 7, marker_size)
    # Add white border
    marker_7 = cv2.copyMakeBorder(marker_7, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    marker_7_bgr = cv2.cvtColor(marker_7, cv2.COLOR_GRAY2BGR)
    
    # Location 2: Right side Table area (x=500, y=120)
    x2, y2 = 500, 120
    img[y2:y2+total_size, x2:x2+total_size] = marker_7_bgr

    # Save the overlaid test image
    cv2.imwrite(str(overlaid_path), img)
    print(f"Saved overlaid test image to {overlaid_path}")

    # Now run the ArUco detector on it!
    # Scale = 6.0 (meaning the table ROI is 6x the marker size)
    detector = AutoTableDetector(aruco_scale_x=6.0, aruco_scale_y=6.0)
    print("Running ArUco table detection...")
    tables = detector.detect(img, method="aruco")

    if not tables:
        print("Error: No ArUco markers detected in the overlaid image!")
        return

    print(f"Success! Detected {len(tables)} table(s) via ArUco.")
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
