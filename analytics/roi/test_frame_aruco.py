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
    img_path = project_root / "unlabeled-2" / "frame_0000486.jpg"
    overlaid_path = project_root / "frame_0000486_with_aruco.jpg"
    out_json = project_root / "analytics" / "config" / "tables_frame_0000486_aruco.json"
    out_preview = project_root / "frame_0000486_with_aruco_detected.jpg"

    print(f"Loading base frame {img_path}...")
    if not img_path.exists():
        print(f"Error: {img_path} not found.")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print("Error: Could not read base frame.")
        return

    h, w = img.shape[:2]
    print(f"Frame dimensions: {w}x{h}")

    # Generate ArUco markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # We use 80x80 marker with a 20px white border for high visibility on a 2K screen
    marker_size = 80
    border = 20
    total_size = marker_size + 2 * border

    # We will overlay 4 markers at the centers of tables identified in yolo mode
    # Format: (marker_id, cx, cy)
    targets = [
        (1, 252, 478),    # table_1 (left side)
        (3, 942, 581),    # table_3 (long center wood table)
        (2, 2202, 1364),  # table_2 (bottom right)
        (5, 2033, 934),   # table_5 (middle right, seated customer)
    ]

    for marker_id, cx, cy in targets:
        # Generate marker
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        # Add white border/quiet zone
        marker_img = cv2.copyMakeBorder(marker_img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

        # Centered paste coordinates
        x_start = int(cx - total_size // 2)
        y_start = int(cy - total_size // 2)

        # Boundaries check
        x_end = x_start + total_size
        y_end = y_start + total_size
        
        if x_start >= 0 and y_start >= 0 and x_end <= w and y_end <= h:
            img[y_start:y_end, x_start:x_end] = marker_bgr
            print(f"Overlaid Tag ID {marker_id} at center [{cx}, {cy}]")
        else:
            print(f"Warning: Tag ID {marker_id} coordinates [{cx}, {cy}] out of bounds.")

    # Save the overlaid test frame
    cv2.imwrite(str(overlaid_path), img)
    print(f"Saved overlaid test frame to {overlaid_path}")

    # Run the ArUco detector on the new overlaid frame!
    # Scale = 6.0 (meaning the table ROI is 6x the marker size)
    detector = AutoTableDetector(aruco_scale_x=6.0, aruco_scale_y=6.0)
    print("Running ArUco table detection on the frame...")
    tables = detector.detect(img, method="aruco")

    if not tables:
        print("Error: No ArUco markers detected in the overlaid frame!")
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
        cv2.polylines(img, [poly], True, color, 4) # Thicker border for 2K image

        cx, cy = int(info["center"][0]), int(info["center"][1])
        cv2.putText(
            img, table_id, (cx - 60, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4, # Larger font for 2K image
        )

    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.imwrite(str(out_preview), img)
    print(f"Saved visual detection preview to {out_preview}")

if __name__ == "__main__":
    main()
