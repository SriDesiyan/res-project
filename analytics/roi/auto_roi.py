"""
Automatic Table ROI Detection — ArUco Markers and YOLOv8 Instance Segmentation.

This module provides two methods to detect dining tables in a reference frame:
1. ArUco markers (default & recommended) — extremely reliable, precise, handles rotation/scale.
2. YOLOv8 instance segmentation (legacy) — segments table shapes from pixels.

Usage (standalone detection):
    python auto_roi.py --video table_wghotel.mp4 --frame 0 --method aruco

Usage (standalone marker generation):
    python auto_roi.py --generate-markers --output-dir markers/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project imports work when run standalone
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from roi_config import save_tables

# COCO class ID for dining table
DINING_TABLE_CLASS = 60


class AutoTableDetector:
    """
    Detects tables from a single image or video frame using ArUco markers or YOLOv8-seg.

    Parameters
    ----------
    model_name : str
        Name of the YOLOv8 segmentation model (e.g. "yolov8n-seg.pt").
    conf : float
        Detection confidence threshold for YOLO/ArUco.
    simplify_epsilon_ratio : float
        Aggressiveness of polygon simplification for YOLO contours.
    min_area_ratio : float
        Min fraction of frame area for table detections.
    aruco_dict_name : str
        Predefined ArUco dictionary (default: "DICT_4X4_50").
    aruco_scale_x : float
        Default table width scale factor relative to marker size (default: 5.0).
    aruco_scale_y : float
        Default table height scale factor relative to marker size (default: 5.0).
    """

    def __init__(
        self,
        model_name: str = "yolov8n-seg.pt",
        conf: float = 0.25,
        simplify_epsilon_ratio: float = 0.015,
        min_area_ratio: float = 0.002,
        aruco_dict_name: str = "DICT_4X4_50",
        aruco_scale_x: float = 5.0,
        aruco_scale_y: float = 5.0,
    ):
        self.model_name = model_name
        self.conf = conf
        self.simplify_epsilon_ratio = simplify_epsilon_ratio
        self.min_area_ratio = min_area_ratio
        self.aruco_dict_name = aruco_dict_name
        self.aruco_scale_x = aruco_scale_x
        self.aruco_scale_y = aruco_scale_y

        self._yolo_model = None

    @property
    def yolo_model(self):
        """Lazy load YOLO model to save startup overhead when using ArUco."""
        if self._yolo_model is None:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.model_name)
        return self._yolo_model

    # ── Public API ─────────────────────────────────────────────────
    def detect(self, frame: np.ndarray, method: str = "aruco") -> dict:
        """
        Run table detection on a frame using the specified method.

        Parameters
        ----------
        frame : np.ndarray
            The image frame to scan.
        method : str
            Either "aruco" or "yolo".

        Returns
        -------
        dict
            ``{table_id: {"polygon": [[x,y], ...], "center": [cx, cy]}}``
        """
        if method == "yolo":
            return self.detect_yolo(frame)
        else:
            return self.detect_aruco(frame)

    def detect_from_video(
        self, video_path: Path, frame_number: int = 0, method: str = "aruco"
    ) -> dict:
        """Extract reference frame(s) from *video_path* and run detection."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video/image source: {video_path}")

        # Check if it is a static image or video
        is_video = True
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 1:
            is_video = False

        if not is_video or method == "yolo":
            # For static images or YOLO, do single frame calibration
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(f"Cannot read frame {frame_number} from {video_path}")
            print(f"[AUTO-ROI] Calibrating from single frame of {video_path.name} via method={method}")
            return self.detect(frame, method=method)

        # For ArUco on a video, implement multi-frame calibration averaging
        print(f"[AUTO-ROI] Calibrating via ArUco with multi-frame averaging over 10 frames starting at #{frame_number}...")
        
        # Dictionary mapping marker_id -> list of corner arrays (each shape (4,2))
        accumulated_corners: dict[int, list[np.ndarray]] = {}
        
        frames_read = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read up to 10 consecutive frames
        while frames_read < 10:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run ArUco detection on this frame
            dict_attr = getattr(cv2.aruco, self.aruco_dict_name, cv2.aruco.DICT_4X4_50)
            dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)
            parameters = cv2.aruco.DetectorParameters()
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            
            corners, ids, rejected = detector.detectMarkers(frame)
            if ids is not None and len(ids) > 0:
                ids = ids.flatten()
                for marker_corners, marker_id in zip(corners, ids):
                    mid = int(marker_id)
                    if mid not in accumulated_corners:
                        accumulated_corners[mid] = []
                    accumulated_corners[mid].append(marker_corners[0]) # corners shape (4,2)
            
            frames_read += 1
            
        cap.release()

        # If no markers were detected across all 10 frames
        if not accumulated_corners:
            print("[AUTO-ROI] No ArUco markers detected in any of the 10 calibration frames.")
            return {}

        # Compute median corners and project table ROIs
        tables: dict = {}
        
        # Load custom configurations if any
        config = {}
        config_path = Path(__file__).parent.parent / "config" / "aruco_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                print(f"[AUTO-ROI] Error reading config {config_path}: {e}")

        for marker_id, corners_list in accumulated_corners.items():
            # Median corners across visible frames (shape: (4, 2))
            median_corners = np.median(np.array(corners_list), axis=0)
            
            marker_id_str = str(marker_id)
            marker_config = config.get(marker_id_str, config.get("default", {}))

            scale_x = float(marker_config.get("scale_x", self.aruco_scale_x))
            scale_y = float(marker_config.get("scale_y", self.aruco_scale_y))
            offset_x = float(marker_config.get("offset_x", 0.0))
            offset_y = float(marker_config.get("offset_y", 0.0))

            polygon, center = self.get_table_polygon_from_marker(
                median_corners, scale_x, scale_y, offset_x, offset_y
            )

            table_id = f"table_{marker_id}"
            tables[table_id] = {
                "polygon": polygon,
                "center": center,
                "marker_id": marker_id,
            }
            print(
                f"  [AUTO-ROI] ArUco Tag {marker_id} calibrated (median over {len(corners_list)}/10 frames) → {table_id}: "
                f"center={center}, scale=({scale_x}, {scale_y})"
            )

        return tables

    # ── Detection Methods ──────────────────────────────────────────
    def detect_aruco(self, frame: np.ndarray) -> dict:
        """Detect ArUco markers and project clean, rotated table ROIs."""
        dict_attr = getattr(cv2.aruco, self.aruco_dict_name, cv2.aruco.DICT_4X4_50)
        dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        corners, ids, rejected = detector.detectMarkers(frame)
        tables: dict = {}

        if ids is None or len(ids) == 0:
            print("[AUTO-ROI] No ArUco markers detected in the frame.")
            return tables

        ids = ids.flatten()

        # Load custom offset/scale configuration if it exists
        config = {}
        config_path = Path(__file__).parent.parent / "config" / "aruco_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                print(f"[AUTO-ROI] Loaded scale/offset configs from {config_path}")
            except Exception as e:
                print(f"[AUTO-ROI] Error reading config {config_path}: {e}")

        for marker_corners, marker_id in zip(corners, ids):
            marker_id_str = str(marker_id)
            marker_config = config.get(marker_id_str, config.get("default", {}))

            scale_x = float(marker_config.get("scale_x", self.aruco_scale_x))
            scale_y = float(marker_config.get("scale_y", self.aruco_scale_y))
            offset_x = float(marker_config.get("offset_x", 0.0))
            offset_y = float(marker_config.get("offset_y", 0.0))

            pts = marker_corners[0]  # shape (4, 2)
            polygon, center = self.get_table_polygon_from_marker(
                pts, scale_x, scale_y, offset_x, offset_y
            )

            table_id = f"table_{marker_id}"
            tables[table_id] = {
                "polygon": polygon,
                "center": center,
                "marker_id": int(marker_id),
            }
            print(
                f"  [AUTO-ROI] ArUco Tag {marker_id} detected → {table_id}: "
                f"center={center}, scale=({scale_x}, {scale_y}), offset=({offset_x}, {offset_y})"
            )

        return tables

    def detect_yolo(self, frame: np.ndarray) -> dict:
        """Run YOLOv8 instance segmentation and extract table contours."""
        h, w = frame.shape[:2]
        frame_area = h * w
        min_area = frame_area * self.min_area_ratio

        results = self.yolo_model(frame, conf=self.conf, verbose=False)[0]
        tables: dict = {}
        table_counter = 1

        if results.masks is None:
            return tables

        for box, mask in zip(results.boxes, results.masks):
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            if class_id != DINING_TABLE_CLASS:
                continue

            poly_points = mask.xy[0].astype(np.float32)
            if len(poly_points) < 3:
                continue

            bbox_area = cv2.contourArea(poly_points)
            if bbox_area < min_area:
                continue

            arc_length = cv2.arcLength(poly_points, closed=True)
            epsilon = self.simplify_epsilon_ratio * arc_length
            approx = cv2.approxPolyDP(poly_points, epsilon, closed=True)
            simplified = approx.reshape(-1, 2).astype(int).tolist()

            if len(simplified) < 3:
                continue

            cx = float(np.mean([p[0] for p in simplified]))
            cy = float(np.mean([p[1] for p in simplified]))

            table_id = f"table_{table_counter}"
            tables[table_id] = {
                "polygon": simplified,
                "center": [round(cx, 1), round(cy, 1)],
            }
            table_counter += 1

            print(
                f"  [AUTO-ROI] YOLO Table {table_id}: {len(simplified)} vertices, "
                f"area={bbox_area:.0f}px², conf={confidence:.2f}"
            )

        return tables

    # ── Mathematical Projection ────────────────────────────────────
    @staticmethod
    def get_table_polygon_from_marker(
        corners: np.ndarray,
        scale_x: float,
        scale_y: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> tuple[list[list[int]], list[float]]:
        """
        Extrapolate 4 corners of a table from the 4 corners of a detected ArUco marker.

        Uses vector math to preserve rotation, translation, and scale (distance scaling).
        """
        # corners order: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
        p0, p1, p2, p3 = corners[0], corners[1], corners[2], corners[3]

        # Centroid
        cx = float(np.mean(corners[:, 0]))
        cy = float(np.mean(corners[:, 1]))
        center = np.array([cx, cy])

        # Local unit/scale coordinate vectors
        v_x = (p1 - p0 + p2 - p3) / 2.0  # Horizontal axis (left to right)
        v_y = (p3 - p0 + p2 - p1) / 2.0  # Vertical axis (top to bottom)

        # Shift the center based on offsets (scaled by the tag's local axes size)
        center_shifted = center + offset_x * v_x + offset_y * v_y

        # Extrapolate table corners outwards from the shifted center
        t0 = center_shifted - (scale_x / 2.0) * v_x - (scale_y / 2.0) * v_y
        t1 = center_shifted + (scale_x / 2.0) * v_x - (scale_y / 2.0) * v_y
        t2 = center_shifted + (scale_x / 2.0) * v_x + (scale_y / 2.0) * v_y
        t3 = center_shifted - (scale_x / 2.0) * v_x + (scale_y / 2.0) * v_y

        polygon = [
            [int(round(t0[0])), int(round(t0[1]))],
            [int(round(t1[0])), int(round(t1[1]))],
            [int(round(t2[0])), int(round(t2[1]))],
            [int(round(t3[0])), int(round(t3[1]))],
        ]
        center_coords = [
            float(round(center_shifted[0], 1)),
            float(round(center_shifted[1], 1)),
        ]
        return polygon, center_coords


# ── Standalone Marker Generation ──────────────────────────────────
def generate_markers(dictionary_name: str, ids: list[int], size: int, output_dir: Path):
    """Generate printable high-contrast ArUco marker PNG images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dict_attr = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)

    print(f"[AUTO-ROI] Generating ArUco markers (Dict: {dictionary_name}) in: {output_dir}")
    for marker_id in ids:
        # Generate the marker image
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
        
        # Add a white quiet zone/border around the marker so it detects better when printed
        border_size = int(size * 0.1)
        bordered_img = cv2.copyMakeBorder(
            marker_img, border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, value=255
        )
        
        output_path = output_dir / f"marker_{marker_id}.png"
        cv2.imwrite(str(output_path), bordered_img)
        print(f"  → Created marker_{marker_id}.png")
    print("[AUTO-ROI] Marker generation complete.")


# ── CLI parser ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    project_root = _script_dir.parent.parent
    parser = argparse.ArgumentParser(
        description="Auto-detect table ROIs (ArUco / YOLOv8-seg) or generate markers."
    )
    # Target Mode
    parser.add_argument(
        "--generate-markers",
        action="store_true",
        help="Generate printable ArUco markers instead of running detection.",
    )
    # Detection arguments
    parser.add_argument(
        "--video",
        type=Path,
        default=project_root / "table_wghotel.mp4",
        help="Path to the input video or image file.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame number to use as the calibration reference (default: 0).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="aruco",
        choices=["aruco", "yolo"],
        help="Calibration detection method: 'aruco' (default) or 'yolo'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_script_dir.parent / "config" / "tables_auto.json",
        help="Output JSON path for detected table definitions.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold for detection (default: 0.15).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-seg.pt",
        help="YOLOv8-seg model name (default: yolov8s-seg.pt).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a preview window with detected polygons overlaid.",
    )
    # Marker generation arguments
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_4X4_50",
        help="ArUco dictionary (default: DICT_4X4_50).",
    )
    parser.add_argument(
        "--marker-ids",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated marker IDs to generate (default: 0,1,2,3,4,5,6,7,8,9).",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=250,
        help="Pixel size of generated markers (default: 250).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "markers",
        help="Directory to save generated markers.",
    )
    return parser.parse_args()


def preview_detections(video_path: Path, frame_number: int, tables: dict):
    """Display an OpenCV window showing detected table polygons."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return

    overlay = frame.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
        (0, 165, 255), (255, 255, 0), (128, 0, 128), (0, 128, 255),
    ]

    for i, (table_id, info) in enumerate(tables.items()):
        color = colors[i % len(colors)]
        poly = np.array(info["polygon"], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(frame, [poly], True, color, 2)

        cx, cy = int(info["center"][0]), int(info["center"][1])
        cv2.putText(
            frame, table_id, (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    win = "Auto-ROI Preview (press any key to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.imshow(win, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    if args.generate_markers:
        try:
            ids = [int(x.strip()) for x in args.marker_ids.split(",")]
        except ValueError:
            print("[ERROR] --marker-ids must be a comma-separated list of integers.")
            return
        generate_markers(args.aruco_dict, ids, args.marker_size, args.output_dir)
        return

    if not args.video.exists():
        print(f"[ERROR] Video/Image not found: {args.video}")
        return

    detector = AutoTableDetector(
        model_name=args.model,
        conf=args.conf,
        aruco_dict_name=args.aruco_dict
    )
    tables = detector.detect_from_video(args.video, args.frame, method=args.method)

    if tables:
        save_tables(tables, args.output)
        print(f"\n[AUTO-ROI] Saved {len(tables)} table(s) → {args.output}")

        if args.preview:
            preview_detections(args.video, args.frame, tables)
    else:
        print(
            f"\n[AUTO-ROI] No tables detected via method='{args.method}'. "
            "Please check marker visibility or try a different frame."
        )


if __name__ == "__main__":
    main()
