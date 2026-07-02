"""
Interactive ROI Drawing Tool.

Opens a reference frame from the video and lets you draw table polygons.

Controls:
  LEFT CLICK  — place a vertex
  RIGHT CLICK — undo last vertex
  N           — finish current table, start next
  S           — save all tables to tables.json and exit
  Q / ESC     — quit without saving
  R           — reset current table

"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from roi_config import save_tables, polygon_center


tables = {}
current_polygon = []
table_counter = 1
drawing = True

COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (0, 165, 255),  # orange
    (255, 255, 0),  # cyan
    (128, 0, 128),  # purple
    (0, 128, 255),  # light orange
]


def get_color(idx):
    return COLORS[idx % len(COLORS)]


def mouse_callback(event, x, y, flags, param):
    global current_polygon

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
        print(f"  Vertex {len(current_polygon)}: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_polygon:
            removed = current_polygon.pop()
            print(f"  Undo: removed ({removed[0]}, {removed[1]})")


def draw_overlay(frame):
    """Draw all completed tables + current in-progress polygon."""
    overlay = frame.copy()

    # Draw completed tables
    for i, (table_id, table_info) in enumerate(tables.items()):
        poly = np.array(table_info["polygon"], dtype=np.int32)
        color = get_color(i)
        cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(overlay, [poly], color=(*color[:3],))
        cx, cy = int(table_info["center"][0]), int(table_info["center"][1])
        cv2.putText(overlay, table_id, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, overlay)

    for i, (table_id, table_info) in enumerate(tables.items()):
        poly = np.array(table_info["polygon"], dtype=np.int32)
        color = get_color(i)
        cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=2)
        cx, cy = int(table_info["center"][0]), int(table_info["center"][1])
        cv2.putText(overlay, table_id, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if current_polygon:
        color = get_color(len(tables))
        pts = np.array(current_polygon, dtype=np.int32)
        for j in range(len(current_polygon) - 1):
            cv2.line(overlay, tuple(current_polygon[j]), tuple(current_polygon[j+1]), color, 2)
        for pt in current_polygon:
            cv2.circle(overlay, tuple(pt), 5, color, -1)

    instructions = [
        f"Drawing: table_{table_counter}",
        f"Vertices: {len(current_polygon)}",
        "",
        "LEFT CLICK = add vertex",
        "RIGHT CLICK = undo vertex",
        "N = finish table, start next",
        "S = save & exit",
        "R = reset current table",
        "Q/ESC = quit (no save)",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(overlay, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    return overlay


def main():
    global current_polygon, table_counter, tables

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Interactive Table ROI Tool")
    parser.add_argument("--video", type=Path, default=project_root / "table_wghotel.mp4")
    parser.add_argument("--frame", type=int, default=4000, help="Frame number to use as reference")
    parser.add_argument("--output", type=Path, default=script_dir.parent / "config" / "tables.json")
    args = parser.parse_args()

    # Extract reference frame
    video_str = str(args.video)
    if video_str.lower().endswith(('.jpg', '.png', '.jpeg')):
        frame = cv2.imread(video_str)
        ret = frame is not None
    else:
        cap = cv2.VideoCapture(video_str)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {args.video}")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()

    if not ret:
        print(f"[ERROR] Cannot read frame {args.frame}")
        return

    print(f"Reference frame: {args.frame} ({frame.shape[1]}x{frame.shape[0]})")
    print(f"Output will be saved to: {args.output}")
    print()

    # Setup window
    window_name = "Table ROI Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"--- Drawing table_{table_counter} ---")

    while True:
        display = draw_overlay(frame)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('n') or key == ord('N'):
            # Finish current table
            if len(current_polygon) >= 3:
                table_id = f"table_{table_counter}"
                center = polygon_center(current_polygon)
                tables[table_id] = {
                    "polygon": current_polygon.copy(),
                    "center": [round(center[0], 1), round(center[1], 1)]
                }
                print(f"[SAVED] {table_id} with {len(current_polygon)} vertices")
                current_polygon = []
                table_counter += 1
                print(f"\n--- Drawing table_{table_counter} ---")
            else:
                print("Need at least 3 vertices to define a table polygon!")

        elif key == ord('r') or key == ord('R'):
            current_polygon = []
            print(f"Reset: table_{table_counter}")

        elif key == ord('s') or key == ord('S'):
            # Save current in-progress polygon if valid
            if len(current_polygon) >= 3:
                table_id = f"table_{table_counter}"
                center = polygon_center(current_polygon)
                tables[table_id] = {
                    "polygon": current_polygon.copy(),
                    "center": [round(center[0], 1), round(center[1], 1)]
                }
                print(f"Saved {table_id} with {len(current_polygon)} vertices")

            if tables:
                save_tables(tables, args.output)
                print(f"\nSaved {len(tables)} tables to {args.output}")
            else:
                print("No tables defined, nothing to save.")
            break

        elif key == ord('q') or key == ord('Q') or key == 27:
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
