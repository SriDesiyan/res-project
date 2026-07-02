"""
Restaurant CCTV Analytics Pipeline — Main Orchestrator.

Wires together all modules:
    Video → PersonTracker → ROI Assignment → OccupancyEngine
                                           → CleanlinessEngine
          → Renderer → Output Video

"""
import sys
import argparse
import time
from pathlib import Path
import sqlite3
import datetime
import csv
import json
import psutil
import subprocess
import gc

import cv2
import numpy as np
import torch

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from roi.roi_config import load_tables, assign_to_table
from roi.table_manager import TableManager
from tracking.person_tracker import PersonTracker
from tracking.session_manager import SessionManager
from occupancy.occupancy_engine import OccupancyEngine
from cleanliness.plate_detector import PlateDetector
from fsm.table_fsm import TableFSMManager
from config.fsm_config import FSM_CONFIG
from visualization.renderer import Renderer
from database.database_manager import DatabaseManager
from database.serving_event_logger import ServingEventLogger
from tracking.serving_detector import detect_waiter_serving, HAND_MODEL_PATH, POSE_MODEL_PATH
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def init_waiter_db():
    db_path = project_root / "waiter_logs.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS waiter_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waiter_id TEXT,
            table_id TEXT,
            timestamp REAL,
            log_type TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_waiter_visit_to_separate_db(waiter_id, table_id, timestamp):
    db_path = project_root / "waiter_logs.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    created_at_str = datetime.datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO waiter_visits (waiter_id, table_id, timestamp, log_type, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (str(waiter_id), str(table_id), float(timestamp), "order taken", created_at_str))
    conn.commit()
    conn.close()


def get_device():
    print("\n" + "="*40)
    print("        DETECTED HARDWARE INFO")
    print("="*40)
    print(f"OS Platform     : {sys.platform}")
    print(f"Python Version  : {sys.version.split()[0]}")
    print(f"PyTorch Version : {torch.__version__}")
    
    cuda_avail = torch.cuda.is_available()
    mps_avail = torch.backends.mps.is_available()
    print(f"CUDA Available  : {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Dev Count  : {torch.cuda.device_count()}")
    print(f"MPS Available   : {mps_avail}")
    
    device = torch.device("cpu")
    if cuda_avail:
        device = torch.device("cuda")
    elif mps_avail:
        device = torch.device("mps")
        
    print(f"Active Device   : {device}")
    print("="*40 + "\n")
    return device


def parse_args():
    project = Path(__file__).parent.parent.resolve()
    parser = argparse.ArgumentParser(description="Restaurant CCTV Analytics Pipeline")
    parser.add_argument("--video", type=Path, default=project / "test video 1.mp4")
    parser.add_argument("--out", type=Path, default=project / "output_analytics.mp4")
    parser.add_argument("--tables", type=Path, default=Path(__file__).parent / "config" / "tables.json")
    parser.add_argument("--start", type=float, default=1500, help="Start time in seconds (default: 0)")
    parser.add_argument("--end", type=float, default=1600, help="End time in seconds (default: 120)")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO detection confidence")
    parser.add_argument("--debug", action="store_true", help="Enable visual debug overlay")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--calibrate", action="store_true",
                        help="Auto-detect tables from an empty-room frame. "
                             "Runs once at startup and saves to --tables path.")
    parser.add_argument("--calibrate-frame", type=int, default=0,
                        help="Frame number to use for auto-calibration (default: 0). "
                             "Choose a frame where the restaurant is empty.")
    parser.add_argument("--calibrate-method", type=str, default="aruco",
                        choices=["aruco", "yolo"],
                        help="Table calibration method: 'aruco' (default) or 'yolo'.")
    parser.add_argument("--aruco-dict", type=str, default="DICT_4X4_50",
                        help="ArUco dictionary (default: DICT_4X4_50).")
    parser.add_argument("--aruco-scale", type=float, default=5.0,
                        help="Scale factor for ArUco marker projection (default: 5.0).")
    parser.add_argument("--auto-roi", action="store_true",
                        help="Enable continuous dwell-clustering to discover and "
                             "refine table ROIs during the pipeline run.")
    parser.add_argument("--roi-refresh-interval", type=float, default=300.0,
                        help="Seconds between dwell-clustering refreshes (default: 300).")
    parser.add_argument("--grace-period", type=float, default=300.0,
                        help="Grace period in seconds for customer database sessions (default: 300.0)")
    parser.add_argument("--use-fsm", action="store_true",
                        help="Enable deterministic Table State Machine tracking.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    # ── Table ROI Management ───────────────────────────────────
    # Use TableManager for dynamic table detection instead of
    # requiring a manually-created tables.json.
    table_mgr = TableManager(
        config_path=args.tables,
        video_path=args.video,
        calibrate_frame=getattr(args, "calibrate_frame", 0),
        enable_auto_detect=args.calibrate or not args.tables.exists(),
        enable_dwell_learning=args.auto_roi,
        refresh_interval_sec=args.roi_refresh_interval,
        calibrate_method=args.calibrate_method,
        aruco_dict=args.aruco_dict,
        aruco_scale=args.aruco_scale,
    )

    tables = table_mgr.get_tables()
    table_ids = table_mgr.get_table_ids()

    if not tables:
        print("[WARNING] No tables detected. The pipeline will run but "
              "occupancy/cleanliness analytics require at least one table.")
        print("  Options:")
        print("    1. Re-run with --calibrate using an empty-room frame")
        print("    2. Re-run with --auto-roi to learn tables from customers")
        print("    3. Manually create tables with:  python3 analytics/roi/roi_tool.py")
    else:
        print(f"Active table ROIs: {table_ids}")

    print("Initializing modules...")
    tracker = PersonTracker(device, conf=args.conf)
    session_manager = SessionManager(similarity_threshold=0.85, timeout_sec=900)
    occupancy = OccupancyEngine(table_ids, tables)
    fsm_manager = TableFSMManager(table_ids)
    plate_detector = PlateDetector(device=device)
    renderer = Renderer(tables)

    db = DatabaseManager()
    db.initialize_db()
    init_waiter_db()
    
    # Initialize MediaPipe models
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2
    )
    mp_hands = vision.HandLandmarker.create_from_options(hand_options)

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False
    )
    mp_pose = vision.PoseLandmarker.create_from_options(pose_options)
    
    # State tracking for database
    last_occ_log_time = 0
    active_db_sessions = {} # customer_track_id -> session_uuid
    lost_customer_timers = {} # customer_track_id -> time_lost
    previous_table_states = {} # table_id -> state
    table_waiter_first_seen = {} # table_id -> frame_time
    table_waiters_logged = {} # table_id -> set of logged waiter IDs
    
    # Order taken tracking state
    waiter_order_writing_start = {}  # (table_id, waiter_id) -> start_frame_time
    waiter_order_writing_not_writing_since = {}  # (table_id, waiter_id) -> timestamp
    order_taken_tables = {}          # table_id -> bool
    
    # Table occupancy locking state
    table_is_locked_occupied = {}    # table_id -> bool
    active_customer_tables = {}      # customer_track_id -> table_id
    table_occupied_start_time = {}   # table_id -> float
    
    # Food serving tracking state
    serving_logger = ServingEventLogger()
    waiter_last_serving_log_time = {}  # waiter_id -> timestamp
    food_served_tables = {}  # table_id -> frame_time when serving was detected
    food_served_debug_info = {}  # table_id -> debug data

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(args.start * fps)
    end_frame = int(args.end * fps) if args.end > 0 else total_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / getattr(args, "frame_step", 1)
    # Resize output to 1280x720 to speed up CPU encoding
    out_w, out_h = 1280, 720
    writer = cv2.VideoWriter(str(args.out), fourcc, out_fps, (out_w, out_h))

    frame_num = start_frame
    print(f"Processing frames {start_frame}–{end_frame} "
          f"({args.start:.0f}s – {args.end:.0f}s)")
    print(f"\n{'='*60}")
    print("Starting Analytics Pipeline")
    print(f"{'='*60}\n")

    cpu_usages = []
    gpu_usages = []
    processed_frames_count = 0
    start_processing_time = time.time()

    # Track table cleaning duration details
    table_cleaning_start = {}      # table_id -> frame_time when waiter arrived at dirty table
    table_cleaning_accumulated = {} # table_id -> accumulated cleaning duration
    completed_cleaning_durations = [] # list of durations

    try:
        while frame_num < end_frame:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = frame_num / fps
            processed_frames_count += 1

            # Periodically clear unused memory to prevent OpenCV/PyTorch OutOfMemoryError
            if processed_frames_count % 300 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if frame_num % 10 == 0:
                cpu_usages.append(psutil.cpu_percent())
                if torch.cuda.is_available():
                    try:
                        res = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                            stderr=subprocess.DEVNULL
                        )
                        gpu_usages.append(float(res.decode("utf-8").strip()))
                    except Exception:
                        gpu_usages.append(0.0)

            # ── Step 1: Track persons ──────────────────────────────
            persons = tracker.process_frame(frame, frame_time)

            # Assign persons to tables first, so we can use table assignments for skipping MediaPipe serving detection
            for person in persons:
                if person.role == "waiter":
                    best_table = None
                    best_dist = -float('inf')
                    for tid, t_info in tables.items():
                        poly = np.array(t_info["polygon"], dtype=np.int32)
                        dist = cv2.pointPolygonTest(poly, person.bottom_center, measureDist=True)
                        if dist > best_dist:
                            best_dist = dist
                            best_table = tid
                    if best_table and best_dist >= -180.0:
                        person.assigned_table = best_table
                    else:
                        person.assigned_table = None
                else:
                    person.assigned_table = assign_to_table(person.bottom_center, tables)

            # Run serving/writing pose detection for all tracked waiters
            waiters = [p for p in persons if p.role == "waiter"]
            
            # ── Build plate-in-ROI lookup for every table ──────────────────────────
            # This is used later as a pipeline-level guard before passing
            # is_serving=True to the FSM.  Only run food detection when there
            # are waiters near a table (avoids unnecessary YOLO inference).
            food_detections_frame = []  # cache for entire frame
            table_plate_in_roi = {}     # table_id -> bool (plate detected inside polygon)

            if waiters:
                # ── MediaPipe Optimization 1: Skip if waiter is far from all tables ──
                active_waiters_for_mp = []
                for person in waiters:
                    # Default flags
                    person.is_serving = False
                    person.is_order_taking = False
                    person.serving_res = {}

                    # Calculate minimum distance to table centers
                    min_center_dist = float('inf')
                    for tid, t_info in tables.items():
                        cx, cy = t_info["center"]
                        wx, wy = person.centroid
                        dist = ((wx - cx)**2 + (wy - cy)**2)**0.5
                        if dist < min_center_dist:
                            min_center_dist = dist

                    # Skip MediaPipe if waiter is not near any table
                    if person.assigned_table is None and min_center_dist >= 600.0:
                        continue
                    active_waiters_for_mp.append(person)

                if active_waiters_for_mp:
                    from tracking.serving_detector import detect_food_in_frame
                    # Shared food detection for all waiters this frame
                    food_detections_frame = detect_food_in_frame(frame, tracker.yolo)

                    # ── Build plate-in-ROI lookup ────────────────────────────────────
                    for tid, t_info in tables.items():
                        poly = np.array(t_info["polygon"], dtype=np.int32)
                        plate_found = False
                        for food in food_detections_frame:
                            fx1, fy1, fx2, fy2 = food['bbox']
                            # Check all four corners and centroid of the food bbox
                            food_cx = (fx1 + fx2) / 2.0
                            food_cy = (fy1 + fy2) / 2.0
                            for pt in [(food_cx, food_cy),
                                       (fx1, fy1), (fx2, fy1),
                                       (fx1, fy2), (fx2, fy2)]:
                                if cv2.pointPolygonTest(poly, pt, measureDist=False) >= 0:
                                    plate_found = True
                                    break
                            if plate_found:
                                break
                        table_plate_in_roi[tid] = plate_found

                    for person in active_waiters_for_mp:
                        # ── MediaPipe Optimization 2: subsample to every 6 frames ──
                        last_mp_check = getattr(person, 'last_mp_check_frame', -100)
                        if frame_num - last_mp_check >= 6 or not hasattr(person, 'last_mp_res'):
                            res = detect_waiter_serving(
                                frame, person.bbox, tracker.yolo,
                                mp_hands, mp_pose,
                                food_detections=food_detections_frame
                            )
                            person.last_mp_res = res
                            person.last_mp_check_frame = frame_num
                        else:
                            res = person.last_mp_res

                        person.is_serving = res.get('is_serving', False)
                        person.is_order_taking = res.get('is_order_taking', False)
                        person.serving_res = res

                        wid = getattr(person, "session_id", f"T{person.track_id}")
                        if frame_num % 30 == 0:  # Reduced spam: every 30 frames
                            print(
                                f"  [SERVE-CHK] F:{frame_num} waiter={wid} "
                                f"is_serving={person.is_serving} "
                                f"conf={res.get('confidence', 0):.2f} "
                                f"methods={res.get('methods', {})}"
                            )

                        # ── Per-table serving signal with pipeline guards ─────────────
                        # Guard 1: Waiter must be inside / near the target table's ROI
                        # Guard 2: Plate must be inside that table's ROI
                        # Guard 3: Hand-to-table centroid distance < configured max
                        # Guard 4: FSM state for that table must be WAITING_FOR_FOOD
                        if person.is_serving:
                            hand_pos = res.get('serving_hand_pos')
                            if hand_pos:
                                hx, hy = hand_pos
                            else:
                                x1b, y1b, x2b, y2b = person.bbox
                                hx, hy = (x1b + x2b) / 2.0, float(y2b)

                            # Find the nearest table
                            best_tid = None
                            best_dist = float('inf')
                            best_cx, best_cy = 0.0, 0.0
                            max_hand_dist = FSM_CONFIG["food_served_hand_to_table_max_px"]
                            for tid, t_info in tables.items():
                                cx, cy = t_info["center"]
                                dist = ((hx - cx)**2 + (hy - cy)**2)**0.5
                                if dist < best_dist:
                                    best_dist = dist
                                    best_tid = tid
                                    best_cx, best_cy = cx, cy

                            if best_tid and best_dist < max_hand_dist:
                                # Guard: waiter must be assigned to this table
                                waiter_in_roi = (
                                    person.assigned_table == best_tid
                                    or person.assigned_table is None  # fallback allow
                                )
                                # Guard: plate must be in this table's ROI
                                plate_ok = True  # Relaxed: waiter holding food near table is sufficient
                                # Guard: FSM must be in WAITING_FOR_FOOD for this table
                                fsm_state_now = fsm_manager.get_state(best_tid)["state"]
                                state_ok = fsm_state_now == "WAITING_FOR_FOOD"
                                # Guard: FOOD_SERVED must not have been fired already
                                food_fired = fsm_manager.tables[best_tid].food_served_fired

                                if waiter_in_roi and plate_ok and state_ok and not food_fired:
                                    food_served_tables[best_tid] = frame_time
                                    food_served_debug_info[best_tid] = {
                                        "wrist": (hx, hy),
                                        "centroid": (best_cx, best_cy),
                                        "distance": best_dist
                                    }
                                    print(
                                        f"[PIPELINE] FOOD SERVED signal for {best_tid} "
                                        f"by {wid} (dist={best_dist:.0f}px "
                                        f"plate_in_roi={plate_ok}) F:{frame_num}"
                                    )
                                else:
                                    if not state_ok:
                                        pass  # silent: table not in WAITING_FOR_FOOD
                                    elif not plate_ok:
                                        print(
                                            f"[PIPELINE] FOOD SERVED rejected: no plate "
                                            f"in ROI of {best_tid} F:{frame_num}"
                                        )
                                    elif food_fired:
                                        pass  # already served this session


            # ── Step 1b: Feed dwell points for auto-ROI learning ──
            for person in persons:
                table_mgr.record_customer(person, frame_time)

            # ── Step 1c: Periodic dwell-clustering refresh ────────
            if table_mgr.maybe_refresh(frame_time):
                tables = table_mgr.get_tables()
                table_ids = table_mgr.get_table_ids()
                # Hot-reload downstream modules with new tables
                occupancy = OccupancyEngine(table_ids, tables)
                fsm_manager = TableFSMManager(table_ids)
                renderer = Renderer(tables)
                print(f"[PIPELINE] Hot-reloaded {len(tables)} table(s)")

            # ── Step 3: Session Management (Re-ID) ─────────────
            resumed_sessions = session_manager.update(persons, frame_time, frame_shape=frame.shape, tables=tables)
            # No longer adjusting entry time in DB to ensure the session duration captures the entire visit including temporary absences
            # for cid, absent_dur in resumed_sessions.items():
            #     if cid in active_db_sessions:
            #         db.adjust_customer_session_entry_time(active_db_sessions[cid], absent_dur)
            #         print(f"[PIPELINE] Adjusted entry time in DB for session {cid} by {absent_dur:.1f}s")

            # Sync session roles back to tracker's locked_waiters and vice versa
            # RULE: tracker.locked_waiters is authoritative - if the tracker has confirmed
            # a waiter via uniform detection, the session role cannot override that.
            for person in persons:
                if person.session_id:
                    sid = person.session_id
                    sess_role = session_manager.active_sessions[sid]["role"]
                    tracker_locked_waiter = person.track_id in tracker.locked_waiters
                    current_hits = tracker.waiter_hits.get(person.track_id, 0)
                    if tracker_locked_waiter:
                        # Tracker is authoritative: this is a confirmed waiter
                        person.role = "waiter"
                        # Also upgrade the session role if it's still 'customer'
                        if sess_role == "customer":
                            session_manager.active_sessions[sid]["role"] = "waiter"
                            print(f"[ROLE-SYNC] F:{frame_num} Track {person.track_id} FORCED to waiter (sess was customer)")
                        # else already waiter
                    elif current_hits > 0:
                        # Person is accumulating waiter evidence - do NOT clear hits!
                        # Let the tracker's own logic decide when to lock.
                        # Don't override role to customer if they have active hits.
                        pass
                    elif sess_role == "waiter":
                        person.role = "waiter"
                        tracker.locked_waiters.add(person.track_id)
                        if tracker.waiter_hits[person.track_id] < 12:
                            tracker.waiter_hits[person.track_id] = 12
                    elif sess_role == "customer":
                        person.role = "customer"
                        # Only clear waiter state if tracker hasn't locked them AND no active hits
                        tracker.locked_waiters.discard(person.track_id)
                        # DO NOT pop waiter_hits here - let the tracker's decay handle it naturally
                elif person.track_id in tracker.locked_waiters:
                    # No session yet, but tracker already locked them as waiter — force role
                    person.role = "waiter"
                    if frame_num % 50 == 0:
                        print(f"[ROLE-SYNC] F:{frame_num} Track {person.track_id} is locked waiter but has no session")

            # ── Step 4: Update occupancy ───────────────────────
            occupancy.update(persons, frame_time)
            occupancy_data = occupancy.get_all_status(frame_time)

            # Update occupancy locking states based on active customer DB sessions
            for occ in occupancy_data:
                tid = occ["table_id"]
                if occ["is_occupied"]:
                    table_is_locked_occupied[tid] = True
                elif table_is_locked_occupied.get(tid, False):
                    # Stay locked if there is still an active session for this table
                    has_active = any(active_customer_tables.get(cid) == tid for cid in active_db_sessions)
                    if not has_active:
                        table_is_locked_occupied[tid] = False

            # Modify occupancy_data and table properties to use is_occupied_effective
            for occ in occupancy_data:
                tid = occ["table_id"]
                is_occupied_effective = table_is_locked_occupied.get(tid, False)
                occ["is_occupied"] = is_occupied_effective
                occupancy.tables[tid].is_occupied = is_occupied_effective
                
                if is_occupied_effective:
                    if tid not in table_occupied_start_time:
                        table_occupied_start_time[tid] = frame_time
                    occupancy.tables[tid].occupied_start_time = table_occupied_start_time[tid]
                else:
                    table_occupied_start_time.pop(tid, None)
                    table_waiters_logged.pop(tid, None)
                    occupancy.tables[tid].occupied_start_time = None

            if frame_time - last_occ_log_time >= 3.0:
                for occ in occupancy_data:
                    db.log_occupancy(
                        timestamp=frame_time,
                        table_id=occ["table_id"],
                        occupancy_count=occ["customer_count"],
                        waiter_count=1 if occ["waiter_present"] else 0,
                        is_occupied=occ["is_occupied"]
                    )
                last_occ_log_time = frame_time

            # ── Step 5: Update cleanliness & DB Sessions ───────
            completed_sessions = []
            current_active_customers = set()

            for occ in occupancy_data:
                tid = occ["table_id"]
                table_obj = occupancy.tables[tid]
                for cid in table_obj.current_customers:
                    current_active_customers.add(cid)
                    if cid not in active_db_sessions:
                        try:
                            numeric_cid = int("".join(c for c in str(cid) if c.isdigit()))
                        except ValueError:
                            numeric_cid = 999
                        suuid = db.create_customer_session(customer_track_id=numeric_cid, table_id=tid, entry_time=frame_time)
                        active_db_sessions[cid] = suuid
                        active_customer_tables[cid] = tid
                        
                # 5b. Waiter DB Metrics
                if occ["waiter_present"]:
                    if tid not in table_waiter_first_seen:
                        table_waiter_first_seen[tid] = frame_time
                        # If customers just arrived, log waiter response time
                        if occ["is_occupied"] and table_obj.occupied_start_time:
                            if not table_waiters_logged.get(tid, False):
                                db.log_waiter_metric(tid, table_obj.occupied_start_time, frame_time)
                                table_waiters_logged[tid] = True
                else:
                    table_waiter_first_seen.pop(tid, None)

                # --- FSM Mode Update ---
                # Determine current FSM state for dirty scanning
                t_state = fsm_manager.get_state(tid)["state"]
                live_obj_count = 0
                if t_state == "DIRTY" and frame_num % 5 == 0:
                    pts = occupancy.table_polygons.get(tid)
                    if pts is not None:
                        live_obj_count = plate_detector.detect_dirty_objects(frame, pts)

                # Get active customer session UUID
                active_session_uuid = None
                for cid in table_obj.current_customers:
                    if cid in active_db_sessions:
                        active_session_uuid = active_db_sessions[cid]
                        break

                # Get writing state
                is_writing = False
                writing_waiter_id = None
                for p in persons:
                    if p.role == "waiter" and getattr(p, "is_order_taking", False):
                        # Check distance to this table's center
                        cx, cy = tables[tid]["center"]
                        wx, wy = p.centroid
                        dist = ((wx - cx)**2 + (wy - cy)**2)**0.5
                        if dist < 600.0:
                            is_writing = True
                            writing_waiter_id = getattr(p, "session_id", f"T{p.track_id}")
                            break

                # Get serving state — only valid if detector fired this table this frame
                # AND pipeline-level guards already confirmed plate-in-ROI + waiter-in-ROI
                is_serving_for_fsm = (food_served_tables.get(tid) == frame_time)

                # Collect serving waiter ID and confidence for FSM ownership guard
                serving_waiter_id_for_fsm = None
                serving_confidence_for_fsm = 0.0
                if is_serving_for_fsm:
                    # Find the waiter assigned to this table who is serving
                    for p in persons:
                        if p.role == "waiter" and getattr(p, "is_serving", False):
                            wid_check = getattr(p, "session_id", f"T{p.track_id}")
                            if (p.assigned_table == tid
                                    or p.assigned_table is None):
                                serving_waiter_id_for_fsm = wid_check
                                serving_confidence_for_fsm = p.serving_res.get(
                                    "confidence", 0.0
                                )
                                break

                # Update Table FSM with all guards (use raw_customer_present OR stabilized verified customer presence to absorb occlusion)
                raw_customer_present = getattr(table_obj, "raw_customer_present", False)
                has_verified_customer = len(table_obj.current_customers) > 0
                customer_present_stable = raw_customer_present or has_verified_customer
                fsm_manager.update(
                    table_id=tid,
                    customer_present=customer_present_stable,
                    waiter_present=occ["waiter_present"],
                    is_writing=is_writing,
                    is_serving=is_serving_for_fsm,
                    dirty_object_count=live_obj_count,
                    frame_time=frame_time,
                    db_manager=db,
                    session_uuid=active_session_uuid,
                    serving_waiter_id=serving_waiter_id_for_fsm,
                    plate_in_roi=table_plate_in_roi.get(tid, False) or is_serving_for_fsm,
                    serving_confidence=serving_confidence_for_fsm,
                )

                # Log waiter visit to separate database if we transitioned to WAITING_FOR_FOOD
                # (ORDER_TAKEN fires the waiter-visit log; WAITING_FOR_FOOD is the persistent state)
                fsm_table = fsm_manager.tables[tid]
                if fsm_table.state == "WAITING_FOR_FOOD" and not getattr(fsm_table, "order_logged", False):
                    logging_wid = writing_waiter_id
                    if not logging_wid:
                        for p in persons:
                            if p.role == "waiter":
                                cx, cy = tables[tid]["center"]
                                wx, wy = p.centroid
                                dist = ((wx - cx)**2 + (wy - cy)**2)**0.5
                                if dist < 600.0:
                                    logging_wid = getattr(p, "session_id", f"T{p.track_id}")
                                    break
                    if logging_wid:
                        log_waiter_visit_to_separate_db(logging_wid, tid, frame_time)
                        print(f"[PIPELINE FSM] Logged waiter {logging_wid} order taken at {tid}")
                        fsm_table.order_logged = True

                # Log serving event to DB once FOOD_SERVED is confirmed by FSM
                if fsm_table.food_served_fired and fsm_table.state in ("FOOD_SERVED", "DINING"):
                    serving_wid = serving_waiter_id_for_fsm or (
                        list(table_obj.current_waiters)[0]
                        if table_obj.current_waiters else "unknown"
                    )
                    if (serving_wid not in waiter_last_serving_log_time
                            or frame_time - waiter_last_serving_log_time.get(serving_wid, 0) > 30.0):
                        # Only log once: check if we already logged for this session
                        prev_log_key = f"{tid}_{fsm_table.dining_time_start}"
                        if prev_log_key not in waiter_last_serving_log_time:
                            food_info = food_served_debug_info.get(tid, {})
                            serving_logger.log_serving(
                                waiter_id=serving_wid,
                                table_id=tid,
                                food_type="food",
                                frame_num=frame_num,
                                confidence=serving_confidence_for_fsm or 1.0
                            )
                            waiter_last_serving_log_time[serving_wid] = frame_time
                            waiter_last_serving_log_time[prev_log_key] = frame_time
                            print(
                                f"[PIPELINE] DB: Serving event logged for {tid} "
                                f"by {serving_wid} at {frame_time:.1f}s"
                            )

                # Synchronize order_taken_tables map for any other components
                order_taken_tables[tid] = fsm_table.state in (
                    "ORDER_TAKEN", "WAITING_FOR_FOOD", "FOOD_SERVED", "DINING"
                )

                current_state = fsm_table.state
                
                # Cleaning duration tracking:
                if current_state == "DIRTY":
                    if occ["waiter_present"]:
                        if tid not in table_cleaning_start:
                            table_cleaning_start[tid] = frame_time
                    else:
                        if tid in table_cleaning_start:
                            dur = frame_time - table_cleaning_start.pop(tid)
                            table_cleaning_accumulated[tid] = table_cleaning_accumulated.get(tid, 0.0) + dur
                
                if current_state == "CLEAN":
                    dur = table_cleaning_accumulated.pop(tid, 0.0)
                    if tid in table_cleaning_start:
                        dur += (frame_time - table_cleaning_start.pop(tid))
                    if dur > 0.0:
                        completed_cleaning_durations.append(dur)
                        print(f"[METRIC] Cleaned table {tid} in {dur:.1f} seconds.")
                


            # 5d. Close Lost DB Sessions (with 5-minute grace period)
            missing_customers = set(active_db_sessions.keys()) - current_active_customers
            
            # Start grace timer for newly missing customers
            for cid in missing_customers:
                if cid not in lost_customer_timers:
                    lost_customer_timers[cid] = frame_time
                    print(f"[PIPELINE] Customer {cid} went missing (grace timer started)")
                    
            # Remove grace timer if customer returned
            for cid in list(lost_customer_timers.keys()):
                if cid in current_active_customers:
                    lost_customer_timers.pop(cid, None)
                    print(f"[PIPELINE] Customer {cid} returned within grace period")

            # Close sessions for customers missing longer than grace period (300 seconds)
            grace_period = getattr(args, "grace_period", 300.0)
            to_close = []
            for cid, lost_time in list(lost_customer_timers.items()):
                if frame_time - lost_time > grace_period:
                    to_close.append(cid)

            for cid in to_close:
                lost_time = lost_customer_timers.pop(cid)
                db.close_customer_session(active_db_sessions[cid], lost_time)
                del active_db_sessions[cid]
                active_customer_tables.pop(cid, None)
                print(f"[PIPELINE] Grace period expired for customer {cid}. Closed session in DB.")

            # Clear food_served_tables entries older than 10 seconds
            for stid in list(food_served_tables.keys()):
                if frame_time - food_served_tables[stid] > 10.0:
                    food_served_tables.pop(stid)
                    food_served_debug_info.pop(stid, None)

            # ── Step 5: Render ─────────────────────────────────
            elapsed = time.time() - t0
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fsm_states = fsm_manager.get_all_states()
            annotated = renderer.render(
                frame,
                persons,
                occupancy_data,
                fsm_states,
                current_fps,
                frame_time,
                food_served_debug_info,
                debug=args.debug
            )
            resized_annotated = cv2.resize(annotated, (1280, 720))
            writer.write(resized_annotated)

            # Skip next (frame_step - 1) frames
            step = getattr(args, "frame_step", 1)
            if step > 1:
                for _ in range(step - 1):
                    cap.grab()
            
            frame_num += step

            # Progress
            if frame_num % 50 == 0:
                total_custs = sum(len(occupancy.tables[t].current_customers) for t in table_ids)
                total_waiters = sum(len(occupancy.tables[t].current_waiters) for t in table_ids)
                print(f"Frame {frame_num} | "
                      f"Persons: {len(persons)} | "
                      f"Customers: {total_custs} | "
                      f"Waiters: {total_waiters} | "
                      f"FPS: {current_fps:.1f}")
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user. Finalizing and saving...")
    finally:
        # Close any remaining active database sessions at the end of the run
        for cid, suuid in active_db_sessions.items():
            exit_time = lost_customer_timers.get(cid, frame_num / fps)
            db.close_customer_session(suuid, exit_time)
        active_db_sessions.clear()
        active_customer_tables.clear()

        final_occ = occupancy.get_all_status(frame_num / fps)
        final_states = fsm_manager.get_all_states()

        cap.release()
        writer.release()

    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Output video: {args.out}")

    # Print final table summary
    print(f"\n{'='*60}")
    print("Table Summary")
    print(f"{'='*60}")
    for occ in final_occ:
        tid = occ["table_id"]
        state = final_states.get(tid, "?")
        print(f"  {tid}: {state} | "
              f"Visits: {occ['total_visits']} | "
              f"Peak: {occ['peak_occupancy']} | "
              f"Waiter visits: {occ['waiter_visits']}")

    # 1. Export CSV logs using python standard csv module
    print(f"\n{'='*60}")
    print("Exporting Database Logs to CSV")
    print(f"{'='*60}")
    
    # 1a. Export occupancy_log.csv
    try:
        conn = sqlite3.connect(str(project_root / "restaurant_analytics.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, table_id, occupancy_count, waiter_count, is_occupied FROM occupancy_logs")
        rows = cursor.fetchall()
        with open(project_root / "occupancy_log.csv", "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["timestamp", "table_id", "occupancy_count", "waiter_count", "is_occupied"])
            csv_writer.writerows(rows)
        print("✓ Exported occupancy_log.csv successfully.")
        conn.close()
    except Exception as e:
        print(f"✗ Failed to export occupancy_log.csv: {e}")

    # 1b. Export waiter_metrics.csv
    try:
        conn = sqlite3.connect(str(project_root / "restaurant_analytics.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT table_id, customer_arrival_time, waiter_first_seen_time, response_time_seconds FROM waiter_service_metrics")
        rows = cursor.fetchall()
        with open(project_root / "waiter_metrics.csv", "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["table_id", "customer_arrival_time", "waiter_first_seen_time", "response_time_seconds"])
            csv_writer.writerows(rows)
        print("✓ Exported waiter_metrics.csv successfully.")
        conn.close()
    except Exception as e:
        print(f"✗ Failed to export waiter_metrics.csv: {e}")

    # 1c. Export serving_events.csv
    try:
        conn = sqlite3.connect(str(project_root / "serving_logs.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT id, waiter_id, table_id, food_type, timestamp, frame_number, confidence FROM serving_events")
        rows = cursor.fetchall()
        with open(project_root / "serving_events.csv", "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["id", "waiter_id", "table_id", "food_type", "timestamp", "frame_number", "confidence"])
            csv_writer.writerows(rows)
        print("✓ Exported serving_events.csv successfully.")
        conn.close()
    except Exception as e:
        print(f"✗ Failed to export serving_events.csv: {e}")

    # 1d. Export cleanliness_log.csv
    try:
        conn = sqlite3.connect(str(project_root / "restaurant_analytics.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT table_id, state, start_time, end_time, duration_seconds FROM table_state_history")
        rows = cursor.fetchall()
        with open(project_root / "cleanliness_log.csv", "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["table_id", "state", "start_time", "end_time", "duration_seconds"])
            csv_writer.writerows(rows)
        print("✓ Exported cleanliness_log.csv successfully.")
        conn.close()
    except Exception as e:
        print(f"✗ Failed to export cleanliness_log.csv: {e}")

    # 2. Compile summary.json report
    print(f"\n{'='*60}")
    print("Compiling Summary JSON Report")
    print(f"{'='*60}")
    
    summary = {
        "total_customers": 0,
        "total_waiters": 0,
        "occupied_tables_count": 0,
        "average_occupancy_duration_seconds": 0.0,
        "order_taking_events_count": 0,
        "food_serving_events_count": 0,
        "average_waiter_response_time_seconds": 0.0,
        "average_dirty_table_duration_seconds": 0.0,
        "dirty_table_durations_by_table": {},
        "average_cleaning_duration_seconds": 0.0,
        "peak_occupancy_periods": []
    }

    try:
        conn_ra = sqlite3.connect(str(project_root / "restaurant_analytics.db"))
        cursor_ra = conn_ra.cursor()

        cursor_ra.execute("SELECT COUNT(DISTINCT customer_track_id) FROM customer_sessions")
        summary["total_customers"] = cursor_ra.fetchone()[0] or 0

        cursor_ra.execute("SELECT COUNT(DISTINCT table_id) FROM occupancy_logs WHERE is_occupied = 1")
        summary["occupied_tables_count"] = cursor_ra.fetchone()[0] or 0

        cursor_ra.execute("SELECT AVG(duration_seconds) FROM customer_sessions WHERE duration_seconds IS NOT NULL")
        val = cursor_ra.fetchone()[0]
        summary["average_occupancy_duration_seconds"] = round(val, 2) if val else 0.0

        cursor_ra.execute("SELECT AVG(response_time_seconds) FROM waiter_service_metrics")
        val = cursor_ra.fetchone()[0]
        summary["average_waiter_response_time_seconds"] = round(val, 2) if val else 0.0

        cursor_ra.execute("SELECT AVG(duration_seconds) FROM table_state_history WHERE state = 'DIRTY' AND duration_seconds IS NOT NULL")
        val = cursor_ra.fetchone()[0]
        summary["average_dirty_table_duration_seconds"] = round(val, 2) if val else 0.0

        cursor_ra.execute("SELECT table_id, AVG(duration_seconds) FROM table_state_history WHERE state = 'DIRTY' AND duration_seconds IS NOT NULL GROUP BY table_id")
        for tid, avg_dur in cursor_ra.fetchall():
            summary["dirty_table_durations_by_table"][tid] = round(avg_dur, 2)

        # Peak occupancy periods calculation
        cursor_ra.execute("""
            SELECT timestamp, SUM(occupancy_count) as total_cust
            FROM occupancy_logs
            GROUP BY timestamp
            ORDER BY total_cust DESC, timestamp ASC
        """)
        occ_rows = cursor_ra.fetchall()
        if occ_rows:
            max_cust = occ_rows[0][1]
            if max_cust > 0:
                peak_times = [r[0] for r in occ_rows if r[1] == max_cust]
                periods = []
                if peak_times:
                    start_p = peak_times[0]
                    prev_p = peak_times[0]
                    for pt in peak_times[1:]:
                        if pt - prev_p > 5.0:
                            periods.append(f"{start_p:.1f}s - {prev_p:.1f}s (Peak: {max_cust} customers)")
                            start_p = pt
                        prev_p = pt
                    periods.append(f"{start_p:.1f}s - {prev_p:.1f}s (Peak: {max_cust} customers)")
                summary["peak_occupancy_periods"] = periods
            else:
                summary["peak_occupancy_periods"] = ["None (0 customers)"]
        else:
            summary["peak_occupancy_periods"] = ["No data"]

        conn_ra.close()
    except Exception as e:
        print(f"✗ Failed to query restaurant_analytics.db: {e}")

    # Distinct waiters union (using persistent session IDs, ignoring raw track IDs)
    waiters_set = set()
    for sid, data in session_manager.active_sessions.items():
        if data.get("role") == "waiter":
            waiters_set.add(sid)
    try:
        conn_wl = sqlite3.connect(str(project_root / "waiter_logs.db"))
        cursor_wl = conn_wl.cursor()
        cursor_wl.execute("SELECT DISTINCT waiter_id FROM waiter_visits")
        for (w_id,) in cursor_wl.fetchall():
            waiters_set.add(str(w_id))
        conn_wl.close()
    except Exception:
        pass
    try:
        conn_sl = sqlite3.connect(str(project_root / "serving_logs.db"))
        cursor_sl = conn_sl.cursor()
        cursor_sl.execute("SELECT DISTINCT waiter_id FROM serving_events")
        for (w_id,) in cursor_sl.fetchall():
            waiters_set.add(str(w_id))
        conn_sl.close()
    except Exception:
        pass
    summary["total_waiters"] = len(waiters_set)

    # Order taking & serving counts
    try:
        conn_wl = sqlite3.connect(str(project_root / "waiter_logs.db"))
        cursor_wl = conn_wl.cursor()
        cursor_wl.execute("SELECT COUNT(*) FROM waiter_visits")
        summary["order_taking_events_count"] = cursor_wl.fetchone()[0] or 0
        conn_wl.close()
    except Exception:
        pass
    try:
        conn_sl = sqlite3.connect(str(project_root / "serving_logs.db"))
        cursor_sl = conn_sl.cursor()
        cursor_sl.execute("SELECT COUNT(*) FROM serving_events")
        summary["food_serving_events_count"] = cursor_sl.fetchone()[0] or 0
        conn_sl.close()
    except Exception:
        pass

    if completed_cleaning_durations:
        summary["average_cleaning_duration_seconds"] = round(sum(completed_cleaning_durations) / len(completed_cleaning_durations), 2)
    else:
        summary["average_cleaning_duration_seconds"] = 0.0

    try:
        with open(project_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        print("✓ Generated summary.json successfully.")
    except Exception as e:
        print(f"✗ Failed to write summary.json: {e}")

    # Print performance profiling report
    end_processing_time = time.time()
    total_time = end_processing_time - start_processing_time
    avg_fps = processed_frames_count / total_time if total_time > 0 else 0
    avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0
    avg_gpu = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0.0
    
    avg_inf = (sum(tracker.yolo_latencies) / len(tracker.yolo_latencies) * 1000.0) if tracker.yolo_latencies else 0.0
    avg_track = (sum(tracker.tracking_latencies) / len(tracker.tracking_latencies) * 1000.0) if tracker.tracking_latencies else 0.0

    print(f"\n{'='*60}")
    print("PERFORMANCE PROFILING REPORT")
    print(f"{'='*60}")
    print(f"Processed Frames           : {processed_frames_count}")
    print(f"Total Processing Time      : {total_time:.2f} seconds")
    print(f"Average FPS                : {avg_fps:.2f}")
    print(f"Average CPU Utilization    : {avg_cpu:.1f}%")
    print(f"Average GPU Utilization    : {avg_gpu:.1f}%")
    print(f"Average Inference Latency  : {avg_inf:.2f} ms")
    print(f"Average Tracking Latency   : {avg_track:.2f} ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()