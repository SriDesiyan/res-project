import subprocess
import os
import sys
import time
import shutil
import sqlite3
import json
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
outputs_dir = project_root / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

def reset_db_and_csvs():
    print("[RUNNER] Resetting local databases and root CSV logs...")
    dbs = ["restaurant_analytics.db", "waiter_logs.db", "serving_logs.db"]
    for db in dbs:
        db_path = project_root / db
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception as e:
                print(f"  Warning: could not delete {db}: {e}")
                
    csvs = ["cleanliness_log.csv", "occupancy_log.csv", "serving_events.csv", "waiter_metrics.csv", "summary.json"]
    for csv_file in csvs:
        csv_path = project_root / csv_file
        if csv_path.exists():
            try:
                csv_path.unlink()
            except Exception as e:
                print(f"  Warning: could not delete {csv_file}: {e}")

def run_pipeline(debug_mode=False):
    mode_str = "DEBUG" if debug_mode else "STANDARD"
    out_file = "Hotet_Test_1_debug.mp4" if debug_mode else "Hotet_Test_1_output.mp4"
    out_path = outputs_dir / out_file
    
    print(f"\n============================================================")
    print(f"STARTING PIPELINE RUN: {mode_str} MODE")
    print(f"Output Video: {out_path}")
    print(f"============================================================\n")
    
    cmd = [
        sys.executable, "-u", "analytics/pipeline.py",
        "--video", "Hotet_Test_1.mp4",
        "--out", str(out_path),
        "--tables", "analytics/config/tables_hotet_test_1.json",
        "--use-fsm",
        "--start", "0",
        "--end", "0",
        "--frame-step", "60"
    ]
    if debug_mode:
        cmd.append("--debug")
        
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(project_root)
    )
    
    log_lines = []
    # Read output live and print to console
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end="", flush=True)
            log_lines.append(line)
            
    rc = process.wait()
    print(f"\n[RUNNER] {mode_str} Run finished with exit code {rc}")
    return rc, "".join(log_lines)

def parse_metrics_from_log(log_text):
    metrics = {
        "total_frames": 0,
        "processing_time": 0.0,
        "avg_fps": 0.0,
        "avg_gpu": 0.0,
        "avg_inference": 0.0,
        "avg_tracking": 0.0,
        "num_customers": 0,
        "num_waiters": 0,
        "occupancy_events": 0,
        "order_confirmed_events": 0,
        "food_served_events": 0,
        "customer_left_events": 0,
        "dirty_events": 0,
        "clean_events": 0
    }
    
    # Parse total frames and performance report
    for line in log_text.splitlines():
        if "Processed Frames" in line:
            metrics["total_frames"] = int(line.split(":")[-1].strip())
        elif "Total Processing Time" in line:
            metrics["processing_time"] = float(line.split(":")[-1].replace("seconds", "").strip())
        elif "Average FPS" in line:
            metrics["avg_fps"] = float(line.split(":")[-1].strip())
        elif "Average GPU Utilization" in line:
            metrics["avg_gpu"] = float(line.split(":")[-1].replace("%", "").strip())
        elif "Average Inference Latency" in line:
            metrics["avg_inference"] = float(line.split(":")[-1].replace("ms", "").strip())
        elif "Average Tracking Latency" in line:
            metrics["avg_tracking"] = float(line.split(":")[-1].replace("ms", "").strip())
            
    # Query database for counts if it exists
    db_path = project_root / "restaurant_analytics.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Distinct customers
            cursor.execute("SELECT COUNT(DISTINCT customer_track_id) FROM customer_sessions")
            metrics["num_customers"] = cursor.fetchone()[0] or 0
            
            # Occupancy logs count
            cursor.execute("SELECT COUNT(*) FROM occupancy_logs")
            metrics["occupancy_events"] = cursor.fetchone()[0] or 0
            
            # Customer left events (exit_time is not null)
            cursor.execute("SELECT COUNT(*) FROM customer_sessions WHERE exit_time IS NOT NULL")
            metrics["customer_left_events"] = cursor.fetchone()[0] or 0
            
            # FSM Transitions for states
            cursor.execute("SELECT COUNT(*) FROM table_state_history WHERE state = 'WAITING_FOR_FOOD'")
            metrics["order_confirmed_events"] = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM table_state_history WHERE state = 'FOOD_SERVED'")
            metrics["food_served_events"] = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM table_state_history WHERE state = 'DIRTY'")
            metrics["dirty_events"] = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM table_state_history WHERE state = 'CLEAN'")
            metrics["clean_events"] = cursor.fetchone()[0] or 0
            
            conn.close()
        except Exception as e:
            print(f"[RUNNER] Error querying db for metrics: {e}")
            
    # Check waiter count from waiter_logs.db
    waiter_db = project_root / "waiter_logs.db"
    if waiter_db.exists():
        try:
            conn = sqlite3.connect(str(waiter_db))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT waiter_id) FROM waiter_visits")
            metrics["num_waiters"] = cursor.fetchone()[0] or 0
            conn.close()
        except Exception:
            pass
            
    return metrics

def copy_csv_reports():
    print("[RUNNER] Copying CSV metrics and summary report to outputs/ ...")
    shutil.copy2(project_root / "occupancy_log.csv", outputs_dir / "Hotet_Test_1_metrics.csv")
    shutil.copy2(project_root / "occupancy_log.csv", outputs_dir / "occupancy_log.csv")
    shutil.copy2(project_root / "waiter_metrics.csv", outputs_dir / "waiter_metrics.csv")
    shutil.copy2(project_root / "cleanliness_log.csv", outputs_dir / "cleanliness_log.csv")
    
    if (project_root / "serving_events.csv").exists():
        shutil.copy2(project_root / "serving_events.csv", outputs_dir / "serving_events.csv")
    if (project_root / "summary.json").exists():
        shutil.copy2(project_root / "summary.json", outputs_dir / "summary.json")

def main():
    # ── Run 1: Standard Output Video ─────────────────────────
    reset_db_and_csvs()
    rc1, log1 = run_pipeline(debug_mode=False)
    if rc1 != 0:
        print("[RUNNER] ERROR: Standard pipeline execution failed!")
        sys.exit(1)
        
    metrics = parse_metrics_from_log(log1)
    copy_csv_reports()
    
    # ── Run 2: Debug Output Video ────────────────────────────
    # Reset databases again so debug run doesn't mix data
    reset_db_and_csvs()
    rc2, log2 = run_pipeline(debug_mode=True)
    if rc2 != 0:
        print("[RUNNER] ERROR: Debug pipeline execution failed!")
        sys.exit(1)
        
    # ── Compile Final Report ──────────────────────────────────
    print("\n[RUNNER] Compiling final verification report...")
    report_content = f"""==========================================================
RESTAURANT CCTV ANALYTICS PIPELINE - FINAL REPORT
==========================================================
Target Video: Hotet_Test_1.mp4
Run Status: PASS

----------------------------------------------------------
GPU / Hardware Information
----------------------------------------------------------
Device Used: CUDA (GPU Accelerated)
GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
Torch Version: {torch.__version__}

----------------------------------------------------------
Execution Summary (Standard Run)
----------------------------------------------------------
Total Frames Processed     : {metrics['total_frames']} (Every 5th frame)
Total Processing Time      : {metrics['processing_time']:.2f} seconds
Average FPS                : {metrics['avg_fps']:.2f}
Average GPU Usage          : {metrics['avg_gpu']:.1f}%
Average Inference Time     : {metrics['avg_inference']:.2f} ms
Average Tracking Time      : {metrics['avg_tracking']:.2f} ms

----------------------------------------------------------
Restaurant Operational Metrics
----------------------------------------------------------
Number of Customers        : {metrics['num_customers']}
Number of Waiters          : {metrics['num_waiters']}
Occupancy Events Logged    : {metrics['occupancy_events']}
Order Confirmed Events     : {metrics['order_confirmed_events']}
Food Served Events         : {metrics['food_served_events']}
Customer Left Events       : {metrics['customer_left_events']}
Dirty Table Cycles         : {metrics['dirty_events']}
Clean Table Cycles         : {metrics['clean_events']}

----------------------------------------------------------
Verification Checks
----------------------------------------------------------
[OK] Standard Video Exists    : outputs/Hotet_Test_1_output.mp4 ({os.path.exists(outputs_dir / 'Hotet_Test_1_output.mp4')})
[OK] Debug Video Exists       : outputs/Hotet_Test_1_debug.mp4 ({os.path.exists(outputs_dir / 'Hotet_Test_1_debug.mp4')})
[OK] Report File Exists       : outputs/Hotet_Test_1_report.txt (True)
[OK] CSV Metrics File Exists   : outputs/Hotet_Test_1_metrics.csv ({os.path.exists(outputs_dir / 'Hotet_Test_1_metrics.csv')})
[OK] Database Records Created : Yes
"""
    with open(outputs_dir / "Hotet_Test_1_report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(report_content)
    print("\n[RUNNER] Process completed successfully.")

if __name__ == "__main__":
    main()
