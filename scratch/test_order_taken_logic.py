import sys
import sqlite3
from pathlib import Path
import shutil

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

# We want to test the order-taken logic logic in isolation
# Let's create a mock database and test the state transitions.

db_path = project_root / "waiter_logs_test.db"

def init_waiter_db():
    if db_path.exists():
        db_path.unlink()
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
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO waiter_visits (waiter_id, table_id, timestamp, log_type, created_at)
        VALUES (?, ?, ?, ?, datetime('now'))
    """, (str(waiter_id), str(table_id), float(timestamp), "order taken"))
    conn.commit()
    conn.close()

class MockPerson:
    def __init__(self, session_id):
        self.session_id = session_id

class MockTable:
    def __init__(self, current_waiters):
        self.current_waiters = current_waiters

def test_order_taken_simulation():
    init_waiter_db()
    
    # State tracking variables
    waiter_order_writing_start = {}
    table_waiters_logged = {}
    order_taken_tables = {}
    table_waiter_absent_since = {}
    
    # Simulation parameters
    tid = "table_1"
    wid = "W1"
    fps = 25.0
    
    table_waiters_logged[tid] = set()
    
    # Frame loop simulation
    print("Starting Order Taken Simulation...")
    for frame in range(1, 150):
        frame_time = frame / fps
        
        # Table occupancy state
        is_occupied = True
        waiter_present = True if frame <= 110 else False # waiter leaves at frame 110
        
        # mock current waiters
        current_waiters = {wid} if waiter_present else set()
        table_obj = MockTable(current_waiters=current_waiters)
        persons = [MockPerson(wid)] if waiter_present else []
        
        # Simulate detect_waiter_serving output
        # Let's say waiter is writing between frame 20 and frame 105
        is_writing = (20 <= frame <= 105)
        
        # Run our logic
        if is_occupied:
            if waiter_present:
                for w in table_obj.current_waiters:
                    table_waiter_absent_since.pop((tid, w), None)
                    
                    if is_writing:
                        if (tid, w) not in waiter_order_writing_start:
                            waiter_order_writing_start[(tid, w)] = frame_time
                        else:
                            write_duration = frame_time - waiter_order_writing_start[(tid, w)]
                            if write_duration >= 3.0:
                                if w not in table_waiters_logged[tid]:
                                    log_waiter_visit_to_separate_db(w, tid, frame_time)
                                    table_waiters_logged[tid].add(w)
                                    print(f"Frame {frame:3d} ({frame_time:.2f}s) | Logged waiter {w} order taken (writing for {write_duration:.2f}s)")
                    else:
                        waiter_order_writing_start.pop((tid, w), None)
            
            # Flash if any active waiter has taken an order during this visit
            if any(w in table_waiters_logged[tid] for w in table_obj.current_waiters):
                order_taken_tables[tid] = True
            else:
                order_taken_tables[tid] = False
                
            # If waiter leaves the table, track absence
            for w in list(table_waiters_logged[tid]):
                if w not in table_obj.current_waiters:
                    if (tid, w) not in table_waiter_absent_since:
                        table_waiter_absent_since[(tid, w)] = frame_time
                    elif frame_time - table_waiter_absent_since[(tid, w)] > 5.0:
                        table_waiters_logged[tid].remove(w)
                        table_waiter_absent_since.pop((tid, w), None)
                        waiter_order_writing_start.pop((tid, w), None)
                        print(f"Frame {frame:3d} ({frame_time:.2f}s) | Waiter {w} absent for 5s, cleared logged state")
        
        # Log visual status at specific milestones
        if frame in (10, 20, 30, 94, 95, 96, 100, 106, 107, 110, 111, 120, 130):
            w_start = waiter_order_writing_start.get((tid, wid), None)
            dur = (frame_time - w_start) if w_start else 0.0
            print(f"Frame {frame:3d} ({frame_time:.2f}s) | present: {str(waiter_present):5s} | writing: {str(is_writing):5s} | write_dur: {dur:.2f}s | logged_set: {list(table_waiters_logged[tid])} | flash: {order_taken_tables.get(tid, False)}")

    # Check SQL DB contents
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM waiter_visits").fetchall()
    print("\nSQLite Database 'waiter_visits' entries:")
    for r in rows:
        print(f"  Row: {r}")
    conn.close()
    
    # Assertions
    assert len(rows) == 1, f"Expected exactly 1 log entry, found {len(rows)}"
    assert rows[0][1] == wid
    assert rows[0][2] == tid
    assert rows[0][4] == "order taken"
    
    if db_path.exists():
        db_path.unlink()
    print("Simulation assertion passed successfully!")

if __name__ == "__main__":
    test_order_taken_simulation()
