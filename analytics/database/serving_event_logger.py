import sqlite3
import datetime
from pathlib import Path

class ServingEventLogger:
    def __init__(self, db_name="serving_logs.db"):
        project_root = Path(__file__).parent.parent.parent.resolve()
        self.db_path = project_root / db_name
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS serving_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                waiter_id TEXT,
                table_id TEXT,
                food_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                frame_number INTEGER,
                confidence REAL
            )
        """)
        conn.commit()
        conn.close()

    def log_serving(self, waiter_id: str, table_id: str, food_type: str, frame_num: int, confidence: float):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        created_at_str = datetime.datetime.utcnow().isoformat()
        cursor.execute("""
            INSERT INTO serving_events (waiter_id, table_id, food_type, timestamp, frame_number, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(waiter_id), str(table_id) if table_id else None, str(food_type), created_at_str, int(frame_num), float(confidence)))
        conn.commit()
        conn.close()
        print(f"[SUCCESS] Logged Serving: Waiter {waiter_id} serving {food_type} at table {table_id}")
