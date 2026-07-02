import sqlite3
from pathlib import Path

db_path = Path("c:/Users/desiy/Downloads/coe-intern-main (1)/coe-intern-main/serving_logs.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()
rows = cursor.execute("SELECT id, waiter_id, table_id, food_type, timestamp, frame_number, confidence FROM serving_events WHERE frame_number >= 37400 AND frame_number <= 40000 ORDER BY frame_number").fetchall()
print(f"Serving events in default window [37400, 40000]: {len(rows)}")
for r in rows:
    print(r)
conn.close()
