import sqlite3
from pathlib import Path

db_path = Path("c:/Users/desiy/Downloads/coe-intern-main (1)/coe-intern-main/restaurant_analytics.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(customer_sessions)")
columns = [col[1] for col in cursor.fetchall()]

cursor.execute("SELECT * FROM customer_sessions ORDER BY id")
rows = cursor.fetchall()
print("Customer Sessions:")
for row in rows:
    row_dict = dict(zip(columns, row))
    print(f"ID {row_dict['id']}: Track {row_dict['customer_track_id']} | Table {row_dict['table_id']} | Session {row_dict['session_uuid']} | Entry: {row_dict['entry_time']} | Exit: {row_dict['exit_time']} | Dur: {row_dict['duration_seconds']}")

conn.close()
