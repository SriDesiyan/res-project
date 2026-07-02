import sqlite3
from pathlib import Path

db_path = Path("c:/Users/desiy/Downloads/coe-intern-main (1)/coe-intern-main/restaurant_analytics.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

cursor.execute("""
    SELECT timestamp, occupancy_count, waiter_count, is_occupied 
    FROM occupancy_logs 
    WHERE table_id = 'table_2' AND timestamp >= 2950.0
    ORDER BY timestamp
""")
rows = cursor.fetchall()
print("Occupancy logs for table_2 (t >= 2950s):")
for r in rows:
    print(f"Time: {r[0]:.2f}s | Cust Count: {r[1]} | Waiter Present: {r[2]} | Is Occupied: {r[3]}")

conn.close()
