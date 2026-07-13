import sqlite3
import os

dbs = [
    'restaurant_analytics.db',
    'analytics/restaurant_analytics.db',
    'waiter_logs.db',
    'serving_logs.db'
]

for db in dbs:
    if not os.path.exists(db):
        print(f"{db} does not exist")
        continue
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"{db} tables: {tables}")
        for t in tables:
            cursor.execute(f"SELECT count(*) FROM [{t}]")
            print(f"  {t}: {cursor.fetchone()[0]} rows")
        conn.close()
    except Exception as e:
        print(f"Error reading {db}: {e}")
