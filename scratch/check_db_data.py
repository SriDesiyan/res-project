import sqlite3
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()

for db_name in ["restaurant_analytics.db", "waiter_logs.db", "serving_logs.db"]:
    db_path = project_root / db_name
    print(f"\n=== Database: {db_name} ({db_path}) ===")
    if not db_path.exists():
        print("  Database does not exist.")
        continue
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        for t in tables:
            t_name = t[0]
            count = cursor.execute(f"SELECT COUNT(*) FROM {t_name}").fetchone()[0]
            print(f"  Table: {t_name:30s} | Rows: {count}")
            # print schema
            schema = cursor.execute(f"PRAGMA table_info({t_name})").fetchall()
            for col in schema:
                print(f"    - {col[1]} ({col[2]})")
        conn.close()
    except Exception as e:
        print(f"  Error: {e}")
