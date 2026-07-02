import sys
sys.path.insert(0, 'analytics')
import sqlite3

conn = sqlite3.connect('analytics/restaurant_analytics.db')
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print(f'Tables: {tables}')

for tbl in tables:
    c.execute(f'SELECT COUNT(*) FROM [{tbl}]')
    cnt = c.fetchone()[0]
    print(f'  {tbl}: {cnt} rows')

if 'table_state_history' in tables:
    c.execute('''
        SELECT table_id, previous_state, new_state, start_time, trigger
        FROM table_state_history
        ORDER BY table_id, start_time
        LIMIT 60
    ''')
    rows = c.fetchall()
    print()
    print('=== FSM State Transitions ===')
    for r in rows:
        prev = str(r[1]) if r[1] else 'None'
        print(f'  {r[0]:10s}  {prev:22s} -> {r[2]:22s}  t={r[3]:.1f}s  {r[4]}')
conn.close()
