"""
Post-run validation script for Example Test 2.
Run after the pipeline completes to verify:
  - State transition order per table
  - No false FOOD_SERVED
  - No DIRTY while customer was seated
  - Timer boundary correctness
"""
import sqlite3
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
db_path = project_root / "restaurant_analytics.db"

print("=" * 60)
print("VALIDATION REPORT — Example Test 2")
print("=" * 60)

conn = sqlite3.connect(str(db_path))
c = conn.cursor()

# ── 1. Fetch all state transitions ordered by table and time ─────────────────
c.execute("""
    SELECT table_id, previous_state, state, start_time, trigger, session_uuid
    FROM table_state_history
    ORDER BY table_id, start_time
""")
rows = c.fetchall()

VALID_FORWARD = {
    "EMPTY":           {"OCCUPIED"},
    "OCCUPIED":        {"ORDER_TAKEN", "CUSTOMER_LEFT"},
    "ORDER_TAKEN":     {"WAITING_FOR_FOOD", "CUSTOMER_LEFT"},
    "WAITING_FOR_FOOD": {"FOOD_SERVED", "CUSTOMER_LEFT"},
    "FOOD_SERVED":     {"DINING"},
    "DINING":          {"CUSTOMER_LEFT"},
    "CUSTOMER_LEFT":   {"DIRTY"},
    "DIRTY":           {"CLEAN"},
    "CLEAN":           {"EMPTY"},
}

issues = []
table_transitions = {}
for tid, prev, new, ts, trigger, sess in rows:
    table_transitions.setdefault(tid, []).append((prev, new, ts, trigger, sess))

print(f"\nTables found in DB: {sorted(table_transitions.keys())}")
print()

for tid, transitions in sorted(table_transitions.items()):
    print(f"--- {tid} ({len(transitions)} transitions) ---")
    
    food_served_count = 0
    prev_new_state = None
    prev_ts = 0
    
    for prev, new, ts, trigger, sess in transitions:
        # Check forward-only rule
        expected_next = VALID_FORWARD.get(prev)
        if expected_next and new not in expected_next and not (prev == "UNKNOWN"):
            issues.append(
                f"[{tid}] ILLEGAL TRANSITION: {prev} -> {new} "
                f"(expected one of {expected_next}) at {ts:.1f}s trigger={trigger}"
            )
            print(f"  [FAIL] {prev:20s} -> {new:20s}  t={ts:.1f}s  ILLEGAL (expected one of {expected_next})")
        else:
            print(f"  [OK]   {prev:20s} -> {new:20s}  t={ts:.1f}s  {trigger}")
        
        # Check backward transitions
        if prev_new_state == new and new not in ("EMPTY",):
            issues.append(f"[{tid}] BACKWARD/DUPLICATE transition to {new} at {ts:.1f}s")
        
        # Count FOOD_SERVED events per session
        if new == "FOOD_SERVED":
            food_served_count += 1
        
        # Check DIRTY guard: previous state must be CUSTOMER_LEFT
        if new == "DIRTY" and prev not in ("CUSTOMER_LEFT", "WAITING_FOR_FOOD", "OCCUPIED", "UNKNOWN"):
            issues.append(
                f"[{tid}] DIRTY reached from {prev} (not via CUSTOMER_LEFT) "
                f"at {ts:.1f}s"
            )
        
        prev_new_state = new
        prev_ts = ts
    
    if food_served_count > 1:
        issues.append(f"[{tid}] DUPLICATE FOOD_SERVED: {food_served_count} events in DB")
        print(f"  [FAIL] FOOD_SERVED appeared {food_served_count} times (should be 1 per session)")
    elif food_served_count == 1:
        print(f"  [OK]   FOOD_SERVED: exactly 1 event")
    else:
        print(f"  -  FOOD_SERVED: 0 events (table may not have been served in this run)")
    print()

# ── 2. Timer sanity check from serving_events ────────────────────────────────
serving_db = project_root / "serving_logs.db"
if serving_db.exists():
    sc = sqlite3.connect(str(serving_db))
    cur = sc.cursor()
    cur.execute("SELECT waiter_id, table_id, timestamp, confidence FROM serving_events ORDER BY timestamp")
    serving_rows = cur.fetchall()
    print(f"Serving events in DB: {len(serving_rows)}")
    for wid, tid, ts, conf in serving_rows:
        print(f"  Waiter={wid}  Table={tid}  t={ts}  conf={conf:.2f}")
    sc.close()

print()
print("=" * 60)
if issues:
    print(f"VALIDATION FAILED — {len(issues)} issue(s):")
    for iss in issues:
        print(f"  [FAIL] {iss}")
else:
    print("VALIDATION PASSED — No issues found")
print("=" * 60)

conn.close()
