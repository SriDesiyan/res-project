import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from cleanliness.cleanliness_engine import CleanlinessEngine

def test_cleanliness_without_cleaning_state():
    engine = CleanlinessEngine(["table_1"])
    
    # Check that CLEANING is not in valid states
    from cleanliness.cleanliness_engine import TableCleanlinessState
    print("Valid states:", TableCleanlinessState.VALID_STATES)
    assert "CLEANING" not in TableCleanlinessState.VALID_STATES
    
    # 1. CLEAN -> OCCUPIED
    print("\n1. CLEAN + occupied -> OCCUPIED")
    res = engine.update("table_1", is_occupied=True, waiter_present=False, dirty_object_count=0, frame_time=1.0)
    state = engine.get_state("table_1")["state"]
    print("State:", state)
    assert state == "OCCUPIED"
    
    # 2. OCCUPIED -> TEMPORARY_ABSENCE
    print("\n2. OCCUPIED + unoccupied -> TEMPORARY_ABSENCE")
    res = engine.update("table_1", is_occupied=False, waiter_present=False, dirty_object_count=0, frame_time=2.0)
    state = engine.get_state("table_1")["state"]
    print("State:", state)
    assert state == "TEMPORARY_ABSENCE"
    
    # 3. TEMPORARY_ABSENCE + waiter present -> stays TEMPORARY_ABSENCE (no CLEANING state)
    print("\n3. TEMPORARY_ABSENCE + waiter present -> stays TEMPORARY_ABSENCE")
    res = engine.update("table_1", is_occupied=False, waiter_present=True, dirty_object_count=2, frame_time=3.0)
    state = engine.get_state("table_1")["state"]
    print("State:", state)
    assert state == "TEMPORARY_ABSENCE"
    
    # 4. TEMPORARY_ABSENCE + timeout with dirty objects -> DIRTY
    print("\n4. TEMPORARY_ABSENCE + timeout -> DIRTY")
    res = engine.update("table_1", is_occupied=False, waiter_present=False, dirty_object_count=2, frame_time=8.0) # > 5s limit
    state = engine.get_state("table_1")["state"]
    print("State:", state)
    assert state == "DIRTY"
    
    # 5. DIRTY + waiter present -> stays DIRTY (no CLEANING state)
    print("\n5. DIRTY + waiter present -> stays DIRTY")
    res = engine.update("table_1", is_occupied=False, waiter_present=True, dirty_object_count=2, frame_time=9.0)
    state = engine.get_state("table_1")["state"]
    print("State:", state)
    assert state == "DIRTY"
    
    # 6. DIRTY + dirty objects cleared -> CLEAN
    print("\n6. DIRTY + dishes cleared -> CLEAN")
    res = engine.update("table_1", is_occupied=False, waiter_present=False, dirty_object_count=0, frame_time=10.0)
    state = engine.get_state("table_1")["state"]
    print("State:", state)
    assert state == "CLEAN"
    
    print("\nAll cleanliness engine tests passed successfully!")

def test_pipeline_locking_simulation():
    # Simulate the pipeline locking logic
    table_is_locked_occupied = {}
    active_customer_tables = {}
    active_db_sessions = {}
    
    # Helper to check if occupied locked
    def get_is_occupied_locked(tid):
        return any(active_customer_tables.get(cid) == tid for cid in active_db_sessions)
        
    tid = "table_1"
    
    print("\nTesting pipeline occupancy locking simulation:")
    # 1. Customer arrives and is verified after delay -> is_occupied = True
    occ_is_occupied = True
    active_db_sessions["C1"] = "session_1"
    active_customer_tables["C1"] = tid
    
    if occ_is_occupied:
        table_is_locked_occupied[tid] = True
    elif table_is_locked_occupied.get(tid, False):
        has_active = get_is_occupied_locked(tid)
        if not has_active:
            table_is_locked_occupied[tid] = False
            
    is_occupied_effective = table_is_locked_occupied.get(tid, False)
    print("  Customer verified. Effective occupied status:", is_occupied_effective)
    assert is_occupied_effective is True
    
    # 2. Customer goes missing (tracker loss), but grace period not expired
    occ_is_occupied = False
    # C1 is still in active_db_sessions
    if occ_is_occupied:
        table_is_locked_occupied[tid] = True
    elif table_is_locked_occupied.get(tid, False):
        has_active = get_is_occupied_locked(tid)
        if not has_active:
            table_is_locked_occupied[tid] = False
            
    is_occupied_effective = table_is_locked_occupied.get(tid, False)
    print("  Tracker lost customer, active session remains. Effective occupied status:", is_occupied_effective)
    assert is_occupied_effective is True # LOCKED!
    
    # 3. Customer grace period expires -> C1 is removed from active sessions
    active_db_sessions.pop("C1")
    
    if occ_is_occupied:
        table_is_locked_occupied[tid] = True
    elif table_is_locked_occupied.get(tid, False):
        has_active = get_is_occupied_locked(tid)
        if not has_active:
            table_is_locked_occupied[tid] = False
            
    is_occupied_effective = table_is_locked_occupied.get(tid, False)
    print("  Session expired. Effective occupied status:", is_occupied_effective)
    assert is_occupied_effective is False # UNLOCKED!
    
    print("\nAll pipeline locking tests passed successfully!")

if __name__ == "__main__":
    test_cleanliness_without_cleaning_state()
    test_pipeline_locking_simulation()
