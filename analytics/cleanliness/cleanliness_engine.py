# analytics/cleanliness/cleanliness_engine.py
"""
Cleanliness Engine — Tracks table cleanliness cycles.
"""

class TableCleanlinessState:
    VALID_STATES = {"CLEAN", "OCCUPIED", "TEMPORARY_ABSENCE", "DIRTY"}

    def __init__(self, table_id: str):
        self.table_id = table_id
        self.state = "CLEAN"
        self.absent_since = None


class CleanlinessEngine:
    def __init__(self, table_ids: list):
        self.tables = {tid: TableCleanlinessState(tid) for tid in table_ids}

    def get_state(self, table_id: str) -> dict:
        if table_id in self.tables:
            return {"state": self.tables[table_id].state}
        return {"state": "CLEAN"}

    def update(self, table_id: str, is_occupied: bool, waiter_present: bool,
               dirty_object_count: int, frame_time: float) -> bool:
        if table_id not in self.tables:
            self.tables[table_id] = TableCleanlinessState(table_id)

        t = self.tables[table_id]
        old_state = t.state

        if t.state == "CLEAN":
            if is_occupied:
                t.state = "OCCUPIED"

        elif t.state == "OCCUPIED":
            if not is_occupied:
                t.state = "TEMPORARY_ABSENCE"
                t.absent_since = frame_time

        elif t.state == "TEMPORARY_ABSENCE":
            if is_occupied:
                t.state = "OCCUPIED"
                t.absent_since = None
            else:
                if t.absent_since is not None and (frame_time - t.absent_since > 5.0):
                    t.state = "DIRTY"
                    t.absent_since = None

        elif t.state == "DIRTY":
            if is_occupied:
                t.state = "OCCUPIED"
            elif dirty_object_count == 0:
                t.state = "CLEAN"

        return t.state != old_state
