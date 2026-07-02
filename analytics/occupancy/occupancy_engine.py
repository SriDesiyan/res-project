"""
Occupancy Engine — Spatial-Centric Per-table Occupancy Analytics.

Uses Bottom-Center anchor points + cv2.pointPolygonTest for direct polygon
containment checks. Includes hysteresis dwell buffer, 300-frame grace period,
blob-based multi-occupant inference, and temporal voting for stable counts.
"""
from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class TableOccupancy:
    table_id: str
    current_customers: set = field(default_factory=set)    
    current_waiters: set = field(default_factory=set)     
    is_occupied: bool = False
    total_visits: int = 0                  
    total_visit_duration: float = 0.0     
    peak_occupancy: int = 0               
    waiter_visits: int = 0                
    idle_since: float = None              
    occupied_start_time: float = None     
    _seen_customer_ids: set = field(default_factory=set)   

    dwell_timers: dict = field(default_factory=dict)       
    grace_timers: dict = field(default_factory=dict)       

    session_checked_in: set = field(default_factory=set)
    session_max_occupants: int = 0

    displayed_count: int = 0
    _candidate_count: int = 0
    _candidate_frames: int = 0

    _single_person_area: float = 0.0
    _area_samples: int = 0
    _occupancy_candidate_start: float = None
    raw_customer_present: bool = False
    _raw_customer_grace: int = 0  # frames since last confirmed raw detection


class OccupancyEngine:
    # table occupancy logic with the customer dwell time threshold, grace period, and blob inference multiplier

    DWELL_THRESHOLD = 75        
    GRACE_PERIOD = 250          
    VELOCITY_LIMIT = 9999.0     
    TEMPORAL_VOTE_FRAMES = 10   
    BLOB_MULTIPLIER = 1.5       
    SPATIAL_TOLERANCE_PX = 15.0 
    OCCUPANCY_DELAY_SECONDS = 0.0  # Removed: DWELL_THRESHOLD=75 frames already provides 3s stability
    RAW_CUSTOMER_GRACE_FRAMES = 5  # frames of YOLO flicker tolerance for raw_customer_present

    def __init__(self, table_ids: list[str], tables: dict):
        self.tables_state = {}
        self.table_polygons = {}  

        for raw_id in table_ids:
            self.tables_state[raw_id] = TableOccupancy(table_id=raw_id)
            if raw_id in tables:
                poly = np.array(tables[raw_id]["polygon"], dtype=np.int32)
                self.table_polygons[raw_id] = poly

    @property
    def tables(self):
        return self.tables_state

    def _get_table_for_point(self, point: tuple) -> str | None:
        # closest roi table to a point
        px, py = float(point[0]), float(point[1])
        best_table = None
        best_dist = -float('inf')  

        for tid, poly in self.table_polygons.items():
            dist = cv2.pointPolygonTest(poly, (px, py), measureDist=True)
            
            if dist > best_dist:
                best_dist = dist
                best_table = tid

       
        if best_dist >= -self.SPATIAL_TOLERANCE_PX:
            return best_table
            
        return None

    def update(self, persons: list, frame_time: float):
       # update table occupancy status
        frame_customers = {tid: set() for tid in self.tables_state}
        frame_waiters = {tid: set() for tid in self.tables_state}

        person_velocities = {getattr(p, "session_id", f"T{p.track_id}"): getattr(p, "velocity", 0.0) for p in persons}
        person_bboxes = {getattr(p, "session_id", f"T{p.track_id}"): p.bbox for p in persons if hasattr(p, "bbox")}

        for person in persons:
            if not person.confirmed:
                continue            
            x1, y1, x2, y2 = person.bbox
            anchor = ((x1 + x2) / 2.0, float(y2))  # (x_center, y_max)
            pid = getattr(person, "session_id", f"T{person.track_id}")

            if person.role == "waiter":
                best_table = None
                best_dist = -float('inf')
                for tid, poly in self.table_polygons.items():
                    dist = cv2.pointPolygonTest(poly, anchor, measureDist=True)
                    if dist > best_dist:
                        best_dist = dist
                        best_table = tid
                if best_table and best_dist >= -180.0:
                    frame_waiters[best_table].add(pid)
            else:
                table_id = self._get_table_for_point(anchor)
                if table_id is not None:
                    frame_customers[table_id].add(pid)

        for table_id, table in self.tables_state.items():
            new_customers = frame_customers[table_id]
            new_waiters = frame_waiters[table_id]
            # Save raw customer presence for FSM to bypass grace-period and Re-ID locking.
            # Apply a short flicker grace (RAW_CUSTOMER_GRACE_FRAMES) to absorb single-frame
            # YOLO detection gaps without letting full grace-period customers inflate this flag.
            if len(new_customers) > 0:
                table.raw_customer_present = True
                table._raw_customer_grace = 0
            else:
                table._raw_customer_grace += 1
                if table._raw_customer_grace > self.RAW_CUSTOMER_GRACE_FRAMES:
                    table.raw_customer_present = False
                # else: keep raw_customer_present=True through the grace window

            for cid in list(table.dwell_timers.keys()):
                if cid not in new_customers:
                    table.grace_timers[cid] = table.grace_timers.get(cid, 0) + 1
                    if table.grace_timers[cid] > self.GRACE_PERIOD:
                        del table.dwell_timers[cid]
                        table.grace_timers.pop(cid, None)
                else:
                    table.grace_timers.pop(cid, None)

            for cid in new_customers:
                vel = person_velocities.get(cid, 0.0)
                if vel <= self.VELOCITY_LIMIT:
                    table.dwell_timers[cid] = table.dwell_timers.get(cid, 0) + 1

            verified_customers = {
                cid for cid, frames in table.dwell_timers.items()
                if frames >= self.DWELL_THRESHOLD
            }

            for cid in verified_customers:
                table.session_checked_in.add(cid)

            raw_count = len(verified_customers)

            blob_inferred = 0
            for cid in verified_customers:
                if cid in person_bboxes:
                    x1, y1, x2, y2 = person_bboxes[cid]
                    w_box = x2 - x1
                    h_box = y2 - y1
                    area = w_box * h_box
                    
                    # Method A: Running Area Comparison
                    if table._area_samples < 50:
                        table._area_samples += 1
                        table._single_person_area += (
                            (area - table._single_person_area) / table._area_samples
                        )
                    if (table._single_person_area > 0
                            and area > table._single_person_area * self.BLOB_MULTIPLIER):
                        blob_inferred += 1
                        
                    # Method B: Aspect Ratio Heuristic (Failsafe for merged horizontal blobs)
                    # A seated individual is almost always taller than wide (or close to 1.0).
                    # If the box is noticeably wider than tall, it encapsulates two adjacent persons.
                    aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
                    if aspect_ratio > 1.6:  # Width catches up to height -> 2 people
                        blob_inferred += 1
                        # Prevent double counting from same box?
                        # Actually we cap at logical table maximum if needed, but let's allow inference stacking for now
                        # as they satisfy separate geometrical truths.
                    
            # Enforce logical limit: Max 4 people per table per timestamp
            effective_count = min(4, raw_count + blob_inferred)
            table.session_max_occupants = max(table.session_max_occupants, effective_count)

            # --- Turnover Counting ---
            for cid in verified_customers:
                if cid not in table._seen_customer_ids:
                    table._seen_customer_ids.add(cid)
                    table.total_visits += 1

            # --- Waiter Visit Counting ---
            for wid in new_waiters:
                if wid not in table.current_waiters:
                    table.waiter_visits += 1

            table.current_customers = verified_customers
            table.current_waiters = new_waiters

            if effective_count == 0 and len(table.session_checked_in) == 0:
                table.displayed_count = 0
                table._candidate_count = 0
                table._candidate_frames = 0
            elif effective_count != table.displayed_count:
                if effective_count == table._candidate_count:
                    table._candidate_frames += 1
                else:
                    table._candidate_count = effective_count
                    table._candidate_frames = 1
                if table._candidate_frames >= self.TEMPORAL_VOTE_FRAMES:
                    table.displayed_count = table._candidate_count
                    table._candidate_frames = 0
            else:
                table._candidate_frames = 0

            if table.displayed_count > 0:
                if table._occupancy_candidate_start is None:
                    table._occupancy_candidate_start = frame_time
                elif frame_time - table._occupancy_candidate_start >= self.OCCUPANCY_DELAY_SECONDS:
                    table.is_occupied = True
            else:
                table.is_occupied = False
                table._occupancy_candidate_start = None
                table.session_checked_in.clear()
                table.session_max_occupants = 0

            table.peak_occupancy = max(table.peak_occupancy, table.displayed_count)

            if table.is_occupied:
                table.idle_since = None
                if table.occupied_start_time is None:
                    table.occupied_start_time = frame_time
            else:
                if table.idle_since is None:
                    table.idle_since = frame_time
                table.occupied_start_time = None

    def get_all_status(self, frame_time: float) -> list[dict]:
        results = []
        for table_id, t in self.tables_state.items():

            idle_sec = 0.0
            if t.idle_since is not None:
                idle_sec = frame_time - t.idle_since

            results.append({
                "table_id": table_id,
                "customer_count": t.displayed_count,
                "waiter_present": len(t.current_waiters) > 0,
                "is_occupied": t.is_occupied,
                "total_visits": t.total_visits,
                "peak_occupancy": t.peak_occupancy,
                "waiter_visits": t.waiter_visits,
                "idle_seconds": round(idle_sec, 1),
                "occupied_seconds": (
                    round(frame_time - t.occupied_start_time, 1)
                    if t.occupied_start_time is not None else 0.0
                ),
            })
        return results

    def get_table_status(self, table_id: str) -> dict:
        t = self.tables_state.get(table_id)
        if not t:
            return {}
        return {
            "table_id": t.table_id,
            "customer_count": t.displayed_count,
            "waiter_present": len(t.current_waiters) > 0,
            "is_occupied": t.is_occupied,
            "total_visits": t.total_visits,
            "peak_occupancy": t.peak_occupancy,
            "waiter_visits": t.waiter_visits,
        }
