# analytics/fsm/table_fsm.py
"""
Deterministic Table State Machine — Production Restaurant Analytics.

Enforced lifecycle (no skipping, no backward, no parallel):

  EMPTY → OCCUPIED → ORDER_TAKEN → WAITING_FOR_FOOD
       → FOOD_SERVED → DINING → CUSTOMER_LEFT → DIRTY → CLEAN → EMPTY

Key invariants:
  * ORDER_TAKEN is a transient checkpoint: it transitions to WAITING_FOR_FOOD
    immediately in the same update cycle. It fires the ORDER_TAKEN DB log and
    starts the Waiting Timer.
  * FOOD_SERVED is a transient confirmation state: the Dining Timer starts here
    immediately. The UI banner is a separate renderer effect. The FSM auto-
    advances to DINING in the next update cycle.
  * Exactly ONE FOOD_SERVED per customer session.
  * Waiter ownership: the waiter who took the order must be the waiter serving.
  * DIRTY is only reachable via CUSTOMER_LEFT (customer must have left first).
"""
import uuid
from analytics.config.fsm_config import FSM_CONFIG


class TableStateFSM:
    # Linear lifecycle — no skipping, no backward transitions.
    LIFECYCLE = [
        "EMPTY",
        "OCCUPIED",
        "ORDER_TAKEN",        # transient checkpoint
        "WAITING_FOR_FOOD",
        "FOOD_SERVED",        # transient confirmation (auto-advance to DINING)
        "DINING",
        "CUSTOMER_LEFT",
        "DIRTY",
        "CLEAN",
    ]

    VALID_TRANSITIONS = {
        "UNKNOWN":         {"EMPTY", "WAITING_FOR_FOOD", "DIRTY"},  # warm-up bootstrap
        "EMPTY":           {"OCCUPIED"},
        "OCCUPIED":        {"ORDER_TAKEN", "CUSTOMER_LEFT"},
        "ORDER_TAKEN":     {"WAITING_FOR_FOOD", "CUSTOMER_LEFT"},                    # transient
        "WAITING_FOR_FOOD": {"FOOD_SERVED", "CUSTOMER_LEFT"},
        "FOOD_SERVED":     {"DINING"},                              # transient
        "DINING":          {"CUSTOMER_LEFT"},
        "CUSTOMER_LEFT":   {"DIRTY"},
        "DIRTY":           {"CLEAN"},
        "CLEAN":           {"EMPTY"},
    }

    def __init__(self, table_id: str):
        self.table_id = table_id
        self.state = "UNKNOWN"
        self.previous_state = None
        self.state_start_time = 0.0
        self.session_start_times = {}

        # ── Warm-up tracking ────────────────────────────────────────────
        self.warm_up_complete = False
        self.first_frame_time = None
        self.warm_up_occupancy_frames = 0
        self.warm_up_dirty_frames = 0
        self.warm_up_total_frames = 0

        # ── Business Timers ─────────────────────────────────────────────
        # Customer Time:  starts at OCCUPIED, stops at CUSTOMER_LEFT
        self.customer_time_start = None   # set on → OCCUPIED
        self.customer_time_stop = None    # set on → CUSTOMER_LEFT

        # Waiting Time:   starts at ORDER_TAKEN, stops at FOOD_SERVED
        self.waiting_time_start = None    # set on → ORDER_TAKEN
        self.waiting_time_stop = None     # set on → FOOD_SERVED

        # Dining Time:    starts at FOOD_SERVED, stops at CUSTOMER_LEFT
        self.dining_time_start = None     # set on → FOOD_SERVED
        self.dining_time_stop = None      # set on → CUSTOMER_LEFT

        # Legacy aliases kept so renderer and DB code can still reference them
        # without a breaking rename.  They are synced inside transition_to().
        self.occupancy_start_time = None  # = customer_time_start
        self.order_confirmed_time = None  # = waiting_time_start
        self.food_wait_start_time = None  # = waiting_time_start
        self.food_served_time = None      # = dining_time_start
        self.customer_left_time = None    # = customer_time_stop
        self.dirty_start_time = None
        self.clean_time = None

        # ── Waiter ownership ────────────────────────────────────────────
        # Set when ORDER_TAKEN fires; must match when FOOD_SERVED is evaluated.
        self.order_taken_waiter_id = None

        # ── Session / logging state ─────────────────────────────────────
        self.session_uuid = None
        self._last_known_session_uuid = None
        self.order_logged = False

        # ── Presence tracking for guard evaluation ──────────────────────
        self.customer_present_start_time = None   # continuous presence start
        self.customer_absent_start_time = None    # continuous absence start

        # ── Gesture / order-taking accumulator ──────────────────────────
        self.gesture_window = []            # list of (timestamp, is_valid)
        self.last_active_gesture_time = None

        # ── FOOD_SERVED confirmation buffer ─────────────────────────────
        # Tracks how long the plate has been continuously detected inside the
        # correct table ROI with an active serving gesture.
        self.food_served_confirmation_start = None  # first valid detection time
        self.food_served_fired = False              # guard: only one per session

        # ── FOOD_SERVED auto-advance ─────────────────────────────────────
        # The state is transient — we advance to DINING on the next cycle.
        self._food_served_advance_pending = False

    # ────────────────────────────────────────────────────────────────────────
    # Transition helpers
    # ────────────────────────────────────────────────────────────────────────

    def can_transition(self, current: str, new_state: str) -> bool:
        """Return True only for strictly valid forward transitions."""
        # Identical state is NOT a valid transition (prevents silent re-fires).
        if current == new_state:
            return False
        return new_state in self.VALID_TRANSITIONS.get(current, set())

    def transition_to(self, new_state: str, frame_time: float, trigger: str,
                      db_manager=None, session_uuid=None) -> bool:
        if not self.can_transition(self.state, new_state):
            print(
                f"[FSM WARNING] Invalid transition: {self.state} -> {new_state} "
                f"for table {self.table_id} at {frame_time:.1f}s. Rejected."
            )
            return False

        old_state = self.state
        self.previous_state = old_state
        self.state = new_state
        self.state_start_time = frame_time

        # ── Session UUID bookkeeping ────────────────────────────────────
        if session_uuid is not None:
            self.session_uuid = session_uuid
            self._last_known_session_uuid = session_uuid
        elif new_state in ("OCCUPIED", "EMPTY"):
            self.session_uuid = None
            self._last_known_session_uuid = None
        else:
            self.session_uuid = self._last_known_session_uuid

        # ── Timer boundary logic ────────────────────────────────────────
        if new_state == "OCCUPIED":
            # If we have a session_uuid and we already recorded a start time for it, restore it!
            if session_uuid and session_uuid in self.session_start_times:
                self.customer_time_start = self.session_start_times[session_uuid]
            else:
                self.customer_time_start = frame_time
                if session_uuid:
                    self.session_start_times[session_uuid] = frame_time
            self.customer_time_stop = None
            self.waiting_time_start = None
            self.waiting_time_stop = None
            self.dining_time_start = None
            self.dining_time_stop = None
            # Sync legacy aliases
            self.occupancy_start_time = self.customer_time_start
            self.order_confirmed_time = None
            self.food_wait_start_time = None
            self.food_served_time = None
            self.customer_left_time = None
            self.dirty_start_time = None
            self.clean_time = None
            self.order_logged = False
            self.food_served_fired = False
            self.order_taken_waiter_id = None
            self.food_served_confirmation_start = None

        elif new_state == "ORDER_TAKEN":
            self.waiting_time_start = frame_time
            # Sync legacy aliases
            self.order_confirmed_time = frame_time
            self.food_wait_start_time = frame_time

        elif new_state == "WAITING_FOR_FOOD":
            # ORDER_TAKEN was just the checkpoint; waiting timer already started.
            pass

        elif new_state == "FOOD_SERVED":
            self.waiting_time_stop = frame_time
            self.dining_time_start = frame_time
            # Sync legacy aliases
            self.food_served_time = frame_time
            self._food_served_advance_pending = True  # auto-advance to DINING

        elif new_state == "DINING":
            # Dining timer already started at FOOD_SERVED
            pass

        elif new_state == "CUSTOMER_LEFT":
            self.customer_time_stop = frame_time
            self.dining_time_stop = frame_time
            # Sync legacy aliases
            self.customer_left_time = frame_time

        elif new_state == "DIRTY":
            self.dirty_start_time = frame_time

        elif new_state == "CLEAN":
            self.clean_time = frame_time

        elif new_state == "EMPTY":
            # Full reset for the next customer session
            self.customer_time_start = None
            self.customer_time_stop = None
            self.waiting_time_start = None
            self.waiting_time_stop = None
            self.dining_time_start = None
            self.dining_time_stop = None
            self.occupancy_start_time = None
            self.order_confirmed_time = None
            self.food_wait_start_time = None
            self.food_served_time = None
            self.customer_left_time = None
            self.dirty_start_time = None
            self.clean_time = None
            self.order_logged = False
            self.food_served_fired = False
            self.order_taken_waiter_id = None
            self.food_served_confirmation_start = None

        print(
            f"[FSM] Table {self.table_id}: {old_state} -> {new_state} "
            f"at {frame_time:.1f}s  trigger={trigger}"
        )

        # ── Database logging ────────────────────────────────────────────
        # Skip logging for UNKNOWN bootstrap and for the silent transient hops.
        if db_manager and old_state != "UNKNOWN":
            db_manager.log_table_state(
                table_id=self.table_id,
                new_state=new_state,
                start_time=frame_time,
                previous_state=old_state,
                trigger=trigger,
                event_id=str(uuid.uuid4()),
                confidence=1.0,
                session_uuid=session_uuid
            )
            if new_state == "DINING":
                db_manager.update_waiter_food_served(self.table_id, frame_time)

        return True

    def force_sequence_to(self, target_state: str, frame_time: float,
                          trigger_suffix: str, db_manager=None,
                          session_uuid=None):
        """Fast-forward through the lifecycle to reach target_state."""
        seq = self.LIFECYCLE
        try:
            start_idx = seq.index(self.state)
            target_idx = seq.index(target_state, start_idx)
        except ValueError:
            return

        for i in range(start_idx, target_idx):
            next_state = seq[i + 1]
            trigger = f"AUTO_{next_state}_{trigger_suffix}"
            self.transition_to(next_state, frame_time, trigger, db_manager, session_uuid)

    # ────────────────────────────────────────────────────────────────────────
    # Main update — called every processed frame
    # ────────────────────────────────────────────────────────────────────────

    def update(self,
               customer_present: bool,
               waiter_present: bool,
               is_writing: bool,
               is_serving: bool,
               dirty_object_count: int,
               frame_time: float,
               db_manager=None,
               session_uuid=None,
               # Extended arguments for FOOD_SERVED multi-condition guard
               serving_waiter_id: str = None,
               plate_in_roi: bool = False,
               serving_confidence: float = 0.0):

        config = FSM_CONFIG

        # ── Phase 1: Warm-Up ────────────────────────────────────────────
        if not self.warm_up_complete:
            if self.first_frame_time is None:
                self.first_frame_time = frame_time
            self.warm_up_total_frames += 1
            if customer_present:
                self.warm_up_occupancy_frames += 1
            if dirty_object_count >= 1:
                self.warm_up_dirty_frames += 1

            if frame_time - self.first_frame_time >= config["warm_up_period_seconds"]:
                self.warm_up_complete = True
                occ_ratio = self.warm_up_occupancy_frames / self.warm_up_total_frames
                dirty_ratio = self.warm_up_dirty_frames / self.warm_up_total_frames

                if occ_ratio >= 0.5:
                    initial_state = "WAITING_FOR_FOOD"
                    self.customer_time_start = self.first_frame_time
                    self.occupancy_start_time = self.first_frame_time
                    self.waiting_time_start = self.first_frame_time
                    self.order_confirmed_time = self.first_frame_time
                    self.food_wait_start_time = self.first_frame_time
                elif dirty_ratio >= 0.5:
                    initial_state = "DIRTY"
                    self.dirty_start_time = self.first_frame_time
                else:
                    initial_state = "EMPTY"

                self.state = initial_state
                self.state_start_time = frame_time
                self.session_uuid = session_uuid
                print(
                    f"[FSM INIT] Table {self.table_id} warm-up complete -> {self.state}"
                )
            return  # Suppress normal update during warm-up

        # ── Phase 2: Transient auto-advances ───────────────────────────
        # ORDER_TAKEN → WAITING_FOR_FOOD (same cycle, next tick)
        if self.state == "ORDER_TAKEN":
            self.transition_to(
                "WAITING_FOR_FOOD", frame_time,
                "ORDER_TAKEN_AUTO_ADVANCE", db_manager, session_uuid
            )
            return

        # FOOD_SERVED → DINING (next tick after the FOOD_SERVED transition)
        if self.state == "FOOD_SERVED" and self._food_served_advance_pending:
            self._food_served_advance_pending = False
            self.transition_to(
                "DINING", frame_time,
                "FOOD_SERVED_AUTO_ADVANCE", db_manager, session_uuid
            )
            return

        # ── Phase 3: Normal state transitions ───────────────────────────

        # ── EMPTY ────────────────────────────────────────────────────────
        if self.state == "EMPTY":
            if customer_present:
                if self.customer_present_start_time is None:
                    self.customer_present_start_time = frame_time
                elif (frame_time - self.customer_present_start_time
                        >= config["occupancy_validation_seconds"]):
                    if self.transition_to(
                            "OCCUPIED", frame_time, "CUSTOMER_SEATED",
                            db_manager, session_uuid):
                        self.customer_present_start_time = None
            else:
                self.customer_present_start_time = None

        # ── OCCUPIED ─────────────────────────────────────────────────────
        elif self.state == "OCCUPIED":
            # NOTE: No is_serving fast-path here — must go through ORDER_TAKEN.
            if customer_present:
                self.customer_absent_start_time = None
                is_valid_gesture = waiter_present and is_writing
                if is_valid_gesture:
                    self.last_active_gesture_time = frame_time
                self.gesture_window.append((frame_time, is_valid_gesture))

                # Expire old window entries
                self.gesture_window = [
                    (t, v) for t, v in self.gesture_window
                    if t >= frame_time - config["gesture_validation_window_seconds"]
                ]
                # Dropout recovery
                if (self.last_active_gesture_time is not None
                        and frame_time - self.last_active_gesture_time
                        > config["gesture_dropout_tolerance_seconds"]):
                    self.gesture_window = []
                    self.last_active_gesture_time = None

                if len(self.gesture_window) > 0:
                    ratio = sum(1 for _, v in self.gesture_window if v) / len(self.gesture_window)
                    if ratio >= config["gesture_acceptance_ratio"]:
                        # Record which waiter triggered the order
                        self.order_taken_waiter_id = serving_waiter_id
                        if self.transition_to(
                                "ORDER_TAKEN", frame_time, "ORDER_CONFIRMED",
                                db_manager, session_uuid):
                            self.gesture_window = []
                            self.last_active_gesture_time = None
            else:
                # Customer left before ordering
                if self.customer_absent_start_time is None:
                    self.customer_absent_start_time = frame_time
                elif (frame_time - self.customer_absent_start_time
                        >= config["customer_exit_timeout_seconds"]):
                    self.transition_to(
                        "CUSTOMER_LEFT", frame_time, "CUSTOMER_LEFT_BEFORE_ORDER",
                        db_manager, session_uuid
                    )
                    self.customer_absent_start_time = None

        # ── WAITING_FOR_FOOD ─────────────────────────────────────────────
        elif self.state == "WAITING_FOR_FOOD":
            # ── FOOD_SERVED multi-condition guard (can happen even if customer is temporarily absent) ──
            if not self.food_served_fired and is_serving:
                # 1. Waiter ownership: must match order-taking waiter
                #    (if no ownership recorded yet, accept any waiter)
                waiter_ok = (
                    self.order_taken_waiter_id is None
                    or serving_waiter_id is None
                    or serving_waiter_id == self.order_taken_waiter_id
                )
                # 2. Plate must be inside the correct table ROI
                plate_ok = plate_in_roi
                # 3. Confidence threshold
                conf_ok = serving_confidence >= config["food_served_min_confidence"]

                if waiter_ok and plate_ok and conf_ok:
                    # 4. Plate must remain in ROI for confirmation period
                    if self.food_served_confirmation_start is None:
                        self.food_served_confirmation_start = frame_time
                        print(
                            f"[FSM] Table {self.table_id}: "
                            f"FOOD_SERVED confirmation started at {frame_time:.1f}s"
                        )
                    elif (frame_time - self.food_served_confirmation_start
                            >= config["food_served_confirmation_seconds"]):
                        # ALL guards passed → fire FOOD_SERVED
                        self.food_served_fired = True
                        self.transition_to(
                            "FOOD_SERVED", frame_time, "FOOD_SERVED_CONFIRMED",
                            db_manager, session_uuid
                        )
                else:
                    # Any failed guard resets the confirmation timer
                    if self.food_served_confirmation_start is not None:
                        print(
                            f"[FSM] Table {self.table_id}: "
                            f"FOOD_SERVED confirmation reset "
                            f"(waiter_ok={waiter_ok}, plate_ok={plate_ok}, "
                            f"conf_ok={conf_ok})"
                        )
                    self.food_served_confirmation_start = None
            else:
                # Not serving — reset confirmation timer
                if not is_serving:
                    self.food_served_confirmation_start = None

            # Customer presence check
            if customer_present:
                self.customer_absent_start_time = None
            else:
                # Customer left while waiting for food
                if self.customer_absent_start_time is None:
                    self.customer_absent_start_time = frame_time
                elif (frame_time - self.customer_absent_start_time
                        >= config["customer_exit_timeout_seconds"]):
                    # Only transition to CUSTOMER_LEFT if food has not been served yet
                    if not self.food_served_fired:
                        self.transition_to(
                            "CUSTOMER_LEFT", frame_time, "CUSTOMER_LEFT_WHILE_WAITING",
                            db_manager, session_uuid
                        )
                        self.customer_absent_start_time = None

        # ── DINING ───────────────────────────────────────────────────────
        elif self.state == "DINING":
            if not customer_present:
                if self.customer_absent_start_time is None:
                    self.customer_absent_start_time = frame_time
                elif (frame_time - self.customer_absent_start_time
                        >= config["customer_exit_timeout_seconds"]):
                    if self.transition_to(
                            "CUSTOMER_LEFT", frame_time, "CUSTOMER_LEFT",
                            db_manager, session_uuid):
                        self.customer_absent_start_time = None
            else:
                self.customer_absent_start_time = None

        # ── CUSTOMER_LEFT ─────────────────────────────────────────────────
        elif self.state == "CUSTOMER_LEFT":
            # Always advance to DIRTY (plate will be on the table)
            # The DIRTY→CLEAN guard handles the actual plate detection.
            self.transition_to(
                "DIRTY", frame_time, "TABLE_NOW_DIRTY",
                db_manager, self._last_known_session_uuid
            )

        # ── DIRTY ────────────────────────────────────────────────────────
        elif self.state == "DIRTY":
            time_in_dirty = frame_time - self.state_start_time
            min_dirty = config.get("dirty_table_min_seconds", 10.0)
            timeout_secs = config["dirty_table_timeout_minutes"] * 60
            if (dirty_object_count == 0 and time_in_dirty >= min_dirty) \
                    or time_in_dirty >= timeout_secs:
                self.transition_to(
                    "CLEAN", frame_time, "TABLE_CLEANED",
                    db_manager, self._last_known_session_uuid
                )

        # ── CLEAN ────────────────────────────────────────────────────────
        elif self.state == "CLEAN":
            clean_validation = config.get("clean_validation_seconds", 5.0)
            if frame_time - self.state_start_time >= clean_validation:
                self.transition_to(
                    "EMPTY", frame_time, "VACANT",
                    db_manager, self._last_known_session_uuid
                )


# ────────────────────────────────────────────────────────────────────────────
# Manager
# ────────────────────────────────────────────────────────────────────────────

class TableFSMManager:
    def __init__(self, table_ids: list):
        self.tables = {tid: TableStateFSM(tid) for tid in table_ids}

    def update(self, table_id: str, customer_present: bool, waiter_present: bool,
               is_writing: bool, is_serving: bool, dirty_object_count: int,
               frame_time: float, db_manager=None, session_uuid=None,
               serving_waiter_id: str = None, plate_in_roi: bool = False,
               serving_confidence: float = 0.0):
        if table_id not in self.tables:
            self.tables[table_id] = TableStateFSM(table_id)
        self.tables[table_id].update(
            customer_present=customer_present,
            waiter_present=waiter_present,
            is_writing=is_writing,
            is_serving=is_serving,
            dirty_object_count=dirty_object_count,
            frame_time=frame_time,
            db_manager=db_manager,
            session_uuid=session_uuid,
            serving_waiter_id=serving_waiter_id,
            plate_in_roi=plate_in_roi,
            serving_confidence=serving_confidence,
        )

    def get_state(self, table_id: str) -> dict:
        if table_id in self.tables:
            t = self.tables[table_id]
            return {"state": t.state, "occupied_start_time": t.occupancy_start_time}
        return {"state": "UNKNOWN", "occupied_start_time": None}

    def get_all_states(self) -> dict:
        result = {}
        for tid, t in self.tables.items():
            result[tid] = {
                # State
                "state": t.state,
                "previous_state": t.previous_state,
                "state_start_time": t.state_start_time,
                "warm_up_complete": t.warm_up_complete,
                # Business timers (new canonical names)
                "customer_time_start": t.customer_time_start,
                "customer_time_stop": t.customer_time_stop,
                "waiting_time_start": t.waiting_time_start,
                "waiting_time_stop": t.waiting_time_stop,
                "dining_time_start": t.dining_time_start,
                "dining_time_stop": t.dining_time_stop,
                # Legacy aliases (retained for DB / pipeline compatibility)
                "occupancy_start_time": t.occupancy_start_time,
                "order_confirmed_time": t.order_confirmed_time,
                "food_wait_start_time": t.food_wait_start_time,
                "food_served_time": t.food_served_time,
                "customer_left_time": t.customer_left_time,
                "dirty_start_time": t.dirty_start_time,
                "clean_time": t.clean_time,
                "session_uuid": t.session_uuid,
                # FOOD_SERVED guard
                "food_served_fired": t.food_served_fired,
            }
        return result
