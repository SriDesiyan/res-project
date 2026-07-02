# analytics/config/fsm_config.py

FSM_CONFIG = {
    # Warm-up initialization threshold in seconds
    "warm_up_period_seconds": 5.0,

    # Occupancy validation parameters
    # How long a customer must remain inside the ROI before EMPTY→OCCUPIED
    "occupancy_validation_seconds": 1.0,  # Reduced from 4.0: DWELL_THRESHOLD already provides 3s stability

    # Waiter order-taking gesture validation parameters
    "gesture_validation_window_seconds": 3.0,
    "gesture_acceptance_ratio": 0.65,
    "gesture_dropout_tolerance_seconds": 1.5,
    "waiter_engagement_seconds": 3.0,

    # Banner showing duration (UI-only effect, does not delay business timing)
    "food_served_banner_seconds": 8.0,

    # ── FOOD_SERVED multi-condition guards ─────────────────────────────────
    # Minimum seconds the plate must remain inside the correct table ROI
    # AND the serving gesture must be active before confirming FOOD_SERVED.
    "food_served_confirmation_seconds": 0.5,
    # Maximum pixel distance from serving hand to table centroid
    "food_served_hand_to_table_max_px": 400.0,
    # Minimum detector confidence to accept a serving detection
    "food_served_min_confidence": 0.45,

    # ── Customer exit detection ────────────────────────────────────────────
    # Seconds a customer must be continuously absent from the ROI before
    # triggering DINING→CUSTOMER_LEFT (prevents flicker / occlusion exits)
    "customer_exit_timeout_seconds": 8.0,

    # ── Timeout / warning parameters ──────────────────────────────────────
    "waiting_for_food_timeout_minutes": 40.0,
    "customer_idle_timeout_minutes": 30.0,
    "dirty_table_timeout_minutes": 30.0,

    # ── Cleanliness transition guards ─────────────────────────────────────
    # Minimum seconds the table must be DIRTY before allowing CLEAN transition
    "dirty_table_min_seconds": 10.0,
    # Seconds to confirm CLEAN state before returning to EMPTY
    "clean_validation_seconds": 5.0,

    # ── Waiter classification tuning ──────────────────────────────────────
    # Number of hit-score points required to lock a track as a waiter
    "waiter_lock_threshold": 6,
    # Hit score increment per frame when uniform matches
    "waiter_hit_increment": 3,
    # Number of consecutive non-match frames before considering an unlock
    "waiter_unlock_streak": 8,
}
