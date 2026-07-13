# analytics/visualization/renderer.py
"""
Renderer — Production HUD overlay for the analytics pipeline.

Draws table polygons, FSM state cards (clean production format),
tracked persons, and FPS.

HUD card displays ONLY:
  Table ID | Status | Customer Time | Waiting Time | Dining Time | Food Status
No debug information is shown.
"""
import cv2
import numpy as np

FSM_STATE_COLORS = {
    "UNKNOWN":          (100, 100, 100),  # gray
    "EMPTY":            (128, 128, 128),  # gray
    "OCCUPIED":         (200, 100,   0),  # blue-ish orange
    "ORDER_TAKEN":      (0,  165, 255),   # orange
    "WAITING_FOR_FOOD": (0,  165, 255),   # orange
    "FOOD_SERVED":      (0,  200,   0),   # green
    "DINING":           (0,  200,   0),   # green
    "CUSTOMER_LEFT":    (80,  80, 200),   # muted blue
    "DIRTY":            (0,    0, 200),   # red
    "CLEAN":            (0,  255,   0),   # light green
}

ROLE_COLORS = {
    "waiter":   (0,   0, 255),   # red
    "customer": (0, 255,   0),   # green
}

# States where the food has been confirmed served (for the Food Status row)
_FOOD_SERVED_STATES = {"FOOD_SERVED", "DINING", "CUSTOMER_LEFT", "DIRTY", "CLEAN"}


class Renderer:
    """Draws all FSM table and person analytics overlays on video frames."""

    def __init__(self, tables: dict):
        """
        Args:
            tables: dict of table_id -> {"polygon": [...], "center": [cx, cy]}
        """
        self.tables = tables

    def draw_table_polygons(self, frame, fsm_states: dict, occupancy_data: list,
                            frame_time: float, food_served_debug_info: dict = None, debug: bool = False):
        """Draw table ROI polygons and production HUD card."""
        occ_lookup = {occ["table_id"]: occ for occ in occupancy_data}

        fh, fw = frame.shape[:2]

        for table_id, table_info in self.tables.items():
            poly = np.array(table_info["polygon"], dtype=np.int32)
            fsm_info = fsm_states.get(table_id, {
                "state": "UNKNOWN",
                "warm_up_complete": False,
                "customer_time_start": None,
                "customer_time_stop": None,
                "waiting_time_start": None,
                "waiting_time_stop": None,
                "dining_time_start": None,
                "dining_time_stop": None,
                "food_served_time": None,
                "food_served_fired": False,
                "session_uuid": None,
            })

            state = fsm_info.get("state", "UNKNOWN")
            color = FSM_STATE_COLORS.get(state, (128, 128, 128))

            # Filled polygon (transparent) - crop and blend locally
            x, y, w, h = cv2.boundingRect(poly)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(fw, x + w), min(fh, y + h)

            if x2 > x1 and y2 > y1:
                sub_frame = frame[y1:y2, x1:x2]
                sub_overlay = sub_frame.copy()
                poly_rel = poly - [x1, y1]
                cv2.fillPoly(sub_overlay, [poly_rel], color)
                cv2.addWeighted(sub_overlay, 0.25, sub_frame, 0.75, 0, sub_frame)

            # Outline drawn on full frame
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

            cx, cy = int(table_info["center"][0]), int(table_info["center"][1])

            # ── HUD Card ─────────────────────────────────────────────────
            # Rows: Table ID | Status | Customer Time | Waiting Time |
            #       Dining Time | Food Status  → 6 rows × 16px + padding
            card_w = 210
            card_h = 116      # 6 rows × 16 + 16px top/bottom padding
            if debug:
                card_h = 138  # Extra space for Session UUID

            bx = cx - card_w // 2
            by = cy - card_h // 2

            cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h),
                          (20, 20, 20), -1)
            cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h),
                          color, 2)

            # Row 1: Table ID + customer count
            y = by + 16
            label = f"TABLE {table_id.split('_')[-1].upper()}"
            cv2.putText(frame, label, (bx + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2)

            occ = occ_lookup.get(table_id, {})
            count = occ.get("customer_count", 0)
            if count > 0:
                ct = f"C:{count}"
                (tw, _), _ = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.putText(frame, ct, (bx + card_w - tw - 8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

            # Row 2: Status
            y += 16
            if not fsm_info.get("warm_up_complete", True) and state == "UNKNOWN":
                display_status = "INITIALIZING"
                status_color = (150, 150, 150)
            else:
                # Show FOOD SERVED banner briefly via blinking text
                food_served_ts = fsm_info.get("food_served_time") or fsm_info.get("dining_time_start")
                is_blinking = (
                    state in ("FOOD_SERVED", "DINING")
                    and food_served_ts is not None
                    and frame_time - food_served_ts < 8.0
                )
                if is_blinking:
                    display_status = "FOOD SERVED"
                    flash_on = int(frame_time * 4) % 2 == 0
                    status_color = (0, 120, 255) if flash_on else (80, 80, 80)
                else:
                    display_status = state.replace("_", " ")
                    status_color = color

            cv2.putText(frame, f"Status: {display_status}", (bx + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 2)

            # ── Timer helper ──────────────────────────────────────────────
            def fmt_timer(start, stop):
                """Format mm:ss, counting up from start; freezes at stop."""
                if start is None:
                    return "--:--"
                end = stop if stop is not None else frame_time
                elapsed = max(0.0, end - start)
                m = int(elapsed) // 60
                s = int(elapsed) % 60
                return f"{m:02d}:{s:02d}"

            # Row 3: Customer Time (OCCUPIED → CUSTOMER_LEFT)
            y += 16
            cust_str = fmt_timer(
                fsm_info.get("customer_time_start"),
                fsm_info.get("customer_time_stop")
            )
            cv2.putText(frame, f"Customer:  {cust_str}", (bx + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            # Row 4: Waiting Time (ORDER_TAKEN → FOOD_SERVED, then freezes)
            y += 16
            wait_str = fmt_timer(
                fsm_info.get("waiting_time_start"),
                fsm_info.get("waiting_time_stop")
            )
            cv2.putText(frame, f"Waiting:   {wait_str}", (bx + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            # Row 5: Dining Time (FOOD_SERVED → CUSTOMER_LEFT, then freezes)
            y += 16
            dining_str = fmt_timer(
                fsm_info.get("dining_time_start"),
                fsm_info.get("dining_time_stop")
            )
            cv2.putText(frame, f"Dining:    {dining_str}", (bx + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            # Row 6: Food Status
            y += 16
            food_served = fsm_info.get("food_served_fired", False) or state in _FOOD_SERVED_STATES
            food_txt = "Food: SERVED" if food_served else "Food: NOT SERVED"
            food_col = (0, 200, 0) if food_served else (160, 160, 160)
            cv2.putText(frame, food_txt, (bx + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, food_col, 1)

            # Debug specific elements
            if debug:
                y += 16
                suuid = fsm_info.get("session_uuid")
                suuid_str = str(suuid)[:8] + "..." if suuid else "None"
                cv2.putText(frame, f"SESS: {suuid_str}", (bx + 8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)

                # Draw centroid circle
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.putText(frame, f"C:({cx},{cy})", (cx + 6, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

                # Draw serving vector
                if food_served_debug_info and table_id in food_served_debug_info:
                    info = food_served_debug_info[table_id]
                    wx, wy = map(int, info["wrist"])
                    cv2.line(frame, (cx, cy), (wx, wy), (0, 165, 255), 2)
                    cv2.circle(frame, (wx, wy), 5, (0, 0, 255), -1)
                    dist = info["distance"]
                    cv2.putText(frame, f"{dist:.0f}px", ((cx + wx) // 2, (cy + wy) // 2 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1)

            # ── Blinking FOOD SERVED banner above table ───────────────────
            food_served_ts = fsm_info.get("food_served_time") or fsm_info.get("dining_time_start")
            if (state in ("FOOD_SERVED", "DINING")
                    and food_served_ts is not None
                    and frame_time - food_served_ts < 8.0):
                flash_on = int(frame_time * 4) % 2 == 0
                if flash_on:
                    min_y = min(p[1] for p in table_info["polygon"])
                    fs_box_y = max(10, min_y - 32 - 10)
                    fs_box_x = cx - 85
                    cv2.rectangle(frame, (fs_box_x, fs_box_y),
                                  (fs_box_x + 170, fs_box_y + 32),
                                  (0, 120, 255), -1)
                    cv2.rectangle(frame, (fs_box_x, fs_box_y),
                                  (fs_box_x + 170, fs_box_y + 32),
                                  (255, 255, 255), 2)
                    cv2.putText(frame, "FOOD SERVED",
                                (fs_box_x + 12, fs_box_y + 23),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return frame

    def draw_persons(self, frame, persons, frame_time: float, debug: bool = False):
        """Draw tracked person bounding boxes with role label."""
        for person in persons:
            if not person.confirmed:
                continue

            x1, y1, x2, y2 = person.bbox
            color = ROLE_COLORS.get(person.role, (128, 128, 128))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if debug:
                # Debug Overlay format: detailed states
                yolo_conf = getattr(person, "yolo_conf", 0.0)
                assigned = person.assigned_table if person.assigned_table else "None"
                sid = person.session_id if person.session_id else "None"

                lines = [
                    f"{person.role.upper()} | T{person.track_id}",
                    f"Sess: {sid}",
                    f"Table: {assigned}",
                    f"YOLO Conf: {yolo_conf:.2f}",
                ]

                if person.role == "waiter":
                    is_srv = getattr(person, "is_serving", False)
                    is_ot = getattr(person, "is_order_taking", False)
                    srv_conf = 0.0
                    if hasattr(person, "serving_res"):
                        srv_conf = person.serving_res.get("confidence", 0.0)
                    lines.append(f"Serv: {is_srv} ({srv_conf:.2f})")
                    lines.append(f"Write: {is_ot}")

                box_h = len(lines) * 12 + 6
                box_w = 160
                cv2.rectangle(frame, (x1, max(0, y1 - box_h - 2)),
                              (x1 + box_w, y1), (15, 15, 15), -1)
                cv2.rectangle(frame, (x1, max(0, y1 - box_h - 2)),
                              (x1 + box_w, y1), color, 1)

                for idx, line in enumerate(lines):
                    ly = y1 - box_h + 12 + idx * 12
                    cv2.putText(frame, line, (x1 + 4, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (240, 240, 240), 1)

                # Draw normalized MediaPipe coordinates if they exist
                if person.role == "waiter" and hasattr(person, "serving_res"):
                    res = person.serving_res
                    pose_res = res.get("pose_results")
                    if pose_res and hasattr(pose_res, "pose_landmarks") and pose_res.pose_landmarks:
                        for lmk in pose_res.pose_landmarks[0]:
                            lx, ly = lmk.x, lmk.y
                            gx = x1 + int(lx * (x2 - x1))
                            gy = y1 + int(ly * (y2 - y1))
                            cv2.circle(frame, (gx, gy), 2, (0, 255, 255), -1)

                    hand_res = res.get("hand_results")
                    if hand_res and hasattr(hand_res, "hand_landmarks") and hand_res.hand_landmarks:
                        for lmk in hand_res.hand_landmarks[0]:
                            lx, ly = lmk.x, lmk.y
                            gx = x1 + int(lx * (x2 - x1))
                            gy = y1 + int(ly * (y2 - y1))
                            cv2.circle(frame, (gx, gy), 2, (255, 0, 255), -1)
            else:
                # Production format
                if person.role == "customer":
                    duration = int(frame_time - person.first_seen)
                    h_t = duration // 3600
                    m_t = (duration % 3600) // 60
                    s_t = duration % 60
                    time_str = f"{h_t}:{m_t:02d}:{s_t:02d}"

                    box_h = 35
                    (tw, _), _ = cv2.getTextSize(
                        "Customer", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    (tw2, _), _ = cv2.getTextSize(
                        time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    box_w = max(tw, tw2) + 10

                    cv2.rectangle(frame, (x1, y1 - box_h - 5),
                                  (x1 + box_w, y1), (0, 0, 0), -1)
                    cv2.putText(frame, "Customer", (x1 + 5, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, time_str, (x1 + 5, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    waiter_id = person.session_id if person.session_id else f"T{person.track_id}"
                    text = f"Waiter {waiter_id}"
                    if getattr(person, "is_serving", False):
                        text += " [SERVING]"
                    (tw, th), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6),
                                  (x1 + tw, y1), color, -1)
                    cv2.putText(frame, text, (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def draw_fps(self, frame, fps: float):
        """Draw FPS counter."""
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return frame

    def draw_summary_panel(self, frame, occupancy_data: list, fsm_states: dict):
        """Draw a compact summary panel in the top-right corner."""
        h_f, w_f = frame.shape[:2]
        panel_w = 260
        panel_h = 30 + len(self.tables) * 28
        x_start = w_f - panel_w - 10
        y_start = 10

        cv2.rectangle(frame, (x_start, y_start),
                      (x_start + panel_w, y_start + panel_h),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (x_start, y_start),
                      (x_start + panel_w, y_start + panel_h),
                      (200, 200, 200), 1)

        cv2.putText(frame, "Table Analytics", (x_start + 8, y_start + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        occ_lookup = {o["table_id"]: o for o in occupancy_data}
        for i, table_id in enumerate(self.tables):
            y = y_start + 40 + i * 24
            info = fsm_states.get(table_id, {"state": "UNKNOWN"})
            st = info.get("state", "UNKNOWN")
            col = FSM_STATE_COLORS.get(st, (100, 100, 100))
            occ = occ_lookup.get(table_id, {})
            count = occ.get("customer_count", 0)

            cv2.circle(frame, (x_start + 12, y - 4), 5, col, -1)
            txt = f"{table_id}: {st} | {count}c"
            cv2.putText(frame, txt, (x_start + 24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)

        return frame

    def render(self, frame, persons, occupancy_data, fsm_states, fps,
               frame_time: float, food_served_debug_info: dict = None, debug: bool = False):
        """Full render pass."""
        frame = self.draw_table_polygons(
            frame, fsm_states, occupancy_data, frame_time, food_served_debug_info, debug=debug)
        frame = self.draw_persons(frame, persons, frame_time, debug=debug)
        frame = self.draw_summary_panel(frame, occupancy_data, fsm_states)
        frame = self.draw_fps(frame, fps)
        return frame
