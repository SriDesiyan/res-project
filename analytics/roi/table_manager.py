"""
Table Manager — Coordinates auto-detection + dwell-clustering + hot-reload.

This is the single entry point the pipeline uses for table ROI management.
It replaces the static ``load_tables`` call with a dynamic manager that:

1. On startup: auto-detects tables if no config exists (empty-room frame).
2. During the run: collects dwell points from tracked customers.
3. Periodically: re-clusters dwell points and merges/updates table ROIs.
4. Exposes ``get_tables()`` for the pipeline to read the current layout.

Usage:
    manager = TableManager(config_path, video_path)
    tables = manager.get_tables()           # initial tables
    manager.record_customer(person, frame_time)  # every frame
    manager.maybe_refresh(frame_time)       # periodically check
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from analytics.tracking.person_tracker import TrackedPerson

import sys
_roi_dir = str(Path(__file__).resolve().parent)
if _roi_dir not in sys.path:
    sys.path.insert(0, _roi_dir)

from roi_config import load_tables, save_tables, polygon_center


class TableManager:
    """
    Dynamic table ROI manager.

    Parameters
    ----------
    config_path : Path
        Path to ``tables.json`` (read / written).
    video_path : Path | None
        Video file used for auto-calibration.
    calibrate_frame : int
        Frame number to extract for auto-detection (should be an
        empty-room frame, e.g. frame 0 or a known off-hours frame).
    enable_auto_detect : bool
        If True, run YOLOv8-seg detection when no config file exists.
    enable_dwell_learning : bool
        If True, continuously collect dwell points and periodically
        re-cluster to discover / update table positions.
    refresh_interval_sec : float
        Minimum seconds between dwell-clustering refreshes.
    velocity_threshold : float
        Maximum velocity (px/frame) below which a customer is
        considered stationary and eligible for dwell recording.
    min_dwell_frames : int
        Customer must be tracked for at least this many frames
        before their position counts as a dwell point.
    """

    def __init__(
        self,
        config_path: Path,
        video_path: Path | None = None,
        calibrate_frame: int = 0,
        enable_auto_detect: bool = True,
        enable_dwell_learning: bool = True,
        refresh_interval_sec: float = 300.0,     # 5 minutes
        velocity_threshold: float = 3.0,
        min_dwell_frames: int = 75,              # ~3s at 25fps
        calibrate_method: str = "aruco",
        aruco_dict: str = "DICT_4X4_50",
        aruco_scale: float = 5.0,
    ):
        self.config_path = Path(config_path)
        self.video_path = video_path
        self.calibrate_frame = calibrate_frame
        self.enable_auto_detect = enable_auto_detect
        self.enable_dwell_learning = enable_dwell_learning
        self.refresh_interval_sec = refresh_interval_sec
        self.velocity_threshold = velocity_threshold
        self.min_dwell_frames = min_dwell_frames
        self.calibrate_method = calibrate_method
        self.aruco_dict = aruco_dict
        self.aruco_scale = aruco_scale

        self._tables: dict = {}
        self._last_refresh_time: float = 0.0
        self._dwell_manager = None
        self._refresh_count: int = 0

        # ── initialise ─────────────────────────────────────────────
        self._initialise()

    # ── lifecycle ──────────────────────────────────────────────────
    def _initialise(self):
        """Load existing config or run auto-detection."""

        # Try loading existing config first
        if self.config_path.exists():
            self._tables = load_tables(self.config_path)
            print(
                f"[TABLE-MGR] Loaded {len(self._tables)} table(s) "
                f"from {self.config_path}"
            )
        elif self.enable_auto_detect and self.video_path is not None:
            print(f"[TABLE-MGR] No tables.json found — running auto-detection via method={self.calibrate_method} …")
            self._run_auto_detect()
        else:
            print(
                "[TABLE-MGR] No tables.json and auto-detect is disabled. "
                "Starting with zero tables."
            )

        # Initialise dwell clustering if enabled
        if self.enable_dwell_learning:
            try:
                from dwell_cluster import DwellClusterManager

                # Compute adaptive eps if we have tables
                eps = 120.0
                if self._tables:
                    diagonals = []
                    for t_info in self._tables.values():
                        poly = np.array(t_info["polygon"])
                        if len(poly) >= 3:
                            diag = np.linalg.norm(np.max(poly, axis=0) - np.min(poly, axis=0))
                            diagonals.append(diag)
                    if diagonals:
                        eps = float(np.mean(diagonals) * 0.35)
                        eps = max(40.0, min(250.0, eps))
                        print(f"[TABLE-MGR] Adaptive DBSCAN eps calculated: {eps:.1f}px (35% of avg table diagonal)")

                self._dwell_manager = DwellClusterManager(eps=eps)

                # Load persisted dwell points if they exist
                dwell_path = self.config_path.parent / "dwell_points.json"
                self._dwell_manager.load_points(dwell_path)

                print(f"[TABLE-MGR] Dwell-learning enabled with eps={eps:.1f}px.")
            except ImportError:
                print(
                    "[TABLE-MGR] scikit-learn not installed — "
                    "dwell-learning disabled."
                )
                self._dwell_manager = None

    def _run_auto_detect(self):
        """Run auto-detection (ArUco or YOLO) and save results."""
        try:
            from auto_roi import AutoTableDetector

            detector = AutoTableDetector(
                aruco_dict_name=self.aruco_dict,
                aruco_scale_x=self.aruco_scale,
                aruco_scale_y=self.aruco_scale,
            )
            tables = detector.detect_from_video(
                self.video_path,
                self.calibrate_frame,
                method=self.calibrate_method,
            )

            if tables:
                self._tables = tables
                save_tables(tables, self.config_path)
                print(
                    f"[TABLE-MGR] Auto-detected {len(tables)} table(s) "
                    f"→ {self.config_path}"
                )
            else:
                print(
                    "[TABLE-MGR] Auto-detection found no tables. "
                    "The pipeline will rely on dwell-learning."
                )
        except Exception as e:
            print(f"[TABLE-MGR] Auto-detection failed: {e}")

    # ── public API ─────────────────────────────────────────────────
    def get_tables(self) -> dict:
        """Return the current table definitions."""
        return self._tables

    def get_table_ids(self) -> list[str]:
        """Return sorted list of current table IDs."""
        return sorted(self._tables.keys())

    def record_customer(self, person, frame_time: float):
        """
        Feed a tracked person into the dwell-point collector.

        Call once per frame for every confirmed customer.
        Only records if the person is stationary and has been
        visible long enough.
        """
        if self._dwell_manager is None:
            return

        if not person.confirmed:
            return
        if person.role != "customer":
            return
        if person.velocity > self.velocity_threshold:
            return
        if person.frame_count < self.min_dwell_frames:
            return

        self._dwell_manager.record_dwell_point(
            person.bottom_center, frame_time
        )

    def maybe_refresh(self, frame_time: float) -> bool:
        """
        Check if it's time to re-cluster dwell points and update tables.

        Returns True if tables were updated.
        """
        if self._dwell_manager is None:
            return False

        elapsed = frame_time - self._last_refresh_time
        if elapsed < self.refresh_interval_sec:
            return False

        self._last_refresh_time = frame_time
        return self._refresh_from_dwell(frame_time)

    def force_refresh(self, frame_time: float) -> bool:
        """Force an immediate dwell-clustering refresh."""
        if self._dwell_manager is None:
            return False
        return self._refresh_from_dwell(frame_time)

    # ── internal ───────────────────────────────────────────────────
    def _refresh_from_dwell(self, frame_time: float) -> bool:
        """
        Run DBSCAN on dwell points and merge discovered tables
        with the existing layout.
        """
        discovered = self._dwell_manager.run_clustering()
        if not discovered:
            return False

        merged = self._merge_tables(self._tables, discovered)

        if merged != self._tables:
            # Back up the previous config
            self._refresh_count += 1
            backup = self.config_path.with_suffix(
                f".backup_{self._refresh_count}.json"
            )
            if self.config_path.exists():
                shutil.copy2(self.config_path, backup)

            self._tables = merged
            save_tables(merged, self.config_path)

            # Persist dwell points for next startup
            dwell_path = self.config_path.parent / "dwell_points.json"
            self._dwell_manager.save_points(dwell_path)

            print(
                f"[TABLE-MGR] Tables updated: {len(merged)} table(s) "
                f"(backup → {backup.name})"
            )
            return True

        return False

    @staticmethod
    def _merge_tables(existing: dict, discovered: dict) -> dict:
        """
        Merge discovered tables with existing ones.

        Strategy:
        - If a discovered table's centre is within 150px of an
          existing table's centre, it is considered the *same* table
          and the existing entry is updated with the new polygon.
        - Otherwise the discovered table is added as a new entry.
        - Existing tables that have no matching discovery are kept
          (they may be occluded or in a dead zone for customers).
        """
        MERGE_RADIUS = 150.0  # pixels

        merged = dict(existing)  # start with a copy of existing
        used_existing = set()

        for disc_id, disc_info in discovered.items():
            dcx, dcy = disc_info["center"]

            best_match = None
            best_dist = float("inf")

            for ex_id, ex_info in existing.items():
                if ex_id in used_existing:
                    continue
                ecx, ecy = ex_info["center"]
                dist = ((dcx - ecx) ** 2 + (dcy - ecy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_match = ex_id

            if best_match is not None and best_dist < MERGE_RADIUS:
                # Update existing table with refined polygon
                merged[best_match] = disc_info
                used_existing.add(best_match)
                print(
                    f"  [MERGE] {disc_id} → updated {best_match} "
                    f"(Δ={best_dist:.0f}px)"
                )
            else:
                # New table discovered — assign next available ID
                max_num = 0
                for tid in merged:
                    try:
                        num = int(tid.split("_")[-1])
                        max_num = max(max_num, num)
                    except ValueError:
                        pass
                new_id = f"table_{max_num + 1}"
                merged[new_id] = disc_info
                print(f"  [MERGE] {disc_id} → new {new_id}")

        return merged
