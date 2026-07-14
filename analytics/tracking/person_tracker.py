"""
analytics/tracking/person_tracker.py
======================================
Person Tracker — YOLO + BoT-SORT + Waiter Classification.

MIGRATION NOTE (Edge AI):
    This module no longer imports torch, YOLO, OSNet, or ResNet50 directly.
    All model inference is routed through the ``BaseInferenceEngine`` interface.
    Waiter colour-heuristic logic (HSV analysis) is pure CPU numpy — unchanged.
    BoT-SORT state management is unchanged.
    Business logic (role locking, hit counting, disappear timeout) is unchanged.

Usage:
    from analytics.inference.engine_factory import create_engine
    from analytics.tracking.person_tracker import PersonTracker

    engine = create_engine(backend="auto")
    tracker = PersonTracker(engine, conf=0.35)
    persons = tracker.process_frame(frame, frame_time)
"""
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from analytics.inference.base_engine import BaseInferenceEngine


class TrackedPerson:
    """Represents a single person being tracked across frames."""

    def __init__(self, track_id: int, bbox: tuple, centroid: tuple, frame_time: float):
        self.track_id = track_id
        self.bbox = bbox                  # (x1, y1, x2, y2)
        self.centroid = centroid           # (cx, cy)
        self.bottom_center = (centroid[0], bbox[3])  # (cx, y_max)
        self.velocity = 0.0               # px/frame
        self.role = "customer"            # "waiter" | "customer"
        self.assigned_table = None        # table_id or None
        self.first_seen = frame_time
        self.last_seen = frame_time
        self.frame_count = 1
        self.confirmed = False            # True after MIN_VISIBILITY frames
        self.visual_embedding = None      # numpy (1, 512) from OSNet
        self.session_id = None

    def update(self, bbox: tuple, centroid: tuple, frame_time: float) -> None:
        self.bbox = bbox
        dx = centroid[0] - self.centroid[0]
        dy = centroid[1] - self.centroid[1]
        self.velocity = (dx ** 2 + dy ** 2) ** 0.5
        self.centroid = centroid
        self.bottom_center = (centroid[0], bbox[3])
        self.last_seen = frame_time
        self.frame_count += 1


class PersonTracker:
    """
    Manages YOLO detection, BoT-SORT tracking, and waiter classification.

    All neural-network calls are delegated to the injected ``engine``.
    This class owns:
      - BoT-SORT track-state dictionary (``self.tracks``)
      - Waiter lock/unlock logic (hit counting, colour heuristics)
      - Track disappear/cleanup logic

    Parameters
    ----------
    engine : BaseInferenceEngine
        The active inference backend (CUDA / CPU / ONNX / Axelera).
    conf : float
        YOLO detection confidence threshold.
    similarity_threshold : float
        Cosine similarity threshold for waiter embedding match.
    """

    MIN_VISIBILITY = 3
    DISAPPEAR_TIMEOUT = 20
    CONFIRM_FRAMES = 2
    WAITER_LOCK_THRESHOLD = 6
    WAITER_HIT_INCREMENT = 3
    WAITER_UNLOCK_STREAK = 8

    def __init__(
        self,
        engine: BaseInferenceEngine,
        conf: float = 0.25,
        similarity_threshold: float = 0.80,
    ) -> None:
        self.engine = engine
        self.conf = conf
        self.similarity_threshold = similarity_threshold

        # Load waiter gallery embeddings (numpy arrays or None)
        self._waiter_emb_np, self._server_emb_np = engine.get_waiter_gallery_embeddings()

        self.tracks: dict = {}
        self.locked_waiters: set = set()
        self.waiter_hits: defaultdict = defaultdict(int)
        self.waiter_non_match_streak: defaultdict = defaultdict(int)
        self.yolo_latencies: list = []
        self.tracking_latencies: list = []

        # Expose yolo reference for detect_food_in_frame compatibility
        # The engine itself is the authoritative model accessor.
        self.yolo = engine  # used by serving_detector.detect_food_in_frame

    # ------------------------------------------------------------------
    # Waiter colour heuristics (pure CPU / numpy — hardware independent)
    # ------------------------------------------------------------------

    def _has_waiter_uniform(self, crop: np.ndarray) -> bool:
        if crop is None or crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_crop, w_crop = hsv.shape[:2]
        upper_body = hsv[int(h_crop * 0.15):int(h_crop * 0.45), :]
        lower_body = hsv[int(h_crop * 0.55):int(h_crop * 0.85), :]
        if upper_body.size == 0 or lower_body.size == 0:
            return False
        mean_upper = cv2.mean(upper_body)
        mean_lower = cv2.mean(lower_body)
        upper_v = mean_upper[2]
        upper_s = mean_upper[1]
        lower_v = mean_lower[2]
        return (upper_v > 175) and (upper_s < 65) and (lower_v < 85)

    def _has_refined_waiter_uniform(self, crop: np.ndarray) -> bool:
        if crop is None or crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_crop, w_crop = hsv.shape[:2]
        x_s = int(w_crop * 0.25)
        x_e = int(w_crop * 0.75)
        upper_body = hsv[int(h_crop * 0.15):int(h_crop * 0.45), x_s:x_e]
        lower_body = hsv[int(h_crop * 0.55):int(h_crop * 0.85), x_s:x_e]
        if upper_body.size == 0 or lower_body.size == 0:
            return False
        white_pixels = (upper_body[:, :, 2] > 175) & (upper_body[:, :, 1] < 65)
        white_ratio = np.mean(white_pixels)
        black_pixels = lower_body[:, :, 2] < 85
        black_ratio = np.mean(black_pixels)
        return (white_ratio >= 0.20) and (black_ratio >= 0.40)

    # ------------------------------------------------------------------
    # Embedding + classification (via engine)
    # ------------------------------------------------------------------

    def _get_embedding_and_classify(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> tuple:
        """
        Extract ResNet50 embedding and classify as waiter/customer.
        Returns (emb_np, max_similarity, is_waiter_match).
        """
        person_crop = frame[y1:y2, x1:x2]
        is_uniform = self._has_waiter_uniform(person_crop)
        is_refined = self._has_refined_waiter_uniform(person_crop)

        if self._waiter_emb_np is None and self._server_emb_np is None:
            return (None, 0.0, is_uniform or is_refined)

        emb_np = self.engine.extract_waiter_embedding(person_crop)  # (1, 2048)

        sims = []
        if self._waiter_emb_np is not None:
            sims.append(float(np.dot(emb_np, self._waiter_emb_np.T)))
        if self._server_emb_np is not None:
            sims.append(float(np.dot(emb_np, self._server_emb_np.T)))
        max_sim = max(sims) if sims else 0.0
        is_match = (max_sim > self.similarity_threshold) or is_uniform or is_refined
        return (emb_np, max_sim, is_match)

    # ------------------------------------------------------------------
    # Main process loop
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, frame_time: float
    ) -> List[TrackedPerson]:
        try:
            return self._process_frame_impl(frame, frame_time)
        except RuntimeError as exc:
            if "cuda" in str(exc).lower() or "device" in str(exc).lower():
                print(f"[PersonTracker] Hardware error: {exc}. Retrying.")
                return self._process_frame_impl(frame, frame_time)
            raise

    def _process_frame_impl(
        self, frame: np.ndarray, frame_time: float
    ) -> List[TrackedPerson]:
        """
        1. YOLO + BoT-SORT via engine.track_persons()
        2. OSNet Re-ID embedding via engine.extract_reid_embedding()
        3. Waiter classification (colour heuristics + ResNet50 embedding)
        """
        h, w = frame.shape[:2]

        # ── Layer 1: YOLO + BoT-SORT ────────────────────────────────────
        results, yolo_lats, track_lats = self.engine.track_persons(frame, self.conf)
        self.yolo_latencies.extend(yolo_lats)
        self.tracking_latencies.extend(track_lats)

        active_ids: set = set()
        active_persons: List[TrackedPerson] = []

        if (results is not None and results[0].boxes is not None
                and results[0].boxes.id is not None):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, conf, track_id in zip(boxes, confs, track_ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                active_ids.add(track_id)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                centroid = (cx, cy)
                bbox = (x1, y1, x2, y2)

                if track_id in self.tracks:
                    self.tracks[track_id].update(bbox, centroid, frame_time)
                else:
                    self.tracks[track_id] = TrackedPerson(track_id, bbox, centroid, frame_time)

                tp = self.tracks[track_id]
                tp.yolo_conf = float(conf)

                if tp.frame_count >= self.MIN_VISIBILITY:
                    tp.confirmed = True

                # ── OSNet Re-ID embedding ──────────────────────────────────
                if tp.visual_embedding is None or tp.frame_count < 10 or tp.frame_count % 30 == 0:
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size > 0:
                        tp.visual_embedding = self.engine.extract_reid_embedding(person_crop)

                # ── Waiter classification ──────────────────────────────────
                if track_id in self.locked_waiters:
                    tp.role = "waiter"
                    if tp.frame_count % 60 == 0:
                        _, _, is_match = self._get_embedding_and_classify(frame, x1, y1, x2, y2)
                        if not is_match:
                            self.waiter_non_match_streak[track_id] += 1
                            if self.waiter_non_match_streak[track_id] >= self.WAITER_UNLOCK_STREAK:
                                self.locked_waiters.discard(track_id)
                                tp.role = "customer"
                                self.waiter_non_match_streak[track_id] = 0
                                print(f"[Tracker] Unlocked waiter track {track_id} after streak")
                        else:
                            self.waiter_non_match_streak[track_id] = 0
                else:
                    last_is_match = getattr(tp, "last_is_match", None)
                    if last_is_match is None or tp.frame_count < 30 or tp.frame_count % 15 == 0:
                        _, _, is_match = self._get_embedding_and_classify(frame, x1, y1, x2, y2)
                        tp.last_is_match = is_match
                    else:
                        is_match = last_is_match

                    if is_match:
                        self.waiter_hits[track_id] = min(
                            30, self.waiter_hits[track_id] + self.WAITER_HIT_INCREMENT
                        )
                        self.waiter_non_match_streak[track_id] = 0
                    else:
                        self.waiter_hits[track_id] = max(
                            0, self.waiter_hits[track_id] - 1
                        )

                    if self.waiter_hits[track_id] >= self.WAITER_LOCK_THRESHOLD:
                        self.locked_waiters.add(track_id)
                        tp.role = "waiter"
                        print(f"[Tracker] Locked track {track_id} as waiter")
                    else:
                        tp.role = "customer"

                active_persons.append(tp)

        # ── Cleanup disappeared tracks ───────────────────────────────────
        disappeared = set(self.tracks.keys()) - active_ids
        to_delete = []
        for tid in disappeared:
            frames_missing = (frame_time - self.tracks[tid].last_seen) * 25
            if frames_missing > self.DISAPPEAR_TIMEOUT:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]
            self.waiter_hits.pop(tid, None)

        return active_persons
