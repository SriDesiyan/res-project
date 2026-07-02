"""
Dwell-Point Clustering — Self-learning table ROI discovery.

Collects centroid coordinates of stationary customers over time and
uses DBSCAN spatial clustering to discover table locations without
any prior configuration.

Usage (programmatic — called from pipeline):
    from dwell_cluster import DwellClusterManager
    dcm = DwellClusterManager()

    # During each frame:
    dcm.record_dwell_point((cx, cy), frame_time)

    # Periodically:
    discovered_tables = dcm.run_clustering()
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None


class DwellClusterManager:
    """
    Discovers table locations by clustering customer dwell positions.

    Parameters
    ----------
    eps : float
        DBSCAN neighbourhood radius in pixels.  Two dwell points
        closer than *eps* are considered neighbours.  Tune based on
        camera resolution and table spacing.
    min_samples : int
        Minimum dwell points in a neighbourhood to form a cluster.
        Higher values suppress noise but need more data.
    max_points : int
        Rolling window size — older dwell points are evicted when
        the buffer exceeds this limit.
    min_cluster_area : float
        Ignore clusters whose convex-hull area is below this value
        (filters out spurious single-chair clusters).
    simplify_epsilon_ratio : float
        Polygon simplification ratio (same as ``auto_roi.py``).
    """

    def __init__(
        self,
        eps: float = 120.0,
        min_samples: int = 15,
        max_points: int = 5000,
        min_cluster_area: float = 500.0,
        simplify_epsilon_ratio: float = 0.02,
    ):
        if DBSCAN is None:
            raise ImportError(
                "scikit-learn is required for dwell clustering.  "
                "Install it with:  pip install scikit-learn"
            )

        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_area = min_cluster_area
        self.simplify_epsilon_ratio = simplify_epsilon_ratio

        # Rolling buffer of (x, y) dwell points
        self._points: deque[tuple[float, float]] = deque(maxlen=max_points)
        self._timestamps: deque[float] = deque(maxlen=max_points)

    # ── data collection ────────────────────────────────────────────
    def record_dwell_point(self, centroid: tuple[float, float], frame_time: float):
        """
        Record a single dwell observation.

        Call this once per frame for every *confirmed customer* whose
        velocity is near zero (i.e. they are seated / stationary).
        """
        self._points.append(centroid)
        self._timestamps.append(frame_time)

    @property
    def point_count(self) -> int:
        return len(self._points)

    def clear(self):
        """Flush all collected dwell points."""
        self._points.clear()
        self._timestamps.clear()

    # ── clustering ─────────────────────────────────────────────────
    def run_clustering(self) -> dict:
        """
        Run DBSCAN on accumulated dwell points and return discovered
        table definitions.

        Returns
        -------
        dict
            ``{table_id: {"polygon": [...], "center": [cx, cy]}}``
            Compatible with ``roi_config.load_tables``.
        """
        if len(self._points) < self.min_samples:
            print(
                f"[DWELL] Not enough points ({len(self._points)}) "
                f"for clustering (need ≥{self.min_samples})."
            )
            return {}

        X = np.array(list(self._points), dtype=np.float64)

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = db.fit_predict(X)

        unique_labels = set(labels)
        unique_labels.discard(-1)  # remove noise label

        if not unique_labels:
            print("[DWELL] No clusters found — all points classified as noise.")
            return {}

        tables: dict = {}
        table_counter = 1

        for label in sorted(unique_labels):
            cluster_mask = labels == label
            cluster_points = X[cluster_mask]

            if len(cluster_points) < 3:
                continue

            # Compute convex hull to get the outer boundary
            hull = cv2.convexHull(cluster_points.astype(np.float32))

            hull_area = cv2.contourArea(hull)
            if hull_area < self.min_cluster_area:
                continue

            # Simplify the hull polygon
            arc_len = cv2.arcLength(hull, closed=True)
            epsilon = self.simplify_epsilon_ratio * arc_len
            approx = cv2.approxPolyDP(hull, epsilon, closed=True)
            simplified = approx.reshape(-1, 2).astype(int).tolist()

            if len(simplified) < 3:
                continue

            # Dilate the simplified polygon by 10% (outward safety margin)
            poly = np.array(simplified, dtype=np.float32)
            centroid = np.mean(poly, axis=0)
            dilated_poly = centroid + (poly - centroid) * 1.10
            dilated = dilated_poly.astype(int).tolist()

            cx = float(np.mean(cluster_points[:, 0]))
            cy = float(np.mean(cluster_points[:, 1]))

            table_id = f"table_{table_counter}"
            tables[table_id] = {
                "polygon": dilated,
                "center": [round(cx, 1), round(cy, 1)],
            }
            table_counter += 1

            print(
                f"  [DWELL] {table_id}: {len(cluster_points)} dwell points → "
                f"{len(simplified)} vertices, area={hull_area:.0f}px²"
            )

        print(f"[DWELL] Discovered {len(tables)} table(s) from "
              f"{len(self._points)} dwell points.")
        return tables

    # ── persistence ────────────────────────────────────────────────
    def save_points(self, path: Path):
        """Persist raw dwell points to JSON for offline analysis."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "points": list(self._points),
            "timestamps": list(self._timestamps),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[DWELL] Saved {len(self._points)} points → {path}")

    def load_points(self, path: Path):
        """Load previously saved dwell points."""
        if not path.exists():
            return
        with open(path, "r") as f:
            data = json.load(f)
        for pt, ts in zip(data["points"], data["timestamps"]):
            self._points.append(tuple(pt))
            self._timestamps.append(ts)
        print(f"[DWELL] Loaded {len(data['points'])} points from {path}")
