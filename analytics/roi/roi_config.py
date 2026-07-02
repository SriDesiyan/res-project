"""
ROI Configuration — Polygon math and table assignment logic.

Provides:
- point_in_polygon: ray-casting algorithm
- assign_to_table: maps a centroid to the nearest table ROI
- load_tables / save_tables: JSON persistence
"""
import json
import numpy as np
from pathlib import Path


def load_tables(json_path: str | Path) -> dict:
    """Load table ROI definitions from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("tables", {})


def save_tables(tables: dict, json_path: str | Path):
    """Save table ROI definitions to JSON."""
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"tables": tables}, f, indent=2)
    print(f"Saved {len(tables)} tables to {json_path}")


def point_in_polygon(point: tuple, polygon: list) -> bool:
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def polygon_center(polygon: list) -> tuple:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def distance(p1: tuple, p2: tuple) -> float:
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


def assign_to_table(centroid: tuple, tables: dict) -> str | None:
    # assign centroid to a table
    if not tables:
        return None

    # Priority 1: check if point is inside a polygon
    for table_id, table_info in tables.items():
        polygon = table_info["polygon"]
        if point_in_polygon(centroid, polygon):
            return table_id

    min_dist = float("inf")
    nearest = None
    for table_id, table_info in tables.items():
        center = tuple(table_info["center"])
        d = distance(centroid, center)
        if d < min_dist:
            min_dist = d
            nearest = table_id

    if min_dist < 200:
        return nearest

    return None


def get_table_polygon_np(table_info: dict) -> np.ndarray:
    return np.array(table_info["polygon"], dtype=np.int32)
