"""
analytics/config/edge_config.py
================================
Centralized configuration for the Edge AI inference backend and all
hardware-specific optimization parameters.

All constants that previously appeared as magic numbers throughout the
pipeline are defined here and imported by name.  Nothing in the inference
or pipeline layer should contain hard-coded optimization values.

Backend Selection
-----------------
``backend``
    "auto"    — detect hardware at runtime, pick best available engine
    "cuda"    — force NVIDIA CUDA (original behaviour)
    "cpu"     — force PyTorch CPU (validation / low-power fallback)
    "axelera" — Axelera Metis AIPU via Voyager SDK
    "onnx"    — ONNX Runtime on CPU (intermediate migration step)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Backend / hardware selection
# ---------------------------------------------------------------------------

EDGE_CONFIG: dict = {

    # ── Hardware backend ────────────────────────────────────────────────────
    "backend": "auto",

    # ── YOLO detection ──────────────────────────────────────────────────────
    # Run YOLO every N frames; BoT-SORT propagates tracks on skipped frames.
    # Set to 1 to match original behaviour (YOLO every frame).
    "yolo_frame_skip": 3,

    # YOLO input resolution (must match the compiled ONNX/AIPU model).
    "yolo_input_size": 640,

    # ── Pose / serving detection ─────────────────────────────────────────────
    # Skip pose inference when no waiter is on screen in this frame.
    "pose_only_when_waiter": True,

    # Subsample pose per waiter: run every N frames (was hardcoded 6 in pipeline.py).
    "pose_frame_subsample": 6,

    # Minimum waiter-to-table-center distance (px) below which pose is skipped.
    "pose_skip_distance_px": 600.0,

    # ── Plate / food detection ───────────────────────────────────────────────
    # Only run plate / food detection when at least one eligible waiter is near a table.
    "plate_only_when_serving": True,

    # ── Re-ID / OSNet embeddings ─────────────────────────────────────────────
    # Run full Re-ID on new tracks only; use cached embedding on known tracks.
    "reid_on_new_track_only": True,

    # Update embedding every N frames for known tracks.
    "reid_update_every_n_frames": 30,

    # ── ROI crop optimization ────────────────────────────────────────────────
    # Run Pose, Plate, and Classification on cropped ROIs instead of full frame.
    "roi_crop_inference": True,

    # ── Model precision ──────────────────────────────────────────────────────
    # "fp32" | "fp16" | "int8"
    # FP32 = original PyTorch precision.
    # FP16 = half-precision (CUDA / Axelera).
    # INT8 = post-training quantization (Axelera / ONNX Runtime).
    "model_precision": "fp32",          # Start with FP32 for validation

    # ── Input sizes ──────────────────────────────────────────────────────────
    "osnet_input_size": (256, 128),     # (height, width) — OSNet standard
    "resnet_input_size": (224, 224),    # ResNet50 standard

    # ── Batching ─────────────────────────────────────────────────────────────
    "batch_size": 1,

    # ── Model caching / lazy loading ─────────────────────────────────────────
    "cache_embeddings": True,
    "lazy_load_models": True,

    # ── Axelera / Voyager SDK ────────────────────────────────────────────────
    # Number of warm-up frames to run through AIPU before timing begins.
    "axelera_warmup_frames": 10,

    # Path to pre-compiled Axelera model artifacts.
    "axelera_model_dir": "weights/axelera",

    # ── ONNX Runtime ─────────────────────────────────────────────────────────
    "onnx_model_dir": "weights/onnx",

    # ── Memory management ────────────────────────────────────────────────────
    # Call gc.collect() (and cuda.empty_cache() on CUDA) every N frames.
    "gc_interval_frames": 300,

    # ── Threading ────────────────────────────────────────────────────────────
    # Maximum number of frames to buffer in the inference queue.
    "inference_queue_maxsize": 4,

    # ── Benchmark collection ─────────────────────────────────────────────────
    # Collect CPU/GPU utilisation every N frames.
    "perf_sample_interval_frames": 10,
}


def get(key: str, default=None):
    """Convenience accessor with optional default."""
    return EDGE_CONFIG.get(key, default)
