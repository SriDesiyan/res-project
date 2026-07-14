"""
analytics/inference/engine_factory.py
=======================================
Hardware-aware factory that selects and instantiates the appropriate
inference engine at runtime.

Auto-detection priority (when backend="auto"):
  1. Axelera Metis AIPU  — if Voyager SDK + .axm files present
  2. ONNX Runtime        — if onnxruntime installed + .onnx files present
  3. CUDA PyTorch        — if torch.cuda.is_available()
  4. CPU PyTorch         — unconditional fallback

Usage
-----
    from analytics.inference.engine_factory import create_engine

    engine = create_engine(
        backend="auto",          # or "cuda" | "cpu" | "onnx" | "axelera"
        project_root=Path("."),
    )

No other module should call torch / YOLO / ONNX / Axelera directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from analytics.inference.base_engine import BaseInferenceEngine


# ---------------------------------------------------------------------------
# Soft-imports for availability probes (no hard crash if missing)
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _ort_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _axelera_available() -> bool:
    try:
        import axelera.runtime  # noqa: F401  type: ignore[import]
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_engine(
    backend: str = "auto",
    project_root: Optional[Path] = None,
    conf: float = 0.25,
) -> BaseInferenceEngine:
    """
    Create and return an inference engine.

    Parameters
    ----------
    backend : str
        "auto" | "cuda" | "cpu" | "onnx" | "axelera"
    project_root : Path | None
        Project root directory.  Defaults to two levels above this file.
    conf : float
        Default YOLO detection confidence threshold.

    Returns
    -------
    BaseInferenceEngine
        A fully initialised engine ready for ``warmup()`` and inference.
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.resolve()

    yolo_path = project_root / "yolov8n.pt"
    embedding_dir = project_root / "embedding"
    onnx_dir = project_root / "weights" / "onnx"
    axelera_dir = project_root / "weights" / "axelera"

    # Locate BoT-SORT config
    tracker_cfg_path = embedding_dir / "tracker_config.yaml"
    tracker_cfg = str(tracker_cfg_path) if tracker_cfg_path.exists() else "botsort.yaml"

    resolved = _resolve_backend(backend, onnx_dir, axelera_dir)
    print(f"[EngineFactory] Requested backend='{backend}' -> resolved='{resolved}'")

    # Mock torch.cuda.is_available to return False for non-CUDA backends
    # to bypass Ultralytics auto-updating onnxruntime-gpu
    if resolved in ("cpu", "onnx", "axelera"):
        try:
            import torch
            torch.cuda.is_available = lambda: False
            print("[EngineFactory] Mocked torch.cuda.is_available to False")
        except ImportError:
            pass

    if resolved == "axelera":
        from analytics.inference.axelera_engine import AxeleraInferenceEngine
        return AxeleraInferenceEngine(
            yolo_model_path=yolo_path,
            tracker_cfg=tracker_cfg,
            axelera_dir=axelera_dir,
            onnx_dir=onnx_dir,
            embedding_dir=embedding_dir,
            conf=conf,
        )

    if resolved == "onnx":
        from analytics.inference.onnx_engine import OnnxInferenceEngine
        return OnnxInferenceEngine(
            yolo_model_path=yolo_path,
            tracker_cfg=tracker_cfg,
            onnx_dir=onnx_dir,
            embedding_dir=embedding_dir,
            conf=conf,
        )

    if resolved == "cuda":
        import torch
        from analytics.inference.cuda_engine import CudaInferenceEngine
        device = torch.device("cuda")
        return CudaInferenceEngine(
            device=device,
            yolo_model_path=yolo_path,
            tracker_cfg=tracker_cfg,
            embedding_dir=embedding_dir,
            conf=conf,
        )

    # cpu
    from analytics.inference.cpu_engine import CpuInferenceEngine
    return CpuInferenceEngine(
        yolo_model_path=yolo_path,
        tracker_cfg=tracker_cfg,
        embedding_dir=embedding_dir,
        conf=conf,
    )


def _resolve_backend(backend: str, onnx_dir: Path, axelera_dir: Path) -> str:
    """
    Resolve "auto" to a concrete backend, or validate explicit requests.
    """
    if backend == "auto":
        if _axelera_available() and any(axelera_dir.glob("*.axm")):
            return "axelera"
        if _ort_available() and any(onnx_dir.glob("*.onnx")):
            return "onnx"
        if _cuda_available():
            return "cuda"
        return "cpu"

    if backend == "axelera":
        if not _axelera_available():
            print("[EngineFactory] WARNING: Voyager SDK not found — falling back to 'onnx'")
            return _resolve_backend("auto", onnx_dir, axelera_dir)
        return "axelera"

    if backend == "onnx":
        if not _ort_available():
            print("[EngineFactory] WARNING: onnxruntime not installed — falling back to 'cpu'")
            return "cuda" if _cuda_available() else "cpu"
        return "onnx"

    if backend == "cuda":
        if not _cuda_available():
            print("[EngineFactory] WARNING: CUDA not available — falling back to 'cpu'")
            return "cpu"
        return "cuda"

    if backend == "cpu":
        return "cpu"

    raise ValueError(
        f"Unknown backend '{backend}'. "
        "Valid values: auto | cuda | cpu | onnx | axelera"
    )
