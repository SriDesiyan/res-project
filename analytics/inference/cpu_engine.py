"""
analytics/inference/cpu_engine.py
=====================================
CpuInferenceEngine — identical to CudaInferenceEngine but forces all
models to run on the CPU.

Use cases:
  1. Validation: Compare outputs against CUDA engine to confirm no regressions.
  2. Deployment: Machines with no GPU.
  3. Intermediate step: Run on RK3588 ARM CPU before AIPU compilation.

No model logic is different from CudaInferenceEngine — every call delegates
to the parent after overriding ``device = cpu``.
"""
from __future__ import annotations

from pathlib import Path

import torch

from analytics.inference.cuda_engine import CudaInferenceEngine


class CpuInferenceEngine(CudaInferenceEngine):
    """
    CPU-forced inference engine — delegates to CudaInferenceEngine with
    ``torch.device("cpu")``.

    Parameters are identical to CudaInferenceEngine.
    """

    def __init__(
        self,
        yolo_model_path: str | Path,
        tracker_cfg: str,
        embedding_dir: str | Path,
        conf: float = 0.25,
    ) -> None:
        super().__init__(
            device=torch.device("cpu"),
            yolo_model_path=yolo_model_path,
            tracker_cfg=tracker_cfg,
            embedding_dir=embedding_dir,
            conf=conf,
        )
        print("[CpuEngine] Initialised (all models on CPU)")

    # ------------------------------------------------------------------
    # Override metadata
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return "cpu"

    def is_cuda_available(self) -> bool:
        # Engine is CPU-forced regardless of hardware
        return False

    def is_axelera_available(self) -> bool:
        return False
