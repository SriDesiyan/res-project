"""
scripts/compile_axelera.py
============================
Voyager SDK model compilation wrapper.

Compiles pre-exported ONNX models to Axelera .axm format for execution
on the Metis AIPU.  Run this on the Axelera development host or Metis board
after running scripts/export_onnx.py.

Prerequisites:
    - Axelera Voyager SDK installed
    - pip install axelera-voyager  (or install from SDK package)
    - ONNX models must exist in weights/onnx/

Output:
    weights/axelera/*.axm  (compiled AIPU models)

Usage:
    python scripts/compile_axelera.py [--int8] [--model all|yolo|osnet|resnet_waiter|resnet_plate]

Phase coverage: Phase 4 (YOLO), Phase 6 (OSNet), Phase 7 (Plate), Phase 8 (INT8)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
ONNX_DIR = project_root / "weights" / "onnx"
AXELERA_DIR = project_root / "weights" / "axelera"
AXELERA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Voyager SDK import (optional — skip gracefully on machines without SDK)
# ---------------------------------------------------------------------------
try:
    import axelera.compiler as _compiler  # type: ignore[import]
    _SDK_AVAILABLE = True
except ImportError:
    _compiler = None
    _SDK_AVAILABLE = False


def compile_model(
    onnx_path: Path,
    output_path: Path,
    quantize_int8: bool = False,
    calibration_data: Path | None = None,
) -> bool:
    """
    Compile a single ONNX model to Axelera .axm.

    Parameters
    ----------
    onnx_path : Path
        Input .onnx file.
    output_path : Path
        Output .axm file.
    quantize_int8 : bool
        Apply INT8 post-training quantization if True.
    calibration_data : Path | None
        Directory with calibration images for INT8 quantization.

    Returns
    -------
    bool
        True if compilation succeeded.
    """
    print(f"\n[Compiler] {onnx_path.name} -> {output_path.name}")

    if not onnx_path.exists():
        print(f"  [SKIP] ONNX file not found: {onnx_path}")
        return False

    if not _SDK_AVAILABLE:
        print("  [SKIP] Voyager SDK not installed.")
        print("         Install the Axelera SDK to enable AIPU compilation.")
        print(f"         ONNX model is ready at: {onnx_path}")
        return False

    try:
        # Voyager SDK compilation call
        # NOTE: Exact API depends on the installed Voyager SDK version.
        #       This follows the standard Voyager SDK Python API.
        compile_config = {
            "input_model": str(onnx_path),
            "output_model": str(output_path),
            "target": "metis",           # Axelera Metis AIPU
            "precision": "int8" if quantize_int8 else "fp16",
        }

        if quantize_int8 and calibration_data and calibration_data.exists():
            compile_config["calibration_data"] = str(calibration_data)

        _compiler.compile(**compile_config)
        print(f"  [PASS ✓] Compiled → {output_path}  (precision={'INT8' if quantize_int8 else 'FP16'})")
        return True

    except Exception as exc:
        print(f"  [FAIL ✗] Compilation error: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compile ONNX models for Axelera Metis AIPU")
    parser.add_argument(
        "--model", default="all",
        choices=["all", "yolo", "osnet", "resnet_waiter", "resnet_plate"],
        help="Which model to compile (default: all)"
    )
    parser.add_argument(
        "--int8", action="store_true",
        help="Apply INT8 post-training quantization (requires calibration data)"
    )
    parser.add_argument(
        "--calibration-dir", type=Path, default=None,
        help="Directory of calibration images for INT8 quantization"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Restaurant Pipeline — Axelera AIPU Model Compilation")
    print(f"  ONNX source:   {ONNX_DIR}")
    print(f"  AIPU output:   {AXELERA_DIR}")
    print(f"  Precision:     {'INT8' if args.int8 else 'FP16'}")
    print(f"  Voyager SDK:   {'Available' if _SDK_AVAILABLE else 'NOT INSTALLED'}")
    print("=" * 60)

    models = {
        "yolo":          (ONNX_DIR / "yolov8n.onnx",         AXELERA_DIR / "yolov8n.axm"),
        "osnet":         (ONNX_DIR / "osnet_x1_0.onnx",      AXELERA_DIR / "osnet_x1_0.axm"),
        "resnet_waiter": (ONNX_DIR / "resnet50_waiter.onnx",  AXELERA_DIR / "resnet50_waiter.axm"),
        "resnet_plate":  (ONNX_DIR / "resnet50_plate.onnx",   AXELERA_DIR / "resnet50_plate.axm"),
    }

    selected = list(models.keys()) if args.model == "all" else [args.model]
    results = {}
    for name in selected:
        onnx_path, axm_path = models[name]
        results[name] = compile_model(
            onnx_path, axm_path,
            quantize_int8=args.int8,
            calibration_data=args.calibration_dir,
        )

    print("\n" + "=" * 60)
    print("  COMPILATION SUMMARY")
    print("=" * 60)
    for name in selected:
        s = "PASS ✓" if results[name] else "SKIP/FAIL"
        print(f"  [{s}] {name}")

    if not _SDK_AVAILABLE:
        print("""
NOTE: Voyager SDK is not installed on this machine.
  To compile for Axelera Metis:
    1. Install the Axelera Voyager SDK from your Axelera partner portal.
    2. Re-run this script on the development host or Metis board.
    3. Copy the .axm files to weights/axelera/ on the target device.
  Until .axm files exist, the AxeleraInferenceEngine falls back to ONNX Runtime.
""")


if __name__ == "__main__":
    main()
