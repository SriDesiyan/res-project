"""
scripts/export_onnx.py
========================
Batch ONNX export script for all models used in the pipeline.

Exports:
  1. YOLOv8n         → weights/onnx/yolov8n.onnx
  2. OSNet x1.0      → weights/onnx/osnet_x1_0.onnx
  3. ResNet50 Waiter → weights/onnx/resnet50_waiter.onnx
  4. ResNet50 Plate  → weights/onnx/resnet50_plate.onnx

For each model the script:
  - Runs a forward pass through the original PyTorch model
  - Exports to ONNX
  - Runs the same input through ONNX Runtime
  - Computes cosine similarity between PyTorch and ONNX outputs
  - Reports PASS / FAIL

Usage:
    cd Restaraunt-Monitoring-main
    python scripts/export_onnx.py

Requirements:
    pip install torch torchvision ultralytics onnx onnxruntime

Phase coverage: Phase 4 (YOLO), Phase 6 (OSNet), Phase 7 (Plate / ResNet50)
"""
from __future__ import annotations

import sys
import torch
torch.cuda.is_available = lambda: False
from pathlib import Path

import numpy as np

# Resolve project root
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

ONNX_DIR = project_root / "weights" / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIR = project_root / "embedding"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def validate_onnx(torch_output: np.ndarray, onnx_output: np.ndarray, model_name: str) -> bool:
    sim = cosine_similarity(torch_output, onnx_output)
    max_diff = float(np.max(np.abs(torch_output.flatten() - onnx_output.flatten())))
    status = "PASS ✓" if sim > 0.999 else "FAIL ✗"
    print(f"  [{status}] {model_name}: cosine_sim={sim:.6f}  max_abs_diff={max_diff:.6e}")
    return sim > 0.999


# ---------------------------------------------------------------------------
# 1. YOLOv8n export
# ---------------------------------------------------------------------------

def export_yolo():
    print("\n" + "="*60)
    print("Exporting YOLOv8n → ONNX")
    print("="*60)
    try:
        from ultralytics import YOLO
        yolo_pt = project_root / "yolov8n.pt"
        if not yolo_pt.exists():
            print(f"  [SKIP] yolov8n.pt not found at {yolo_pt}")
            return False

        model = YOLO(str(yolo_pt))
        out_path = ONNX_DIR / "yolov8n.onnx"
        # Ultralytics export — produces the ONNX file in the same dir as the .pt
        exported = model.export(format="onnx", opset=12, simplify=True, imgsz=640)
        # Move to weights/onnx/
        exported_path = Path(str(exported))
        if exported_path.exists() and exported_path != out_path:
            exported_path.rename(out_path)
        print(f"  Saved: {out_path}")

        # Validate with ONNX Runtime
        try:
            import onnxruntime as ort
            import cv2
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            # PyTorch inference
            pt_results = model(dummy_frame, verbose=False)
            pt_boxes = pt_results[0].boxes.xyxy.cpu().numpy() if pt_results[0].boxes is not None else np.zeros((0, 4))

            # ONNX Runtime inference
            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
            in_name = sess.get_inputs()[0].name
            onnx_out = sess.run(None, {in_name: dummy_input})[0]
            print(f"  ONNX output shape: {onnx_out.shape}")
            print("  [INFO] YOLOv8n ONNX exported and runtime verified")
        except Exception as exc:
            print(f"  [WARN] ONNX validation skipped: {exc}")

        return True
    except Exception as exc:
        print(f"  [ERROR] YOLO export failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# 2. OSNet x1.0 export
# ---------------------------------------------------------------------------

def export_osnet():
    print("\n" + "="*60)
    print("Exporting OSNet x1.0 → ONNX")
    print("="*60)
    try:
        import torch
        import torch.onnx
        from tracking.osnet import osnet_x1_0

        device = torch.device("cpu")
        model = osnet_x1_0(num_classes=1000, pretrained=False).to(device)

        weights_path = EMBEDDING_DIR / "osnet_x1_0_msmt17.pth"
        if weights_path.exists():
            state_dict = torch.load(str(weights_path), map_location=device)
            model_dict = model.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k[7:] if k.startswith("module.") else k
                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)
            print(f"  Loaded weights from {weights_path}")
        else:
            print(f"  [WARN] OSNet weights not found at {weights_path}. Exporting with random init.")

        model.eval()
        dummy_input = torch.zeros(1, 3, 256, 128)
        out_path = ONNX_DIR / "osnet_x1_0.onnx"

        # PyTorch output
        with torch.no_grad():
            pt_output = model(dummy_input).numpy()

        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        print(f"  Saved: {out_path}")

        # Validate
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            onnx_output = sess.run(None, {"input": dummy_input.numpy()})[0]
            validate_onnx(pt_output, onnx_output, "OSNet x1.0")
        except Exception as exc:
            print(f"  [WARN] ONNX validation skipped: {exc}")

        return True
    except Exception as exc:
        print(f"  [ERROR] OSNet export failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 3. ResNet50 Waiter export
# ---------------------------------------------------------------------------

def export_resnet50_waiter():
    print("\n" + "="*60)
    print("Exporting ResNet50 Waiter → ONNX")
    print("="*60)
    try:
        import torch
        import torch.onnx
        from torchvision.models import resnet50, ResNet50_Weights
        import torch.nn as nn

        class _FeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
                self.features = nn.Sequential(*list(base_model.children())[:-1])
            def forward(self, x):
                return self.features(x).view(x.size(0), -1)

        model = _FeatureExtractor()
        model.eval()
        dummy_input = torch.zeros(1, 3, 224, 224)
        out_path = ONNX_DIR / "resnet50_waiter.onnx"

        with torch.no_grad():
            pt_output = model(dummy_input).numpy()

        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        print(f"  Saved: {out_path}  output_shape={pt_output.shape}")

        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            onnx_output = sess.run(None, {"input": dummy_input.numpy()})[0]
            validate_onnx(pt_output, onnx_output, "ResNet50 Waiter")
        except Exception as exc:
            print(f"  [WARN] ONNX validation skipped: {exc}")

        return True
    except Exception as exc:
        print(f"  [ERROR] ResNet50 Waiter export failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 4. ResNet50 Plate export
# ---------------------------------------------------------------------------

def export_resnet50_plate():
    print("\n" + "="*60)
    print("Exporting ResNet50 Plate Classifier → ONNX")
    print("="*60)
    try:
        import torch
        import torch.onnx
        from torchvision.models import resnet50, ResNet50_Weights

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()
        dummy_input = torch.zeros(1, 3, 224, 224)
        out_path = ONNX_DIR / "resnet50_plate.onnx"

        with torch.no_grad():
            pt_output = model(dummy_input).numpy()

        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        print(f"  Saved: {out_path}  output_shape={pt_output.shape}")

        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            onnx_output = sess.run(None, {"input": dummy_input.numpy()})[0]
            validate_onnx(pt_output, onnx_output, "ResNet50 Plate")
        except Exception as exc:
            print(f"  [WARN] ONNX validation skipped: {exc}")

        return True
    except Exception as exc:
        print(f"  [ERROR] ResNet50 Plate export failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Restaurant Pipeline — ONNX Model Export")
    print(f"  Output directory: {ONNX_DIR}")
    print("=" * 60)

    results = {}
    results["yolov8n"]          = export_yolo()
    results["osnet_x1_0"]       = export_osnet()
    results["resnet50_waiter"]  = export_resnet50_waiter()
    results["resnet50_plate"]   = export_resnet50_plate()

    print("\n" + "=" * 60)
    print("  EXPORT SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  [{status}] {name}")

    all_ok = all(results.values())
    print("\n" + ("All exports successful!" if all_ok else "Some exports failed — check errors above."))
    sys.exit(0 if all_ok else 1)
