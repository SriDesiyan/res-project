# CUDA Dependency Audit — Restaurant CCTV Analytics Pipeline

This document logs all GPU/CUDA dependencies in the Restaurant CCTV Analytics Pipeline and classifies them for CPU and Axelera Metis AIPU compatibility.

---

## 1. CUDA/GPU API Usage & Allocations

### 1.1 `torch.cuda` Calls & Device Assignments
- **`torch.cuda.is_available()`**
  - **Location**: `analytics/inference/engine_factory.py` (Line 39, Line 106), `analytics/pipeline.py` (Line 282, Line 291), `analytics/inference/cuda_engine.py` (Line 209, Line 396, Line 399), `classify/*.py` (utility scripts), and `bow_server_video_pipeline.py`.
  - **Purpose**: Checks if NVIDIA GPU is present to automatically resolve the backend.
  - **Classification**: **CPU/Axelera Compatible** (returns `False` on RK3588; handled via auto-fallback logic or mocked dynamically).
- **`torch.cuda.empty_cache()`**
  - **Location**: `analytics/pipeline.py` (Line 285), `analytics/inference/cuda_engine.py` (Line 210).
  - **Purpose**: Releases unused GPU memory.
  - **Classification**: **CPU/Axelera Compatible** (noop or safe to bypass on non-CUDA hardware).
- **`device = torch.device("cuda")` / `device="cuda"`**
  - **Location**: `analytics/inference/engine_factory.py` (Line 135) and `analytics/inference/cuda_engine.py` (Line 85).
  - **Purpose**: Directs models and tensors to GPU memory.
  - **Classification**: **Requires Conversion** (must map to `"cpu"` for CPU/ONNX, or route memory buffers through Axelera Metis hardware APIs).

### 1.2 GPU Tensor Allocations & Moves (`.to(device)`)
- **YOLOv8 Model Move**: `self.yolo.to(device)` (`analytics/inference/cuda_engine.py`: Line 112)
- **Feature Extractor Move**: `self._extractor = _FeatureExtractor().to(device)` (`analytics/inference/cuda_engine.py`: Line 115)
- **Plate Model Move**: `self._plate_model = resnet50(...).to(device)` (`analytics/inference/cuda_engine.py`: Line 127)
- **OSNet Move**: `self._reid_extractor = osnet_x1_0(...).to(device)` (`analytics/inference/cuda_engine.py`: Line 132)
- **Input Tensor Transfers**:
  - Re-ID: `tensor = self._reid_transform(img_pil).unsqueeze(0).to(self.device)` (`cuda_engine.py`: Line 292)
  - Waiter embedding: `tensor = self._resnet_transform(top_img).unsqueeze(0).to(self.device)` (`cuda_engine.py`: Line 303)
  - Plate preprocessing: `torch.from_numpy(rgb)...to(self.device)` (`cuda_engine.py`: Line 375)
- **Classification**: **Requires Conversion** (tensor movements to `"cuda"` must be replaced by standard NumPy/CPU tensors or Axelera input descriptor buffers).

---

## 2. CUDA-Only & GPU-Specific Imports

- **`torch` & `torch.cuda`**: Used heavily throughout `cuda_engine.py`, `pipeline.py`, and classification helpers.
  - **Classification**: **Requires Conversion** (for edge deployment, PyTorch overhead should be bypassed by running ONNX Runtime on the host CPU and offloading model inference to the Axelera AIPU).
- **`torchvision.transforms`**: Used for input tensor normalization on GPU.
  - **Classification**: **Requires Conversion** (can be replaced with lightweight, hardware-independent NumPy/OpenCV pre-processing logic).

---

## 3. Inference Models Compatibility Matrix

| Inference Model | Weights / Format | Host CPU Compatible | Axelera Metis AIPU Compatible | Migration Path / Status |
| :--- | :--- | :---: | :---: | :--- |
| **YOLOv8 Object Detection** | `yolov8n.pt` (PyTorch) | Yes (CPU) | **Yes** (AIPU) | Export to `.onnx` -> Quantize/Compile to `.axm` |
| **OSNet x1.0 Re-ID** | `osnet_x1_0_msmt17.pth` | Yes (CPU) | **Yes** (AIPU) | Export to `.onnx` -> Quantize/Compile to `.axm` |
| **ResNet50 Waiter Classifier** | Torchvision State Dict | Yes (CPU) | **Yes** (AIPU) | Export to `.onnx` -> Quantize/Compile to `.axm` |
| **ResNet50 Plate Classifier** | Torchvision State Dict | Yes (CPU) | **Yes** (AIPU) | Export to `.onnx` -> Quantize/Compile to `.axm` |
| **MediaPipe Pose Landmarker** | `pose_landmarker.task` | **Yes (CPU)** | No | **Host CPU Fallback** (non-standard operations run on RK3588 host CPU) |
| **MediaPipe Hand Landmarker** | `hand_landmarker.task` | **Yes (CPU)** | No | **Host CPU Fallback** (non-standard operations run on RK3588 host CPU) |

---

## 4. Summary & Recommendation

1. **Host CPU Fallback for MediaPipe**: MediaPipe landmarks contain post-processing calculators that cannot be mapped to the Metis hardware. These models must continue running on the Rockchip RK3588 host CPU.
2. **Model Conversion to ONNX**: The remaining neural networks (YOLOv8, OSNet, and the ResNet50 models) must be exported to ONNX format.
3. **Axelera Metis AIPU Target**: ONNX models must be compiled to `.axm` format using the Axelera Voyager SDK compiler to utilize the AIPU acceleration.
4. **Dynamic CUDA Mocking**: To prevent Ultralytics and PyTorch from seeking CUDA and hanging on the edge board, a dynamic mocking strategy for `torch.cuda.is_available = lambda: False` should be implemented for non-CUDA backends.
