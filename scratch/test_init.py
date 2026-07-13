import sys
import os
import torch
import cv2
import json

def test_initialization():
    print("--- Running Pre-Run Checks ---")
    
    # 1. PyTorch / CUDA Check
    print("Checking PyTorch & CUDA...")
    print(f"Torch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Index: {torch.cuda.current_device()}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory} bytes")
    else:
        print("GPU is unavailable.")
        
    # 2. Verify Weights Exist
    weights = {
        "YOLOv8": "yolov8n.pt",
        "OSNet": "embedding/osnet_x1_0_msmt17.pth",
        "MediaPipe Pose": "embedding/pose_landmarker.task",
        "MediaPipe Hand": "embedding/hand_landmarker.task",
        "Tables JSON": "analytics/config/tables.json"
    }
    
    missing_files = []
    for name, path in weights.items():
        exists = os.path.exists(path)
        print(f"{name} ({path}): {'EXISTS' if exists else 'MISSING'}")
        if not exists:
            missing_files.append(path)
            
    if missing_files:
        print(f"ERROR: Missing files: {missing_files}")
        return False
        
    # 3. Load YOLOv8
    print("Initializing YOLOv8 model...")
    try:
        from ultralytics import YOLO
        device = "cuda" if cuda_available else "cpu"
        yolo_model = YOLO("yolov8n.pt")
        yolo_model.to(device)
        print("YOLOv8 loaded successfully.")
    except Exception as e:
        print(f"ERROR loading YOLOv8: {e}")
        return False
        
    # 4. Initialize OSNet definition
    print("Initializing OSNet weights...")
    try:
        from analytics.tracking.osnet import osnet_x1_0
        model = osnet_x1_0(pretrained=False)
        model.eval()
        state_dict = torch.load("embedding/osnet_x1_0_msmt17.pth", map_location="cpu")
        model_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        if cuda_available:
            model = model.cuda()
        print("OSNet loaded successfully.")
    except Exception as e:
        print(f"ERROR loading OSNet: {e}")
        return False
        
    # 5. Initialize MediaPipe
    print("Initializing MediaPipe Task Files...")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        
        # Test loading task files as byte buffers
        with open("embedding/pose_landmarker.task", "rb") as f:
            pose_data = f.read()
        with open("embedding/hand_landmarker.task", "rb") as f:
            hand_data = f.read()
        print("MediaPipe task files verified and readable.")
    except Exception as e:
        print(f"ERROR loading MediaPipe: {e}")
        return False
        
    # 6. Database Initialization
    print("Initializing Database connection...")
    try:
        from analytics.database.database_manager import DatabaseManager
        db_manager = DatabaseManager()
        db_manager.initialize_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"ERROR initializing database: {e}")
        return False
        
    # 7. Table Polygons Load
    print("Loading Table Polygons config...")
    try:
        with open("analytics/config/tables.json", "r") as f:
            tables_data = json.load(f)
        print(f"Table Polygons config loaded successfully with {len(tables_data)} tables.")
    except Exception as e:
        print(f"ERROR loading table config: {e}")
        return False
        
    print("--- Pre-Run Checks PASSED ---")
    return True

if __name__ == "__main__":
    success = test_initialization()
    sys.exit(0 if success else 1)
