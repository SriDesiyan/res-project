import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

class PlateDetector:
    # Relevant ImageNet indices:
    # 504: coffee mug, 923: plate, 968: cup, 868: tray, 809: soup bowl, 
    # 659: mixing bowl, 440: beer bottle, 737: pop bottle, 898: water bottle, 907: wine bottle
    DISHWARE_INDICES = {504, 923, 968, 868, 809, 659, 440, 737, 898, 907}

    def __init__(self, device="cpu"):
        self.device = device
        # load pretrained model
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights).to(device)
        self.model.eval()
        self.preprocess = weights.transforms()

    def detect_dirty_objects(self, frame, polygon_pts) -> int:
        # 1-> if detects dirty dishware else 0
        if polygon_pts is None or len(polygon_pts) == 0:
            return 0
        h, w = frame.shape[:2]
        # turn into a rectangular crop box for comparison
        x1 = int(max(0, np.min(polygon_pts[:, 0])))
        y1 = int(max(0, np.min(polygon_pts[:, 1])))
        x2 = int(min(w, np.max(polygon_pts[:, 0])))
        y2 = int(min(h, np.max(polygon_pts[:, 1])))
        
        if x2 - x1 < 20 or y2 - y1 < 20:
            return 0
            
        crop = frame[y1:y2, x1:x2]
        
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        try:
            img_tensor = self.preprocess(torch.from_numpy(crop_rgb).permute(2, 0, 1)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(img_tensor)
        except RuntimeError as e:
            if "cuda" in str(e).lower() or "device" in str(e).lower():
                print(f"\n[WARNING] PlateDetector CUDA error: {e}. Falling back to CPU.")
                self.device = "cpu"
                self.model = self.model.to("cpu")
                img_tensor = self.preprocess(torch.from_numpy(crop_rgb).permute(2, 0, 1)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(img_tensor)
            else:
                raise e
            
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_idx = torch.topk(probs, 5)
        
        top5_indices = set(top5_idx.cpu().numpy())
        
        matches = self.DISHWARE_INDICES.intersection(top5_indices)
        
        if len(matches) > 0:
            return len(matches)
            
        return 0
