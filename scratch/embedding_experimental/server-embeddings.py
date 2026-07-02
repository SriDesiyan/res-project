import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
import numpy as np

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the classification head (fc layer)
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # Flatten

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_top_40(img: Image.Image) -> Image.Image:
    """Crop the top 40% of the image to focus on the chest/tie/vest area."""
    w, h = img.size
    return img.crop((0, 0, w, int(h * 0.4)))

"""def check_maroon_white_condition(img: Image.Image) -> bool:
    
    Check if the image contains a significant amount of maroon and white colors.
    Specifically looking for a maroon vest over a white shirt.
    
    # Convert PIL Image to OpenCV HSV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    # Maroon (Dark Red)
    lower_maroon1 = np.array([0, 50, 20])
    upper_maroon1 = np.array([10, 255, 150])
    lower_maroon2 = np.array([160, 50, 20])
    upper_maroon2 = np.array([180, 255, 150])
    
    # White
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    
    # Create masks
    mask_maroon1 = cv2.inRange(hsv, lower_maroon1, upper_maroon1)
    mask_maroon2 = cv2.inRange(hsv, lower_maroon2, upper_maroon2)
    mask_maroon = cv2.bitwise_or(mask_maroon1, mask_maroon2)
    
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Calculate percentages
    total_pixels = hsv.shape[0] * hsv.shape[1]
    maroon_pixels = cv2.countNonZero(mask_maroon)
    white_pixels = cv2.countNonZero(mask_white)
    
    maroon_ratio = maroon_pixels / total_pixels
    white_ratio = white_pixels / total_pixels
    
    # We want to see some maroon and some white (e.g. >2% of the cropped area for each)
    if maroon_ratio > 0.02 and white_ratio > 0.02:
        return True
    return False
"""
def compute_server_embedding(gallery_dir: str, save_path: str = "server_average_embedding.pt"):
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading ResNet50 Feature Extractor...")
    model = FeatureExtractor().to(device)
    model.eval()
    transform = get_transforms()
    
    gallery_path = Path(gallery_dir)
    print(f"Extracting features from {gallery_path}...")
    gallery_embeddings = []
    
    # Note: Using "server" directory if it exists, otherwise checking the provided directory
    if not gallery_path.exists():
        print(f"[ERROR] Gallery directory {gallery_path} does not exist.")
        return None

    valid_exts = {".jpg", ".jpeg", ".png"}
    for img_path in gallery_path.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in valid_exts and "aug" not in img_path.stem.lower():
            try:
                img = Image.open(img_path).convert("RGB")
                top_img = extract_top_40(img) # Focus on vest/shirt area
                
                # Check for the maroon + white condition on the top 40% of the image
                """if not check_maroon_white_condition(top_img):
                    print(f"Skipping {img_path.name}: Did not pass maroon+white color condition.")
                    continue"""
                    
                tensor = transform(top_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor)
                    # L2 normalize
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    gallery_embeddings.append(emb)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    if not gallery_embeddings:
        print("[ERROR] No valid server images found that match the maroon+white condition.")
        return None
        
    print(f"Loaded {len(gallery_embeddings)} reference server images matching condition. Computing average embedding...")
    stacked = torch.cat(gallery_embeddings, dim=0)
    avg_emb = torch.mean(stacked, dim=0, keepdim=True)
    avg_emb = torch.nn.functional.normalize(avg_emb, p=2, dim=1)
    
    # Save the embedding for later use
    torch.save(avg_emb, save_path)
    print(f"Average server embedding successfully saved to {save_path}!")
    print(f"Shape of embedding: {avg_emb.shape}")
    
    return avg_emb

if __name__ == "__main__":
    # You can change the gallery directory to labelled/server if you have a separate folder for servers.
    # Otherwise, it filters the labelled/waiter folder for people matching the server uniform condition.
    project_root = Path(__file__).resolve().parent.parent
    compute_server_embedding(
        gallery_dir=str(project_root / "labelled" / "server"), 
        save_path=str(project_root / "embedding" / "server_average_embedding.pt")
    )
