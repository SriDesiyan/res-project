import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
from pathlib import Path

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
    """Crop the top 40% of the image to focus on the chest/tie area."""
    w, h = img.size
    return img.crop((0, 0, w, int(h * 0.4)))

def compute_average_embedding(gallery_dir: str, save_path: str = "waiter_average_embedding.pt"):
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading ResNet50 Feature Extractor...")
    model = FeatureExtractor().to(device)
    model.eval()
    transform = get_transforms()
    
    gallery_path = Path(gallery_dir)
    print(f"Extracting features from {gallery_path}...")
    gallery_embeddings = []
    
    if not gallery_path.exists():
        print(f"[ERROR] Gallery directory {gallery_path} does not exist.")
        return None

    valid_exts = {".jpg", ".jpeg", ".png"}
    for img_path in gallery_path.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in valid_exts and "aug" not in img_path.stem.lower():
            try:
                img = Image.open(img_path).convert("RGB")
                img = extract_top_40(img) # Focus on tie area
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor)
                    # L2 normalize
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    gallery_embeddings.append(emb)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    if not gallery_embeddings:
        print("[ERROR] No valid gallery images found.")
        return None
        
    print(f"Loaded {len(gallery_embeddings)} reference waiter images. Computing average embedding...")
    stacked = torch.cat(gallery_embeddings, dim=0)
    avg_emb = torch.mean(stacked, dim=0, keepdim=True)
    avg_emb = torch.nn.functional.normalize(avg_emb, p=2, dim=1)
    
    # Save the embedding for later use
    torch.save(avg_emb, save_path)
    print(f"Average embedding successfully saved to {save_path}!")
    print(f"Shape of embedding: {avg_emb.shape}")
    
    return avg_emb

if __name__ == "__main__":
    compute_average_embedding(gallery_dir="labelled/waiter", save_path="waiter_average_embedding.pt")
