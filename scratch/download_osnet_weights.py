import urllib.request
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
target_dir = project_root / "embedding"
target_file = target_dir / "osnet_x1_0_msmt17.pth"

# Let's try downloading from mikel-brostrom's yolo_tracking releases
url = "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0/osnet_x1_0_msmt17.pt"

print(f"Downloading OSNet weights from {url}...")
print(f"Saving to {target_file}...")

target_dir.mkdir(parents=True, exist_ok=True)

try:
    # Use urllib to download the file (add a User-Agent to avoid blocking)
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response:
        with open(target_file, 'wb') as out_file:
            out_file.write(response.read())
    print("Download completed successfully!")
    print(f"File size: {target_file.stat().st_size} bytes")
except Exception as e:
    print(f"Failed to download weights: {e}")
