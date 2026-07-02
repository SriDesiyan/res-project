import urllib.request
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
target_dir = project_root / "embedding"
target_file = target_dir / "osnet_x1_0_msmt17.pth"

# GDrive download URL
url = "https://drive.google.com/uc?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x&export=download"

print(f"Downloading OSNet weights from {url}...")
print(f"Saving to {target_file}...")

target_dir.mkdir(parents=True, exist_ok=True)

try:
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
