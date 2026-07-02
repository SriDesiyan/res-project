import shutil
import os
from pathlib import Path

project_root = Path("/Users/gaurisudharsinip/Desktop/wgtech")
waiter_dir = project_root / "labelled" / "waiter"
sample_dir = project_root / "waiter-sample"

# 1. Clean labelled/waiter
print("Cleaning labelled/waiter...")
if waiter_dir.exists():
    for f in waiter_dir.iterdir():
        if f.is_file():
            f.unlink()
else:
    waiter_dir.mkdir(parents=True, exist_ok=True)

# 2. Copy images from waiter-sample to labelled/waiter
print(f"Copying files from {sample_dir} to {waiter_dir}...")
for f in sample_dir.iterdir():
    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        shutil.copy2(f, waiter_dir / f.name)
        print(f"  Copied {f.name}")

print("Dataset cleaning complete.")
