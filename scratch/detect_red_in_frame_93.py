import cv2
import numpy as np

img = cv2.imread("/Users/gaurisudharsinip/Desktop/wgtech/scratch/frame_93.jpg")
h, w, c = img.shape
print(f"Image size: {w}x{h}")

# Search for red pixels
red_mask = (img[:, :, 2] > 200) & (img[:, :, 1] < 50) & (img[:, :, 0] < 50)
y_indices, x_indices = np.where(red_mask)

if len(x_indices) > 0:
    x1, x2 = np.min(x_indices), np.max(x_indices)
    y1, y2 = np.min(y_indices), np.max(y_indices)
    print(f"Found red pixels! Bounding box of red pixels: [{x1}, {y1}, {x2}, {y2}]")
else:
    print("No red pixels found in frame 93.")
