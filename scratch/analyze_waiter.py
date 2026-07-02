import cv2
import numpy as np
import json

# Load the image
img_path = "/Users/gaurisudharsinip/.gemini/antigravity-ide/brain/57927b26-be82-40df-ac0c-91a54bc6ac32/media__1781276110321.jpg"
img = cv2.imread(img_path)
h, w, c = img.shape
print(f"Image size: {w}x{h}")

# The waiter bounding box is red. In BGR, red is (0, 0, 255).
# Let's search for red pixels.
# Since BGR values might be slightly compressed or varied, let's look for pixels where R is very high and G, B are low.
# For example, R > 200, G < 50, B < 50.
red_mask = (img[:, :, 2] > 200) & (img[:, :, 1] < 50) & (img[:, :, 0] < 50)
y_indices, x_indices = np.where(red_mask)

# Find connected components or bounding boxes of these red pixels
# Let's just find the bounding box of the red pixels on the right side of the image (x > w/2)
right_side_mask = x_indices > (w / 2)
x_right = x_indices[right_side_mask]
y_right = y_indices[right_side_mask]

if len(x_right) > 0:
    x1, x2 = np.min(x_right), np.max(x_right)
    y1, y2 = np.min(y_right), np.max(y_right)
    print(f"Detected red bounding box on right side: [{x1}, {y1}, {x2}, {y2}]")
    anchor = ((x1 + x2) / 2.0, float(y2))
    print(f"Waiter anchor (bottom-center): {anchor}")
    
    # Let's load the table_1 polygon
    tables_path = "/Users/gaurisudharsinip/Desktop/wgtech/analytics/config/tables.json"
    with open(tables_path) as f:
        tables = json.load(f)["tables"]
    
    for tid, t_info in tables.items():
        poly = np.array(t_info["polygon"], dtype=np.int32)
        dist = cv2.pointPolygonTest(poly, anchor, measureDist=True)
        print(f"Distance to {tid}: {dist:.2f} px")
else:
    print("No red pixels found on the right side.")
