import sys
import numpy as np
import cv2
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from tracking.serving_detector import detect_waiter_serving

class MockYolo:
    def __init__(self):
        self.names = {41: 'cup'}
    def __call__(self, frame, classes=None, conf=0.15, verbose=False):
        class MockBoxList:
            @property
            def xyxy(self):
                return MockTensor([])
            @property
            def cls(self):
                return MockTensor([])
            @property
            def conf(self):
                return MockTensor([])
        class MockResult:
            def __init__(self):
                self.boxes = MockBoxList()
        class MockTensor:
            def __init__(self, data):
                self.data = np.array(data)
            def cpu(self):
                return self
            def numpy(self):
                return self.data
            def int(self):
                return self
        return [MockResult()]

class MockHandsResult:
    def __init__(self, landmarks_list):
        self.hand_landmarks = landmarks_list

class MockHandsDetector:
    def __init__(self, landmarks_list):
        self.landmarks = landmarks_list
    def detect(self, mp_image):
        return MockHandsResult(self.landmarks)

class MockPoseResult:
    def __init__(self, landmarks_list):
        self.pose_landmarks = landmarks_list

class MockPoseDetector:
    def __init__(self, landmarks_list):
        self.landmarks = landmarks_list
    def detect(self, mp_image):
        return MockPoseResult(self.landmarks)

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def test_black_plate_detection():
    # 1. Create a frame that is mostly black in the region around the wrist crop to simulate a black plate in hand
    # Frame size: 200x200
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Let's say the waiter is at bbox [10, 10, 190, 190]
    waiter_bbox = [10, 10, 190, 190]
    
    # MediaPipe pose landmarks inside crop (180x180)
    # Crop is from y=0 to y=200, x=0 to x=200 (since pad is 20, actually full image is processed)
    # Wrist landmark indices: left_wrist (15)
    # normalized coordinates in crop: left_wrist is at (0.5, 0.5), which translates to wx = 100, wy = 100
    pose_landmarks = [MockLandmark(0, 0) for _ in range(33)]
    pose_landmarks[11] = MockLandmark(0.3, 0.3)  # left shoulder
    pose_landmarks[12] = MockLandmark(0.7, 0.3)  # right shoulder
    pose_landmarks[13] = MockLandmark(0.3, 0.5)  # left elbow
    pose_landmarks[14] = MockLandmark(0.7, 0.5)  # right elbow
    pose_landmarks[15] = MockLandmark(0.5, 0.5)  # left wrist
    pose_landmarks[16] = MockLandmark(0.9, 0.9)  # right wrist far away
    
    pose_detector = MockPoseDetector([pose_landmarks])
    hands_detector = MockHandsDetector([]) # not used if pose check is done
    yolo_model = MockYolo()
    
    # Crop area around wrist is max(0, wx-r) to min(crop_w, wx+r)
    # wx = 100, wy = 100. Crop radius = 45. Crop is [55:145, 55:145].
    # Let's paint this area black (Value = 10, Saturation = 10) in BGR so that HSV has Value < 50
    # In BGR: (10, 10, 10) gives HSV: Hue=0, Sat=0, Val=10.
    frame[50:150, 50:150] = (10, 10, 10)
    
    res = detect_waiter_serving(
        frame=frame,
        waiter_bbox=waiter_bbox,
        yolo_model=yolo_model,
        hands_detector=hands_detector,
        pose_detector=pose_detector
    )
    
    print("\nBlack Plate Detection Result:")
    print("Is serving:", res['is_serving'])
    print("Is black plate:", res['is_black_plate'])
    print("Confidence:", res['confidence'])
    print("Food type:", res['food_type'])
    print("Method breakdown:", res['methods'])
    
    assert res['is_serving'] is True
    assert res['is_black_plate'] is True
    assert res['food_type'] == 'black plate'
    assert res['methods']['black_plate_hand'] is True
    
    print("\nBlack plate detection tests passed successfully!")

if __name__ == "__main__":
    test_black_plate_detection()
