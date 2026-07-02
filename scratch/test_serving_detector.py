import sys
from pathlib import Path
import numpy as np
# import pytest
import sqlite3

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from tracking.serving_detector import detect_waiter_serving
from database.serving_event_logger import ServingEventLogger

class MockYolo:
    def __init__(self, detections):
        self.detections = detections
        self.names = {39: 'bottle', 41: 'cup', 45: 'bowl'}

    def __call__(self, frame, classes=None, conf=0.15, verbose=False):
        class MockBox:
            def __init__(self, bbox, cls_id, conf_val):
                self.xyxy = np.array([bbox])
                self.cls = np.array([cls_id])
                self.conf = np.array([conf_val])

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        boxes = []
        for det in self.detections:
            if classes is None or det['class_id'] in classes:
                boxes.append(MockBox(det['bbox'], det['class_id'], det['confidence']))
        
        return [MockResult(MockBoxList(boxes))]

class MockBoxList:
    def __init__(self, boxes):
        self.boxes = boxes
    @property
    def xyxy(self):
        return MockTensor([b.xyxy[0] for b in self.boxes])
    @property
    def cls(self):
        return MockTensor([b.cls[0] for b in self.boxes])
    @property
    def conf(self):
        return MockTensor([b.conf[0] for b in self.boxes])

class MockTensor:
    def __init__(self, data):
        self.data = np.array(data)
    def cpu(self):
        return self
    def numpy(self):
        return self.data
    def int(self):
        return self

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class MockHandsResult:
    def __init__(self, landmarks_list):
        self.hand_landmarks = landmarks_list

class MockHandsDetector:
    def __init__(self, landmarks_list):
        self.landmarks_list = landmarks_list
    def detect(self, mp_image):
        return MockHandsResult(self.landmarks_list)

class MockPoseResult:
    def __init__(self, landmarks_list):
        self.pose_landmarks = landmarks_list

class MockPoseDetector:
    def __init__(self, landmarks_list):
        self.landmarks_list = landmarks_list
    def detect(self, mp_image):
        return MockPoseResult(self.landmarks_list)

def test_serving_detector():
    # Set up mock objects
    waiter_bbox = [100, 100, 200, 300]
    
    # 1. Mock YOLO detections: a cup near the waiter
    detections = [
        {'bbox': [110, 120, 140, 150], 'class_id': 41, 'confidence': 0.85}
    ]
    yolo_model = MockYolo(detections)
    
    # 2. Mock MediaPipe Hand landmarks inside waiter_bbox, near the food item
    # Landmark indices check wrist (0), finger bases
    hand_landmarks = [MockLandmark(0.12, 0.13) for _ in range(21)]  # normalised x=0.12, y=0.13
    # Frame shape is 1000x1000, so hand is at (120, 130), near food center at (125, 135)
    hands_detector = MockHandsDetector([hand_landmarks])
    
    # 3. Mock MediaPipe Pose landmarks for waiter serving pose
    # Left shoulder (11) at (150, 150), Left wrist (15) at (150, 100) (wrist above shoulder)
    pose_landmarks = [MockLandmark(0, 0) for _ in range(33)]
    pose_landmarks[11] = MockLandmark(0.15, 0.15)  # left shoulder (150, 150)
    pose_landmarks[12] = MockLandmark(0.18, 0.15)  # right shoulder (180, 150)
    pose_landmarks[13] = MockLandmark(0.15, 0.20)  # left elbow (150, 200)
    pose_landmarks[14] = MockLandmark(0.18, 0.20)  # right elbow (180, 200)
    pose_landmarks[15] = MockLandmark(0.15, 0.10)  # left wrist (150, 100) -> above shoulder
    pose_landmarks[16] = MockLandmark(0.40, 0.40)  # right wrist far away
    pose_detector = MockPoseDetector([pose_landmarks])
    
    # Create fake 1000x1000 image frame
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    result = detect_waiter_serving(
        frame=frame,
        waiter_bbox=waiter_bbox,
        yolo_model=yolo_model,
        hands_detector=hands_detector,
        pose_detector=pose_detector
    )
    
    print("\nHybrid Serving Detection Result:")
    print("Is serving:", result['is_serving'])
    print("Confidence:", result['confidence'])
    print("Food type:", result['food_type'])
    print("Method breakdown:", result['methods'])
    
    assert result['is_serving'] is True
    assert result['confidence'] >= 0.80
    assert result['food_type'] == 'cup'
    
    # Verify Serving Logger
    logger = ServingEventLogger("test_serving_logs.db")
    logger.log_serving("S10", "table_1", result['food_type'], frame_num=42, confidence=result['confidence'])
    
    # Verify SQLite entry
    conn = sqlite3.connect("test_serving_logs.db")
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM serving_events ORDER BY id DESC LIMIT 1").fetchone()
    print("Logged Row in DB:", row)
    assert row is not None
    assert row[1] == "S10"
    assert row[2] == "table_1"
    assert row[3] == "cup"
    assert row[5] == 42
    assert abs(row[6] - result['confidence']) < 1e-5
    conn.close()
    
    import os
    os.remove("test_serving_logs.db")
    print("Unit test passed successfully!")

if __name__ == "__main__":
    test_serving_detector()
