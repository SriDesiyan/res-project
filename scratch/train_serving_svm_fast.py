import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from sklearn.svm import SVC

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from analytics.tracking.person_tracker import PersonTracker
from analytics.tracking.serving_detector import detect_food_in_frame, detect_waiter_serving
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from analytics.tracking.serving_detector import HAND_MODEL_PATH, POSE_MODEL_PATH

def calculate_angle_3pt(a_x, a_y, b_x, b_y, c_x, c_y):
    ba = np.array([a_x - b_x, a_y - b_y])
    bc = np.array([c_x - b_x, c_y - b_y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def generate_synthetic_samples():
    synth_features = []
    synth_labels = []
    
    # 1. Class 0: Idle poses (arms straight down)
    # [le_x, le_y, re_x, re_y, lw_x, lw_y, rw_x, rw_y]
    # Normal coordinates relative to shoulder center (distance normalized by shoulder width):
    # Left shoulder is around -0.5, 0.0. Right shoulder is around 0.5, 0.0.
    # Idle Left elbow: x=-0.5, y=0.7. Left wrist: x=-0.5, y=1.3.
    # Idle Right elbow: x=0.5, y=0.7. Right wrist: x=0.5, y=1.3.
    for _ in range(50):
        le_x = -0.5 + np.random.normal(0, 0.05)
        le_y = 0.7 + np.random.normal(0, 0.05)
        re_x = 0.5 + np.random.normal(0, 0.05)
        re_y = 0.7 + np.random.normal(0, 0.05)
        lw_x = -0.5 + np.random.normal(0, 0.05)
        lw_y = 1.3 + np.random.normal(0, 0.05)
        rw_x = 0.5 + np.random.normal(0, 0.05)
        rw_y = 1.3 + np.random.normal(0, 0.05)
        
        synth_features.append([le_x, le_y, re_x, re_y, lw_x, lw_y, rw_x, rw_y])
        synth_labels.append(0)
        
    # 2. Class 0: Notepad writing / order taking poses (wrists close together, elbows tucked)
    # Left elbow: x=-0.4, y=0.6. Left wrist: x=-0.1, y=0.4.
    # Right elbow: x=0.4, y=0.6. Right wrist: x=0.1, y=0.4.
    for _ in range(50):
        le_x = -0.4 + np.random.normal(0, 0.04)
        le_y = 0.6 + np.random.normal(0, 0.04)
        re_x = 0.4 + np.random.normal(0, 0.04)
        re_y = 0.6 + np.random.normal(0, 0.04)
        lw_x = -0.1 + np.random.normal(0, 0.03)
        lw_y = 0.4 + np.random.normal(0, 0.03)
        rw_x = 0.1 + np.random.normal(0, 0.03)
        rw_y = 0.4 + np.random.normal(0, 0.03)
        
        synth_features.append([le_x, le_y, re_x, re_y, lw_x, lw_y, rw_x, rw_y])
        synth_labels.append(0)
        
    # 3. Class 1: Serving / carrying poses (arms bent and extended / flared)
    # Left elbow: x=-0.7, y=0.5. Left wrist: x=-0.8, y=0.2. (raised)
    # Right elbow: x=0.7, y=0.5. Right wrist: x=0.8, y=0.2.
    for _ in range(50):
        le_x = -0.7 + np.random.normal(0, 0.05)
        le_y = 0.5 + np.random.normal(0, 0.05)
        re_x = 0.7 + np.random.normal(0, 0.05)
        re_y = 0.5 + np.random.normal(0, 0.05)
        lw_x = -0.8 + np.random.normal(0, 0.05)
        lw_y = 0.2 + np.random.normal(0, 0.05)
        rw_x = 0.8 + np.random.normal(0, 0.05)
        rw_y = 0.2 + np.random.normal(0, 0.05)
        
        synth_features.append([le_x, le_y, re_x, re_y, lw_x, lw_y, rw_x, rw_y])
        synth_labels.append(1)
        
    return synth_features, synth_labels

def main():
    video_path = project_root / "new.mp4"
    if not video_path.exists():
        print(f"Video {video_path} not found!")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    # Initialize MediaPipe models
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2
    )
    mp_hands = vision.HandLandmarker.create_from_options(hand_options)

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False
    )
    mp_pose = vision.PoseLandmarker.create_from_options(pose_options)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracker = PersonTracker(device, conf=0.20)
    
    # Frame ranges to scan where waiter is active
    ranges = [(1240, 1500), (7900, 8350)]
    
    features = []
    labels = []
    
    print("Collecting waiter pose features (processing every 5th frame)...")
    
    total_frames_to_process = sum((end_f - start_f) // 5 for start_f, end_f in ranges)
    processed_count = 0
    
    for start_f, end_f in ranges:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for frame_num in range(start_f, end_f):
            if (frame_num - start_f) % 5 != 0:
                continue
                
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_time = frame_num / fps
            persons = tracker.process_frame(frame, frame_time)
            
            # Find waiter
            waiters = [p for p in persons if p.role == "waiter"]
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Progress: {processed_count}/{total_frames_to_process} frames processed. Samples collected so far: {len(features)}")
                sys.stdout.flush()
                
            if not waiters:
                continue
                
            food_detections = detect_food_in_frame(frame, tracker.yolo)
            
            for w in waiters:
                waiter_x1, waiter_y1, waiter_x2, waiter_y2 = w.bbox
                h_frame, w_frame = frame.shape[:2]
                pad = 20
                px1, py1 = max(0, waiter_x1 - pad), max(0, waiter_y1 - pad)
                px2, py2 = min(w_frame, waiter_x2 + pad), min(h_frame, waiter_y2 + pad)
                
                waiter_crop = frame[py1:py2, px1:px2]
                crop_h, crop_w = waiter_crop.shape[:2]
                
                if crop_h > 20 and crop_w > 20:
                    rgb_crop = cv2.cvtColor(waiter_crop, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                    pose_results = mp_pose.detect(mp_image)
                    
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks[0]
                        ls = landmarks[11]
                        rs = landmarks[12]
                        le = landmarks[13]
                        re = landmarks[14]
                        lw = landmarks[15]
                        rw = landmarks[16]
                        
                        # Calculate features
                        cx = (ls.x + rs.x) / 2.0
                        cy = (ls.y + rs.y) / 2.0
                        dist = np.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2) + 1e-6
                        
                        le_x_norm = (le.x - cx) / dist
                        le_y_norm = (le.y - cy) / dist
                        re_x_norm = (re.x - cx) / dist
                        re_y_norm = (re.y - cy) / dist
                        lw_x_norm = (lw.x - cx) / dist
                        lw_y_norm = (lw.y - cy) / dist
                        rw_x_norm = (rw.x - cx) / dist
                        rw_y_norm = (rw.y - cy) / dist
                        
                        feat = [le_x_norm, le_y_norm, re_x_norm, re_y_norm, lw_x_norm, lw_y_norm, rw_x_norm, rw_y_norm]
                        
                        # Heuristics for auto-labeling
                        # Angle calculations
                        left_angle = calculate_angle_3pt(ls.x, ls.y, le.x, le.y, lw.x, lw.y)
                        right_angle = calculate_angle_3pt(rs.x, rs.y, re.x, re.y, rw.x, rw.y)
                        
                        wrist_dist = np.sqrt((lw.x - rw.x)**2 + (lw.y - rw.y)**2)
                        
                        # Check if serving
                        res = detect_waiter_serving(frame, w.bbox, tracker.yolo, mp_hands, mp_pose, food_detections)
                        
                        is_serving_heuristic = res['is_serving']
                        is_order_taking_heuristic = res['is_order_taking']
                        
                        is_positive = False
                        is_negative = False
                        
                        if is_serving_heuristic:
                            is_positive = True
                        elif (60 < left_angle < 115 and lw.y < le.y) or (60 < right_angle < 115 and rw.y < re.y):
                            if not (wrist_dist < 0.18 and left_angle < 75 and right_angle < 75):
                                is_positive = True
                                
                        if is_order_taking_heuristic:
                            is_negative = True
                        elif wrist_dist < 0.18 and left_angle < 75 and right_angle < 75:
                            is_negative = True
                        elif (lw.y > le.y + 0.10 and rw.y > re.y + 0.10) or (left_angle > 140 and right_angle > 140):
                            is_negative = True
                            
                        if is_positive and not is_negative:
                            features.append(feat)
                            labels.append(1)
                        elif is_negative and not is_positive:
                            features.append(feat)
                            labels.append(0)

    cap.release()
    
    # Generate synthetic samples
    synth_feats, synth_labs = generate_synthetic_samples()
    
    # Combine
    all_features = list(features) + synth_feats
    all_labels = list(labels) + synth_labs
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\nFinal dataset size: {len(X)} samples.")
    print(f"  Positives (Serving): {np.sum(y == 1)} (real: {len(features)})")
    print(f"  Negatives (Not Serving): {np.sum(y == 0)}")
    
    # Train linear SVM
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X, y)
    
    train_acc = clf.score(X, y)
    print(f"SVM Training Accuracy: {train_acc * 100.0:.2f}%")
    
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]
    print("\nDecision boundary coefficients (coef_):")
    print(list(coef))
    print("\nIntercept (intercept_):")
    print(intercept)
    
    # Print Decision Boundary Formula code snippet
    print("\nCopy-pasteable Python evaluation snippet:")
    print("def eval_serving_svm(feats):")
    print(f"    coef = {list(coef)}")
    print(f"    intercept = {intercept}")
    print("    score = sum(f * c for f, c in zip(feats, coef)) + intercept")
    print("    return score > 0")

if __name__ == "__main__":
    main()
