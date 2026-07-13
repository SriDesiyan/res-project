"""
Person Tracker — YOLO + BoT-SORT + Waiter Classification.

Wraps YOLO detection with BoT-SORT tracking and integrates the existing
ResNet50 waiter identification pipeline. Each tracked person carries full
state for table assignment and occupancy analytics.
"""
import time
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from collections import defaultdict
from .osnet import osnet_x1_0


class TrackedPerson:

    def __init__(self, track_id: int, bbox: tuple, centroid: tuple, frame_time: float):
        self.track_id = track_id
        self.bbox = bbox                  # (x1, y1, x2, y2)
        self.centroid = centroid           # (cx, cy)
        self.bottom_center = (centroid[0], bbox[3]) # (cx, y_max)
        self.velocity = 0.0               # px/frame
        self.role = "customer"            # "waiter" | "customer"
        self.assigned_table = None        # table_id or None
        self.first_seen = frame_time      # timestamp
        self.last_seen = frame_time
        self.frame_count = 1              # total frames visible
        self.confirmed = False            # True after MIN_VISIBILITY frames
        self.visual_embedding = None      # torch.Tensor of shape (1, 2048)
        self.session_id = None            # persistent session ID mapping

    def update(self, bbox: tuple, centroid: tuple, frame_time: float):
        self.bbox = bbox
        
        dx = centroid[0] - self.centroid[0]
        dy = centroid[1] - self.centroid[1]
        self.velocity = (dx**2 + dy**2) ** 0.5
        
        self.centroid = centroid
        self.bottom_center = (centroid[0], bbox[3])
        self.last_seen = frame_time
        self.frame_count += 1


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class PersonTracker:
    """
    Manages YOLO detection, BoT-SORT tracking, and waiter classification.

    Parameters:
        conf: YOLO detection confidence threshold
        similarity_threshold: cosine similarity threshold for waiter classification
        min_visibility: frames before a person is "confirmed"
        disappear_timeout: frames before removing a missing track
    """

    MIN_VISIBILITY = 3
    DISAPPEAR_TIMEOUT = 20
    CONFIRM_FRAMES = 2
    # Waiter classification tuning
    WAITER_LOCK_THRESHOLD = 6    # hits needed to lock (was 12)
    WAITER_HIT_INCREMENT = 3     # score per uniform-match frame (was 1)
    WAITER_UNLOCK_STREAK = 8     # consecutive non-match frames before unlock

    def __init__(self, device, conf=0.25, similarity_threshold=0.80):
        self.device = device
        self.conf = conf
        self.similarity_threshold = similarity_threshold

        project_root = Path(__file__).parent.parent.parent.resolve()
        self.yolo = YOLO(str(project_root / "yolov8n.pt"))
        self.yolo.to(device)

        embedding_dir = project_root / "embedding"
        self.tracker_cfg = str(embedding_dir / "tracker_config.yaml")
        if not Path(self.tracker_cfg).exists():
            self.tracker_cfg = "botsort.yaml"  

        self.extractor = FeatureExtractor().to(device)
        self.extractor.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # OSNet Re-ID Extractor
        self.reid_extractor = osnet_x1_0(num_classes=1000, pretrained=False).to(device)
        reid_weight_path = embedding_dir / "osnet_x1_0_msmt17.pth"
        if reid_weight_path.exists():
            state_dict = torch.load(reid_weight_path, map_location=device)
            model_dict = self.reid_extractor.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
            model_dict.update(new_state_dict)
            self.reid_extractor.load_state_dict(model_dict)
            print(f"[PersonTracker] Loaded OSNet Re-ID weights from {reid_weight_path}")
        else:
            print(f"[WARNING] OSNet weights not found at {reid_weight_path}. Running with random initialization.")
        self.reid_extractor.eval()

        self.reid_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        waiter_emb_path = embedding_dir / "waiter_average_embedding.pt"
        server_emb_path = embedding_dir / "server_average_embedding.pt"

        self.waiter_emb = None
        self.server_emb = None

        if waiter_emb_path.exists():
            self.waiter_emb = torch.load(waiter_emb_path, map_location=device)
            self.waiter_emb = torch.nn.functional.normalize(self.waiter_emb, p=2, dim=1)
        if server_emb_path.exists():
            self.server_emb = torch.load(server_emb_path, map_location=device)
            self.server_emb = torch.nn.functional.normalize(self.server_emb, p=2, dim=1)

        self.tracks = {}                       # track_id -> TrackedPerson
        self.locked_waiters = set()            # permanently locked waiter IDs
        self.waiter_hits = defaultdict(int)    # track_id -> cumulative waiter hits
        self.waiter_non_match_streak = defaultdict(int)  # track_id -> consecutive non-match frames
        self.yolo_latencies = []
        self.tracking_latencies = []

    def _extract_top_40(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((0, 0, w, int(h * 0.4)))

    def _has_waiter_uniform(self, crop) -> bool:
        if crop is None or crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_crop, w_crop = hsv.shape[:2]
        
        # Divide into upper (shirt) and lower (pants)
        upper_body = hsv[int(h_crop * 0.15):int(h_crop * 0.45), :]
        lower_body = hsv[int(h_crop * 0.55):int(h_crop * 0.85), :]
        
        if upper_body.size == 0 or lower_body.size == 0:
            return False
            
        mean_upper = cv2.mean(upper_body)
        mean_lower = cv2.mean(lower_body)
        
        upper_s = mean_upper[1]
        upper_v = mean_upper[2]
        lower_v = mean_lower[2]
        
        # Upper shirt: Brightness/Value > 175, low Saturation < 65
        # Lower pants: Brightness/Value < 85
        return (upper_v > 175) and (upper_s < 65) and (lower_v < 85)

    def _has_refined_waiter_uniform(self, crop) -> bool:
        if crop is None or crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_crop, w_crop = hsv.shape[:2]
        
        # Sliced width to exclude background (central 50%)
        x_slice_start = int(w_crop * 0.25)
        x_slice_end = int(w_crop * 0.75)
        
        upper_body = hsv[int(h_crop * 0.15):int(h_crop * 0.45), x_slice_start:x_slice_end]
        lower_body = hsv[int(h_crop * 0.55):int(h_crop * 0.85), x_slice_start:x_slice_end]
        
        if upper_body.size == 0 or lower_body.size == 0:
            return False
            
        # Check pixel percentage for white shirt (exact user thresholds):
        white_pixels = (upper_body[:, :, 2] > 175) & (upper_body[:, :, 1] < 65)
        white_ratio = np.mean(white_pixels)
        
        # Check pixel percentage for black pants (exact user threshold):
        black_pixels = (lower_body[:, :, 2] < 85)
        black_ratio = np.mean(black_pixels)
        
        # Require at least 20% of upper body to be white shirt and 40% of lower body to be black pants
        return (white_ratio >= 0.20) and (black_ratio >= 0.40)

    def _get_embedding_and_classify(self, frame, x1, y1, x2, y2) -> tuple:
        """
        Run ResNet50 similarity against waiter/server embeddings.
        Returns (emb, max_similarity, is_waiter_match)
        """
        person_crop = frame[y1:y2, x1:x2]
        is_uniform_match = self._has_waiter_uniform(person_crop)
        is_refined_match = self._has_refined_waiter_uniform(person_crop)

        if self.waiter_emb is None and self.server_emb is None:
            return (None, 0.0, is_uniform_match or is_refined_match)

        img_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        top_img = self._extract_top_40(img_pil)
        tensor = self.transform(top_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.extractor(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            sims = []
            if self.waiter_emb is not None:
                sims.append(torch.mm(emb, self.waiter_emb.t()).item())
            if self.server_emb is not None:
                sims.append(torch.mm(emb, self.server_emb.t()).item())

            max_sim = max(sims) if sims else 0.0

        is_match = (max_sim > self.similarity_threshold) or is_uniform_match or is_refined_match
        return (emb.cpu(), max_sim, is_match)

    def process_frame(self, frame, frame_time: float) -> list[TrackedPerson]:
        try:
            return self._process_frame_impl(frame, frame_time)
        except RuntimeError as e:
            if "cuda" in str(e).lower() or "device" in str(e).lower():
                print(f"\n[WARNING] PersonTracker CUDA error: {e}. Falling back to CPU.")
                self.device = torch.device("cpu")
                self.extractor = self.extractor.to(self.device)
                self.reid_extractor = self.reid_extractor.to(self.device)
                self.yolo.to("cpu")
                return self._process_frame_impl(frame, frame_time)
            else:
                raise e

    def _process_frame_impl(self, frame, frame_time: float) -> list[TrackedPerson]:
        """
        Process a single frame:
        1. YOLO detect persons
        2. BoT-SORT track
        3. Classify waiter/customer
        4. Update track state

        Returns list of active TrackedPerson objects.
        """
        h, w = frame.shape[:2]

        # Layer 1: YOLO + BoT-SORT
        t_yolo_start = time.time()
        results = self.yolo.track(
            frame, classes=[0], conf=self.conf,
            persist=True, tracker=self.tracker_cfg, verbose=False,
            device=self.device
        )
        t_yolo_end = time.time()
        
        yolo_inf = 0.0
        if results and len(results) > 0 and hasattr(results[0], 'speed'):
            yolo_inf = results[0].speed.get('inference', 0.0) # in ms
            
        self.yolo_latencies.append(yolo_inf / 1000.0) # convert to seconds
        self.tracking_latencies.append(max(0.0, (t_yolo_end - t_yolo_start) - (yolo_inf / 1000.0)))

        active_ids = set()
        active_persons = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, conf, track_id in zip(boxes, confs, track_ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                active_ids.add(track_id)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                centroid = (cx, cy)
                bbox = (x1, y1, x2, y2)

                if track_id in self.tracks:
                    self.tracks[track_id].update(bbox, centroid, frame_time)
                else:
                    self.tracks[track_id] = TrackedPerson(track_id, bbox, centroid, frame_time)

                tp = self.tracks[track_id]
                tp.yolo_conf = float(conf)

                if tp.frame_count >= self.MIN_VISIBILITY:
                    tp.confirmed = True

                # ── OSNet Re-ID embedding (gallery building) ──────────────────────
                # Run for first 10 frames, then every 30 frames.
                if tp.visual_embedding is None or tp.frame_count < 10 or tp.frame_count % 30 == 0:
                    person_crop = frame[y1:y2, x1:x2]
                    img_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                    reid_tensor = self.reid_transform(img_pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        reid_emb = self.reid_extractor(reid_tensor)
                        reid_emb = torch.nn.functional.normalize(reid_emb, p=2, dim=1)
                    tp.visual_embedding = reid_emb.cpu()

                # ── Waiter classification ─────────────────────────────────────────
                # Skip expensive ResNet50 re-check for tracks already locked as waiter.
                # This saves CPU and prevents flicker on confirmed waiters.
                if track_id in self.locked_waiters:
                    tp.role = "waiter"
                    # Periodic check (every 60 frames) to allow unlock on sustained mismatch
                    if tp.frame_count % 60 == 0:
                        _, _, is_match = self._get_embedding_and_classify(frame, x1, y1, x2, y2)
                        if not is_match:
                            self.waiter_non_match_streak[track_id] += 1
                            if self.waiter_non_match_streak[track_id] >= self.WAITER_UNLOCK_STREAK:
                                self.locked_waiters.discard(track_id)
                                tp.role = "customer"
                                self.waiter_non_match_streak[track_id] = 0
                                print(f"[Tracker] Unlocked waiter track {track_id} after streak")
                        else:
                            self.waiter_non_match_streak[track_id] = 0
                else:
                    # Not yet locked — run classifier (first 30 frames every frame,
                    # then every 15 frames for speed).
                    last_is_match = getattr(tp, "last_is_match", None)
                    if last_is_match is None or tp.frame_count < 30 or tp.frame_count % 15 == 0:
                        emb, max_sim, is_match = self._get_embedding_and_classify(
                            frame, x1, y1, x2, y2
                        )
                        tp.last_is_match = is_match
                    else:
                        is_match = last_is_match

                    if is_match:
                        # Asymmetric: fast accumulation (+3), slow decay (-1)
                        self.waiter_hits[track_id] = min(
                            30, self.waiter_hits[track_id] + self.WAITER_HIT_INCREMENT
                        )
                        self.waiter_non_match_streak[track_id] = 0
                    else:
                        self.waiter_hits[track_id] = max(
                            0, self.waiter_hits[track_id] - 1
                        )

                    if self.waiter_hits[track_id] >= self.WAITER_LOCK_THRESHOLD:
                        self.locked_waiters.add(track_id)
                        tp.role = "waiter"
                        print(f"[Tracker] Locked track {track_id} as waiter")
                    else:
                        tp.role = "customer"

                active_persons.append(tp)

        disappeared = set(self.tracks.keys()) - active_ids
        to_delete = []
        for tid in disappeared:
            frames_missing = (frame_time - self.tracks[tid].last_seen) * 25  
            if frames_missing > self.DISAPPEAR_TIMEOUT:
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]
            self.waiter_hits.pop(tid, None)
            # Don't remove from locked_waiters — keeps the lock for re-ID for 15 minutes 

        return active_persons
