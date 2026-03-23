import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
from .config import (
    LOCK_SIZE_THRESHOLD,
    ANOMALY_BUFFER_SIZE,
    LOCK_DURATION_FRAMES,
    ATTENTION_THRESHOLD,
    ATTENTION_LOCK_MULTIPLIER,
)

class ObjectAnalyzer:
    def __init__(self):
        # Stores history of object features (area, aspect ratio, motion, confidence)
        self.feature_history = deque(maxlen=2000)

        # ML Model to detect anomalies (online)
        self.clf = IsolationForest(contamination=0.05, random_state=42)
        self.is_fitted = False

        # Track active locks: {track_id: frames_remaining}
        self.lock_timers = {}

        # Keep last boxes for each track to compute motion (attention)
        self.last_boxes = {}

    def extract_features(self, track, frame_shape):
        """Extracts adaptive features from the track for attention and anomaly detection."""
        box = track.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        img_area = frame_shape[0] * frame_shape[1]

        norm_area = (w * h) / (img_area + 1e-9)
        aspect_ratio = (w / h) if h > 0 else 0.0

        # Motion: normalized delta movement between frames (relative to diagonal)
        track_id = int(track.id)
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        motion = 0.0
        if track_id in self.last_boxes:
            prev_box = self.last_boxes[track_id]
            prev_center = np.array([(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2])
            delta = center - prev_center
            diag = np.linalg.norm([frame_shape[0], frame_shape[1]]) + 1e-9
            motion = np.linalg.norm(delta) / diag

        self.last_boxes[track_id] = box

        confidence = float(track.conf.cpu().numpy()) if hasattr(track, "conf") else 0.0

        return [norm_area, aspect_ratio, motion, confidence]

    def update_locks(self, tracks, frame_shape):
        """Update locking metadata for tracked objects.

        The system uses a combination of:
        - Absolute size thresholds (large target lock)
        - Online anomaly detection (Isolation Forest)
        - Adaptive z-score based outlier detection
        - Attention scoring (motion + size + confidence)
        """
        metadata = {}
        current_features = []

        # 1. Feature Extraction
        for track in tracks:
            feats = self.extract_features(track, frame_shape)
            current_features.append(feats)
            self.feature_history.append(feats)

        # 2. Train Anomaly Detector (Online Learning)
        if len(self.feature_history) >= ANOMALY_BUFFER_SIZE and not self.is_fitted:
            self.clf.fit(list(self.feature_history))
            self.is_fitted = True

        # Prepare adaptive statistics for z-score based anomaly detection
        history = np.array(self.feature_history)
        if history.shape[0] > 0:
            mean = history.mean(axis=0)
            std = history.std(axis=0) + 1e-9
        else:
            mean = np.zeros((4,))
            std = np.ones((4,))

        # 3. Apply Locking Rules
        for i, track in enumerate(tracks):
            track_id = int(track.id)
            norm_area, aspect_ratio, motion, confidence = current_features[i]
            z_scores = (np.array(current_features[i]) - mean) / std

            status = "SCANNING"
            is_locked = False

            # Check existing lock
            if track_id in self.lock_timers:
                self.lock_timers[track_id] -= 1
                if self.lock_timers[track_id] > 0:
                    status = f"LOCKED ({self.lock_timers[track_id]})"
                    is_locked = True
                else:
                    del self.lock_timers[track_id]  # Expired

            # Check for new lock triggers
            if not is_locked:
                trigger = False
                lock_duration = LOCK_DURATION_FRAMES

                # Attention Score (motion + size + confidence)
                attention_score = (norm_area * 0.6) + (motion * 0.3) + (confidence * 0.1)
                if attention_score >= ATTENTION_THRESHOLD:
                    status = "ATTENTION"
                    lock_duration = int(LOCK_DURATION_FRAMES * ATTENTION_LOCK_MULTIPLIER)
                    trigger = True

                # Trigger A: Large Object
                if not trigger and norm_area > LOCK_SIZE_THRESHOLD:
                    status = "TARGET (SIZE)"
                    trigger = True

                # Trigger B: Adaptive Z-Score (outliers in size/motion)
                if not trigger and abs(z_scores[0]) > 2.5:
                    status = "ADAPTIVE OUTLIER"
                    trigger = True

                # Trigger C: Anomaly (ML)
                if not trigger and self.is_fitted:
                    if self.clf.predict([current_features[i]])[0] == -1:
                        status = "ANOMALY"
                        trigger = True

                if trigger:
                    lock_duration = max(1, lock_duration)
                    self.lock_timers[track_id] = lock_duration
                    is_locked = True

            metadata[track_id] = {"locked": is_locked, "status": status}

        return metadata
