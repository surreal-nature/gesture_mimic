"""Pose estimation and keypoint processing for gesture mimic.

Extracts human upper-body keypoints from RGB frames using MediaPipe Pose,
normalizes them relative to shoulder center / torso length, and computes
velocity and acceleration features for temporal context.

The output feature vector replaces raw RGB images as ACT policy input,
dramatically reducing the learning problem from perception+control to
control-only — critical for small datasets (50 episodes).

Usage:
    estimator = PoseEstimator()
    for frame in video_frames:
        features = estimator.process_frame(frame)  # (feature_dim,) np array
    estimator.close()
"""

import numpy as np

UPPER_BODY_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_index": 19,
    "right_index": 20,
}

RIGHT_ARM_LANDMARKS = {
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
    "right_index": 20,
    "right_hip": 24,
    "left_shoulder": 11,
    "left_hip": 23,
}

LEFT_ARM_LANDMARKS = {
    "left_shoulder": 11,
    "left_elbow": 13,
    "left_wrist": 15,
    "left_index": 19,
    "left_hip": 23,
    "right_shoulder": 12,
    "right_hip": 24,
}

LANDMARK_PRESETS = {
    "upper_body": UPPER_BODY_LANDMARKS,
    "right_arm": RIGHT_ARM_LANDMARKS,
    "left_arm": LEFT_ARM_LANDMARKS,
}


class PoseEstimator:
    """Extracts normalized keypoints with temporal features from RGB frames.

    Args:
        landmarks: Dict of {name: mediapipe_index} or a preset name
            ("upper_body", "right_arm", "left_arm").
        include_velocity: Include first-order temporal derivatives.
        include_acceleration: Include second-order temporal derivatives.
        smoothing_alpha: Exponential moving average factor for keypoint
            smoothing (0 = no smoothing, 0.5 = heavy smoothing).
            Reduces MediaPipe frame-to-frame jitter.
        use_3d: Use 3D (x,y,z) coordinates. If False, use 2D (x,y) only.
        model_complexity: MediaPipe model complexity (0=lite, 1=full, 2=heavy).
    """

    def __init__(
        self,
        landmarks="upper_body",
        include_velocity=True,
        include_acceleration=True,
        smoothing_alpha=0.3,
        use_3d=True,
        model_complexity=2,
    ):
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        if isinstance(landmarks, str):
            self.landmarks = LANDMARK_PRESETS[landmarks]
        else:
            self.landmarks = landmarks

        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration
        self.smoothing_alpha = smoothing_alpha
        self.use_3d = use_3d
        self.n_coords = 3 if use_3d else 2

        self._sorted_names = sorted(self.landmarks.keys())
        self._prev_keypoints = None
        self._prev_velocity = None
        self._smoothed_keypoints = None
        self._detection_failures = 0

    def extract_keypoints(self, rgb_frame):
        """Extract raw keypoints from an RGB uint8 frame.

        Returns dict of {name: np.array([x, y, z])} or None if detection fails.
        """
        results = self.pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None

        keypoints = {}
        for name, idx in self.landmarks.items():
            lm = results.pose_landmarks.landmark[idx]
            if self.use_3d:
                keypoints[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            else:
                keypoints[name] = np.array([lm.x, lm.y], dtype=np.float32)
        return keypoints

    def normalize_keypoints(self, keypoints):
        """Normalize keypoints relative to shoulder center, scaled by torso.

        Provides scale, translation, and camera invariance.
        """
        if keypoints is None:
            return None

        left_sh = keypoints.get("left_shoulder", np.zeros(self.n_coords))
        right_sh = keypoints.get("right_shoulder", np.zeros(self.n_coords))
        shoulder_center = (left_sh + right_sh) / 2.0

        left_hip = keypoints.get("left_hip", np.zeros(self.n_coords))
        right_hip = keypoints.get("right_hip", np.zeros(self.n_coords))
        hip_center = (left_hip + right_hip) / 2.0

        torso_length = np.linalg.norm(shoulder_center - hip_center)
        if torso_length < 0.01:
            torso_length = 1.0

        normalized = {}
        for name, coords in keypoints.items():
            normalized[name] = (coords - shoulder_center) / torso_length
        return normalized

    def keypoints_to_vector(self, normalized_keypoints):
        """Convert normalized keypoints dict to flat feature vector."""
        if normalized_keypoints is None:
            return np.zeros(len(self._sorted_names) * self.n_coords, dtype=np.float32)

        parts = [normalized_keypoints[name] for name in self._sorted_names]
        return np.concatenate(parts).astype(np.float32)

    def _apply_smoothing(self, current):
        """Exponential moving average to reduce MediaPipe jitter."""
        if self.smoothing_alpha <= 0 or self._smoothed_keypoints is None:
            self._smoothed_keypoints = current.copy()
            return current
        alpha = self.smoothing_alpha
        self._smoothed_keypoints = alpha * self._smoothed_keypoints + (1 - alpha) * current
        return self._smoothed_keypoints.copy()

    def process_frame(self, rgb_frame):
        """Full pipeline: extract, normalize, smooth, compute temporal features.

        Args:
            rgb_frame: (H, W, 3) uint8 RGB numpy array.

        Returns:
            Feature vector of shape (feature_dim,) as float32 numpy array.
            If pose detection fails, uses previous frame's keypoints (or zeros).
        """
        keypoints = self.extract_keypoints(rgb_frame)

        if keypoints is None:
            self._detection_failures += 1
            if self._smoothed_keypoints is not None:
                position = self._smoothed_keypoints.copy()
            else:
                position = np.zeros(len(self._sorted_names) * self.n_coords, dtype=np.float32)
        else:
            normalized = self.normalize_keypoints(keypoints)
            position = self.keypoints_to_vector(normalized)
            position = self._apply_smoothing(position)

        features = [position]

        if self.include_velocity:
            if self._prev_keypoints is not None:
                velocity = position - self._prev_keypoints
            else:
                velocity = np.zeros_like(position)
            features.append(velocity)

            if self.include_acceleration:
                if self._prev_velocity is not None:
                    acceleration = velocity - self._prev_velocity
                else:
                    acceleration = np.zeros_like(position)
                features.append(acceleration)
                self._prev_velocity = velocity.copy()

        self._prev_keypoints = position.copy()
        return np.concatenate(features).astype(np.float32)

    def reset(self):
        """Reset temporal state between episodes."""
        self._prev_keypoints = None
        self._prev_velocity = None
        self._smoothed_keypoints = None

    @property
    def feature_dim(self):
        """Total dimension of output feature vector."""
        n_base = len(self._sorted_names) * self.n_coords
        dim = n_base
        if self.include_velocity:
            dim += n_base
        if self.include_acceleration:
            dim += n_base
        return dim

    @property
    def feature_names(self):
        """Human-readable names for each dimension of the feature vector."""
        axes = ["x", "y", "z"] if self.use_3d else ["x", "y"]
        names = []
        for name in self._sorted_names:
            for axis in axes:
                names.append(f"kp.{name}.{axis}")
        if self.include_velocity:
            for name in self._sorted_names:
                for axis in axes:
                    names.append(f"kp.{name}.vel.{axis}")
        if self.include_acceleration:
            for name in self._sorted_names:
                for axis in axes:
                    names.append(f"kp.{name}.acc.{axis}")
        return names

    @property
    def detection_failure_count(self):
        return self._detection_failures

    def close(self):
        self.pose.close()
