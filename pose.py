"""Mediapipe Pose wrapper — detects 33 full-body landmarks."""

import mediapipe as mp
import numpy as np


class PoseEstimator:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,       # 0=lite, 1=full, 2=heavy
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_pose = mp.solutions.pose
        self.pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, bgr_frame: np.ndarray):
        """
        Run pose detection on a BGR frame.
        Returns mediapipe results object (results.pose_landmarks or None).
        """
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True
        return results

    def get_landmark_coords(self, results, frame_shape: tuple) -> dict:
        """
        Returns a dict of {landmark_name: (x_px, y_px, z, visibility)}
        for every detected landmark, scaled to pixel coordinates.
        """
        if not results.pose_landmarks:
            return {}

        h, w = frame_shape[:2]
        coords = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            name = self._mp_pose.PoseLandmark(idx).name
            coords[name] = (int(lm.x * w), int(lm.y * h), lm.z, lm.visibility)
        return coords

    def close(self):
        self.pose.close()
