"""Skeletal overlay and posture annotation helpers."""

import cv2
import mediapipe as mp
import numpy as np

_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles
_mp_pose = mp.solutions.pose

# Custom drawing specs
LANDMARK_SPEC = _mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
CONNECTION_SPEC = _mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)


def draw_skeleton(frame: np.ndarray, results) -> np.ndarray:
    """Draws landmarks and connections on the frame in-place."""
    if results.pose_landmarks:
        _mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            _mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_SPEC,
            connection_drawing_spec=CONNECTION_SPEC,
        )
    return frame


def draw_info_panel(frame: np.ndarray, fps: float, landmark_count: int) -> np.ndarray:
    """Renders a small HUD with FPS and detected landmark count."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, f"Landmarks: {landmark_count}/33", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return frame


def analyze_posture(coords: dict) -> str:
    """
    Simple shoulder-level check.
    Returns a posture label: 'Balanced', 'Left shoulder high', or 'Right shoulder high'.
    """
    left = coords.get("LEFT_SHOULDER")
    right = coords.get("RIGHT_SHOULDER")
    if not left or not right:
        return "No shoulders detected"

    diff = left[1] - right[1]   # y increases downward
    if abs(diff) < 15:
        return "Posture: Balanced"
    elif diff > 0:
        return "Posture: Right shoulder high"
    else:
        return "Posture: Left shoulder high"


def draw_posture_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Renders the posture label at the bottom of the frame."""
    h, w = frame.shape[:2]
    color = (0, 255, 0) if "Balanced" in label else (0, 140, 255)
    cv2.putText(frame, label, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
