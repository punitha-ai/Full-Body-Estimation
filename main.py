"""
Full-body pose estimation — webcam entry point.

Controls:
  Q  — quit
  S  — save current frame as snapshot.png
  1/2/3 — switch model complexity (lite / full / heavy)
"""

import time
import cv2
from pose import PoseEstimator
from visualizer import draw_skeleton, draw_info_panel, analyze_posture, draw_posture_label


def run(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    estimator = PoseEstimator(model_complexity=1)
    prev_time = time.time()

    print("Running — press Q to quit, S to save snapshot, 1/2/3 to change model complexity.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Empty frame, retrying...")
            continue

        results = estimator.process(frame)
        coords = estimator.get_landmark_coords(results, frame.shape)

        # Skeletal overlay
        draw_skeleton(frame, results)

        # FPS + landmark count HUD
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now
        draw_info_panel(frame, fps, len(coords))

        # Posture analysis
        posture = analyze_posture(coords)
        draw_posture_label(frame, posture)

        cv2.imshow("Full-Body Pose Estimation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("snapshot.png", frame)
            print("Snapshot saved as snapshot.png")
        elif key == ord("1"):
            estimator.close()
            estimator = PoseEstimator(model_complexity=0)
            print("Switched to LITE model")
        elif key == ord("2"):
            estimator.close()
            estimator = PoseEstimator(model_complexity=1)
            print("Switched to FULL model")
        elif key == ord("3"):
            estimator.close()
            estimator = PoseEstimator(model_complexity=2)
            print("Switched to HEAVY model")

    estimator.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
