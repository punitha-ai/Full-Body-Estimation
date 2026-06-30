# Full Body Pose Estimation Using MediaPipe

## Introduction

Full Body Pose Estimation is a computer vision technique that identifies and tracks key points of the human body from images or video streams. This project uses MediaPipe Pose and OpenCV to perform real-time human pose detection through a webcam. The system detects body landmarks and visualizes them as a skeletal structure, allowing accurate tracking of human posture and movement.

The project demonstrates how modern AI-powered computer vision models can understand and analyze human body positions without requiring specialized hardware or sensors.


## Project Objective

The primary objective of this project is to build a real-time pose estimation system capable of detecting and tracking human body landmarks from live video input. By leveraging MediaPipe's pose detection framework, the application can identify body joints and generate a visual representation of the user's posture.

This project serves as a foundation for developing advanced applications such as fitness monitoring, gesture recognition, sports analytics, rehabilitation systems, and human-computer interaction solutions.


## How the Project Works

The application captures video frames from a webcam using OpenCV. Each frame is processed by MediaPipe Pose, which uses machine learning models to detect key body landmarks. The detected landmarks are then connected to form a skeletal structure representing the human body.

The system continuously processes incoming frames, allowing real-time tracking of body movements and posture changes.

### Workflow

1. Capture video from the webcam.
2. Convert the frame into a format suitable for MediaPipe processing.
3. Detect body landmarks using MediaPipe Pose.
4. Extract landmark coordinates.
5. Draw landmarks and skeletal connections.
6. Display the processed video feed with pose annotations.

---

## Key Features

* Real-time full-body pose detection.
* Detection of 33 body landmarks.
* Live skeletal visualization.
* Accurate tracking of body movements.
* Lightweight and efficient implementation.
* Easy integration with other computer vision applications.
* Works directly with a standard webcam.

---

## Technologies Used

### MediaPipe

MediaPipe is Google's open-source framework for building multimodal machine learning pipelines. The Pose solution provides highly accurate body landmark detection and tracking.

### OpenCV

OpenCV is used for video capture, frame processing, and displaying the output video stream.

### Python

Python serves as the primary programming language for implementing the project and integrating the required libraries.

---

## Body Landmarks Detected

The model detects 33 key body landmarks, including:

* Nose
* Eyes
* Ears
* Mouth
* Shoulders
* Elbows
* Wrists
* Fingers
* Hips
* Knees
* Ankles
* Feet

