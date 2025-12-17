# 🎥 Smart Video Surveillance System

A real-time **computer vision surveillance system** built using **YOLO (Ultralytics)** and **OpenCV**, designed to process live video streams (camera, RTSP, or video files), perform object detection, and save the processed output video efficiently using multithreading.

---

## 🚀 Features

- ✅ Real-time object detection using **YOLOv8**
- ✅ Supports:
  - Webcam
  - Video files
  - RTSP streams
- ✅ Multithreaded architecture (capture vs inference)
- ✅ Smooth FPS calculation (processing FPS vs output FPS)
- ✅ Annotated output video with:
  - Bounding boxes
  - Confidence scores
  - Timestamp
  - FPS overlay
- ✅ Efficient frame queue to reduce latency
- ✅ Saves processed video to disk

---

## 🧠 System Architecture

Video Source ──▶ Frame Capture ──▶ Queue ──▶ YOLO Inference Thread ──▶ Video Writer

- **Main thread**: reads frames from source
- **Worker thread**: runs YOLO inference and video writing
- **Queue**: buffers frames and controls latency

---

## 📂 Project Structure


Smart Video Surveillance System/
│
├── main.py # Entry point (video capture & pipeline setup)
├── detector.py # YOLO inference worker (threaded)
├── output.mp4 # Processed output video
├── requirements.txt # Python dependencies
├── README.md # Project documentation
