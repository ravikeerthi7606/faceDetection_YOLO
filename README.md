# faceDetection_YOLO
📌 Overview

Custom YOLOv11-based object detection system integrated with OpenCV LBPH face recognition.
Designed for fast, accurate person detection and recognition using a custom dataset.

🛠 Tech Stack

Python 3.x

YOLOv11 (Ultralytics)

OpenCV

Label Studio

VS Code

📂 Project Structure
yolo_model/
├── dataset_split/
├── runs/
├── face_recognition.py
├── train.py
├── predict.py
├── data.yaml
└── README.md


⚙️ Setup
python -m venv yolo-env
yolo-env\Scripts\activate   # Windows
pip install -r requirements.txt
🏋️ Training
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
🔍 Inference
yolo detect predict model=runs/detect/train/weights/best.pt source=test.jpg

Example output:

image 1/1: 640x640 1 hari, 10.2ms
Speed: 4.2ms preprocess, 10.2ms inference, 1.5ms postprocess
🧠 Face Recognition
cv2.face.LBPHFaceRecognizer_create()

Pipeline:

Grayscale conversion

Face detection

Train recognizer

Predict identity

🚀 Future Improvements

Real-time webcam detection

Multi-person recognition

Flask/FastAPI deployment

Docker & Cloud deployment

👨‍💻 Author

Ravi Keerthi R
YOLO R&D Project