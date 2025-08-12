# 🚦 Smart Traffic Analyzer

The **Smart Traffic Analyzer** is an AI-powered web application that analyzes traffic images and videos to detect:  
- 🚗 **Vehicle counts**  
- 🚦 **Red light violations**  
- 🖁 **Wrong-way driving**  
- 📊 **Vehicle speeds** (average & maximum)  

It uses **YOLOv8** object detection models and a custom tracking algorithm to process media files, annotate results, and provide downloadable reports.

---

## 📌 Features

- **Image Analysis**
  - Detects vehicles
  - Detects red light status
  - Identifies wrong-way vehicles
  - Outputs annotated images

- **Video Analysis**
  - Detects and counts vehicles
  - Tracks vehicles using `SimpleTracker`
  - Calculates average & max speed
  - Identifies wrong-way driving
  - Detects red light violations
  - Generates annotated video output
  - Produces CSV speed report

- **Web Interface**
  - Built with Flask
  - Upload images or videos
  - View processed results directly in browser
  - Download processed video & CSV report

---

## 🛠 Tech Stack

- **Backend**: Python, Flask  
- **Computer Vision**: OpenCV, NumPy  
- **Object Detection**: [YOLOv8](https://github.com/ultralytics/ultralytics)  
- **Tracking**: Custom `SimpleTracker`  
- **Frontend**: HTML, CSS (Bootstrap 5)  

---
## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/Smart-Traffic-Analyzer.git
cd Smart-Traffic-Analyzer
```

## 2️⃣ Create & activate a virtual environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the App

1. Place YOLO model weights (.pt files) in the project root.

2. Start the Flask server:

```bash
python flask_app.py
```

3. Open your browser and go to:

```cpp
http://127.0.0.1:5000/
```

4. Upload an image or video to analyze traffic.

## 📊 Output

- Image Analysis → Annotated image preview in browser

- Video Analysis:

   - Live preview in browser

   - Download:

       - Processed Video (MP4)

       - Speed Report (CSV)

   - Summary statistics:

       - 🚗 Vehicle Count

       - 🚦 Red Light Violations

       - 🖁 Wrong-Way Vehicles

       - ⚡ Average Speed

       - 🚀 Max Speed


## 📌 Notes

- Videos are encoded in H.264 (avc1) for browser compatibility.

- Large model files (.pt) are excluded from GitHub.

- CSV reports include speed data per vehicle ID with timestamps.

