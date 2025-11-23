# Cricket Video Analysis System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🏏 Overview

AI-powered cricket video analysis system that detects and analyzes cricket shots using computer vision and deep learning. The system provides automated feedback on batting technique, shot classification, and performance metrics for both desktop and mobile platforms.

## ✨ Features

- **Object Detection**: Real-time detection of players, bat, ball, and wickets using YOLOv8
- **Pose Estimation**: Body keypoint extraction using MediaPipe/MoveNet for technique analysis
- **Shot Classification**: Automatic identification of cricket shots (cover drive, pull, cut, etc.)
- **Performance Metrics**: Technical analysis including swing angle, footwork, timing, and ball trajectory
- **Automated Feedback**: GPT-4 powered coaching suggestions and performance insights
- **Multi-Platform**: Desktop prototype with Android mobile deployment
- **Real-time Analysis**: Process live camera feeds or recorded videos

## 🏗️ Project Structure

```
cricket-video-analysis/
├── src/
│   ├── detection/          # Object detection modules
│   │   ├── yolo_detector.py
│   │   └── tracker.py
│   ├── pose/               # Pose estimation modules
│   │   ├── pose_estimator.py
│   │   └── keypoint_analyzer.py
│   ├── classification/     # Shot classification
│   │   ├── shot_classifier.py
│   │   └── action_detector.py
│   ├── analysis/           # Metrics and analysis
│   │   ├── metrics_calculator.py
│   │   └── feedback_generator.py
│   ├── utils/              # Utility functions
│   │   ├── video_processor.py
│   │   └── config.py
│   └── mobile/             # Mobile deployment code
│       └── android/
├── models/                 # Trained models
│   ├── yolo/
│   └── pose/
├── data/                   # Sample data and datasets
│   ├── videos/
│   └── annotations/
├── notebooks/              # Jupyter notebooks for experiments
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA for GPU acceleration

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sitaraman-newlife/cricket-video-analysis.git
cd cricket-video-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models:
```bash
python scripts/download_models.py
```

### Quick Start

```python
from src.analysis.cricket_analyzer import CricketAnalyzer

# Initialize analyzer
analyzer = CricketAnalyzer()

# Analyze video
results = analyzer.analyze_video('path/to/cricket_video.mp4')

# Get feedback
feedback = analyzer.generate_feedback(results)
print(feedback)
```

## 📱 Mobile Deployment

### Android

The mobile version uses TensorFlow Lite for efficient on-device inference.

1. Convert models to TFLite:
```bash
python scripts/convert_to_tflite.py
```

2. Open Android project in Android Studio:
```bash
cd src/mobile/android
```

3. Build and run on your device

## 🔧 Technical Stack

- **Computer Vision**: OpenCV, YOLOv8
- **Pose Estimation**: MediaPipe, MoveNet
- **Deep Learning**: PyTorch, TensorFlow
- **API Integration**: OpenAI GPT-4
- **Mobile**: TensorFlow Lite, Android SDK
- **Utilities**: NumPy, Pandas, Matplotlib

## 📊 Usage Examples

### Desktop Analysis

```python
# Process a video file
from src.detection.yolo_detector import YOLODetector
from src.pose.pose_estimator import PoseEstimator

detector = YOLODetector(model_path='models/yolo/cricket.pt')
pose_est = PoseEstimator()

# Process video
for frame in video_frames:
    detections = detector.detect(frame)
    keypoints = pose_est.estimate(frame)
    # Analyze and visualize
```

### Real-time Camera Analysis

```python
# Use webcam or phone camera
analyzer = CricketAnalyzer(source=0)  # 0 for webcam
analyzer.start_realtime_analysis()
```

## 🎯 Roadmap

- [x] Basic object detection (bat, ball, player)
- [x] Pose estimation integration
- [ ] Shot classification model training
- [ ] Metrics calculation module
- [ ] GPT-4 feedback integration
- [ ] Android app development
- [ ] Real-time mobile inference
- [ ] Cloud deployment
- [ ] Multi-player analysis
- [ ] Bowling action analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sitaraman**
- GitHub: [@sitaraman-newlife](https://github.com/sitaraman-newlife)

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- MediaPipe by Google
- OpenCV community
- OpenAI GPT-4

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an active development project. The system is currently in the prototype phase with ongoing improvements.
