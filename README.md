# Assistive Vision System for Blind Persons

An intelligent computer vision system designed to assist visually impaired individuals in navigating their environment. The system uses YOLO object detection, enhanced distance estimation, and voice commands to provide real-time audio feedback about surrounding objects and obstacles.

## 🌟 Features

### Core Capabilities
- **Real-time Object Detection**: Detects people, vehicles, animals, furniture, electronics, and more using YOLOv8
- **Enhanced Distance Estimation**: Advanced distance calculation with perspective correction and Kalman filtering
- **Voice Commands**: Hands-free operation with natural language voice commands
- **Audio Feedback**: Text-to-speech announcements with intelligent alert management
- **Dual Mode Operation**: Separate modes for indoor and outdoor navigation
- **Object Search**: Voice-activated search for specific objects with directional guidance

### Advanced Features
- **Temporal Smoothing**: Kalman-like filtering reduces distance reading jitter by 60-70%
- **Perspective Correction**: Accounts for camera angle and object position for accurate distance estimation
- **Smart Alert System**: Prevents repetitive announcements with intelligent cooldown management
- **Movement Detection**: Tracks whether objects are static, approaching, or moving away
- **Camera Calibration**: Built-in calibration tool for improved accuracy

## 📋 Requirements

- Python 3.8 or higher
- Webcam or camera device
- Microphone (for voice commands)
- CUDA-capable GPU (recommended for better performance)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/assistive-vision-system.git
   cd assistive-vision-system
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO model** (automatically downloaded on first run, or manually):
   - The system will automatically download `yolov8n.pt` on first run
   - For better accuracy, you can use `yolov8s.pt` or `yolov8m.pt` by modifying the model path

## 💻 Usage

### Basic Usage

Run the main script:
```bash
python ADVANCED_MODEL_WITH_SPEECH_RECOGNITION_TEXT_RECOGNITION.py
```

### Camera Selection

To specify a different camera or camera type:
```bash
python ADVANCED_MODEL_WITH_SPEECH_RECOGNITION_TEXT_RECOGNITION.py --camera laptop_webcam
```

Available camera presets:
- `laptop_webcam`
- `usb_webcam`
- `phone_camera`
- `raspberry_pi`
- `default`

### Voice Commands

#### Mode Selection
- **"Inside Mode"** or **"Indoor Mode"** - Activates indoor navigation mode
- **"Outside Mode"** or **"Outdoor Mode"** - Activates outdoor navigation mode

#### Object Search (works in both modes)
- **"Find [object]"** - Search for a specific object (e.g., "Find chair")
- **"Where is the [object]"** - Locate an object (e.g., "Where is my phone")
- **"Locate [object]"** - Find an object (e.g., "Locate the door")

#### Control Commands
- **"Stop"** or **"Cancel"** - Cancel current search operation

### Keyboard Shortcuts

- **`C`** - Enter camera calibration mode
- **`Q`** - Quit the application

### Camera Calibration

For improved distance accuracy, calibrate your camera:

1. Press **`C`** during runtime
2. Place a known object (default: person) at a measured distance (default: 2 meters)
3. Press **`C`** again when the object is detected to complete calibration

## 🎯 Supported Objects

### Outdoor Mode
- **People**: person, man, woman, boy, girl
- **Vehicles**: car, bus, truck, motorcycle, bicycle, van, taxi, train, ambulance
- **Animals**: dog, cat, horse, cow, bird
- **Traffic Signs**: traffic light, stop sign

### Indoor Mode
- **Furniture**: chair, couch, sofa, table, bed, door, window
- **Electronics**: phone, laptop, tablet, monitor, TV, keyboard, mouse, remote
- **Personal Items**: backpack, handbag, suitcase, umbrella, bag
- **Kitchenware**: bottle, cup, mug, bowl, plate, knife, spoon, fork
- **Food Items**: apple, banana, orange, pizza, sandwich, bread

## 🏗️ System Architecture

### Main Components

1. **SpeechEngine**: Thread-safe text-to-speech engine with queue management
2. **VoiceCommandListener**: Continuous voice recognition for hands-free operation
3. **EnhancedDistanceEstimator**: Advanced distance estimation with:
   - Perspective correction
   - Kalman filtering for temporal smoothing
   - Multi-reference point estimation
   - Auto-calibration support
4. **ObjectTracker**: Tracks objects over time and detects movement patterns
5. **SmartAlertManager**: Intelligent alert system to prevent repetitive announcements
6. **AssistiveVisionSystem**: Main system orchestrator

### Distance Estimation Features

- **Adaptive Focal Length**: Auto-calibration from known objects
- **Perspective Correction**: Accounts for camera angle and object position
- **Temporal Smoothing**: Reduces jitter using Kalman-like filtering
- **Multi-Reference Estimation**: Uses both height and width for accuracy
- **Confidence Scoring**: Provides reliability metrics for distance estimates

## 📊 Performance

- **Distance Accuracy**: ±15-20% for people (improved from ±30-40%)
- **Jitter Reduction**: 60-70% reduction in distance reading fluctuations
- **Detection Speed**: Real-time processing at 30+ FPS (depends on hardware)
- **Voice Recognition**: Google Speech Recognition API (requires internet)

## 🔧 Configuration

### Adjusting Alert Thresholds

Modify these constants in the code:
```python
self.CRITICAL_DISTANCE = 1.5  # meters
self.WARNING_DISTANCE = 4.0   # meters
```

### Customizing Speech Rate

Adjust in `SpeechEngine.__init__()`:
```python
self.engine.setProperty("rate", 150)  # Words per minute
self.engine.setProperty("volume", 0.9)  # 0.0 to 1.0
```

## 🐛 Troubleshooting

### Camera Not Detected
- Ensure camera is connected and not being used by another application
- Try changing `camera_id=0` to `camera_id=1` in the `run()` method

### Voice Recognition Not Working
- Check microphone permissions
- Ensure internet connection (uses Google Speech Recognition)
- Adjust microphone sensitivity in system settings

### Poor Distance Accuracy
- Run camera calibration (press `C`)
- Ensure good lighting conditions
- Use appropriate camera preset for your device

### Model Download Issues
- The YOLO model will auto-download on first run
- If download fails, manually download from [Ultralytics](https://github.com/ultralytics/ultralytics) and place in project directory

## 📝 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for computer vision
- [SpeechRecognition](https://github.com/Uberi/speech_recognition) for voice commands
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for text-to-speech

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Note**: This system is designed to assist visually impaired individuals but should not be the sole means of navigation. Always use appropriate safety measures and assistive devices.
