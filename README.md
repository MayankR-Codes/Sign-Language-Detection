# Sign Language Detection using YOLOv8

A real-time sign language detection system built using YOLOv8 and OpenCV, capable of recognizing sign language gestures through webcam or image input.

## Features

- **Real-time Detection**: Process live webcam feed for instant sign language recognition
- **Image Testing**: Test the model on individual images
- **Model Training**: Train the model from scratch using custom dataset
- **Performance Metrics**: View model performance including Precision, Recall, F1 Score, and mAP50
- **CPU Support**: Optimized for CPU-based inference and training

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- CUDA (optional, for GPU acceleration)

## Project Structure

```
Sign Language Prediction/
├── app.py              # Main application file
├── data.yaml           # Dataset configuration
├── Dataset/
│   ├── images/         # Training and validation images
│   │   ├── train/
│   │   └── val/
│   └── labels/         # YOLO format annotations
│       ├── train/
│       └── val/
└── runs/               # Training outputs and model weights
```

## Installation

1. Create a virtual environment:
```bash
python -m venv myenv
```

2. Activate the virtual environment:
```bash
# Windows
myenv\Scripts\activate

# Linux/Mac
source myenv/bin/activate
```

3. Install dependencies:
```bash
pip install ultralytics opencv-python
```

## Usage

1. **Running the Application**:
```bash
python app.py
```

2. **Menu Options**:
   - Train model from scratch
   - Test using webcam
   - Test single image
   - Show model performance
   - Exit

## Model Training

The model is trained using YOLOv8 with the following configurations:
- Image size: 800x800
- Batch size: 4
- Epochs: 300
- Data augmentation: Enabled (flip, rotation, HSV augmentation)
- Learning rate: 0.001 → 0.0001

## Performance

The model's performance can be evaluated using:
- Precision
- Recall
- F1 Score
- mAP50 (mean Average Precision at IoU=50%)

## Contributing

Feel free to open issues and pull requests for improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.