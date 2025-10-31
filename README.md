# Sign Language Detection using YOLOv8

A real-time sign language detection system built using YOLOv8 and OpenCV, capable of recognizing sign language gestures through webcam or image input. This project aims to bridge communication gaps by providing instant sign language recognition, making it easier for people to understand and interpret sign language gestures.

## Project Overview

This system uses YOLOv8, a state-of-the-art object detection model, specifically trained to recognize various sign language gestures. The model is optimized for CPU usage, making it accessible without requiring specialized hardware. Key aspects include:

- **Real-time Processing**: Achieves real-time detection with minimal latency
- **High Accuracy**: Utilizes YOLOv8's advanced architecture for accurate gesture recognition
- **User-friendly Interface**: Simple menu-driven interface for easy interaction
- **Multiple Input Options**: Supports both webcam feed and image inputs

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

## Model Architecture and Training

### YOLOv8 Configuration
The model utilizes YOLOv8's nano configuration (yolov8n) which offers a good balance between speed and accuracy, making it suitable for real-time applications. The model is trained with the following specifications:

#### Training Parameters
- **Image size**: 800x800 (optimized for hand gesture detection)
- **Batch size**: 4 (CPU-optimized)
- **Epochs**: 300 (extensive training for better accuracy)
- **Device**: CPU (accessible implementation)
- **Workers**: 2 (optimized for multi-core processors)

#### Data Augmentation
Robust augmentation techniques are employed to improve model generalization:
- Horizontal flip (probability: 0.5)
- Vertical flip (probability: 0.5)
- Rotation (±15 degrees)
- Translation (±20%)
- Scale variation (±50%)
- HSV color space augmentation:
  - Hue: ±0.015
  - Saturation: ±0.7
  - Value: ±0.4

#### Training Optimization
- **Learning rate**: Adaptive (0.001 → 0.0001)
- **Early stopping**: 50 epochs patience
- **Pretrained weights**: None (trained from scratch)
- **Model freezing**: None (full model training)

## Performance and Metrics

The model's performance is evaluated using standard object detection metrics:

### Key Metrics
- **Precision**: Accuracy of positive predictions
  - Formula: TP / (TP + FP)
  - Indicates how many detected signs are correct
  
- **Recall**: Ability to find all relevant instances
  - Formula: TP / (TP + FN)
  - Shows proportion of actual signs correctly identified
  
- **F1 Score**: Harmonic mean of precision and recall
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)
  - Balances precision and recall
  
- **mAP50**: Mean Average Precision at 50% IoU
  - Evaluates detection quality
  - Considers both location accuracy and class prediction

### Real-world Performance
- Real-time processing at 15-20 FPS on CPU
- Low latency response (200-300ms)
- Robust performance under various lighting conditions
- Effective detection range: 0.5-3 meters

## Application Use Cases

1. **Educational Settings**
   - Sign language learning tools
   - Interactive teaching aids
   - Student practice validation

2. **Communication Assistance**
   - Real-time sign language interpretation
   - Communication bridge for hearing-impaired individuals
   - Public space accessibility improvement

3. **Research and Development**
   - Sign language pattern analysis
   - Gesture recognition research
   - Human-computer interaction studies

## Future Improvements

- [ ] Add support for continuous gesture recognition
- [ ] Implement sentence formation from multiple gestures
- [ ] Add GPU support for faster processing
- [ ] Expand the dataset for more sign variations
- [ ] Implement transfer learning for better initialization
- [ ] Add support for different sign language systems

## Contributing

Feel free to open issues and pull requests for improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.