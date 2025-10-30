# SIGN LANGUAGE DETECTION - YOLOv8 FROM SCRATCH (CPU)

from ultralytics import YOLO
import os
from pathlib import Path
import cv2


data_yaml_path = r"C:\Users\KIIT\Desktop\Sign Language Prediction\data.yaml"

model_name = "sign_language_cpu_scratch18"

weights_path = Path(f"runs/detect/{model_name}/weights/best.pt")

val_images = r"C:\Users\KIIT\Desktop\Sign Language Prediction\Dataset\Images\val"

def train_model():
    print("Starting YOLOv8 training on Sign Language dataset (CPU mode)...\n")
    
    model = YOLO("yolov8n.yaml")  # Nano model for CPU-friendly training
    
    model.train(
        data=data_yaml_path,
        epochs=300,          # train longer for better accuracy
        imgsz=800,           # larger image size for better hand detection
        batch=4,
        device='cpu',
        workers=2,
        name=model_name,
        pretrained=False,
        augment=True,
        flipud=0.5,
        fliplr=0.5,
        degrees=15,
        translate=0.2,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        lr0=0.001,
        lrf=0.0001,
        freeze=0,
        patience=50
    )
    print("\n Training completed!")

def test_webcam():
    if not weights_path.exists():
        print(f"\nModel weights not found: {weights_path}")
        return
    
    print("\nInitializing webcam detection...")
    try:
        model = YOLO(str(weights_path))
        
        # Try different camera indices
        for camera_index in range(2):  # Try both 0 and 1
            print(f"Attempting to access camera {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Successfully connected to camera {camera_index}")
                break
            cap.release()
        else:
            print("ERROR: Could not access any webcam! Please check:")
            return

        print("\nWebcam activated! Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam")
                break
            
            # Run prediction
            results = model.predict(frame, conf=0.5, device='cpu')
            result = results[0]
            
            # Add prediction text if any detection
            if len(result.boxes) > 0:
                confidence = result.boxes[0].conf.item()
                class_id = result.boxes[0].cls.item()
                class_name = model.names[int(class_id)]
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw predictions on frame
            frame = result.plot()
            
            cv2.imshow("Sign Language Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam closed.")

def show_model_metrics():
    if not weights_path.exists():
        print(f"\nModel weights not found: {weights_path}")
        return
    
    print("\nEvaluating model performance...")
    try:
        model = YOLO(str(weights_path))
        
        # Run validation to get metrics
        results = model.val(data=data_yaml_path, device='cpu')
        
        # Extract metrics
        metrics = results.results_dict
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        map50 = metrics.get('metrics/mAP50(B)', 0)
        
        # Calculate F1 score manually using precision and recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Print metrics in a formatted way
        print("\nMODEL PERFORMANCE METRICS")
        print("=" * 40)
        print(f"Precision    : {precision:.4f}")
        print(f"Recall       : {recall:.4f}")
        print(f"F1 Score     : {f1_score:.4f}")
        print(f"mAP50        : {map50:.4f}")
        print("=" * 40)
    
    except Exception as e:
        print(f"\nAn error occurred while calculating metrics: {str(e)}")
        print("Please check if the model and validation data are properly set up.")

def test_single_image():
    if not weights_path.exists():
        print(f"\nModel weights not found: {weights_path}")
        return
    
    # Get image path from user
    image_path = input("\nEnter the full path of the image to test: ").strip().strip('"').strip("'")
    
    if not os.path.exists(image_path):
        print("Image file not found! Please check the path and try again.")
        return
    
    print("\nRunning prediction on the image...")
    try:
        model = YOLO(str(weights_path))
        
        # Run prediction
        results = model.predict(
            source=image_path,
            conf=0.5,
            save=False,  # Don't save the image
            show=False,
            device='cpu'
        )
        
        # Get prediction details
        result = results[0]
        if len(result.boxes) > 0:
            confidence = result.boxes[0].conf.item()
            class_id = result.boxes[0].cls.item()
            class_name = model.names[int(class_id)]
            
            print("\nPrediction Results:")
            print("=" * 30)
            print(f"Detected Sign : {class_name}")
            print(f"Confidence    : {confidence:.4f} ({confidence*100:.1f}%)")
            print("=" * 30)
        else:
            print("\nNo signs detected in the image!")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

def main_menu():
    while True:
        print("  SIGN LANGUAGE DETECTION MENU  ")
        print("1. Train model from scratch")
        print("2. Test using webcam")
        print("3. Test single image")
        print("4. Show model performance")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            train_model()
        elif choice == '2':
            test_webcam()
        elif choice == '3':
            test_single_image()
        elif choice == '4':
            show_model_metrics()
        elif choice == '5':
            print("\nExiting program. Goodbye!")
            break
        else:
            print("\nInvalid choice! Please enter a number from 1-5.")

if __name__ == "__main__":
    main_menu()