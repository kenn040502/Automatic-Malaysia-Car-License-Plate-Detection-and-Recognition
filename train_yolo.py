from ultralytics import YOLO
import torch
import os

def train_yolo_model(model_type="yolov8t.pt", epochs=10, batch=16, imgsz=640, run_name="yolov8_model"):
    """
    Trains a YOLOv8 model with the given parameters.

    Parameters:
        model_type (str): The pre-trained YOLOv8 model to use.
        epochs (int): Number of training epochs.
        batch (int): Batch size used during training.
        imgsz (int): Input image resolution.
        run_name (str): The name of the run folder inside 'runs/train/'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚗 Training YOLO model: {model_type}")
    print(f"📦 Epochs: {epochs}, Batch size: {batch}, Image size: {imgsz}")
    print(f"💻 Using device: {device.upper()}")

    # Train the model
    model = YOLO(model_type)
    model.train(
        data="dataset/dataset.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        project="runs/train",
        pretrained=True,
        device=device
    )

    print("✅ Training complete!")

# === Entry point ===
if __name__ == "__main__":
    train_yolo_model(
        model_type="yolov8s.pt",  
        epochs=50,
        batch=16,
        imgsz=640,
        run_name="yolov8_model" 
    )
