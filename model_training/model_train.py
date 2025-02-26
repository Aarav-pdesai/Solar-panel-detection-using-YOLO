from ultralytics import YOLO
import torch

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = model.train(
        data="C:/Users/aarav/OneDrive/Desktop/IIT_GN_TASK/Solar-panel-detection-using-YOLO/dataset.yaml", 
        epochs=50,
        imgsz=640, 
        device=device  
    )
