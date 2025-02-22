from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(
    data="Solar-panel-detection-using-YOLO/dataset.yaml", 
    epochs=50,  
    imgsz=640,  
    batch=4,  
    workers=2,  
    optimizer="SGD",  
    lr0=0.01,  
    momentum=0.9, 
    weight_decay=0.0005,  
    val=True,
    
)