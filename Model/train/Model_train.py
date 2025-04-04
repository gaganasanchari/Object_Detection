from ultralytics import YOLO

# Load a YOLO model (no pre-trained weights)
model = YOLO("yolov8n.yaml")  # Replace with your custom YAML model config

# Train the model from scratch
results = model.train(data="data.yaml", epochs=50, imgsz=640)
