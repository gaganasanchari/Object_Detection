from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/train/exp/weights/best.pt")  # Update with the correct path to your trained model

# Validate on the validation dataset (defined in data.yaml)
results = model.val()

# Print results summary
print(results)
