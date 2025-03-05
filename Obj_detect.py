from ultralytics import YOLO
import cv2
import os 
import glob

input_image_folder = "/dataset_Old/images"

output_image_folder ="/images/Obj_Detection/Objs_Identified"
output_boxes_folder ="/images/Obj_Detection/labels_boxes"

# Load pre-trained YOLOv8 model (you can also train your own)
model = YOLO("yolov8n.pt")  # 'yolov8n.pt' is the pre-trained model

# Load images
image_files = glob.glob(os.path.join(input_image_folder, "Imges_*.jpg"))
for image_path in image_files:
	image = cv2.imread(image_path)

	# Run YOLO inference (object detection)
	results = model(image_path)

	# Draw bounding boxes
	for result in results:
    		for box in result.boxes:
        		x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
        		class_id = int(box.cls[0])  # Class ID
        		confidence = box.conf[0].item()  # Confidence score

        		# Get class name from YOLO model
        		class_name = model.names[class_id]

        		# Draw rectangle and label
        		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        		cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

	txt_output_path = os.path.join(output_boxes_folder,"output_"+os.path.basename(image_path.strip('.jpg'))+".txt")

	with open(txt_output_path, "w") as f:
    		for result in results:
        		for box in result.boxes:
            			class_id = int(box.cls)  # Class ID
            			x_center, y_center, width, height = box.xywhn.tolist()[0]  # Normalized coordinates
            			confidence = box.conf.tolist()[0]  # Confidence score

            			# Write results in YOLO format: class_id x_center y_center width height confidence
            			f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")

	output_image_path = os.path.join(output_image_folder, f"detected_{os.path.basename(image_path)}")
	cv2.imwrite(output_image_path,image)

	print(f"Detection results saved to {txt_output_path}")
