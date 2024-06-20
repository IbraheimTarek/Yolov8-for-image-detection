import torch
import cv2
import numpy as np
import sys
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Using the YOLOv8n model for lightweight and faster inference

def predict_components(img_path):
    img = cv2.imread(img_path)
    results = model(img)

    result = []
    for detection in results[0].boxes:
        box = detection.xyxy[0].cpu().numpy()
        confidence = detection.conf[0].cpu().numpy()
        class_id = int(detection.cls[0].cpu().numpy())
        label = model.names[class_id]

        if confidence > 0.5:
            x1, y1, x2, y2 = box
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            result.append({"label": label, "confidence": float(confidence), "box": [x, y, w, h]})

            # Draw bounding box and label with increased font size
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1.6, color, 2)

    return result, img
