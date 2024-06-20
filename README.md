# Yolov8-for-object-recognation-app

## Overview
a Stream-lit application that uses Yolov8 pretrained model to perform object detection on an input image, filters the detections by confidence, draws bounding boxes and labels on the image, and returns the detection results along with the annotated image.

## YOLOv8 Model additions
### 1-Loading the YOLOv8 Model:
### 2-Processing the Detection Results:
```
result = []
for detection in results[0].boxes:
    box = detection.xyxy[0].cpu().numpy()
    confidence = detection.conf[0].cpu().numpy()
    class_id = int(detection.cls[0].cpu().numpy())
    label = model.names[class_id]
```
### 3-Filtering Detections by Confidence:
```
if confidence > 0.5:
    ...
    result.append({"label": label, "confidence": float(confidence), "box": [x, y, w, h]})
```
Filters out detections with a confidence score less than 0.5.
Converts the bounding box coordinates to integers and calculates the width and height.
### Drawing Bounding Boxes and Labels on the Image using OpenCV:
```
color = (0, 255, 0)
cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1.6, color, 2)
```
Returns the list of detected components (result) and the modified image (img) with drawn bounding boxes and labels.
## Flask application to run the model prediction
definning a post method to use the predict_components method after sending the image to the model
## Streamlit UI
the streamlit ui consists of the following components
### 1-Display COCO class names in a collapsible section (expander)
### 2-File uploader
### 3-Button to recognize components
### 4-Display the result and the confidence level

## Test Results and comparison
### Test 1
these are the results of yolov8 model on the first image that consists of humans and alot of object around them
![Test 1 Image](https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/results/test_1_v8.jpg)
these are the confidence levels
![Test 2 Image](https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/results/test_1_v8_photo2.jpg)
these are the results of yolov3 model on the first image
![Test 33 Image](https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/results/test_1_v3.jpg)
these are the confidence levels
![Test 333 Image](https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/results/test_1_v3_photo2.jpg)
Yolov3 got a better prediction and more objects are been detected
### Test 2
Yolov8 model on the second image that consists of three sheep
![Test 3333 Image](https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/results/test_2_v8.jpg)
Yolov3 model on the second image that consists of three sheep
![Test 44444 Image](https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/results/test_2_v3.jpg)
Yolov3 got a better prediction and the three sheep are detected
### Test 4
#### comparison between yoloV3, yoloV8 and CNN
YoloV8 failed in classifying 4 dogs and missed 1 dog on the right
![Test 444 Image](https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/results/test_4_v8.jpg)
YoloV3 succeeded in classifying the 4 dogs and missed 1 dog on the left
![Test 4444 Image](https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/results/test_4_v3.jpg)
CNN failed
![Test 4444 Image](https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/results/test_4_CNN.jpg)
### Result
the yolov3 has a better predictions and high accuracy but very large size
the yolov8 got a good predicions and accuracy also has a very small size and very fast
the Pretrained CNN terribly failed as the images objects have alot of different views and high context 
# Vidoe

## Contributors
- [Ibraheim Tarek](https://github.com/IbraheimTarek)
