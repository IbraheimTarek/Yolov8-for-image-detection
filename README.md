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
1. Display COCO class names in a collapsible section (expander)
2. File uploader
3. Button to recognize components
4. Display the result and the confidence level 

## Test Results and comparison
### Test 1
These are the results of yolov8 model on the first image that consists of humans and a lot of objects around them
<img src="https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/test_results/test_1_v3.jpg" alt="Test 1 Image" width="80%"/>

These are the confidence levels:
person: 0.90 person: 0.89 chair: 0.84 person: 0.79 chair: 0.78 person: 0.76 chair: 0.71 chair: 0.67 chair: 0.67 person: 0.55 person: 0.52 person: 0.51 <br/>
These are the results of yolov3 model on the first image
<img src="https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/test_results/test_1_v3.jpg" alt="Test 33 Image" width="80%"/>

These are the confidence levels:
person: 0.99 person: 0.99 chair: 0.95 person: 0.98 backpack: 0.68 person: 0.93 person: 0.63 person: 0.70 person: 0.93 person: 0.77 person: 0.60 diningtable: 0.54 chair: 0.83 chair: 0.98 chair: 0.80 chair: 0.78 cup: 0.82 <br/>
Yolov3 got a better prediction and more objects are been detected as the confidence level is bigger than 0.5

### Test 2
Yolov8 model on the second image that consists of three sheep<br/>
<img src="https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/test_results/test_2_v8.jpg" alt="Test 3333 Image" width="80%"/>

Yolov3 model on the second image that consists of three sheep<br/>
<img src="https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/test_results/test_2_v3.jpg" alt="Test 44444 Image" width="80%"/><br/>

Yolov3 failed in detecting one sheep.

### Test 4
#### Comparison between yoloV3, yoloV8 and CNN
YoloV8 failed in classifying 4 dogs and missed 1 dog on the right: dog: 0.93 dog: 0.80 sheep: 0.77 sheep: 0.70 sheep: 0.67 sheep: 0.57 <br/>
<img src="https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/test_results/test_4_v8.jpg" alt="Test 444 Image" width="80%"/><br/>
YoloV3 succeeded in classifying the 4 dogs and missed 1 dog on the left<br/>
<img src="https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/test_results/test_4_v3.jpg" alt="Test 4444 Image" width="80%"/><br/>
CNN failed<br/>
<img src="https://github.com/IbraheimTarek/Yolov8-for-image-detection/blob/main/test_results/test_4_CNN.jpg" alt="Test 4444 Image" width="80%"/>

### Result
The yolov3 has better predictions and higher accuracy but a very large size.
The yolov8 got good predictions and accuracy, also has a very small size and is very fast.
The Pretrained CNN terribly failed as the images objects have a lot of different views and high context.

# Video

## Contributors
- [Ibraheim Tarek](https://github.com/IbraheimTarek)