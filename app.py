from flask import Flask, request, jsonify
from yolo_model import predict_components
import os
import cv2

app = Flask(__name__)

upload_folder = '/home/ibraheimtarek/Yolov8-for-image-detection/uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    try:
        predictions, img = predict_components(file_path)

        # Save the processed image with bounding boxes
        output_path = os.path.join(upload_folder, "output_" + file.filename)
        cv2.imwrite(output_path, img)

        # Convert predictions to a format suitable for JSON response
        components = [{"label": pred["label"], "confidence": pred["confidence"]} for pred in predictions]

        return jsonify({"components": components, "image_path": output_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(file_path)

# Comment out or remove the app.run() line
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)
