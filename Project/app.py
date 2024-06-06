from flask import Flask, request, redirect, jsonify, render_template, Response
import cv2
import numpy as np
import pymysql
from datetime import datetime
import os

app = Flask(__name__)

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'yolo_db'

# Initialize MySQL
db = pymysql.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB']
)
cursor = db.cursor()

# Path to YOLO configuration and weights files
yolo_folder = "yolo-coco"  # Folder name where YOLO files are stored
yolo_cfg_path = os.path.join(os.path.dirname(__file__), yolo_folder, 'yolov3.cfg')
yolo_weights_path = os.path.join(os.path.dirname(__file__), yolo_folder, 'yolov3.weights')

# YOLO setup
net = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[layer_id - 1] for layer_id in unconnected_layers]
classes = []
with open(os.path.join(os.path.dirname(__file__), yolo_folder, "coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

@app.route('/', methods=['GET'])
def index():
    try:
        cursor.execute("SELECT class, SUM(number_of_instances) FROM objects GROUP BY class")
        results = cursor.fetchall()
        print("Results from database:", results)
        return render_template('index.html', results=results)
    except Exception as e:
        print("Error fetching data from database:", e)
        return render_template('error.html', error=str(e))

import base64

@app.route('/classify', methods=['POST'])
def classify():
    try:
        file = request.files['image']
        print("File received:", file)
        
        # Read the image data
        image_data = file.read()
        
        # Convert the image data to a numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode the image using OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print("Image decoded")

        # YOLO object detection
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        print("YOLO object detection performed")

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print("Non-maximum suppression performed")

        for i in range(len(boxes)):
            if i in indexes:
                class_name = classes[class_ids[i]]
                cursor.execute("SELECT * FROM objects WHERE class=%s", (class_name,))
                record = cursor.fetchone()
                if record:
                    cursor.execute("UPDATE objects SET number_of_instances = number_of_instances + 1, time_of_adding = %s WHERE class = %s",
                                   (datetime.now(), class_name))
                    print("Updated existing object in database:", class_name)
                else:
                    cursor.execute("INSERT INTO objects (class, number_of_instances) VALUES (%s, %s)",
                                   (class_name, 1))
                    print("Inserted new object into database:", class_name)
                db.commit()
        
        # Draw bounding boxes on the image outside the loop
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                class_name = classes[class_ids[i]]
                color = (0, 255, 0)  # Green color for the bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert the modified image to base64 for displaying in HTML
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return render_template('result.html', img_base64=img_base64)  # Render template with image
    except Exception as e:
        print("Error during classification:", e)
        return render_template('error.html', error=str(e))

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # YOLO object detection
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    class_name = classes[class_ids[i]]
                    color = (0, 255, 0)  # Green color for the bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_feed')
def live_feed():
    return render_template('live_feed.html')

if __name__ == '__main__':
    app.run(debug=True)
