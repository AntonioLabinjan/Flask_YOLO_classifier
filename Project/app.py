from flask import Flask, request, redirect, jsonify, render_template
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
        print("Objects detected by YOLO:", class_ids)

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

        return redirect('/')  # Redirect to the homepage after processing the image
    except Exception as e:
        print("Error during classification:", e)
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
