import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

# Define a function to convert RGB color to its name
def rgb_to_name(color):
    # Define a dictionary of common colors and their names
    common_colors = {
        (255, 0, 0): 'red',
        (0, 255, 0): 'green',
        (0, 0, 255): 'blue',
        (255, 255, 0): 'yellow',
        (0, 255, 255): 'cyan',
        (255, 0, 255): 'magenta',
        (255, 255, 255): 'white',
        (0, 0, 0): 'black'
    }

    # Get the closest color name from the dictionary
    closest_color = min(common_colors.keys(), key=lambda x: np.linalg.norm(np.array(x) - np.array(color)))
    color_name = common_colors[closest_color]

    return color_name

# Load the YOLOv3 object detection model
net = cv2.dnn.readNet('yolov3.weights', os.path.join(os.getcwd(), 'yolov3.cfg'))

# Load the class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define the colors for the detected objects
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the video capture device
cap = cv2.VideoCapture(0)

# Check if the video capture device is working properly
if not cap.isOpened():
    print('Error: Unable to open video capture device.')
    exit()

# Process the video frame by frame
while True:
    # Capture the current frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print('Error: Unable to capture video frame.')
        break

    # Prepare the frame for the model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Perform object detection
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process the detection results
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x, center_y, w, h = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype('int')
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), colors[class_id], 2)
                cv2.putText(frame, classes[class_id], (center_x - w // 2, center_y - h // 2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_id], 2)

                # Get the dominant color of the detected object
                roi = frame[center_y - h // 2:center_y + h // 2, center_x - w // 2:center_x + w // 2]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.reshape((1, -1, 3))