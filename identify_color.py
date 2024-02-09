import cv2
import numpy as np

# Load the SSD object detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Load the class names
classes = None
with open('coco.names', 'rt') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the input image
image = cv2.imread('input.jpg')

# Preprocess the input image
height, width, _ = image.shape
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

# Set the input blob for the model
net.setInput(blob)

# Perform object detection
outputs = net.forward('detection_out')

# Postprocess the detection results
detections = []
for output in outputs:
    for detection in output:
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
            detections.append((class_id, confidence, x, y, w, h))

# Filter out detections with low confidence
detections = [detection for detection in detections if detection[1] > 0.5]

# Sort detections by confidence
detections = sorted(detections, key=lambda x: x[1], reverse=True)

# Draw bounding boxes around detected objects
for class_id, confidence, x, y, w, h in detections:
    color = (0, 255, 0)
    label = f'{classes[class_id]} {confidence:.2f}'
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Display the output image
cv2.imshow('Output', image)
cv2.waitKey(0)
cv2.destroyAllWindows()