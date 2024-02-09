import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(-1, 3)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, colors = cv2.kmeans(image, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return colors[0]

def rgb_to_name(rgb):
    name = None
    r, g, b = map(int, rgb)
    rgb_tuple = (r / 255.0, g / 255.0, b / 255.0)
    closest_name = sorted(colors.CSS4_COLORS.items(), key=lambda x: np.linalg.norm(np.array(x[1]) - rgb_tuple))[0][0]

    return closest_name

def scanner():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Scanner", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            # Convert the image to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the range of the color you want to detect
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([140, 255, 255])

            # Threshold the image to get only the pixels within the defined color range
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Find the contours of the object in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Get the bounding box of the largest contour
                x, y, w, h = cv2.boundingRect(contours[0])

                # Crop the ROI around the object
                roi = frame[y:y+h, x:x+w]

                # Get the dominant color of the ROI
                dominant_color = get_dominant_color(roi)

                # Display the color name
                color_name = rgb_to_name(dominant_color)
                cv2.putText(roi, color_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the ROI using matplotlib
                plt.axis('off')
                plt.imshow(roi)
                plt.title(f"Dominant Color: {color_name}")
                plt.show()

def close_camera():
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    scanner()
    close_camera()