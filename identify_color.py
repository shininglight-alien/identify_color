import cv2
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

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("s"):
                break

    cap.release()

    # Get the dominant color of the scene
    dominant_color = get_dominant_color(frame)

    # Display the color name
    color_name = rgb_to_name(dominant_color)
    cv2.putText(frame, color_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the color name using matplotlib
    import matplotlib.pyplot as plt
    plt.axis('off')
    plt.imshow(frame)
    plt.title(f"Dominant Color: {color_name}")
    plt.show()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()