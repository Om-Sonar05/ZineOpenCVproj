import cv2
import numpy as np

def detect_shape_color_size(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for different shapes
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])

    green_lower = np.array([40, 50, 50])
    green_upper = np.array([70, 255, 255])

    blue_lower = np.array([80, 50, 50])
    blue_upper = np.array([130, 255, 255])

    # Detect colors in the image
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

    # Find contours of the detected colors
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect shapes and print them out
    for contour in red_contours:
        shape = 'Circle' if cv2.isContourConvex(contour) else 'Rectangle'
        color = 'Red'
        size = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, f'Shape: {shape}, Color: {color}, Size: {size}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for contour in green_contours:
        shape = 'Circle' if cv2.isContourConvex(contour) else 'Rectangle'
        color = 'Green'
        size = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'Shape: {shape}, Color: {color}, Size: {size}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for contour in blue_contours:
        shape = 'Circle' if cv2.isContourConvex(contour) else 'Rectangle'
        color = 'Blue'
        size = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, f'Shape: {shape}, Color: {color}, Size: {size}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def main():
    # Create a camera object
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect shape, color, and size
        detect_shape_color_size(frame)

        # Display the resulting frame
        cv2.imshow('Color Detection', frame)

        # Exit the program if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
