import cv2
import numpy as np

def get_dominant_color(roi):
    pixels = roi.reshape((-1, 3))
    k = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, _, dominant_color = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = np.uint8(dominant_color[0])
    return dominant_color

def color_name(rgb_color):
    color_names = {
        "Red": np.array([0, 0, 255]),
        "Green": np.array([0, 255, 0]),
        "Blue": np.array([255, 0, 0])
    }

    min_distance = float('inf')
    detected_color = "Unknown"

    for name, known_color in color_names.items():
        distance = np.linalg.norm(rgb_color - known_color)
        if distance < min_distance:
            min_distance = distance
            detected_color = name

    return detected_color

def detect_objects(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([80, 100, 100])
    upper_blue = np.array([130, 255, 255])

    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
    mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    combined_mask = mask_red + mask_green + mask_blue

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 700:
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]

            dominant_color = get_dominant_color(roi)
            color = color_name(dominant_color)
            cv2.putText(frame, f"Color: {color}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            shape, size = get_shape_and_size(contour)
            cv2.putText(frame, f"Shape: {shape}, Size: {size:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def get_shape_and_size(contour):
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    num_sides = len(approx)

    if num_sides == 4:
        aspect_ratio = float(cv2.boundingRect(approx)[2]) / cv2.boundingRect(approx)[3]
        shape = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    else:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity > 0.7:
            shape = "Circle"
        else:
            shape = "Unknown"

    area = cv2.contourArea(contour)

    return shape, area

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture video.")
            break

        detect_objects(frame)
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
