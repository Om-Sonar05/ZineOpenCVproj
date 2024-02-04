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

# Define object-specific variables
dist = 0
focal = 515
pixels = 39.67
width = 5.4

# Basic constants for OpenCV functions
kernel = np.ones((3, 3), 'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
org = (0, 20)
fontScale = 0.6
thickness = 2

# Function to find the distance from the camera for red objects
def get_dist(rectangle_params, image, color):
    # Unpack the tuple with two values
    (center, dimensions, angle) = rectangle_params
    # Find the number of pixels covered
    pixels = dimensions[0]
    print(pixels)
    # Calculate distance only for red objects
    if color == "Red":
        dist = (width * focal) / pixels
        # Convert distance to string
        dist_str = f'{dist:.2f}'
        # Write on the image
        image = cv2.putText(image, f'Distance of red square from cam in cm: {dist_str}', org, font, fontScale, (0, 0, 255), thickness, cv2.LINE_AA)

    return image

# Extract Frames 
cap = cv2.VideoCapture(0)

cv2.namedWindow('Object Detection and Distance Measure', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detection and Distance Measure', 1366, 768)

# Loop to capture video frames
while True:
    ret, img = cap.read()

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Predefined mask for red, green, and blue color detection
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv_img, lower_red, upper_red)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    lower_blue = np.array([100, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # Combine masks for red, green, and blue
    combined_mask = mask_red + mask_green + mask_blue

    # Find contours using the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 700:
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y+h, x:x+w]

            dominant_color = get_dominant_color(roi)
            color = color_name(dominant_color)
            cv2.putText(img, f"Color: {color}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            shape, size = get_shape_and_size(contour)
            cv2.putText(img, f"Shape: {shape}, Size: {size:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get object distance and display it on the image
            rect = cv2.minAreaRect(contour)
            img = get_dist(rect, img, color)

    cv2.imshow('Object Detection and Distance Measure', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
