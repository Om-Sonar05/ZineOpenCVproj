import cv2
import numpy as np

def get_contour_info(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    area = cv2.contourArea(contour)
    return len(approx), area

def detect_color(frame, contour):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # Ensure the mask is of type CV_8U
    cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)  # Use thickness=cv2.FILLED to fill the contour
    mean_color = cv2.mean(frame, mask=mask)
    return mean_color[:3]

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 0:
                sides, area = get_contour_info(contour)

                if area > 0:
                    color = detect_color(frame, contour)

                    if sides == 3:
                        shape = "Triangle"
                    elif sides == 4:
                        # Check for squares and rectangles based on aspect ratio
                        _, _, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h

                        if 0.95 <= aspect_ratio <= 1.05:
                            shape = "Square"
                        else:
                            shape = "Rectangle"
                    elif sides > 4:
                        # Detect circles using the minimum enclosing circle
                        (_, _), radius = cv2.minEnclosingCircle(contour)
                        if radius > 10:  # Adjust this threshold based on your requirements
                            shape = "Circle"
                        else:
                            shape = "Unknown"
                    else:
                        shape = "Unknown"

                    size = round(np.sqrt(area), 2)

                    if size > 0.00:
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                        x, y, _, _ = cv2.boundingRect(contour)
                        cv2.putText(frame, f"{shape}, Color: {color}, Size: {size}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()