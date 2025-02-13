import cv2
import numpy as np

def detect_lines_from_frame(frame):
    """
    Detects white, yellow, and black lines in the frame using color thresholding and Hough Transform.
    Returns a frame with only the detected lines in white on a black background.
    """
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for black, white, and yellow lines
    black_lower = np.array([0, 0, 0], dtype=np.uint8)  # Lower bound for black (Hue: any, Saturation: 0, Value: 0)
    black_upper = np.array([180, 255, 50], dtype=np.uint8)  # Upper bound for black (low value range)

    white_lower = np.array([0, 0, 200], dtype=np.uint8)  # Lower bound for white
    white_upper = np.array([180, 25, 255], dtype=np.uint8)  # Upper bound for white

    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)  # Lower bound for yellow
    yellow_upper = np.array([40, 255, 255], dtype=np.uint8)  # Upper bound for yellow

    # Create masks for black, white, and yellow lines
    black_mask = cv2.inRange(hsv_frame, black_lower, black_upper)
    white_mask = cv2.inRange(hsv_frame, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)

    # Combine the masks for white and yellow lines
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Bitwise-AND the mask with the frame to extract the lines
    result_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(result_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=150)

    # Create an empty black image to draw the detected lines
    line_img = np.zeros_like(frame)

    # Draw lines on the black image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 10)

    return line_img

def main():
    # Open webcam (device index 0 for the first webcam)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Detect lines in the frame
        line_frame = detect_lines_from_frame(frame)

        # Show the result
        cv2.imshow("Detected Lines", line_frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
