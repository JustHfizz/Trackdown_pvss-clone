import cv2
from ultralytics import YOLO
import numpy as np

def main():
    model_path = r"C:\Users\SRIKANTH\Downloads\yolo11x.pt"// edit the model path
    print("Loading YOLO model...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # RTSP URLs for each camera
    camera_urls = [
        "rtsp://username:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0",
            ]

    # Initialize VideoCapture objects for each camera
    caps = [cv2.VideoCapture(url, cv2.CAP_FFMPEG) for url in camera_urls]

    # Check if all streams are opened
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Could not open video stream for Camera {i+1}")
            return

    print("All video streams opened successfully.")

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black placeholder if no frame
            else:
                # YOLO inference
                results = model(frame)
                frame = results[0].plot()

            # Resize each frame to a smaller size for grid display (e.g., 640x480 -> 320x240)
            frame = cv2.resize(frame, (400, 240))
            frames.append(frame)

        # Create a grid layout (2x2 for 4 cameras)
        top_row = np.hstack(frames[:2])  # Combine Camera 1 and Camera 2
        bottom_row = np.hstack(frames[2:])  # Combine Camera 3 and Camera 4
        grid = np.vstack([top_row, bottom_row])  # Combine top and bottom rows

        # Display the grid in a single window
        cv2.imshow("Multi-Camera YOLO Detection", grid)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all VideoCapture objects and close windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
