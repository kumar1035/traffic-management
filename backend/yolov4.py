import cv2 as cv
import time
from collections import deque
import numpy as np
from scipy.signal import find_peaks

def detect_cars(video_file):
    """
    Detect cars in the given video_file using YOLOv4-tiny.
    Returns the mean peak value of car counts over a 30-second window.
    """

    # Confidence and Non-Max Suppression thresholds
    Conf_threshold = 0.4
    NMS_threshold = 0.4

    # Define colors (if you need to visualize bounding boxes)
    COLORS = [
        (0, 255, 0),   (0, 0, 255),   (255, 0, 0),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    # Load class names from file (make sure 'classes.txt' is in the same folder)
    class_name = []
    with open('classes.txt', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    # Load YOLOv4-tiny model (make sure .weights and .cfg are present)
    net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

    # If you have a CUDA-compatible GPU, uncomment these lines:
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    # Open the video file
    cap = cv.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return 0

    starting_time = time.time()
    frame_counter = 0

    # Keep track of car counts over time
    car_counts = deque()  # Each element: (timestamp, car_count)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_counter += 1

        # Detect objects
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

        # Count the number of 'car' detections
        car_count = 0
        for (classid, score, box) in zip(classes, scores, boxes):
            # Adjust to your classes.txt if needed
            if class_name[classid] == "car":
                car_count += 1

        # Current timestamp
        current_time = time.time()
        car_counts.append((current_time, car_count))

        # Remove data older than 30 seconds from the left
        while car_counts and car_counts[0][0] < current_time - 30:
            car_counts.popleft()

    cap.release()

    # Analyze the final deque of car counts
    car_count_values = [count for _, count in car_counts]

    # Find peaks in the car count sequence
    peaks, _ = find_peaks(car_count_values)

    # Calculate mean of peak values
    if len(peaks) > 0:
        mean_peak_value = np.mean([car_count_values[i] for i in peaks])
    else:
        mean_peak_value = 0

    return mean_peak_value
