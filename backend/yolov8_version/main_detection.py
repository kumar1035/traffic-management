import cv2
import time
import os
from ultralytics import YOLO

def detect_cars(frame, model):
    """
    Detect cars in a frame using YOLOv8 model.
    """
    results = model.predict(source=frame, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()  # draw boxes and labels
    car_count = len(results[0].boxes)  # Count detected cars
    return annotated_frame, car_count

def process_video(video_path, model, output_video_path='OutputVideo_YOLOv8.avi'):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Output video setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 4)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 4)
    output_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, output_size)

    # Start detection loop
    start_time = time.time()
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, output_size)
        frame_counter += 1

        # Run YOLOv8 inference
        annotated_frame, car_count = detect_cars(frame, model)

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_counter / elapsed_time

        # Display FPS and car count
        cv2.rectangle(annotated_frame, (18, 20), (300, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f'FPS: {round(fps, 2)}', (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f'Cars: {car_count}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("YOLOv8 Detection", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Detection complete, video saved as", output_video_path)

if __name__ == "__main__":
    # Ensure YOLOv8 weights are present
    weights_path = 'yolov8n.pt'  # Ensure the correct YOLOv8 weights file path
    model = YOLO(weights_path)

    # Set video input and output paths
    video_path = 'pexels-alex-pelsh-6896028.mp4'
    output_video_path = 'OutputVideo_YOLOv8.avi'

    # Process the video
    process_video(video_path, model, output_video_path)
