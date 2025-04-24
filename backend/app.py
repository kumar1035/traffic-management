from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from yolov8_version.main_detection import detect_cars  # Import detect_cars function
from ultralytics import YOLO  # Make sure YOLO is imported for model loading
from algo import optimize_traffic

app = Flask(__name__)
CORS(app)

# Ensure 'uploads' folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
weights_path = 'yolov8n.pt'  # Make sure the correct path to the YOLOv8 weights file
model = YOLO(weights_path)

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('videos')
    if len(files) != 4:
        return jsonify({'error': 'Please upload exactly 4 videos'}), 400

    video_paths = []
    for i, file in enumerate(files):
        video_path = os.path.join(UPLOAD_FOLDER, f'video_{i}.mp4')
        file.save(video_path)
        video_paths.append(video_path)

    try:
        # Step 1: Detect cars in each video
        num_cars_list = []
        for video_file in video_paths:
            # Ensure to pass the model to detect_cars function
            result = detect_cars(video_file, model)
            
            # Assuming the result contains the count of cars as the first element of the tuple
            num_cars = result[1]  # This might need to be adjusted based on the output format of detect_cars
            
            print(f"Detected {num_cars} cars in {video_file}")
            num_cars_list.append(int(num_cars))  # Now num_cars is an integer

        # Step 2: Optimize traffic timings based on car counts
        optimized_timings = optimize_traffic(num_cars_list)
        print(f"Optimized timings: {optimized_timings}")

        return jsonify(optimized_timings)
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': 'Server encountered an error'}), 500

if __name__ == '__main__':
    app.run(debug=True)


#python app.py