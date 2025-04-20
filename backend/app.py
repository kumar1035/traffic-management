# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# from yolov4 import detect_cars  # We'll use the detect_cars function from yolov4.py

# app = Flask(__name__)
# CORS(app)

# @app.route('/')
# def home():
#     return "Flask app is running!"

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     # Expecting exactly 4 videos in the 'videos' form-data field
#     files = request.files.getlist('videos')
#     if len(files) != 4:
#         return jsonify({'error': 'Please upload exactly 4 videos'}), 400

#     # Save uploaded videos to 'uploads' folder
#     video_paths = []
#     for i, file in enumerate(files):
#         video_path = os.path.join('uploads', f'video_{i}.mp4')
#         file.save(video_path)
#         video_paths.append(video_path)

#     # Detect cars in each video
#     num_cars_list = []
#     for video_file in video_paths:
#         num_cars = detect_cars(video_file)
#         num_cars_list.append(num_cars)

#     # Instead of calling `optimize_traffic`, just return the detected counts
#     return jsonify({'car_counts': num_cars_list})

# if __name__ == '__main__':
#     # Ensure 'uploads' folder exists
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from yolov4 import detect_cars
from algo import optimize_traffic

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
            num_cars = detect_cars(video_file)
            print(f"Detected {num_cars} cars in {video_file}")
            num_cars_list.append(int(num_cars))

        # Step 2: Optimize traffic timings based on car counts
        optimized_timings = optimize_traffic(num_cars_list)
        print(f"Optimized timings: {optimized_timings}")

        return jsonify(optimized_timings)
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': 'Server encountered an error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
