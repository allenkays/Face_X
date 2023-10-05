#!/usr/bin/env python3
"""
routes.py
"""
import os
from flask import (
    Blueprint, Flask, render_template, request,
    redirect, url_for, flash, Response
)
import matplotlib
import matplotlib.pyplot as plt
from app.face_training_recognition import FaceRecognizer
from app.face_detection import FaceDetector
from app.plotter import Plotter
from io import BytesIO

matplotlib.use('Agg')

app = Flask(__name__)
main_routes = Blueprint('main', __name__)
app.config['UPLOAD_FOLDER'] = './images/uploads'
app.config['SESSION_COOKIE_SAMESITE'] = 'None'

# Set threshold for face recognition
threshold = 0.6  # Adjust as needed

# Initialize the FaceRecognizer
input_directory = 'images/training_dataset'  # Directory of known faces
recognizer = FaceRecognizer(input_directory, threshold=threshold)
recognizer.load_known_faces()


def generate_image(image):
    """
    Function to generate image bytes from a given image
    """
    _, img_encoded = cv2.imencode('.jpg', image)
    return img_encoded.tobytes()


def generate_frames(video_file):
    cap = cv2.VideoCapture(video_file)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@main_routes.route('/video', methods=['GET'])
def video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        video_file = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_file)

        return Response(
            generate_frames(video_file),
            mimetype='multipart/x-mixed-replace;boundary=frame'
        )


@main_routes.route('/')
def index():
    """
    Define the route for the home page
    """
    return render_template('index.html')


@main_routes.route('/detect', methods=['POST'])
def detect_faces():
    """
    Route for uploading an image or video and performing face detection
    """
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Check if the uploaded file is an image or video
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Perform face detection using face_recognition for images
            detector = FaceDetector()  # Create an instance of FaceDetector
            detected_faces = detector.detect_faces(filename)

            if len(detected_faces) == 0:
                # Handles the case where no faces are detected
                detected_faces = [(0, 0, 0, 0)]
                return "No faces detected"

            # Create a plot with detected faces
            plot_buf = Plotter.plot_detected_faces(filename, detected_faces)

            # Convert the plot to bytes for display
            plot_bytes = plot_buf.getvalue()
            plot_buf.close()

            return Response(plot_bytes, content_type='image/png')

        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # Call the recognize_faces_in_video method to get video results
            video_results = recognizer.recognize_faces_in_video(filename)

            # Render the video_results.html template with video_results data
            return render_template(
                'video_results.html',
                video_results=video_results
            )
        else:
            return "Unsupported file format"


@main_routes.route('/recognize', methods=['POST'])
def recognize_faces():
    """
    Route for recognizing faces in an uploaded image or video
    """
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Check if the uploaded file is an image or video
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Call the recognize_faces_in_image method
            recognized_results, detected_faces = (
                recognizer.recognize_faces_in_image(filename)
            )

            if not recognized_results:
                # Handles the case where no faces are recognized
                recognized_results = ["No faces recognized"]

            if len(detected_faces) == 0:
                # Handles the case where no faces are detected
                detected_faces = [(0, 0, 0, 0)]
                return "No faces detected"

            # Display the recognized faces using the Plotter class
            plot_buf = Plotter.display_recognized_faces(
                filename, recognized_results, detected_faces
            )

            # Convert the plot to bytes for response
            plot_bytes = plot_buf.getvalue()
            plot_buf.close()

            return Response(plot_bytes, content_type='image/png')
        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # Call the recognize_faces_in_video method to get video results
            video_results = recognizer.recognize_faces_in_video(filename)

            # Render the video_results.html template with video_results data
            return render_template(
                'video_results.html',
                video_results=video_results
            )
        else:
            return "Unsupported file format"


# Register the main_routes Blueprint
app.register_blueprint(main_routes)

if __name__ == '__main__':
    app.run(debug=True)
