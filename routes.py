#!/usr/bin/env python3
"""
routes.py
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from flask import Blueprint, Flask, render_template, request, redirect, url_for, flash, Response
from app.face_training_recognition import FaceRecognizer
from app.face_detection import FaceDetector
from app.plotter import Plotter
from io import BytesIO


app = Flask(__name__)
main_routes = Blueprint('main', __name__)
app.config['UPLOAD_FOLDER'] = 'images/uploads'
known_faces_directory = '/home/eq/Documents/Projects_Ex/Face_X_Final/images/training_dataset'
face_recognizer = FaceRecognizer(known_faces_directory)

# Set threshold for face recognition
threshold = 0.6  # Adjust as needed

# Initialize the FaceRecognizer
input_directory = 'images/training_dataset' # Directory of known faces
recognizer = FaceRecognizer(input_directory, threshold=threshold)
recognizer.load_known_faces()


def generate_image(image):
    """
    Function to generate image bytes from a given image
    """
    _, img_encoded = cv2.imencode('.jpg', image)
    return img_encoded.tobytes()


@main_routes.route('/')
def index():
    """
    Define the route for the home page
    """
    return render_template('index.html')


@main_routes.route('/detect', methods=['POST'])
def detect_faces():
    """
    Route for uploading an image and performing face detection
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

        # Perform face detection using Haar Cascade
        cascade_path = '/home/eq/Documents/Projects_Ex/Face_X_Final/data/haar_face.xml'  # Update with your cascade path
        img = cv2.imread(filename)
        detector = FaceDetector(cascade_path)
        detected_faces = detector.detect_faces(img)

        if len(detected_faces) == 0:
            # Handles the case where no faces are detected
            detected_faces = [(0, 0, 0, 0)]
            return "No faces detected"

        # Create a plot with detected faces
        plot_buf = Plotter.plot_detected_faces(img, detected_faces)

        # Convert the plot to bytes for display
        plot_bytes = plot_buf.getvalue()
        plot_buf.close()

        return Response(plot_bytes, content_type='image/png')

@main_routes.route('/recognize', methods=['POST'])
def recognize_faces():
    """
    Route for recognizing faces in an uploaded image
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

        # Call the recognize_faces_in_image method to recognize faces
        recognized_results, detected_faces = recognizer.recognize_faces_in_image(filename)

        if not recognized_results:
            # Handles the case where no faces are recognized
            recognized_results = ["No faces recognized"]
        
        if len(detected_faces) == 0:
            # Handles the case where no faces are detected
            detected_faces = [(0, 0, 0, 0)]
            return "No faces detected"
        
        # Display the recognized faces using the Plotter class
        plot_buf = Plotter.display_recognized_faces(filename, recognized_results, detected_faces)

        # Convert the plot to bytes for response
        plot_bytes = plot_buf.getvalue()
        plot_buf.close()

        return Response(plot_bytes, content_type='image/png')

# Register the main_routes Blueprint
app.register_blueprint(main_routes)


if __name__ == '__main__':
    app.run(debug=True)
