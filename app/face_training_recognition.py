#!/usr/bin/env python3
"""
This module trains and does facial recognition
"""
import os
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FaceRecognizer:
    """
    This is the facial recognition class
    """
    def __init__(self, input_dir, threshold=0.6):
        """
        Facial recognition constructor

        Args:
            input_dir (str): Path to training folder
            threshold (float): Facial recognition threshold
        """
        self.input_dir = input_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.threshold = threshold  # Adjust the threshold as needed

    def load_known_faces(self):
        """
        List only immediate subdirectories (children) of the input directory
        These are the directories containing known faces grouped per directory
        """
        subdirectories = [
            d for d in os.listdir(
                self.input_dir
            )
            if os.path.isdir(os.path.join(self.input_dir, d))]

        for person in subdirectories:
            person_dir = os.path.join(self.input_dir, person)

            # Load and encode the known faces in the directory
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(image_path)

                # Detect faces in the image
                face_locations = face_recognition.face_locations(image)

                # Check if at least one face is detected before encoding
                if len(face_locations) > 0:
                    face_encoding = face_recognition.face_encodings(image)[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(person)

                    # Print encoding for debugging
                    # print(f"Encoded face for {person}: {face_encoding}")

    def recognize_faces_in_image(self, test_image_path):
        """
        Facial recognition

        Args:
            image_path (str): Path to image to be checked

        Returns:
            results (list): List containing facial coordinates
        """
        # Load the test image
        test_image = face_recognition.load_image_file(test_image_path)

        # Find all face locations and face encodings in the test image
        face_locations = face_recognition.face_locations(test_image)

        if not face_locations:
            # No faces detected in the image, return an empty result
            return [], []

        face_encodings = face_recognition.face_encodings(
            test_image, face_locations
        )
        # Initialize a lists to store results & detected face locations
        recognized_results = []
        detected_faces = []

        # Compare face encodings in the test image with known face encodings
        for face_encoding, face_location in zip(
            face_encodings, face_locations
        ):
            # Calculate the face distance (similarity) to known faces
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )

            if len(face_distances) == 0:
                # No known faces to compare with;
                # set a default value for min_distance
                min_distance = 1.0
            else:
                # Find the closest match (smallest distance)
                min_distance = min(face_distances)

            # Calculate the confidence level as a percentage (0-100)
            confidence_level = (1 - min_distance) * 100

            if min_distance < self.threshold:
                # If below the threshold, consider it a match and get the name
                index = list(face_distances).index(min_distance)
                name = self.known_face_names[index]
                recognized_result = (
                    f"Recognized: {name}\
                    \nLevel of Confidence: {confidence_level:.2f}%"
                )
            else:
                recognized_result = "Unknown"

            recognized_results.append(recognized_result)
            detected_faces.append(face_location)

        return recognized_results, detected_faces

    def recognize_faces_in_video(self, video_path):
        """
        Facial recognition in video

        Args:
            video_path (str): Path to the video file

        Returns:
            results (list): List of facial recognition results for each frame
        """
        # Open the video file
        video_capture = cv2.VideoCapture(video_path)

        results = []

        while True:
            # Read a single frame from the video
            ret, frame = video_capture.read()

            if not ret:
                break

            # Convert the frame to RGB (required for face_recognition library)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_frame, face_locations
            )

            recognized_results = []

            for face_encoding, face_location in zip(
                face_encodings, face_locations
            ):
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )

                if len(face_distances) == 0:
                    min_distance = 1.0
                else:
                    min_distance = min(face_distances)

                confidence_level = (1 - min_distance) * 100

                if min_distance < self.threshold:
                    index = list(face_distances).index(min_distance)
                    name = self.known_face_names[index]
                    recognized_result = (
                        f"Recognized: {name}\
                        \nLevel of Confidence: {confidence_level:.2f}%"
                    )
                else:
                    recognized_result = "Unknown"

                recognized_results.append(recognized_result)

            results.append(recognized_results)

        # Release the video capture object
        video_capture.release()

        return results


'''
if __name__ == '__main__':
    input_directory = 'images/training_dataset'
    recognizer = FaceRecognizer(input_directory)

    # Load known faces from the provided directory
    recognizer.load_known_faces()

    # Test image for recognition
    test_image_path = 'images/uploads/jerry_seinfeld/1.jpg'
    # Recognize faces in the test image
    recognized_results = recognizer.recognize_faces_in_image(test_image_path)

    # Display the recognized faces in a new window
    recognizer.display_recognized_faces(test_image_path)

    # Print the recognized results
    for result in recognized_results:
        print(result)
'''
