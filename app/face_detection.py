#!/usr/bin/env python3
"""
This is a class containing methods to detect faces in a
photo or video
"""

# pylint:disable=no-member
import cv2
import os


class FaceDetector:
    """
    Class for detecting faces in an image using OpenCV's Haar cascade.
    """
    def __init__(self, cascade_path):
        """
        Initializes the FaceDetector with the given Haar cascade xml file

        Args:
            cascade_path (str): Path to haar cascade file
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image):
        """
        Detects faces in the given image.

        Args:
            image (numpy.ndarray): The image to detect faces from

        Returns:
            List of tuples: Bounding box coordinates of the detected faces
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        return faces


"""
if __name__ == '__main__':
    img = cv2.imread('group 2.jpg')

    # Get the absolute path to the XML file
    cascade_path = (
        '/home/eq/Documents/Projects_Ex/Face_X_Final/data/haar_face.xml'
    )

    # Initialize the FaceDetector with the absolute file path
    detector = FaceDetector(cascade_path)

    # Detect faces in the image
    detected_faces = detector.detect_faces(img)

    # Output number of detected faces
    print(f'Number of faces found = {len(detected_faces)}')

    # draw rectangle around detected faces
    for (x,y,w,h) in detected_faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
