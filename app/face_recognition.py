#!/usr/bin/python3

import cv2
import dlib
import numpy as np
from app.face_alignment import FaceAligner
from app.face_encoding import FaceEncoder
from app.feature_extraction import FeatureExtractor
from flask import request

class FacialRecognizer:
    """
    Class for performing facial recognition using face detection, alignment, and encoding.
    """

    def __init__(self, shape_predictor_path, database):
        """
        Initializes the FacialRecognizer with the provided shape predictor path and Database.

        Args:
            shape_predictor_path (str): The path to the shape predictor file.
            database (Database): The Database instance containing face encodings.
        """
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self.database = database
        self.face_alignment = FaceAligner(shape_predictor_path)
        self.feature_extractor = FeatureExtractor(shape_predictor_path)
        self.face_encoder = FaceEncoder(self.feature_extractor)

    def recognize_faces(self, image):
        """
        Performs face recognition on the given image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            List[str]: A list of recognized person names corresponding to the faces.
        """
        if 'image' not in request.files:
            return 'No file uploaded.', 400

        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        aligned_faces = self.face_alignment.align_face(image)
        encodings = self.face_encoder.encode_faces(aligned_faces)
        recognized_names = []
        for encoding in encodings:
            match_index = self.database.find_match(encoding)
            if match_index is not None:
                recognized_names.append(self.database.get_name(match_index))
            else:
                recognized_names.append("Unknown")
        return recognized_names
