import face_recognition


class FaceDetector:
    """
    Class for detecting faces in an image using the face_recognition library.
    """
    def __init__(self):
        pass

    def detect_faces(self, image_path):
        """
        Detects faces in the given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            List of tuples: Bounding box coordinates of the detected faces.
        """
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        return face_locations

    def detect_faces_in_video(self, video_path):
        """
        Detects faces in a video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List of lists: A list of faces detected in each frame.
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Error: Could not open video file.")

        detected_faces_in_video = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break  # End of the video

            # Detect faces in the current frame
            face_locations = self.detect_faces_in_frame(frame)

            # Append the detected faces to the result list
            detected_faces_in_video.append(face_locations)

        # Release the video capture object
        cap.release()

        return detected_faces_in_video

    def detect_faces_in_frame(self, frame):
        """
        Detects faces in a single frame.

        Args:
            frame (numpy.ndarray): The frame to detect faces from.

        Returns:
            List of tuples: Bounding box coordinates of the detected faces.
        """
        face_locations = face_recognition.face_locations(frame)
        return face_locations
