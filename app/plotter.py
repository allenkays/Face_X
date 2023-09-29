#!/usr/bin/env python3
"""
This is the plotter class that displays the image back to the user
"""
import cv2
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO


class Plotter:
    """
    plotter class containing a methods to display detected faces, recognized faces
    """
    @staticmethod
    def plot_detected_faces(image, detected_faces):
        """
        Method to display detected faces

        Args:
            image (str): Path to image
            detected_faces (tuple): Tuple showing facial coordinates

        Returns:
            buf (BytesIO object)
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        ax = plt.gca()

        for (x, y, w, h) in detected_faces:
            # Draw a rectangle around detected faces
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none'))

        ax.axis('off')

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')  # Specify 'bbox_inches' to avoid saving extra whitespace
        buf.seek(0)
        return buf


    @staticmethod
    def display_recognized_faces(test_image_path, recognized_results, detected_faces):
        """
        Method to display recognized faces

        Args:
            test_image_path (str): Path to the test image
            recognized_results (list): List containing faces 
            detected_faces (tuple): Tuple of coordinates for bounding box
        """
        if not test_image_path:
            raise ValueError("test_image_path is not provided.")
        
        if not recognized_results:
            raise ValueError("recognized_results is empty.")
        
        if not detected_faces:
            raise ValueError("detected_faces is empty.")

        test_image = face_recognition.load_image_file(test_image_path)

        plt.figure(figsize=(8, 6))
        plt.imshow(test_image)

        ax = plt.gca()

        for face_location, result in zip(detected_faces, recognized_results):
            top, right, bottom, left = face_location

            rect_border = patches.Rectangle(
                (left, top),
                right - left,
                bottom - top,
                linewidth=2,
                edgecolor='g',
                facecolor='none'
            )
            ax.add_patch(rect_border)

            ax.annotate(
                result,
                xy=(left, bottom + 25),
                color='w',
                fontsize=8,
                weight='bold'
            )

        plt.axis('off')

        # Save the plot to a BytesIO buffer
        plot_buf = BytesIO()
        plt.savefig(plot_buf, format='png')
        plot_buf.seek(0)

        return plot_buf
