#!/usr/bin/env python3
"""
This is the plotter class that displays the image back to the user
"""
import cv2
import face_recognition
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from io import BytesIO


class Plotter:
    """
    plotter class methods to display detected faces/recognized faces
    """
    @staticmethod
    def plot_detected_faces(image_path, face_locations):
        """
        Plot detected faces on the image.

        Args:
            image_path (str): The path to the image to plot on.
            face_locations (list): List of tuples containing face locations.

        Returns:
            BytesIO: BytesIO object containing the plotted image.
        """
        # Load the image using face_recognition
        image = face_recognition.load_image_file(image_path)

        # Create a figure and axes for plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)

        for (top, right, bottom, left) in face_locations:
            # Create a rectangle patch for each detected face
            rect = patches.Rectangle(
                (left, top), right - left, bottom - top,
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)

        # Create a BytesIO buffer to save the image
        buf = BytesIO()
        
        # Use FigureCanvasAgg to render the figure to the BytesIO buffer
        canvas = FigureCanvas(fig)
        canvas.print_figure(buf, format='png')
        buf.seek(0)
        plt.close(fig)  # Close the figure to free up resources

        return buf

    @staticmethod
    def display_recognized_faces(
        test_image_path, recognized_results, detected_faces
    ):
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
