import cv2
import numpy as np
from simple_logger import Logger


__all__ = ["DistanceCalculator", "DistanceCalculatorParams"]


class DistanceCalculatorParams:
    """Parameters used by the :class:`.DistanceCalculator`."""

    def __init__(self):
        """Initializes the rectangle detector parameters to their default value.

        :ivar rectangle_length: Length of the rectangle used for distance calculation.
        """
        self.rectangle_length = 50

class DistanceCalculator:
    """Class responsible for calculating the offset distance between the center of the image
    and the center of detected rectangles."""

    def __init__(self, input_rectangles: list, input_image_shape: tuple, params=DistanceCalculatorParams()):
        """
        Initializes the DistanceCalculator with the required parameters.

        :param input_rectangles: List of detected rectangles (each rectangle should have a `get_vertex_points` method).
        :param input_image_shape: A tuple `(height, width)` representing the dimensions of the image.
        :param params: Distance calculator parameters to be used for distance calculation.
        """
        self.rectangles = input_rectangles
        self.image_shape = input_image_shape
        self.params = params
        self.offsets = []

    def calculate_offsets(self) -> list:
        """
        Calculates the offset for each rectangle and logs the results.

        :return: A list of offsets, where each offset is the distance (in multiples of the rectangle length)
                 between the center of the image and the center of a rectangle.
        """
        if not self.rectangles:
            Logger.warning("No rectangles available to calculate offsets.")
            return []

        image_center = np.array([self.image_shape[1] / 2, self.image_shape[0] / 2])  # (x, y)

        for rectangle in self.rectangles:
            # Get the vertex points of the rectangle
            rectangle_vertices = rectangle.get_vertex_points()

            # Calculate the center of the rectangle
            rectangle_center = np.mean(rectangle_vertices, axis=0)

            # Calculate the Euclidean distance between the rectangle center and the image center
            distance = np.linalg.norm(rectangle_center - image_center)

            # Calculate the offset as a multiple of the rectangle length
            offset = distance / self.params.rectangle_length
            self.offsets.append(offset)

            # Log the offset
            Logger.info(f"Offset for rectangle: {offset:.2f} times the basic rectangle length")
