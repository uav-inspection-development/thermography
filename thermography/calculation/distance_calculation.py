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

    def __init__(self, input_rectangles: list, input_image_shape: tuple, input_image_rtk: list, params=DistanceCalculatorParams()):
        """
        Initializes the DistanceCalculator with the required parameters.

        :param input_rectangles: List of detected rectangles (each rectangle should have a `get_vertex_points` method).
        :param input_image_shape: A tuple `(height, width)` representing the dimensions of the image.
        :param params: Distance calculator parameters to be used for distance calculation.
        """
        # TODO:
        self.rectangles = input_rectangles
        self.image_shape = input_image_shape
        self.image_rtk = input_image_rtk
        self.params = params
        self.rectangle_positions = []
        self.earth_radius = 6378137  # Earth's radius in meters

    def calculate_distances(self) -> list:
        """
        Calculates the distance for each rectangle and logs the results.

        :return: A list of offsets, where each offset is the distance (in multiples of the rectangle length)
                 between the center of the image and the center of a rectangle.
        """
        if not self.rectangles:
            Logger.warning("No rectangles available to calculate offsets.")
            return []

        image_center = np.array([self.image_shape[1] / 2, self.image_shape[0] / 2])  # (x, y)
        lat, lon, alt = self.image_rtk

        for rectangle in self.rectangles:
            # Get the vertex points of the rectangle
            rectangle_vertices = rectangle.get_vertex_points()

            # Calculate the center of the rectangle
            rectangle_center = np.mean(rectangle_vertices, axis=0)

            # Calculate the pixel offset from the image center
            pixel_offset = rectangle_center - image_center  # (dx, dy)

            # Calculate the real-world distance per pixel using the rectangle's real-world length
            # and its pixel length (distance between two opposite vertices)
            rectangle_pixel_length = np.linalg.norm(rectangle_vertices[0] - rectangle_vertices[1])
            meters_per_pixel = self.params.rectangle_length / rectangle_pixel_length

            # Convert pixel offset to real-world distances (in meters)
            dx_meters = pixel_offset[0] * meters_per_pixel
            dy_meters = pixel_offset[1] * meters_per_pixel

            # Adjust latitude and longitude based on the real-world distances
            # Assuming a simple flat-earth approximation for small distances
            
            new_lat = lat + (dy_meters / self.earth_radius) * (180 / np.pi)
            new_lon = lon + (dx_meters / (self.earth_radius * np.cos(np.radians(lat)))) * (180 / np.pi)

            # Altitude remains unchanged (assuming no vertical offset)
            new_alt = alt

            # Store the new position
            self.rectangle_positions.append((new_lat, new_lon, new_alt))

            # Log the new position
            Logger.info(f"New position: Latitude={new_lat:.6f}, Longitude={new_lon:.6f}, Altitude={new_alt:.2f}")
