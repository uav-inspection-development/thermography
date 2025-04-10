import numpy as np
from simple_logger import Logger

from thermography.utils.geometry import aspect_ratio, area, sort_rectangle

__all__ = ["RectangleDetector", "RectangleDetectorParams"]


class RectangleDetectorParams:
    """Parameters used by the :class:`.RectangleDetector`."""

    def __init__(self):
        """Initializes the rectangle detector parameters to their default value.

        :ivar aspect_rato: Expected rectangle aspect ratio.
        :ivar aspect_ratio_relative_deviation: Detected rectangles whose aspect ratio deviates from :attr:`self.aspect_ratio` more than this parameter are ignored.
        :ivar min_area: Minimal surface of detected rectangles. Smaller rectangles are rejected.
        """
        self.aspect_ratio = 1.5
        self.aspect_ratio_relative_deviation = 0.35

        self.min_area = 20 * 40


class RectangleDetector:
    """Class responsible for detecting rectangles given a structured intersection list."""

    def __init__(self, input_intersections: dict, params: RectangleDetectorParams = RectangleDetectorParams()):
        """Initializes the rectangle detector with the input intersections and the rectangle detection parameters."""
        self.intersections = input_intersections
        self.params = params

        self.rectangles = []

    def detect(self) -> None:
        """Detects the rectangles from the input intersections.
        """
        Logger.debug("Detecting rectangles")
        # Iterate over each pair of clusters.
        num_clusters = int((np.sqrt(8 * len(self.intersections) + 1) + 1) / 2)
        for cluster_index_i in range(num_clusters):
            for cluster_index_j in range(cluster_index_i + 1, num_clusters):
                if (cluster_index_i, cluster_index_j) in self.intersections:
                    Logger.debug("Detecting rectangles between cluster {} and cluster {}".format(cluster_index_i,
                                                                                                 cluster_index_j))
                    self.__detect_rectangles_between_clusters(cluster_index_i, cluster_index_j)

    @staticmethod
    def fulfills_ratio(rectangle: np.ndarray, expected_ratio: float, deviation: float) -> bool:
        """Computes wether a rectangle defined as a set of four coordinates fulfills a predefined aspect ratio within a maximal deviation.

        :param rectangle: Rectangle to be tested defined as a set of four pixel coordinates as a numpy array of shape `[4,2]`.
        :param expected_ratio: Expected aspect ratio of the rectangle.
        :param deviation: Maximal deviation between the query rectangle and the :attr:`expected_ratio` in order to accept or not the ratio test.
        :return: A boolean set to True if the aspect relative deviation between its aspect ratio and the :attr:`expected_ratio` is smaller than the :attr:`deviation` threshold.
        """
        ratio = aspect_ratio(rectangle)

        if np.abs(expected_ratio - ratio) / expected_ratio < deviation:
            return True
        if np.abs(expected_ratio - 1.0 / ratio) / expected_ratio < deviation:
            return True
        return False

    def calculate_rectangle_dimensions(rectangle: np.ndarray) -> tuple:
        """
        Calculates the row length and column length of a rectangle.

        :param rectangle: A numpy array of shape `[4, 2]` representing the four corners of the rectangle.
                        The corners should be ordered as [top-left, top-right, bottom-right, bottom-left].
        :return: A tuple `(row_length, col_length)` where:
                - `row_length` is the horizontal distance (width) of the rectangle.
                - `col_length` is the vertical distance (height) of the rectangle.
        """
        # Calculate the row length (distance between top-left and top-right)
        row_length = np.linalg.norm(rectangle[0] - rectangle[1])

        # Calculate the column length (distance between top-left and bottom-left)
        col_length = np.linalg.norm(rectangle[0] - rectangle[3])

        return row_length, col_length

    def get_average_dimensions(self) -> tuple:
        """
        Calculates and returns the average row length and column length of the detected rectangles.

        :return: A tuple `(avg_row_length, avg_col_length)` where:
                - `avg_row_length` is the average horizontal distance (width) of the rectangles.
                - `avg_col_length` is the average vertical distance (height) of the rectangles.
                If no rectangles are detected, returns `(0.0, 0.0)`.
        """
        if not self.rectangles:
            Logger.warning("No rectangles detected to calculate average dimensions.")
            return 0.0, 0.0

        total_row_length = 0.0
        total_col_length = 0.0

        for rectangle in self.rectangles:
            row_length, col_length = self.calculate_rectangle_dimensions(rectangle)
            total_row_length += row_length
            total_col_length += col_length

        avg_row_length = total_row_length / len(self.rectangles)
        avg_col_length = total_col_length / len(self.rectangles)

        Logger.info(f"Average row length: {avg_row_length:.2f}, Average col length: {avg_col_length:.2f}")
        return avg_row_length, avg_col_length

    def __detect_rectangles_between_clusters(self, cluster_index_i: int, cluster_index_j: int):
        intersections_i_j = self.intersections[cluster_index_i, cluster_index_j]
        rectangles_between_cluster_i_j = []
        # Iterate over all segments in cluster i, and all intersections between that segment and cluster j.
        for segment_index_i, intersections_with_i in intersections_i_j.items():
            if segment_index_i + 1 not in intersections_i_j:
                continue

            intersections_with_i_plus = intersections_i_j[segment_index_i + 1]
            for segment_index_j, intersection in intersections_with_i.items():
                if segment_index_j + 1 not in intersections_with_i:
                    continue
                if segment_index_j in intersections_with_i_plus and segment_index_j + 1 in intersections_with_i_plus:
                    coord1 = intersections_with_i[segment_index_j]
                    coord2 = intersections_with_i[segment_index_j + 1]
                    coord3 = intersections_with_i_plus[segment_index_j]
                    coord4 = intersections_with_i_plus[segment_index_j + 1]
                    rectangle = np.array([coord1, coord2, coord4, coord3])
                    rectangle = sort_rectangle(rectangle)
                    if self.fulfills_ratio(rectangle, self.params.aspect_ratio,
                                           self.params.aspect_ratio_relative_deviation) and \
                                    area(rectangle) >= self.params.min_area:
                        rectangles_between_cluster_i_j.append(rectangle)

        self.rectangles.extend(rectangles_between_cluster_i_j)
