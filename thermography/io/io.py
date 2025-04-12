import os
import exifread
import cv2
import progressbar
from simple_logger import Logger
from typing import Optional

from . import Modality

__all__ = ["ImageLoader", "VideoLoader"]


class ImageLoader:
    """Class responsible for loading a single image file into a numpy array."""

    def __init__(self, image_folder_path: str, mode: Modality = Modality.DEFAULT):
        """Initializes and loads the image associated to the file indicated by the path passed as argument.

        :param image_folder_path: Absolute path to the folder containing the images to be loaded.
        :param mode: Modality to be used when loading the image.
        """
        Logger.debug("Loading images from folder: {}".format(image_folder_path))
        self.image_folder_path = image_folder_path
        self.mode = mode
        self.frames = []  # List to store loaded images
        self.image_paths = []  # List to store paths of the loaded images
        self.rtk_data_list = []  # List to store RTK data for each image
        self.__load_images()

    @property
    def num_frames(self) -> int:
        """Returns the number of images loaded by this object."""
        return len(self.frames)

    @property
    def image_folder_path(self) -> str:
        """Returns the absolute path to the folder containing the images."""
        return self.__image_folder_path

    @image_folder_path.setter
    def image_folder_path(self, folder: str):
        if not os.path.isdir(folder):
            Logger.fatal("Image folder {} does not exist".format(folder))
            raise FileNotFoundError("Image folder {} not found".format(folder))
        self.__image_folder_path = folder

    def show_image(self, index: int, title: str = "", wait: int = 0) -> None:
        """
        Displays the image at the specified index.

        :param index: Index of the image to display.
        :param title: Title to be added to the displayed image.
        :param wait: Time to wait until the displayed window is closed. If set to 0, the image does not close.
        """
        if index < 0 or index >= len(self.frames):
            Logger.error("Index out of range: {}".format(index))
            raise IndexError("Index out of range: {}".format(index))

        cv2.imshow(title + " (image)" if len(title) > 0 else "", self.frames[index])
        cv2.waitKey(wait)

    @property
    def image_path(self) -> str:
        """Returns the absolute path to the image loaded by this object."""
        return self.__image_path

    @image_path.setter
    def image_path(self, path: str):
        if not os.path.exists(path):
            raise FileExistsError("Image file {} not found".format(self.image_path))
        self.__image_path = path

    def _extract_rtk_data(self, image_path: str) -> Optional[list]:
        """Extracts RTK precise position (latitude, longitude, altitude) from the image metadata if available.

        :param image_path: Path to the image file.
        :return: A list with latitude, longitude, and altitude, or [None, None, None] if RTK data is not available.
        """
        try:
            with open(image_path, 'rb') as image_file:
                tags = exifread.process_file(image_file)

                # Extract GPS data
                latitude = tags.get('GPS GPSLatitude')
                latitude_ref = tags.get('GPS GPSLatitudeRef')
                longitude = tags.get('GPS GPSLongitude')
                longitude_ref = tags.get('GPS GPSLongitudeRef')
                altitude = tags.get('GPS GPSAltitude')

                if latitude and longitude and altitude:
                    # Convert latitude and longitude to decimal degrees
                    lat = self._convert_to_decimal(latitude, str(latitude_ref))
                    lon = self._convert_to_decimal(longitude, str(longitude_ref))
                    alt = float(altitude.values[0])  # Altitude in meters

                    Logger.info(f"RTK data extracted for {image_path}: Latitude={lat}, Longitude={lon}, Altitude={alt}")
                    return [lat, lon, alt]
                else:
                    Logger.warning(f"RTK data not found in image metadata for {image_path}.")
                    return [None, None, None]
        except Exception as e:
            Logger.error(f"Error extracting RTK data from {image_path}: {e}")
            return [None, None, None]

    @staticmethod
    def _convert_to_decimal(value, ref) -> float:
        """Converts GPS coordinates from DMS to decimal degrees.

        :param value: GPS coordinate in DMS format.
        :param ref: Reference ('N', 'S', 'E', 'W').
        :return: Decimal degree representation of the coordinate.
        """
        d, m, s = [float(x.num) / float(x.den) for x in value.values]
        decimal = d + (m / 60.0) + (s / 3600.0)
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal

    def __load_images(self):
        """Loads all images from the specified folder and extracts RTK data."""
        # Get all image files in the folder (e.g., .jpg, .png, etc.)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = sorted(
            [f for f in os.listdir(self.image_folder_path) if f.lower().endswith(valid_extensions)]
        )

        if not image_files:
            Logger.error("No valid image files found in folder: {}".format(self.image_folder_path))
            raise ValueError("No valid image files found in folder: {}".format(self.image_folder_path))

        Logger.info("Found {} images in folder: {}".format(len(image_files), self.image_folder_path))

        # Load each image and extract RTK data
        for image_file in image_files:
            image_path = os.path.join(self.image_folder_path, image_file)
            image = cv2.imread(image_path, self.mode)
            if image is None:
                Logger.warning("Could not load image: {}".format(image_path))
                continue

            self.frames.append(image)
            self.image_paths.append(image_path)

            # Extract RTK data for the current image
            rtk_data = self._extract_rtk_data(image_path)
            self.rtk_data_list.append(rtk_data)

            Logger.debug(f"Loaded image: {image_path}, RTK data: {rtk_data}")

        Logger.info("Successfully loaded {} images.".format(len(self.frames)))


class VideoLoader:
    """Class responsible for laoding a video into a sequence of numpy arrays representing the single video frames."""

    def __init__(self, video_path: str, start_frame: int = 0, end_frame: int = None):
        """Loads the frames associated to the video indicated by the path passed as argument.

        :param video_path: Absolute path to the video to be loaded.
        :param start_frame: Start frame of the video to be considered (inclusive).
        :param end_frame: End frame of the video to be considered (non inclusive). If set to None, the video will be loaded until the last frame.
        """
        Logger.debug("Loading video at {}".format(video_path))
        self.video_path = video_path

        self.start_frame = start_frame
        self.end_frame = end_frame
        Logger.debug("Start frame: {}, end frame: {}".format(self.start_frame, self.end_frame))

        self.frames = []
        self.__load_video(cv2.VideoCapture(self.video_path))

    @property
    def num_frames(self) -> int:
        """Returns the number of frames loaded by this object."""
        return self.end_frame - self.start_frame

    @property
    def video_path(self) -> str:
        """Returns the absolute path associated to the video loaded by this object."""
        return self.__video_path

    @video_path.setter
    def video_path(self, path: str):
        if not os.path.exists(path):
            Logger.fatal("Video path {} does not exist".format(path))
            raise FileNotFoundError("Video file {} not found".format(path))
        self.__video_path = path

    def __load_video(self, video_raw: cv2.VideoCapture):
        if not video_raw.isOpened():
            Logger.error("Unable to read {} feed".format(self.video_path))

        self.frames = []

        num_video_frames = int(video_raw.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.end_frame is None or self.end_frame > num_video_frames:
            Logger.warning("Setting end_frame to {}".format(num_video_frames))
            self.end_frame = num_video_frames

        num_frames = 0

        # Skip the first frames until the self_start frame.
        video_raw.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        Logger.info("Loading {} frames...".format(self.end_frame - self.start_frame))
        bar = progressbar.ProgressBar(maxval=self.num_frames,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(self.end_frame - self.start_frame):
            ret = video_raw.grab()
            if not ret:
                Logger.error("Could not load frame {}".format(i + self.start_frame))
                raise ValueError("Could not load frame {}".format(i + self.start_frame))

            self.frames.append(video_raw.retrieve()[1])
            num_frames += 1
            bar.update(num_frames)

        bar.finish()
        video_raw.release()
