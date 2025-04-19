import cv2
import numpy as np
from skimage.morphology import skeletonize
from simple_logger import Logger
from scipy.stats import mode
import matplotlib.pyplot as plt
from scipy import stats

__all__ = ["CrackDetectorParams", "CrackDetector"]


class CrackDetectorParams:
    """Parameters used by the :class:`.CrackDetector`."""
    def __init__(self):
        """Initializes the crack cell parameters to their default value.

        :ivar crack_thresh: Threshold for detecting cracks in the grayscale image.
        :ivar busbar_thresh: Threshold for detecting busbars in the grayscale image.
        :ivar busbar_num: Number of busbars in the cell.
        :ivar extract_brightness_mode: Method used to extract brightness from the image.
        :ivar extract_brightness_norm: Normalization factor for brightness extraction.
        """
        self.crack_thresh = 50
        self.busbar_thresh = 200
        self.busbar_num = 0
        self.extract_brightness_mode = "avg_inactive_only"
        self.extract_brightness_norm = 255


class CrackDetector:
    """Class responsible for detecting cracks and busbars in grayscale images.

    The approach taken to detect cracks and busbars in the input grayscale image is the following:

        1. Perform thresholding to detect cracks and busbars based on intensity values.
        2. Extract skeletons of the detected cracks and busbars for further analysis.
        3. Merge the skeletons of cracks and busbars to create a unified representation.
        4. Analyze the merged skeleton to extract features such as inactive area proportion, crack length, and brightness.

    This class provides methods to detect cracks, busbars, and inactive areas, as well as to compute various features
    related to the structural integrity of the input image.

    """
    def __init__(self, input_image: np.ndarray, params: CrackDetectorParams = CrackDetectorParams()):
        """
        :param input_image: Input EL image module to be analyzed.
        :param params: Parameters used for crack detection.
        """
        self.img = input_image
        self.params = params

        self.crack = self._detect_cracks()
        self.busbar = self._detect_busbars()
        self._merge = self.crack + self.busbar
        self._merge[self._merge > 1] = 1

        self.busbar_num = self.params.busbar_num

        self.ske_crack = self._extract_crack_skeleton()
        self.ske_busbar = self._extract_busbar_skeleton()
        self.ske_merge = self._merge_crack_busbar()

        self.inactive_area = None
        self.features = {}

    def _detect_cracks(self) -> np.ndarray:
        """Automatically detect cracks in the grayscale image."""
        Logger.debug("Detecting cracks in the image")
        _, crack_mask = cv2.threshold(self.img, self.params.crack_thresh, 255, cv2.THRESH_BINARY_INV)
        return crack_mask.astype(np.uint8)

    def _detect_busbars(self) -> np.ndarray:
        """Automatically detect busbars in the grayscale image."""
        Logger.debug("Detecting busbars in the image")
        _, busbar_mask = cv2.threshold(self.img, self.params.busbar_thresh, 255, cv2.THRESH_BINARY)
        return busbar_mask.astype(np.uint8)

    def _extract_crack_skeleton(self):
        """Extract crack skeleton"""
        ske_crack = self.skeleton_crack(self.crack)
        return ske_crack

    def _extract_busbar_skeleton(self):
        """Extract busbar skeleton"""
        ske_busbar = self.extend_busbar(self.busbar)
        busbar_num = mode(np.sum(ske_busbar, axis=0), keepdims=False).mode
        if busbar_num != self.busbar_num:
            raise ValueError("Busbar number is not correct")

        return ske_busbar

    def _merge_crack_busbar(self):
        """Merge crack and busbar skeletons"""
        ske_merge = self.ske_crack + self.ske_busbar
        ske_merge[ske_merge > 1] = 1
        return ske_merge

    def extract_inactive_area(self):
        """Extract inactive area"""
        self.inactive_area, self.features["inactive_prop"] = self.detect_inactive(self.crack, self.busbar)
        return self.inactive_area, self.features["inactive_prop"]

    def extract_crack_length(self):
        """Extract crack length (din pixel)"""
        self.features["crack_length"] = self.ske_crack.sum()
        return self.features["crack_length"]

    def extract_brightness(self, mode, norm=255):
        """Extract brightness
        :param mode: The mode to use for brightness extraction.
        if mode == "avg_all": average brightness of the raw image
        if mode == "avg_inactive": average brightness of the inactive area and treat the active area as 1
        if mode == "avg_inactive_only": average brightness of the inactive area only and return 1 for intact cells

        :param norm: Normalization factor for brightness extraction.
        :return: The brightness value based on the selected mode.
        """
        if mode == "avg_all":
            self.features["brightness_cell"] = self.avg_grayscale(self.img, norm=norm)
            return self.features["brightness_cell"]
        elif mode == "avg_inactive":
            if self.inactive_area is None:
                Logger.warning("Inactive area is not extracted. Extracting now...")
                self.extract_inactive_area()
            self.features["brightness_inactive"] = self.avg_grayscale2(self.img, self.inactive_area, norm=norm)
            return self.features["brightness_inactive"]
        elif mode == "avg_inactive_only":
            if self.inactive_area is None:
                Logger.warnings("Inactive area is not extracted. Extracting now...")
                self.extract_inactive_area()
            self.features["brightness_inactive_only"] = self.avg_grayscale3(self.img, self.inactive_area, norm=norm)
            return self.features["brightness_inactive_only"]

    def extract_features(self):
        """Extract all features"""
        self.extract_inactive_area()
        self.extract_crack_length()
        self.extract_brightness(mode=self.params.extract_brightness_mode, norm=self.params.extract_brightness_norm)
        return self.features

    def plot(self):
        """plot raw image, masks, skeletons and inactive area"""
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes[0, 0].imshow(self.img, cmap="gray")
        axes[0, 0].set_title("Raw image")
        axes[0, 1].imshow(self._merge, cmap="gray")
        axes[0, 1].set_title("Masks")
        axes[1, 0].imshow(self.ske_merge, cmap="gray")
        axes[1, 0].set_title("Skeleton")
        axes[1, 1].imshow(self.inactive_area, cmap="gray")
        axes[1, 1].set_title("Inactive area")
        plt.tight_layout()
        plt.show()

    def skeleton_crack(mask_crack):
        """Skeletonize crack masks

        :param mask_crack: Mask of cracks.
        :return: Skeletonized crack mask.
        """
        return skeletonize(mask_crack).astype(np.uint8)


    def extend_busbar(self, mask_busbar, kernel_size=(10, 100)):
        """Connect and extend broken busbars. Return the skeleton of the busbar masks

        :param mask_busbar: Masks of busbars
        :param kernel_size: Kernel used to do morphological operation. Details available on
        https://docs.opencv.org/4.5.4/d9/d61/tutorial_py_morphological_ops.html
        :return: Skeletonized busbar masks.
        """
        kernel = np.ones(kernel_size, np.uint8)
        closing = cv2.morphologyEx(mask_busbar, cv2.MORPH_CLOSE, kernel)
        ske_busbar = skeletonize(closing).astype(np.uint8)

        return ske_busbar


    def locate_busbar(self, ske_busbar):
        """Get position of busbars

        :param ske_busbar: Skeletonized busbar masks.
        :return: Positions of busbars.
        """
        numlist_busbar = []
        for i in np.linspace(10, ske_busbar.shape[-1] - 10, 10, dtype=int):
            numlist_busbar.append(len(np.argwhere(ske_busbar[:, i] == 1)))

        num_busbar = stats.mode(numlist_busbar).mode

        pos_busbar = np.zeros((num_busbar, 1))
        for i in np.linspace(10, ske_busbar.shape[-1] - 10, 100, dtype=int):
            tem_pos = np.argwhere(ske_busbar[:, i] == 1)

            if len(tem_pos) == num_busbar:
                pos_busbar = np.hstack((pos_busbar, tem_pos))

        pos_busbar = np.delete(pos_busbar, 0, axis=1)
        pos_busbar = pos_busbar.mean(axis=1, dtype=int).tolist()

        return pos_busbar


    def skeleton_cell(self, ske_crack, pos_busbar):
        """Get the skeleton of a cell. Crack has the value of -1, busbar is 1, other area is 0.

        :param ske_crack: Skeleton of crack masks.
        :param pos_busbar: Position of busbars.
        :return: Skeleton of cell.
        """
        ske_cell = ske_crack * -1
        for i in pos_busbar:
            ske_cell[i, :] = 1

        return ske_cell


    def stop_diff(self, val):
        """Check whether diffusion should stop. If meet busbar(val=1) or crack(val=-1), stop diffusion.

        :param val: Grayscale value of a pixel.
        :return: True if diffusion should stop, False otherwise.
        """
        return val == 1 or val == -1


    def diff_up(self, image, row, col):
        """Diffuse the electrons up. The busbars are horizontally aligned.

        :param image: Skeleton of cell.
        :param row: Row position of current pixel.
        :param col: Column position of current pixel.
        """
        current = row - 1
        while not (current < 0 or self.stop_diff(image[current, col])):
            image[current, col] = 1
            current -= 1


    def diff_down(self, image, row, col):
        """Diffuse the electrons down. The busbars are horizontally aligned

        :param image: Skeleton of cell.
        :param row: Row position of current pixel.
        :param col: Column position of current pixel.
        """
        end = image.shape[0]
        current = row + 1
        while not (current > end - 1 or self.stop_diff(image[current, col])):
            image[current, col] = 1
            current += 1


    def diffuse_line(self, image, row):
        """Diffuse the electrons from one busbar. The busbars are horizontally aligned.

        :param image: Skeleton of cell.
        :param row: Position of current busbar.
        """
        for j in range(image.shape[-1]):
            self.diff_up(image, row, j)
            self.diff_down(image, row, j)


    def diffuse(self, image, pos_busbar):
        """Diffuse the electrons from all busbars. The busbars are horizontally aligned.

        :param image: Skeleton of cell.
        :param pos_busbar: Position of busbars.
        :return: Diffused cell
        """
        image_c = np.copy(image)
        for i in pos_busbar:
            self.diffuse_line(image_c, i)
        return image_c


    def count_area(self, cell_diff):
        """Count worst-case percentage of inactive area.

        :param cell_diff: Diffused cell image.
        :return: Percentage of inactive area.
        """
        inactive_area = np.zeros(cell_diff.shape).astype(np.uint8)
        inactive_area[cell_diff == 0] = 1
        return inactive_area.sum() / (inactive_area.shape[0] * inactive_area.shape[1])


    def detect_inactive(self, mask_crack, mask_busbar, extend_kernel=(10, 100)):
        """Detect the worst-case isolated area and calculate its proportion.

        :param mask_crack: Mask of cracks.
        :param mask_busbar: Masks of busbars.
        :param extend_kernel: Kernel used to do morphological operation to extend busbar. Details available on
        https://docs.opencv.org/4.5.4/d9/d61/tutorial_py_morphological_ops.html

        :return inactive_area: Binary isolated area.
        :return inactive_prop: Percentage of inactive area. Not in the form of %.
        """
        ske_crack = self.skeleton_crack(mask_crack)
        ske_busbar = self.extend_busbar(mask_busbar, kernel_size=extend_kernel)
        pos_busbar = self.locate_busbar(ske_busbar)
        ske_cell = self.skeleton_cell(ske_crack, pos_busbar)
        cell_diff = self.diffuse(ske_cell, pos_busbar)

        inactive_area = np.zeros(cell_diff.shape).astype(np.uint8)
        inactive_area[cell_diff == 0] = 1
        inactive_prop = inactive_area.sum() / (inactive_area.shape[0] * inactive_area.shape[1])

        return inactive_area, inactive_prop


    def crack_length(self, mask_crack):
        """Compute the length of crack

        :param mask_crack: Mask of cracks.
        :return: Length of crack in pixels.
        """
        ske_crack = self.skeleton_crack(mask_crack)
        return ske_crack.sum()


    def avg_grayscale(self, raw_image, norm=255):
        """Average brightness of the raw image

        :param raw_image: Raw image.
        :param norm: Normalization factor for brightness extraction.
        :return: Average brightness of the raw image.
        """
        return (raw_image/norm).sum() / (raw_image.shape[0] * raw_image.shape[1])


    def avg_grayscale2(self, raw_img, inactive_area, norm=255):
        """Average grayscale of inactive area, treat unisolated area as white
        If my isolate area is small but grayscale of that area is severe, the metric value is not bad.
        Because the white pixels around the small severe part compensate for it.

        :param raw_img: Raw image.
        :param inactive_area: Binary isolated area.
        :param norm: Normalization factor for brightness extraction.
        :return: Average brightness of the inactive area.
        """
        raw_inactive = raw_img/norm * inactive_area + (1 - inactive_area)
        return raw_inactive.sum()/(raw_inactive.shape[0] * raw_inactive.shape[1])


    def avg_grayscale3(self, raw_img, inactive_area, norm=255):
        """Average grayscale of inactive area ONLY. For intact cell, the metric is just 1
        If my isolate area is small but grayscale of that area is severe, the metric value is also severe.

        :param raw_img: Raw image.
        :param inactive_area: Binary isolated area.
        :param norm: Normalization factor for brightness extraction.
        :return: Average brightness of the inactive area.
        """
        raw_inactive = raw_img / norm * inactive_area
        area = inactive_area.sum()
        return raw_inactive.sum() / area if area else 1
