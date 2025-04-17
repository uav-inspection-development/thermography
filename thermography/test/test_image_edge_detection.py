import cv2
import numpy as np
import matplotlib.pyplot as plt

# 确保 EdgeDetector 类和 EdgeDetectorParams 类在当前目录下
from thermography.detection.edge_detection import EdgeDetector, EdgeDetectorParams

def test_edge_detector():
    # load image
    image_path = "../../examples/EL_img/EL_img2.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load image. Please check the file path.")
        return

    params = EdgeDetectorParams()

    # Create EdgeDetector object
    edge_detector = EdgeDetector(input_image=image, params=params)

    # Perform edge detection
    edge_detector.detect()

    # Obtain detected edge images
    edge_image = edge_detector.edge_image

    # Display the original image and edge image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Edge Image")
    plt.imshow(edge_image, cmap="gray")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    test_edge_detector()