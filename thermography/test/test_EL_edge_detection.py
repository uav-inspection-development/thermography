import cv2
import numpy as np
import matplotlib.pyplot as plt

# 确保 EdgeDetector 类和 EdgeDetectorParams 类在当前目录下
from thermography.detection.edge_detection import EdgeDetector, EdgeDetectorParams

def test_edge_detector():
    # 加载图像
    image_path = "../../examples/EL_img/EL_img2.jpg"  # 替换为你的测试图像路径
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load image. Please check the file path.")
        return

    params = EdgeDetectorParams()

    # 创建 EdgeDetector 对象
    edge_detector = EdgeDetector(input_image=image, params=params)

    # 执行边缘检测
    edge_detector.detect()

    # 获取检测到的边缘图像
    edge_image = edge_detector.edge_image

    # 显示原始图像和边缘图像
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