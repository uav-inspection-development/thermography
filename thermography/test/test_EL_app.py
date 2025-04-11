import cv2
import numpy as np
from thermography.thermo_app import ThermoApp
import thermography as tg
import os
# 确保路径正确
input_image_path = "../../examples/EL_img/EL_img2.jpg"

SETTINGS_DIR = tg.settings.get_settings_dir()
camera_param_file = os.path.join(SETTINGS_DIR, "camera_parameters.json")

# 加载图片
frame = cv2.imread(input_image_path)
if frame is None:
    print("Error: Unable to load image. Please check the file path.")
    exit()

# 创建 ThermoApp 实例
app = ThermoApp(input_video_path=input_image_path, camera_param_file=camera_param_file)

# 设置输入帧
app.last_input_frame = frame

# 假设图片不需要去畸变处理
app.should_undistort_image = False

# 调用 step 方法处理图片
if app.step(frame_id=0, frame=frame):
    print("Detection successful!")
else:
    print("Detection failed!")

# 显示检测结果
cv2.imshow("Filtered segments", app.create_segment_image())
cv2.imshow("Detected rectangles", app.create_rectangle_image())
cv2.imshow("Global map", app.create_module_map_image())
cv2.waitKey(0)
cv2.destroyAllWindows()