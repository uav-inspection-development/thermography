import cv2
import numpy as np
from thermography.thermo_app import ThermoApp
import thermography as tg
import os

input_image_path = "../../examples/EL_img/EL_img1.jpg"

SETTINGS_DIR = tg.settings.get_settings_dir()
camera_param_file = os.path.join(SETTINGS_DIR, "camera_parameters.json")

# load image
frame = cv2.imread(input_image_path)
if frame is None:
    print("Error: Unable to load image. Please check the file path.")
    exit()

# create ThermoApp example
app = ThermoApp(input_video_path=input_image_path, camera_param_file=camera_param_file)

# set input frame
app.last_input_frame = frame

# Assuming that the image does not require distortion processing
app.should_undistort_image = False

# Call the step method to process images
if app.step(frame_id=0, frame=frame):
    print("Detection successful!")
else:
    print("Detection failed!")

# Display test results
cv2.imshow("Filtered segments", app.create_segment_image())
# cv2.imshow("Detected rectangles", app.create_rectangle_image())
cv2.imshow("Global map", app.create_module_map_image())
cv2.waitKey(0)
cv2.destroyAllWindows()