import os
import argparse
import thermography as tg
from thermography.io import setup_logger, LogLevel


def _main():
    SETTINGS_DIR = tg.settings.get_settings_dir()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Thermography Application")
    parser.add_argument("--camera_param_file", required=True, help="Path to the camera parameters JSON file", default=os.path.join(SETTINGS_DIR, "camera_parameters.json"))
    parser.add_argument("--input_video", help="Path to the input video file")
    parser.add_argument("--input_image_folder", help="Path to the folder containing input images")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame (inclusive, default: 0)")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame (exclusive, default: None)")
    args = parser.parse_args()

    # Check for conflicting inputs
    if args.input_video and args.input_image_folder:
        print("Error: Both --input_video and --input_image are provided. Please specify only one.")
        exit(1)
    elif not args.input_video and not args.input_image_folder:
        print("Error: Neither --input_video nor --input_image is provided. Please specify one.")
        exit(1)

    app = tg.App(input_path=args.input_video or args.input_image_folder, camera_param_file=args.camera_param_file)

    # Load video or image based on the input
    if args.input_video:
        app.load_video(start_frame=args.start_frame, end_frame=args.end_frame)
    elif args.input_image_folder:
        app.load_image(start_frame=args.start_frame, end_frame=args.end_frame)

    app.run()


if __name__ == '__main__':
    setup_logger(console_log_level=LogLevel.INFO, file_log_level=LogLevel.DEBUG)
    _main()
