Traffic and Face Detection with YOLOv8
This project uses YOLOv8 models to perform vehicle detection, traffic density calculation, and face blurring in video footage. It processes videos frame by frame, detecting vehicles, counting them, changing box colors based on traffic density, and detecting and blurring human faces in the video.

Features
Vehicle detection using yolov8n.pt.
Traffic density calculation based on vehicle count.
Face detection and blurring using yolov8l_100e.pt.
Modular structure for easy scalability and maintenance.
Project Structure
config.py: Stores configuration values such as video paths, codec, FPS, and thresholds for traffic density.
video_processor.py: Manages video file input and output, handling video reading and frame writing.
traffic_analyzer.py: Performs vehicle detection and counting using YOLOv8.
face_blurring.py: Detects human faces in the video and applies a blurring effect to anonymize them.
drawing_utils.py: Contains helper functions to draw bounding boxes, display vehicle count, and show traffic status on the video frames.
traffic_status.py: Determines traffic density status based on the number of detected vehicles.
main.py: The main execution file that ties everything together and runs the full pipeline.
