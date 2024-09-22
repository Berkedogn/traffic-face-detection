import logging
from config import CONFIG
from video_processor import VideoProcessor
from traffic_analyzer import TrafficAnalyzer
from drawing_utils import determine_box_color, draw_boxes, display_info
from traffic_status import get_traffic_status
from face_blurring import FaceBlurrer

logging.basicConfig(level=logging.INFO)

def main():
    # Video ve model işlemleri
    video_processor = VideoProcessor(CONFIG['video_path'], CONFIG['output_path'], CONFIG['codec'], CONFIG['fps'])
    traffic_analyzer = TrafficAnalyzer(CONFIG['model_path'], CONFIG['vehicle_classes'])
    face_blurrer = FaceBlurrer("yolov8l_100e.pt")  # Yüz tespit modeli

    # Ana döngü
    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        # Araç tespiti
        vehicle_count, detected_objects = traffic_analyzer.analyze_frame(frame)

        # Trafik durumu belirleme ve kutu rengi
        traffic_status = get_traffic_status(vehicle_count, CONFIG['low_traffic_threshold'], CONFIG['medium_traffic_threshold'])
        box_color = determine_box_color(vehicle_count, CONFIG['low_traffic_threshold'], CONFIG['medium_traffic_threshold'])

        # Araç kutularını çiz ve bilgileri göster
        draw_boxes(frame, detected_objects, box_color)
        display_info(frame, vehicle_count, traffic_status)

        # Yüz bulanıklaştırma işlemi
        frame = face_blurrer.blur_faces(frame)

        # Çıktı frame'i kaydet
        video_processor.write_frame(frame)

    # Kaynakları serbest bırak
    video_processor.release_resources()

if __name__ == "__main__":
    main()
