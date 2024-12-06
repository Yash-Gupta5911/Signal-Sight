import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
from datetime import datetime

# Paths for input and output
SOURCE_VIDEO_PATH = r"C:\Users\LENOVO\Desktop\s2\vehicles.mp4"
CURRENT_TIME = datetime.now().strftime("%d-%m-%Y_%I-%M-%p")
OUTPUT_VIDEO_PATH = f"output_{CURRENT_TIME}.mp4"

# Perspective transformation points (adjust based on your scene)
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def resize_frame(frame, target_width=800):
    """Resize frame for display without distorting aspect ratio."""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_height = int(target_width / aspect_ratio)
    return cv2.resize(frame, (target_width, new_height))

if __name__ == "__main__":
    # YOLO settings
    CONFIDENCE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.7

    # Load video and YOLO model
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    model = YOLO("yolov8x.pt")  # Replace with your YOLO model path

    # Initialize ByteTrack and annotators
    byte_tracker = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=CONFIDENCE_THRESHOLD
    )
    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness, trace_length=video_info.fps * 2, position=sv.Position.BOTTOM_CENTER
    )

    # Polygon zone for detection filtering
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_info.fps, video_info.resolution_wh)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, stream=True)  # Stream mode returns a generator

        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
            detections = detections.with_nms(threshold=IOU_THRESHOLD)

            # Apply polygon zone filtering
            detections = detections[polygon_zone.trigger(detections)]

            # Update ByteTrack with detections
            detections = byte_tracker.update_with_detections(detections=detections)

            # Transform points for speed calculation
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            # Debug vehicle speeds
            labels = []
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    # Insufficient data to calculate speed yet
                    labels.append(f"#{tracker_id}")
                else:
                    # Calculate speed
                    start = coordinates[tracker_id][-1]
                    end = coordinates[tracker_id][0]
                    distance = abs(start - end)  # Change in position in pixels
                    time = len(coordinates[tracker_id]) / video_info.fps  # Time in seconds
                    speed = (distance / time) * 3.6  # Convert m/s to km/h
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

            # Display and save frame
            resized_frame = resize_frame(annotated_frame)
            cv2.imshow("Annotated Frame", resized_frame)
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
