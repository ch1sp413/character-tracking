"""
detect_and_track.py

Offline detection + tracking script with Comet ML integration. 
- Uses YOLOv5 for detection.
- Uses OpenCV CSRT for tracking.
- Logs metrics (IoU, YOLO confidence, bounding-box area, reacquisitions, etc.) to:
  1) Comet ML (real-time experiment management).
  2) A local CSV file ("tracker_metrics.csv").

Author: Roger Ribas
"""

import cv2
import torch
import csv
import time
from collections import deque

########################
# 1) Comet ML Integration
########################
try:
    from comet_ml import Experiment

    # Create a Comet experiment
    experiment = Experiment(
        api_key="<YOUR_API_KEY>",          # Replace with your actual Comet API Key
        project_name="<YOUR_PROJECT_NAME>",        # Replace with your Comet project name
        workspace="<YOUR_WROKSPACE_NAME>"          # Replace with your Comet workspace
    )
    # Optionally set a name or other metadata
    experiment.set_name("detect_and_track_v1")
    experiment.add_tags(["offline-tracking", "CSRT", "YOLOv5"])
except ImportError:
    experiment = None
    print("Comet ML not installed or import failed. Proceeding without Comet logging.")

########################
# 2) IoU Calculation
########################
def iou(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Each box is a tuple (x, y, w, h) in pixel coordinates.

    boxA, boxB: (x, y, w, h)

    Returns: IoU as a float between 0.0 and 1.0
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0

    return interArea / float(unionArea)

########################
# 3) Load YOLO Model
########################
# If you cloned YOLOv5 locally, you could do source='local' and point to the local path. 
# If not, source='github' downloads it from GitHub's ultralytics/yolov5 on first run.

model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='<PATH_TO_BEST.PT>',  # Path to your trained YOLO weights
    source='github'
)

# Confidence threshold for YOLO detection
model.conf = 0.25

def detect_character(frame):
    """
    Runs a YOLO inference on the given frame.

    Returns: (bbox, conf)
        bbox = (x, y, w, h) of the best detection (highest confidence),
        conf = float detection confidence,
        or (None, None) if no detections found.
    """
    results = model(frame)          # YOLO inference
    detections = results.xyxy[0]    # format: [x1, y1, x2, y2, conf, class]

    if len(detections) > 0:
        # Pick the detection with the highest confidence
        best = max(detections, key=lambda det: det[4])
        x1, y1, x2, y2, conf, cls = best

        # Convert to int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        return (x1, y1, w, h), float(conf)
    else:
        return None, None


def main():
    ########################
    # 4) Initial Setup
    ########################
    video_path = '<PATH_TO_INPUT_VIDEO_FILE>'   # Path to your input gameplay video
    output_path = '<PATH_TO_OUTPUT_VIDEO_FILE>'   # Where you'll save the annotated video

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        return

    # Gather video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer for the annotated output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ########################
    # 5) Hyperparams & Variables
    ########################
    DETECTION_INTERVAL = 30  # Re-run YOLO detection every 30 frames
    iou_window_size = 30     # Rolling IoU window
    iou_values = deque(maxlen=iou_window_size)

    tracker = None
    tracking_active = False
    frames_since_detection = 0

    last_detected_bbox = None
    last_detection_conf = 0.0
    reacquisition_count = 0

    # For timing and CSV logging
    start_time = time.time()
    frame_index = 0
    csv_filename = 'tracker_metrics.csv'

    ########################
    # 6) Comet ML: Log Some Hyperparams
    ########################
    if experiment is not None:
        experiment.log_parameter("detection_interval", DETECTION_INTERVAL)
        experiment.log_parameter("model_conf_threshold", model.conf)
        experiment.log_parameter("iou_window_size", iou_window_size)
        experiment.log_parameter("video_path", video_path)
        experiment.log_parameter("output_path", output_path)

    ########################
    # 7) CSV Setup
    ########################
    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Header row
        writer.writerow([
            'frame',
            'iou',
            'avg_iou',
            'yolo_confidence',
            'bbox_area',
            'reacquisition_count',
            'elapsed_time_sec'
        ])

        ########################
        # 8) Main Loop
        ########################
        while True:
            ret, frame_mat = cap.read()
            if not ret:
                break  # No more frames

            frame = frame_mat
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Periodically re-detect or if not currently tracking
            if not tracking_active or frames_since_detection >= DETECTION_INTERVAL:
                bbox, conf = detect_character(frame)
                if bbox is not None:
                    # Initialize or re-initialize the CSRT tracker
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, bbox)

                    tracking_active = True
                    frames_since_detection = 0

                    last_detected_bbox = bbox
                    last_detection_conf = conf
                else:
                    tracking_active = False
            else:
                # We have an active tracker → update
                success, tracked_bbox = tracker.update(frame)

                if not success:
                    # The tracker lost the character → reacquire
                    reacquisition_count += 1
                    bbox, conf = detect_character(frame)
                    if bbox is not None:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, bbox)
                        tracking_active = True
                        frames_since_detection = 0

                        last_detected_bbox = bbox
                        last_detection_conf = conf
                    else:
                        tracking_active = False
                else:
                    # Convert floats to ints
                    tracked_bbox = tuple(map(int, tracked_bbox))

                    # Compute IoU with the last YOLO detection
                    current_iou = 0.0
                    if last_detected_bbox is not None:
                        current_iou = iou(tracked_bbox, last_detected_bbox)
                        iou_values.append(current_iou)

                    # Rolling average IoU
                    avg_iou = sum(iou_values) / len(iou_values) if iou_values else 0.0

                    # Draw bounding box on the frame
                    x, y, w, h = tracked_bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Bbox area
                    box_area = w * h

                    # Write to CSV
                    writer.writerow([
                        frame_index,
                        f"{current_iou:.4f}",
                        f"{avg_iou:.4f}",
                        f"{last_detection_conf:.4f}",
                        box_area,
                        reacquisition_count,
                        f"{elapsed_time:.2f}"
                    ])

                    # COMET: Log metrics
                    if experiment is not None:
                        experiment.log_metric("iou", current_iou, step=frame_index)
                        experiment.log_metric("avg_iou", avg_iou, step=frame_index)
                        experiment.log_metric("yolo_confidence", last_detection_conf, step=frame_index)
                        experiment.log_metric("bbox_area", box_area, step=frame_index)
                        experiment.log_metric("reacquisition_count", reacquisition_count, step=frame_index)
            
            ########################
            # 9) Display Overlay
            ########################
            if tracking_active:
                status_text = "Tracking: ON"
                color = (0, 255, 0)
            else:
                status_text = "Tracking: OFF"
                color = (0, 0, 255)

            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # Write the frame to output video
            out.write(frame)

            # Optionally show a real-time preview
            cv2.imshow('Character Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frames_since_detection += 1
            frame_index += 1

    ########################
    # 10) Cleanup
    ########################
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Finished processing. CSV saved to '{csv_filename}'.")
    if experiment is not None:
        experiment.end()  # finalize Comet experiment

########################
# 11) Entry Point
########################
if __name__ == "__main__":
    main()