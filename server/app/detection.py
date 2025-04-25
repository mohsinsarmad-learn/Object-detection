import cv2
from ultralytics import YOLO

def load_model(model_path: str) -> YOLO:
    """
    Load and return a YOLO model from .pt weights.
    """
    return YOLO(model_path)

def process_video(
    model: YOLO,
    input_path: str,
    output_path: str,
    conf_thresh: float = 0.5
) -> None:
    """
    Run inference on the input video frame by frame,
    draw bounding boxes, and save annotated video.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    names  = model.names  # e.g. ['circle','rectangle','triangle']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_thresh)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label  = names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        out.write(frame)

    cap.release()
    out.release()
