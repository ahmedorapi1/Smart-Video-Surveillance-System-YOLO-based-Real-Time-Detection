import cv2
from detector import ModelInference


def open_capture(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    return cap


def safe_fps(cap, fallback=25.0):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        return float(fallback)
    return float(fps)


def main(
    source: str,
    output_path: str,
    target_output_fps: float = 10.0,
    model_path: str = "yolov8n.pt",
    conf: float = 0.5,
    device=None,
):
    cap = open_capture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera/video source")
        return

    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("[ERROR] Could not read first frame.")
        cap.release()
        return

    h, w = frame.shape[:2]
    frame_size = (w, h)

    src_fps = safe_fps(cap, fallback=25.0)

    out_fps = min(float(target_output_fps), float(src_fps))
    out_fps = max(2.0, out_fps)

    worker = ModelInference(
        output_path=output_path,
        model_path=model_path,
        frame_size=frame_size,
        model_input_size=(640, 640),
        conf=conf,
        writer_fps=out_fps,
        queue_size=2,
        device=device,
        overlay=True,
    )

    worker.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            worker.push_frame(frame)
    finally:
        worker.stop()
        cap.release()


if __name__ == "__main__":
    source = input("Enter video path / RTSP url / webcam index (0): ").strip() or "0"

    main(
        source=source,
        output_path=r"C:\Users\hp\Desktop\CV_projects\Smart Video Surveillance System\output.mp4",
        target_output_fps=15,
        model_path="yolov8n.pt",
        conf=0.5,
        device=0
    )
