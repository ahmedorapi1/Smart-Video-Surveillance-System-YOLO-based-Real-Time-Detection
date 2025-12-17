from ultralytics import YOLO
import cv2
import threading
import time
import queue


class ModelInference:
    def __init__(
        self,
        output_path: str,
        frame_size: tuple,
        model_path: str,
        model_input_size=(640, 640),
        conf: float = 0.5,
        writer_fps: float = 10.0,
        queue_size: int = 2,
        device=None,
        overlay: bool = True,
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.model_input_size = model_input_size
        self.frame_size = frame_size
        self.conf = conf
        self.writer_fps = float(writer_fps)
        self.device = device
        self.overlay = overlay

        self.frame_queue = queue.Queue(maxsize=queue_size)
        self._running = False
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._write_interval = 1.0 / self.writer_fps if self.writer_fps > 0 else 0.0
        self._last_write_time = 0.0

        self._init_session()

    def _init_session(self):
        self.model = YOLO(self.model_path)
        self.writer = self.build_writer()

    def build_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, self.writer_fps, self.frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.output_path}")
        return writer

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join()
        self.writer.release()
        cv2.destroyAllWindows()

    def push_frame(self, frame):
        if self.frame_queue.full():
            try:
                _ = self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def _run(self):
        prev_time = time.time()
        smooth_fps = 0.0
        alpha = 0.1

        while self._running:
            frame = self.frame_queue.get()
            if frame is None:
                break
            if frame is None or frame.size == 0:
                continue

            now = time.time()
            dt = now - prev_time
            prev_time = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            smooth_fps = (1 - alpha) * smooth_fps + alpha * inst_fps if smooth_fps > 0 else inst_fps

            results = self.model.predict(
                source=frame,
                conf=self.conf,
                verbose=False,
                device=self.device
            )
            annotated = results[0].plot()

            if self.overlay:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    annotated, f"PROC FPS: {smooth_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )
                cv2.putText(
                    annotated, f"OUT FPS: {self.writer_fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
                cv2.putText(
                    annotated, ts, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )

            t = time.time()
            if self._write_interval <= 0 or (t - self._last_write_time) >= self._write_interval:
                w, h = self.frame_size
                if annotated.shape[1] != w or annotated.shape[0] != h:
                    annotated = cv2.resize(annotated, (w, h))
                self.writer.write(annotated)
                self._last_write_time = t
