from __future__ import annotations

import csv
import cv2
import platform
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class U20Camera:
    """Threaded camera interface with per-frame host timestamps.

    Every frame that is actually written to the MP4 receives a timestamp from
    ``time.perf_counter()``.  The timestamps are written to a CSV sidecar whose
    ``video_frame_index`` is guaranteed to match the frame index in the MP4.
    """

    def __init__(
        self,
        camera_index: int = 0,
        output_folder: str | Path = "captures",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        window_name: str = "U20CAM Live View",
        recording_queue_size: int = 1000,
    ) -> None:
        self.camera_index = camera_index
        self.output_folder = Path(output_folder)
        self.width = width
        self.height = height
        self.fps = fps
        self.window_name = window_name

        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.camera: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None

        self.actual_width = width
        self.actual_height = height
        self.actual_fps = float(fps)

        self._capture_thread: Optional[threading.Thread] = None
        self._recording_thread: Optional[threading.Thread] = None

        self._stop_capture_event = threading.Event()
        self._stop_recording_event = threading.Event()

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_host_time_s: Optional[float] = None

        self._recording_queue: queue.Queue = queue.Queue(
            maxsize=recording_queue_size
        )

        self.is_recording = False
        self.is_viewing = False

        self.video_path: Optional[Path] = None
        self.frame_timestamps_path: Optional[Path] = None
        self._frame_times_file = None
        self._frame_times_writer = None

        self._host_time_zero_s: Optional[float] = None
        self._capture_sequence = 0
        self._recorded_frame_count = 0
        self._dropped_frame_count = 0
        self._first_recorded_frame_host_time_s: Optional[float] = None
        self._last_recorded_frame_host_time_s: Optional[float] = None

    # ------------------------------------------------------------------
    # Camera setup / acquisition
    # ------------------------------------------------------------------

    def _open_camera(self) -> cv2.VideoCapture:
        system = platform.system()

        if system == "Darwin":
            backend = cv2.CAP_AVFOUNDATION
            backend_name = "AVFoundation"
        elif system == "Windows":
            backend = cv2.CAP_DSHOW
            backend_name = "DirectShow"
        else:
            backend = cv2.CAP_ANY
            backend_name = "Default"

        print(f"Opening camera using {backend_name} backend...")
        camera = cv2.VideoCapture(self.camera_index, backend)

        if not camera.isOpened():
            print("Preferred backend failed. Trying default backend...")
            camera.release()
            camera = cv2.VideoCapture(self.camera_index)

        return camera

    def start(self) -> None:
        """Open the camera and start the acquisition thread."""
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.camera = self._open_camera()

        if self.camera is None or not self.camera.isOpened():
            self.camera = None
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. "
                "Check camera permissions/index."
            )

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)

        self.actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(self.camera.get(cv2.CAP_PROP_FPS))
        if self.actual_fps <= 0:
            self.actual_fps = float(self.fps)

        print(
            f"Camera started: {self.actual_width}x{self.actual_height} "
            f"@ {self.actual_fps:.3f} FPS"
        )

        with self._lock:
            self._latest_frame = None
            self._latest_frame_host_time_s = None

        self._capture_sequence = 0
        self._stop_capture_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="U20CameraCapture",
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Only this thread calls ``camera.read()``."""
        frame_interval = 1.0 / max(self.actual_fps, 1e-9)

        try:
            while not self._stop_capture_event.is_set():
                loop_start = time.perf_counter()

                if self.camera is None:
                    break

                success, frame = self.camera.read()
                host_time_s = time.perf_counter()

                if not success or frame is None:
                    print("Warning: Could not read frame from camera.")
                    time.sleep(0.01)
                    continue

                capture_sequence = self._capture_sequence
                self._capture_sequence += 1

                with self._lock:
                    self._latest_frame = frame
                    self._latest_frame_host_time_s = host_time_s
                    recording = self.is_recording

                if recording:
                    try:
                        self._recording_queue.put_nowait(
                            (capture_sequence, host_time_s, frame.copy())
                        )
                    except queue.Full:
                        self._dropped_frame_count += 1

                elapsed = time.perf_counter() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            print("Camera capture thread stopped.")

    # ------------------------------------------------------------------
    # Frame access / image capture
    # ------------------------------------------------------------------

    def get_latest_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_latest_frame_with_time(self):
        with self._lock:
            if self._latest_frame is None:
                return None, None
            return self._latest_frame.copy(), self._latest_frame_host_time_s

    def save_frame(self) -> Optional[Path]:
        frame = self.get_latest_frame()
        if frame is None:
            print("No camera frame available to save.")
            return None

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        image_path = self.output_folder / f"u20cam_{timestamp}.jpg"

        if not cv2.imwrite(str(image_path), frame):
            print(f"Error: Could not save image to {image_path}")
            return None

        print(f"Image saved: {image_path}")
        return image_path

    # ------------------------------------------------------------------
    # Recording + timestamp sidecar
    # ------------------------------------------------------------------

    def start_recording(
        self,
        filename: Optional[str] = None,
        host_time_zero_s: Optional[float] = None,
    ) -> Path:
        """Start MP4 recording and per-frame timestamp logging.

        Parameters
        ----------
        filename:
            MP4 filename.
        host_time_zero_s:
            Shared ``time.perf_counter()`` zero used by the DAQ and camera.
            If omitted, a new zero is taken here.
        """
        if self.camera is None:
            raise RuntimeError("Camera is not started. Call start() first.")
        if self._capture_thread is None or not self._capture_thread.is_alive():
            raise RuntimeError("Camera capture thread is not running.")
        if self.is_recording or self._recording_thread is not None:
            raise RuntimeError("Camera is already recording.")

        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            filename = f"u20cam_{timestamp}.mp4"

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.video_path = self.output_folder / filename
        self.frame_timestamps_path = self.video_path.with_name(
            f"{self.video_path.stem}_frame_timestamps.csv"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.actual_fps,
            (self.actual_width, self.actual_height),
        )

        if not self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
            self.video_path = None
            raise RuntimeError("Could not create video file.")

        self._frame_times_file = self.frame_timestamps_path.open(
            "w", newline="", encoding="utf-8"
        )
        self._frame_times_writer = csv.writer(self._frame_times_file)
        self._frame_times_writer.writerow(
            [
                "video_frame_index",
                "capture_sequence",
                "host_time_s",
                "host_elapsed_s",
            ]
        )
        self._frame_times_file.flush()

        while not self._recording_queue.empty():
            try:
                self._recording_queue.get_nowait()
            except queue.Empty:
                break

        self._host_time_zero_s = (
            time.perf_counter()
            if host_time_zero_s is None
            else float(host_time_zero_s)
        )
        self._recorded_frame_count = 0
        self._dropped_frame_count = 0
        self._first_recorded_frame_host_time_s = None
        self._last_recorded_frame_host_time_s = None

        self._stop_recording_event.clear()
        with self._lock:
            self.is_recording = True

        self._recording_thread = threading.Thread(
            target=self._recording_loop,
            name="U20CameraRecording",
            daemon=True,
        )
        self._recording_thread.start()

        print(f"Recording started: {self.video_path}")
        print(f"Frame timestamps: {self.frame_timestamps_path}")
        return self.video_path

    def _recording_loop(self) -> None:
        try:
            while (
                not self._stop_recording_event.is_set()
                or not self._recording_queue.empty()
            ):
                try:
                    capture_sequence, host_time_s, frame = (
                        self._recording_queue.get(timeout=0.1)
                    )
                except queue.Empty:
                    continue

                with self._lock:
                    writer = self.video_writer

                if writer is None:
                    continue

                writer.write(frame)

                video_frame_index = self._recorded_frame_count
                host_elapsed_s = (
                    host_time_s - self._host_time_zero_s
                    if self._host_time_zero_s is not None
                    else float("nan")
                )

                if self._frame_times_writer is not None:
                    self._frame_times_writer.writerow(
                        [
                            video_frame_index,
                            capture_sequence,
                            f"{host_time_s:.9f}",
                            f"{host_elapsed_s:.9f}",
                        ]
                    )

                if self._first_recorded_frame_host_time_s is None:
                    self._first_recorded_frame_host_time_s = host_time_s
                self._last_recorded_frame_host_time_s = host_time_s

                self._recorded_frame_count += 1
                if (
                    self._frame_times_file is not None
                    and self._recorded_frame_count % 100 == 0
                ):
                    self._frame_times_file.flush()
        finally:
            print("Video recording thread stopped.")

    def request_stop_recording(self) -> None:
        """Stop accepting new frames immediately; finalization can follow later."""
        with self._lock:
            self.is_recording = False
        self._stop_recording_event.set()

    def stop_recording(self) -> Optional[Path]:
        """Stop recording, drain queued frames, and close MP4/timestamp files."""
        if (
            not self.is_recording
            and self._recording_thread is None
            and self.video_writer is None
        ):
            return self.video_path

        self.request_stop_recording()

        if self._recording_thread is not None and self._recording_thread.is_alive():
            self._recording_thread.join()
        self._recording_thread = None

        with self._lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

        if self._frame_times_file is not None:
            self._frame_times_file.flush()
            self._frame_times_file.close()
            self._frame_times_file = None
            self._frame_times_writer = None

        completed_path = self.video_path
        print(f"Recording stopped: {completed_path}")
        print(
            f"Recorded frames: {self._recorded_frame_count}; "
            f"dropped before writer: {self._dropped_frame_count}"
        )
        return completed_path

    @property
    def recorded_frame_count(self) -> int:
        return self._recorded_frame_count

    @property
    def dropped_frame_count(self) -> int:
        return self._dropped_frame_count

    @property
    def first_recorded_frame_host_time_s(self) -> Optional[float]:
        return self._first_recorded_frame_host_time_s

    @property
    def last_recorded_frame_host_time_s(self) -> Optional[float]:
        return self._last_recorded_frame_host_time_s

    # ------------------------------------------------------------------
    # Live view (main thread on macOS)
    # ------------------------------------------------------------------

    def view(self) -> None:
        if self.camera is None:
            self.start()

        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("U20Camera.view() must be called from the main thread.")

        if self.is_viewing:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        with self._lock:
            self.is_viewing = True

    def update_view(self) -> None:
        if not self.is_viewing:
            return

        frame = self.get_latest_frame()
        if frame is not None:
            display_frame = frame.copy()
            with self._lock:
                recording = self.is_recording

            if recording:
                text = "RECORDING"
                color = (0, 0, 255)
            else:
                text = "LIVE"
                color = (255, 255, 255)

            cv2.putText(
                display_frame,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, display_frame)

        cv2.waitKey(1)

        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                self.stop_view()
        except cv2.error:
            self.stop_view()

    def stop_view(self) -> None:
        if not self.is_viewing:
            return

        with self._lock:
            self.is_viewing = False

        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass

        print("Camera view stopped.")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        self.stop_view()
        self.stop_recording()

        self._stop_capture_event.set()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join()
        self._capture_thread = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        with self._lock:
            self._latest_frame = None
            self._latest_frame_host_time_s = None

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        print("Camera released.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
