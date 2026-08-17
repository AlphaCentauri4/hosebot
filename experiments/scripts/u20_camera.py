
import cv2
import platform
import threading
import time
import queue

from datetime import datetime
from pathlib import Path
from typing import Optional


class U20Camera:
    """
    Threaded camera interface.

    Separate threads are used for:

        - camera acquisition
        - video recording
        - live display

    The main application thread is therefore free to run the
    experiment, PID controller, DAQ, etc.

    Typical usage:

        camera = U20Camera(camera_index=1)

        camera.start()
        camera.view()

        camera.start_recording("experiment.mp4")

        # Experiment runs here without being blocked by the camera.

        camera.stop_recording()
        camera.release()
    """

    def __init__(
        self,
        camera_index: int = 1,
        output_folder: str | Path = "captures",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        window_name: str = "U20CAM Live View",
    ):
        self.camera_index = camera_index
        self.output_folder = Path(output_folder)

        self.width = width
        self.height = height
        self.fps = fps
        self.window_name = window_name

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.camera: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None

        self.actual_width = width
        self.actual_height = height
        self.actual_fps = fps

        self._capture_thread: Optional[threading.Thread] = None
        self._recording_thread: Optional[threading.Thread] = None
        self._view_thread: Optional[threading.Thread] = None

        self._stop_capture_event = threading.Event()
        self._stop_recording_event = threading.Event()
        self._stop_view_event = threading.Event()

        self._lock = threading.Lock()

        self._latest_frame = None

        self._recording_queue = queue.Queue(
            maxsize=1000
        )

        self.is_recording = False
        self.is_viewing = False

        self.video_path: Optional[Path] = None

    # ==================================================================
    # Camera setup
    # ==================================================================

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

        print(
            f"Opening camera using {backend_name} backend..."
        )

        camera = cv2.VideoCapture(
            self.camera_index,
            backend,
        )

        if not camera.isOpened():
            print(
                "Preferred backend failed. "
                "Trying default backend..."
            )

            camera.release()

            camera = cv2.VideoCapture(
                self.camera_index
            )

        return camera

    def start(self) -> None:
        """
        Open the camera and start frame acquisition.
        """

        if (
            self._capture_thread is not None
            and self._capture_thread.is_alive()
        ):
            return

        self.camera = self._open_camera()

        if not self.camera.isOpened():
            self.camera = None

            raise RuntimeError(
                f"Could not open camera index "
                f"{self.camera_index}.\n"
                "Try another camera index or check "
                "camera permissions."
            )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )

        self.camera.set(
            cv2.CAP_PROP_FPS,
            self.fps,
        )

        self.actual_width = int(
            self.camera.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        self.actual_height = int(
            self.camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        self.actual_fps = self.camera.get(
            cv2.CAP_PROP_FPS
        )

        if self.actual_fps <= 0:
            self.actual_fps = self.fps

        print(
            f"Camera started: "
            f"{self.actual_width}x"
            f"{self.actual_height} @ "
            f"{self.actual_fps:.1f} FPS"
        )

        with self._lock:
            self._latest_frame = None

        self._stop_capture_event.clear()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="U20CameraCapture",
            daemon=True,
        )

        self._capture_thread.start()

    # ==================================================================
    # Camera acquisition
    # ==================================================================

    def _capture_loop(self) -> None:
        """
        The only method that calls camera.read().
        """

        frame_interval = 1.0 / self.actual_fps

        try:
            while not self._stop_capture_event.is_set():

                loop_start = time.perf_counter()

                if self.camera is None:
                    break

                success, frame = self.camera.read()

                if not success or frame is None:
                    print(
                        "Warning: Could not read frame "
                        "from camera."
                    )

                    time.sleep(0.01)
                    continue

                with self._lock:
                    self._latest_frame = frame

                    recording = self.is_recording

                if recording:
                    try:
                        self._recording_queue.put_nowait(
                            frame.copy()
                        )
                    except queue.Full:
                        self._dropped_frame_count += 1

                elapsed = (
                    time.perf_counter()
                    - loop_start
                )

                sleep_time = (
                    frame_interval
                    - elapsed
                )

                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            print(
                "Camera capture thread stopped."
            )

    # ==================================================================
    # Frame access
    # ==================================================================

    def get_latest_frame(self):
        with self._lock:

            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()

    # ==================================================================
    # Image capture
    # ==================================================================

    def save_frame(self) -> Optional[Path]:
        frame = self.get_latest_frame()

        if frame is None:
            print(
                "No camera frame available to save."
            )

            return None

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S-%f"
        )

        image_path = (
            self.output_folder
            / f"u20cam_{timestamp}.jpg"
        )

        if not cv2.imwrite(
            str(image_path),
            frame,
        ):
            print(
                f"Error: Could not save image to "
                f"{image_path}"
            )

            return None

        print(
            f"Image saved: {image_path}"
        )

        return image_path

    # ==================================================================
    # Video recording
    # ==================================================================

    def start_recording(
        self,
        filename: Optional[str] = None,
    ) -> Path:
        if self.camera is None:
            raise RuntimeError(
                "Camera is not started. "
                "Call start() first."
            )

        if (
            self._capture_thread is None
            or not self._capture_thread.is_alive()
        ):
            raise RuntimeError(
                "Camera capture thread is not running."
            )

        if self.is_recording:
            raise RuntimeError(
                "Camera is already recording."
            )

        if filename is None:
            timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S-%f"
            )

            filename = (
                f"u20cam_{timestamp}.mp4"
            )

        self.video_path = (
            self.output_folder / filename
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.actual_fps,
            (
                self.actual_width,
                self.actual_height,
            ),
        )

        if not self.video_writer.isOpened():

            self.video_writer.release()
            self.video_writer = None
            self.video_path = None

            raise RuntimeError(
                "Could not create video file."
            )

        while not self._recording_queue.empty():
            try:
                self._recording_queue.get_nowait()
            except queue.Empty:
                break

        self._stop_recording_event.clear()

        with self._lock:
            self.is_recording = True

        self._recording_thread = threading.Thread(
            target=self._recording_loop,
            name="U20CameraRecording",
            daemon=True,
        )

        self._recording_thread.start()

        print(
            f"Recording started: "
            f"{self.video_path}"
        )

        return self.video_path

    def _recording_loop(self) -> None:
        """
        Write captured frames to disk.

        This thread never calls camera.read().
        """

        try:
            while (
                not self._stop_recording_event.is_set()
                or not self._recording_queue.empty()
            ):

                try:
                    frame = (
                        self._recording_queue.get(
                            timeout=0.1
                        )
                    )

                except queue.Empty:
                    continue

                with self._lock:
                    writer = self.video_writer

                if writer is not None:
                    writer.write(frame)

        finally:
            print(
                "Video recording thread stopped."
            )

    def stop_recording(self) -> Optional[Path]:
        with self._lock:

            if not self.is_recording:
                return self.video_path

            self.is_recording = False

        self._stop_recording_event.set()

        if (
            self._recording_thread is not None
            and self._recording_thread.is_alive()
        ):
            self._recording_thread.join()

        self._recording_thread = None

        with self._lock:

            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

        completed_path = self.video_path

        print(
            f"Recording stopped: "
            f"{completed_path}"
        )

        return completed_path

    # ==================================================================
    # Live view
    # ==================================================================

    def view(self) -> None:
        """
        Start the live camera view.

        This method is NON-BLOCKING.

        It starts a separate display thread and immediately
        returns to the caller.

        No keyboard input is accepted from the user.
        """

        if self.camera is None:
            self.start()

        if (
            self._view_thread is not None
            and self._view_thread.is_alive()
        ):
            return

        self._stop_view_event.clear()

        self._view_thread = threading.Thread(
            target=self._view_loop,
            name="U20CameraView",
            daemon=True,
        )

        self._view_thread.start()

    def _view_loop(self) -> None:
        """
        Display the most recent frame.

        This thread does not call camera.read().
        """

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        with self._lock:
            self.is_viewing = True

        try:
            while not self._stop_view_event.is_set():

                frame = self.get_latest_frame()

                if frame is not None:

                    display_frame = frame.copy()

                    with self._lock:
                        recording = self.is_recording

                    if recording:
                        text = "RECORDING"
                        color = (
                            0,
                            0,
                            255,
                        )
                    else:
                        text = "LIVE"
                        color = (
                            255,
                            255,
                            255,
                        )

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

                    cv2.imshow(
                        self.window_name,
                        display_frame,
                    )

                cv2.waitKey(1)

                try:
                    visible = cv2.getWindowProperty(
                        self.window_name,
                        cv2.WND_PROP_VISIBLE,
                    )

                    if visible < 1:
                        break

                except cv2.error:
                    break

                time.sleep(0.01)

        finally:

            with self._lock:
                self.is_viewing = False

            try:
                cv2.destroyWindow(
                    self.window_name
                )
            except cv2.error:
                pass

            print(
                "Camera view stopped."
            )

    def stop_view(self) -> None:
        """
        Stop the live view without stopping the camera
        or video recording.
        """

        self._stop_view_event.set()

        if (
            self._view_thread is not None
            and self._view_thread.is_alive()
        ):
            self._view_thread.join()

        self._view_thread = None

    # ==================================================================
    # Cleanup
    # ==================================================================

    def release(self) -> None:
        """
        Stop viewing, recording, acquisition, and release
        the camera.
        """

        self.stop_view()

        self.stop_recording()

        self._stop_capture_event.set()

        if (
            self._capture_thread is not None
            and self._capture_thread.is_alive()
        ):
            self._capture_thread.join()

        self._capture_thread = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        with self._lock:
            self._latest_frame = None

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        print(
            "Camera released."
        )

    # ==================================================================
    # Context manager
    # ==================================================================

    def __enter__(self):
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.release()
