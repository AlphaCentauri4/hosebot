import cv2
import platform
from datetime import datetime
from pathlib import Path


def open_camera(camera_index: int = 0):
    """Open the camera using the preferred backend for the current OS."""

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

    camera = cv2.VideoCapture(camera_index, backend)

    # Fall back to the default backend if needed.
    if not camera.isOpened():
        print("Preferred backend failed. Trying default backend...")
        camera.release()
        camera = cv2.VideoCapture(camera_index)

    return camera


def main() -> None:
    output_folder = Path("captures")
    output_folder.mkdir(parents=True, exist_ok=True)

    camera_index = 0
    camera = open_camera(camera_index)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}.\n"
            "Try another camera index (1, 2, ...) or check camera permissions."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)

    window_name = "U20CAM Live View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Camera started.")
    print("Press S to save the current frame.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("Error: Could not read a frame from the camera.")
                break

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                "S: Save image",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_frame,
                "Q: Quit",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S")):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                image_path = output_folder / f"u20cam_{timestamp}.jpg"

                if cv2.imwrite(str(image_path), frame):
                    print(f"Image saved: {image_path}")

                    cv2.putText(
                        display_frame,
                        "IMAGE SAVED",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imshow(window_name, display_frame)
                    cv2.waitKey(400)
                else:
                    print(f"Error: Could not save image to {image_path}")

            if key in (ord("q"), ord("Q")):
                print("Stopping camera.")
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Camera window closed.")
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()