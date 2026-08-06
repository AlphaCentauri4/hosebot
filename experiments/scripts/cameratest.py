import cv2
from datetime import datetime
from pathlib import Path


def main() -> None:
    # Folder where captured images will be stored.
    output_folder = Path("captures")
    output_folder.mkdir(parents=True, exist_ok=True)

    # On macOS, explicitly use the AVFoundation camera backend.
    # Change 0 to another index if the U20CAM is not camera 0.
    camera_index = 0
    camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check macOS camera permissions and try another index."
        )

    # Request the preferred camera resolution and frame rate.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)

    window_name = "U20CAM Live View"

    # Create a resizable live-view window.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Camera started.")
    print("Press S to save the current frame.")
    print("Press Q to quit.")

    try:
        while True:
            # Read one frame from the camera.
            success, frame = camera.read()

            if not success or frame is None:
                print("Error: Could not read a frame from the camera.")
                break

            # Create a copy so text overlays do not affect the saved image.
            display_frame = frame.copy()

            # Live command text shown inside the camera window.
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

            # Show the live camera stream.
            cv2.imshow(window_name, display_frame)

            # waitKey reads keyboard input while keeping the window responsive.
            key = cv2.waitKey(1) & 0xFF

            # Save the original frame when S is pressed.
            # The saved image does not include the command overlay.
            if key in (ord("s"), ord("S")):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                image_path = output_folder / f"u20cam_{timestamp}.jpg"

                saved = cv2.imwrite(str(image_path), frame)

                if saved:
                    print(f"Image saved: {image_path}")

                    # Temporarily display confirmation in the live window.
                    cv2.putText(
                        display_frame,
                        "IMAGE SAVED",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, display_frame)
                    cv2.waitKey(400)
                else:
                    print(f"Error: Could not save image to {image_path}")

            # Stop the program when Q is pressed.
            if key in (ord("q"), ord("Q")):
                print("Stopping camera.")
                break

            # Also stop when the user closes the live-view window.
            if cv2.getWindowProperty(
                window_name,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                print("Camera window closed.")
                break

    finally:
        # Always release the camera and close OpenCV windows.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()