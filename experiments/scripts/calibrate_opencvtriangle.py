#   To run this script
#
#   python calibrate_opencvtriangle.py --source 20260803_180357_10_20_75_glue.MOV --output 20260803_180357_10_20_75_glue.json --length-per-pixel 0.075 --length-unit mm

#
#
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CALIBRATION_WINDOW = "Triangle calibration"
MASK_WINDOW = "Calibrated mask"

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
VIDEOS_DIRECTORY = SCRIPT_DIRECTORY / "videos"


def nothing(_: int) -> None:
    pass


def parse_source(source: str) -> str | int:
    try:
        return int(source)
    except ValueError:
        return str((VIDEOS_DIRECTORY / source).resolve())


def parse_output(output: str) -> Path:
    return (VIDEOS_DIRECTORY / output).resolve()


def create_trackbars(initial: dict[str, int]) -> None:
    cv2.namedWindow(CALIBRATION_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)

    cv2.createTrackbar(
        "H lower",
        CALIBRATION_WINDOW,
        initial["h_lower"],
        179,
        nothing,
    )
    cv2.createTrackbar(
        "H upper",
        CALIBRATION_WINDOW,
        initial["h_upper"],
        179,
        nothing,
    )
    cv2.createTrackbar(
        "S lower",
        CALIBRATION_WINDOW,
        initial["s_lower"],
        255,
        nothing,
    )
    cv2.createTrackbar(
        "S upper",
        CALIBRATION_WINDOW,
        initial["s_upper"],
        255,
        nothing,
    )
    cv2.createTrackbar(
        "V lower",
        CALIBRATION_WINDOW,
        initial["v_lower"],
        255,
        nothing,
    )
    cv2.createTrackbar(
        "V upper",
        CALIBRATION_WINDOW,
        initial["v_upper"],
        255,
        nothing,
    )

    cv2.createTrackbar(
        "Open kernel",
        CALIBRATION_WINDOW,
        initial["open_kernel"],
        30,
        nothing,
    )
    cv2.createTrackbar(
        "Close kernel",
        CALIBRATION_WINDOW,
        initial["close_kernel"],
        30,
        nothing,
    )
    cv2.createTrackbar(
        "Min area",
        CALIBRATION_WINDOW,
        initial["min_area"],
        100_000,
        nothing,
    )


def get_trackbar_values() -> dict[str, int]:
    values = {
        "h_lower": cv2.getTrackbarPos(
            "H lower", CALIBRATION_WINDOW
        ),
        "h_upper": cv2.getTrackbarPos(
            "H upper", CALIBRATION_WINDOW
        ),
        "s_lower": cv2.getTrackbarPos(
            "S lower", CALIBRATION_WINDOW
        ),
        "s_upper": cv2.getTrackbarPos(
            "S upper", CALIBRATION_WINDOW
        ),
        "v_lower": cv2.getTrackbarPos(
            "V lower", CALIBRATION_WINDOW
        ),
        "v_upper": cv2.getTrackbarPos(
            "V upper", CALIBRATION_WINDOW
        ),
        "open_kernel": cv2.getTrackbarPos(
            "Open kernel", CALIBRATION_WINDOW
        ),
        "close_kernel": cv2.getTrackbarPos(
            "Close kernel", CALIBRATION_WINDOW
        ),
        "min_area": cv2.getTrackbarPos(
            "Min area", CALIBRATION_WINDOW
        ),
    }

    # Morphology kernels should be odd and at least 1.
    values["open_kernel"] = make_odd(
        max(1, values["open_kernel"])
    )
    values["close_kernel"] = make_odd(
        max(1, values["close_kernel"])
    )

    return values


def make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def create_green_mask(
    roi_frame: np.ndarray,
    parameters: dict[str, int],
) -> np.ndarray:
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    lower = np.array(
        [
            parameters["h_lower"],
            parameters["s_lower"],
            parameters["v_lower"],
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            parameters["h_upper"],
            parameters["s_upper"],
            parameters["v_upper"],
        ],
        dtype=np.uint8,
    )

    mask = cv2.inRange(hsv, lower, upper)

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            parameters["open_kernel"],
            parameters["open_kernel"],
        ),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            parameters["close_kernel"],
            parameters["close_kernel"],
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    return mask


def largest_valid_contour(
    mask: np.ndarray,
    min_area: float,
) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < min_area:
        return None

    return contour


def read_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()

    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read frame {frame_index} from the video."
        )

    return frame


def save_calibration(
    output_path: Path,
    source: str,
    frame: np.ndarray,
    frame_index: int,
    roi: tuple[int, int, int, int],
    parameters: dict[str, int],
    additional: dict[str, Any],
) -> None:
    x, y, width, height = roi

    calibration = {
        "schema_version": 1,
        "video": {
            "source": str(source),
            "calibration_frame_index": int(frame_index),
            "frame_width_px": int(frame.shape[1]),
            "frame_height_px": int(frame.shape[0]),
        },
        "roi": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        },
        "hsv": {
            "lower": [
                int(parameters["h_lower"]),
                int(parameters["s_lower"]),
                int(parameters["v_lower"]),
            ],
            "upper": [
                int(parameters["h_upper"]),
                int(parameters["s_upper"]),
                int(parameters["v_upper"]),
            ],
        },
        "morphology": {
            "open_kernel": int(parameters["open_kernel"]),
            "close_kernel": int(parameters["close_kernel"]),
            "open_iterations": 1,
            "close_iterations": 1,
        },
        "tracking": {
            "min_area_px2": float(parameters["min_area"]),
            "centerline_sections": int(
                additional["centerline_sections"]
            ),
            "endpoint_fraction": float(
                additional["endpoint_fraction"]
            ),
            "plot_history_seconds": float(
                additional["plot_history_seconds"]
            ),
        },
        "calibration": {
            "length_per_pixel": (
                None
                if additional["length_per_pixel"] is None
                else float(additional["length_per_pixel"])
            ),
            "length_unit": str(additional["length_unit"]),
        },
    }

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file first, then replace the destination.
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            calibration,
            file,
            indent=4,
            allow_nan=False,
        )
        file.flush()

    temporary_path.replace(output_path)

    print(f"Calibration saved to:\n{output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select a triangle ROI and calibrate HSV thresholds."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video filename or camera index.",
    )
    parser.add_argument(
        "--output",
        default="triangle_calibration.json",
        help="Output calibration JSON file.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Initial calibration frame index.",
    )
    parser.add_argument(
        "--centerline-sections",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--endpoint-fraction",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--plot-history",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--length-per-pixel",
        type=float,
        default=None,
        help="Optional physical calibration, e.g. mm/pixel.",
    )
    parser.add_argument(
        "--length-unit",
        default="px",
        help="Physical unit, for example mm.",
    )
    args = parser.parse_args()

    source = parse_source(args.source)
    output_path = parse_output(args.output)
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = max(0, args.frame)

    if total_frames > 0:
        frame_index = min(frame_index, total_frames - 1)

    frame = read_frame(capture, frame_index)

    print("Select the triangle region, then press Enter or Space.")
    print("Press C to cancel ROI selection.")

    selected = cv2.selectROI(
        "Select triangle ROI",
        frame,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow("Select triangle ROI")

    x, y, width, height = map(int, selected)

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("No valid ROI was selected.")

    initial = {
        "h_lower": 35,
        "h_upper": 90,
        "s_lower": 60,
        "s_upper": 255,
        "v_lower": 40,
        "v_upper": 255,
        "open_kernel": 5,
        "close_kernel": 9,
        "min_area": 500,
    }

    create_trackbars(initial)

    print("")
    print("Calibration controls:")
    print("  A / D : previous or next frame")
    print("  J / L : jump backward or forward 10 frames")
    print("  S     : save calibration")
    print("  Q/Esc : quit without saving")

    saved = False

    while True:
        roi_frame = frame[
            y : y + height,
            x : x + width,
        ].copy()

        parameters = get_trackbar_values()
        mask = create_green_mask(roi_frame, parameters)

        contour = largest_valid_contour(
            mask,
            parameters["min_area"],
        )

        preview = roi_frame.copy()

        if contour is not None:
            area = cv2.contourArea(contour)

            cv2.drawContours(
                preview,
                [contour],
                -1,
                (0, 255, 255),
                2,
            )

            moments = cv2.moments(contour)

            if abs(moments["m00"]) > 1e-9:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])

                cv2.circle(
                    preview,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1,
                )

            status = f"Detected area: {area:.0f} px2"
            status_color = (0, 255, 0)
        else:
            status = "No valid contour"
            status_color = (0, 0, 255)

        cv2.putText(
            preview,
            f"Frame: {frame_index}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            status,
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "S: save   A/D: frame   J/L: +/-10",
            (10, preview.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        masked_color = cv2.bitwise_and(
            roi_frame,
            roi_frame,
            mask=mask,
        )

        cv2.imshow(CALIBRATION_WINDOW, preview)
        cv2.imshow(MASK_WINDOW, masked_color)

        raw_key = cv2.waitKey(30)

        if raw_key < 0:
            continue

        key = raw_key & 0xFF

        # Convert uppercase ASCII letters to lowercase.
        if ord("A") <= key <= ord("Z"):
            key = ord(chr(key).lower())

        if key in (27, ord("q")):
            break

        if key == ord("s"):
            print("Saving calibration...")

            additional = {
                "centerline_sections": int(args.centerline_sections),
                "endpoint_fraction": float(args.endpoint_fraction),
                "plot_history_seconds": float(args.plot_history),
                "length_per_pixel": (
                    None
                    if args.length_per_pixel is None
                    else float(args.length_per_pixel)
                ),
                "length_unit": (
                    str(args.length_unit)
                    if args.length_per_pixel is not None
                    else "px"
                ),
            }

            try:
                save_calibration(
                    output_path=output_path,
                    source=str(source),
                    frame=frame,
                    frame_index=int(frame_index),
                    roi=(int(x), int(y), int(width), int(height)),
                    parameters=parameters,
                    additional=additional,
                )
            except Exception as error:
                print(f"Could not save calibration: {error}")
                continue

            saved = True
            print("Calibration saved successfully.")
            break

        frame_delta = 0

        if key == ord("a"):
            frame_delta = -1
        elif key == ord("d"):
            frame_delta = 1
        elif key == ord("j"):
            frame_delta = -10
        elif key == ord("l"):
            frame_delta = 10

        if frame_delta != 0:
            new_index = max(0, frame_index + frame_delta)

            if total_frames > 0:
                new_index = min(new_index, total_frames - 1)

            try:
                frame = read_frame(capture, new_index)
                frame_index = new_index
            except RuntimeError as error:
                print(error)

    capture.release()
    cv2.destroyAllWindows()

    if not saved:
        print("Calibration was not saved.")


if __name__ == "__main__":
    main()