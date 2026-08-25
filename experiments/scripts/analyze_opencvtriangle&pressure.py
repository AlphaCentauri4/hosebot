#python analyze_opencvtriangle.py 20260803_180357_10_20_7.5_glue
# run first basic_plotter.py to get the preprocessed dataframe
# then calibrate_opencvtriangle.py to produce the json with calibration parameters


#python 'analyze_opencvtriangle&pressure.py' 20260803_180357_10_20_7.5_glue

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import matplotlib
import cmcrameri.cm as cmc
import matplotlib.colors as colors
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import UnivariateSpline
except ImportError:  # optional; a polynomial fallback is used
    UnivariateSpline = None


ANALYSIS_WINDOW = "Triangle synchronized analysis"
MASK_WINDOW = "Triangle mask"
PLAYBACK_TRACKBAR = "Controller sample"

# Project layout, resolved relative to this Python file.
SCRIPT_DIR = Path(__file__).resolve().parent
#FILES_DIR = SCRIPT_DIR / "files"
VIDEOS_DIR = SCRIPT_DIR / "videos"
FIGURES_DIR = SCRIPT_DIR / "figures"

def format_axes(axes):
    for ax in axes:
        ax.xaxis.set_tick_params(labelsize=ticks_size)
        ax.yaxis.set_tick_params(labelsize=ticks_size)
        ax.grid(False)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
ticks_size = 14

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class Calibration:
    roi_x: int
    roi_y: int
    roi_width: int
    roi_height: int

    hsv_lower: np.ndarray
    hsv_upper: np.ndarray

    open_kernel: int
    close_kernel: int
    open_iterations: int
    close_iterations: int

    min_area: float
    centerline_sections: int
    endpoint_fraction: float

    length_per_pixel: float
    length_unit: str


@dataclass
class Measurement:
    frame_index: int
    video_time_s: float

    tip_x_px: float
    tip_y_px: float

    fixed_base_x_px: float
    fixed_base_y_px: float

    tip_longitudinal: float
    tip_transverse: float

    centerline_rms: float = np.nan
    centerline_max: float = np.nan


@dataclass
class SynchronizedData:
    controller_time: np.ndarray
    elapsed_time: np.ndarray
    flow1: np.ndarray
    flow2: np.ndarray
    tip_transverse: np.ndarray
    video_frame_index: np.ndarray


# ----------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------

def make_odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def resolve_path(
    path_text: str,
    reference_directory: Path,
) -> Path:
    path = Path(path_text).expanduser()

    if path.is_absolute():
        return path.resolve()

    current_candidate = Path.cwd() / path
    reference_candidate = reference_directory / path

    if current_candidate.exists():
        return current_candidate.resolve()

    if reference_candidate.exists():
        return reference_candidate.resolve()

    return current_candidate.resolve()


def get_video_time_seconds(
    frame_index: int,
    fps: float,
) -> float:
    return frame_index / fps


# ----------------------------------------------------------------------
# Calibration loading
# ----------------------------------------------------------------------

def load_calibration(
    path: Path,
) -> tuple[Calibration, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration JSON does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if raw.get("schema_version") != 1:
        raise ValueError(
            "Unsupported or missing calibration schema_version."
        )

    roi = raw["roi"]
    hsv = raw["hsv"]
    morphology = raw["morphology"]
    tracking = raw["tracking"]
    physical = raw.get("calibration", {})

    length_per_pixel = physical.get("length_per_pixel")

    if length_per_pixel is None:
        scale = 1.0
        unit = "px"
    else:
        scale = float(length_per_pixel)
        unit = str(physical.get("length_unit", "unit"))

    calibration = Calibration(
        roi_x=int(roi["x"]),
        roi_y=int(roi["y"]),
        roi_width=int(roi["width"]),
        roi_height=int(roi["height"]),
        hsv_lower=np.asarray(
            hsv["lower"],
            dtype=np.uint8,
        ),
        hsv_upper=np.asarray(
            hsv["upper"],
            dtype=np.uint8,
        ),
        open_kernel=make_odd(
            morphology.get("open_kernel", 5)
        ),
        close_kernel=make_odd(
            morphology.get("close_kernel", 9)
        ),
        open_iterations=max(
            0,
            int(morphology.get("open_iterations", 1)),
        ),
        close_iterations=max(
            0,
            int(morphology.get("close_iterations", 1)),
        ),
        min_area=float(
            tracking.get("min_area_px2", 500.0)
        ),
        centerline_sections=max(
            10,
            int(tracking.get("centerline_sections", 80)),
        ),
        endpoint_fraction=float(
            tracking.get("endpoint_fraction", 0.06)
        ),
        length_per_pixel=scale,
        length_unit=unit,
    )

    return calibration, raw


# ----------------------------------------------------------------------
# Controller CSV loading
# ----------------------------------------------------------------------

def load_controller_data(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Controller CSV does not exist: {path}"
        )

    dataframe = pd.read_csv(path)

    required = {
        "controller_time",
        "flow1",
        "flow2",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Controller CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    dataframe = dataframe[
        ["controller_time", "flow1", 'flow2']
    ].copy()

    dataframe["controller_time"] = pd.to_numeric(
        dataframe["controller_time"],
        errors="coerce",
    )

    dataframe["flow1"] = pd.to_numeric(
        dataframe["flow1"],
        errors="coerce",
    )
    dataframe["flow2"] = pd.to_numeric(
        dataframe["flow2"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["controller_time"]
    )

    dataframe = dataframe.sort_values(
        "controller_time"
    )

    dataframe = dataframe.drop_duplicates(
        subset="controller_time",
        keep="last",
    )

    dataframe = dataframe.reset_index(drop=True)

    if len(dataframe) < 2:
        raise ValueError(
            "The controller CSV must contain at least two valid timestamps."
        )

    controller_time = dataframe[
        "controller_time"
    ].to_numpy(dtype=np.float64)

    if np.any(np.diff(controller_time) <= 0):
        raise ValueError(
            "controller_time must be strictly increasing."
        )

    return dataframe


# ----------------------------------------------------------------------
# Triangle tracking
# ----------------------------------------------------------------------

class TriangleTracker:
    """
    Tracks the triangle relative to a fixed original base midpoint.

    The original triangle is assumed to be isosceles. The base midpoint,
    original longitudinal axis, and original transverse axis are initialized
    from the first valid segmented frame and then remain fixed.
    """

    def __init__(
        self,
        calibration: Calibration,
    ) -> None:
        self.calibration = calibration

        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                calibration.open_kernel,
                calibration.open_kernel,
            ),
        )

        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                calibration.close_kernel,
                calibration.close_kernel,
            ),
        )

        self.fixed_base: np.ndarray | None = None
        self.initial_tip: np.ndarray | None = None
        self.fixed_longitudinal: np.ndarray | None = None
        self.fixed_transverse: np.ndarray | None = None

        self.previous_tip: np.ndarray | None = None

    def reset_reference(self) -> None:
        self.fixed_base = None
        self.initial_tip = None
        self.fixed_longitudinal = None
        self.fixed_transverse = None
        self.previous_tip = None

    def get_roi(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        x0 = self.calibration.roi_x
        y0 = self.calibration.roi_y
        x1 = x0 + self.calibration.roi_width
        y1 = y0 + self.calibration.roi_height

        if (
            x0 < 0
            or y0 < 0
            or x1 > frame.shape[1]
            or y1 > frame.shape[0]
        ):
            raise ValueError(
                "The calibrated ROI lies outside the video frame."
            )

        return frame[y0:y1, x0:x1], x0, y0

    def segment(
        self,
        roi_frame: np.ndarray,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(
            roi_frame,
            cv2.COLOR_BGR2HSV,
        )

        mask = cv2.inRange(
            hsv,
            self.calibration.hsv_lower,
            self.calibration.hsv_upper,
        )

        if self.calibration.open_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                self.open_kernel,
                iterations=self.calibration.open_iterations,
            )

        if self.calibration.close_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                self.close_kernel,
                iterations=self.calibration.close_iterations,
            )

        return mask

    def largest_contour(
        self,
        mask: np.ndarray,
    ) -> np.ndarray | None:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        if cv2.contourArea(contour) < self.calibration.min_area:
            return None

        return contour

    @staticmethod
    def principal_axis(
        points_xy: np.ndarray,
    ) -> np.ndarray:
        centered = points_xy - np.mean(
            points_xy,
            axis=0,
        )

        covariance = np.cov(
            centered,
            rowvar=False,
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        axis = eigenvectors[
            :,
            np.argmax(eigenvalues),
        ]

        norm = float(np.linalg.norm(axis))

        if norm < 1e-12:
            raise ValueError(
                "Could not calculate the principal axis."
            )

        return axis / norm

    @staticmethod
    def endpoint_width(
        points_xy: np.ndarray,
        axis: np.ndarray,
        positive_end: bool,
        fraction: float = 0.12,
    ) -> float:
        centroid = np.mean(
            points_xy,
            axis=0,
        )

        normal = np.array(
            [-axis[1], axis[0]],
            dtype=np.float64,
        )

        relative = points_xy - centroid
        longitudinal = relative @ axis
        transverse = relative @ normal

        minimum = float(np.min(longitudinal))
        maximum = float(np.max(longitudinal))
        span = max(maximum - minimum, 1.0)
        band = fraction * span

        if positive_end:
            values = transverse[
                longitudinal >= maximum - band
            ]
        else:
            values = transverse[
                longitudinal <= minimum + band
            ]

        if len(values) < 3:
            return np.inf

        return float(np.ptp(values))

    def orient_initial_axis(
        self,
        points_xy: np.ndarray,
        axis: np.ndarray,
    ) -> np.ndarray:
        positive_width = self.endpoint_width(
            points_xy,
            axis,
            positive_end=True,
        )

        negative_width = self.endpoint_width(
            points_xy,
            axis,
            positive_end=False,
        )

        return (
            axis
            if positive_width <= negative_width
            else -axis
        )

    def estimate_tip_from_axis(
        self,
        contour_points: np.ndarray,
        axis: np.ndarray,
    ) -> np.ndarray:
        centroid = np.mean(
            contour_points,
            axis=0,
        )

        projection = (
            contour_points - centroid
        ) @ axis

        minimum = float(np.min(projection))
        maximum = float(np.max(projection))
        span = max(maximum - minimum, 1.0)

        band = (
            self.calibration.endpoint_fraction
            * span
        )

        tip_points = contour_points[
            projection >= maximum - band
        ]

        if len(tip_points) == 0:
            return contour_points[
                np.argmax(projection)
            ].copy()

        return np.median(
            tip_points,
            axis=0,
        )

    def estimate_initial_base_midpoint(
        self,
        contour_points: np.ndarray,
        longitudinal_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centroid = np.mean(
            contour_points,
            axis=0,
        )

        transverse_axis = np.array(
            [
                -longitudinal_axis[1],
                longitudinal_axis[0],
            ],
            dtype=np.float64,
        )

        relative = contour_points - centroid

        longitudinal = (
            relative @ longitudinal_axis
        )

        transverse = (
            relative @ transverse_axis
        )

        minimum = float(np.min(longitudinal))
        maximum = float(np.max(longitudinal))
        span = max(maximum - minimum, 1.0)

        base_band_width = max(
            0.12 * span,
            2.0,
        )

        base_selector = (
            longitudinal
            <= minimum + base_band_width
        )

        base_points = contour_points[
            base_selector
        ]

        base_transverse = transverse[
            base_selector
        ]

        if len(base_points) < 4:
            raise ValueError(
                "Not enough points to determine the base."
            )

        transverse_span = float(
            np.ptp(base_transverse)
        )

        corner_band = max(
            0.08 * transverse_span,
            1.0,
        )

        transverse_min = float(
            np.min(base_transverse)
        )

        transverse_max = float(
            np.max(base_transverse)
        )

        corner_1_points = base_points[
            base_transverse
            <= transverse_min + corner_band
        ]

        corner_2_points = base_points[
            base_transverse
            >= transverse_max - corner_band
        ]

        if (
            len(corner_1_points) == 0
            or len(corner_2_points) == 0
        ):
            raise ValueError(
                "Could not identify both base corners."
            )

        corner_1 = np.median(
            corner_1_points,
            axis=0,
        )

        corner_2 = np.median(
            corner_2_points,
            axis=0,
        )

        base_midpoint = 0.5 * (
            corner_1 + corner_2
        )

        return (
            base_midpoint,
            corner_1,
            corner_2,
        )

    def initialize_fixed_geometry(
        self,
        contour_points: np.ndarray,
    ) -> None:
        axis = self.principal_axis(
            contour_points
        )

        axis = self.orient_initial_axis(
            contour_points,
            axis,
        )

        initial_tip = self.estimate_tip_from_axis(
            contour_points,
            axis,
        )

        base_midpoint, _, _ = (
            self.estimate_initial_base_midpoint(
                contour_points,
                axis,
            )
        )

        direction = initial_tip - base_midpoint
        length = float(np.linalg.norm(direction))

        if length < 5.0:
            raise ValueError(
                "The initial base-to-tip length is too small."
            )

        self.fixed_base = base_midpoint.copy()
        self.initial_tip = initial_tip.copy()

        self.fixed_longitudinal = (
            direction / length
        )

        self.fixed_transverse = np.array(
            [
                -self.fixed_longitudinal[1],
                self.fixed_longitudinal[0],
            ],
            dtype=np.float64,
        )

        self.previous_tip = initial_tip.copy()

    def orient_current_axis(
        self,
        contour_points: np.ndarray,
        axis: np.ndarray,
    ) -> np.ndarray:
        if self.previous_tip is not None:
            positive_tip = self.estimate_tip_from_axis(
                contour_points,
                axis,
            )

            negative_tip = self.estimate_tip_from_axis(
                contour_points,
                -axis,
            )

            positive_distance = float(
                np.linalg.norm(
                    positive_tip - self.previous_tip
                )
            )

            negative_distance = float(
                np.linalg.norm(
                    negative_tip - self.previous_tip
                )
            )

            return (
                axis
                if positive_distance <= negative_distance
                else -axis
            )

        if self.fixed_longitudinal is not None:
            return (
                axis
                if np.dot(
                    axis,
                    self.fixed_longitudinal,
                ) >= 0
                else -axis
            )

        return self.orient_initial_axis(
            contour_points,
            axis,
        )

    def detect_current_tip(
        self,
        contour_points: np.ndarray,
    ) -> np.ndarray:
        axis = self.principal_axis(
            contour_points
        )

        axis = self.orient_current_axis(
            contour_points,
            axis,
        )

        tip = self.estimate_tip_from_axis(
            contour_points,
            axis,
        )

        self.previous_tip = tip.copy()
        return tip

    @staticmethod
    def filled_contour_mask(
        contour: np.ndarray,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        mask = np.zeros(
            shape[:2],
            dtype=np.uint8,
        )

        cv2.drawContours(
            mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )

        return mask

    def extract_centerline(
        self,
        object_mask: np.ndarray,
        tip: np.ndarray,
    ) -> np.ndarray | None:
        if self.fixed_base is None:
            return None

        chord = tip - self.fixed_base
        length = float(np.linalg.norm(chord))

        if length < 5.0:
            return None

        chord_axis = chord / length

        chord_normal = np.array(
            [-chord_axis[1], chord_axis[0]],
            dtype=np.float64,
        )

        rows, columns = np.nonzero(
            object_mask
        )

        if len(rows) < 20:
            return None

        pixels = np.column_stack(
            (columns, rows)
        ).astype(np.float64)

        relative = pixels - self.fixed_base

        longitudinal = relative @ chord_axis
        transverse = relative @ chord_normal

        base_tolerance = max(
            0.02 * length,
            2.0,
        )

        valid = (
            (longitudinal >= -base_tolerance)
            & (longitudinal <= length)
        )

        longitudinal = longitudinal[valid]
        transverse = transverse[valid]

        if len(longitudinal) < 20:
            return None

        edges = np.linspace(
            0.0,
            length,
            self.calibration.centerline_sections + 1,
        )

        points = [
            self.fixed_base.copy()
        ]

        for section_index, (left, right) in enumerate(
            zip(edges[:-1], edges[1:])
        ):
            selector = (
                (longitudinal >= left)
                & (longitudinal < right)
            )

            if np.count_nonzero(selector) < 3:
                continue

            section_s = float(
                np.median(
                    longitudinal[selector]
                )
            )

            section_n = float(
                np.median(
                    transverse[selector]
                )
            )

            if section_index < 3:
                denominator = max(
                    edges[min(3, len(edges) - 1)],
                    1.0,
                )

                blend = float(
                    np.clip(
                        section_s / denominator,
                        0.0,
                        1.0,
                    )
                )

                section_n *= blend

            point = (
                self.fixed_base
                + section_s * chord_axis
                + section_n * chord_normal
            )

            points.append(point)

        points.append(tip.copy())

        centerline = np.asarray(
            points,
            dtype=np.float64,
        )

        if len(centerline) < 5:
            return None

        return centerline

    def calculate_centerline_deviation(
        self,
        centerline: np.ndarray,
        tip: np.ndarray,
    ) -> tuple[float, float]:
        if self.fixed_base is None:
            return np.nan, np.nan

        chord = tip - self.fixed_base
        length = float(np.linalg.norm(chord))

        if length < 1e-9:
            return np.nan, np.nan

        chord_axis = chord / length

        chord_normal = np.array(
            [-chord_axis[1], chord_axis[0]],
            dtype=np.float64,
        )

        relative = centerline - self.fixed_base

        longitudinal = relative @ chord_axis
        transverse = relative @ chord_normal

        selector = (
            (longitudinal > 0)
            & (longitudinal < length)
        )

        deviations = transverse[selector]

        if len(deviations) == 0:
            return np.nan, np.nan

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(deviations)
                )
            )
        )

        maximum = float(
            np.max(
                np.abs(deviations)
            )
        )

        return rms, maximum

    def process_tip_only(
        self,
        frame: np.ndarray,
        frame_index: int,
        video_time_s: float,
    ) -> Measurement | None:
        roi_frame, x0, y0 = self.get_roi(
            frame
        )

        mask = self.segment(
            roi_frame
        )

        contour = self.largest_contour(
            mask
        )

        if contour is None:
            return None

        contour_points = contour[
            :,
            0,
            :,
        ].astype(
            np.float64,
            copy=False,
        )

        if self.fixed_base is None:
            try:
                self.initialize_fixed_geometry(
                    contour_points
                )
            except ValueError:
                return None

        assert self.fixed_base is not None
        assert self.fixed_longitudinal is not None
        assert self.fixed_transverse is not None

        tip_roi = self.detect_current_tip(
            contour_points
        )

        relative_tip = (
            tip_roi - self.fixed_base
        )

        longitudinal_px = float(
            relative_tip
            @ self.fixed_longitudinal
        )

        transverse_px = float(
            relative_tip
            @ self.fixed_transverse
        )

        scale = (
            self.calibration.length_per_pixel
        )

        tip_global = (
            tip_roi
            + np.array([x0, y0])
        )

        base_global = (
            self.fixed_base
            + np.array([x0, y0])
        )

        return Measurement(
            frame_index=frame_index,
            video_time_s=video_time_s,
            tip_x_px=float(tip_global[0]),
            tip_y_px=float(tip_global[1]),
            fixed_base_x_px=float(
                base_global[0]
            ),
            fixed_base_y_px=float(
                base_global[1]
            ),
            tip_longitudinal=(
                longitudinal_px * scale
            ),
            tip_transverse=(
                transverse_px * scale
            ),
        )

    def process_full(
        self,
        frame: np.ndarray,
        frame_index: int,
        video_time_s: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Measurement | None,
    ]:
        roi_frame, x0, y0 = self.get_roi(
            frame
        )

        mask = self.segment(
            roi_frame
        )

        contour = self.largest_contour(
            mask
        )

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x0, y0),
            (
                x0 + self.calibration.roi_width,
                y0 + self.calibration.roi_height,
            ),
            (180, 180, 180),
            1,
        )

        if contour is None:
            return overlay, mask, None

        contour_points = contour[
            :,
            0,
            :,
        ].astype(
            np.float64,
            copy=False,
        )

        if self.fixed_base is None:
            try:
                self.initialize_fixed_geometry(
                    contour_points
                )
            except ValueError:
                return overlay, mask, None

        assert self.fixed_base is not None
        assert self.fixed_longitudinal is not None
        assert self.fixed_transverse is not None

        tip_roi = self.detect_current_tip(
            contour_points
        )

        object_mask = self.filled_contour_mask(
            contour,
            roi_frame.shape,
        )

        centerline_roi = self.extract_centerline(
            object_mask,
            tip_roi,
        )

        rms_px = np.nan
        maximum_px = np.nan

        if centerline_roi is not None:
            rms_px, maximum_px = (
                self.calculate_centerline_deviation(
                    centerline_roi,
                    tip_roi,
                )
            )

        relative_tip = (
            tip_roi - self.fixed_base
        )

        longitudinal_px = float(
            relative_tip
            @ self.fixed_longitudinal
        )

        transverse_px = float(
            relative_tip
            @ self.fixed_transverse
        )

        scale = (
            self.calibration.length_per_pixel
        )

        offset = np.array(
            [x0, y0],
            dtype=np.float64,
        )

        tip_global = tip_roi + offset
        base_global = self.fixed_base + offset

        contour_global = contour.copy()
        contour_global[:, 0, 0] += x0
        contour_global[:, 0, 1] += y0

        cv2.drawContours(
            overlay,
            [contour_global],
            -1,
            (0, 255, 255),
            2,
        )

        base_point = tuple(
            np.round(base_global).astype(int)
        )

        tip_point = tuple(
            np.round(tip_global).astype(int)
        )

        # Cyan fixed-base-to-current-tip line.
        cv2.line(
            overlay,
            base_point,
            tip_point,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Magenta centerline anchored at the fixed base midpoint.
        if centerline_roi is not None:
            centerline_global = (
                centerline_roi + offset
            )

            cv2.polylines(
                overlay,
                [
                    np.round(
                        centerline_global
                    ).astype(np.int32)
                ],
                False,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.circle(
            overlay,
            base_point,
            7,
            (255, 0, 0),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            overlay,
            tip_point,
            7,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        measurement = Measurement(
            frame_index=frame_index,
            video_time_s=video_time_s,
            tip_x_px=float(tip_global[0]),
            tip_y_px=float(tip_global[1]),
            fixed_base_x_px=float(
                base_global[0]
            ),
            fixed_base_y_px=float(
                base_global[1]
            ),
            tip_longitudinal=(
                longitudinal_px * scale
            ),
            tip_transverse=(
                transverse_px * scale
            ),
            centerline_rms=rms_px * scale,
            centerline_max=maximum_px * scale,
        )

        return overlay, mask, measurement


# ----------------------------------------------------------------------
# Fast preanalysis
# ----------------------------------------------------------------------

def preanalyze_tip_position(
    capture: cv2.VideoCapture,
    tracker: TriangleTracker,
    total_frames: int,
    fps: float,
    frame_step: int,
) -> dict[int, Measurement]:
    frame_step = max(1, int(frame_step))

    tracker.reset_reference()

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        0,
    )

    measurements: dict[int, Measurement] = {}

    print(
        "Preanalyzing tip position "
        f"every {frame_step} frame(s)..."
    )

    last_percent = -1
    frame_index = 0

    while frame_index < total_frames:
        ok, frame = capture.read()

        if not ok or frame is None:
            break

        if frame_index % frame_step == 0:
            video_time_s = (
                frame_index / fps
            )

            measurement = tracker.process_tip_only(
                frame=frame,
                frame_index=frame_index,
                video_time_s=video_time_s,
            )

            if measurement is not None:
                measurements[
                    frame_index
                ] = measurement

        percent = int(
            100
            * (frame_index + 1)
            / total_frames
        )

        if percent != last_percent:
            print(
                f"\rPreanalysis: {percent:3d}%",
                end="",
                flush=True,
            )
            last_percent = percent

        frame_index += 1

    print()

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        0,
    )

    return measurements


def save_tip_positions_csv(
    path: Path,
    measurements: dict[int, Measurement],
    unit: str,
) -> None:
    """Save raw video-time tip positions in the figures directory."""
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "frame_index": measurement.frame_index,
            "video_time_s": measurement.video_time_s,
            f"tip_transverse_{unit}": measurement.tip_transverse,
        }
        for _, measurement in sorted(measurements.items())
    ]

    if not rows:
        raise ValueError("No tip-position measurements are available to save.")

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def initialize_tracker_reference(
    capture: cv2.VideoCapture,
    tracker: TriangleTracker,
    total_frames: int,
    fps: float,
    maximum_search_frames: int = 500,
) -> None:
    """
    Initialize the fixed triangle geometry before interactive playback.

    This is required when synchronized data are loaded from cache and the
    preanalysis pass is skipped.
    """
    tracker.reset_reference()

    search_limit = min(
        total_frames,
        maximum_search_frames,
    )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        0,
    )

    for frame_index in range(search_limit):
        ok, frame = capture.read()

        if not ok or frame is None:
            break

        measurement = tracker.process_tip_only(
            frame=frame,
            frame_index=frame_index,
            video_time_s=frame_index / fps,
        )

        if measurement is not None:
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                0,
            )
            return

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        0,
    )

    raise RuntimeError(
        "Could not initialize the fixed triangle geometry "
        "from the first video frames."
    )


# ----------------------------------------------------------------------
# Synchronization
# ----------------------------------------------------------------------

def synchronize_video_and_controller(
    controller_dataframe: pd.DataFrame,
    video_measurements: dict[int, Measurement],
    fps: float,
    total_frames: int,
) -> SynchronizedData:
    controller_time = controller_dataframe[
        "controller_time"
    ].to_numpy(dtype=np.float64)

    flow1 = controller_dataframe[
        "flow1"
    ].to_numpy(dtype=np.float64)
    flow2 = controller_dataframe[
        "flow2"
    ].to_numpy(dtype=np.float64)

    elapsed_time = (
        controller_time
        - controller_time[0]
    )

    measurement_frames = np.asarray(
        sorted(video_measurements),
        dtype=np.int64,
    )

    if len(measurement_frames) == 0:
        raise ValueError(
            "No valid video measurements are available."
        )

    video_time = np.asarray(
        [
            video_measurements[
                int(frame)
            ].video_time_s
            for frame in measurement_frames
        ],
        dtype=np.float64,
    )

    video_tip = np.asarray(
        [
            video_measurements[
                int(frame)
            ].tip_transverse
            for frame in measurement_frames
        ],
        dtype=np.float64,
    )

    valid = (
        np.isfinite(video_time)
        & np.isfinite(video_tip)
    )

    video_time = video_time[valid]
    video_tip = video_tip[valid]

    if len(video_time) == 0:
        raise ValueError(
            "All video tip measurements are invalid."
        )

    order = np.argsort(video_time)

    video_time = video_time[order]
    video_tip = video_tip[order]

    video_time, unique_indices = np.unique(
        video_time,
        return_index=True,
    )

    video_tip = video_tip[
        unique_indices
    ]

    if len(video_time) == 1:
        synchronized_tip = np.full(
            elapsed_time.shape,
            video_tip[0],
            dtype=np.float64,
        )
    else:
        synchronized_tip = np.interp(
            elapsed_time,
            video_time,
            video_tip,
            left=video_tip[0],
            right=video_tip[-1],
        )

    # Zero-order hold for video frames:
    # repeated controller samples can map to the same frame.
    frame_indices = np.floor(
        elapsed_time * fps
    ).astype(np.int64)

    frame_indices = np.clip(
        frame_indices,
        0,
        total_frames - 1,
    )

    return SynchronizedData(
        controller_time=controller_time,
        elapsed_time=elapsed_time,
        flow1=flow1,
        flow2=flow2,
        tip_transverse=synchronized_tip,
        video_frame_index=frame_indices,
    )

def save_synchronized_csv(
    path: Path,
    synchronized: SynchronizedData,
    unit: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        {
            "controller_time": synchronized.controller_time,
            "elapsed_time_s": synchronized.elapsed_time,
            "video_frame_index": synchronized.video_frame_index,
            f"tip_transverse_{unit}": synchronized.tip_transverse,
            "flow1": synchronized.flow1,
            "flow2": synchronized.flow2,
        }
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)


def load_synchronized_csv(
    path: Path,
    unit: str,
    controller_dataframe: pd.DataFrame,
    total_frames: int,
) -> SynchronizedData:
    dataframe = pd.read_csv(path)

    tip_column = (
        f"tip_transverse_{unit}"
    )

    required = {
        "controller_time",
        "elapsed_time_s",
        "video_frame_index",
        tip_column,
        "flow1",
        "flow2",
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Cached synchronized CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    if dataframe.empty:
        raise ValueError(
            "Cached synchronized CSV is empty."
        )

    cached_controller_time = pd.to_numeric(
        dataframe["controller_time"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    current_controller_time = (
        controller_dataframe[
            "controller_time"
        ].to_numpy(dtype=np.float64)
    )

    if (
        len(cached_controller_time)
        != len(current_controller_time)
    ):
        raise ValueError(
            "Cached synchronized CSV does not match "
            "the current controller-data length."
        )

    if not np.allclose(
        cached_controller_time,
        current_controller_time,
        rtol=1e-9,
        atol=1e-12,
        equal_nan=False,
    ):
        raise ValueError(
            "Cached synchronized CSV uses a different "
            "controller_time timeline."
        )

    frame_indices = pd.to_numeric(
        dataframe["video_frame_index"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if np.any(~np.isfinite(frame_indices)):
        raise ValueError(
            "Cached video_frame_index contains invalid values."
        )

    frame_indices = frame_indices.astype(
        np.int64
    )

    if (
        np.min(frame_indices) < 0
        or np.max(frame_indices) >= total_frames
    ):
        raise ValueError(
            "Cached video-frame indices are outside "
            "the current video."
        )

    tip_transverse = pd.to_numeric(
        dataframe[tip_column],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    elapsed_time = pd.to_numeric(
        dataframe["elapsed_time_s"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    flow1 = pd.to_numeric(
        dataframe["flow1"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    flow2 = pd.to_numeric(
        dataframe["flow2"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if np.any(~np.isfinite(tip_transverse)):
        raise ValueError(
            "Cached tip displacement contains invalid values."
        )

    if np.any(~np.isfinite(elapsed_time)):
        raise ValueError(
            "Cached elapsed time contains invalid values."
        )
    return SynchronizedData(
        controller_time=cached_controller_time,
        elapsed_time=elapsed_time,
        flow1=flow1,
        flow2=flow2,
        tip_transverse=tip_transverse,
        video_frame_index=frame_indices,
    )

# ----------------------------------------------------------------------
# Live plotting
# ----------------------------------------------------------------------

class LivePlot:
    def __init__(
        self,
        synchronized: SynchronizedData,
        unit: str,
    ) -> None:
        self.synchronized = synchronized

        plt.ion()

        self.figure, self.axes = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            sharex=True,
        )

        self.tip_line, = self.axes[0].plot(
            [],
            [],
            label="Tip transverse position",
        )

        self.tip_marker, = self.axes[0].plot(
            [],
            [],
            marker="o",
            linestyle="None",
            label="Current sample",
        )

        self.flow_line, = self.axes[1].plot(
            [],
            [],
            label="flow1",
        )

        self.flow_marker, = self.axes[1].plot(
            [],
            [],
            marker="o",
            linestyle="None",
            label="Current sample",
        )

        self.axes[0].set_ylabel(
            f"Tip displacement [{unit}]"
        )

        self.axes[1].set_ylabel(
            "flow1"
        )

        self.axes[1].set_xlabel(
            "controller_time [s]"
        )

        for axis in self.axes:
            axis.grid(True)
            axis.legend()

        self.axes[0].set_xlim(
            synchronized.controller_time[0],
            synchronized.controller_time[-1],
        )

        self.axes[1].set_xlim(
            synchronized.controller_time[0],
            synchronized.controller_time[-1],
        )

        self.figure.tight_layout()
        self.figure.show()

    def update(
        self,
        controller_index: int,
    ) -> None:
        controller_index = int(
            np.clip(
                controller_index,
                0,
                len(
                    self.synchronized.controller_time
                ) - 1,
            )
        )

        stop = controller_index + 1

        visible_time = (
            self.synchronized.controller_time[
                :stop
            ]
        )

        visible_tip = (
            self.synchronized.tip_transverse[
                :stop
            ]
        )

        visible_flow = (
            self.synchronized.flow1[
                :stop
            ]
        )

        self.tip_line.set_data(
            visible_time,
            visible_tip,
        )

        self.flow_line.set_data(
            visible_time,
            visible_flow,
        )

        current_time = float(
            self.synchronized.controller_time[
                controller_index
            ]
        )

        current_tip = float(
            self.synchronized.tip_transverse[
                controller_index
            ]
        )

        current_flow = float(
            self.synchronized.flow1[
                controller_index
            ]
        )

        self.tip_marker.set_data(
            [current_time],
            [current_tip],
        )

        self.flow_marker.set_data(
            [current_time],
            [current_flow],
        )

        for axis in self.axes:
            axis.relim()
            axis.autoscale_view(
                scalex=False,
                scaley=True,
            )

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)


# ----------------------------------------------------------------------
# Playback controller
# ----------------------------------------------------------------------

class PlaybackController:
    def __init__(
        self,
        number_of_samples: int,
    ) -> None:
        self.number_of_samples = max(
            1,
            int(number_of_samples),
        )

        self.pending_index: int | None = None
        self.updating_trackbar = False

        cv2.namedWindow(
            ANALYSIS_WINDOW,
            cv2.WINDOW_NORMAL,
        )

        cv2.createTrackbar(
            PLAYBACK_TRACKBAR,
            ANALYSIS_WINDOW,
            0,
            self.number_of_samples - 1,
            self._on_trackbar,
        )

    def _on_trackbar(
        self,
        index: int,
    ) -> None:
        if self.updating_trackbar:
            return

        self.pending_index = int(
            np.clip(
                index,
                0,
                self.number_of_samples - 1,
            )
        )

    def consume_seek(
        self,
    ) -> int | None:
        index = self.pending_index
        self.pending_index = None
        return index

    def update_position(
        self,
        index: int,
    ) -> None:
        index = int(
            np.clip(
                index,
                0,
                self.number_of_samples - 1,
            )
        )

        self.updating_trackbar = True

        cv2.setTrackbarPos(
            PLAYBACK_TRACKBAR,
            ANALYSIS_WINDOW,
            index,
        )

        self.updating_trackbar = False



# ----------------------------------------------------------------------
# Pressure reconstruction extension
# ----------------------------------------------------------------------

@dataclass
class PressureConfig:
    """Configuration for the inverse beam-pressure reconstruction.

    Equations (12)--(16) of Triangular_Valve_Model copy.pdf are implemented
    explicitly for the ideal triangular beam. A finite-nose correction is also
    reported because the experimental valve has a rounded, non-zero-width tip.
    """

    young_modulus_pa: float = 1.0e6
    thickness_m: float = 1.0e-3
    base_width_m: float = 5.0e-3
    apex_width_m: float = 0.25e-3
    air_density_kg_m3: float = 1.204
    pressure_coefficient: float = 1.0
    inlet_area_m2: float = 8.0e-3 * 7.5e-3
    standard_temperature_k: float = 288.15
    standard_pressure_pa: float = 101325.0
    actual_temperature_k: float = 293.15
    actual_pressure_pa: float = 101325.0
    curvature_smoothing_m: float = 2.0e-5
    local_fit_fraction: float = 0.10
    x_min_fraction: float = 0.03
    x_max_fraction: float = 0.90
    n_profile: int = 80
    use_finite_tip_correction: bool = True
    pressure_scale_clip_pa: float | None = None
    pressure_model: str = "uniform"


@dataclass
class PressureSnapshot:
    frame_index: int
    video_time_s: float
    x_m: np.ndarray
    curvature_1_m: np.ndarray
    pressure_uniform_pa: np.ndarray
    pressure_nonuniform_pa: np.ndarray
    distributed_load_uniform_N_m: np.ndarray
    distributed_load_nonuniform_N_m: np.ndarray
    force_uniform_N: float
    force_uniform_finite_tip_N: float
    force_nonuniform_N: float
    moment_uniform_Nm: float
    moment_uniform_finite_tip_Nm: float
    moment_nonuniform_Nm: float
    generalized_force_uniform_N: float
    generalized_force_nonuniform_N: float
    curvature_clamp_1_m: float
    delta_p_uniform_pdf_pa: float
    delta_p_uniform_finite_tip_pa: float
    velocity_m_s: float
    dynamic_pressure_pa: float
    total_flow_slpm: float


@dataclass
class PressureAnalysis:
    controller_time: np.ndarray
    total_flow_slpm: np.ndarray
    velocity_m_s: np.ndarray
    dynamic_pressure_pa: np.ndarray
    flow_pressure_scale_pa: np.ndarray
    delta_p_uniform_pdf_pa: np.ndarray
    delta_p_uniform_finite_tip_pa: np.ndarray
    delta_p_nonuniform_mean_pa: np.ndarray
    force_uniform_pdf_N: np.ndarray
    force_uniform_finite_tip_N: np.ndarray
    force_nonuniform_N: np.ndarray
    moment_uniform_pdf_Nm: np.ndarray
    moment_uniform_finite_tip_Nm: np.ndarray
    moment_nonuniform_Nm: np.ndarray
    snapshots: dict[int, PressureSnapshot]
    geometry: dict[str, float]


def _flow_slpm_to_actual_m3_s(
    flow_slpm: float | np.ndarray,
    cfg: PressureConfig,
) -> np.ndarray:
    """Convert SLPM to actual volumetric flow using ideal-gas scaling."""
    q_std = np.asarray(flow_slpm, dtype=float) * 1.0e-3 / 60.0
    return q_std * (
        cfg.actual_temperature_k / cfg.standard_temperature_k
    ) * (
        cfg.standard_pressure_pa / cfg.actual_pressure_pa
    )


def _airspeed_from_flow(flow_slpm: float, cfg: PressureConfig) -> float:
    q_actual = float(_flow_slpm_to_actual_m3_s(flow_slpm, cfg))
    if cfg.inlet_area_m2 <= 0:
        return np.nan
    return q_actual / cfg.inlet_area_m2


def _dynamic_pressure_from_flow(flow_slpm: float, cfg: PressureConfig) -> float:
    u = _airspeed_from_flow(flow_slpm, cfg)
    return 0.5 * cfg.air_density_kg_m3 * u * u


def _extract_centerline_physical(
    tracker: TriangleTracker,
    frame: np.ndarray,
) -> np.ndarray | None:
    """Reuse the original segmentation/tracking core without changing it."""
    roi_frame, _, _ = tracker.get_roi(frame)
    mask = tracker.segment(roi_frame)
    contour = tracker.largest_contour(mask)
    if contour is None:
        return None
    contour_points = contour[:, 0, :].astype(np.float64, copy=False)
    if tracker.fixed_base is None:
        tracker.initialize_fixed_geometry(contour_points)
    tip = tracker.detect_current_tip(contour_points)
    object_mask = tracker.filled_contour_mask(contour, roi_frame.shape)
    centerline = tracker.extract_centerline(object_mask, tip)
    if centerline is None:
        return None

    # Convert ROI pixels into the fixed undeformed beam coordinates.
    base = tracker.fixed_base
    e1 = tracker.fixed_longitudinal
    e2 = tracker.fixed_transverse
    scale = tracker.calibration.length_per_pixel
    rel = centerline - base
    x = rel @ e1 * scale
    y = rel @ e2 * scale
    order = np.argsort(x)
    xy = np.column_stack((x[order], y[order]))
    _, unique = np.unique(xy[:, 0], return_index=True)
    xy = xy[np.sort(unique)]
    if len(xy) < 8:
        return None
    return xy


def _smooth_curvature_from_centerline(
    xy: np.ndarray,
    cfg: PressureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate small-deflection curvature kappa=d2y/dx2 robustly."""
    x = xy[:, 0]
    y = xy[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 8:
        return np.array([]), np.array([])

    # Restrict to a monotone interior domain. The apex region is excluded from
    # the inverse pressure because b(x) becomes small and Eq. (20) amplifies noise.
    x_min = max(float(np.min(x)), cfg.x_min_fraction * np.max(x))
    x_max = min(float(np.max(x)), cfg.x_max_fraction * np.max(x))
    selector = (x >= x_min) & (x <= x_max)
    x = x[selector]
    y = y[selector]
    if len(x) < 8:
        return np.array([]), np.array([])

    if UnivariateSpline is not None:
        # s has units of y^2; use a noise-scale estimate rather than forcing an
        # arbitrary polynomial degree through every pixel-level fluctuation.
        s = len(x) * cfg.curvature_smoothing_m**2
        spline = UnivariateSpline(x, y, s=s, k=3)
        kappa = spline.derivative(2)(x)
        return x, np.asarray(kappa, dtype=float)

    # Numpy fallback: local quadratic fits.
    kappa = np.full_like(x, np.nan, dtype=float)
    half = max(3, int(round(cfg.local_fit_fraction * len(x))))
    for i in range(len(x)):
        lo = max(0, i-half)
        hi = min(len(x), i+half+1)
        if hi-lo < 5:
            continue
        coeff = np.polyfit(x[lo:hi]-x[i], y[lo:hi], 2)
        kappa[i] = 2.0 * coeff[0]
    return x, kappa


def _beam_width(x: np.ndarray, cfg: PressureConfig) -> np.ndarray:
    """Finite-nose triangular width. Set apex_width_m=0 for the PDF idealization."""
    L = max(float(np.max(x)), 1e-12)
    b = cfg.base_width_m - (
        cfg.base_width_m - cfg.apex_width_m
    ) * x / L
    return np.maximum(b, max(cfg.apex_width_m, 1e-9))


def _uniform_pressure_from_clamp_curvature(
    kappa0: float,
    L: float,
    cfg: PressureConfig,
) -> tuple[float, float]:
    """Return Eq. (15) and its finite-nose correction.

    PDF Eq. (15): dp = E h^3 kappa0 /(2 L^2).
    For b(x)=b0-(b0-ba)x/L, the exact clamp-moment correction is
    dp = E b0 h^3 kappa0 / [2 L^2 (b0+2 ba)].
    """
    dp_pdf = (
        cfg.young_modulus_pa
        * cfg.thickness_m**3
        * kappa0
        / (2.0 * L**2)
    )
    denom = 2.0 * L**2 * (
        cfg.base_width_m + 2.0 * cfg.apex_width_m
    )
    dp_finite = (
        cfg.young_modulus_pa
        * cfg.base_width_m
        * cfg.thickness_m**3
        * kappa0
        / denom
    )
    return float(dp_pdf), float(dp_finite)


def _clamp_curvature_from_centerline(
    xy: np.ndarray,
    cfg: PressureConfig,
) -> float:
    """Estimate kappa(0) from a local quadratic fit in the undeformed x coordinate."""
    x = xy[:, 0]
    y = xy[:, 1]
    L = float(np.max(x))
    nfit = max(6, int(round(cfg.local_fit_fraction * len(x))))
    selector = x <= max(cfg.x_min_fraction * L, x[min(nfit-1, len(x)-1)])
    xx = x[selector]
    yy = y[selector]
    if len(xx) < 5:
        return np.nan
    xx0 = xx - xx[0]
    coeff = np.polyfit(xx0, yy, 2)
    return float(2.0 * coeff[0])


def _nonuniform_pressure_from_curvature(
    x: np.ndarray,
    kappa: np.ndarray,
    cfg: PressureConfig,
) -> np.ndarray:
    """Implement Eq. (20): dp=Eh^3/[12 b] d2[b kappa]/dx2."""
    valid = np.isfinite(x) & np.isfinite(kappa)
    if np.count_nonzero(valid) < 8:
        return np.full_like(x, np.nan, dtype=float)
    xv = x[valid]
    kv = kappa[valid]
    b = _beam_width(xv, cfg)
    Bk = b * kv
    if UnivariateSpline is not None:
        s = len(xv) * (cfg.curvature_smoothing_m / max(xv[-1], 1e-12))**2 * max(np.nanmax(np.abs(Bk))**2, 1e-18)
        spline = UnivariateSpline(xv, Bk, s=s, k=3)
        second = spline.derivative(2)(xv)
    else:
        first = np.gradient(Bk, xv)
        second = np.gradient(first, xv)
    dp = (
        cfg.young_modulus_pa * cfg.thickness_m**3
        / (12.0 * b)
    ) * second
    out = np.full_like(x, np.nan, dtype=float)
    out[valid] = dp
    return out


def _pressure_forces(
    x: np.ndarray,
    dp: np.ndarray,
    cfg: PressureConfig,
) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(dp)
    if np.count_nonzero(valid) < 2:
        return np.nan, np.nan, np.nan
    xv = x[valid]
    dpv = dp[valid]
    b = _beam_width(xv, cfg)
    q = b * dpv
    F = float(np.trapezoid(q, xv))
    M = float(np.trapezoid(q * xv, xv))
    # A simple tip-normalized cantilever trial mode; this is only a generalized
    # force diagnostic, not a replacement for the full structural model.
    L = max(float(np.max(xv)), 1e-12)
    xi = xv / L
    phi = 0.5 * xi**2 * (3.0 - xi)
    Qg = float(np.trapezoid(q * phi, xv))
    return F, M, Qg


def reconstruct_pressure_snapshot(
    tracker: TriangleTracker,
    frame: np.ndarray,
    frame_index: int,
    video_time_s: float,
    total_flow_slpm: float,
    cfg: PressureConfig,
) -> PressureSnapshot | None:
    xy = _extract_centerline_physical(tracker, frame)
    if xy is None:
        return None
    x, kappa = _smooth_curvature_from_centerline(xy, cfg)
    if len(x) < 8:
        return None
    L = float(np.max(xy[:, 0]))
    kappa0 = _clamp_curvature_from_centerline(xy, cfg)
    dp_pdf, dp_finite = _uniform_pressure_from_clamp_curvature(kappa0, L, cfg)
    dp_uniform = np.full_like(x, dp_pdf)
    if cfg.pressure_model == "nonuniform":
        dp_nonuniform = _nonuniform_pressure_from_curvature(x, kappa, cfg)
        if cfg.pressure_scale_clip_pa is not None:
            dp_nonuniform = np.clip(dp_nonuniform, -cfg.pressure_scale_clip_pa, cfg.pressure_scale_clip_pa)
    else:
        dp_nonuniform = np.full_like(x, np.nan)
    b = _beam_width(x, cfg)
    q_uniform = b * dp_uniform
    q_nonuniform = b * dp_nonuniform
    # Uniform-pressure resultants are evaluated analytically over the full beam.
    # For the ideal PDF triangle (ba=0): F=dp*b0*L/2 and M=dp*b0*L^2/6.
    Fu = dp_pdf * cfg.base_width_m * L / 2.0
    Mu = dp_pdf * cfg.base_width_m * L**2 / 6.0
    _, _, Gu = _pressure_forces(x, dp_uniform, cfg)
    if cfg.pressure_model == "nonuniform":
        Fn, Mn, Gn = _pressure_forces(x, dp_nonuniform, cfg)
    else:
        Fn, Mn, Gn = np.nan, np.nan, np.nan
    # The finite-nose uniform solution has the same constant pressure but a
    # different width law. Evaluate its total force/moment on the full beam.
    x_full = np.linspace(0.0, L, max(40, len(x)))
    dp_finite_profile = np.full_like(x_full, dp_finite)
    Ff = dp_finite * L * (cfg.base_width_m + cfg.apex_width_m) / 2.0
    Mf = dp_finite * L**2 * (cfg.base_width_m + 2.0*cfg.apex_width_m) / 6.0
    return PressureSnapshot(
        frame_index=frame_index,
        video_time_s=video_time_s,
        x_m=x,
        curvature_1_m=kappa,
        pressure_uniform_pa=dp_uniform,
        pressure_nonuniform_pa=dp_nonuniform,
        distributed_load_uniform_N_m=q_uniform,
        distributed_load_nonuniform_N_m=q_nonuniform,
        force_uniform_N=Fu,
        force_uniform_finite_tip_N=Ff,
        force_nonuniform_N=Fn,
        moment_uniform_Nm=Mu,
        moment_uniform_finite_tip_Nm=Mf,
        moment_nonuniform_Nm=Mn,
        generalized_force_uniform_N=Gu,
        generalized_force_nonuniform_N=Gn,
        curvature_clamp_1_m=kappa0,
        delta_p_uniform_pdf_pa=dp_pdf,
        delta_p_uniform_finite_tip_pa=dp_finite,
        velocity_m_s=_airspeed_from_flow(total_flow_slpm, cfg),
        dynamic_pressure_pa=_dynamic_pressure_from_flow(total_flow_slpm, cfg),
        total_flow_slpm=total_flow_slpm,
    )


def preanalyze_pressure(
    capture: cv2.VideoCapture,
    tracker: TriangleTracker,
    synchronized: SynchronizedData,
    total_frames: int,
    fps: float,
    cfg: PressureConfig,
    frame_step: int = 1,
) -> PressureAnalysis:
    """Full pressure/force preanalysis. It runs before interactive video playback."""
    frame_step = max(1, int(frame_step))
    print(f"Pressure model: {cfg.pressure_model}")
    if cfg.pressure_model == "uniform":
        print("Uniform pressure inversion (Eq. 15) enabled; Eq. 20 is OFF.")
    else:
        print("Non-uniform pressure inversion (Eq. 20) enabled; this is slower and noise-sensitive.")
    tracker.reset_reference()
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Establish the fixed geometry using the original tracker logic.
    initialize_tracker_reference(capture, tracker, total_frames, fps)

    snapshots: dict[int, PressureSnapshot] = {}
    frame_indices = np.arange(0, total_frames, frame_step, dtype=int)
    last_percent = -1
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        # Flow is synchronized onto the video time using the existing controller arrays.
        t = frame_index / fps
        total_flow = float(np.interp(
            t,
            synchronized.elapsed_time,
            synchronized.flow1 + synchronized.flow2,
            left=synchronized.flow1[0] + synchronized.flow2[0],
            right=synchronized.flow1[-1] + synchronized.flow2[-1],
        ))
        snapshot = reconstruct_pressure_snapshot(
            tracker, frame, int(frame_index), t, total_flow, cfg
        )
        if snapshot is not None:
            snapshots[int(frame_index)] = snapshot
        percent = int(100 * (frame_index + 1) / max(total_frames, 1))
        if percent != last_percent:
            print(f"\rPressure preanalysis: {percent:3d}%", end="", flush=True)
            last_percent = percent
    print()
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not snapshots:
        raise RuntimeError("Pressure preanalysis produced no valid centerline snapshots.")

    frames = np.asarray(sorted(snapshots), dtype=float)
    times = np.asarray([snapshots[int(i)].video_time_s for i in frames])
    def interp(attr: str) -> np.ndarray:
        values = np.asarray([getattr(snapshots[int(i)], attr) for i in frames], dtype=float)
        return np.interp(
            synchronized.elapsed_time,
            times,
            values,
            left=values[0],
            right=values[-1],
        )

    geometry = {
        "length_m": float(np.nanmedian([np.max(s.x_m) for s in snapshots.values()])),
        "base_width_m": cfg.base_width_m,
        "apex_width_m": cfg.apex_width_m,
        "young_modulus_pa": cfg.young_modulus_pa,
        "thickness_m": cfg.thickness_m,
        "inlet_area_m2": cfg.inlet_area_m2,
        "pressure_model": cfg.pressure_model,
    }

    return PressureAnalysis(
        controller_time=synchronized.controller_time,
        total_flow_slpm=synchronized.flow1 + synchronized.flow2,
        velocity_m_s=interp("velocity_m_s"),
        dynamic_pressure_pa=interp("dynamic_pressure_pa"),
        flow_pressure_scale_pa=cfg.pressure_coefficient * interp("dynamic_pressure_pa"),
        delta_p_uniform_pdf_pa=interp("delta_p_uniform_pdf_pa"),
        delta_p_uniform_finite_tip_pa=interp("delta_p_uniform_finite_tip_pa"),
        delta_p_nonuniform_mean_pa=np.interp(
            synchronized.elapsed_time,
            times,
            np.asarray([float(np.nanmean(snapshots[int(i)].pressure_nonuniform_pa)) for i in frames]),
            left=float(np.nanmean(snapshots[int(frames[0])].pressure_nonuniform_pa)),
            right=float(np.nanmean(snapshots[int(frames[-1])].pressure_nonuniform_pa)),
        ),
        force_uniform_pdf_N=interp("force_uniform_N"),
        force_uniform_finite_tip_N=interp("force_uniform_finite_tip_N"),
        force_nonuniform_N=interp("force_nonuniform_N"),
        moment_uniform_pdf_Nm=interp("moment_uniform_Nm"),
        moment_uniform_finite_tip_Nm=interp("moment_uniform_finite_tip_Nm"),
        moment_nonuniform_Nm=interp("moment_nonuniform_Nm"),
        snapshots=snapshots,
        geometry=geometry,
    )


def save_pressure_analysis(
    path: Path,
    analysis: PressureAnalysis,
) -> None:
    df = pd.DataFrame({
        "controller_time": analysis.controller_time,
        "total_flow_slpm": analysis.total_flow_slpm,
        "air_speed_m_s": analysis.velocity_m_s,
        "dynamic_pressure_pa": analysis.dynamic_pressure_pa,
        "flow_pressure_scale_pa": analysis.flow_pressure_scale_pa,
        "delta_p_uniform_pdf_pa": analysis.delta_p_uniform_pdf_pa,
        "delta_p_uniform_finite_tip_pa": analysis.delta_p_uniform_finite_tip_pa,
        "delta_p_nonuniform_mean_pa": analysis.delta_p_nonuniform_mean_pa,
        "force_uniform_pdf_N": analysis.force_uniform_pdf_N,
        "force_uniform_finite_tip_N": analysis.force_uniform_finite_tip_N,
        "force_nonuniform_N": analysis.force_nonuniform_N,
        "moment_uniform_pdf_Nm": analysis.moment_uniform_pdf_Nm,
        "moment_uniform_finite_tip_Nm": analysis.moment_uniform_finite_tip_Nm,
        "moment_nonuniform_Nm": analysis.moment_nonuniform_Nm,
    })
    df.to_csv(path, index=False)

    profile_path = path.with_name(path.stem + "_profiles.csv")
    rows = []
    for frame_index, snap in sorted(analysis.snapshots.items()):
        for j, (x, k, pu, pn) in enumerate(zip(
            snap.x_m, snap.curvature_1_m, snap.pressure_uniform_pa, snap.pressure_nonuniform_pa
        )):
            rows.append({
                "frame_index": frame_index,
                "video_time_s": snap.video_time_s,
                "x_m": x,
                "curvature_1_m": k,
                "pressure_uniform_pa": pu,
                "pressure_nonuniform_pa": pn,
                "load_uniform_N_m": snap.distributed_load_uniform_N_m[j],
                "load_nonuniform_N_m": snap.distributed_load_nonuniform_N_m[j],
            })
    pd.DataFrame(rows).to_csv(profile_path, index=False)


def make_pressure_preanalysis_plots(
    experiment: str,
    analysis: PressureAnalysis,
    output_dir: Path,
) -> None:
    """Create diagnostic comparisons before interactive playback."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    t = analysis.controller_time
    axes[0].plot(t, analysis.total_flow_slpm, label="Qin = QL + QR")
    axes[0].set_ylabel("Flow [SLPM]")
    axes[0].legend()
    axes[1].plot(t, analysis.velocity_m_s, label="inlet air speed")
    axes[1].set_ylabel("U [m/s]")
    axes[1].legend()
    axes[2].plot(t, analysis.dynamic_pressure_pa, label="dynamic pressure 1/2 rho U^2")
    axes[2].plot(t, analysis.flow_pressure_scale_pa, label="Cp * dynamic pressure")
    axes[2].plot(t, analysis.delta_p_uniform_pdf_pa, label="Eq. 15 pressure")
    axes[2].plot(t, analysis.delta_p_uniform_finite_tip_pa, label="finite-nose correction")
    if analysis.geometry.get("pressure_model") == "nonuniform":
        axes[2].plot(t, analysis.delta_p_nonuniform_mean_pa, label="Eq. 20 mean inferred pressure")
    axes[2].set_ylabel("Pressure [Pa]")
    axes[2].set_xlabel("controller time [s]")
    axes[2].legend()
    fig.suptitle(experiment + " — pressure/flow preanalysis")
    fig.tight_layout()
    fig.savefig(output_dir / f"pressure_preanalysis_{experiment}.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Force comparison.
    fig2, ax = plt.subplots(figsize=(9, 6))
    ax.plot(t, analysis.force_uniform_pdf_N, label="uniform pressure force")
    if analysis.geometry.get("pressure_model") == "nonuniform":
        ax.plot(t, analysis.force_nonuniform_N, label="non-uniform pressure force")
    ax.set_xlabel("controller time [s]")
    ax.set_ylabel("Transverse force [N]")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / f"pressure_force_comparison_{experiment}.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Representative pressure-profile comparison: lowest-flow and maximum-flow snapshots.
    frames = sorted(analysis.snapshots)
    chosen = [frames[0], frames[len(frames)//2], frames[-1]] if len(frames) >= 3 else frames
    fig3, axes3 = plt.subplots(len(chosen), 1, figsize=(9, 3.5*len(chosen)), squeeze=False)
    for ax, frame_index in zip(axes3.ravel(), chosen):
        snap = analysis.snapshots[frame_index]
        ax.plot(snap.x_m * 1e3, snap.pressure_uniform_pa, label="uniform Eq. 15")
        if analysis.geometry.get("pressure_model") == "nonuniform":
            ax.plot(snap.x_m * 1e3, snap.pressure_nonuniform_pa, label="non-uniform Eq. 20")
        ax.axhline(snap.dynamic_pressure_pa, ls="--", label="dynamic pressure")
        ax.axhline(snap.dynamic_pressure_pa * cfg.pressure_coefficient, ls=":", label="Cp * dynamic pressure")
        ax.set_ylabel("Delta p [Pa]")
        ax.set_title(f"t={snap.video_time_s:.3f} s, Qin={snap.total_flow_slpm:.2f} SLPM")
        ax.legend()
    axes3.ravel()[-1].set_xlabel("x from clamp [mm]")
    fig3.tight_layout()
    fig3.savefig(output_dir / f"pressure_profiles_comparison_{experiment}.png", dpi=200, bbox_inches="tight")
    plt.show()


class PressureLivePlot(LivePlot):
    """Original live plot plus pressure, force, and current profile panels."""

    def __init__(self, synchronized, unit, pressure_analysis: PressureAnalysis):
        super().__init__(synchronized, unit)
        self.pressure_analysis = pressure_analysis
        self.figure.clf()
        self.axes = self.figure.subplots(4, 1)
        self.tip_line, = self.axes[0].plot([], [], label="Tip transverse")
        self.tip_marker, = self.axes[0].plot([], [], marker="o", linestyle="None")
        self.pressure_line, = self.axes[1].plot([], [], label="Eq. 15 uniform Δp")
        self.pressure_finite_line, = self.axes[1].plot([], [], label="finite-nose Δp")
        self.pressure_nonuniform_line, = self.axes[1].plot([], [], label="Eq. 20 mean Δp")
        self.dynamic_pressure_line, = self.axes[1].plot([], [], label="½ρU²", linestyle="--")
        self.flow_pressure_line, = self.axes[1].plot([], [], label="Cp·½ρU²", linestyle=":")
        self.force_uniform_line, = self.axes[2].plot([], [], label="F uniform")
        self.force_nonuniform_line, = self.axes[2].plot([], [], label="F non-uniform")
        self.force_marker, = self.axes[2].plot([], [], marker="o", linestyle="None")
        self.profile_uniform_line, = self.axes[3].plot([], [], label="uniform")
        self.profile_nonuniform_line, = self.axes[3].plot([], [], label="non-uniform Eq. 20")
        self.profile_dynamic_line, = self.axes[3].plot([], [], label="dynamic-pressure scale", linestyle="--")
        self.use_nonuniform = pressure_analysis.geometry.get("pressure_model") == "nonuniform"
        for line in (self.pressure_nonuniform_line, self.force_nonuniform_line, self.profile_nonuniform_line):
            line.set_visible(self.use_nonuniform)
        self.axes[0].set_ylabel(f"Tip displacement [{unit}]")
        self.axes[1].set_ylabel("Pressure [Pa]")
        self.axes[2].set_ylabel("Force [N]")
        self.axes[3].set_ylabel("Δp [Pa]")
        self.axes[3].set_xlabel("x from clamp [mm]")
        for ax in self.axes[:3]:
            ax.grid(True)
            ax.legend()
            ax.set_xlim(synchronized.controller_time[0], synchronized.controller_time[-1])
        self.axes[3].grid(True)
        self.axes[3].legend()
        self.figure.tight_layout()
        self.figure.show()

    def update(self, controller_index: int, current_frame_index: int | None = None):
        i = int(np.clip(controller_index, 0, len(self.synchronized.controller_time)-1))
        t = self.synchronized.controller_time[:i+1]
        self.tip_line.set_data(t, self.synchronized.tip_transverse[:i+1])
        self.tip_marker.set_data([t[-1]], [self.synchronized.tip_transverse[i]])
        self.pressure_line.set_data(t, self.pressure_analysis.delta_p_uniform_pdf_pa[:i+1])
        self.pressure_finite_line.set_data(t, self.pressure_analysis.delta_p_uniform_finite_tip_pa[:i+1])
        if self.use_nonuniform:
            self.pressure_nonuniform_line.set_data(t, self.pressure_analysis.delta_p_nonuniform_mean_pa[:i+1])
        self.dynamic_pressure_line.set_data(t, self.pressure_analysis.dynamic_pressure_pa[:i+1])
        self.flow_pressure_line.set_data(t, self.pressure_analysis.flow_pressure_scale_pa[:i+1])
        self.force_uniform_line.set_data(t, self.pressure_analysis.force_uniform_pdf_N[:i+1])
        if self.use_nonuniform:
            self.force_nonuniform_line.set_data(t, self.pressure_analysis.force_nonuniform_N[:i+1])
        self.force_marker.set_data([t[-1]], [self.pressure_analysis.force_nonuniform_N[i]])
        if current_frame_index is not None:
            snap = self.pressure_analysis.snapshots.get(int(current_frame_index))
            if snap is not None:
                self.profile_uniform_line.set_data(snap.x_m * 1e3, snap.pressure_uniform_pa)
                if self.use_nonuniform:
                    self.profile_nonuniform_line.set_data(snap.x_m * 1e3, snap.pressure_nonuniform_pa)
                self.profile_dynamic_line.set_data(snap.x_m * 1e3, np.full_like(snap.x_m, snap.dynamic_pressure_pa * self.pressure_analysis.flow_pressure_scale_pa[i] / max(snap.dynamic_pressure_pa, 1e-12)))
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        self.axes[3].relim()
        self.axes[3].autoscale_view(scalex=True, scaley=True)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)


def overlay_pressure_text(
    frame: np.ndarray,
    snapshot: PressureSnapshot | None,
) -> np.ndarray:
    if snapshot is None:
        return frame
    lines = [
        f"U_in = {snapshot.velocity_m_s:.3f} m/s",
        f"q_dyn = {snapshot.dynamic_pressure_pa:.2f} Pa",
        f"Δp Eq15 = {snapshot.delta_p_uniform_pdf_pa:.2f} Pa",
        f"Δp Eq20 mean = {np.nanmean(snapshot.pressure_nonuniform_pa):.2f} Pa",
        f"F Eq15 = {snapshot.force_uniform_N:.5g} N",
        f"F Eq20 = {snapshot.force_nonuniform_N:.5g} N",
    ]
    for j, line in enumerate(lines):
        cv2.putText(
            frame, line, (15, 300 + 25*j),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return frame

# ----------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize triangle video tracking with flow and reconstruct beam pressure. "
            "Uniform-pressure inversion is the default; Eq. 20 is opt-in."
        )
    )

    parser.add_argument(
        "experiment",
        help=(
            "Experiment name X. The script loads files/videos/X.mov, "
            "files/videos/X_calibration.json, and "
            "files/figures/X_processed.csv."
        ),
    )

    parser.add_argument(
        "--preanalysis-step",
        type=int,
        default=1,
        help="Analyze every Nth video frame during preanalysis.",
    )

    parser.add_argument(
        "--force-preanalysis",
        action="store_true",
        help="Ignore existing cached CSV files and rebuild them.",
    )

    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Show the segmentation mask.",
    )

    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Start synchronized playback paused.",
    )

    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback-speed multiplier.",
    )

    parser.add_argument("--pressure-youngs-modulus-mpa", type=float, default=1.0,
                        help="Young's modulus used by inverse beam model [MPa].")
    parser.add_argument("--valve-thickness-mm", type=float, default=1.0,
                        help="Silicone valve thickness [mm].")
    parser.add_argument("--valve-base-width-mm", type=float, default=5.0,
                        help="Triangle base width [mm].")
    parser.add_argument("--valve-apex-width-mm", type=float, default=0.25,
                        help="Finite rounded-nose width used for the corrected triangular geometry [mm].")
    parser.add_argument("--inlet-width-mm", type=float, default=8.0,
                        help="Inlet channel width [mm].")
    parser.add_argument("--inlet-depth-mm", type=float, default=7.5,
                        help="Inlet channel depth [mm].")
    parser.add_argument("--air-density", type=float, default=1.204,
                        help="Air density [kg/m^3].")
    parser.add_argument("--pressure-coefficient", type=float, default=1.0,
                        help="Reference Cp multiplying dynamic pressure; diagnostic only, not a beam-pressure inversion.")
    parser.add_argument("--pressure-model", choices=("uniform", "nonuniform"), default="uniform",
                        help="Pressure inversion model. Default: uniform Eq. 15. Use nonuniform for Eq. 20.")
    parser.add_argument("--pressure-preanalysis-step", type=int, default=1,
                        help="Analyze every Nth video frame for pressure reconstruction.")
    parser.add_argument("--curvature-smoothing-um", type=float, default=20.0,
                        help="Centerline smoothing length scale [um].")
    parser.add_argument("--pressure-clip-pa", type=float, default=None,
                        help="Optional symmetric clip for Eq.20 inferred pressure [Pa].")

    args = parser.parse_args()

    if args.playback_speed <= 0:
        raise ValueError("--playback-speed must be greater than zero.")

    experiment = args.experiment.strip()
    if not experiment:
        raise ValueError("The experiment name cannot be empty.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    video_path = VIDEOS_DIR / f"{experiment}.MOV"
    config_path = VIDEOS_DIR / f"{experiment}.json"
    controller_csv_path = FIGURES_DIR / f"{experiment}_processed.csv"
    synchronized_csv_path = FIGURES_DIR / f"{experiment}_synchronized.csv"
    tip_positions_csv_path = FIGURES_DIR / f"{experiment}_tippos.csv"
    plot_path_qqtip = FIGURES_DIR / f"qq_tip_{experiment}.png"
    plot_path_qtiptime = FIGURES_DIR / f"qtip_time_{experiment}.png"
    plot_path_qin_tiptime = FIGURES_DIR / f"qin_tiptime_{experiment}.png"

    required_paths = {
        "video": video_path,
        "calibration JSON": config_path,
        "processed controller CSV": controller_csv_path,
    }

    missing = [
        f"{label}: {path}"
        for label, path in required_paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing experiment input file(s):\n" + "\n".join(missing)
        )

    calibration, raw_config = load_calibration(
        config_path
    )

    controller_dataframe = load_controller_data(
        controller_csv_path
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(
            "The video does not report a valid FPS."
        )

    if total_frames <= 0:
        capture.release()
        raise RuntimeError(
            "The video does not report a valid frame count."
        )

    video_duration_s = (
        (total_frames - 1) / fps
    )

    controller_start = float(
        controller_dataframe[
            "controller_time"
        ].iloc[0]
    )

    controller_end = float(
        controller_dataframe[
            "controller_time"
        ].iloc[-1]
    )

    controller_duration_s = (
        controller_end - controller_start
    )

    print(
        f"Video: {video_path.name}"
    )
    print(
        f"Video FPS: {fps:.6f}"
    )
    print(
        f"Video duration: {video_duration_s:.6f} s"
    )
    print(
        "Controller interval: "
        f"{controller_start:.6f} to "
        f"{controller_end:.6f} s"
    )
    print(
        "Controller duration: "
        f"{controller_duration_s:.6f} s"
    )

    tracker = TriangleTracker(
        calibration
    )

    synchronized: SynchronizedData | None = None

    use_cache = (
        synchronized_csv_path.exists()
        and not args.force_preanalysis
    )

    if use_cache:
        print(
            "Existing synchronized CSV found."
        )
        print(
            "Attempting to skip preanalysis..."
        )

        try:
            synchronized = load_synchronized_csv(
                path=synchronized_csv_path,
                unit=calibration.length_unit,
                controller_dataframe=controller_dataframe,
                total_frames=total_frames,
            )

            print(
                "Cached synchronized data loaded successfully:"
            )
            print(
                synchronized_csv_path
            )

        except (
            ValueError,
            OSError,
            pd.errors.ParserError,
        ) as error:
            print(
                "Cached synchronized CSV cannot be used:"
            )
            print(error)
            print(
                "A new preanalysis will be performed."
            )

            synchronized = None

    if synchronized is None:
        video_measurements = preanalyze_tip_position(
            capture=capture,
            tracker=tracker,
            total_frames=total_frames,
            fps=fps,
            frame_step=args.preanalysis_step,
        )

        if not video_measurements:
            capture.release()
            raise RuntimeError(
                "No valid triangle measurements were found."
            )

        save_tip_positions_csv(
            path=tip_positions_csv_path,
            measurements=video_measurements,
            unit=calibration.length_unit,
        )

        print("Tip-position CSV saved:")
        print(tip_positions_csv_path)

        synchronized = (
            synchronize_video_and_controller(
                controller_dataframe=(
                    controller_dataframe
                ),
                video_measurements=(
                    video_measurements
                ),
                fps=fps,
                total_frames=total_frames,
            )
        )

        save_synchronized_csv(
            path=synchronized_csv_path,
            synchronized=synchronized,
            unit=calibration.length_unit,
        )

        print(
            "Synchronized CSV saved:"
        )
        print(
            synchronized_csv_path
        )

    # Ensure the fixed reference geometry is available even when cache
    # loading skipped the video preanalysis.
    initialize_tracker_reference(
        capture=capture,
        tracker=tracker,
        total_frames=total_frames,
        fps=fps,
    )

    ## Some plots!
    df_synch = pd.read_csv(synchronized_csv_path)

    fig = plt.figure(figsize=(8, 7))
    ax0 = fig.add_subplot(111)

    ax0.set_title(
        synchronized_csv_path.name,
        fontsize=ticks_size,
    )

    axes = [ax0]
    format_axes(axes)

    ax0.set_ylabel(
        "$Q_{\\mathrm{left}}$ [SLPM]",
        fontsize=ticks_size,
    )

    ax0.set_xlabel(
        "$Q_{\\mathrm{right}}$ [SLPM]",
        fontsize=ticks_size,
    )

    cmap = cmc.hawaii

    print(df_synch.columns)

    # Change this name if your synchronized file uses another unit,
    # for example tip_transverse_px.
    values = df_synch["tip_transverse_mm"]

    scatter = ax0.scatter(
        df_synch["flow1"],
        df_synch["flow2"],
        c=values,
        cmap=cmap,
        norm=colors.Normalize(
            values.min(),
            values.max(),
        ),
        alpha=0.5,
        marker=".",
    )

    cbar = fig.colorbar(
        scatter,
        ax=ax0,
    )

    cbar.set_label(
        "Tip transverse displacement [mm]"
    )

    ax0.set_xlim(0, 100)
    ax0.set_ylim(0, 100)

    minflow = 3

    for qtot in [30, 50, 70]:
        qtotx = np.linspace(
            minflow,
            qtot,
        )

        qtoty = np.linspace(
            qtot,
            minflow,
        )

        ax0.plot(
            qtotx,
            qtoty,
            color="silver",
            ls="--",
            linewidth=0.5,
        )

        ax0.text(
            np.mean(qtotx) - 15,
            np.mean(qtoty) + 10,
            "%d SLPM" % qtot,
            color="k",
            rotation=-45,
        )

    ax0.plot(
        np.linspace(minflow, 100),
        np.linspace(minflow, 100),
        color="k",
        ls="--",
    )

    plt.tight_layout()
    plt.savefig(
        plot_path_qqtip,
        dpi=200,
        bbox_inches="tight",
    )

    print("Scatter plot saved:")
    print(plot_path_qqtip)

    plt.show()



    fig = plt.figure(figsize=(8, 7))
    ax0 = fig.add_subplot(111)
    ax0.set_title(experiment,fontsize=ticks_size)

    axes = [ax0]
    format_axes(axes)

    ax0.set_ylabel(
        "Tip transverse displacement [mm]",
        fontsize=ticks_size,
    )

    ax0.set_xlabel(
        "$\\frac{Q_{\\mathrm{left}}-Q_{\\mathrm{right}}}{Q_{\\mathrm{left}}+Q_{\\mathrm{right}}}$",
        fontsize=ticks_size,
    )
    cmap = cmc.managua #managua roma berlin_r

    cs = []

    #cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(df_synch["flow1"].min(), df_synch["flow1"].max())

    for i in range(len(df_synch["flow1"])):
        cs.append(cmap(i/len(df_synch["flow1"])))   
    values = df_synch["controller_time"]

    scatter = ax0.scatter(
    (df_synch["flow1"]-df_synch["flow2"])/(df_synch["flow1"]+df_synch["flow2"]),
    df_synch["tip_transverse_mm"],
    c=values,
    cmap=plt.get_cmap("plasma"),#cmc.managua,
    norm=colors.Normalize(values.min(), values.max()),
    alpha=0.7,
    marker="."
    )

    cbar = fig.colorbar(scatter, ax=ax0)
    cbar.set_label("Time [s]")
    #ax0.set_xlim(0,100)
    #ax0.set_ylim(0,100)
    #minflow = 3#min(df["flow_1"].min(),df["flow_2"].min())
    #maxflow = max(df_synch["flow1"].max(),df_synch["flow2"].max())
    #for qtot in [30,50,70]:
    #    qtotx = np.linspace(minflow, qtot)
    #    qtoty = np.linspace(qtot,minflow)
    #    ax0.plot(qtotx,qtoty, color='silver', ls='--', linewidth=.5)
    #    ax0.text(np.mean(qtotx)-15,np.mean(qtoty)+10,'%d SLPM'%(qtot), color='k',rotation=-45)
    #ax0.plot(np.linspace(minflow,100),np.linspace(minflow,100),color='k', ls='--')
    ax0.set_title(experiment,fontsize=ticks_size)



    plt.tight_layout()

    plotname = f"qtip_time{experiment}.png"

    plt.savefig(
        plot_path_qtiptime,
        dpi=200,
        bbox_inches="tight",
    )
    print("Scatter plot saved:")
    print(plot_path_qtiptime)
    plt.show()




    fig = plt.figure(figsize=(8, 7))
    ax0 = fig.add_subplot(111)
    ax0.set_title(experiment,fontsize=ticks_size)

    axes = [ax0]
    format_axes(axes)

    ax0.set_ylabel(
        "Tip transverse displacement [mm]",
        fontsize=ticks_size,
    )

    ax0.set_xlabel(
        "$Q_{\\mathrm{left}}+Q_{\\mathrm{right}}$",
        fontsize=ticks_size,
    )
    cmap = cmc.managua #managua roma berlin_r

    cs = []

    #cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(df_synch["flow1"].min(), df_synch["flow1"].max())

    for i in range(len(df_synch["flow1"])):
        cs.append(cmap(i/len(df_synch["flow1"])))   
    values = df_synch["controller_time"]

    scatter = ax0.scatter(
    (df_synch["flow1"]+df_synch["flow2"]),
    df_synch["tip_transverse_mm"],
    c=values,
    cmap=plt.get_cmap("plasma"),#cmc.managua,
    norm=colors.Normalize(values.min(), values.max()),
    alpha=0.7,
    marker="."
    )

    cbar = fig.colorbar(scatter, ax=ax0)
    cbar.set_label("Time [s]")
    #ax0.set_xlim(0,100)
    #ax0.set_ylim(0,100)
    #minflow = 3#min(df["flow_1"].min(),df["flow_2"].min())
    #maxflow = max(df_synch["flow1"].max(),df_synch["flow2"].max())
    #for qtot in [30,50,70]:
    #    qtotx = np.linspace(minflow, qtot)
    #    qtoty = np.linspace(qtot,minflow)
    #    ax0.plot(qtotx,qtoty, color='silver', ls='--', linewidth=.5)
    #    ax0.text(np.mean(qtotx)-15,np.mean(qtoty)+10,'%d SLPM'%(qtot), color='k',rotation=-45)
    #ax0.plot(np.linspace(minflow,100),np.linspace(minflow,100),color='k', ls='--')
    ax0.set_title(experiment,fontsize=ticks_size)

    plt.tight_layout()

    plotname = f"qin_tip_time{experiment}.png"

    plt.savefig(
        plot_path_qin_tiptime,
        dpi=200,
        bbox_inches="tight",
    )
    print("Scatter plot saved:")
    print(plot_path_qin_tiptime)
    plt.show()


    # ------------------------------------------------------------------
    # Pressure/force preanalysis. This deliberately happens before the
    # interactive video window is shown, as requested.
    # ------------------------------------------------------------------
    pressure_cfg = PressureConfig(
        young_modulus_pa=args.pressure_youngs_modulus_mpa * 1.0e6,
        thickness_m=args.valve_thickness_mm * 1.0e-3,
        base_width_m=args.valve_base_width_mm * 1.0e-3,
        apex_width_m=args.valve_apex_width_mm * 1.0e-3,
        air_density_kg_m3=args.air_density,
        pressure_coefficient=args.pressure_coefficient,
        inlet_area_m2=(args.inlet_width_mm * 1.0e-3) * (args.inlet_depth_mm * 1.0e-3),
        curvature_smoothing_m=args.curvature_smoothing_um * 1.0e-6,
        pressure_scale_clip_pa=args.pressure_clip_pa,
        pressure_model=args.pressure_model,
    )

    pressure_analysis = preanalyze_pressure(
        capture=capture,
        tracker=tracker,
        synchronized=synchronized,
        total_frames=total_frames,
        fps=fps,
        cfg=pressure_cfg,
        frame_step=args.pressure_preanalysis_step,
    )

    pressure_csv_path = FIGURES_DIR / f"{experiment}_pressure_analysis.csv"
    save_pressure_analysis(pressure_csv_path, pressure_analysis)
    print("Pressure analysis CSV saved:")
    print(pressure_csv_path)
    make_pressure_preanalysis_plots(
        experiment,
        pressure_analysis,
        FIGURES_DIR,
    )

    # Graphics are created only after synchronized and pressure data are available.
    live_plot = PressureLivePlot(
        synchronized=synchronized,
        unit=calibration.length_unit,
        pressure_analysis=pressure_analysis,
    )

    playback = PlaybackController(
        number_of_samples=len(
            synchronized.controller_time
        )
    )

    if args.show_mask:
        cv2.namedWindow(
            MASK_WINDOW,
            cv2.WINDOW_NORMAL,
        )

    paused = bool(
        args.start_paused
    )

    controller_index = 0

    previous_video_frame_index: int | None = None

    current_overlay: np.ndarray | None = None
    current_mask: np.ndarray | None = None

    running = True
    force_frame_refresh = True

    try:
        while running:
            requested_index = (
                playback.consume_seek()
            )

            if requested_index is not None:
                controller_index = requested_index
                paused = True
                force_frame_refresh = True

            video_frame_index = int(
                synchronized.video_frame_index[
                    controller_index
                ]
            )

            if (
                force_frame_refresh
                or previous_video_frame_index
                != video_frame_index
            ):
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    video_frame_index,
                )

                ok, frame = capture.read()

                if not ok or frame is None:
                    raise RuntimeError(
                        "Could not read video frame "
                        f"{video_frame_index}."
                    )

                video_time_s = (
                    video_frame_index / fps
                )

                (
                    current_overlay,
                    current_mask,
                    frame_measurement,
                ) = tracker.process_full(
                    frame=frame,
                    frame_index=video_frame_index,
                    video_time_s=video_time_s,
                )

                if frame_measurement is None:
                    cv2.putText(
                        current_overlay,
                        "Triangle not detected",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                previous_video_frame_index = (
                    video_frame_index
                )

            if current_overlay is None:
                raise RuntimeError(
                    "No video overlay is available."
                )

            display_frame = current_overlay.copy()

            current_pressure_snapshot = pressure_analysis.snapshots.get(video_frame_index)
            display_frame = overlay_pressure_text(display_frame, current_pressure_snapshot)

            controller_time = float(
                synchronized.controller_time[
                    controller_index
                ]
            )

            elapsed_time_s = float(
                synchronized.elapsed_time[
                    controller_index
                ]
            )

            tip_position = float(
                synchronized.tip_transverse[
                    controller_index
                ]
            )

            flow1 = float(
                synchronized.flow1[
                    controller_index
                ]
            )

            flow2 = float(
                synchronized.flow2[
                    controller_index
                ]
            )

            status = (
                "PAUSED"
                if paused
                else "PLAYING"
            )

            status_color = (
                (0, 220, 255)
                if paused
                else (0, 255, 0)
            )

            text_lines = [
                status,
                (
                    "controller_time = "
                    f"{controller_time:.6f} s"
                ),
                (
                    "Elapsed time = "
                    f"{elapsed_time_s:.3f} s"
                ),
                (
                    "Video time = "
                    f"{video_frame_index / fps:.3f} s"
                ),
                (
                    "Video frame = "
                    f"{video_frame_index} / "
                    f"{total_frames - 1}"
                ),
                (
                    "Tip transverse = "
                    f"{tip_position:+.5f} "
                    f"{calibration.length_unit}"
                ),
                f"flow1 = {flow1:.5f}",
                f"flow2 = {flow2:.5f}",
                (
                    "Space: play/pause | "
                    "A/D: previous/next sample | "
                    "Q: quit"
                ),
            ]

            for line_index, text in enumerate(
                text_lines
            ):
                color = (
                    status_color
                    if line_index == 0
                    else (255, 255, 255)
                )

                cv2.putText(
                    display_frame,
                    text,
                    (
                        15,
                        28 + 27 * line_index,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.61,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(
                ANALYSIS_WINDOW,
                display_frame,
            )

            if (
                args.show_mask
                and current_mask is not None
            ):
                cv2.imshow(
                    MASK_WINDOW,
                    current_mask,
                )

            live_plot.update(
                controller_index,
                current_frame_index=video_frame_index,
            )

            playback.update_position(
                controller_index
            )

            force_frame_refresh = False

            if (
                not paused
                and controller_index
                < len(
                    synchronized.controller_time
                ) - 1
            ):
                controller_dt = float(
                    synchronized.controller_time[
                        controller_index + 1
                    ]
                    - synchronized.controller_time[
                        controller_index
                    ]
                )

                delay_ms = max(
                    1,
                    int(
                        round(
                            1000.0
                            * controller_dt
                            / args.playback_speed
                        )
                    ),
                )

                # Prevent OpenCV from becoming unresponsive for very
                # large gaps in controller_time.
                delay_ms = min(
                    delay_ms,
                    1000,
                )
            else:
                delay_ms = 20

            raw_key = cv2.waitKey(
                delay_ms
            )

            if raw_key < 0:
                key = -1
            else:
                key = raw_key & 0xFF

                if ord("A") <= key <= ord("Z"):
                    key = ord(
                        chr(key).lower()
                    )

            if key in (
                27,
                ord("q"),
            ):
                running = False
                continue

            if key == ord(" "):
                paused = not paused
                continue

            if key == ord("a"):
                paused = True

                controller_index = max(
                    0,
                    controller_index - 1,
                )

                force_frame_refresh = True
                continue

            if key == ord("d"):
                paused = True

                controller_index = min(
                    len(
                        synchronized.controller_time
                    ) - 1,
                    controller_index + 1,
                )

                force_frame_refresh = True
                continue

            if not paused:
                if controller_index < (
                    len(
                        synchronized.controller_time
                    ) - 1
                ):
                    controller_index += 1
                    force_frame_refresh = True
                else:
                    paused = True

    finally:
        capture.release()
        cv2.destroyAllWindows()

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()