from __future__ import annotations

from pathlib import Path
import argparse
import csv
import re
from typing import Optional

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


CONTROLLER_COLUMNS = [
    "time_s",
    "elapsed_us",
    "pres_in",
    "flow_in",
    "flow_left",
    "flow_right",
    "pres_left",
    "pres_right",
]


# ----------------------------------------------------------------------
# Inlet Reynolds-number configuration
# ----------------------------------------------------------------------
#
# Geometry inspected from Hosebot_Body_v2.8 STL:
#   - circular inlet/connector bore: approximately 3.9 mm diameter
#   - circular inlet throat/orifice: 3.0 mm diameter
#
# Reynolds number is evaluated at the 3.0 mm throat by default.
#
# Because flow_in is reported in SLPM, the calculation is naturally written
# from the standard-condition mass flow:
#
#       Re = 4 * rho_std * Q_std / (pi * mu * D)
#
# where Q_std is converted from SLPM to m^3/s.
#
# This is equivalent to rho * V * D / mu.  Using mass flow is convenient
# because rho*Q is conserved between the standard-flow reference and the
# pressurized inlet (neglecting the small pressure dependence of viscosity).
#
DEFAULT_INLET_DIAMETER_MM = 3.0
STL_INLET_PORT_DIAMETER_MM = 3.9

# Air properties near 20 degC / 1 atm.  If your flow meter defines "standard"
# at another reference condition, override the density from the command line.
DEFAULT_AIR_DENSITY_STD_KG_M3 = 1.2041
DEFAULT_AIR_DYNAMIC_VISCOSITY_PA_S = 1.81e-5

SLPM_TO_M3_S = 1.0e-3 / 60.0


class ExperimentData:
    """Load, plot, and analyze one controller CSV.

    By default every valid row in the CSV is used. ``n_repetitions`` is optional
    and only adds repetition labels; it does not remove data.

    Transition detection is based on sharp finite-time changes in
    ``Q_left - Q_right``.

    Important distinction:
    - ``flat_run_threshold`` decides only whether an entire experiment is flat.
    - transition validation uses separate pre/post state limits and a minimum
      state change, so a real transition may start or finish above 5 SLPM.
    - before/after jump windows are separated from the event by a guard window,
      preventing the transition itself from contaminating the state medians.
    """

    def __init__(
        self,
        expfilename: str,
        data_dir: str | Path = "experiment_data",
        figure_dir: str | Path = "figures",
        n_repetitions: Optional[int] = None,
        repetition: Optional[int] = None,
        timecutoff: float = 0.0,
        inlet_diameter_mm: float = DEFAULT_INLET_DIAMETER_MM,
        air_density_std_kg_m3: float = DEFAULT_AIR_DENSITY_STD_KG_M3,
        air_dynamic_viscosity_pa_s: float = DEFAULT_AIR_DYNAMIC_VISCOSITY_PA_S,
    ) -> None:
        self.expfilename = expfilename
        self.data_dir = Path(data_dir)
        self.figure_dir = Path(figure_dir)
        self.n_repetitions = n_repetitions
        self.repetition = repetition
        self.timecutoff = float(timecutoff)

        self.inlet_diameter_mm = float(inlet_diameter_mm)
        self.air_density_std_kg_m3 = float(air_density_std_kg_m3)
        self.air_dynamic_viscosity_pa_s = float(air_dynamic_viscosity_pa_s)

        if self.inlet_diameter_mm <= 0:
            raise ValueError("inlet_diameter_mm must be greater than zero.")
        if self.air_density_std_kg_m3 <= 0:
            raise ValueError("air_density_std_kg_m3 must be greater than zero.")
        if self.air_dynamic_viscosity_pa_s <= 0:
            raise ValueError("air_dynamic_viscosity_pa_s must be greater than zero.")

        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.color_left = "red"
        self.color_right = "royalblue"
        self.ticks_size = 14

        self.csv_path = self.data_dir / f"{self.expfilename}.csv"
        self.df = self._read_controller_csv()

        if self.repetition is None:
            self.data = self._get_all_data()
        else:
            self.data = self._get_repetition_data(self.repetition)

        self.last_transition_results: Optional[dict] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _read_controller_csv(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        recovered_rows: list[list[float]] = []
        skipped_rows = 0

        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)

            for line_number, row in enumerate(reader, start=1):
                if not row:
                    continue

                if row[0].strip().lower() in {"time_s", "time"}:
                    continue

                if len(row) != len(CONTROLLER_COLUMNS):
                    skipped_rows += 1
                    continue

                try:
                    values = [float(value) for value in row]
                except ValueError:
                    print(f"Skipping invalid row at line {line_number}: {row}")
                    skipped_rows += 1
                    continue

                recovered_rows.append(values)

        if not recovered_rows:
            raise RuntimeError(
                f"No valid controller rows were found in {self.csv_path}."
            )

        # Keep every valid datapoint, including equal-valued consecutive rows.
        df = pd.DataFrame(recovered_rows, columns=CONTROLLER_COLUMNS)
        df = df.reset_index(drop=True)

        df["experiment_time"] = df["time_s"] - df["time_s"].iloc[0]

        n_rows = len(df)

        if self.n_repetitions is None:
            df["repetition"] = 0
            df["controller_time"] = df["experiment_time"]
            print(f"Read {len(df)} controller rows.")
            print(f"Ignored {skipped_rows} malformed rows.")
            print("No repetition count supplied; using the entire dataset.")
            return df

        if not isinstance(self.n_repetitions, int):
            raise TypeError("n_repetitions must be an integer or None.")
        if self.n_repetitions <= 0:
            raise ValueError("n_repetitions must be greater than zero.")
        if self.n_repetitions > n_rows:
            raise RuntimeError(
                f"Number of valid rows ({n_rows}) is too small for "
                f"{self.n_repetitions} repetitions."
            )

        # np.array_split preserves every row.
        repetition_labels = np.empty(n_rows, dtype=int)
        split_indices = np.array_split(np.arange(n_rows), self.n_repetitions)
        repetition_sizes = []

        for rep, indices in enumerate(split_indices):
            repetition_labels[indices] = rep
            repetition_sizes.append(len(indices))

        df["repetition"] = repetition_labels
        df["controller_time"] = (
            df.groupby("repetition")["time_s"]
            .transform(lambda values: values - values.iloc[0])
        )

        print(f"Read {len(df)} controller rows.")
        print(f"Ignored {skipped_rows} malformed rows.")
        print(
            f"Split data into {self.n_repetitions} repetitions "
            f"with row counts {repetition_sizes}."
        )
        return df

    def _get_all_data(self) -> pd.DataFrame:
        df = self.df[self.df["experiment_time"] >= self.timecutoff].copy()
        if df.empty:
            raise RuntimeError(
                "No controller data remains after "
                f"applying timecutoff={self.timecutoff}."
            )
        return df

    def _get_repetition_data(self, repetition: int) -> pd.DataFrame:
        if self.n_repetitions is None:
            raise ValueError(
                "Cannot select a repetition because n_repetitions was not supplied."
            )
        if repetition < 0:
            raise ValueError("Repetition must be non-negative.")
        if repetition >= self.n_repetitions:
            raise ValueError(
                f"Repetition {repetition} does not exist. Valid repetitions are "
                f"0 through {self.n_repetitions - 1}."
            )

        df = self.df[self.df["repetition"] == repetition].copy()
        df = df[df["controller_time"] >= self.timecutoff].copy()

        if df.empty:
            raise RuntimeError(
                "No controller data remains after "
                f"applying repetition={repetition} and timecutoff={self.timecutoff}."
            )
        return df

    def set_repetition(self, repetition: Optional[int]) -> None:
        self.repetition = repetition
        if repetition is None:
            self.data = self._get_all_data()
        else:
            self.data = self._get_repetition_data(repetition)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_axes(self, axes) -> None:
        for ax in axes:
            ax.xaxis.set_tick_params(labelsize=self.ticks_size)
            ax.yaxis.set_tick_params(labelsize=self.ticks_size)
            ax.grid(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)

    def _select_plot_data(self, full_experiment: bool):
        if full_experiment:
            return self._get_all_data(), "experiment_time"

        if self.repetition is None:
            raise ValueError(
                "To plot one repetition, supply n_repetitions and select "
                "repetition=<index> or call set_repetition(index)."
            )
        return self.data, "controller_time"

    # ------------------------------------------------------------------
    # Inlet Reynolds number
    # ------------------------------------------------------------------

    @property
    def inlet_diameter_m(self) -> float:
        """Circular inlet/orifice diameter in metres."""
        return self.inlet_diameter_mm * 1.0e-3

    def inlet_reynolds_from_slpm(self, flow_in_slpm) -> np.ndarray:
        """Return inlet Reynolds number from standard volumetric flow.

        ``flow_in_slpm`` is the measured standard flow in SLPM.

        For a circular passage,

            Re = rho * V * D / mu
               = 4 * m_dot / (pi * mu * D)
               = 4 * rho_std * Q_std / (pi * mu * D)

        where Q_std is the standard volumetric flow.  The absolute value is
        used because Reynolds number is a magnitude.
        """
        flow = np.asarray(flow_in_slpm, dtype=float)
        q_std_m3_s = np.abs(flow) * SLPM_TO_M3_S

        reynolds = (
            4.0
            * self.air_density_std_kg_m3
            * q_std_m3_s
            / (
                np.pi
                * self.air_dynamic_viscosity_pa_s
                * self.inlet_diameter_m
            )
        )

        return reynolds

    # ------------------------------------------------------------------
    # Transition-analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seconds_to_samples(
        time_values,
        seconds: float,
        minimum: int = 1,
        odd: bool = False,
    ) -> int:
        values = np.asarray(time_values, dtype=float)
        dt = np.diff(values)
        dt = dt[np.isfinite(dt) & (dt > 0)]

        if dt.size == 0 or seconds <= 0:
            samples = minimum
        else:
            median_dt = float(np.median(dt))
            samples = max(minimum, int(round(float(seconds) / median_dt)))

        if odd and samples % 2 == 0:
            samples += 1
        return samples

    @staticmethod
    def _rolling_median(values, window: int) -> np.ndarray:
        return (
            pd.Series(np.asarray(values, dtype=float))
            .rolling(window, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )

    @staticmethod
    def _rolling_mean(values, window: int) -> np.ndarray:
        return (
            pd.Series(np.asarray(values, dtype=float))
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )

    @staticmethod
    def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        """Return inclusive-exclusive intervals for True runs."""
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0:
            return []

        padded = np.concatenate(([False], mask, [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        return [(int(start), int(end)) for start, end in zip(starts, ends)]

    @staticmethod
    def _merge_intervals(
        intervals: list[tuple[int, int]],
        max_gap_samples: int,
    ) -> list[tuple[int, int]]:
        if not intervals:
            return []

        merged = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap_samples:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _window_median(values: np.ndarray, start: int, end: int) -> float:
        start = max(0, int(start))
        end = min(len(values), int(end))
        if end <= start:
            return float("nan")
        return float(np.median(values[start:end]))

    @staticmethod
    def _zero_transition_summary() -> dict:
        return {
            "pin_up_avg": 0.0,
            "Qin_up_avg": 0.0,
            "pin_up_std": 0.0,
            "Qin_up_std": 0.0,
            "pin_down_avg": 0.0,
            "Qin_down_avg": 0.0,
            "pin_down_std": 0.0,
            "Qin_down_std": 0.0,
        }

    @staticmethod
    def _summarize_exact_events(
        events: list[dict],
        expected_transitions: int,
        prefix: str,
    ) -> dict:
        """Summarize only when one valid event exists for every expected ramp."""
        if len(events) != expected_transitions:
            return {
                f"pin_{prefix}_avg": 0.0,
                f"Qin_{prefix}_avg": 0.0,
                f"pin_{prefix}_std": 0.0,
                f"Qin_{prefix}_std": 0.0,
            }

        pin = np.asarray([event["pin"] for event in events], dtype=float)
        qin = np.asarray([event["Qin"] for event in events], dtype=float)
        ddof = 1 if expected_transitions > 1 else 0

        return {
            f"pin_{prefix}_avg": float(np.mean(pin)),
            f"Qin_{prefix}_avg": float(np.mean(qin)),
            f"pin_{prefix}_std": float(np.std(pin, ddof=ddof)),
            f"Qin_{prefix}_std": float(np.std(qin, ddof=ddof)),
        }

    def _pressure_plateaus(
        self,
        time_values: np.ndarray,
        pin_smooth: np.ndarray,
        expected_cycles: int,
        plateau_fraction: float = 0.90,
        merge_gap_s: float = 1.0,
        min_plateau_s: float = 0.5,
    ) -> list[tuple[int, int]]:
        """Find one high-pressure plateau for each pressure cycle.

        Primary method: contiguous regions near the global high pressure.
        Fallback: split the experiment into equal time chunks and find the local
        high-pressure region in each chunk. The fallback is only for cycle
        segmentation; transition detection itself is not based on equal chunks.
        """
        n = len(pin_smooth)
        if n < expected_cycles * 3:
            return []

        p_low = float(np.nanpercentile(pin_smooth, 5.0))
        p_high = float(np.nanpercentile(pin_smooth, 95.0))
        span = p_high - p_low
        if not np.isfinite(span) or span <= 1e-9:
            return []

        threshold = p_low + plateau_fraction * span
        intervals = self._true_runs(pin_smooth >= threshold)

        merge_gap = self._seconds_to_samples(
            time_values, merge_gap_s, minimum=1
        )
        min_plateau = self._seconds_to_samples(
            time_values, min_plateau_s, minimum=1
        )
        intervals = self._merge_intervals(intervals, merge_gap)
        intervals = [
            (start, end)
            for start, end in intervals
            if end - start >= min_plateau
        ]

        if len(intervals) >= expected_cycles:
            if len(intervals) > expected_cycles:
                # Prefer long high-pressure regions, then restore chronological order.
                scored = sorted(
                    intervals,
                    key=lambda interval: (
                        interval[1] - interval[0],
                        float(np.mean(pin_smooth[interval[0]:interval[1]])),
                    ),
                    reverse=True,
                )[:expected_cycles]
                intervals = sorted(scored, key=lambda interval: interval[0])
            return intervals

        # Fallback: one local top region in each approximately equal cycle.
        fallback: list[tuple[int, int]] = []
        chunks = np.array_split(np.arange(n), expected_cycles)

        for chunk in chunks:
            if chunk.size < 3:
                return []

            start = int(chunk[0])
            end = int(chunk[-1]) + 1
            local = pin_smooth[start:end]
            local_low = float(np.nanpercentile(local, 5.0))
            local_high = float(np.nanpercentile(local, 95.0))
            local_span = local_high - local_low

            if not np.isfinite(local_span) or local_span <= 1e-9:
                return []

            local_threshold = local_low + plateau_fraction * local_span
            local_mask = local >= local_threshold
            local_runs = self._true_runs(local_mask)

            if local_runs:
                # Choose the run that contains the local maximum, otherwise the longest.
                local_peak = int(np.argmax(local))
                containing = [
                    interval
                    for interval in local_runs
                    if interval[0] <= local_peak < interval[1]
                ]
                if containing:
                    local_start, local_end = containing[0]
                else:
                    local_start, local_end = max(
                        local_runs, key=lambda interval: interval[1] - interval[0]
                    )
            else:
                local_peak = int(np.argmax(local))
                local_start = max(0, local_peak - 1)
                local_end = min(len(local), local_peak + 2)

            fallback.append((start + local_start, start + local_end))

        return fallback

    def _build_pressure_cycles(
        self,
        time_values: np.ndarray,
        pin_smooth: np.ndarray,
        expected_cycles: int,
    ) -> list[dict]:
        plateaus = self._pressure_plateaus(
            time_values,
            pin_smooth,
            expected_cycles,
        )
        if len(plateaus) != expected_cycles:
            return []

        centers = [int((start + end - 1) // 2) for start, end in plateaus]
        boundaries = [0]
        for left, right in zip(centers[:-1], centers[1:]):
            boundaries.append(int((left + right) // 2))
        boundaries.append(len(pin_smooth))

        cycles = []
        for ramp, ((plateau_start, plateau_end), peak, start, end) in enumerate(
            zip(plateaus, centers, boundaries[:-1], boundaries[1:]),
            start=1,
        ):
            if end - start < 5:
                continue
            cycles.append(
                {
                    "ramp": ramp,
                    "start": int(start),
                    "end": int(end),
                    "peak": int(peak),
                    "plateau_start": int(plateau_start),
                    "plateau_end": int(plateau_end),
                }
            )
        return cycles

    @staticmethod
    def _rolling_median_extreme(
        abs_signal: np.ndarray,
        start: int,
        end: int,
        state_samples: int,
        mode: str,
    ) -> float:
        start = max(0, int(start))
        end = min(len(abs_signal), int(end))
        if end <= start:
            return float("nan")

        segment = abs_signal[start:end]
        if len(segment) < state_samples:
            return float(np.median(segment))

        medians = (
            pd.Series(segment)
            .rolling(state_samples, min_periods=state_samples)
            .median()
            .dropna()
            .to_numpy(dtype=float)
        )
        if medians.size == 0:
            return float(np.median(segment))
        if mode == "min":
            return float(np.min(medians))
        if mode == "max":
            return float(np.max(medians))
        raise ValueError("mode must be 'min' or 'max'.")

    def _find_transition_in_phase(
        self,
        *,
        ramp: int,
        direction: str,
        phase_start: int,
        phase_end: int,
        time_values: np.ndarray,
        qdiff_smooth: np.ndarray,
        pin_smooth: np.ndarray,
        qin_smooth: np.ndarray,
        up_pre_max_slpm: float,
        broken_threshold_slpm: float,
        down_post_max_slpm: float,
        min_state_change_slpm: float,
        jump_samples: int,
        guard_samples: int,
        state_samples: int,
        state_horizon_samples: int,
        slope_samples: int,
        critical_samples: int,
        min_jump_slpm: float,
        min_pressure_slope_bar_s: float,
    ) -> Optional[dict]:
        """Return the strongest validated transition in one pressure-ramp phase.

        The transition time is identified from a finite-time change in Qdiff.
        The 5-SLPM whole-run flat criterion is intentionally *not* used here.

        A small guard region around each candidate keeps the transition itself
        out of the before/after jump windows.

        UP validation:
            pre state <= up_pre_max_slpm
            post state >= broken_threshold_slpm
            post - pre >= min_state_change_slpm

        DOWN validation:
            pre state >= broken_threshold_slpm
            post state <= down_post_max_slpm
            pre - post >= min_state_change_slpm
        """
        n = len(time_values)
        phase_start = max(0, int(phase_start))
        phase_end = min(n, int(phase_end))

        jump_margin = jump_samples + guard_samples
        margin = max(jump_margin, slope_samples, state_samples + guard_samples)
        search_start = max(phase_start + margin, margin)
        search_end = min(phase_end - margin, n - margin)

        if search_end <= search_start:
            return None

        abs_qdiff = np.abs(qdiff_smooth)
        candidates = []

        for index in range(search_start, search_end):
            before_start = index - guard_samples - jump_samples
            before_end = index - guard_samples
            after_start = index + guard_samples
            after_end = index + guard_samples + jump_samples

            q_before = self._window_median(
                qdiff_smooth,
                before_start,
                before_end,
            )
            q_after = self._window_median(
                qdiff_smooth,
                after_start,
                after_end,
            )
            if not np.isfinite(q_before) or not np.isfinite(q_after):
                continue

            signed_jump = q_after - q_before
            jump_slpm = abs(signed_jump)
            if jump_slpm < min_jump_slpm:
                continue

            # Pressure direction is estimated over a wider window than Qdiff.
            p_before = self._window_median(
                pin_smooth,
                index - slope_samples,
                index,
            )
            p_after = self._window_median(
                pin_smooth,
                index,
                index + slope_samples,
            )
            if not np.isfinite(p_before) or not np.isfinite(p_after):
                continue

            t_before = self._window_median(
                time_values,
                index - slope_samples,
                index,
            )
            t_after = self._window_median(
                time_values,
                index,
                index + slope_samples,
            )
            dt = t_after - t_before
            if not np.isfinite(dt) or dt <= 0:
                continue

            pressure_slope = (p_after - p_before) / dt

            if direction == "up":
                if pressure_slope < min_pressure_slope_bar_s:
                    continue

                # Immediate robust state before the event, excluding the guard.
                pre_level = self._window_median(
                    abs_qdiff,
                    index - guard_samples - state_samples,
                    index - guard_samples,
                )

                # The broken state may establish after the sharp jump, so search
                # forward for the strongest persistent state.
                post_level = self._rolling_median_extreme(
                    abs_qdiff,
                    index + guard_samples,
                    min(phase_end, index + state_horizon_samples),
                    state_samples,
                    mode="max",
                )

                if not np.isfinite(pre_level) or not np.isfinite(post_level):
                    continue
                if pre_level > up_pre_max_slpm:
                    continue
                if post_level < broken_threshold_slpm:
                    continue

                state_change = post_level - pre_level
                if state_change < min_state_change_slpm:
                    continue

                branch_reference = q_after

            elif direction == "down":
                if pressure_slope > -min_pressure_slope_bar_s:
                    continue

                # The broken state can be present for several seconds before the
                # sharp return event. Find a robust high state in the preceding
                # horizon.
                pre_level = self._rolling_median_extreme(
                    abs_qdiff,
                    max(phase_start, index - state_horizon_samples),
                    index - guard_samples,
                    state_samples,
                    mode="max",
                )

                # The signal does not need to cross 5 SLPM at the event. It only
                # needs to settle to the configured low-state limit afterward.
                post_level = self._rolling_median_extreme(
                    abs_qdiff,
                    index + guard_samples,
                    min(phase_end, index + state_horizon_samples),
                    state_samples,
                    mode="min",
                )

                if not np.isfinite(pre_level) or not np.isfinite(post_level):
                    continue
                if pre_level < broken_threshold_slpm:
                    continue
                if post_level > down_post_max_slpm:
                    continue

                state_change = pre_level - post_level
                if state_change < min_state_change_slpm:
                    continue

                branch_reference = q_before

            else:
                raise ValueError("direction must be 'up' or 'down'.")

            # Primary score: finite-time jump. State contrast helps separate
            # a real branch transition from smaller oscillations/noise.
            score = (
                jump_slpm
                * (1.0 + min(abs(pressure_slope), 1.0))
                * (1.0 + min(state_change / max(min_state_change_slpm, 1e-9), 2.0))
            )

            lo = max(0, index - critical_samples)
            hi = min(n, index + critical_samples + 1)

            candidates.append(
                {
                    "ramp": int(ramp),
                    "direction": direction,
                    "index": int(index),
                    "time_s": float(time_values[index]),
                    "pin": float(np.median(pin_smooth[lo:hi])),
                    "Qin": float(np.median(qin_smooth[lo:hi])),
                    "qdiff_before": float(q_before),
                    "qdiff_after": float(q_after),
                    "signed_jump_slpm": float(signed_jump),
                    "jump_slpm": float(jump_slpm),
                    "pressure_slope_bar_s": float(pressure_slope),
                    "pre_state_abs_slpm": float(pre_level),
                    "post_state_abs_slpm": float(post_level),
                    "state_change_slpm": float(state_change),
                    "branch_sign": int(np.sign(branch_reference)),
                    "score": float(score),
                }
            )

        if not candidates:
            return None

        # Exactly one transition per ramp and direction.
        return max(candidates, key=lambda event: event["score"])

    def _finite_window_jump_trace(
        self,
        qdiff_smooth: np.ndarray,
        jump_samples: int,
        guard_samples: int,
    ) -> np.ndarray:
        """Signed guarded finite-window jump used in the diagnostic subplot."""
        n = len(qdiff_smooth)
        trace = np.full(n, np.nan, dtype=float)
        margin = jump_samples + guard_samples

        for index in range(margin, n - margin):
            q_before = np.median(
                qdiff_smooth[
                    index - guard_samples - jump_samples:
                    index - guard_samples
                ]
            )
            q_after = np.median(
                qdiff_smooth[
                    index + guard_samples:
                    index + guard_samples + jump_samples
                ]
            )
            trace[index] = q_after - q_before

        return trace

    def detect_flowdiff_transitions(

        self,
        df: Optional[pd.DataFrame] = None,
        time_col: str = "experiment_time",
        expected_transitions: int = 3,
        flat_run_threshold: float = 5.0,
        up_pre_max_slpm: float = 15.0,
        broken_threshold_slpm: float = 15.0,
        down_post_max_slpm: float = 10.0,
        min_state_change_slpm: float = 15.0,
        smooth_window_s: float = 0.15,
        jump_window_s: float = 0.60,
        guard_window_s: float = 0.10,
        state_window_s: float = 0.50,
        state_horizon_s: float = 8.0,
        pressure_slope_window_s: float = 0.75,
        min_pressure_slope_bar_s: float = 0.005,
        critical_window_s: float = 0.10,
        min_jump_slpm: float = 4.0,
    ) -> dict:
        """Detect one forward and one backward transition in each pressure ramp.

        Fundamental design:
        -------------------
        1. Smooth Q_left-Q_right and p_in.
        2. If the *entire* smoothed run stays below ``flat_run_threshold``,
           classify the experiment as flat and return zeros.
        3. Locate the three pressure cycles and search rising/falling phases
           independently.
        4. Detect the event from a guarded finite-time change in Qdiff.
        5. Use p_in slope only to classify up versus down.
        6. Validate transitions with separate state criteria:
              UP:   pre <= up_pre_max, post >= broken_threshold,
                    post-pre >= min_state_change.
              DOWN: pre >= broken_threshold, post <= down_post_max,
                    pre-post >= min_state_change.
        7. The event is never forced to coincide with a +/-5 SLPM crossing.
        """
        if df is None:
            df = self._get_all_data()

        if expected_transitions <= 0:
            raise ValueError("expected_transitions must be greater than zero.")
        if flat_run_threshold <= 0:
            raise ValueError("flat_run_threshold must be greater than zero.")
        if up_pre_max_slpm <= 0:
            raise ValueError("up_pre_max_slpm must be greater than zero.")
        if broken_threshold_slpm <= 0:
            raise ValueError("broken_threshold_slpm must be greater than zero.")
        if down_post_max_slpm <= 0:
            raise ValueError("down_post_max_slpm must be greater than zero.")
        if min_state_change_slpm <= 0:
            raise ValueError("min_state_change_slpm must be greater than zero.")
        if jump_window_s <= 0:
            raise ValueError("jump_window_s must be greater than zero.")
        if guard_window_s < 0:
            raise ValueError("guard_window_s cannot be negative.")
        if state_horizon_s <= 0:
            raise ValueError("state_horizon_s must be greater than zero.")
        if min_jump_slpm <= 0:
            raise ValueError("min_jump_slpm must be greater than zero.")

        needed = [
            time_col,
            "flow_left",
            "flow_right",
            "flow_in",
            "pres_in",
        ]
        work = df[needed].copy()
        for column in needed:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work = (
            work.replace([np.inf, -np.inf], np.nan)
            .dropna()
            .reset_index(drop=True)
        )

        result = {
            "flat": True,
            "expected_transitions": int(expected_transitions),
            "flat_run_threshold_slpm": float(flat_run_threshold),
            "up_pre_max_slpm": float(up_pre_max_slpm),
            "broken_threshold_slpm": float(broken_threshold_slpm),
            "down_post_max_slpm": float(down_post_max_slpm),
            "min_state_change_slpm": float(min_state_change_slpm),
            "up_events": [],
            "down_events": [],
            "cycles": [],
            "diagnostics": {},
        }
        result.update(self._zero_transition_summary())

        if len(work) < 20:
            self.last_transition_results = result
            return result

        time_values = work[time_col].to_numpy(dtype=float)
        qdiff_raw = (
            work["flow_left"].to_numpy(dtype=float)
            - work["flow_right"].to_numpy(dtype=float)
        )

        smooth_samples = self._seconds_to_samples(
            time_values, smooth_window_s, minimum=3, odd=True
        )
        # A second short mean pass suppresses sample-scale sensor noise without
        # moving the large transition appreciably.
        mean_samples = self._seconds_to_samples(
            time_values, max(smooth_window_s / 2.0, 0.02), minimum=1, odd=True
        )
        jump_samples = self._seconds_to_samples(
            time_values, jump_window_s, minimum=2
        )
        guard_samples = self._seconds_to_samples(
            time_values, guard_window_s, minimum=0
        )
        state_samples = self._seconds_to_samples(
            time_values, state_window_s, minimum=2
        )
        state_horizon_samples = self._seconds_to_samples(
            time_values, state_horizon_s, minimum=state_samples + 1
        )
        slope_samples = self._seconds_to_samples(
            time_values, pressure_slope_window_s, minimum=2
        )
        critical_samples = self._seconds_to_samples(
            time_values, critical_window_s, minimum=1
        )

        qdiff_smooth = self._rolling_median(qdiff_raw, smooth_samples)
        qdiff_smooth = self._rolling_mean(qdiff_smooth, mean_samples)

        pin_raw = work["pres_in"].to_numpy(dtype=float)
        qin_raw = work["flow_in"].to_numpy(dtype=float)
        pin_smooth = self._rolling_median(pin_raw, smooth_samples)
        pin_smooth = self._rolling_mean(pin_smooth, mean_samples)
        qin_smooth = self._rolling_median(qin_raw, smooth_samples)

        abs_qdiff = np.abs(qdiff_smooth)
        jump_trace = self._finite_window_jump_trace(qdiff_smooth, jump_samples, guard_samples)

        result["diagnostics"] = {
            "time_values": time_values,
            "qdiff_smooth": qdiff_smooth,
            "pin_smooth": pin_smooth,
            "qin_smooth": qin_smooth,
            "jump_trace": jump_trace,
        }

        # Global flat-run test only. This is the sole use of the 5-SLPM-style
        # threshold; transition events have separate state limits.
        if float(np.nanmax(abs_qdiff)) < flat_run_threshold:
            self.last_transition_results = result
            return result

        result["flat"] = False

        cycles = self._build_pressure_cycles(
            time_values,
            pin_smooth,
            expected_transitions,
        )
        result["cycles"] = cycles

        if len(cycles) != expected_transitions:
            print(
                f"[Transition detector] Could identify {len(cycles)} pressure "
                f"cycles; expected {expected_transitions}. Saving zero summaries."
            )
            self.last_transition_results = result
            return result

        up_events = []
        down_events = []

        for cycle in cycles:
            ramp = cycle["ramp"]
            cycle_start = cycle["start"]
            cycle_end = cycle["end"]
            peak = cycle["peak"]

            # Search each ramp independently. Using the peak as the dividing point
            # leaves the full rising/falling portions available to the detector;
            # we do not cut the search at an arbitrary pressure threshold.
            up_event = self._find_transition_in_phase(
                ramp=ramp,
                direction="up",
                phase_start=cycle_start,
                phase_end=peak + 1,
                time_values=time_values,
                qdiff_smooth=qdiff_smooth,
                pin_smooth=pin_smooth,
                qin_smooth=qin_smooth,
                up_pre_max_slpm=up_pre_max_slpm,
                broken_threshold_slpm=broken_threshold_slpm,
                down_post_max_slpm=down_post_max_slpm,
                min_state_change_slpm=min_state_change_slpm,
                jump_samples=jump_samples,
                guard_samples=guard_samples,
                state_samples=state_samples,
                state_horizon_samples=state_horizon_samples,
                slope_samples=slope_samples,
                critical_samples=critical_samples,
                min_jump_slpm=min_jump_slpm,
                min_pressure_slope_bar_s=min_pressure_slope_bar_s,
            )

            down_event = self._find_transition_in_phase(
                ramp=ramp,
                direction="down",
                phase_start=peak,
                phase_end=cycle_end,
                time_values=time_values,
                qdiff_smooth=qdiff_smooth,
                pin_smooth=pin_smooth,
                qin_smooth=qin_smooth,
                up_pre_max_slpm=up_pre_max_slpm,
                broken_threshold_slpm=broken_threshold_slpm,
                down_post_max_slpm=down_post_max_slpm,
                min_state_change_slpm=min_state_change_slpm,
                jump_samples=jump_samples,
                guard_samples=guard_samples,
                state_samples=state_samples,
                state_horizon_samples=state_horizon_samples,
                slope_samples=slope_samples,
                critical_samples=critical_samples,
                min_jump_slpm=min_jump_slpm,
                min_pressure_slope_bar_s=min_pressure_slope_bar_s,
            )

            if up_event is not None:
                up_events.append(up_event)
            if down_event is not None:
                down_events.append(down_event)

        result["up_events"] = sorted(up_events, key=lambda event: event["ramp"])
        result["down_events"] = sorted(down_events, key=lambda event: event["ramp"])

        result.update(
            self._summarize_exact_events(
                result["up_events"], expected_transitions, "up"
            )
        )
        result.update(
            self._summarize_exact_events(
                result["down_events"], expected_transitions, "down"
            )
        )

        self.last_transition_results = result
        return result

    def _print_transition_results(self, results: dict) -> None:
        print("\nTransition analysis")
        print("-------------------")

        if results["flat"]:
            print(
                f"|Q_left-Q_right| stayed below "
                f"{results['flat_run_threshold_slpm']:.3g} SLPM after smoothing."
            )
            print("Saving zero transition statistics.")
            return

        for direction in ("up", "down"):
            events = results[f"{direction}_events"]
            print(
                f"{direction.upper()}: {len(events)}/"
                f"{results['expected_transitions']} ramps detected"
            )

            by_ramp = {event["ramp"]: event for event in events}
            for ramp in range(1, results["expected_transitions"] + 1):
                event = by_ramp.get(ramp)
                if event is None:
                    print(f"  ramp {ramp}: NOT FOUND")
                    continue
                print(
                    f"  ramp {ramp}: t={event['time_s']:.3f} s, "
                    f"p_in={event['pin']:.6g} bar, "
                    f"Q_in={event['Qin']:.6g} SLPM, "
                    f"jump={event['signed_jump_slpm']:+.3f} SLPM, "
                    f"state_change={event['state_change_slpm']:.3f} SLPM, "
                    f"dp/dt={event['pressure_slope_bar_s']:+.4f} bar/s"
                )

            if len(events) != results["expected_transitions"]:
                print(
                    f"  Missing at least one {direction} transition; "
                    "summary for this direction is zero."
                )

        print(
            "  up avg:   "
            f"(p_in={results['pin_up_avg']:.6g}, "
            f"Q_in={results['Qin_up_avg']:.6g})"
        )
        print(
            "  up std:   "
            f"(p_in={results['pin_up_std']:.6g}, "
            f"Q_in={results['Qin_up_std']:.6g})"
        )
        print(
            "  down avg: "
            f"(p_in={results['pin_down_avg']:.6g}, "
            f"Q_in={results['Qin_down_avg']:.6g})"
        )
        print(
            "  down std: "
            f"(p_in={results['pin_down_std']:.6g}, "
            f"Q_in={results['Qin_down_std']:.6g})"
        )

    def save_transition_events(self, results: dict) -> Path:
        """Save accepted individual transition events in the run's figures folder."""
        path = self.figure_dir / f"transition_events_{self.expfilename}.csv"
        rows = []

        for direction in ("up", "down"):
            by_ramp = {event["ramp"]: event for event in results[f"{direction}_events"]}
            for ramp in range(1, results["expected_transitions"] + 1):
                event = by_ramp.get(ramp)
                if event is None:
                    rows.append(
                        {
                            "ramp": ramp,
                            "direction": direction,
                            "detected": 0,
                            "time_s": 0.0,
                            "pin_bar": 0.0,
                            "Qin_slpm": 0.0,
                            "qdiff_before_slpm": 0.0,
                            "qdiff_after_slpm": 0.0,
                            "signed_jump_slpm": 0.0,
                            "pressure_slope_bar_s": 0.0,
                            "pre_state_abs_slpm": 0.0,
                            "post_state_abs_slpm": 0.0,
                            "state_change_slpm": 0.0,
                            "branch_sign": 0,
                        }
                    )
                    continue

                rows.append(
                    {
                        "ramp": ramp,
                        "direction": direction,
                        "detected": 1,
                        "time_s": event["time_s"],
                        "pin_bar": event["pin"],
                        "Qin_slpm": event["Qin"],
                        "qdiff_before_slpm": event["qdiff_before"],
                        "qdiff_after_slpm": event["qdiff_after"],
                        "signed_jump_slpm": event["signed_jump_slpm"],
                        "pressure_slope_bar_s": event["pressure_slope_bar_s"],
                        "pre_state_abs_slpm": event["pre_state_abs_slpm"],
                        "post_state_abs_slpm": event["post_state_abs_slpm"],
                        "state_change_slpm": event["state_change_slpm"],
                        "branch_sign": event["branch_sign"],
                    }
                )

        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Transition events saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Time-history plot
    # ------------------------------------------------------------------

    def plot_time_history(
        self,
        save: bool = True,
        dpi: int = 200,
        full_experiment: bool = True,
    ):
        """Plot flow, pressure, and inlet Reynolds number versus time."""
        df, time_col = self._select_plot_data(full_experiment)

        reynolds_in = self.inlet_reynolds_from_slpm(df["flow_in"])

        fig = plt.figure(figsize=(7, 9))
        ax0 = fig.add_subplot(311)
        ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)
        self.format_axes([ax0, ax1, ax2])

        for ax in (ax0, ax1, ax2):
            ax.set_xlabel("Time [s]", fontsize=self.ticks_size)

        # --------------------------------------------------------------
        # Flow
        # --------------------------------------------------------------
        ax0.plot(
            df[time_col],
            df["flow_left"],
            color=self.color_left,
            label=r"$Q_{\mathrm{left}}$",
        )
        ax0.plot(
            df[time_col],
            df["flow_right"],
            color=self.color_right,
            label=r"$Q_{\mathrm{right}}$",
        )
        ax0.plot(
            df[time_col],
            df["flow_left"] + df["flow_right"],
            color="silver",
            label=r"$Q_{\mathrm{left}}+Q_{\mathrm{right}}$",
        )
        ax0.plot(
            df[time_col],
            df["flow_in"],
            color="k",
            label=r"$Q_{\mathrm{in}}$",
        )
        ax0.set_ylabel("$Q$ [SLPM]", fontsize=self.ticks_size)
        ax0.legend(fontsize=self.ticks_size)

        # --------------------------------------------------------------
        # Pressure
        # --------------------------------------------------------------
        ax1.plot(
            df[time_col],
            df["pres_in"],
            color="k",
            label=r"$p_{\mathrm{in}}$",
        )
        ax1.plot(
            df[time_col],
            df["pres_left"],
            color=self.color_left,
            label=r"$p_{\mathrm{left}}$",
        )
        ax1.plot(
            df[time_col],
            df["pres_right"],
            color=self.color_right,
            label=r"$p_{\mathrm{right}}$",
        )
        ax1.set_ylabel("$p$ [bar]", fontsize=self.ticks_size)
        ax1.legend(fontsize=self.ticks_size)

        # --------------------------------------------------------------
        # Inlet Reynolds number
        # --------------------------------------------------------------
        ax2.plot(
            df[time_col],
            reynolds_in,
            color="k",
            linewidth=1.2,
            label=rf"$Re_{{\mathrm{{in}}}}$ ($D={self.inlet_diameter_mm:g}$ mm)",
        )
        ax2.set_ylabel(r"$Re_{\mathrm{in}}$ [-]", fontsize=self.ticks_size)
        ax2.legend(fontsize=self.ticks_size)

        # Repetition boundaries, if repetition labels are being used.
        if (
            full_experiment
            and self.n_repetitions is not None
            and self.n_repetitions > 1
        ):
            rep_starts = df.groupby("repetition")[time_col].min().iloc[1:]
            for ax in (ax0, ax1, ax2):
                for rep_start in rep_starts:
                    ax.axvline(
                        rep_start,
                        color="silver",
                        linestyle="--",
                        linewidth=0.8,
                    )

        fig.tight_layout()

        if save:
            suffix = "all" if full_experiment else f"rep{self.repetition}"
            path = self.figure_dir / (
                f"expdata_time_{self.expfilename}_{suffix}.png"
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure: {path}")

        return fig

    # ------------------------------------------------------------------
    # Reynolds number versus inlet flow
    # ------------------------------------------------------------------

    def plot_reynolds_flowin(
        self,
        title: str = "",
        save: bool = True,
        dpi: int = 200,
        full_experiment: bool = True,
    ):
        """Plot inlet Reynolds number as a function of measured inlet flow."""
        df, _ = self._select_plot_data(full_experiment)

        qin = df["flow_in"].to_numpy(dtype=float)
        reynolds_in = self.inlet_reynolds_from_slpm(qin)

        valid = np.isfinite(qin) & np.isfinite(reynolds_in)
        qin = qin[valid]
        reynolds_in = reynolds_in[valid]

        # Since Re is a deterministic function of Qin for fixed D/rho/mu,
        # sort by |Qin| so the saved curve is clean rather than repeatedly
        # tracing the same line in acquisition-time order.
        order = np.argsort(np.abs(qin))
        qin_plot = np.abs(qin[order])
        reynolds_plot = reynolds_in[order]

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=self.ticks_size)
        self.format_axes([ax])

        ax.plot(
            qin_plot,
            reynolds_plot,
            color="k",
            linewidth=1.5,
        )

        ax.set_xlabel(
            r"$|Q_{\mathrm{in}}|$ [SLPM]",
            fontsize=self.ticks_size,
        )
        ax.set_ylabel(
            r"$Re_{\mathrm{in}}$ [-]",
            fontsize=self.ticks_size,
        )

        # Put the physical assumptions directly on the figure.
        annotation = (
            rf"$D={self.inlet_diameter_mm:g}$ mm"
            "\n"
            rf"$\rho_{{std}}={self.air_density_std_kg_m3:.4g}$ kg m$^{{-3}}$"
            "\n"
            rf"$\mu={self.air_dynamic_viscosity_pa_s:.3g}$ Pa s"
        )
        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(10, self.ticks_size - 2),
        )

        fig.tight_layout()

        if save:
            suffix = "all" if full_experiment else f"rep{self.repetition}"
            path = self.figure_dir / (
                f"reynolds_qin_{self.expfilename}_{suffix}.png"
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure: {path}")

        return fig

    # ------------------------------------------------------------------
    # Flow-difference / transition plot
    # ------------------------------------------------------------------

    def plot_time_flowdiff(
        self,
        save: bool = True,
        dpi: int = 200,
        full_experiment: bool = True,
        expected_transitions: int = 3,
        flat_run_threshold: float = 5.0,
        up_pre_max_slpm: float = 15.0,
        broken_threshold_slpm: float = 15.0,
        down_post_max_slpm: float = 10.0,
        min_state_change_slpm: float = 15.0,
        smooth_window_s: float = 0.15,
        jump_window_s: float = 0.60,
        guard_window_s: float = 0.10,
        state_window_s: float = 0.50,
        state_horizon_s: float = 8.0,
        pressure_slope_window_s: float = 0.75,
        min_pressure_slope_bar_s: float = 0.005,
        critical_window_s: float = 0.10,
        min_jump_slpm: float = 4.0,
    ):
        df, time_col = self._select_plot_data(full_experiment)

        results = self.detect_flowdiff_transitions(
            df=df,
            time_col=time_col,
            expected_transitions=expected_transitions,
            flat_run_threshold=flat_run_threshold,
            up_pre_max_slpm=up_pre_max_slpm,
            broken_threshold_slpm=broken_threshold_slpm,
            down_post_max_slpm=down_post_max_slpm,
            min_state_change_slpm=min_state_change_slpm,
            smooth_window_s=smooth_window_s,
            jump_window_s=jump_window_s,
            guard_window_s=guard_window_s,
            state_window_s=state_window_s,
            state_horizon_s=state_horizon_s,
            pressure_slope_window_s=pressure_slope_window_s,
            min_pressure_slope_bar_s=min_pressure_slope_bar_s,
            critical_window_s=critical_window_s,
            min_jump_slpm=min_jump_slpm,
        )
        self._print_transition_results(results)

        diagnostics = results["diagnostics"]
        time_values = diagnostics.get("time_values", df[time_col].to_numpy(dtype=float))
        qdiff_smooth = diagnostics.get(
            "qdiff_smooth",
            (df["flow_left"] - df["flow_right"]).to_numpy(dtype=float),
        )
        jump_trace = diagnostics.get(
            "jump_trace",
            np.full(len(df), np.nan, dtype=float),
        )

        fig = plt.figure(figsize=(7, 9))
        ax0 = fig.add_subplot(311)
        ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)
        self.format_axes([ax0, ax1, ax2])

        for ax in (ax0, ax1, ax2):
            ax.set_xlabel("Time [s]", fontsize=self.ticks_size)

        qdiff_raw = df["flow_left"].to_numpy(dtype=float) - df["flow_right"].to_numpy(dtype=float)
        qsum = df["flow_left"].to_numpy(dtype=float) + df["flow_right"].to_numpy(dtype=float)

        ax0.plot(
            df[time_col],
            qdiff_raw,
            color="k",
            linewidth=1.0,
            alpha=0.35,
        )
        ax0.plot(
            time_values,
            qdiff_smooth,
            color="k",
            linewidth=1.5,
            label=r"$Q_{\mathrm{left}}-Q_{\mathrm{right}}$",
        )
        ax0.plot(
            df[time_col],
            qsum,
            color="silver",
            linewidth=1.5,
            label=r"$Q_{\mathrm{left}}+Q_{\mathrm{right}}$",
        )
        ax0.axhline(flat_run_threshold, color="0.65", linestyle=":", linewidth=0.9)
        ax0.axhline(-flat_run_threshold, color="0.65", linestyle=":", linewidth=0.9)
        ax0.set_ylabel("$Q$ [SLPM]", fontsize=self.ticks_size)
        ax0.legend(fontsize=self.ticks_size)

        ax1.plot(
            df[time_col], df["pres_in"],
            color="k", label=r"$p_{\mathrm{in}}$"
        )
        ax1.plot(
            df[time_col], df["pres_left"],
            color=self.color_left, label=r"$p_{\mathrm{left}}$"
        )
        ax1.plot(
            df[time_col], df["pres_right"],
            color=self.color_right, label=r"$p_{\mathrm{right}}$"
        )
        ax1.set_ylabel("$p$ [bar]", fontsize=self.ticks_size)
        ax1.legend(fontsize=self.ticks_size)

        # Diagnostic finite-time change. This is deliberately not np.diff();
        # sample-to-sample differencing is dominated by acquisition noise.
        ax2.plot(time_values, jump_trace, color="k", linewidth=1.0)
        ax2.axhline(min_jump_slpm, color="0.65", linestyle=":", linewidth=0.9)
        ax2.axhline(-min_jump_slpm, color="0.65", linestyle=":", linewidth=0.9)
        ax2.set_ylabel(r"$\Delta_{\tau}(Q_L-Q_R)$ [SLPM]", fontsize=self.ticks_size)

        # Lightly show pressure-cycle centers to make missed-cycle problems visible.
        for cycle in results.get("cycles", []):
            peak_time = float(time_values[cycle["peak"]])
            for ax in (ax0, ax1, ax2):
                ax.axvline(peak_time, color="0.85", linestyle="--", linewidth=0.7)

        for event in results["up_events"]:
            for ax in (ax0, ax1, ax2):
                ax.axvline(event["time_s"], color="green", linestyle="--", linewidth=1.2)
            ax0.text(
                event["time_s"],
                qdiff_smooth[event["index"]],
                f" U{event['ramp']}",
                fontsize=9,
                va="bottom",
            )

        for event in results["down_events"]:
            for ax in (ax0, ax1, ax2):
                ax.axvline(event["time_s"], color="darkorange", linestyle="--", linewidth=1.2)
            ax0.text(
                event["time_s"],
                qdiff_smooth[event["index"]],
                f" D{event['ramp']}",
                fontsize=9,
                va="bottom",
            )

        if full_experiment and self.n_repetitions is not None and self.n_repetitions > 1:
            rep_starts = df.groupby("repetition")[time_col].min().iloc[1:]
            for ax in (ax0, ax1, ax2):
                for rep_start in rep_starts:
                    ax.axvline(rep_start, color="silver", linestyle="--", linewidth=0.6)

        fig.tight_layout()

        if save:
            suffix = "all" if full_experiment else f"rep{self.repetition}"
            path = self.figure_dir / (
                f"expdata_timeflowdiff_{self.expfilename}_{suffix}.png"
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure: {path}")
            self.save_transition_events(results)

        return fig

    # ------------------------------------------------------------------
    # Flow versus flow
    # ------------------------------------------------------------------

    def plot_flow_flow(
        self,
        title: str = "",
        save: bool = True,
        dpi: int = 200,
        full_experiment: bool = True,
    ):
        df, time_col = self._select_plot_data(full_experiment)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=self.ticks_size)
        self.format_axes([ax])

        ax.set_ylabel(r"$Q_{\mathrm{left}}$ [SLPM]", fontsize=self.ticks_size)
        ax.set_xlabel(r"$Q_{\mathrm{right}}$ [SLPM]", fontsize=self.ticks_size)

        values = df[time_col]
        scatter = ax.scatter(
            df["flow_left"],
            df["flow_right"],
            c=values,
            cmap=plt.get_cmap("plasma"),
            norm=colors.Normalize(values.min(), values.max()),
            alpha=0.7,
            marker=".",
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Time [s]", fontsize=self.ticks_size)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        minflow = 3
        for qtot in [30, 50, 70]:
            qtotx = np.linspace(minflow, qtot)
            qtoty = np.linspace(qtot, minflow)
            ax.plot(qtotx, qtoty, color="silver", linestyle="--", linewidth=0.5)
            ax.text(
                np.mean(qtotx) - 15,
                np.mean(qtoty) + 10,
                f"{qtot:d} SLPM",
                color="k",
                rotation=-45,
            )

        ax.plot(
            np.linspace(minflow, 100),
            np.linspace(minflow, 100),
            color="k",
            linestyle="--",
        )

        fig.tight_layout()

        if save:
            suffix = "all" if full_experiment else f"rep{self.repetition}"
            path = self.figure_dir / f"qq_{self.expfilename}_{suffix}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure: {path}")

        return fig

    # ------------------------------------------------------------------
    # Flow versus flow
    # ------------------------------------------------------------------

    def plot_pin_flowin(
        self,
        title: str = "",
        save: bool = True,
        dpi: int = 200,
        full_experiment: bool = True,
    ):
        df, time_col = self._select_plot_data(full_experiment)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=self.ticks_size)
        self.format_axes([ax])

        ax.set_ylabel(r"$p_{\mathrm{in}}$ [bar]", fontsize=self.ticks_size)
        ax.set_xlabel(r"$Q_{\mathrm{in}}$ [SLPM]", fontsize=self.ticks_size)

        values = df[time_col]
        scatter = ax.scatter(
            df["flow_in"],
            df["pres_in"],
            c=values,
            cmap=plt.get_cmap("plasma"),
            norm=colors.Normalize(values.min(), values.max()),
            alpha=0.7,
            marker=".",
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Time [s]", fontsize=self.ticks_size)

        #ax.set_xlim(0, 100)
        #ax.set_ylim(0, 100)

        
        

        fig.tight_layout()

        if save:
            suffix = "all" if full_experiment else f"rep{self.repetition}"
            path = self.figure_dir / f"pin_flowin_{self.expfilename}_{suffix}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure: {path}")

        return fig

    # ------------------------------------------------------------------
    # Flow versus pressure
    # ------------------------------------------------------------------

    def plot_flow_pressure(
        self,
        save: bool = True,
        dpi: int = 200,
        full_experiment: bool = True,
    ):
        df, _ = self._select_plot_data(full_experiment)

        fig = plt.figure(figsize=(7, 7))
        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)
        self.format_axes([ax0, ax1])

        for ax in (ax0, ax1):
            ax.set_ylabel("$p$ [bar]", fontsize=self.ticks_size)
            ax.set_xlabel("$Q$ [SLPM]", fontsize=self.ticks_size)

        ax0.plot(df["flow_left"], df["pres_left"], color=self.color_left)
        ax1.plot(df["flow_right"], df["pres_right"], color=self.color_right)

        fig.tight_layout()

        if save:
            suffix = "all" if full_experiment else f"rep{self.repetition}"
            path = self.figure_dir / f"qp_{self.expfilename}_{suffix}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure: {path}")

        return fig

    # ------------------------------------------------------------------
    # All plots
    # ------------------------------------------------------------------

    def plot_all(
        self,
        title: str = "",
        save: bool = True,
        show: bool = True,
        dpi: int = 200,
        full_experiment: bool = True,
        expected_transitions: int = 3,
        flat_run_threshold: float = 5.0,
        up_pre_max_slpm: float = 15.0,
        broken_threshold_slpm: float = 15.0,
        down_post_max_slpm: float = 10.0,
        min_state_change_slpm: float = 15.0,
        smooth_window_s: float = 0.15,
        jump_window_s: float = 0.60,
        guard_window_s: float = 0.10,
        state_window_s: float = 0.50,
        state_horizon_s: float = 8.0,
        pressure_slope_window_s: float = 0.75,
        min_pressure_slope_bar_s: float = 0.005,
        critical_window_s: float = 0.10,
        min_jump_slpm: float = 4.0,
    ):
        figures = []

        figures.append(
            self.plot_time_history(
                save=save,
                dpi=dpi,
                full_experiment=full_experiment,
            )
        )
        figures.append(
            self.plot_reynolds_flowin(
                title=title,
                save=save,
                dpi=dpi,
                full_experiment=full_experiment,
            )
        )
        figures.append(
            self.plot_flow_flow(
                title=title,
                save=save,
                dpi=dpi,
                full_experiment=full_experiment,
            )
        )
        figures.append(
            self.plot_pin_flowin (
                title=title,
                save=save,
                dpi=dpi,
                full_experiment=full_experiment,
            )
        )
        figures.append(
            self.plot_flow_pressure(
                save=save,
                dpi=dpi,
                full_experiment=full_experiment,
            )
        )
        figures.append(
            self.plot_time_flowdiff(
                save=save,
                dpi=dpi,
                full_experiment=full_experiment,
                expected_transitions=expected_transitions,
                flat_run_threshold=flat_run_threshold,
                up_pre_max_slpm=up_pre_max_slpm,
                broken_threshold_slpm=broken_threshold_slpm,
                down_post_max_slpm=down_post_max_slpm,
                min_state_change_slpm=min_state_change_slpm,
                smooth_window_s=smooth_window_s,
                jump_window_s=jump_window_s,
                guard_window_s=guard_window_s,
                state_window_s=state_window_s,
                state_horizon_s=state_horizon_s,
                pressure_slope_window_s=pressure_slope_window_s,
                min_pressure_slope_bar_s=min_pressure_slope_bar_s,
                critical_window_s=critical_window_s,
                min_jump_slpm=min_jump_slpm,
            )
        )

        if save:
            print(f"All figures saved in: {self.figure_dir}")

        if show:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press Enter to close plots and continue...")
            plt.close("all")
            print("Plots closed.")

        return figures


# ----------------------------------------------------------------------
# Batch processing
# ----------------------------------------------------------------------


def _looks_like_controller_csv(csv_path: Path) -> bool:
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if not row:
                    continue

                stripped = [value.strip() for value in row]
                if stripped == CONTROLLER_COLUMNS:
                    return True

                if len(stripped) == len(CONTROLLER_COLUMNS):
                    try:
                        [float(value) for value in stripped]
                    except ValueError:
                        return False
                    return True

                return False
    except (OSError, UnicodeError, csv.Error):
        return False

    return False


def find_controller_csv(run_dir: Path) -> Optional[Path]:
    """Find the controller CSV for one run and ignore timestamp sidecars."""
    preferred = run_dir / f"{run_dir.name}.csv"
    if preferred.is_file() and _looks_like_controller_csv(preferred):
        return preferred

    candidates = []
    for csv_path in sorted(run_dir.glob("*.csv")):
        name = csv_path.name.lower()
        if name.endswith("_daq_timestamps.csv"):
            continue
        if name.endswith("_frame_timestamps.csv"):
            continue
        if _looks_like_controller_csv(csv_path):
            candidates.append(csv_path)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        print(
            f"Skipping {run_dir}: found multiple possible controller CSV files: "
            + ", ".join(path.name for path in candidates)
        )

    return None


def parse_run_parameters(run_name: str) -> tuple[float, float]:
    """Extract the two numeric parameters after the leading run prefix.

    Examples
    --------
    v10_24_32bis -> (24, 32)
    v10_18_22    -> (18, 22)
    """
    parts = run_name.split("_")
    values = []

    for token in parts[1:]:
        match = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)", token)
        if match is None:
            continue
        values.append(float(match.group(0)))
        if len(values) == 2:
            break

    while len(values) < 2:
        values.append(float("nan"))

    return values[0], values[1]


def _format_pair(first: float, second: float) -> str:
    if float(first) == 0.0 and float(second) == 0.0:
        return "(0, 0)"
    return f"({float(first):.9g}, {float(second):.9g})"


def build_transition_summary_row(run_name: str, results: dict) -> dict:
    parameter_1, parameter_2 = parse_run_parameters(run_name)

    return {
        "parameter_1": float(parameter_1),
        "parameter_2": float(parameter_2),

        "pin_up_avg": float(results["pin_up_avg"]),
        "Qin_up_avg": float(results["Qin_up_avg"]),

        "pin_up_std": float(results["pin_up_std"]),
        "Qin_up_std": float(results["Qin_up_std"]),

        "pin_down_avg": float(results["pin_down_avg"]),
        "Qin_down_avg": float(results["Qin_down_avg"]),

        "pin_down_std": float(results["pin_down_std"]),
        "Qin_down_std": float(results["Qin_down_std"]),
    }

def write_transition_summary(summary_path: str | Path, rows: list[dict]) -> None:
    columns = [
        "parameter_1",
        "parameter_2",
        "pin_up_avg",
        "Qin_up_avg",
        "pin_up_std",
        "Qin_up_std",
        "pin_down_avg",
        "Qin_down_avg",
        "pin_down_std",
        "Qin_down_std",
    ]

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(rows, columns=columns)

    # Explicitly ensure all columns are numeric.
    summary_df = summary_df.astype(float)

    summary_df.to_csv(summary_path, index=False)

    print(f"Transition summary saved: {summary_path.resolve()}")
def plot_all_runs(
    runs_dir: str | Path = "runs",
    n_repetitions: Optional[int] = None,
    timecutoff: float = 0.0,
    dpi: int = 200,
    show: bool = False,
    expected_transitions: int = 3,
    flat_run_threshold: float = 5.0,
    up_pre_max_slpm: float = 15.0,
    broken_threshold_slpm: float = 15.0,
    down_post_max_slpm: float = 10.0,
    min_state_change_slpm: float = 15.0,
    smooth_window_s: float = 0.15,
    jump_window_s: float = 0.60,
    guard_window_s: float = 0.10,
    state_window_s: float = 0.50,
    state_horizon_s: float = 8.0,
    pressure_slope_window_s: float = 0.75,
    min_pressure_slope_bar_s: float = 0.005,
    critical_window_s: float = 0.10,
    min_jump_slpm: float = 4.0,
    inlet_diameter_mm: float = DEFAULT_INLET_DIAMETER_MM,
    air_density_std_kg_m3: float = DEFAULT_AIR_DENSITY_STD_KG_M3,
    air_dynamic_viscosity_pa_s: float = DEFAULT_AIR_DYNAMIC_VISCOSITY_PA_S,
) -> None:
    """Process every immediate subfolder in ``runs_dir``.

    Figures and detailed transition events are written to each run's
    ``figures/`` directory. The six-column global transition summary is written
    next to the ``runs/`` directory.
    """
    runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir.resolve()}")
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {runs_dir.resolve()}")

    run_dirs = sorted(
        path for path in runs_dir.iterdir()
        if path.is_dir() and path.name != "figures"
    )

    if not run_dirs:
        print(f"No run folders found in: {runs_dir.resolve()}")
        return

    processed = 0
    skipped = 0
    failed = 0
    summary_rows: list[dict] = []
    summary_path = runs_dir.parent / "transition_summary.csv"

    print(f"Scanning {runs_dir.resolve()} for experiment folders...\n")
    print(
        "Inlet Reynolds configuration: "
        f"D={inlet_diameter_mm:g} mm, "
        f"rho_std={air_density_std_kg_m3:g} kg/m^3, "
        f"mu={air_dynamic_viscosity_pa_s:g} Pa s\n"
    )

    for run_dir in run_dirs:
        controller_csv = find_controller_csv(run_dir)

        if controller_csv is None:
            print(f"Skipping {run_dir.name}: no unique controller CSV found.\n")
            skipped += 1
            continue

        figure_dir = run_dir / "figures"
        print("=" * 72)
        print(f"Run:     {run_dir.name}")
        print(f"CSV:     {controller_csv}")
        print(f"Figures: {figure_dir}")

        try:
            experiment = ExperimentData(
                expfilename=controller_csv.stem,
                data_dir=run_dir,
                figure_dir=figure_dir,
                n_repetitions=n_repetitions,
                timecutoff=timecutoff,
                inlet_diameter_mm=inlet_diameter_mm,
                air_density_std_kg_m3=air_density_std_kg_m3,
                air_dynamic_viscosity_pa_s=air_dynamic_viscosity_pa_s,
            )

            experiment.plot_all(
                title=run_dir.name,
                save=True,
                show=show,
                dpi=dpi,
                full_experiment=True,
                expected_transitions=expected_transitions,
                flat_run_threshold=flat_run_threshold,
                up_pre_max_slpm=up_pre_max_slpm,
                broken_threshold_slpm=broken_threshold_slpm,
                down_post_max_slpm=down_post_max_slpm,
                min_state_change_slpm=min_state_change_slpm,
                smooth_window_s=smooth_window_s,
                jump_window_s=jump_window_s,
                guard_window_s=guard_window_s,
                state_window_s=state_window_s,
                state_horizon_s=state_horizon_s,
                pressure_slope_window_s=pressure_slope_window_s,
                min_pressure_slope_bar_s=min_pressure_slope_bar_s,
                critical_window_s=critical_window_s,
                min_jump_slpm=min_jump_slpm,
            )

            results = experiment.last_transition_results
            if results is None:
                results = ExperimentData._zero_transition_summary()

            summary_rows.append(build_transition_summary_row(run_dir.name, results))
            processed += 1

        except Exception as exc:
            failed += 1
            print(f"Failed to process {run_dir.name}: {exc}")

        finally:
            plt.close("all")

        print()

    if summary_rows:
        write_transition_summary(summary_path, summary_rows)

    print("=" * 72)
    print("Batch processing complete.")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan runs/<experiment>/ folders, plot every controller CSV, detect "
            "one up/down flow-asymmetry transition per pressure ramp, and write "
            "a global transition_summary.csv next to runs/."
        )
    )

    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing experiment folders (default: runs).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help=(
            "Optional repetition labels for plotting. This does not remove data "
            "and is not required by transition detection."
        ),
    )
    parser.add_argument(
        "--timecutoff",
        type=float,
        default=0.0,
        help="Ignore data before this experiment time in seconds (default: 0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved figure resolution (default: 200).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each run's figures interactively. Batch default is save-only.",
    )

    parser.add_argument(
        "--inlet-diameter-mm",
        type=float,
        default=DEFAULT_INLET_DIAMETER_MM,
        help=(
            "Circular diameter used for inlet Reynolds number in mm "
            f"(default: {DEFAULT_INLET_DIAMETER_MM:g}, the STL throat/orifice). "
            f"The STL connector bore is approximately {STL_INLET_PORT_DIAMETER_MM:g} mm."
        ),
    )
    parser.add_argument(
        "--air-density-std",
        type=float,
        default=DEFAULT_AIR_DENSITY_STD_KG_M3,
        help=(
            "Air density at the SLPM reference condition in kg/m^3 "
            f"(default: {DEFAULT_AIR_DENSITY_STD_KG_M3:g})."
        ),
    )
    parser.add_argument(
        "--air-dynamic-viscosity",
        type=float,
        default=DEFAULT_AIR_DYNAMIC_VISCOSITY_PA_S,
        help=(
            "Air dynamic viscosity in Pa s "
            f"(default: {DEFAULT_AIR_DYNAMIC_VISCOSITY_PA_S:g})."
        ),
    )

    parser.add_argument(
        "--expected-transitions",
        type=int,
        default=3,
        help="Expected pressure ramps / up and down transitions (default: 3).",
    )
    parser.add_argument(
        "--flat-run-threshold",
        type=float,
        default=5.0,
        help=(
            "If the entire smoothed |Q_left-Q_right| trace stays below this "
            "value, classify the whole run as flat and save zeros (default: 5 SLPM)."
        ),
    )
    parser.add_argument(
        "--up-pre-max",
        type=float,
        default=15.0,
        help=(
            "Largest allowed persistent |Qdiff| immediately before an UP "
            "transition (default: 15 SLPM)."
        ),
    )
    parser.add_argument(
        "--broken-threshold",
        type=float,
        default=15.0,
        help=(
            "Minimum persistent |Qdiff| representing the broken/asymmetric "
            "state (default: 15 SLPM)."
        ),
    )
    parser.add_argument(
        "--down-post-max",
        type=float,
        default=10.0,
        help=(
            "Largest allowed persistent |Qdiff| after a DOWN transition "
            "(default: 10 SLPM)."
        ),
    )
    parser.add_argument(
        "--min-state-change",
        type=float,
        default=15.0,
        help=(
            "Minimum persistent change in |Qdiff| between states for an accepted "
            "transition (default: 15 SLPM)."
        ),
    )
    parser.add_argument(
        "--transition-smooth",
        type=float,
        default=0.15,
        help="Median smoothing window in seconds (default: 0.15).",
    )
    parser.add_argument(
        "--jump-window",
        type=float,
        default=0.60,
        help=(
            "Length of each before/after Qdiff median window used for the "
            "finite-time jump (default: 0.60 s)."
        ),
    )
    parser.add_argument(
        "--guard-window",
        type=float,
        default=0.10,
        help=(
            "Time excluded on each side of a candidate before computing the "
            "before/after jump medians (default: 0.10 s)."
        ),
    )
    parser.add_argument(
        "--state-window",
        type=float,
        default=0.50,
        help="Window for robust state medians in seconds (default: 0.50).",
    )
    parser.add_argument(
        "--state-horizon",
        type=float,
        default=8.0,
        help=(
            "How far before/after a candidate to confirm the broken/low state "
            "(default: 8.0 s)."
        ),
    )
    parser.add_argument(
        "--pressure-slope-window",
        type=float,
        default=0.75,
        help="Window used to estimate p_in slope around a candidate (default: 0.75 s).",
    )
    parser.add_argument(
        "--min-pressure-slope",
        type=float,
        default=0.005,
        help=(
            "Minimum magnitude of p_in slope in bar/s for up/down classification "
            "(default: 0.005)."
        ),
    )
    parser.add_argument(
        "--critical-window",
        type=float,
        default=0.10,
        help=(
            "Median window around the detected event for reporting p_in and Q_in "
            "(default: 0.10 s)."
        ),
    )
    parser.add_argument(
        "--transition-min-jump",
        type=float,
        default=4.0,
        help=(
            "Minimum guarded finite-window jump in Qdiff for a candidate "
            "(default: 4 SLPM)."
        ),
    )

    args = parser.parse_args()

    plot_all_runs(
        runs_dir=args.runs_dir,
        n_repetitions=args.repetitions,
        timecutoff=args.timecutoff,
        dpi=args.dpi,
        show=args.show,
        expected_transitions=args.expected_transitions,
        flat_run_threshold=args.flat_run_threshold,
        up_pre_max_slpm=args.up_pre_max,
        broken_threshold_slpm=args.broken_threshold,
        down_post_max_slpm=args.down_post_max,
        min_state_change_slpm=args.min_state_change,
        smooth_window_s=args.transition_smooth,
        jump_window_s=args.jump_window,
        guard_window_s=args.guard_window,
        state_window_s=args.state_window,
        state_horizon_s=args.state_horizon,
        pressure_slope_window_s=args.pressure_slope_window,
        min_pressure_slope_bar_s=args.min_pressure_slope,
        critical_window_s=args.critical_window,
        min_jump_slpm=args.transition_min_jump,
        inlet_diameter_mm=args.inlet_diameter_mm,
        air_density_std_kg_m3=args.air_density_std,
        air_dynamic_viscosity_pa_s=args.air_dynamic_viscosity,
    )


if __name__ == "__main__":
    main()
