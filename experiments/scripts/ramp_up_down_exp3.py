"""Synchronized pressure-ramp experiment.

This version timestamps every DAQ sample and every video frame with the same
``time.perf_counter()`` clock.

For backward compatibility, the main DAQ CSV keeps the original schema used by
basic_plotter.py.  Host-clock timestamps are written to a separate
``*_daq_timestamps.csv`` sidecar keyed by sample_index.  Video timestamps are
written to ``*_frame_timestamps.csv`` keyed by video_frame_index.
"""




from __future__ import annotations

import json
import time
from pathlib import Path

import dynamixel_easy_sdk
import numpy as np
from simple_pid import PID

from basic_plotter import ExperimentData
from data_acquisition import DataAcquisition
from motor_control import AutoGROQS6
from u20_camera import U20Camera


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MOTOR_PORT = "/dev/cu.usbserial-FT9MG6RZ"
MOTOR_ID = 13
MOTOR_BAUDRATE = 57600

TORQUE_CONST = 0.352
MAX_TORQUE = 16
MAX_CURRENT = 150

DAQ_PORT = "/dev/cu.usbmodem1101"

GOAL_PRESSURE = 1
PID_KP = 0.4
PID_KI = 0.05
PID_KD = 0.02
PID_SAMPLE_TIME_S = 0.05

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_WARMUP_S = 0.5

RAMP_PERIOD_S = 30.0
SET_DELAY_S = 0.05
P_MIN = 0.0
P_MAX = 3.0
REPETITIONS = 3
HOLD_BEFORE_EACH_REP_S = 10.0
HOLD_AFTER_RUN_S = 10.0
CONTROL_LOOP_SLEEP_S = 0.025

RUNS_DIR = Path("runs")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    temporary.replace(path)


def finite_or_none(value):
    if value is None:
        return None
    return float(value)


def build_pressure_intervals() -> np.ndarray:
    divisions = max(2, round(RAMP_PERIOD_S / SET_DELAY_S))
    return np.concatenate(
        [
            np.linspace(P_MIN, P_MAX, divisions),
            np.linspace(P_MAX, P_MIN, divisions),
        ]
    )


def run_pid_for_duration(
    *,
    duration_s: float,
    setpoint: float,
    pid: PID,
    daq: DataAcquisition,
    motor: AutoGROQS6,
    camera: U20Camera,
) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        pid.setpoint = setpoint
        pressure = daq.get_channel("pres_in")

        if pressure is not None:
            control = pid(pressure)
            if control is not None:
                motor.setPercentagePosition(control)

        camera.update_view()
        time.sleep(CONTROL_LOOP_SLEEP_S)


def run_pressure_ramp(
    *,
    pressure_intervals: np.ndarray,
    pid: PID,
    daq: DataAcquisition,
    motor: AutoGROQS6,
    camera: U20Camera,
) -> None:
    for set_pressure in pressure_intervals:
        pid.setpoint = float(set_pressure)
        interval_start = time.perf_counter()

        while time.perf_counter() - interval_start < SET_DELAY_S:
            pressure = daq.get_channel("pres_in")

            if pressure is not None:
                control = pid(pressure)
                if control is not None:
                    motor.setPercentagePosition(control)

            camera.update_view()
            time.sleep(CONTROL_LOOP_SLEEP_S)


def make_final_sync_metadata(
    *,
    initial_metadata: dict,
    daq: DataAcquisition,
    camera: U20Camera,
    common_stop_request_host_s: float,
) -> dict:
    metadata = dict(initial_metadata)

    daq_first = daq.first_sample_host_time_s
    daq_last = daq.last_sample_host_time_s
    video_first = camera.first_recorded_frame_host_time_s
    video_last = camera.last_recorded_frame_host_time_s

    metadata.update(
        {
            "common_stop_request_host_s": common_stop_request_host_s,
            "daq_csv": str(daq.csv_path) if daq.csv_path is not None else None,
            "daq_timestamps_csv": (
                str(daq.timestamps_path)
                if daq.timestamps_path is not None
                else None
            ),
            "video_file": str(camera.video_path) if camera.video_path else None,
            "video_frame_timestamps_csv": (
                str(camera.frame_timestamps_path)
                if camera.frame_timestamps_path
                else None
            ),
            "daq_sample_count": daq.sample_count,
            "video_frame_count": camera.recorded_frame_count,
            "video_dropped_frame_count": camera.dropped_frame_count,
            "first_daq_sample_host_time_s": finite_or_none(daq_first),
            "last_daq_sample_host_time_s": finite_or_none(daq_last),
            "first_video_frame_host_time_s": finite_or_none(video_first),
            "last_video_frame_host_time_s": finite_or_none(video_last),
            "actual_video_fps_reported": float(camera.actual_fps),
        }
    )

    if None not in (daq_first, daq_last, video_first, video_last):
        overlap_start = max(daq_first, video_first)
        overlap_end = min(daq_last, video_last)
        overlap_duration = max(0.0, overlap_end - overlap_start)

        metadata.update(
            {
                "overlap_start_host_time_s": overlap_start,
                "overlap_end_host_time_s": overlap_end,
                "overlap_duration_s": overlap_duration,
                "daq_host_duration_s": daq_last - daq_first,
                "video_host_duration_s": video_last - video_first,
                "first_video_minus_first_daq_s": video_first - daq_first,
            }
        )

    return metadata


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    input("Close pre valve.")
    print("Initializing motor...")

    motor = AutoGROQS6(
        port=MOTOR_PORT,
        id=MOTOR_ID,
        baudrate=MOTOR_BAUDRATE,
        max_current=MAX_CURRENT,
    )
    motor.connect()
    motor.autoCalibration()

    motor.setPercentagePosition(1)
    while not motor.setPositionFinished():
        time.sleep(0.01)

    motor.setPercentagePosition(0)

    daq = DataAcquisition(
        port=DAQ_PORT,
        channel_names=[
            "pres_in",
            "flow_in",
            "flow_left",
            "flow_right",
            "pres_left",
            "pres_right",
        ],
        channel_rescale=[7, 200, 200, 200, 7, 7],
        output_directory=RUNS_DIR,
    )

    # Calibration remains a separate pre-run operation.
    daq.connect()
    daq.calibrate()
    daq.stop()

    while not motor.setPositionFinished():
        time.sleep(0.01)

    print("Initialization complete.")
    input("Open pre valve.")

    pid = PID(PID_KP, PID_KI, PID_KD, setpoint=GOAL_PRESSURE)
    pid.sample_time = PID_SAMPLE_TIME_S
    pid.output_limits = (0, 1)

    pressure_intervals = build_pressure_intervals()

    camera = U20Camera(
        camera_index=CAMERA_INDEX,
        output_folder=RUNS_DIR,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
    )

    try:
        while True:
            this_name = input("Run name: ").strip()
            if not this_name:
                print("Run name cannot be empty.")
                continue

            run_dir = RUNS_DIR / this_name
            if run_dir.exists() and any(run_dir.iterdir()):
                answer = input(
                    f"{run_dir} already contains files. Continue and overwrite? [y/N]: "
                ).strip().lower()
                if answer != "y":
                    continue
            run_dir.mkdir(parents=True, exist_ok=True)

            sync_path = run_dir / f"{this_name}_sync.json"

            run_started = False
            try:
                # ---------------------------------------------------------
                # Prepare both devices before choosing the common host t0.
                # ---------------------------------------------------------
                camera.output_folder = run_dir
                camera.start()
                camera.view()

                warmup_start = time.perf_counter()
                while time.perf_counter() - warmup_start < CAMERA_WARMUP_S:
                    camera.update_view()
                    time.sleep(0.01)

                # Reopening the port resets the Arduino.  Do this BEFORE t0.
                daq.connect()

                # ---------------------------------------------------------
                # One common perf_counter zero for ALL sample/frame times.
                # ---------------------------------------------------------
                host_time_zero_s = time.perf_counter()

                daq.start(
                    f"{this_name}/{this_name}",
                    host_time_zero_s=host_time_zero_s,
                )
                camera.start_recording(
                    f"{this_name}.mp4",
                    host_time_zero_s=host_time_zero_s,
                )
                run_started = True

                protocol_start_host_s = time.perf_counter()

                sync_metadata = {
                    "schema_version": 2,
                    "experiment": this_name,
                    "clock": "time.perf_counter",
                    "host_time_zero_s": host_time_zero_s,
                    "protocol_start_host_s": protocol_start_host_s,
                    "protocol_start_elapsed_s": (
                        protocol_start_host_s - host_time_zero_s
                    ),
                    "camera_requested_fps": CAMERA_FPS,
                    "camera_requested_width_px": CAMERA_WIDTH,
                    "camera_requested_height_px": CAMERA_HEIGHT,
                    "repetitions": REPETITIONS,
                    "ramp_period_s": RAMP_PERIOD_S,
                    "set_delay_s": SET_DELAY_S,
                    "p_min": P_MIN,
                    "p_max": P_MAX,
                    "status": "running",
                }
                write_json(sync_path, sync_metadata)

                print("\nSynchronized acquisition started.")
                print(f"DAQ CSV: {daq.csv_path}")
                print(f"DAQ timestamps: {daq.timestamps_path}")
                print(f"Video: {camera.video_path}")
                print(f"Frame times: {camera.frame_timestamps_path}")
                print(
                    "Use host_elapsed_s / host_time_s for synchronization, "
                    "not frame_index / nominal FPS.\n"
                )

                # ---------------------------------------------------------
                # Experiment protocol
                # ---------------------------------------------------------
                for rep in range(REPETITIONS):
                    print(f"Repetition {rep + 1}/{REPETITIONS}")
                    pid.reset()

                    run_pid_for_duration(
                        duration_s=HOLD_BEFORE_EACH_REP_S,
                        setpoint=float(pressure_intervals[0]),
                        pid=pid,
                        daq=daq,
                        motor=motor,
                        camera=camera,
                    )

                    run_pressure_ramp(
                        pressure_intervals=pressure_intervals,
                        pid=pid,
                        daq=daq,
                        motor=motor,
                        camera=camera,
                    )

                run_pid_for_duration(
                    duration_s=HOLD_AFTER_RUN_S,
                    setpoint=float(pressure_intervals[-1]),
                    pid=pid,
                    daq=daq,
                    motor=motor,
                    camera=camera,
                )

                protocol_end_host_s = time.perf_counter()
                sync_metadata["protocol_end_host_s"] = protocol_end_host_s
                sync_metadata["protocol_duration_s"] = (
                    protocol_end_host_s - protocol_start_host_s
                )

                # ---------------------------------------------------------
                # Request BOTH streams to stop almost simultaneously.
                # request_* calls are non-blocking; finalization happens after.
                # ---------------------------------------------------------
                common_stop_request_host_s = time.perf_counter()
                camera.request_stop_recording()
                daq.request_stop()

                # These may block while files/queues are flushed, but no new
                # synchronized samples/frames are accepted after the requests.
                camera.stop_recording()
                daq.stop()
                run_started = False

                sync_metadata["status"] = "complete"
                final_metadata = make_final_sync_metadata(
                    initial_metadata=sync_metadata,
                    daq=daq,
                    camera=camera,
                    common_stop_request_host_s=common_stop_request_host_s,
                )
                write_json(sync_path, final_metadata)

                camera.release()
                motor.setPercentagePosition(0)

                print("\nRun complete.")
                if "overlap_duration_s" in final_metadata:
                    print(
                        "True DAQ/video overlap: "
                        f"{final_metadata['overlap_duration_s']:.6f} s"
                    )
                print(f"Sync metadata: {sync_path}")

                # Plotting is post-processing.  A plotting/parser error must
                # never invalidate a run whose DAQ/video files were already
                # closed successfully.
                try:
                    experiment = ExperimentData(
                        expfilename=this_name,
                        data_dir=str(run_dir),
                        figure_dir=str(run_dir / "figures"),
                        n_repetitions=REPETITIONS,
                        repetition=1,
                    )
                    experiment.plot_all(title="", show=True, save=True)
                except Exception as plot_error:
                    print(
                        "\nRun acquisition is complete, but basic_plotter "
                        "could not make the plots:"
                    )
                    print(plot_error)
                    print(
                        "The raw CSV, DAQ timestamp sidecar, video, frame "
                        "timestamps, and sync JSON have been preserved."
                    )

            except Exception:
                # If a run fails, stop both streams before propagating the error.
                if run_started:
                    stop_request = time.perf_counter()
                    camera.request_stop_recording()
                    daq.request_stop()
                    camera.stop_recording()
                    daq.stop()

                    failure_metadata = {
                        "schema_version": 2,
                        "experiment": this_name,
                        "clock": "time.perf_counter",
                        "status": "failed",
                        "common_stop_request_host_s": stop_request,
                    }
                    write_json(sync_path, failure_metadata)

                camera.release()
                raise

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except dynamixel_easy_sdk.DxlRuntimeError:
        print("No status packet received.")

    finally:
        # Idempotent cleanup.
        try:
            camera.request_stop_recording()
            camera.stop_recording()
            camera.release()
        except Exception:
            pass

        try:
            daq.request_stop()
            daq.stop()
        except Exception:
            pass

        motor._disableTorque()
        motor.updateState()
        print("State saved.")


if __name__ == "__main__":
    main()
