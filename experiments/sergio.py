from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import serial
from serial.tools import list_ports


# ---------------------------------------------------------------------
# Configuration, just for 2 channels A0 and A1
# ---------------------------------------------------------------------

SERIAL_PORT = "/dev/cu.usbmodem1101" #Change this to COM3 or whatever port for Windows :) 
BAUD_RATE = 250_000
SERIAL_TIMEOUT_SECONDS = 1.0

OUTPUT_DIRECTORY = Path("experiment_data")
FLUSH_EVERY_N_SAMPLES = 100

CHANNEL_NAMES = ["A0", "A1"]

# Initial zero-reference acquisition period
CALIBRATION_DURATION_SECONDS = 5.0

# Arduino Mega ADC range is 0 to 1023.
# The requested normalization uses 1024 in the denominator.
ADC_FULL_SCALE = 1024.0

# Keep False to implement exactly:
# (ADC - calibration) / (1024 - calibration)
#
# Set True to multiply the result by 100.
STORE_AS_PERCENT_0_TO_100 = False


def show_available_ports() -> None:
    """Print serial ports currently visible to Python."""
    ports = list(list_ports.comports())

    if not ports:
        print("No serial ports were detected.")
        return

    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")


def create_output_paths() -> tuple[Path, Path]:
    """Create timestamped CSV and plot filenames."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = OUTPUT_DIRECTORY / f"{file_timestamp}.csv"
    plot_path = OUTPUT_DIRECTORY / f"{file_timestamp}_plot.png"

    return csv_path, plot_path


def acquire_data(csv_path: Path) -> int:
    """
    Continuously read Arduino data and save raw measurements to CSV.

    Expected Arduino format:
        elapsed_us<TAB>A0<TAB>A1

    Stop acquisition with Ctrl+C.
    """
    sample_count = 0

    print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE} baud...")

    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=SERIAL_TIMEOUT_SECONDS,
        )
    except serial.SerialException as exc:
        print(f"Could not open serial port: {exc}")
        show_available_ports()
        return 0

    print("Serial port opened.")
    print(
        f"The first {CALIBRATION_DURATION_SECONDS:g} seconds "
        "will be used for zero-reference calibration."
    )

    # Opening the serial port normally resets the Arduino.
    ser.reset_input_buffer()

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(
                [
                    "time_s",
                    "elapsed_us",
                    *CHANNEL_NAMES,
                ]
            )
            csv_file.flush()

            print(f"Recording to: {csv_path.resolve()}")
            print("Press Ctrl+C to stop acquisition.")

            calibration_message_printed = False

            while True:
                raw_line = ser.readline()

                if not raw_line:
                    continue

                try:
                    line = raw_line.decode(
                        "ascii",
                        errors="strict",
                    ).strip()
                except UnicodeDecodeError:
                    continue

                if not line:
                    continue

                fields = line.split()

                if len(fields) != len(CHANNEL_NAMES) + 1:
                    print(f"Skipped malformed line: {line!r}")
                    continue

                try:
                    elapsed_us = int(fields[0])
                    channel_values = [int(value) for value in fields[1:]]
                except ValueError:
                    print(f"Skipped non-numeric line: {line!r}")
                    continue

                time_s = elapsed_us / 1_000_000.0

                writer.writerow(
                    [
                        f"{time_s:.6f}",
                        elapsed_us,
                        *channel_values,
                    ]
                )

                sample_count += 1

                if sample_count % FLUSH_EVERY_N_SAMPLES == 0:
                    csv_file.flush()

                if (
                    not calibration_message_printed
                    and time_s >= CALIBRATION_DURATION_SECONDS
                ):
                    print(
                        "\nCalibration interval completed. "
                        "Continuing experiment acquisition."
                    )
                    calibration_message_printed = True

                if sample_count % 1000 == 0:
                    phase = (
                        "CALIBRATING"
                        if time_s < CALIBRATION_DURATION_SECONDS
                        else "RECORDING"
                    )

                    print(
                        f"\r{phase}: {sample_count:,} samples, "
                        f"{time_s:.2f} s elapsed",
                        end="",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\nAcquisition stopped by the user.")

    except serial.SerialException as exc:
        print(f"\nSerial communication error: {exc}")

    finally:
        ser.close()
        print("Serial port closed.")

    return sample_count


def calibrate_csv(csv_path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Calculate calibration averages and add normalized channel columns.

    For each channel:

        normalized =
            (ADC - calibration_average)
            / (1024 - calibration_average)

    The processed data replaces the original CSV.
    """
    try:
        data = pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError(f"Could not read CSV: {exc}") from exc

    if data.empty:
        raise RuntimeError("The CSV contains no data.")

    required_columns = ["time_s", "elapsed_us", *CHANNEL_NAMES]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"CSV is missing required columns: {missing_columns}"
        )

    # Include samples starting at zero and strictly before 5 seconds.
    calibration_mask = (
        (data["time_s"] >= 0.0)
        & (data["time_s"] < CALIBRATION_DURATION_SECONDS)
    )

    calibration_data = data.loc[calibration_mask, CHANNEL_NAMES]

    if calibration_data.empty:
        raise RuntimeError(
            "No samples were found in the calibration interval."
        )

    calibration_averages: dict[str, float] = {}

    for channel_name in CHANNEL_NAMES:
        calibration_value = calibration_data[channel_name].mean()
        calibration_averages[channel_name] = float(calibration_value)

        denominator = ADC_FULL_SCALE - calibration_value

        if denominator <= 0:
            raise RuntimeError(
                f"Cannot calibrate {channel_name}: calibration average "
                f"is {calibration_value:.6f}, producing a denominator "
                f"of {denominator:.6f}."
            )

        calibrated_column_name = f"{channel_name}%"

        calibrated_values = (
            data[channel_name] - calibration_value
        ) / denominator

        if STORE_AS_PERCENT_0_TO_100:
            calibrated_values = calibrated_values * 100.0

        data[calibrated_column_name] = calibrated_values

    # Rewrite the CSV, now including calibrated columns.
    data.to_csv(
        csv_path,
        index=False,
        float_format="%.8f",
    )

    return data, calibration_averages


def create_plot(
    data: pd.DataFrame,
    calibration_averages: dict[str, float],
    plot_path: Path,
) -> None:
    """Display and save calibrated channel readings."""
    figure, axis = plt.subplots(figsize=(12, 6))

    for channel_name in CHANNEL_NAMES:
        calibrated_column_name = f"{channel_name}%"

        axis.plot(
            data["time_s"],
            data[calibrated_column_name],
            label=calibrated_column_name,
            linewidth=0.8,
        )

    # Mark the end of the initial calibration interval.
    axis.axvline(
        CALIBRATION_DURATION_SECONDS,
        linestyle="--",
        linewidth=1.0,
        label="Calibration end",
    )

    unit_label = (
        "Normalized reading (%)"
        if STORE_AS_PERCENT_0_TO_100
        else "Normalized reading (fraction)"
    )

    calibration_text = "\n".join(
        f"{channel}: {average:.3f} ADC"
        for channel, average in calibration_averages.items()
    )

    axis.text(
        0.99,
        0.98,
        f"Calibration averages\n{calibration_text}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
        },
    )

    axis.set_title("Calibrated Arduino Analog Channels")
    axis.set_xlabel("Time from experiment start (s)")
    axis.set_ylabel(unit_label)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    try:
        figure.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Plot saved to: {plot_path.resolve()}")
    except OSError as exc:
        print(f"Could not save the plot: {exc}")

    plt.show()
    plt.close(figure)

    try:
        figure, axis = plt.subplots(figsize=(12, 6))

        axis.plot(
                    data["A0%"],
                    data["A1%"],
                    linewidth=0.8, color='k'
                )
        plt.show()
        plt.close(figure)

    except:
        print(f"Could not save the plot: {exc}")

def main() -> None:
    csv_path, plot_path = create_output_paths()

    sample_count = acquire_data(csv_path)

    if sample_count == 0:
        print("No valid samples were recorded.")
        sys.exit(1)

    print(f"Recorded {sample_count:,} valid samples.")

    try:
        data, calibration_averages = calibrate_csv(csv_path)
    except RuntimeError as exc:
        print(f"Calibration failed: {exc}")
        sys.exit(1)

    print("\nCalibration averages:")

    for channel_name, average in calibration_averages.items():
        print(f"  {channel_name}: {average:.6f} ADC counts")

    print(f"Calibrated CSV saved to: {csv_path.resolve()}")

    create_plot(
        data=data,
        calibration_averages=calibration_averages,
        plot_path=plot_path,
    )


if __name__ == "__main__":
    main()