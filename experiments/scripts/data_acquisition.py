from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import serial
from serial.tools import list_ports


class DataAcquisition:
    """
    Reads two-channel Arduino data over serial, calibrates a zero-reference,
    logs raw + normalized samples to CSV, and exposes the latest normalized
    reading for a live control loop to poll.

    Typical usage from another script:

        daq = DataAcquisition(port="/dev/cu.usbmodem1101")
        daq.connect()
        daq.calibrate()      # blocks for `calibration_duration_s` seconds
        daq.start()          # begins background logging thread

        pressure = daq.get_channel("A0")   # non-blocking, latest value

        daq.stop()            # when done
    """

    def __init__(
        self,
        port: str = "/dev/cu.usbmodem1101",
        baud_rate: int = 250_000,
        timeout_s: float = 1.0,
        channel_names: Optional[list[str]] = None,
        calibration_duration_s: float = 5.0,
        adc_full_scale: float = 1024.0,
        as_percent: bool = False,
        output_directory: str | Path = "experiment_data",
        flush_every_n_samples: int = 100,
        channel_rescale=None,
        reset_settle_timeout_s: float = 5.0,
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout_s = timeout_s
        self.channel_names = channel_names or ["A0", "A1"]
        self.calibration_duration_s = calibration_duration_s
        self.adc_full_scale = adc_full_scale
        self.as_percent = as_percent
        self.output_directory = Path(output_directory)
        self.flush_every_n_samples = flush_every_n_samples
        self.channel_rescale = dict([channel_names[i], channel_rescale[i]] for i in range(len(channel_names)))
        self.reset_settle_timeout_s = reset_settle_timeout_s

        self.ser: Optional[serial.Serial] = None
        self.calibration_averages: dict[str, float] = {}

        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer = None

        self._latest_raw: dict[str, int] = {}
        self._latest_normalized: dict[str, float] = {}
        self._latest_time_s: Optional[float] = None
        self._sample_count = 0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the serial port. Raises RuntimeError on failure."""
        print(f"Opening serial port {self.port} at {self.baud_rate} baud...")
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout_s,
            )
        except serial.SerialException as exc:
            self._show_available_ports()
            raise RuntimeError(f"Could not open serial port: {exc}") from exc

        # Opening the serial port normally resets the Arduino.
        self.ser.reset_input_buffer()
        print("Serial port opened.")

        self._wait_for_fresh_reset()

    def _wait_for_fresh_reset(self, timeout_s: Optional[float] = None) -> None:
        """
        Block until data from a genuinely fresh Arduino boot starts
        arriving, discarding anything read before that.

        Opening the serial port toggles DTR and resets the Arduino, but
        bytes already in flight from the *previous* session (still in the
        Arduino's UART buffer, the OS receive buffer, or the USB-serial
        chip's FIFO) can keep arriving for a short while after reopening,
        before the reboot actually takes effect. reset_input_buffer()
        only clears what's already buffered at the moment it's called, so
        it can't catch those late-arriving stale bytes on its own.

        A genuine reset shows up as elapsed time dropping (a new boot
        starts counting from ~0) rather than continuing to climb from
        wherever the old session left off. We read and discard lines
        until we see that drop, or until the timeout elapses without ever
        seeing one (e.g. this really is the very first connection, so
        there's no stale data to skip past).
        """
        if timeout_s is None:
            timeout_s = self.reset_settle_timeout_s

        deadline = time.monotonic() + timeout_s
        last_elapsed_us: Optional[int] = None

        while time.monotonic() < deadline:
            parsed = self._read_one_line()
            if parsed is None:
                continue

            _, elapsed_us, _ = parsed

            if last_elapsed_us is None:
                if elapsed_us < 500_000:
                    # First line we've seen already looks like a fresh
                    # boot (well under a second in) - nothing stale to skip.
                    return
            elif elapsed_us < last_elapsed_us:
                # Timer went backwards: this line is from a new boot.
                return

            last_elapsed_us = elapsed_us

        print(
            "[DataAcquisition] Warning: never observed a clean Arduino "
            "reset within the settle window; proceeding with whatever "
            "data is arriving. Early samples may be stale."
        )

    @staticmethod
    def _show_available_ports() -> None:
        ports = list(list_ports.comports())
        if not ports:
            print("No serial ports were detected.")
            return
        print("Available serial ports:")
        for port in ports:
            print(f"  {port.device}: {port.description}")

    def _read_one_line(self) -> Optional[tuple[float, int, list[int]]]:
        """Read and parse a single line. Returns (time_s, elapsed_us, values) or None."""
        raw_line = self.ser.readline()
        if not raw_line:
            return None

        try:
            line = raw_line.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError:
            return None

        if not line:
            return None

        fields = line.split()
        if len(fields) != len(self.channel_names) + 1:
            return None

        try:
            elapsed_us = int(fields[0])
            channel_values = [int(value) for value in fields[1:]]
        except ValueError:
            return None

        time_s = elapsed_us / 1_000_000.0
        return time_s, elapsed_us, channel_values

    # ------------------------------------------------------------------
    # Calibration (blocking, run once before start())
    # ------------------------------------------------------------------

    def calibrate(self) -> dict[str, float]:
        """
        Block for `calibration_duration_s` seconds, collecting samples to
        compute a per-channel zero-reference average. Must be called after
        connect() and before start().
        """
        if self.ser is None:
            raise RuntimeError("Call connect() before calibrate().")

        print(
            f"Calibrating for {self.calibration_duration_s:g} seconds "
            "(keep the system at zero-reference conditions)..."
        )

        sums = {name: 0.0 for name in self.channel_names}
        counts = {name: 0 for name in self.channel_names}
        start_time = time.monotonic()

        while (time.monotonic() - start_time) < self.calibration_duration_s:
            parsed = self._read_one_line()
            if parsed is None:
                continue
            _, _, channel_values = parsed
            for name, value in zip(self.channel_names, channel_values):
                sums[name] += value
                counts[name] += 1

        self.calibration_averages = {}
        for name in self.channel_names:
            if counts[name] == 0:
                raise RuntimeError(
                    f"No samples were received for channel {name} during calibration."
                )
            average = sums[name] / counts[name]
            denominator = self.adc_full_scale - average
            if denominator <= 0:
                raise RuntimeError(
                    f"Cannot calibrate {name}: calibration average is "
                    f"{average:.6f}, producing a denominator of {denominator:.6f}."
                )
            self.calibration_averages[name] = average

        print("Calibration averages:")
        for name, average in self.calibration_averages.items():
            print(f"  {name}: {average:.6f} ADC counts")

        return self.calibration_averages

    def _normalize(self, name: str, raw_value: int) -> float:
        average = self.calibration_averages[name]
        scaler = self.channel_rescale[name]
        denominator = self.adc_full_scale - average
        normalized = (raw_value - average) / denominator
        # if self.as_percent:
        #     normalized *= 100.0
        return normalized*scaler

    # ------------------------------------------------------------------
    # Background acquisition thread
    # ------------------------------------------------------------------

    def start(self, file_name=None) -> None:
        """Open the CSV log and start the background reading thread."""
        if self.ser is None:
            raise RuntimeError("Call connect() before start().")
        if not self.calibration_averages:
            raise RuntimeError("Call calibrate() before start().")
        if self._thread is not None:
            raise RuntimeError("start() was already called.")

        self.output_directory.mkdir(parents=True, exist_ok=True)
        csv_name = file_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = self.output_directory / f"{csv_name}.csv"

        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        header = ["time_s", "elapsed_us"]
        for name in self.channel_names:
            header.append(name)
            # header.append(f"{name}%")
        self._csv_writer.writerow(header)
        self._csv_file.flush()

        print(f"Logging to: {self._csv_path.resolve()}")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self._thread.start()

    def _acquisition_loop(self) -> None:
        while not self._stop_event.is_set():
            parsed = self._read_one_line()
            if parsed is None:
                continue

            time_s, elapsed_us, channel_values = parsed
            normalized_values = [
                self._normalize(name, value)
                for name, value in zip(self.channel_names, channel_values)
            ]

            row = [f"{time_s:.6f}", elapsed_us]
            for raw, norm in zip(channel_values, normalized_values):
                # row.append(raw)
                row.append(f"{norm:.8f}")
            self._csv_writer.writerow(row)

            self._sample_count += 1
            if self._sample_count % self.flush_every_n_samples == 0:
                self._csv_file.flush()

            with self._lock:
                self._latest_time_s = time_s
                # self._latest_raw = dict(zip(self.channel_names, channel_values))
                self._latest_normalized = dict(zip(self.channel_names, normalized_values))

    # ------------------------------------------------------------------
    # Live polling API (call from your control loop)
    # ------------------------------------------------------------------

    def get_channel(self, name: str) -> Optional[float]:
        """Latest normalized (calibrated) reading for one channel, or None if no data yet."""
        with self._lock:
            return self._latest_normalized.get(name)

    def get_latest(self) -> dict:
        """Latest normalized readings for all channels, plus timestamp. Non-blocking."""
        with self._lock:
            return {
                "time_s": self._latest_time_s,
                "normalized": dict(self._latest_normalized),
                "raw": dict(self._latest_raw),
            }

    @property
    def sample_count(self) -> int:
        return self._sample_count

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop the background thread, flush and close the CSV, close the serial port."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            print(f"CSV closed: {self._csv_path.resolve()}")

        if self.ser is not None:
            self.ser.close()
            self.ser = None
            print("Serial port closed.")

    def __enter__(self) -> "DataAcquisition":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


if __name__ == "__main__":
    # Standalone smoke test: log for 15 seconds then plot, same behavior
    # as the original single-file script (minus the reprocessing step,
    # since normalization now happens live).
    import matplotlib.pyplot as plt
    import pandas as pd

    daq = DataAcquisition(
        port="COM7",
        channel_names=["pres_in", "flow_in", "flow_left", "flow_right", "pres_left", "pres_right"],
        channel_rescale=[7, 200, 200, 200, 7, 7]
        )
    daq.connect()
    daq.calibrate()
    daq.start()

    try:
        print("Recording for 15 seconds. Press Ctrl+C to stop early.")
        start = time.monotonic()
        while (time.monotonic() - start) < 2.0:
            pressure = daq.get_channel("pres_in")
            print(pressure)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        daq.stop()

    # data = pd.read_csv(daq._csv_path)
    # fig, ax = plt.subplots(figsize=(12, 6))
    # for name in daq.channel_names:
    #     ax.plot(data["time_s"], data[f"{name}%"], label=f"{name}%", linewidth=0.8)
    # ax.set_xlabel("Time from experiment start (s)")
    # ax.set_ylabel("Normalized reading (fraction)")
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    # fig.tight_layout()
    # plt.show()