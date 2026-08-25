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
    """Serial DAQ with a host ``perf_counter()`` timestamp on every sample."""

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
    ) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.timeout_s = timeout_s
        self.channel_names = channel_names or ["A0", "A1"]
        self.calibration_duration_s = calibration_duration_s
        self.adc_full_scale = adc_full_scale
        self.as_percent = as_percent
        self.output_directory = Path(output_directory)
        self.flush_every_n_samples = flush_every_n_samples

        if channel_rescale is None:
            channel_rescale = [1.0] * len(self.channel_names)
        if len(channel_rescale) != len(self.channel_names):
            raise ValueError("channel_rescale must match channel_names length.")
        self.channel_rescale = dict(zip(self.channel_names, channel_rescale))

        self.reset_settle_timeout_s = reset_settle_timeout_s

        self.ser: Optional[serial.Serial] = None
        self.calibration_averages: dict[str, float] = {}

        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer = None

        # Host-clock timestamps are intentionally stored in a sidecar CSV.
        # This keeps the main controller CSV schema backward-compatible with
        # basic_plotter.py and any existing analysis scripts.
        self._timestamps_path: Optional[Path] = None
        self._timestamps_file = None
        self._timestamps_writer = None

        self._latest_raw: dict[str, int] = {}
        self._latest_normalized: dict[str, float] = {}
        self._latest_time_s: Optional[float] = None
        self._latest_host_time_s: Optional[float] = None

        self._sample_count = 0
        self._first_sample_host_time_s: Optional[float] = None
        self._last_sample_host_time_s: Optional[float] = None
        self._host_time_zero_s: Optional[float] = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self.ser is not None and self.ser.is_open:
            return

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

        self.ser.reset_input_buffer()
        print("Serial port opened.")
        self._wait_for_fresh_reset()

    def _wait_for_fresh_reset(self, timeout_s: Optional[float] = None) -> None:
        if timeout_s is None:
            timeout_s = self.reset_settle_timeout_s

        deadline = time.monotonic() + timeout_s
        last_elapsed_us: Optional[int] = None

        while time.monotonic() < deadline:
            parsed = self._read_one_line()
            if parsed is None:
                continue

            _, elapsed_us, _, _ = parsed

            if last_elapsed_us is None:
                if elapsed_us < 500_000:
                    return
            elif elapsed_us < last_elapsed_us:
                return

            last_elapsed_us = elapsed_us

        print(
            "[DataAcquisition] Warning: never observed a clean Arduino reset "
            "within the settle window; proceeding with incoming data."
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

    def _read_one_line(
        self,
    ) -> Optional[tuple[float, int, list[int], float]]:
        """Return device time, elapsed_us, values, and host perf_counter time."""
        if self.ser is None:
            return None

        raw_line = self.ser.readline()
        host_time_s = time.perf_counter()

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
        return time_s, elapsed_us, channel_values, host_time_s

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self) -> dict[str, float]:
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
            _, _, channel_values, _ = parsed
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
                    f"Cannot calibrate {name}: average={average:.6f}, "
                    f"denominator={denominator:.6f}."
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
        return normalized * scaler

    # ------------------------------------------------------------------
    # Background acquisition
    # ------------------------------------------------------------------

    def start(
        self,
        file_name: Optional[str] = None,
        host_time_zero_s: Optional[float] = None,
    ) -> None:
        if self.ser is None:
            raise RuntimeError("Call connect() before start().")
        if not self.calibration_averages:
            raise RuntimeError("Call calibrate() before start().")
        if self._thread is not None:
            raise RuntimeError("start() was already called.")

        self.output_directory.mkdir(parents=True, exist_ok=True)
        csv_name = file_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = self.output_directory / f"{csv_name}.csv"
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        # IMPORTANT: preserve the legacy controller CSV format expected by
        # basic_plotter.py:
        # time_s, elapsed_us, pres_in, flow_in, flow_left, flow_right,
        # pres_left, pres_right
        header = ["time_s", "elapsed_us"]
        header.extend(self.channel_names)
        self._csv_writer.writerow(header)
        self._csv_file.flush()

        # Store the shared-host-clock timestamp for every DAQ sample in a
        # separate sidecar keyed by sample_index.  This gives exact software
        # synchronization without changing the controller CSV schema.
        self._timestamps_path = self._csv_path.with_name(
            f"{self._csv_path.stem}_daq_timestamps.csv"
        )
        self._timestamps_file = self._timestamps_path.open(
            "w", newline="", encoding="utf-8"
        )
        self._timestamps_writer = csv.writer(self._timestamps_file)
        self._timestamps_writer.writerow(
            [
                "sample_index",
                "host_time_s",
                "host_elapsed_s",
                "time_s",
                "elapsed_us",
            ]
        )
        self._timestamps_file.flush()

        self._host_time_zero_s = (
            time.perf_counter()
            if host_time_zero_s is None
            else float(host_time_zero_s)
        )
        self._sample_count = 0
        self._first_sample_host_time_s = None
        self._last_sample_host_time_s = None

        print(f"Logging to: {self._csv_path.resolve()}")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._acquisition_loop,
            name="DataAcquisition",
            daemon=True,
        )
        self._thread.start()

    def _acquisition_loop(self) -> None:
        while not self._stop_event.is_set():
            parsed = self._read_one_line()
            if parsed is None:
                continue

            # If stop was requested while readline() was blocked, do not append
            # a sample after the common stop request.
            if self._stop_event.is_set():
                break

            time_s, elapsed_us, channel_values, host_time_s = parsed
            normalized_values = [
                self._normalize(name, value)
                for name, value in zip(self.channel_names, channel_values)
            ]

            host_elapsed_s = (
                host_time_s - self._host_time_zero_s
                if self._host_time_zero_s is not None
                else float("nan")
            )

            sample_index = self._sample_count

            # Legacy/main controller CSV.
            row = [
                f"{time_s:.6f}",
                elapsed_us,
            ]
            row.extend(f"{value:.8f}" for value in normalized_values)
            self._csv_writer.writerow(row)

            # Synchronization sidecar. sample_index is the zero-based row
            # number in the data section of the main controller CSV.
            if self._timestamps_writer is not None:
                self._timestamps_writer.writerow(
                    [
                        sample_index,
                        f"{host_time_s:.9f}",
                        f"{host_elapsed_s:.9f}",
                        f"{time_s:.6f}",
                        elapsed_us,
                    ]
                )

            if self._first_sample_host_time_s is None:
                self._first_sample_host_time_s = host_time_s
            self._last_sample_host_time_s = host_time_s

            self._sample_count += 1
            if self._sample_count % self.flush_every_n_samples == 0:
                self._csv_file.flush()
                if self._timestamps_file is not None:
                    self._timestamps_file.flush()

            with self._lock:
                self._latest_time_s = time_s
                self._latest_host_time_s = host_time_s
                self._latest_raw = dict(zip(self.channel_names, channel_values))
                self._latest_normalized = dict(
                    zip(self.channel_names, normalized_values)
                )

    def request_stop(self) -> None:
        """Request acquisition to stop immediately without waiting for file close."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Live API
    # ------------------------------------------------------------------

    def get_channel(self, name: str) -> Optional[float]:
        with self._lock:
            return self._latest_normalized.get(name)

    def get_latest(self) -> dict:
        with self._lock:
            return {
                "time_s": self._latest_time_s,
                "host_time_s": self._latest_host_time_s,
                "normalized": dict(self._latest_normalized),
                "raw": dict(self._latest_raw),
            }

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def first_sample_host_time_s(self) -> Optional[float]:
        return self._first_sample_host_time_s

    @property
    def last_sample_host_time_s(self) -> Optional[float]:
        return self._last_sample_host_time_s

    @property
    def csv_path(self) -> Optional[Path]:
        return self._csv_path

    @property
    def timestamps_path(self) -> Optional[Path]:
        return self._timestamps_path

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self.request_stop()

        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.timeout_s + 1.0))
            if self._thread.is_alive():
                print(
                    "[DataAcquisition] Warning: acquisition thread did not "
                    "exit before timeout."
                )
            self._thread = None

        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            if self._csv_path is not None:
                print(f"CSV closed: {self._csv_path.resolve()}")

        if self._timestamps_file is not None:
            self._timestamps_file.flush()
            self._timestamps_file.close()
            self._timestamps_file = None
            self._timestamps_writer = None
            if self._timestamps_path is not None:
                print(
                    "DAQ timestamps closed: "
                    f"{self._timestamps_path.resolve()}"
                )

        if self.ser is not None:
            self.ser.close()
            self.ser = None
            print("Serial port closed.")

    def __enter__(self) -> "DataAcquisition":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
