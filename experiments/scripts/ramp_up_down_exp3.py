"""
Overarching controller: reads live pressure from the Arduino via DataAcquisition
and drives the valve motor via AutoGROQS6 through a PID loop.
"""

import time

from simple_pid import PID

from data_acquisition import DataAcquisition
from motor_control import AutoGROQS6
import dynamixel_easy_sdk
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import itertools as it 


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MOTOR_PORT = "COM5"
MOTOR_ID = 3
MOTOR_BAUDRATE = 57600

TORQUE_CONST = 0.352      # Nm/A, at stall
MAX_TORQUE = 16           # Nm
MAX_CURRENT = MAX_TORQUE / TORQUE_CONST
MAX_CURRENT = 150

DAQ_PORT = "COM8"   # change to e.g. "COM3" on Windows
# PRESSURE_CHANNEL = "p_in"             # whichever channel is the pressure sensor
# FLOW_1 = "A1"
# FLOW_2 = "A2"
# FLOW_3 = "A3"


daq_channels = ["p_in", "flow_in", "flow_left", "flow_right", "p_left", "p_right"]
daq_channels_rescale=[7, 200, 200, 200, 7, 7]
csv_channels = ["flow_in", "flow_left", "flow_right", "p_in", "p_left", "p_right"]


GOAL_PRESSURE = 1   # bar

PID_KP, PID_KI, PID_KD = 1, 0.5, 0.05
PID_SAMPLE_TIME_S = 0.01

REP_GAP = 3 # s


period = 20 # s
set_delay = 0.05 # s
no_divisions = round(period/set_delay)

p_min = 0
p_max = 3
reps = 3


def main() -> None:
    input("Close pre valve.")
    print("Initializing.")
    motor = AutoGROQS6(
        port=MOTOR_PORT,
        id=MOTOR_ID,
        baudrate=MOTOR_BAUDRATE,
        max_current=MAX_CURRENT,
    )
    motor.connect()
    # motor.loadCalibration()
    motor.autoCalibration()
    motor.setPercentagePosition(1)
    while True:
        if motor.setPositionFinished():
            break

    daq = DataAcquisition(
        port=DAQ_PORT,
        channel_names=daq_channels,
        channel_rescale=daq_channels_rescale
        )
    daq.connect()
    calibrations = daq.calibrate()

    motor.setPercentagePosition(0)
    while True:
        if motor.setPositionFinished():
            break
    
    print("Homing complete.")
    input("Open pre valve.")
    # daq.start()
    # time.sleep(1)
    # while True:
    #     p_min_hat = daq.get_channel("p_in")

    #     motor.setPercentagePosition(1)
    #     while True:
    #         if motor.setPositionFinished():
    #             break

    #     p_max_hat = daq.get_channel("p_in")
    #     motor.setPercentagePosition(0)
    #     while True:
    #         if motor.setPositionFinished():
    #             break
        
    #     print(f"[software, physical] max pressure: {p_max}, {p_max_hat}")
    #     print(f"[software, physical] min pressure: {p_min}, {p_min_hat}")
    #     if (p_max_hat<p_max) or (p_min_hat>p_min):
    #         input("Pressure bounds check failed. Adjust wall pressure.")
    #     else:
    #         input("Pressure bounds check passed.")
    #         break
    # daq.stop()
    # input("Experiment will now begin.")



    pid = PID(PID_KP, PID_KI, PID_KD, setpoint=GOAL_PRESSURE)
    pid.sample_time = PID_SAMPLE_TIME_S
    pid.output_limits = (0, 1)   # setPercentagePosition only accepts [0, 1]

    output_directory = Path("data/exp3")

    output_directory.mkdir(parents=True, exist_ok=True)
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _csv_path = output_directory / f"{file_timestamp}.csv"
    _csv_file = _csv_path.open("w", newline="", encoding="utf-8")
    _csv_writer = csv.writer(_csv_file)
    header = ["time"] + csv_channels
    _csv_writer.writerow(header)

    # pressure_intervals = np.concatenate([
    #     np.arange(p_min, p_max, delay_interval), 
    #     np.arange(p_max, p_min - delay_interval, -delay_interval)
    #     ])
    
    pressure_intervals = np.concatenate([
        np.linspace(p_min, p_max, no_divisions),
        np.linspace(p_max, p_min, no_divisions)
    ])
    
    global_start_time = time.time()
    try:
        save_name = input("Input save name (e.g. l20_w10_h7.5): ")
        daq.start(file_name=save_name)
        for rep in range(reps):
            rep_start = time.time()
            while (time.time()-rep_start)<REP_GAP:
                pass

            print("_________________Rep: ", rep)
            for set_pressure in pressure_intervals:
                pid.setpoint = set_pressure
                # print("Set pressure: ", set_pressure)
                start_time = time.time()
                while (time.time()-start_time) < set_delay:
                    pressure = daq.get_channel("p_in")
                    print(set_pressure, pressure)
                    if pressure is None:
                        # No sample received yet; skip this cycle.
                        time.sleep(PID_SAMPLE_TIME_S)
                        continue
                    
                    control = pid(pressure)
                    if control is not None:
                        motor.setPercentagePosition(control)
                        # pass
                    _csv_writer.writerow(
                        [time.time()-global_start_time]
                        + [daq.get_channel(ch) for ch in csv_channels]
                        )
        daq.stop()

    except KeyboardInterrupt:
        print("\nStopped by user.")
        motor._disableTorque()
        motor.updateState()
        print("State saved")

    except Exception as e:
        print(e)
        motor._disableTorque()
        motor.updateState()
        print("State saved")

    except dynamixel_easy_sdk.DxlRuntimeError:
        motor.updateState()
        print("No status packet received.")
        pass

    finally:
        motor._disableTorque()
        motor.updateState()
        print("State saved.")
        daq.stop()


if __name__ == "__main__":
    main()
