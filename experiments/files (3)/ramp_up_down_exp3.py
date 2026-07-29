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
PRESSURE_CHANNEL = "A0"             # whichever channel is the pressure sensor
FLOW_1 = "A1"
FLOW_2 = "A2"
FLOW_3 = "A3"
FLOW_4 = "A4"




GOAL_PRESSURE = 1   # normalized fraction, in the same units DataAcquisition reports

PID_KP, PID_KI, PID_KD = 1, 0.5, 0.05
PID_SAMPLE_TIME_S = 0.01


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
        channel_names=["A0", "A1", "A2", "A3", "A4"]
        )
    daq.connect()
    calibrations = daq.calibrate()

    motor.setPercentagePosition(0.05)
    while True:
        if motor.setPositionFinished():
            break
    
    print("Initialization complete.")
    input("Open pre valve.")

    pid = PID(PID_KP, PID_KI, PID_KD, setpoint=GOAL_PRESSURE)
    pid.sample_time = PID_SAMPLE_TIME_S
    pid.output_limits = (0, 1)   # setPercentagePosition only accepts [0, 1]

    output_directory = Path("exp3")

    output_directory.mkdir(parents=True, exist_ok=True)
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _csv_path = output_directory / f"{file_timestamp}.csv"
    _csv_file = _csv_path.open("w", newline="", encoding="utf-8")
    _csv_writer = csv.writer(_csv_file)
    header = ["time", "p_in", "flow_0", "flow_1", "flow_2", "flow_3"]
    _csv_writer.writerow(header)


    period = 5 # s
    set_delay = 0.05 # s
    no_divisions = round(period/set_delay)

    p_min = 0
    p_max = 0.5
    reps = 3

    # pressure_intervals = np.concatenate([
    #     np.arange(p_min, p_max, delay_interval), 
    #     np.arange(p_max, p_min - delay_interval, -delay_interval)
    #     ])
    
    pressure_intervals = np.concatenate([
        np.linspace(p_min, p_max, no_divisions),
        np.linspace(p_max, p_min, no_divisions)
    ])
    
    daq.start()
    global_start_time = time.time()
    try:
        for rep in range(reps):
            print("_________________Rep: ", rep)
            for set_pressure in pressure_intervals:
                pid.setpoint = set_pressure
                print("Set pressure: ", set_pressure)
                start_time = time.time()
                while (time.time()-start_time) < set_delay:
                    pressure = daq.get_channel(PRESSURE_CHANNEL)
                    if pressure is None:
                        # No sample received yet; skip this cycle.
                        time.sleep(PID_SAMPLE_TIME_S)
                        continue
                    else:
                        pressure*=7
                    
                    control = pid(pressure)
                    if control is not None:
                        # motor.setPercentagePosition(control)
                        pass
                    _csv_writer.writerow([
                        time.time()-global_start_time, 
                        pressure,
                        daq.get_channel(FLOW_1),
                        daq.get_channel(FLOW_2),
                        daq.get_channel(FLOW_3),
                        daq.get_channel(FLOW_4)
                        ])

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
