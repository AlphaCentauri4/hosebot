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

GOAL_PRESSURE = 1   # normalized fraction, in the same units DataAcquisition reports

PID_KP, PID_KI, PID_KD = -1, -0.5, -0.05
PID_SAMPLE_TIME_S = 0.01


def main() -> None:
    motor = AutoGROQS6(
        port=MOTOR_PORT,
        id=MOTOR_ID,
        baudrate=MOTOR_BAUDRATE,
        max_current=MAX_CURRENT,
    )
    motor.connect()
    motor.loadCalibration()
    motor.setPercentagePosition(1)
    while True:
        if motor.setPositionFinished():
            break



    daq = DataAcquisition(port=DAQ_PORT)
    input("Close pre valve. Open post valve.")
    daq.connect()
    calibrations = daq.calibrate()
    daq.start()
    input("Open pre valve.")


    # motor.autoCalibration()

    pid = PID(PID_KP, PID_KI, PID_KD, setpoint=GOAL_PRESSURE)
    pid.sample_time = PID_SAMPLE_TIME_S
    pid.output_limits = (0, 1)   # setPercentagePosition only accepts [0, 1]

    pressure_levels = [4, 5, 6]

    motor.setPercentagePosition(1)

    output_directory = Path("data/exp1")

    output_directory.mkdir(parents=True, exist_ok=True)
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _csv_path = output_directory / f"{file_timestamp}.csv"
    _csv_file = _csv_path.open("w", newline="", encoding="utf-8")
    _csv_writer = csv.writer(_csv_file)
    header = ["pressure_level", "max_pressure", "min_pressure"]    
    _csv_writer.writerow(header)

    try:
        for level in pressure_levels:
            # input("Close post valve.")
            input(f"Adjust pressure to: {level} bars")
            while True:
                wallPressure = daq.get_channel(PRESSURE_CHANNEL)*7
                print(f"Current pressure is: {wallPressure} bars")
                confirm = input("Type 'y' to confirm. Press 'enter' to redo.")
                if confirm=="y":
                    # input("Open post valve.")
                    break

            print("Please wait.")
            motor.setPercentagePosition(0)
            while True:
                if motor.setPositionFinished():
                    time.sleep(2)
                    break
            maxPresure = daq.get_channel(PRESSURE_CHANNEL)*7

            motor.setPercentagePosition(1)
            while True:
                if motor.setPositionFinished():
                    time.sleep(2)
                    break
            minPresure = daq.get_channel(PRESSURE_CHANNEL)*7
            _csv_writer = csv.writer(_csv_file)
                    
            _csv_writer.writerow([wallPressure, maxPresure, minPresure])
            _csv_file.flush()

            
            
            
            # if pressure is None:
            #     # No sample received yet; skip this cycle.
            #     time.sleep(PID_SAMPLE_TIME_S)
            #     continue



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

    except DxlRuntimeError:
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
