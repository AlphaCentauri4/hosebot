"""
Overarching controller: reads live pressure from the Arduino via DataAcquisition
and drives the valve motor via AutoGROQS6 through a PID loop.
"""

import time

from simple_pid import PID

from data_acquisition import DataAcquisition
from motor_control2 import AutoGROQS6
import dynamixel_easy_sdk

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
    daq = DataAcquisition(port=DAQ_PORT)
    daq.connect()
    calibrations = daq.calibrate()
    daq.start()

    motor = AutoGROQS6(
        port=MOTOR_PORT,
        id=MOTOR_ID,
        baudrate=MOTOR_BAUDRATE,
        max_current=MAX_CURRENT,
    )
    motor.connect()
    motor.autoCalibration()

    pid = PID(PID_KP, PID_KI, PID_KD, setpoint=GOAL_PRESSURE)
    pid.sample_time = PID_SAMPLE_TIME_S
    pid.output_limits = (0, 1)   # setPercentagePosition only accepts [0, 1]

    try:
        while True:
            # pressure = (daq.get_channel(PRESSURE_CHANNEL) - calibrations["A0"]) / ((1024 - calibrations["A0"])*7)
            pressure = daq.get_channel(PRESSURE_CHANNEL)*7
            if pressure is None:
                # No sample received yet; skip this cycle.
                time.sleep(PID_SAMPLE_TIME_S)
                continue
                
            control = pid(pressure)
            print(pressure, pid.components, control)
            if control is not None:
                motor.setPercentagePosition(control)

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
