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
from u20_camera import U20Camera
from basic_plotter import ExperimentData


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MOTOR_PORT = "COM8"
MOTOR_ID = 13
MOTOR_BAUDRATE = 57600

TORQUE_CONST = 0.352      # Nm/A, at stall
MAX_TORQUE = 16           # Nm
MAX_CURRENT = MAX_TORQUE / TORQUE_CONST
MAX_CURRENT = 150

DAQ_PORT = "COM7"   # change to e.g. "COM3" on Windows
PRESSURE_CHANNEL = "A0"             # whichever channel is the pressure sensor
FLOW_0 = "A1"
FLOW_1 = "A2"
FLOW_2 = "A3"
P_1 = "A4"
P_2 = "A5"

GOAL_PRESSURE = 1   # normalized fraction, in the same units DataAcquisition reports

# PID_KP, PID_KI, PID_KD = 0.5, 0.1, 0.05
# PID_KP, PID_KI, PID_KD = 0.5, 0.5, 0.1
PID_KP = 0.4
PID_KI = 0.05 #0.05
PID_KD = 0.02

PID_SAMPLE_TIME_S = 0.05


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

    motor.setPercentagePosition(0)

    daq = DataAcquisition(
        port="COM7",
        channel_names=["pres_in", "flow_in", "flow_left", "flow_right", "pres_left", "pres_right",],
        channel_rescale=[7, 200, 200, 200, 7, 7],
        output_directory="runs"
        )
    daq.connect()
    calibrations = daq.calibrate()
    daq.stop()

    while True:
        if motor.setPositionFinished():
            break
    
    print("Initialization complete.")
    input("Open pre valve.")

    pid = PID(PID_KP, PID_KI, PID_KD, setpoint=GOAL_PRESSURE)
    pid.sample_time = PID_SAMPLE_TIME_S
    pid.output_limits = (0, 1)   # setPercentagePosition only accepts [0, 1]

    output_directory = Path("runs")

    output_directory.mkdir(parents=True, exist_ok=True)
    # file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    camera = U20Camera(
        camera_index=1,
        output_folder="runs",
        width=1280,
        height=720,
        fps=30,
    )

    period = 30 # s
    set_delay = 0.05 # s
    no_divisions = round(period/set_delay)

    p_min = 0
    p_max = 3
    reps = 3

    # pressure_intervals = np.concatenate([
    #     np.arange(p_min, p_max, delay_interval), 
    #     np.arange(p_max, p_min - delay_interval, -delay_interval)
    #     ])
    
    pressure_intervals = np.concatenate([
        np.linspace(p_min, p_max, no_divisions),
        np.linspace(p_max, p_min, no_divisions)
    ])
    
    # global_start_time = time.time()
    try:
        while True:
            this_name = input("Run name: ")

            Path(f"runs/{this_name}").mkdir(parents=True, exist_ok=True)

            # _csv_path = output_directory / f"{this_name}.csv"
            # _csv_file = _csv_path.open("w", newline="", encoding="utf-8")
            # _csv_writer = csv.writer(_csv_file)
            # header = ["repetition", "time", "p_in", "flow_0", "flow_1", "flow_2", "p1", "p2"]
            # _csv_writer.writerow(header)

            daq.connect()
            daq.start(f"{this_name}/{this_name}")
            camera.output_folder = Path(f"runs/{this_name}")
            camera.start()
            camera.start_recording(f"{this_name}.mp4")
            camera.view()

            for rep in range(reps):
                # print("_________________Rep: ", rep)
                pid.reset()
                rep_start_time = time.time()

                while (time.time() - rep_start_time)<10:
                    pid.setpoint = pressure_intervals[0]
                    pressure = daq.get_channel("pres_in")
                    control = pid(pressure)
                    # print(pressure, pid.setpoint, pid.components, control)
                    if control is not None:
                        motor.setPercentagePosition(control)
                    time.sleep(0.025)


                for set_pressure in pressure_intervals:
                    pid.setpoint = set_pressure
                    #print("Set pressure: ", set_pressure)
                    start_time = time.time()
                    while (time.time()-start_time) < set_delay:
                        pressure = daq.get_channel("pres_in")
                        # if pressure is None:
                        #     # No sample received yet; skip this cycle.
                        #     time.sleep(PID_SAMPLE_TIME_S)
                        #     continue
                        
                        control = pid(pressure)
                        # print('pset - pexp = %.2f'%(set_pressure-pressure))
                        # print(pressure, pid.setpoint, pid.components, control)
                        if control is not None:
                            motor.setPercentagePosition(control)
                        time.sleep(0.025)
            
            rep_end_time = time.time()
            while (time.time() - rep_end_time)<10:
                pid.setpoint = pressure_intervals[-1]
                pressure = daq.get_channel("pres_in")
                control = pid(pressure)
                # print(pressure, pid.setpoint, pid.components, control)
                if control is not None:
                    motor.setPercentagePosition(control)
                time.sleep(0.025)

            camera.release()
            daq.stop()
            motor.setPercentagePosition(0)
            experiment = ExperimentData(
                expfilename=this_name,
                data_dir=f"runs/{this_name}",
                figure_dir=f"runs/{this_name}/figures",
                n_repetitions=reps,
                repetition=1,
            )

            experiment.plot_all(
                title="",
                show=True,
                save=True,
            )


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
        camera.release()


if __name__ == "__main__":
    main()