from dynamixel_easy_sdk import *
from simple_pid import PID
import json
import multiprocessing
import time

class AutoGROQS6:
    def __init__(self, port, id, baudrate, max_current):
        self.port = port
        self.id = id
        self.baudrate = baudrate
        self.max_current = max_current

    def connect(self):
        connector = Connector(self.port, self.baudrate)
        self.motor = connector.createMotor(self.id)

        self.motor.disableTorque()
        self.motor.setOperatingMode(OperatingMode.CURRENT_BASED_POSITION)

        self.motor.enableTorque()
        # print(self.motor._checkTorqueStatus(1))

    def autoCalibration(self):
        minPos = self._gotoExtreme(1, 0)
        time.sleep(2)
        maxPos = self._gotoExtreme(-1, 50000, True)

        print("maxPos", maxPos)
        
        self.max = maxPos + 4000
        self.min = minPos - 4000
        print(f"Max: {self.max}")
        print(f"Min: {self.min}")


        self._disableTorque()
        self.motor.setOperatingMode(OperatingMode.CURRENT_BASED_POSITION)
        self._enableTorque()
        self.setPercentagePosition(1)
        self.updateState()


    def manualCalibration(self):
        autoqs6._disableTorque()
        input("Set max.")
        self.max = self._getPresentPosition()
        print(f"Max: {self.max}")
        
        # input("Set min.")
        # self.min = self._getPresentPosition()
        # print(f"Min: {self.min}")

        self.min = self.max + 45000

        self.updateState()

    def setCalibration(self, maxPos, minPos):
        self.min = minPos
        self.max = maxPos

        self.updateState()

    def loadCalibration(self):
        with open("state.json") as f:
            state = json.load(f)
        
        diffPosition = self._getPresentPosition() - state["presentPosition"]
        print(state["maxPosition"] + diffPosition)
        print(state["minPosition"] + diffPosition)
        self.setCalibration(state["maxPosition"] + diffPosition, state["minPosition"] + diffPosition)
        self.updateState()

    def setPercentagePosition(self, percentage):

        # self.motor.setOperatingMode(OperatingMode.CURRENT_BASED_POSITION)
        if (percentage>1 or percentage<0):
            raise UnboundLocalError
        self._enableTorque()
        # actualMin = min(self.min, self.max)
        # actualMax = max(self.min, self.max)
        # position = actualMin + (actualMax-actualMin)*percentage
        position = self.min - (self.min - self.max)*percentage
        # print(position)
        self.motor.setGoalPosition(int(position))
        self.positionPercentage = percentage
        self.targetPosition = int(position)
        return self._getPresentPosition()

    def updateState(self):
        state = {
            "presentPosition": self.targetPosition,
            "maxPosition": self.max,
            "minPosition": self.min
        }
        with open("state.json", "w") as f:
            json.dump(state, f)

    def _gotoExtreme(self, direction, slowPos, delay=False):
        print(direction)
        self.motor.disableTorque()
        self.motor.setOperatingMode(OperatingMode.VELOCITY)
        self.motor.enableTorque()
        # self.motor.setGoalPosition(1048575*direction)
        self.motor.setGoalVelocity(150*direction)
        if delay:
            time.sleep(0.5)
        initialPos = self._getPresentPosition()
        while True:
            try:
                if abs(self._getPresentPosition() - initialPos) < slowPos and abs(self._getPresentPosition() - initialPos) > 2000:
                    print("diff", self._getPresentPosition() - initialPos)
                    self.motor.setGoalVelocity(400*direction)
                else:
                    self.motor.setGoalVelocity(150*direction)
                    
                # print(f"{self.motor.getPresentCurrent()}     {self.max_current}")
                chkXtrm = self._checkExtreme()
                if (chkXtrm):
                    print("DISABLED")
                    self.motor.disableTorque()
                    # print(self._getPresentPosition())
                    return self._getPresentPosition()
            except KeyboardInterrupt:
                print("KYBOARD")
                self.motor.setGoalVelocity(0)
                self.motor.disableTorque()
            except DxlRuntimeError:
                print("No status packets received.")
                pass
            except Exception as e:
                print("jgkerbjkgebkjgebkgje")
                self.motor.disableTorque()
                print(e)
                raise RuntimeError
        self.motor.disableTorque()
        self.motor.setOperatingMode(OperatingMode.CURRENT_BASED_POSITION)
        self.motor.enableTorque()

    def _checkExtreme(self):
        print(abs(self.motor.getPresentCurrent()), abs(self.motor.getPresentVelocity()) )
        test1 = abs(self.motor.getPresentCurrent()) > 60
        test2 = abs(self.motor.getPresentVelocity()) < 140
        return test1 and test2

    def _enableTorque(self):
        self.motor.enableTorque()

    def _disableTorque(self):
        self.motor.disableTorque()

    def _getPresentPosition(self):
        return self.motor.getPresentPosition()

# torque_const = 0.352 # 3.452            # at stall, Nm/A
# max_torque = 14                # N/m
# max_current = max_torque / torque_const
max_current = 60

autoqs6 = AutoGROQS6(
    port        = "COM5",
    id          = 3,
    baudrate    = 57600,
    max_current = max_current
    )

autoqs6.connect()
# autoqs6._gotoExtreme(-1, 0, True)
autoqs6.autoCalibration()
# autoqs6.manualCalibration()
# autoqs6.loadCalibration()
# autoqs6.setPercentagePosition(1)
# time.sleep(3)
# autoqs6.setPercentagePosition(0.5)