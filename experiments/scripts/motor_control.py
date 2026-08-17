from dynamixel_easy_sdk import *
from simple_pid import PID
import json
import multiprocessing
import time

# Sentinel returned by _safe_call when every retry has failed. A plain
# `None` would be ambiguous, since some SDK calls can legitimately
# return None on success.
_COMM_FAILED = object()


class AutoGROQS6:
    def __init__(self, port, id, baudrate, max_current):
        self.port = port
        self.id = id
        self.baudrate = baudrate
        self.max_current = max_current
        self._lastKnownPosition = None

    # ------------------------------------------------------------------
    # Communication helper
    # ------------------------------------------------------------------

    def _safe_call(self, func, *args, retries=5, retry_delay=0.01, raise_on_fail=False, **kwargs):
        """
        Call a Dynamixel SDK function, retrying on DxlRuntimeError
        (e.g. SDK_COMM_RX_CORRUPT). These errors are almost always a
        transient serial glitch — a corrupted or dropped status packet —
        and clear up on the next attempt, so we retry a few times before
        giving up.

        Returns the function's result on success, or the _COMM_FAILED
        sentinel if all retries are exhausted (unless raise_on_fail=True,
        in which case the last exception is re-raised).
        """
        last_exc = None
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except DxlRuntimeError as e:
                last_exc = e
                if attempt < retries - 1:
                    time.sleep(retry_delay)

        name = getattr(func, "__name__", func)
        print(f"[AutoGROQS6] Comm failure after {retries} attempts on {name}: {last_exc}")
        if raise_on_fail:
            raise last_exc
        return _COMM_FAILED

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def connect(self):
        connector = Connector(self.port, self.baudrate)
        self.motor = connector.createMotor(self.id)

        self._disableTorque()
        self._safe_call(self.motor.setOperatingMode, OperatingMode.CURRENT_BASED_POSITION)
        self._enableTorque()
        # print(self.motor._checkTorqueStatus(1))

    def autoCalibration(self):
        maxPos = self._gotoExtreme(-1, 0, True)

        self.max = maxPos + 4000
        self.min = maxPos + 55000
        print(f"Max: {self.max}")
        print(f"Min: {self.min}")

        self._disableTorque()
        self._safe_call(self.motor.setOperatingMode, OperatingMode.CURRENT_BASED_POSITION)
        self._enableTorque()
        self.targetPosition = self.max
        self.updateState()

    def autoCalibrationSlow(self):
        minPos = self._gotoExtreme(1, 0)
        time.sleep(2)
        maxPos = self._gotoExtreme(-1, 50000, True)

        self.max = maxPos + 4000
        self.min = minPos - 4000

        self._disableTorque()
        self._safe_call(self.motor.setOperatingMode, OperatingMode.CURRENT_BASED_POSITION)
        self._enableTorque()
        self.targetPosition = self.max
        self.updateState()

    def manualCalibration(self):
        self._disableTorque()
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
        self.targetPosition = state["presentPosition"]
        self.setCalibration(state["maxPosition"] + diffPosition, state["minPosition"] + diffPosition)

    def setPercentagePosition(self, percentage):
        if percentage > 1 or percentage < 0:
            raise UnboundLocalError

        position = self.min - (self.min - self.max) * percentage

        result = self._safe_call(self.motor.setGoalPosition, int(position))
        if result is _COMM_FAILED:
            print("No status packet received.")
            # We can't be sure the command landed, so trust the motor's
            # actual reported position instead of the one we tried to set.
            actual = self._getPresentPosition()
            if actual is not None:
                self.targetPosition = actual
        else:
            self.targetPosition = int(position)
            self.positionPercentage = percentage

        self.updateState()

    def setPositionFinished(self):
        current = self._getPresentPosition()
        if current is None:
            return False
        return abs(self.targetPosition - current) < 11

    def updateState(self):
        state = {
            "presentPosition": self.targetPosition,
            "maxPosition": self.max,
            "minPosition": self.min
        }
        with open("state.json", "w") as f:
            json.dump(state, f)

    def _gotoExtreme(self, direction, slowPos, delay=False):
        self._disableTorque()
        self._safe_call(self.motor.setOperatingMode, OperatingMode.VELOCITY)
        self._enableTorque()
        self._safe_call(self.motor.setGoalVelocity, 150 * direction)
        if delay:
            time.sleep(0.5)

        # This read is critical to the whole routine, so retry hard for it
        # rather than silently proceeding with a bad initial position.
        initialPos = self._getPresentPosition(retries=20)
        if initialPos is None:
            raise RuntimeError("Could not read initial motor position - check the connection.")

        while True:
            try:
                currentPos = self._getPresentPosition()
                if currentPos is None:
                    # Transient read failure, just try again next loop.
                    continue

                if abs(currentPos - initialPos) < slowPos and abs(currentPos - initialPos) > 2000:
                    self._safe_call(self.motor.setGoalVelocity, 400 * direction)
                else:
                    self._safe_call(self.motor.setGoalVelocity, 150 * direction)

                if self._checkExtreme():
                    self._disableTorque()
                    return self._getPresentPosition()
            except KeyboardInterrupt:
                self._safe_call(self.motor.setGoalVelocity, 0)
                self._disableTorque()
            except DxlRuntimeError:
                # Shouldn't normally get here since motor calls above go
                # through _safe_call, but stay defensive.
                pass
            except Exception as e:
                self._disableTorque()
                print(e)
                raise RuntimeError

    def _checkExtreme(self):
        current = self._safe_call(self.motor.getPresentCurrent)
        velocity = self._safe_call(self.motor.getPresentVelocity)
        if current is _COMM_FAILED or velocity is _COMM_FAILED:
            # Can't confirm we're at the extreme this cycle; try again next loop.
            return False
        return abs(current) > 60 and abs(velocity) < 140

    def _enableTorque(self):
        self._safe_call(self.motor.enableTorque)

    def _disableTorque(self):
        self._safe_call(self.motor.disableTorque)

    def _getPresentPosition(self, retries=5, retry_delay=0.01):
        position = self._safe_call(self.motor.getPresentPosition, retries=retries, retry_delay=retry_delay)
        if position is _COMM_FAILED:
            if self._lastKnownPosition is not None:
                print("[AutoGROQS6] Using last known position after repeated comm failure.")
                return self._lastKnownPosition
            return None
        self._lastKnownPosition = position
        return position