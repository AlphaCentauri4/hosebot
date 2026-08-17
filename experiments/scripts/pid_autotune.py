"""
pid_autotune.py

Automated PID tuning for the pressure-control rig (Arduino DAQ + AutoGROQS6 valve motor).

Two tuning strategies are provided and can be run independently or in sequence:

1. Relay (Astrom-Hagglund) autotuning
   - Safe, standard method for finding the ultimate gain/period of the closed loop
     without ever running the loop with untuned gains.
   - Toggles the valve position between two fixed levels around a setpoint and
     watches the resulting pressure oscillation.
   - Produces initial Kp/Ki/Kd estimates via three standard tuning rules
     (Ziegler-Nichols PID, Tyreus-Luyben, Pessen Integral).

2. Nelder-Mead refinement on real hardware
   - Starts from the relay-tuning estimate (or a manual guess) and runs real
     step-response experiments, scoring each parameter set with a cost function
     that penalizes tracking error, overshoot, AND control-signal chatter
     (rapid back-and-forth in the valve command -- this is what "bouncing"
     looks like from the PID's point of view).
   - Uses scipy.optimize.minimize(method="Nelder-Mead"), a derivative-free
     simplex optimizer well suited to noisy, expensive-to-evaluate (hardware)
     objective functions -- no gradient needed, and it tolerates a certain
     amount of trial-to-trial noise.

USAGE
-----
    # Full pipeline: relay test, then refine with real step tests
    python pid_autotune.py --mode both

    # Relay test only (just want Ku/Tu and rule-based gains, no hardware refinement)
    python pid_autotune.py --mode relay

    # Refine a manual guess without running the relay test
    python pid_autotune.py --mode optimize --kp 0.25 --ki 0.03 --kd 0.02

SAFETY
------
- --max-pv sets a hard pressure ceiling; any trial that crosses it immediately
  zeroes the valve command and is scored as a failure (very high cost) rather
  than crashing the script.
- Every stage prompts you before actuating the valve, so you can get hands
  clear / check the rig between the relay test and the optimization runs.
- Ctrl+C at any point falls through to `finally`, which zeros the valve and
  disables torque, same as your main control script.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from simple_pid import PID

from data_acquisition import DataAcquisition
from motor_control import AutoGROQS6

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# ---------------------------------------------------------------------
# Configuration (mirrors control.py)
# ---------------------------------------------------------------------

MOTOR_PORT = "COM8"
MOTOR_ID = 3
MOTOR_BAUDRATE = 57600
MAX_CURRENT = 150

DAQ_PORT = "COM7"
GOAL_PRESSURE = 1

OUTPUT_DIR = Path("runs")


# ---------------------------------------------------------------------
# Stage 1: relay autotune
# ---------------------------------------------------------------------

@dataclass
class RelayAutotuneResult:
    times: np.ndarray
    pvs: np.ndarray
    ku: float
    tu: float
    amplitude: float
    relay_amplitude: float
    gains: dict = field(default_factory=dict)


def relay_autotune(
    daq: DataAcquisition,
    motor: AutoGROQS6,
    *,
    channel: str = "pres_in",
    setpoint: float,
    relay_amplitude: float = 0.15,
    base_output: float = 0.4,
    hysteresis: float = 0.02,
    min_cycles: int = 5,
    max_time: float = 60.0,
    sample_dt: float = 0.02,
    output_limits: tuple[float, float] = (0.0, 1.0),
) -> RelayAutotuneResult:
    """Run a relay-feedback test and return Ku/Tu plus rule-based PID gains.

    The valve is switched between `base_output + relay_amplitude` and
    `base_output - relay_amplitude` every time the pressure crosses the
    setpoint (with `hysteresis` to avoid chattering on sensor noise). This
    forces the loop into a stable limit-cycle oscillation whose period (Tu)
    and the resulting pressure amplitude (a) let us back out the ultimate
    gain: Ku = 4*d / (pi*a), where d is the relay half-amplitude.
    """
    high = min(base_output + relay_amplitude, output_limits[1])
    low = max(base_output - relay_amplitude, output_limits[0])
    d = (high - low) / 2.0

    times: list[float] = []
    pvs: list[float] = []
    switch_times: list[float] = []

    state_high = True
    motor.setPercentagePosition(high)

    t0 = time.time()
    print("Relay autotune running... (Ctrl+C to abort)")
    while True:
        now = time.time()
        t = now - t0
        if t > max_time:
            motor.setPercentagePosition(0.0)
            raise TimeoutError(
                "Relay autotune did not complete enough cycles within max_time. "
                "Try a larger --relay-amplitude or --max-time."
            )

        pv = daq.get_channel(channel)
        if pv is None:
            time.sleep(sample_dt)
            continue

        times.append(t)
        pvs.append(pv)

        error = setpoint - pv
        if state_high and error < -hysteresis:
            state_high = False
            motor.setPercentagePosition(low)
            switch_times.append(now)
        elif not state_high and error > hysteresis:
            state_high = True
            motor.setPercentagePosition(high)
            switch_times.append(now)

        if len(switch_times) >= 2 * min_cycles + 1:
            break

        time.sleep(sample_dt)

    motor.setPercentagePosition(base_output)

    # Use only the last min_cycles full periods so the initial transient
    # (before the oscillation settles into a clean limit cycle) is excluded.
    usable_switches = switch_times[-(2 * min_cycles):]
    half_periods = np.diff(usable_switches)
    Tu = float(np.mean(half_periods) * 2)

    settle_start = usable_switches[0] - t0
    t_arr = np.array(times)
    pv_arr = np.array(pvs)
    window = pv_arr[t_arr >= settle_start]

    a = float((window.max() - window.min()) / 2.0)
    if a <= 1e-6:
        raise RuntimeError(
            "Oscillation amplitude was too small to measure reliably; "
            "increase --relay-amplitude and try again."
        )

    Ku = 4 * d / (np.pi * a)

    gains = {
        "ziegler_nichols_pid": {
            "Kp": 0.6 * Ku,
            "Ki": 1.2 * Ku / Tu,
            "Kd": 0.075 * Ku * Tu,
        },
        # Gentler than classic Z-N -- usually the right choice if the system
        # is prone to overshoot/bouncing, at the cost of slower settling.
        "tyreus_luyben": {
            "Kp": Ku / 3.2,
            "Ki": Ku / (3.2 * 2.2 * Tu),
            "Kd": Ku * Tu / (3.2 * 6.3),
        },
        # More aggressive than classic Z-N -- faster settling, more risk of
        # overshoot. Useful as an upper bound / starting point for the
        # optimizer if Tyreus-Luyben turns out too sluggish.
        "pessen_integral": {
            "Kp": 0.7 * Ku,
            "Ki": 1.75 * Ku / Tu,
            "Kd": 0.105 * Ku * Tu,
        },
    }

    return RelayAutotuneResult(
        times=t_arr, pvs=pv_arr, ku=Ku, tu=Tu, amplitude=a,
        relay_amplitude=d, gains=gains,
    )


# ---------------------------------------------------------------------
# Stage 2: real-hardware step response + Nelder-Mead refinement
# ---------------------------------------------------------------------

def run_step_response(
    daq: DataAcquisition,
    motor: AutoGROQS6,
    kp: float,
    ki: float,
    kd: float,
    *,
    setpoint_low: float,
    setpoint_high: float,
    hold_time: float = 3.0,
    step_time: float = 6.0,
    sample_dt: float = 0.02,
    output_limits: tuple[float, float] = (0.0, 1.0),
    max_pv: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Settle at setpoint_low, then step to setpoint_high and record the response."""
    pid = PID(kp, ki, kd, setpoint=setpoint_low)
    pid.sample_time = None  # we drive the timing loop ourselves
    pid.output_limits = output_limits

    t0 = time.time()
    while time.time() - t0 < hold_time:
        pv = daq.get_channel("pres_in")
        if pv is None:
            time.sleep(sample_dt)
            continue
        motor.setPercentagePosition(pid(pv))
        time.sleep(sample_dt)

    pid.setpoint = setpoint_high
    times, pvs, sps, outs = [], [], [], []
    t1 = time.time()
    while time.time() - t1 < step_time:
        t = time.time() - t1
        pv = daq.get_channel("pres_in")
        if pv is None:
            time.sleep(sample_dt)
            continue

        if max_pv is not None and pv > max_pv:
            motor.setPercentagePosition(0.0)
            raise RuntimeError(
                f"Pressure exceeded safety limit ({pv:.2f} > {max_pv:.2f}); trial aborted."
            )

        control = pid(pv)
        motor.setPercentagePosition(control)

        times.append(t)
        pvs.append(pv)
        sps.append(pid.setpoint)
        outs.append(control)
        time.sleep(sample_dt)

    motor.setPercentagePosition(0.0)
    return np.array(times), np.array(pvs), np.array(sps), np.array(outs)


def pid_cost(
    params: np.ndarray,
    *,
    daq: DataAcquisition,
    motor: AutoGROQS6,
    setpoint_low: float,
    setpoint_high: float,
    max_pv: float,
    log: list[dict],
) -> float:
    kp, ki, kd = params
    if kp < 0 or ki < 0 or kd < 0:
        return 1e6  # keep the simplex out of physically meaningless territory

    try:
        t, pv, sp, out = run_step_response(
            daq, motor, kp, ki, kd,
            setpoint_low=setpoint_low, setpoint_high=setpoint_high, max_pv=max_pv,
        )
    except RuntimeError as exc:
        print(f"  [trial failed: {exc}]")
        return 1e6

    if len(t) < 2:
        return 1e6

    error = sp - pv
    iae = float(np.trapz(np.abs(error), t))
    overshoot = float(max(0.0, pv.max() - setpoint_high))
    # Mean absolute step-to-step change in the control signal -- this is
    # exactly what "bouncing" looks like, so penalize it directly.
    chatter = float(np.mean(np.abs(np.diff(out))))

    cost = iae + 5.0 * overshoot + 2.0 * chatter
    log.append({
        "kp": kp, "ki": ki, "kd": kd, "cost": cost,
        "iae": iae, "overshoot": overshoot, "chatter": chatter,
    })
    print(
        f"  Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f} -> cost={cost:.4f} "
        f"(IAE={iae:.3f}, overshoot={overshoot:.3f}, chatter={chatter:.4f})"
    )
    return cost


def optimize_pid(
    daq: DataAcquisition,
    motor: AutoGROQS6,
    initial_guess: list[float],
    *,
    setpoint_low: float,
    setpoint_high: float,
    max_pv: float,
    max_iter: int = 25,
):
    log: list[dict] = []
    cost_fn = partial(
        pid_cost, daq=daq, motor=motor,
        setpoint_low=setpoint_low, setpoint_high=setpoint_high,
        max_pv=max_pv, log=log,
    )
    result = minimize(
        cost_fn,
        x0=np.array(initial_guess, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-2, "adaptive": True},
    )
    return result, log


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_relay_result(result: RelayAutotuneResult, out_dir: Path) -> None:
    if not HAVE_MPL:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.times, result.pvs, label="pressure")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pressure")
    ax.set_title(f"Relay autotune: Ku={result.ku:.3f}, Tu={result.tu:.3f}s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "relay_autotune.png", dpi=150)
    plt.close(fig)


def plot_optimization_log(log: list[dict], out_dir: Path) -> None:
    if not HAVE_MPL or not log:
        return
    costs = [entry["cost"] for entry in log]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(costs, marker="o")
    ax.set_xlabel("trial")
    ax.set_ylabel("cost (IAE + overshoot + chatter)")
    ax.set_title("Nelder-Mead refinement progress")
    fig.tight_layout()
    fig.savefig(out_dir / "optimization_progress.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Autotune the pressure-control PID loop.")
    parser.add_argument("--mode", choices=["relay", "optimize", "both"], default="both")
    parser.add_argument("--setpoint", type=float, default=GOAL_PRESSURE)
    parser.add_argument("--relay-amplitude", type=float, default=0.25)
    parser.add_argument("--base-output", type=float, default=0.4)
    parser.add_argument("--kp", type=float, default=None, help="initial Kp (needed if --mode optimize)")
    parser.add_argument("--ki", type=float, default=None, help="initial Ki (needed if --mode optimize)")
    parser.add_argument("--kd", type=float, default=None, help="initial Kd (needed if --mode optimize)")
    parser.add_argument(
        "--rule", choices=["ziegler_nichols_pid", "tyreus_luyben", "pessen_integral"],
        default="tyreus_luyben",
        help="which relay-tuning rule seeds the optimizer (default: gentler tyreus_luyben)",
    )
    parser.add_argument("--max-pv", type=float, default=3.5, help="hard safety ceiling on pres_in")
    parser.add_argument("--max-iter", type=int, default=25, help="Nelder-Mead iteration budget")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "pid_autotune_results.json"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    motor = AutoGROQS6(port=MOTOR_PORT, id=MOTOR_ID, baudrate=MOTOR_BAUDRATE, max_current=MAX_CURRENT)
    motor.connect()
    motor.autoCalibration()
    motor.setPercentagePosition(0)

    daq = DataAcquisition(
        port=DAQ_PORT,
        channel_names=["pres_in", "flow_in", "flow_left", "flow_right", "pres_left", "pres_right"],
        channel_rescale=[7, 200, 200, 200, 7, 7],
        output_directory=str(OUTPUT_DIR),
    )
    daq.connect()
    daq.calibrate()

    results: dict = {}

    try:
        if args.mode in ("relay", "both"):
            input("\nReady for relay autotune. Confirm the pre-valve is open, then press Enter.")
            relay_result = relay_autotune(
                daq, motor,
                setpoint=args.setpoint,
                relay_amplitude=args.relay_amplitude,
                base_output=args.base_output,
            )
            print(f"\nRelay test complete: Ku={relay_result.ku:.4f}, Tu={relay_result.tu:.4f}s")
            for rule, gains in relay_result.gains.items():
                print(f"  {rule}: Kp={gains['Kp']:.4f} Ki={gains['Ki']:.4f} Kd={gains['Kd']:.4f}")
            plot_relay_result(relay_result, OUTPUT_DIR)

            results["relay"] = {
                "ku": relay_result.ku, "tu": relay_result.tu, "gains": relay_result.gains,
            }
            chosen = relay_result.gains[args.rule]
            initial_guess = [chosen["Kp"], chosen["Ki"], chosen["Kd"]]
        else:
            if None in (args.kp, args.ki, args.kd):
                raise SystemExit("With --mode optimize you must pass --kp, --ki, and --kd.")
            initial_guess = [args.kp, args.ki, args.kd]

        if args.mode in ("optimize", "both"):
            input(
                "\nReady for Nelder-Mead refinement -- this runs several real step tests "
                f"(seed: Kp={initial_guess[0]:.4f} Ki={initial_guess[1]:.4f} Kd={initial_guess[2]:.4f}). "
                "Press Enter to start."
            )
            opt_result, log = optimize_pid(
                daq, motor, initial_guess,
                setpoint_low=args.setpoint * 0.2,
                setpoint_high=args.setpoint,
                max_pv=args.max_pv,
                max_iter=args.max_iter,
            )
            kp, ki, kd = opt_result.x
            print(
                f"\nOptimized gains: Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f} "
                f"(final cost={opt_result.fun:.4f}, {len(log)} trials)"
            )
            plot_optimization_log(log, OUTPUT_DIR)
            results["optimized"] = {
                "Kp": float(kp), "Ki": float(ki), "Kd": float(kd),
                "cost": float(opt_result.fun), "log": log,
            }

    finally:
        motor.setPercentagePosition(0)
        motor._disableTorque()
        motor.updateState()
        daq.stop()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()