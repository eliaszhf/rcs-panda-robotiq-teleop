"""Safety-gated zero-motion Cartesian hold test for one real Panda.

The test never homes the arm and never connects to a gripper or camera.  It
starts the Cartesian impedance controller at the measured TCP pose, monitors
that pose for two seconds, and stops if drift exceeds one millimetre.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from panda_hardware_keyboard_collect import (
    joints_have_margin,
    tool_within_workspace,
    verify_franka_hand_defaults,
)
from panda_hardware_micro_motion import asserted_collisions, create_arm_only_env


HOLD_SECONDS = 2.0
MAX_DRIFT_METERS = 0.001
CONFIRM_PREFIX = "HOLD-CARTESIAN-2S"


def hold_action() -> dict:
    return {"tquat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    robot_ip = str(config["robot_ip"])
    expected = f"{CONFIRM_PREFIX} {robot_ip}"
    if args.confirm != expected:
        raise SystemExit(f"confirmation must be exactly: {expected}")

    workspace_min = np.asarray(config["workspace_min"], dtype=float)
    workspace_max = np.asarray(config["workspace_max"], dtype=float)
    tool_box_min = np.asarray(config["tool_box_min"], dtype=float)
    tool_box_max = np.asarray(config["tool_box_max"], dtype=float)
    joint_margin = np.deg2rad(float(config["joint_limit_margin_deg"]))

    env = None
    try:
        env = create_arm_only_env(config)
        observation, _ = env.reset()
        start = np.asarray(observation["tquat"], dtype=float)
        robot = env.get_wrapper_attr("robot")
        verify_franka_hand_defaults(robot.get_state().robot_state)
        joint_limits = np.asarray(robot.get_config().joint_limits, dtype=float)

        if not tool_within_workspace(
            start, tool_box_min, tool_box_max, workspace_min, workspace_max
        ):
            raise RuntimeError("initial tool envelope is outside the configured workspace")
        if not joints_have_margin(observation["joints"], joint_limits, joint_margin):
            raise RuntimeError("initial joints are inside the configured joint-limit margin")
        collisions = asserted_collisions(observation)
        if collisions:
            raise RuntimeError(f"initial collision signal: {collisions}")

        print(f"START TCP: {start[:3]}")
        observation, _, terminated, truncated, info = env.step(hold_action())
        print(f"HOLD TARGET: {np.asarray(info.get('absolute_action', []), dtype=float)[:3]}")
        if terminated or truncated:
            raise RuntimeError(f"environment stopped while engaging hold: {info}")

        deadline = time.monotonic() + HOLD_SECONDS
        max_drift = 0.0
        final = start
        while time.monotonic() < deadline:
            time.sleep(0.05)
            observation = env.get_wrapper_attr("get_robot_obs")()
            final = np.asarray(observation["tquat"], dtype=float)
            drift = float(np.linalg.norm(final[:3] - start[:3]))
            max_drift = max(max_drift, drift)
            if drift > MAX_DRIFT_METERS:
                raise RuntimeError(f"Cartesian hold drift exceeded 1 mm: {drift:.6f} m")
            collisions = asserted_collisions(observation)
            if collisions:
                raise RuntimeError(f"collision signal during hold: {collisions}")

        print(f"FINAL TCP: {final[:3]}; delta={final[:3] - start[:3]}")
        print(f"MAX DRIFT: {max_drift:.6f} m")
        print("Two-second Cartesian hold test complete.")
    finally:
        if env is not None:
            env.close()
            print("Hardware environment closed.")


if __name__ == "__main__":
    main()
