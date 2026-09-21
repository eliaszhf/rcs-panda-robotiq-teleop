"""Safety-gated 2 mm up-and-return test for one real Panda.

This script never homes the arm and does not connect to a gripper or camera.
It starts at the measured pose, moves +2 mm along Panda base-frame z, and then
moves -2 mm to return to the measured starting height.
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


DISTANCE_METERS = 0.002
WARMUP_SECONDS = 3.0
SETTLE_SECONDS = 3.0
CONFIRM_PREFIX = "MOVE-2MM-AND-RETURN"


def movement_actions(distance: float = DISTANCE_METERS) -> tuple[dict, dict]:
    if not np.isclose(distance, DISTANCE_METERS):
        raise ValueError(f"micro-motion distance must remain exactly {DISTANCE_METERS} m")
    identity = [0.0, 0.0, 0.0, 1.0]
    return (
        {"tquat": [0.0, 0.0, distance, *identity]},
        # With CONFIGURED_ORIGIN this is the recorded reset pose, even if the
        # first target was not tracked accurately. Never subtract 2 mm from
        # the measured pose, because that can compound a tracking error.
        {"tquat": [0.0, 0.0, 0.0, *identity]},
    )


def asserted_collisions(observation: dict) -> list[str]:
    state = observation.get("robot_state", {})
    asserted: list[str] = []
    for key in ("joint_collision", "cartesian_collision"):
        value = np.asarray(state.get(key, []), dtype=float)
        if value.size and np.any(value > 0.0):
            asserted.append(f"{key}={value.tolist()}")
    return asserted


def create_arm_only_env(config: dict):
    from rcs import common
    from rcs.envs.base import RelativeTo
    from rcs_panda.configs import DefaultPandaHardwareEnv

    creator = DefaultPandaHardwareEnv()
    creator.ip = config["robot_ip"]
    cfg = creator.config()
    cfg.gripper_cfg = None
    cfg.camera_cfgs = None
    cfg.wrapper_cfg.home_on_reset = False
    cfg.relative_to = RelativeTo.CONFIGURED_ORIGIN
    cfg.max_relative_movement = (DISTANCE_METERS, np.deg2rad(1.0))

    robot_cfg = cfg.robot_cfg
    robot_cfg.async_control = True
    # Wait for the measured starting pose to settle before accepting the
    # Cartesian increment.
    robot_cfg.blocking_move_on_start = True
    robot_cfg.ignore_realtime = False
    robot_cfg.policy_rate = int(config["frequency"])
    robot_cfg.speed_factor = float(config["speed_factor"])
    robot_cfg.tcp_offset = common.Pose(
        translation=np.asarray(config["tcp_translation"], dtype=float),
        quaternion=np.asarray(config["tcp_quaternion"], dtype=float),
    )
    robot_cfg.attachment_site = "attachment"

    # Preserve the shared Desk "Franka Hand" preset; do not call setLoad().
    robot_cfg.load_parameters = None
    return creator.create_env(cfg)


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
    up_action, return_action = movement_actions()

    env = None
    try:
        env = create_arm_only_env(config)
        observation, _ = env.reset()
        start = np.asarray(observation["tquat"], dtype=float)
        robot = env.get_wrapper_attr("robot")
        verify_franka_hand_defaults(robot.get_state().robot_state)
        joint_limits = np.asarray(robot.get_config().joint_limits, dtype=float)

        if not joints_have_margin(observation["joints"], joint_limits, joint_margin):
            raise RuntimeError("initial joints are inside the configured joint-limit margin")
        collisions = asserted_collisions(observation)
        if collisions:
            raise RuntimeError(f"initial collision signal: {collisions}")
        for dz in (0.0, DISTANCE_METERS):
            candidate = start.copy()
            candidate[2] += dz
            if not tool_within_workspace(candidate, tool_box_min, tool_box_max, workspace_min, workspace_max):
                raise RuntimeError(f"tool envelope would leave workspace at dz={dz}")

        print(f"RESET TCP: {start[:3]}")
        # The asynchronous OSC starts on the first action. Start it with the
        # configured-origin target, let the repository controller's startup
        # transient settle, and only then define the reference for direction.
        observation, _, terminated, truncated, info = env.step(return_action)
        if terminated or truncated:
            raise RuntimeError(f"environment stopped during OSC warm-up: {info}")
        time.sleep(WARMUP_SECONDS)
        start = np.asarray(env.get_wrapper_attr("get_robot_obs")()["tquat"], dtype=float)
        print(f"STABILIZED START TCP: {start[:3]}")
        if not tool_within_workspace(
            start, tool_box_min, tool_box_max, workspace_min, workspace_max
        ):
            raise RuntimeError("tool envelope left workspace during OSC warm-up")

        observation, _, terminated, truncated, info = env.step(up_action)
        print(f"UP TARGET: {np.asarray(info.get('absolute_action', []), dtype=float)[:3]}")
        if terminated or truncated:
            raise RuntimeError(f"environment stopped during +2 mm move: {info}")
        # The repository controller is asynchronous: env.step() publishes the
        # target and returns before the arm necessarily reaches it. Allow a
        # conservative settling window before measuring the achieved pose.
        time.sleep(SETTLE_SECONDS)
        raised = np.asarray(env.get_wrapper_attr("get_robot_obs")()["tquat"], dtype=float)
        print(f"RAISED TCP: {raised[:3]}; delta={raised[:3] - start[:3]}")

        observation, _, terminated, truncated, info = env.step(return_action)
        print(f"RETURN TARGET: {np.asarray(info.get('absolute_action', []), dtype=float)[:3]}")
        if terminated or truncated:
            raise RuntimeError(f"environment stopped during return move: {info}")
        time.sleep(SETTLE_SECONDS)
        final = np.asarray(env.get_wrapper_attr("get_robot_obs")()["tquat"], dtype=float)
        print(f"FINAL TCP: {final[:3]}; delta_from_start={final[:3] - start[:3]}")
        if np.linalg.norm(final[:3] - start[:3]) > 0.0015:
            raise RuntimeError("return pose differs from start by more than 1.5 mm")
        print("2 mm up-and-return test complete.")
    finally:
        if env is not None:
            env.close()
            print("Hardware environment closed.")


if __name__ == "__main__":
    main()
