"""Safety-gated single-Panda client for a remote RoboMME pi05 policy server.

The default mode queries one action chunk and never sends a policy motion
command, although hardware initialization does open the gripper. Passing
--execute enables bounded joint control only after three explicit operator
confirmations. Keep an operator at the emergency stop at all times.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from panda_hardware_keyboard_collect import (
    HOME_CONFIRM_PREFIX,
    HOME_TOLERANCE_DEG,
    MAX_LOW_CONTROL_SECONDS,
    MIN_CONTROL_COMMAND_SUCCESS_RATE,
    collision_report,
    control_command_success_rate,
    home_path_within_workspace,
    joint_margin_violations,
    joints_have_margin,
    parse_camera_specs,
    preflight_cameras,
    raw_terminal,
    read_pending_keys,
    realtime_host_warnings,
    tool_within_workspace,
    verify_franka_hand_defaults,
)
from pi05_websocket_client import Pi05WebSocketClient


OBSERVE_CONFIRM_PREFIX = "OBSERVE-PI05-PANDA"
EXECUTE_CONFIRM_PREFIX = "EXECUTE-PI05-PANDA"
RUN_CONFIRM_PREFIX = "RUN-PI05"
EXPECTED_ACTION_HORIZON = 20
EXPECTED_ACTION_DIM = 8


def resize_rgb(image: np.ndarray, size: int = 256) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB image must have shape HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"RGB image must use uint8, got {image.dtype}")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Image preprocessing requires opencv-python") from error
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def build_policy_observation(
    observation: dict[str, Any], instruction: str, camera_name: str = "third_person"
) -> dict[str, Any]:
    try:
        image = resize_rgb(observation["frames"][camera_name]["rgb"]["data"])
        joints = np.asarray(observation["right"]["joints"], dtype=np.float32)
        gripper = np.asarray(observation["right"]["gripper"], dtype=np.float32)
    except KeyError as error:
        raise ValueError(f"Missing policy observation field: {error}") from error
    if joints.shape != (7,) or gripper.shape != (1,):
        raise ValueError(f"Expected joints=(7,) and gripper=(1,), got {joints.shape}, {gripper.shape}")
    state = np.concatenate((joints, gripper)).astype(np.float32, copy=False)
    return {
        "observation/image": image,
        "observation/wrist_image": np.zeros_like(image),
        "observation/state": state,
        "prompt": instruction.strip().lower(),
    }


def validate_action_chunk(response: dict[str, Any]) -> np.ndarray:
    if "actions" not in response:
        raise ValueError(f"Policy response has no 'actions' field: {sorted(response)}")
    actions = np.asarray(response["actions"], dtype=float)
    if actions.shape != (EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM):
        raise ValueError(
            f"Expected policy actions shape {(EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM)}, "
            f"got {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy actions contain NaN or infinity")
    return actions


def bounded_joint_target(
    current: np.ndarray,
    proposed: np.ndarray,
    joint_limits: np.ndarray,
    margin_radians: float,
    max_step_radians: float,
    max_model_delta_radians: float,
) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    proposed = np.asarray(proposed, dtype=float)
    limits = np.asarray(joint_limits, dtype=float)
    if current.shape != (7,) or proposed.shape != (7,) or limits.shape != (2, 7):
        raise ValueError("Joint target validation requires current=(7,), proposed=(7,), limits=(2,7)")
    if not np.all(np.isfinite([current, proposed])):
        raise ValueError("Joint target contains NaN or infinity")
    delta = proposed - current
    if np.max(np.abs(delta)) > max_model_delta_radians:
        raise ValueError(
            "Policy joint target is too far from measured state: "
            f"max delta={np.rad2deg(np.max(np.abs(delta))):.2f} deg"
        )
    safe_low = limits[0] + margin_radians
    safe_high = limits[1] - margin_radians
    target = current + np.clip(delta, -max_step_radians, max_step_radians)
    if np.any(target < safe_low) or np.any(target > safe_high):
        raise ValueError("Bounded policy target enters the configured joint-limit margin")
    return target


def model_gripper_to_binary(value: float) -> int:
    if not np.isfinite(value):
        raise ValueError("Policy gripper action is not finite")
    return 1 if value >= 0.0 else 0


def create_joint_hardware_env(config: dict[str, Any]):
    from rcs import common
    from rcs.envs.base import ControlMode, RelativeTo
    from rcs_panda.configs import DROIDEnv
    from rcs_panda.creators import HardwareCameraCreatorConfig

    creator = DROIDEnv()
    creator.robot_ip = str(config["robot_ip"])
    creator.gripper_serial_number = str(config["gripper_serial"])
    cfg = creator.config()
    cfg.control_mode = ControlMode.JOINTS
    cfg.relative_to = RelativeTo.NONE
    cfg.max_relative_movement = None
    cfg.wrapper_cfg.home_on_reset = False
    cfg.wrapper_cfg.binary_gripper = True
    cfg.wrapper_cfg.include_depth = False

    cameras = parse_camera_specs(config.get("cameras"))
    cfg.camera_cfgs = {
        "realsense": HardwareCameraCreatorConfig(
            camera_type_id="realsense",
            camera_cfgs={
                name: common.BaseCameraConfig(
                    identifier=serial,
                    resolution_width=int(config["camera_width"]),
                    resolution_height=int(config["camera_height"]),
                    frame_rate=int(config["camera_fps"]),
                )
                for name, serial in cameras.items()
            },
            kwargs={
                "enable_ir_emitter": False,
                "enable_ir": False,
                "enable_imu": False,
                "align_depth_to_color": False,
            },
        )
    }

    robot_cfg = cfg.robot_cfgs["right"]
    robot_cfg.tcp_offset = common.Pose(
        translation=np.asarray(config["tcp_translation"], dtype=float),
        quaternion=np.asarray(config["tcp_quaternion"], dtype=float),
    )
    robot_cfg.q_home = np.deg2rad(np.asarray(config["home_joints_deg"], dtype=float))
    robot_cfg.load_parameters = None
    robot_cfg.speed_factor = float(config["speed_factor"])
    robot_cfg.policy_rate = int(config["frequency"])
    robot_cfg.ignore_realtime = False
    robot_cfg.blocking_move_on_start = True
    robot_cfg.attachment_site = "attachment"

    gripper_cfg = cfg.gripper_cfgs["right"]
    gripper_cfg.serial_device = str(config["gripper_device"])
    gripper_cfg.speed = float(config["gripper_speed"])
    gripper_cfg.force = float(config["gripper_force"])
    return creator.create_env(cfg)


def target_tcp_is_safe(
    kinematics: Any,
    target_joints: np.ndarray,
    tcp_offset: Any,
    config: dict[str, Any],
) -> bool:
    pose = kinematics.forward(target_joints, tcp_offset)
    tcp_tquat = np.concatenate((pose.translation(), pose.rotation_q()))
    return tool_within_workspace(
        tcp_tquat,
        np.asarray(config["tool_box_min"], dtype=float),
        np.asarray(config["tool_box_max"], dtype=float),
        np.asarray(config["workspace_min"], dtype=float),
        np.asarray(config["workspace_max"], dtype=float),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--camera-name", default="third_person")
    parser.add_argument("--response-timeout", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--actions-per-query", type=int, default=1)
    parser.add_argument("--max-joint-step-deg", type=float, default=0.5)
    parser.add_argument("--max-model-delta-deg", type=float, default=5.0)
    parser.add_argument("--confirm", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    robot_ip = str(config["robot_ip"])
    expected = f"{EXECUTE_CONFIRM_PREFIX if args.execute else OBSERVE_CONFIRM_PREFIX} {robot_ip}"
    if args.confirm != expected:
        raise SystemExit(f"confirmation must be exactly: {expected}")
    if not (1 <= args.actions_per_query <= EXPECTED_ACTION_HORIZON):
        raise SystemExit(f"--actions-per-query must be between 1 and {EXPECTED_ACTION_HORIZON}")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    if not (0.0 < args.max_joint_step_deg <= args.max_model_delta_deg <= 20.0):
        raise SystemExit(
            "require 0 < --max-joint-step-deg <= --max-model-delta-deg <= 20"
        )
    if not (1 <= args.port <= 65535):
        raise SystemExit("--port must be between 1 and 65535")
    if args.response_timeout <= 0.0:
        raise SystemExit("--response-timeout must be positive")
    if not args.instruction.strip():
        raise SystemExit("--instruction must not be empty")

    cameras = parse_camera_specs(config.get("cameras"))
    if args.camera_name not in cameras:
        raise SystemExit(f"camera {args.camera_name!r} is not configured; available: {sorted(cameras)}")
    preflight_cameras(cameras)
    for warning in realtime_host_warnings():
        print(f"REALTIME WARNING: {warning}")

    env = None
    client = None
    try:
        client = Pi05WebSocketClient(
            args.host, args.port, response_timeout=args.response_timeout
        )
        print(f"Connected to pi05 policy at ws://{args.host}:{args.port}")
        print(f"Policy metadata: {client.metadata}")
        print("Initialization performs Franka error recovery and OPENS the Robotiq gripper.")
        if input(f"Type 'START {robot_ip}' to create the hardware environment: ").strip() != f"START {robot_ip}":
            raise SystemExit("Cancelled before opening either hardware driver.")

        env = create_joint_hardware_env(config)
        observation, _ = env.reset()
        robot = env.get_wrapper_attr("robot")["right"]
        verify_franka_hand_defaults(robot.get_state().robot_state)
        joint_limits = np.asarray(robot.get_config().joint_limits, dtype=float)
        joint_margin = np.deg2rad(float(config["joint_limit_margin_deg"]))
        kinematics = robot.get_ik()
        if kinematics is None:
            raise RuntimeError("Cannot validate policy targets without Panda kinematics")
        if collision_report(observation):
            raise RuntimeError(f"Initial collision signal: {collision_report(observation)}")
        if not joints_have_margin(observation["right"]["joints"], joint_limits, joint_margin):
            raise RuntimeError("Initial joints are inside the configured joint-limit margin")
        if not tool_within_workspace(
            np.asarray(observation["right"]["tquat"], dtype=float),
            np.asarray(config["tool_box_min"], dtype=float),
            np.asarray(config["tool_box_max"], dtype=float),
            np.asarray(config["workspace_min"], dtype=float),
            np.asarray(config["workspace_max"], dtype=float),
        ):
            raise RuntimeError("Initial tool envelope is outside the configured workspace")

        if args.execute:
            home_joints = np.deg2rad(np.asarray(config["home_joints_deg"], dtype=float))
            if not home_path_within_workspace(
                kinematics,
                observation["right"]["joints"],
                home_joints,
                robot.get_config().tcp_offset,
                np.asarray(config["tool_box_min"], dtype=float),
                np.asarray(config["tool_box_max"], dtype=float),
                np.asarray(config["workspace_min"], dtype=float),
                np.asarray(config["workspace_max"], dtype=float),
            ):
                raise RuntimeError("Sampled path to home would leave the configured workspace")
            home_confirmation = f"{HOME_CONFIRM_PREFIX} {robot_ip}"
            if input(f"Type '{home_confirmation}' to move to Home: ").strip() != home_confirmation:
                raise SystemExit("Cancelled before home motion.")
            robot.move_home()
            observation, _ = env.reset()
            measured_home = np.asarray(observation["right"]["joints"], dtype=float)
            if not np.allclose(
                measured_home,
                home_joints,
                rtol=0.0,
                atol=np.deg2rad(HOME_TOLERANCE_DEG),
            ):
                raise RuntimeError(
                    "Home motion did not reach the configured pose within "
                    f"{HOME_TOLERANCE_DEG} deg"
                )
            if collision_report(observation):
                raise RuntimeError(
                    f"Collision signal asserted after Home: {collision_report(observation)}"
                )
            if not joints_have_margin(measured_home, joint_limits, joint_margin):
                raise RuntimeError("Home pose is inside the configured joint-limit margin")

        client.reset()
        started = time.monotonic()
        response = client.infer(
            build_policy_observation(observation, args.instruction, args.camera_name)
        )
        actions = validate_action_chunk(response)
        latency = time.monotonic() - started
        current = np.asarray(observation["right"]["joints"], dtype=float)
        max_first_delta = float(np.max(np.abs(actions[0, :7] - current)))
        print(
            f"POLICY AUDIT: shape={actions.shape}, latency={latency:.3f}s, "
            f"first max joint delta={np.rad2deg(max_first_delta):.2f} deg, "
            f"first gripper={actions[0, 7]:.3f}"
        )
        audited_target = bounded_joint_target(
            current,
            actions[0, :7],
            joint_limits,
            joint_margin,
            np.deg2rad(args.max_joint_step_deg),
            np.deg2rad(args.max_model_delta_deg),
        )
        if not target_tcp_is_safe(
            kinematics, audited_target, robot.get_config().tcp_offset, config
        ):
            raise RuntimeError("First bounded policy target would leave the configured workspace")
        if not args.execute:
            print(
                "OBSERVE ONLY: no policy motion command was sent; "
                "hardware initialization did open the gripper."
            )
            return

        run_confirmation = f"{RUN_CONFIRM_PREFIX} {robot_ip}"
        if input(f"Type '{run_confirmation}' to execute bounded policy actions: ").strip() != run_confirmation:
            raise SystemExit("Cancelled before policy execution.")

        period = 1.0 / float(config["frequency"])
        max_step = np.deg2rad(args.max_joint_step_deg)
        max_model_delta = np.deg2rad(args.max_model_delta_deg)
        low_control_ticks = 0
        completed_steps = 0
        with raw_terminal():
            while completed_steps < args.max_steps:
                if "\x1b" in read_pending_keys(0, 0.0):
                    print("ESC received; stopping policy execution.")
                    break
                started = time.monotonic()
                response = client.infer(
                    build_policy_observation(observation, args.instruction, args.camera_name)
                )
                actions = validate_action_chunk(response)
                for policy_action in actions[: args.actions_per_query]:
                    current = np.asarray(observation["right"]["joints"], dtype=float)
                    target = bounded_joint_target(
                        current,
                        policy_action[:7],
                        joint_limits,
                        joint_margin,
                        max_step,
                        max_model_delta,
                    )
                    if not target_tcp_is_safe(
                        kinematics, target, robot.get_config().tcp_offset, config
                    ):
                        raise RuntimeError("Policy target would move the tool envelope outside workspace")
                    observation, _, terminated, truncated, info = env.step(
                        {
                            "right": {
                                "joints": target,
                                "gripper": [model_gripper_to_binary(float(policy_action[7]))],
                            }
                        }
                    )
                    collisions = collision_report(observation)
                    if collisions:
                        raise RuntimeError(f"SAFETY STOP: collision signal asserted: {collisions}")
                    violations = joint_margin_violations(
                        observation["right"]["joints"], joint_limits, joint_margin
                    )
                    if violations:
                        raise RuntimeError("SAFETY STOP: " + "; ".join(violations))
                    if not tool_within_workspace(
                        np.asarray(observation["right"]["tquat"], dtype=float),
                        np.asarray(config["tool_box_min"], dtype=float),
                        np.asarray(config["tool_box_max"], dtype=float),
                        np.asarray(config["workspace_min"], dtype=float),
                        np.asarray(config["workspace_max"], dtype=float),
                    ):
                        raise RuntimeError(
                            "SAFETY STOP: measured tool envelope left the workspace"
                        )
                    success_rate = control_command_success_rate(observation)
                    if success_rate is not None and success_rate < MIN_CONTROL_COMMAND_SUCCESS_RATE:
                        low_control_ticks += 1
                    else:
                        low_control_ticks = 0
                    if low_control_ticks >= max(
                        1, int(float(config["frequency"]) * MAX_LOW_CONTROL_SECONDS)
                    ):
                        raise RuntimeError("SAFETY STOP: sustained low libfranka command success rate")
                    if terminated or truncated:
                        raise RuntimeError(f"Environment stopped: {info}")
                    completed_steps += 1
                    if completed_steps >= args.max_steps:
                        break
                    remaining = period - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
                    started = time.monotonic()
                print(f"executed {completed_steps}/{args.max_steps} policy steps")
    finally:
        if env is not None:
            env.close()
            print("Hardware environment closed.")
        if client is not None:
            client.close()
            print("Policy connection closed.")


if __name__ == "__main__":
    main()
