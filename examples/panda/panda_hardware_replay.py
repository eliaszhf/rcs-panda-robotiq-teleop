"""Validate or replay one recorded Panda hardware episode.

The default mode is offline validation: it reads the Parquet data and session
metadata but imports no hardware driver and sends no command.  Real replay is
deliberately protected by several confirmations and reuses the collector's
home-path, workspace, joint-margin, and collision checks.
"""

from __future__ import annotations

import argparse
import json
import select
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from panda_hardware_keyboard_collect import (
    HOME_CONFIRM_PREFIX,
    HOME_PATH_SAMPLES,
    HOME_TOLERANCE_DEG,
    MAX_LOW_CONTROL_SECONDS,
    MAX_STEP_METERS,
    MIN_CONTROL_COMMAND_SUCCESS_RATE,
    MIN_POST_GRIPPER_RECORD_SECONDS,
    collision_report,
    create_hardware_env,
    home_path_within_workspace,
    joint_margin_violations,
    joints_have_margin,
    raw_terminal,
    tool_within_workspace,
    verify_franka_hand_defaults,
)


EXECUTE_CONFIRM_PREFIX = "REPLAY-REAL-PANDA"
FINAL_CONFIRM_PREFIX = "REPLAY-EPISODE"
START_TOLERANCE_METERS = 0.005
START_TOLERANCE_DEGREES = 3.0
MEDIAN_PERIOD_TOLERANCE = 0.2
MAX_PERIOD_FACTOR = 2.0
MAX_DUPLICATE_CAMERA_FRACTION = 0.02


@dataclass(frozen=True)
class ReplayStep:
    step: int
    timestamp: float
    action: dict[str, Any]
    recorded_tquat: np.ndarray
    recorded_target_tquat: np.ndarray
    recorded_joints: np.ndarray
    joint_collision: np.ndarray
    cartesian_collision: np.ndarray
    success: bool
    frame_timestamp: float | None
    camera_available: bool | None
    control_command_success_rate: float | None


@dataclass(frozen=True)
class Episode:
    uuid: str
    instruction: str
    steps: list[ReplayStep]


def quaternion_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(np.rad2deg(2.0 * np.arccos(np.clip(abs(np.dot(first, second)), 0.0, 1.0))))


def target_tquat_from_origin(origin_tquat: np.ndarray, relative_tquat: np.ndarray) -> np.ndarray:
    """Match RelativeActionSpace's CONFIGURED_ORIGIN Cartesian transform."""
    from rcs import common

    origin_tquat = np.asarray(origin_tquat, dtype=float)
    relative_tquat = np.asarray(relative_tquat, dtype=float)
    origin = common.Pose(translation=origin_tquat[:3], quaternion=origin_tquat[3:])
    offset = common.Pose(translation=relative_tquat[:3], quaternion=relative_tquat[3:])
    return np.concatenate((origin_tquat[:3] + relative_tquat[:3], (offset * origin).rotation_q()))


def load_episode(dataset: Path, requested_uuid: str | None = None) -> Episode:
    if not dataset.exists():
        raise ValueError(f"dataset does not exist: {dataset}")
    connection = duckdb.connect()
    try:
        uuids = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT uuid FROM read_parquet(?) ORDER BY uuid", [str(dataset)]
            ).fetchall()
        ]
        if requested_uuid is None:
            if len(uuids) != 1:
                raise ValueError(
                    f"dataset contains {len(uuids)} episodes; choose one with --uuid: {uuids}"
                )
            episode_uuid = uuids[0]
        elif requested_uuid not in uuids:
            raise ValueError(f"UUID {requested_uuid!r} not found; available UUIDs: {uuids}")
        else:
            episode_uuid = requested_uuid

        rows = connection.execute(
            "SELECT step, timestamp, env_action, obs.right.tquat, "
            "info.right.absolute_action, obs.right.joints, "
            "obs.right.robot_state.joint_collision, "
            "obs.right.robot_state.cartesian_collision, instruction, success, "
            "info.frame_timestamp, info.camera_available, "
            "obs.right.robot_state.control_command_success_rate "
            "FROM read_parquet(?) WHERE uuid = ? ORDER BY step",
            [str(dataset), episode_uuid],
        ).fetchall()
    except duckdb.Error as error:
        raise ValueError(f"cannot read Panda replay fields from {dataset}: {error}") from error
    finally:
        connection.close()

    if not rows:
        raise ValueError(f"episode {episode_uuid} contains no steps")
    instructions = {str(row[8]) for row in rows}
    if len(instructions) != 1:
        raise ValueError(f"episode has inconsistent instructions: {sorted(instructions)}")
    steps = [
        ReplayStep(
            step=int(row[0]),
            timestamp=float(row[1]),
            action=row[2],
            recorded_tquat=np.asarray(row[3], dtype=float),
            recorded_target_tquat=np.asarray(row[4], dtype=float),
            recorded_joints=np.asarray(row[5], dtype=float),
            joint_collision=np.asarray(row[6], dtype=float),
            cartesian_collision=np.asarray(row[7], dtype=float),
            success=bool(row[9]),
            frame_timestamp=None if row[10] is None else float(row[10]),
            camera_available=None if row[11] is None else bool(row[11]),
            control_command_success_rate=None if row[12] is None else float(row[12]),
        )
        for row in rows
    ]
    return Episode(episode_uuid, instructions.pop(), steps)


def load_session_metadata(dataset: Path) -> dict[str, Any]:
    metadata_path = dataset / "_session.json" if dataset.is_dir() else dataset.parent / "_session.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read required session metadata {metadata_path}: {error}") from error
    if metadata.get("relative_action_reference") != "configured_session_origin":
        raise ValueError("only configured_session_origin recordings can be replayed on hardware")
    return metadata


def validate_metadata(config: dict[str, Any], metadata: dict[str, Any]) -> None:
    scalar_pairs = {
        "robot_ip": "robot_ip",
        "gripper_serial": "gripper_serial",
        "frequency": "frequency",
    }
    array_pairs = {
        "workspace_min": "workspace_min",
        "workspace_max": "workspace_max",
        "tool_box_min": "tool_box_min_tcp_frame",
        "tool_box_max": "tool_box_max_tcp_frame",
        "home_joints_deg": "home_joints_deg",
        "tcp_translation": "flange_to_tcp_translation_m",
        "tcp_quaternion": "flange_to_tcp_quaternion_xyzw",
    }
    mismatches = [
        f"{config_key}: config={config.get(config_key)!r}, recording={metadata.get(metadata_key)!r}"
        for config_key, metadata_key in scalar_pairs.items()
        if config.get(config_key) != metadata.get(metadata_key)
    ]
    for config_key, metadata_key in array_pairs.items():
        configured = np.asarray(config.get(config_key), dtype=float)
        recorded = np.asarray(metadata.get(metadata_key), dtype=float)
        if configured.shape != recorded.shape or not np.allclose(configured, recorded, rtol=0.0, atol=1e-9):
            mismatches.append(
                f"{config_key}: config={configured.tolist()}, recording={recorded.tolist()}"
            )
    if mismatches:
        raise ValueError("config does not match the recording:\n  " + "\n  ".join(mismatches))


def validate_episode_timing(episode: Episode, metadata: dict[str, Any]) -> None:
    """Reject timing/communication defects that would corrupt action chunks."""
    frequency = float(metadata["frequency"])
    expected_period = 1.0 / frequency
    timestamps = np.asarray([item.timestamp for item in episode.steps], dtype=float)
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("episode contains non-finite host timestamps")
    intervals = np.diff(timestamps)
    if intervals.size == 0 or np.any(intervals <= 0.0):
        raise ValueError("episode timestamps must be strictly increasing")
    median_period = float(np.median(intervals))
    if not np.isclose(
        median_period,
        expected_period,
        rtol=MEDIAN_PERIOD_TOLERANCE,
        atol=0.002,
    ):
        raise ValueError(
            "recording rate does not match metadata: "
            f"expected {frequency:.2f} Hz ({expected_period * 1000:.1f} ms), "
            f"median interval is {median_period * 1000:.1f} ms"
        )
    max_interval = float(np.max(intervals))
    if max_interval > expected_period * MAX_PERIOD_FACTOR:
        raise ValueError(
            f"recording contains a {max_interval:.3f} s timing gap; "
            f"maximum allowed at {frequency:.2f} Hz is "
            f"{expected_period * MAX_PERIOD_FACTOR:.3f} s"
        )

    if metadata.get("cameras"):
        if any(item.camera_available is not True or item.frame_timestamp is None for item in episode.steps):
            raise ValueError("recording contains a missing camera frame")
        frame_timestamps = np.asarray(
            [item.frame_timestamp for item in episode.steps], dtype=float
        )
        duplicate_fraction = float(np.mean(np.diff(frame_timestamps) <= 0.0))
        if duplicate_fraction > MAX_DUPLICATE_CAMERA_FRACTION:
            raise ValueError(
                f"camera frame timestamps repeat in {duplicate_fraction:.1%} of transitions; "
                f"maximum allowed is {MAX_DUPLICATE_CAMERA_FRACTION:.1%}"
            )

    gripper_states = [int(item.action["right"]["gripper"][0]) for item in episode.steps]
    changes = [index for index in range(1, len(gripper_states)) if gripper_states[index] != gripper_states[index - 1]]
    if any(item.success for item in episode.steps) and changes:
        required_tail = float(
            metadata.get(
                "minimum_post_gripper_record_seconds",
                MIN_POST_GRIPPER_RECORD_SECONDS,
            )
        )
        tail = timestamps[-1] - timestamps[changes[-1]]
        if tail < required_tail:
            raise ValueError(
                f"successful episode records only {tail:.3f} s after its last gripper command; "
                f"at least {required_tail:.3f} s is required"
            )

    threshold = float(
        metadata.get(
            "minimum_control_command_success_rate",
            MIN_CONTROL_COMMAND_SUCCESS_RATE,
        )
    )
    max_low_seconds = float(
        metadata.get("maximum_low_control_seconds", MAX_LOW_CONTROL_SECONDS)
    )
    maximum_low_steps = max(1, int(np.ceil(frequency * max_low_seconds)))
    low_run = 0
    for item in episode.steps:
        rate = item.control_command_success_rate
        low_run = low_run + 1 if rate is not None and rate < threshold else 0
        if low_run >= maximum_low_steps:
            raise ValueError(
                "libfranka control command success rate remained below "
                f"{threshold:.2f} for at least {max_low_seconds:.1f} s"
            )


def validate_episode(
    episode: Episode, metadata: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected_steps = list(range(len(episode.steps)))
    actual_steps = [item.step for item in episode.steps]
    if actual_steps != expected_steps:
        raise ValueError("recorded step numbers are not contiguous from zero")
    validate_episode_timing(episode, metadata)

    workspace_min = np.asarray(metadata["workspace_min"], dtype=float)
    workspace_max = np.asarray(metadata["workspace_max"], dtype=float)
    tool_box_min = np.asarray(metadata["tool_box_min_tcp_frame"], dtype=float)
    tool_box_max = np.asarray(metadata["tool_box_max_tcp_frame"], dtype=float)
    max_step = float(metadata["step_meters"])
    if not np.isfinite(max_step) or not (0.0 < max_step <= MAX_STEP_METERS):
        raise ValueError(
            f"recorded step_meters must be > 0 and <= {MAX_STEP_METERS}, got {max_step!r}"
        )
    previous_offset = np.zeros(3, dtype=float)
    previous_quaternion = np.array([0.0, 0.0, 0.0, 1.0])

    for item in episode.steps:
        try:
            robot_action = item.action["right"]
            tquat = np.asarray(robot_action["tquat"], dtype=float)
            gripper = np.asarray(robot_action["gripper"], dtype=int)
        except (KeyError, TypeError) as error:
            raise ValueError(f"step {item.step} has an invalid action structure") from error
        arrays = (tquat, item.recorded_tquat, item.recorded_target_tquat, item.recorded_joints)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError(f"step {item.step} contains a non-finite value")
        if tquat.shape != (7,) or item.recorded_tquat.shape != (7,) or item.recorded_target_tquat.shape != (7,):
            raise ValueError(f"step {item.step} has an invalid TQuat shape")
        if item.recorded_joints.shape != (7,) or gripper.shape != (1,) or gripper[0] not in (0, 1):
            raise ValueError(f"step {item.step} has invalid joints or gripper data")
        if not np.isclose(np.linalg.norm(tquat[3:]), 1.0, atol=1e-5):
            raise ValueError(f"step {item.step} action quaternion is not normalized")
        translation_increment = float(np.linalg.norm(tquat[:3] - previous_offset))
        rotation_increment = quaternion_angle_degrees(tquat[3:], previous_quaternion)
        if translation_increment > max_step + 1e-8 or rotation_increment > 1.0 + 1e-6:
            raise ValueError(
                f"step {item.step} exceeds recorded control limits: "
                f"translation={translation_increment:.6f} m, rotation={rotation_increment:.3f} deg"
            )
        if not tool_within_workspace(
            item.recorded_target_tquat,
            tool_box_min,
            tool_box_max,
            workspace_min,
            workspace_max,
        ):
            raise ValueError(f"step {item.step} recorded target leaves the configured workspace")
        if np.any(item.joint_collision > 0.0) or np.any(item.cartesian_collision > 0.0):
            raise ValueError(f"step {item.step} contains a recorded collision signal")
        previous_offset = tquat[:3]
        previous_quaternion = tquat[3:]
    return workspace_min, workspace_max, tool_box_min, tool_box_max


def hardware_args(config: dict[str, Any]) -> argparse.Namespace:
    values = dict(config)
    values.update(
        cameras={},
        require_cameras=False,
        include_depth=False,
        camera_width=int(config.get("camera_width", 640)),
        camera_height=int(config.get("camera_height", 480)),
        camera_fps=int(config.get("camera_fps", 30)),
    )
    return argparse.Namespace(**values)


def replay_hardware(
    episode: Episode,
    config: dict[str, Any],
    bounds: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    robot_ip = str(config["robot_ip"])
    workspace_min, workspace_max, tool_box_min, tool_box_max = bounds
    args = hardware_args(config)
    env = None
    try:
        print("Initialization performs Franka error recovery and OPENS the Robotiq gripper.")
        if input(f"Type 'START {robot_ip}' to create the hardware environment: ").strip() != f"START {robot_ip}":
            raise SystemExit("Cancelled before opening either hardware driver.")
        env = create_hardware_env(robot_ip, str(config["gripper_serial"]), args)
        observation, _ = env.reset()
        robot = env.get_wrapper_attr("robot")["right"]
        verify_franka_hand_defaults(robot.get_state().robot_state)
        joint_limits = np.asarray(robot.get_config().joint_limits, dtype=float)
        joint_margin = np.deg2rad(float(config["joint_limit_margin_deg"]))
        start_joints = np.asarray(observation["right"]["joints"], dtype=float)
        home_joints = np.deg2rad(np.asarray(config["home_joints_deg"], dtype=float))
        if not joints_have_margin(home_joints, joint_limits, joint_margin):
            raise RuntimeError("configured home is inside the joint-limit margin")
        if collision_report(observation):
            raise RuntimeError(f"initial collision signal: {collision_report(observation)}")
        kinematics = robot.get_ik()
        if kinematics is None or not home_path_within_workspace(
            kinematics,
            start_joints,
            home_joints,
            robot.get_config().tcp_offset,
            tool_box_min,
            tool_box_max,
            workspace_min,
            workspace_max,
            samples=HOME_PATH_SAMPLES,
        ):
            raise RuntimeError("sampled path to home is unavailable or leaves the workspace")
        expected_home = f"{HOME_CONFIRM_PREFIX} {robot_ip}"
        if input(f"Type '{expected_home}' to move to the recorded Home: ").strip() != expected_home:
            raise SystemExit("Cancelled before home motion.")
        robot.move_home()
        observation, _ = env.reset()
        measured_joints = np.asarray(observation["right"]["joints"], dtype=float)
        if not np.allclose(measured_joints, home_joints, rtol=0.0, atol=np.deg2rad(HOME_TOLERANCE_DEG)):
            raise RuntimeError("home motion did not reach the configured joint pose")

        measured_start = np.asarray(observation["right"]["tquat"], dtype=float)
        recorded_start = episode.steps[0].recorded_tquat
        position_error = float(np.linalg.norm(measured_start[:3] - recorded_start[:3]))
        orientation_error = quaternion_angle_degrees(measured_start[3:], recorded_start[3:])
        if position_error > START_TOLERANCE_METERS or orientation_error > START_TOLERANCE_DEGREES:
            raise RuntimeError(
                "current Home does not match the recorded start: "
                f"position error={position_error:.6f} m, orientation error={orientation_error:.3f} deg"
            )
        if not tool_within_workspace(
            measured_start, tool_box_min, tool_box_max, workspace_min, workspace_max
        ):
            raise RuntimeError("measured Home tool envelope is outside the workspace")

        final_confirmation = f"{FINAL_CONFIRM_PREFIX} {episode.uuid}"
        print("Reset the scene to the recorded initial state; keep the emergency stop reachable.")
        if input(f"Type '{final_confirmation}' to begin motion: ").strip() != final_confirmation:
            raise SystemExit("Cancelled before replay motion.")

        period = 1.0 / float(config["frequency"])
        next_tick = time.monotonic()
        replay_origin = measured_start.copy()
        with raw_terminal():
            for index, item in enumerate(episode.steps):
                timeout = max(0.0, next_tick - time.monotonic())
                readable, _, _ = select.select([0], [], [], timeout)
                if readable and input_character() == "\x1b":
                    raise KeyboardInterrupt("operator aborted replay with Escape")
                commanded_target = target_tquat_from_origin(
                    replay_origin, np.asarray(item.action["right"]["tquat"], dtype=float)
                )
                if not tool_within_workspace(
                    commanded_target,
                    tool_box_min,
                    tool_box_max,
                    workspace_min,
                    workspace_max,
                ):
                    raise RuntimeError(
                        f"SAFETY STOP before step {item.step}: command would leave workspace"
                    )
                observation, _, terminated, truncated, info = env.step(item.action)
                measured = np.asarray(observation["right"]["tquat"], dtype=float)
                if not tool_within_workspace(
                    measured, tool_box_min, tool_box_max, workspace_min, workspace_max
                ):
                    raise RuntimeError(f"SAFETY STOP at step {item.step}: tool left workspace")
                if not joints_have_margin(observation["right"]["joints"], joint_limits, joint_margin):
                    violations = joint_margin_violations(
                        observation["right"]["joints"], joint_limits, joint_margin
                    )
                    raise RuntimeError(
                        f"SAFETY STOP at step {item.step}: " + "; ".join(violations)
                    )
                collisions = collision_report(observation)
                if collisions:
                    raise RuntimeError(f"SAFETY STOP at step {item.step}: {collisions}")
                if terminated or truncated:
                    raise RuntimeError(f"environment stopped at step {item.step}: {info}")
                if index % int(config["frequency"]) == 0:
                    print(f"replay {index}/{len(episode.steps)} TCP={measured[:3]}")
                next_tick = max(next_tick + period, time.monotonic())
        print(f"Replay complete: {len(episode.steps)} steps.")
    finally:
        if env is not None:
            env.close()
            print("Hardware environment closed.")


def input_character() -> str:
    import sys

    return sys.stdin.read(1).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--uuid", help="Required only when the dataset contains multiple episodes.")
    parser.add_argument("--execute", action="store_true", help="Actually connect to and move hardware.")
    parser.add_argument("--confirm", help=f"With --execute, must equal '{EXECUTE_CONFIRM_PREFIX} <robot-ip>'.")
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        metadata = load_session_metadata(args.dataset)
        validate_metadata(config, metadata)
        # Step size is an episode property. Preserve the recorded value so old
        # and fine-control recordings remain valid after the lab default changes.
        config["step"] = metadata["step_meters"]
        episode = load_episode(args.dataset, args.uuid)
        bounds = validate_episode(episode, metadata)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"Replay validation failed: {error}") from error

    duration = episode.steps[-1].timestamp - episode.steps[0].timestamp
    max_offset = np.max(
        np.abs([step.action["right"]["tquat"][:3] for step in episode.steps]), axis=0
    )
    print(f"VALID episode UUID: {episode.uuid}")
    print(f"instruction: {episode.instruction}")
    print(f"steps: {len(episode.steps)}; recorded duration: {duration:.2f} s")
    print(f"maximum absolute xyz offset from Home: {max_offset} m")
    print(f"recorded success: {any(step.success for step in episode.steps)}")

    if not args.execute:
        print("DRY RUN ONLY: no hardware driver was imported and no command was sent.")
        return
    expected = f"{EXECUTE_CONFIRM_PREFIX} {config['robot_ip']}"
    if args.confirm != expected:
        raise SystemExit(f"Hardware replay requires --confirm {expected!r}")
    replay_hardware(episode, config, bounds)


if __name__ == "__main__":
    main()
