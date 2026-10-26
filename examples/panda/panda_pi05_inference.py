"""Safety-gated Panda joint client for a remote RoboMME pi05 policy server.

The default mode queries one action chunk and never sends a policy motion
command, although hardware initialization does open the gripper. Passing
--execute bounds the checkpoint's absolute joint prediction and sends that
target directly to the Panda joint controller. Forward kinematics still audits
every proposed target against Cartesian step and workspace limits.
Execution starts only after explicit operator confirmations. Keep an operator
at the emergency stop.
"""

from __future__ import annotations

import argparse
import copy
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
MODEL_GRIPPER_OPEN_THRESHOLD = 0.0
HOLD_CONFIRM_PREFIX = "JOINT-HOLD"
EXPECTED_ACTION_HORIZON = 20
EXPECTED_ACTION_DIM = 8
MAX_FK_TRANSLATION_ERROR_METERS = 0.01
MAX_FK_ROTATION_ERROR_DEGREES = 3.0
PANDA_MJCF_ATTACHMENT_YAW_DEGREES = 135.0
HOLD_CHECK_MAX_JOINT_DEGREES = 0.05
HOLD_CHECK_MAX_TCP_METERS = 0.0005
HOLD_CHECK_MAX_ROTATION_DEGREES = 0.1
HOLD_CHECK_SECONDS = 1.0


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


def refresh_hardware_observation(
    observation: dict[str, Any], robot: Any, camera_set: Any, camera_name: str
) -> dict[str, Any]:
    """Refresh policy-critical state and image without sending an action."""

    frameset = camera_set.get_latest_frames()
    if frameset is None or camera_name not in frameset.frames:
        raise RuntimeError(f"No fresh frame available from camera {camera_name!r}")
    pose = robot.get_cartesian_position()
    fresh = dict(observation)
    fresh["right"] = dict(observation["right"])
    fresh["right"]["joints"] = np.asarray(robot.get_joint_position(), dtype=float)
    fresh["right"]["tquat"] = np.concatenate((pose.translation(), pose.rotation_q()))
    fresh["frames"] = dict(observation["frames"])
    fresh["frames"][camera_name] = copy.deepcopy(observation["frames"][camera_name])
    fresh["frames"][camera_name]["rgb"]["data"] = frameset.frames[
        camera_name
    ].camera.color.data
    return fresh


def wait_for_joint_settle(
    robot: Any,
    target_joints: np.ndarray,
    position_tolerance_radians: float,
    *,
    velocity_tolerance: float = 0.01,
    stable_samples: int = 5,
    timeout_seconds: float = 5.0,
) -> None:
    """Wait until a blocking move has actually stopped at its target."""

    deadline = time.monotonic() + timeout_seconds
    stable = 0
    while time.monotonic() < deadline:
        state = robot.get_state().robot_state
        joints = np.asarray(state.q, dtype=float)
        velocity = np.asarray(state.dq, dtype=float)
        if (
            np.max(np.abs(joints - target_joints)) <= position_tolerance_radians
            and np.max(np.abs(velocity)) <= velocity_tolerance
        ):
            stable += 1
            if stable >= stable_samples:
                return
        else:
            stable = 0
        time.sleep(0.05)
    raise RuntimeError("Robot did not settle at the requested joint target")


def wait_for_velocity_settle(
    robot: Any,
    *,
    velocity_tolerance: float = 0.01,
    stable_samples: int = 3,
    timeout_seconds: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    stable = 0
    while time.monotonic() < deadline:
        velocity = np.asarray(robot.get_state().robot_state.dq, dtype=float)
        if np.max(np.abs(velocity)) <= velocity_tolerance:
            stable += 1
            if stable >= stable_samples:
                return
        else:
            stable = 0
        time.sleep(0.05)
    raise RuntimeError("Robot did not settle after the policy command")


def run_configured_origin_hold_check(
    env: Any,
    robot: Any,
    gripper_state: int,
    frequency: float,
    *,
    duration_seconds: float = HOLD_CHECK_SECONDS,
    max_joint_radians: float = np.deg2rad(HOLD_CHECK_MAX_JOINT_DEGREES),
    max_tcp_meters: float = HOLD_CHECK_MAX_TCP_METERS,
    max_rotation_degrees: float = HOLD_CHECK_MAX_ROTATION_DEGREES,
    joint_target: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Continuously command a zero-motion target and reject physical drift."""

    if frequency <= 0.0 or duration_seconds <= 0.0:
        raise ValueError("hold-check frequency and duration must be positive")

    state_before = robot.get_state().robot_state
    start_joints = np.asarray(state_before.q, dtype=float)
    start_pose = robot.get_cartesian_position()
    start_tcp = np.concatenate((start_pose.translation(), start_pose.rotation_q()))
    if joint_target is None:
        hold_command = {
            "tquat": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        }
        hold_label = "CONFIGURED-ORIGIN"
    else:
        joint_target = np.asarray(joint_target, dtype=float)
        if joint_target.shape != (7,) or not np.all(np.isfinite(joint_target)):
            raise ValueError("joint hold target must contain seven finite values")
        hold_command = {"joints": joint_target.copy()}
        hold_label = "JOINT-TARGET"
    period = 1.0 / frequency
    ticks = max(1, int(np.ceil(duration_seconds * frequency)))
    joint_motion = 0.0
    tcp_motion = 0.0
    tcp_rotation = 0.0
    measured_joints = start_joints.copy()

    for _ in range(ticks):
        tick_started = time.monotonic()
        _, _, terminated, truncated, info = env.step(
            {
                "right": {
                    **hold_command,
                    "gripper": [int(gripper_state)],
                }
            }
        )
        if terminated or truncated:
            raise RuntimeError(f"Environment stopped during configured-origin hold: {info}")
        state_after = robot.get_state().robot_state
        measured_joints = np.asarray(state_after.q, dtype=float)
        measured_pose = robot.get_cartesian_position()
        measured_tcp = np.concatenate(
            (measured_pose.translation(), measured_pose.rotation_q())
        )
        joint_motion = max(
            joint_motion, float(np.max(np.abs(measured_joints - start_joints)))
        )
        tcp_motion = max(
            tcp_motion, float(np.linalg.norm(measured_tcp[:3] - start_tcp[:3]))
        )
        tcp_rotation = max(
            tcp_rotation,
            quaternion_angle_degrees(start_tcp[3:], measured_tcp[3:]),
        )
        if joint_motion > max_joint_radians:
            raise RuntimeError(
                "SAFETY STOP: configured-origin hold moved joints farther than allowed: "
                f"joint_motion={np.rad2deg(joint_motion):.3f} deg, "
                f"tcp_motion={tcp_motion * 1000.0:.2f} mm, "
                f"tcp_rotation={tcp_rotation:.3f} deg, "
                f"joint_delta_deg={np.rad2deg(measured_joints - start_joints)}, "
                f"tcp_delta_mm={(measured_tcp[:3] - start_tcp[:3]) * 1000.0}"
            )
        if tcp_motion > max_tcp_meters:
            raise RuntimeError(
                "SAFETY STOP: configured-origin hold translated farther than allowed: "
                f"joint_motion={np.rad2deg(joint_motion):.3f} deg, "
                f"tcp_motion={tcp_motion * 1000.0:.2f} mm, "
                f"tcp_rotation={tcp_rotation:.3f} deg, "
                f"joint_delta_deg={np.rad2deg(measured_joints - start_joints)}, "
                f"tcp_delta_mm={(measured_tcp[:3] - start_tcp[:3]) * 1000.0}"
            )
        if tcp_rotation > max_rotation_degrees:
            raise RuntimeError(
                "SAFETY STOP: configured-origin hold rotated farther than allowed: "
                f"joint_motion={np.rad2deg(joint_motion):.3f} deg, "
                f"tcp_motion={tcp_motion * 1000.0:.2f} mm, "
                f"tcp_rotation={tcp_rotation:.3f} deg, "
                f"joint_delta_deg={np.rad2deg(measured_joints - start_joints)}, "
                f"tcp_delta_mm={(measured_tcp[:3] - start_tcp[:3]) * 1000.0}"
            )
        remaining = period - (time.monotonic() - tick_started)
        if remaining > 0.0:
            time.sleep(remaining)

    print(
        f"{hold_label} HOLD CHECK: "
        f"max_joint_motion={np.rad2deg(joint_motion):.3f} deg, "
        f"max_tcp_motion={tcp_motion * 1000.0:.2f} mm, "
        f"max_tcp_rotation={tcp_rotation:.3f} deg, "
        f"start_joints_deg={np.rad2deg(start_joints)}, "
        f"measured_joints_deg={np.rad2deg(measured_joints)}"
    )
    return joint_motion, tcp_motion, tcp_rotation


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
) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    proposed = np.asarray(proposed, dtype=float)
    limits = np.asarray(joint_limits, dtype=float)
    if current.shape != (7,) or proposed.shape != (7,) or limits.shape != (2, 7):
        raise ValueError("Joint target validation requires current=(7,), proposed=(7,), limits=(2,7)")
    if not np.all(np.isfinite([current, proposed])):
        raise ValueError("Joint target contains NaN or infinity")
    delta = proposed - current
    safe_low = limits[0] + margin_radians
    safe_high = limits[1] - margin_radians
    # Preserve the seven-joint direction learned from coordinated
    # demonstrations. Per-joint clipping changes the relative joint motion and
    # can turn a Cartesian reach into an unintended bowing trajectory.
    max_abs_delta = float(np.max(np.abs(delta)))
    scale = 1.0 if max_abs_delta <= max_step_radians else max_step_radians / max_abs_delta
    target = current + delta * scale
    if np.any(target < safe_low) or np.any(target > safe_high):
        raise ValueError("Bounded policy target enters the configured joint-limit margin")
    return target


def referenced_tquat_action(reference_tcp: np.ndarray, target_tcp: np.ndarray) -> np.ndarray:
    """Convert an absolute TCP target to an RCS reference-relative action."""

    reference_tcp = np.asarray(reference_tcp, dtype=float)
    target_tcp = np.asarray(target_tcp, dtype=float)
    if reference_tcp.shape != (7,) or target_tcp.shape != (7,):
        raise ValueError("Referenced TCP conversion requires two seven-element poses")
    if not np.all(np.isfinite([reference_tcp, target_tcp])):
        raise ValueError("Referenced TCP conversion received NaN or infinity")
    from rcs import common

    origin_pose = common.Pose(
        translation=reference_tcp[:3], quaternion=reference_tcp[3:]
    )
    target_pose = common.Pose(translation=target_tcp[:3], quaternion=target_tcp[3:])
    rotation_offset = target_pose * origin_pose.inverse()
    return np.concatenate(
        (target_tcp[:3] - reference_tcp[:3], rotation_offset.rotation_q())
    )


def model_gripper_to_binary(value: float) -> int:
    if not np.isfinite(value):
        raise ValueError("Policy gripper action is not finite")
    # The pi05 dataset converter explicitly encodes gripper actions as signed
    # values: +1 is open and -1 is closed.
    return 1 if value >= MODEL_GRIPPER_OPEN_THRESHOLD else 0


def quaternion_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("Quaternion comparison requires two four-element arrays")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("Quaternion comparison requires nonzero quaternions")
    dot = abs(float(np.dot(first / first_norm, second / second_norm)))
    return float(np.rad2deg(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


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
                "enable_depth": False,
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
    joint_kp = np.asarray(config.get("joint_kp", robot_cfg.kp), dtype=float)
    joint_kd = np.asarray(config.get("joint_kd", robot_cfg.kd), dtype=float)
    if (
        joint_kp.shape != (7,)
        or joint_kd.shape != (7,)
        or not np.all(np.isfinite(joint_kp))
        or not np.all(np.isfinite(joint_kd))
        or np.any(joint_kp <= 0.0)
        or np.any(joint_kd <= 0.0)
    ):
        raise ValueError("joint_kp and joint_kd must contain seven positive finite values")
    robot_cfg.kp = joint_kp
    robot_cfg.kd = joint_kd
    osc_translation_stiffness = np.asarray(
        config.get("osc_translation_stiffness", [150.0, 150.0, 150.0]),
        dtype=float,
    )
    if (
        osc_translation_stiffness.shape != (3,)
        or not np.all(np.isfinite(osc_translation_stiffness))
        or np.any(osc_translation_stiffness <= 0.0)
        or np.any(osc_translation_stiffness > 500.0)
    ):
        raise ValueError(
            "osc_translation_stiffness must contain three finite values in (0, 500]"
        )
    robot_cfg.kp_p = osc_translation_stiffness
    gravity_mass = config.get(
        "joint_gravity_compensation_total_mass_kg",
        config.get("gravity_compensation_total_mass_kg"),
    )
    gravity_com = config.get("gravity_compensation_total_com_m")
    if (gravity_mass is None) != (gravity_com is None):
        raise ValueError(
            "gravity compensation requires both total mass and total center of mass"
        )
    if gravity_mass is not None:
        gravity_com_array = np.asarray(gravity_com, dtype=float)
        if (
            not np.isfinite(float(gravity_mass))
            or float(gravity_mass) <= 0.0
            or gravity_com_array.shape != (3,)
            or not np.all(np.isfinite(gravity_com_array))
        ):
            raise ValueError("invalid gravity compensation total mass or center of mass")
        robot_cfg.gravity_compensation_total_mass = float(gravity_mass)
        robot_cfg.gravity_compensation_total_com = gravity_com_array
    robot_cfg.speed_factor = float(config["speed_factor"])
    robot_cfg.policy_rate = int(config["frequency"])
    robot_cfg.ignore_realtime = False
    robot_cfg.async_control = True
    robot_cfg.blocking_move_on_start = True
    robot_cfg.attachment_site = "attachment"

    gripper_cfg = cfg.gripper_cfgs["right"]
    gripper_cfg.serial_device = str(config["gripper_device"])
    gripper_cfg.speed = float(config["gripper_speed"])
    gripper_cfg.force = float(config["gripper_force"])
    return creator.create_env(cfg)


def absolute_tcp_from_joint_target(
    kinematics: Any,
    target_joints: np.ndarray,
    tcp_offset: Any,
) -> np.ndarray:
    """Map a seven-joint checkpoint target to one absolute base-frame TCP pose."""

    target_joints = np.asarray(target_joints, dtype=float)
    if target_joints.shape != (7,) or not np.all(np.isfinite(target_joints)):
        raise ValueError("Joint-to-TCP conversion requires seven finite joint values")
    from rcs import common

    # Panda's MJCF "attachment" body is located at the physical flange but is
    # rotated +135 degrees about local Z. The hardware Cartesian API exposes
    # the physical flange frame instead. Remove that fixed model-frame rotation
    # before applying the configured flange-to-TCP transform.
    half_yaw = np.deg2rad(-PANDA_MJCF_ATTACHMENT_YAW_DEGREES) / 2.0
    attachment_to_flange = common.Pose(
        translation=np.zeros(3),
        quaternion=np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)]),
    )
    attachment_pose = kinematics.forward(target_joints)
    pose = attachment_pose * attachment_to_flange * tcp_offset
    tcp_tquat = np.concatenate((pose.translation(), pose.rotation_q()))
    if tcp_tquat.shape != (7,) or not np.all(np.isfinite(tcp_tquat)):
        raise ValueError("Panda forward kinematics returned an invalid TCP pose")
    quaternion_norm = float(np.linalg.norm(tcp_tquat[3:]))
    if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(
            f"Panda forward kinematics returned a non-unit quaternion: norm={quaternion_norm}"
        )
    return tcp_tquat


def validate_fk_matches_measured(
    fk_tcp: np.ndarray,
    measured_tcp: np.ndarray,
    max_translation_error: float = MAX_FK_TRANSLATION_ERROR_METERS,
    max_rotation_error_degrees: float = MAX_FK_ROTATION_ERROR_DEGREES,
) -> None:
    fk_tcp = np.asarray(fk_tcp, dtype=float)
    measured_tcp = np.asarray(measured_tcp, dtype=float)
    if fk_tcp.shape != (7,) or measured_tcp.shape != (7,):
        raise ValueError("FK validation requires two seven-element TCP poses")
    translation_error = float(np.linalg.norm(fk_tcp[:3] - measured_tcp[:3]))
    rotation_error = quaternion_angle_degrees(fk_tcp[3:], measured_tcp[3:])
    if translation_error > max_translation_error or rotation_error > max_rotation_error_degrees:
        raise RuntimeError(
            "FK TCP disagrees with measured Panda TCP: "
            f"translation={translation_error * 1000.0:.1f} mm, "
            f"rotation={rotation_error:.2f} deg"
        )


def validate_tcp_step(
    measured_tcp: np.ndarray,
    target_tcp: np.ndarray,
    max_translation_meters: float,
    max_rotation_degrees: float,
) -> tuple[float, float]:
    measured_tcp = np.asarray(measured_tcp, dtype=float)
    target_tcp = np.asarray(target_tcp, dtype=float)
    if measured_tcp.shape != (7,) or target_tcp.shape != (7,):
        raise ValueError("TCP step validation requires two seven-element poses")
    translation = float(np.linalg.norm(target_tcp[:3] - measured_tcp[:3]))
    rotation = quaternion_angle_degrees(target_tcp[3:], measured_tcp[3:])
    if translation > max_translation_meters or rotation > max_rotation_degrees:
        raise ValueError(
            "Policy TCP target exceeds the per-step Cartesian safety limit: "
            f"translation={translation * 1000.0:.1f} mm, rotation={rotation:.2f} deg"
        )
    return translation, rotation


def bound_joint_target_to_tcp_step(
    current_joints: np.ndarray,
    target_joints: np.ndarray,
    measured_tcp: np.ndarray,
    kinematics: Any,
    tcp_offset: Any,
    max_translation_meters: float,
    max_rotation_degrees: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Scale one coordinated joint step until its FK TCP step is admissible."""

    current_joints = np.asarray(current_joints, dtype=float)
    target_joints = np.asarray(target_joints, dtype=float)
    measured_tcp = np.asarray(measured_tcp, dtype=float)
    if current_joints.shape != (7,) or target_joints.shape != (7,):
        raise ValueError("Cartesian joint scaling requires two seven-joint vectors")
    if measured_tcp.shape != (7,):
        raise ValueError("Cartesian joint scaling requires a seven-element measured TCP pose")
    if max_translation_meters <= 0.0 or max_rotation_degrees <= 0.0:
        raise ValueError("Cartesian joint scaling limits must be positive")

    def evaluate(scale: float) -> tuple[np.ndarray, float, float]:
        joints = current_joints + scale * (target_joints - current_joints)
        tcp = absolute_tcp_from_joint_target(kinematics, joints, tcp_offset)
        translation = float(np.linalg.norm(tcp[:3] - measured_tcp[:3]))
        rotation = quaternion_angle_degrees(tcp[3:], measured_tcp[3:])
        return tcp, translation, rotation

    target_tcp, translation, rotation = evaluate(1.0)
    if translation <= max_translation_meters and rotation <= max_rotation_degrees:
        return target_joints, target_tcp, 1.0, translation, rotation

    _, zero_translation, zero_rotation = evaluate(0.0)
    if zero_translation > max_translation_meters or zero_rotation > max_rotation_degrees:
        raise RuntimeError(
            "Measured TCP disagrees with current-joint FK by more than the per-step limit"
        )

    low = 0.0
    high = 1.0
    for _ in range(32):
        middle = 0.5 * (low + high)
        _, middle_translation, middle_rotation = evaluate(middle)
        if (
            middle_translation <= max_translation_meters
            and middle_rotation <= max_rotation_degrees
        ):
            low = middle
        else:
            high = middle

    # Stay just inside the boundary despite floating-point/FK roundoff.
    scale = low * (1.0 - 1e-6)
    scaled_joints = current_joints + scale * (target_joints - current_joints)
    scaled_tcp, translation, rotation = evaluate(scale)
    validate_tcp_step(
        measured_tcp,
        scaled_tcp,
        max_translation_meters,
        max_rotation_degrees,
    )
    return scaled_joints, scaled_tcp, scale, translation, rotation


def validate_cartesian_outcome(
    start_tcp: np.ndarray,
    measured_tcp: np.ndarray,
    *,
    max_achieved_translation_meters: float,
    max_achieved_rotation_degrees: float,
) -> tuple[float, float]:
    achieved_translation = float(np.linalg.norm(measured_tcp[:3] - start_tcp[:3]))
    achieved_rotation = quaternion_angle_degrees(start_tcp[3:], measured_tcp[3:])
    if achieved_translation > max_achieved_translation_meters:
        raise RuntimeError(
            "SAFETY STOP: measured TCP translation exceeded the absolute step limit: "
            f"limit={max_achieved_translation_meters * 1000.0:.2f} mm, "
            f"measured={achieved_translation * 1000.0:.2f} mm"
        )
    if achieved_rotation > max_achieved_rotation_degrees:
        raise RuntimeError(
            "SAFETY STOP: measured TCP rotation exceeded the absolute step limit: "
            f"limit={max_achieved_rotation_degrees:.3f} deg, "
            f"measured={achieved_rotation:.3f} deg"
        )
    return achieved_translation, achieved_rotation


def target_tcp_is_safe(
    target_tcp: np.ndarray,
    config: dict[str, Any],
) -> bool:
    return tool_within_workspace(
        target_tcp,
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
    parser.add_argument("--hold-check-only", action="store_true")
    parser.add_argument(
        "--skip-hold-check",
        action="store_true",
        help="Skip the separate zero-motion joint-controller hold preflight.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum policy steps; use 0 to run until ESC or a safety stop.",
    )
    parser.add_argument(
        "--actions-per-query",
        type=int,
        default=5,
        help="Consume this many consecutive actions from each 20-step policy chunk.",
    )
    parser.add_argument("--max-joint-step-deg", type=float, default=0.5)
    parser.add_argument("--max-tcp-step-mm", type=float, default=5.0)
    parser.add_argument("--max-tcp-rotation-deg", type=float, default=1.0)
    parser.add_argument("--confirm", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    robot_ip = str(config["robot_ip"])
    expected = f"{EXECUTE_CONFIRM_PREFIX if args.execute else OBSERVE_CONFIRM_PREFIX} {robot_ip}"
    if args.confirm != expected:
        raise SystemExit(f"confirmation must be exactly: {expected}")
    if args.hold_check_only and not args.execute:
        raise SystemExit("--hold-check-only requires --execute")
    if args.skip_hold_check and not args.execute:
        raise SystemExit("--skip-hold-check requires --execute")
    if args.skip_hold_check and args.hold_check_only:
        raise SystemExit("--skip-hold-check and --hold-check-only are incompatible")
    if not (1 <= args.actions_per_query <= EXPECTED_ACTION_HORIZON):
        raise SystemExit(f"--actions-per-query must be between 1 and {EXPECTED_ACTION_HORIZON}")
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be non-negative (0 means unlimited)")
    if not (0.0 < args.max_joint_step_deg <= 20.0):
        raise SystemExit("require 0 < --max-joint-step-deg <= 20")
    if not (0.0 < args.max_tcp_step_mm <= 5.0):
        raise SystemExit("require 0 < --max-tcp-step-mm <= 5")
    if not (0.0 < args.max_tcp_rotation_deg <= 1.0):
        raise SystemExit("require 0 < --max-tcp-rotation-deg <= 1")
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
        camera_set = env.get_wrapper_attr("camera_set")
        verify_franka_hand_defaults(robot.get_state().robot_state)
        joint_limits = np.asarray(robot.get_config().joint_limits, dtype=float)
        joint_margin = np.deg2rad(float(config["joint_limit_margin_deg"]))
        kinematics = robot.get_ik()
        if kinematics is None:
            raise RuntimeError("Cannot validate policy targets without Panda kinematics")
        measured_tcp = np.asarray(observation["right"]["tquat"], dtype=float)
        current_fk_tcp = absolute_tcp_from_joint_target(
            kinematics,
            np.asarray(observation["right"]["joints"], dtype=float),
            robot.get_config().tcp_offset,
        )
        validate_fk_matches_measured(current_fk_tcp, measured_tcp)
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
            wait_for_joint_settle(
                robot,
                home_joints,
                np.deg2rad(HOME_TOLERANCE_DEG),
            )
            observation, _ = env.reset()
            observation = refresh_hardware_observation(
                observation, robot, camera_set, args.camera_name
            )
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

            if args.skip_hold_check:
                print(
                    "WARNING: configured-origin hold preflight skipped by operator; "
                    "the first joint-controller command will be the first bounded policy target."
                )
            else:
                hold_confirmation = f"{HOLD_CONFIRM_PREFIX} {robot_ip}"
                if input(
                    f"Type '{hold_confirmation}' to verify a zero-motion joint hold: "
                ).strip() != hold_confirmation:
                    raise SystemExit("Cancelled before joint hold check.")
                gripper_state = int(
                    float(
                        np.asarray(
                            observation["right"]["gripper"], dtype=float
                        ).reshape(-1)[0]
                    )
                    >= 0.5
                )
                run_configured_origin_hold_check(
                    env,
                    robot,
                    gripper_state,
                    float(config["frequency"]),
                    joint_target=measured_home,
                )
                observation = refresh_hardware_observation(
                    observation, robot, camera_set, args.camera_name
                )
                if args.hold_check_only:
                    print("HOLD CHECK ONLY: passed; no policy action was requested or sent.")
                    return

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
        )
        print(
            "JOINT AUDIT DEG: "
            f"raw_delta={np.rad2deg(actions[0, :7] - current)}, "
            f"bounded_delta={np.rad2deg(audited_target - current)}"
        )
        (
            audited_target,
            audited_tcp,
            audit_cartesian_scale,
            audit_translation,
            audit_rotation,
        ) = bound_joint_target_to_tcp_step(
            current,
            audited_target,
            np.asarray(observation["right"]["tquat"], dtype=float),
            kinematics,
            robot.get_config().tcp_offset,
            args.max_tcp_step_mm / 1000.0,
            args.max_tcp_rotation_deg,
        )
        print(
            "TCP AUDIT: "
            f"measured_xyz={np.asarray(observation['right']['tquat'], dtype=float)[:3]}, "
            f"target_xyz={audited_tcp[:3]}, "
            f"cartesian_scale={audit_cartesian_scale:.3f}, "
            f"translation={audit_translation * 1000.0:.2f} mm, "
            f"rotation={audit_rotation:.3f} deg"
        )
        if not target_tcp_is_safe(audited_tcp, config):
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
        low_control_ticks = 0
        completed_steps = 0
        last_command_time: float | None = None
        with raw_terminal():
            unlimited_steps = args.max_steps == 0
            while unlimited_steps or completed_steps < args.max_steps:
                if "\x1b" in read_pending_keys(0, 0.0):
                    print("ESC received; stopping policy execution.")
                    break
                if last_command_time is not None:
                    remaining = period - (time.monotonic() - last_command_time)
                    if remaining > 0:
                        time.sleep(remaining)
                observation = refresh_hardware_observation(
                    observation, robot, camera_set, args.camera_name
                )
                response = client.infer(
                    build_policy_observation(observation, args.instruction, args.camera_name)
                )
                actions = validate_action_chunk(response)
                for action_index, policy_action in enumerate(actions[: args.actions_per_query]):
                    if action_index > 0 and last_command_time is not None:
                        remaining = period - (time.monotonic() - last_command_time)
                        if remaining > 0:
                            time.sleep(remaining)
                        observation = refresh_hardware_observation(
                            observation, robot, camera_set, args.camera_name
                        )
                    current = np.asarray(observation["right"]["joints"], dtype=float)
                    start_tcp = np.asarray(
                        observation["right"]["tquat"], dtype=float
                    ).copy()
                    raw_joint_delta = policy_action[:7] - current
                    target = bounded_joint_target(
                        current,
                        policy_action[:7],
                        joint_limits,
                        joint_margin,
                        max_step,
                    )
                    (
                        target,
                        target_tcp,
                        cartesian_scale,
                        step_translation,
                        step_rotation,
                    ) = bound_joint_target_to_tcp_step(
                        current,
                        target,
                        start_tcp,
                        kinematics,
                        robot.get_config().tcp_offset,
                        args.max_tcp_step_mm / 1000.0,
                        args.max_tcp_rotation_deg,
                    )
                    if not target_tcp_is_safe(target_tcp, config):
                        raise RuntimeError("Policy target would move the tool envelope outside workspace")
                    raw_gripper = float(policy_action[7])
                    binary_gripper = model_gripper_to_binary(raw_gripper)
                    observation, _, terminated, truncated, info = env.step(
                        {
                            "right": {
                                "joints": target,
                                "gripper": [binary_gripper],
                            }
                        }
                    )
                    last_command_time = time.monotonic()
                    # The upstream joint controller is asynchronous. Let its interpolation
                    # window elapse, then require the arm to settle before
                    # auditing or closing a one-step run.
                    time.sleep(period)
                    wait_for_velocity_settle(
                        robot,
                        velocity_tolerance=0.01,
                        stable_samples=3,
                        timeout_seconds=max(1.0, 5.0 * period),
                    )
                    observation = refresh_hardware_observation(
                        observation, robot, camera_set, args.camera_name
                    )
                    measured_joints = np.asarray(
                        observation["right"]["joints"], dtype=float
                    )
                    measured_tcp = np.asarray(
                        observation["right"]["tquat"], dtype=float
                    )
                    print(
                        "COMMAND TRACE: "
                        f"start_joints_deg={np.rad2deg(current)}, "
                        f"target_joints_deg={np.rad2deg(target)}, "
                        "action_reference=absolute-joint-target, "
                        f"measured_joints_deg={np.rad2deg(measured_joints)}, "
                        f"start_xyz={start_tcp[:3]}, measured_xyz={measured_tcp[:3]}"
                    )
                    achieved_joint_max = float(
                        np.max(np.abs(measured_joints - current))
                    )
                    achieved_translation, achieved_rotation = validate_cartesian_outcome(
                        start_tcp,
                        measured_tcp,
                        max_achieved_translation_meters=args.max_tcp_step_mm / 1000.0,
                        max_achieved_rotation_degrees=args.max_tcp_rotation_deg,
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
                    print(
                        f"STEP AUDIT {completed_steps}: "
                        f"requested_translation={step_translation * 1000.0:.2f} mm, "
                        f"requested_rotation={step_rotation:.3f} deg, "
                        f"cartesian_scale={cartesian_scale:.3f}, "
                        f"raw_max_joint_delta={np.rad2deg(np.max(np.abs(raw_joint_delta))):.3f} deg, "
                        f"raw_gripper={raw_gripper:.4f}, commanded_gripper={binary_gripper}, "
                        f"achieved_joint_max={np.rad2deg(achieved_joint_max):.3f} deg, "
                        f"achieved_translation={achieved_translation * 1000.0:.2f} mm, "
                        f"achieved_rotation={achieved_rotation:.3f} deg, "
                        f"measured_xyz={measured_tcp[:3]}"
                    )
                    if not unlimited_steps and completed_steps >= args.max_steps:
                        break
                step_limit = "unlimited" if unlimited_steps else str(args.max_steps)
                print(f"executed {completed_steps}/{step_limit} policy steps")
    finally:
        if env is not None:
            env.close()
            print("Hardware environment closed.")
        if client is not None:
            client.close()
            print("Policy connection closed.")


if __name__ == "__main__":
    main()
