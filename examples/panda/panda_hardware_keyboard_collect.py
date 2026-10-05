"""Safety-gated keyboard collection for one real Panda + Robotiq 2F-85.

This is event-driven teleoperation sampled at a fixed rate.  Before collection,
it moves to a configured joint-space home pose after a separate operator
confirmation. Hardware use still requires an operator at the emergency stop
and a correctly configured Desk.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import ipaddress
import json
import os
import select
import sys
import termios
import time
import tty
from collections.abc import Iterator
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


MAX_STEP_METERS = 0.005
MAX_SPEED_FACTOR = 0.1
MAX_INITIAL_GRIPPER_SPEED = 50.0
MAX_INITIAL_GRIPPER_FORCE = 50.0
MIN_POST_GRIPPER_RECORD_SECONDS = 1.0
MIN_CONTROL_COMMAND_SUCCESS_RATE = 0.9
MAX_LOW_CONTROL_SECONDS = 1.0
RECORD_BATCH_SECONDS = 5
CONFIRM_PREFIX = "ENABLE-REAL-PANDA"
HOME_CONFIRM_PREFIX = "MOVE-HOME"
HOME_PATH_SAMPLES = 101
HOME_TOLERANCE_DEG = 1.0
DESK_END_EFFECTOR_PRESET = "Franka Hand"
FRANKA_HAND_MASS_KG = 0.73
FRANKA_HAND_COM_M = np.array([-0.01, 0.0, 0.03])
FRANKA_HAND_INERTIA_KG_M2 = np.diag([0.001, 0.0025, 0.0017])
FRANKA_HAND_F_T_EE = np.array(
    [
        [0.7071, 0.7071, 0.0, 0.0],
        [-0.7071, 0.7071, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1034],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
KEY_DELTAS = {
    "w": (1.0, 0.0, 0.0),
    "s": (-1.0, 0.0, 0.0),
    "a": (0.0, 1.0, 0.0),
    "d": (0.0, -1.0, 0.0),
    "r": (0.0, 0.0, 1.0),
    "f": (0.0, 0.0, -1.0),
}


def delta_for_key(key: str, step_meters: float = MAX_STEP_METERS) -> np.ndarray | None:
    unit_delta = KEY_DELTAS.get(key.lower())
    return None if unit_delta is None else np.asarray(unit_delta, dtype=float) * step_meters


def make_action(delta: np.ndarray, gripper_state: int) -> dict:
    return {
        "right": {
            "tquat": [*np.asarray(delta, dtype=float), 0.0, 0.0, 0.0, 1.0],
            "gripper": [int(gripper_state)],
        }
    }


def accumulated_offset(current_offset: np.ndarray, key: str, step_meters: float) -> np.ndarray:
    """Return a persistent session-origin offset after one keyboard event."""
    offset = np.asarray(current_offset, dtype=float).copy()
    delta = delta_for_key(key, step_meters)
    return offset if delta is None else offset + delta


def realtime_host_warnings() -> list[str]:
    """Return actionable host warnings without changing system settings."""
    governors = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"):
        try:
            governors.add(path.read_text(encoding="utf-8").strip())
        except OSError:
            continue
    warnings = []
    if governors and governors != {"performance"}:
        warnings.append(
            "CPU governors are " + ", ".join(sorted(governors))
            + "; use the performance governor for libfranka control"
        )
    return warnings


def read_pending_keys(fd: int, timeout: float) -> str:
    """Read all currently buffered terminal input after waiting at most timeout."""
    readable, _, _ = select.select([fd], [], [], max(0.0, timeout))
    if not readable:
        return ""
    return os.read(fd, 4096).decode(errors="ignore").lower()


def control_command_success_rate(observation: dict) -> float | None:
    value = observation.get("right", {}).get("robot_state", {}).get(
        "control_command_success_rate"
    )
    if value is None:
        return None
    rate = float(value)
    return rate if np.isfinite(rate) else None


def workspace_allows(
    current_tcp: np.ndarray,
    delta: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> bool:
    target = np.asarray(current_tcp, dtype=float) + np.asarray(delta, dtype=float)
    return bool(np.all(target >= workspace_min) and np.all(target <= workspace_max))


def box_corners(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Return the eight corners of an axis-aligned box."""
    return np.asarray(
        [
            [x, y, z]
            for x in (box_min[0], box_max[0])
            for y in (box_min[1], box_max[1])
            for z in (box_min[2], box_max[2])
        ],
        dtype=float,
    )


def tool_corners_in_base(
    tcp_tquat: np.ndarray,
    tool_box_min: np.ndarray,
    tool_box_max: np.ndarray,
) -> np.ndarray:
    """Transform a conservative tool box from TCP coordinates to Panda base."""
    from rcs import common

    tcp_tquat = np.asarray(tcp_tquat, dtype=float)
    if tcp_tquat.shape != (7,) or not np.all(np.isfinite(tcp_tquat)):
        raise ValueError(f"TCP TQuat must be seven finite values, got {tcp_tquat}")
    pose = common.Pose(translation=tcp_tquat[:3], quaternion=tcp_tquat[3:])
    return box_corners(tool_box_min, tool_box_max) @ pose.rotation_m().T + pose.translation()


def tool_within_workspace(
    tcp_tquat: np.ndarray,
    tool_box_min: np.ndarray,
    tool_box_max: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> bool:
    corners = tool_corners_in_base(tcp_tquat, tool_box_min, tool_box_max)
    return bool(np.all(corners >= workspace_min) and np.all(corners <= workspace_max))


def joints_have_margin(q: np.ndarray, limits: np.ndarray, margin_radians: float) -> bool:
    q = np.asarray(q, dtype=float)
    limits = np.asarray(limits, dtype=float)
    return bool(
        q.shape == (7,)
        and limits.shape == (2, 7)
        and np.all(np.isfinite(q))
        and np.all(q >= limits[0] + margin_radians)
        and np.all(q <= limits[1] - margin_radians)
    )


def joint_margin_violations(
    q: np.ndarray, limits: np.ndarray, margin_radians: float
) -> list[str]:
    """Describe joints outside the configured safe interval (one-based indices)."""
    q = np.asarray(q, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if q.shape != (7,) or limits.shape != (2, 7):
        return [f"invalid joint/limit shapes: q={q.shape}, limits={limits.shape}"]
    safe_low = limits[0] + margin_radians
    safe_high = limits[1] - margin_radians
    violations = []
    for index, (measured, low, high) in enumerate(zip(q, safe_low, safe_high), start=1):
        if not np.isfinite(measured) or measured < low or measured > high:
            violations.append(
                f"J{index}={np.rad2deg(measured):.2f} deg "
                f"(safe {np.rad2deg(low):.2f}..{np.rad2deg(high):.2f} deg)"
            )
    return violations


def home_path_within_workspace(
    kinematics: Any,
    start_joints: np.ndarray,
    home_joints: np.ndarray,
    tcp_offset: Any,
    tool_box_min: np.ndarray,
    tool_box_max: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    samples: int = HOME_PATH_SAMPLES,
) -> bool:
    """Check the sampled joint interpolation used as a home-path preflight."""
    start_joints = np.asarray(start_joints, dtype=float)
    home_joints = np.asarray(home_joints, dtype=float)
    if start_joints.shape != (7,) or home_joints.shape != (7,):
        raise ValueError("home path requires two seven-joint poses")
    if samples < 2:
        raise ValueError("home path requires at least two samples")
    for alpha in np.linspace(0.0, 1.0, samples):
        pose = kinematics.forward((1.0 - alpha) * start_joints + alpha * home_joints, tcp_offset)
        tcp_tquat = np.concatenate((pose.translation(), pose.rotation_q()))
        if not tool_within_workspace(
            tcp_tquat, tool_box_min, tool_box_max, workspace_min, workspace_max
        ):
            return False
    return True


def collision_report(observation: dict) -> list[str]:
    """Return collision fields asserted by libfranka in an RCS observation."""
    robot_state = observation.get("right", {}).get("robot_state", {})
    asserted = []
    for key in ("joint_collision", "cartesian_collision"):
        value = np.asarray(robot_state.get(key, []), dtype=float)
        if value.size and np.any(value > 0.0):
            asserted.append(f"{key}={value.tolist()}")
    return asserted


def parse_camera_specs(specs: Mapping[str, str] | Sequence[str] | None) -> dict[str, str]:
    """Normalize JSON mappings or repeated CLI ``NAME=SERIAL`` camera specs."""
    if specs is None:
        return {}
    if isinstance(specs, Mapping):
        cameras = {str(name).strip(): str(serial).strip() for name, serial in specs.items()}
    else:
        cameras = {}
        for spec in specs:
            if "=" not in spec:
                raise ValueError(f"camera must use NAME=SERIAL syntax, got {spec!r}")
            name, serial = spec.split("=", 1)
            name, serial = name.strip(), serial.strip()
            if name in cameras:
                raise ValueError(f"duplicate camera name: {name}")
            cameras[name] = serial
    if any(not name or not serial for name, serial in cameras.items()):
        raise ValueError("camera names and serial numbers must not be empty")
    return cameras


def missing_camera_serials(cameras: Mapping[str, str], connected_serials: set[str]) -> dict[str, str]:
    """Return configured cameras that are not present in a RealSense enumeration."""
    normalized_connected = {serial.lower().lstrip("0") for serial in connected_serials}
    return {
        name: serial
        for name, serial in cameras.items()
        if serial.lower().lstrip("0") not in normalized_connected
    }


def verify_franka_hand_defaults(robot_state: Any) -> None:
    """Refuse motion unless Desk still exposes the shared Franka Hand preset."""
    actual = {
        "mass": float(robot_state.m_ee),
        "com": np.asarray(robot_state.F_x_Cee, dtype=float),
        "inertia": np.asarray(robot_state.I_ee, dtype=float).reshape(3, 3, order="F"),
        "transform": np.asarray(robot_state.F_T_EE, dtype=float).reshape(4, 4, order="F"),
        "load_mass": float(robot_state.m_load),
    }
    expected = {
        "mass": FRANKA_HAND_MASS_KG,
        "com": FRANKA_HAND_COM_M,
        "inertia": FRANKA_HAND_INERTIA_KG_M2,
        "transform": FRANKA_HAND_F_T_EE,
        "load_mass": 0.0,
    }
    tolerances = {
        "mass": 5e-4,
        "com": 5e-4,
        "inertia": 5e-5,
        "transform": 5e-4,
        "load_mass": 5e-4,
    }
    mismatches = [
        name
        for name in expected
        if not np.allclose(actual[name], expected[name], rtol=0.0, atol=tolerances[name])
    ]
    if mismatches:
        details = "; ".join(
            f"{name}: actual={np.asarray(actual[name]).tolist()} expected={np.asarray(expected[name]).tolist()}"
            for name in mismatches
        )
        raise RuntimeError(
            f"Desk end-effector does not match '{DESK_END_EFFECTOR_PRESET}'; refusing motion. {details}"
        )


def preflight_cameras(cameras: Mapping[str, str]) -> None:
    """Verify configured RealSense serials before connecting to either robot driver."""
    if not cameras:
        return
    try:
        import pyrealsense2 as rs
        from rcs_realsense.camera import RealSenseCameraSet
    except ImportError as error:
        raise RuntimeError("RealSense cameras require the rcs_realsense extension") from error
    devices = RealSenseCameraSet.enumerate_connected_devices(rs.context())
    connected = {device.serial for device in devices.values()}
    missing = missing_camera_serials(cameras, connected)
    if missing:
        details = ", ".join(f"{name}={serial}" for name, serial in sorted(missing.items()))
        raise RuntimeError(f"configured RealSense cameras are not connected: {details}")


def validate_safety_args(
    args: argparse.Namespace,
) -> tuple[str, Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    required = (
        "robot_ip",
        "gripper_serial",
        "gripper_device",
        "instruction",
        "output",
        "workspace_min",
        "workspace_max",
        "tool_box_min",
        "tool_box_max",
        "tcp_translation",
        "tcp_quaternion",
        "home_joints_deg",
    )
    missing = [name for name in required if getattr(args, name, None) in (None, "")]
    if missing:
        raise ValueError(f"missing required configuration values: {', '.join(missing)}")
    robot_ip = str(ipaddress.ip_address(args.robot_ip))
    try:
        gripper_device = Path(args.gripper_device).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"gripper device does not exist: {args.gripper_device}") from error
    if not gripper_device.name.startswith("ttyUSB"):
        raise ValueError(f"gripper device must resolve to /dev/ttyUSB*, got {gripper_device}")
    args.gripper_device = str(gripper_device)
    expected = f"{CONFIRM_PREFIX} {robot_ip}"
    if args.confirm != expected:
        raise ValueError(f"confirmation must be exactly: {expected}")
    if not (0.0 < args.step <= MAX_STEP_METERS):
        raise ValueError(f"--step must be > 0 and <= {MAX_STEP_METERS}")
    if not (0.0 < args.speed_factor <= MAX_SPEED_FACTOR):
        raise ValueError(f"--speed-factor must be > 0 and <= {MAX_SPEED_FACTOR}")
    if not (1 <= args.frequency <= 30):
        raise ValueError("--frequency must be between 1 and 30 Hz")
    if not (20.0 <= args.gripper_speed <= MAX_INITIAL_GRIPPER_SPEED):
        raise ValueError(f"--gripper-speed must be between 20 and {MAX_INITIAL_GRIPPER_SPEED} mm/s")
    if not (20.0 <= args.gripper_force <= MAX_INITIAL_GRIPPER_FORCE):
        raise ValueError(f"--gripper-force must be between 20 and {MAX_INITIAL_GRIPPER_FORCE} N")
    cameras = parse_camera_specs(args.cameras)
    if args.require_cameras:
        missing_cameras = {"wrist", "third_person"} - cameras.keys()
        if missing_cameras:
            raise ValueError(
                "camera configuration must include wrist and third_person; missing: "
                + ", ".join(sorted(missing_cameras))
            )
    if not (320 <= args.camera_width <= 1920 and 240 <= args.camera_height <= 1080):
        raise ValueError("camera dimensions must be between 320x240 and 1920x1080")
    if not (1 <= args.camera_fps <= 60):
        raise ValueError("--camera-fps must be between 1 and 60")

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        raise ValueError("--output must be an absolute path")
    if output.exists():
        raise ValueError(f"refusing to reuse existing output path: {output}")

    workspace_min = np.asarray(args.workspace_min, dtype=float)
    workspace_max = np.asarray(args.workspace_max, dtype=float)
    tool_box_min = np.asarray(args.tool_box_min, dtype=float)
    tool_box_max = np.asarray(args.tool_box_max, dtype=float)
    for label, low, high in (
        ("workspace", workspace_min, workspace_max),
        ("tool-box", tool_box_min, tool_box_max),
    ):
        if low.shape != (3,) or high.shape != (3,) or not np.all(np.isfinite([low, high])):
            raise ValueError(f"{label} bounds must each be three finite values")
        if not np.all(low < high):
            raise ValueError(f"every {label} minimum coordinate must be below its maximum")
    if not np.all(tool_box_min <= 0.0) or not np.all(tool_box_max >= 0.0):
        raise ValueError("the tool box must contain the TCP origin [0, 0, 0]")
    tcp_translation = np.asarray(args.tcp_translation, dtype=float)
    tcp_quaternion = np.asarray(args.tcp_quaternion, dtype=float)
    if tcp_translation.shape != (3,) or not np.all(np.isfinite(tcp_translation)):
        raise ValueError("--tcp-translation must contain three finite values")
    if tcp_quaternion.shape != (4,) or not np.all(np.isfinite(tcp_quaternion)):
        raise ValueError("--tcp-quaternion must contain four finite values in xyzw order")
    if not np.isclose(np.linalg.norm(tcp_quaternion), 1.0, atol=1e-3):
        raise ValueError("--tcp-quaternion must be normalized")
    if not (0.0 < args.joint_limit_margin_deg <= 20.0):
        raise ValueError("--joint-limit-margin-deg must be > 0 and <= 20")
    home_joints_deg = np.asarray(args.home_joints_deg, dtype=float)
    if home_joints_deg.shape != (7,) or not np.all(np.isfinite(home_joints_deg)):
        raise ValueError("--home-joints-deg must contain seven finite values")
    return robot_ip, output, workspace_min, workspace_max, tool_box_min, tool_box_max, cameras


@contextlib.contextmanager
def raw_terminal() -> Iterator[None]:
    if not sys.stdin.isatty():
        raise RuntimeError("Real keyboard teleoperation requires an interactive terminal.")
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def create_hardware_env(robot_ip: str, gripper_serial: str, args: argparse.Namespace):
    from rcs import common
    from rcs.envs.base import RelativeTo
    from rcs_panda.configs import DROIDEnv
    from rcs_panda.creators import HardwareCameraCreatorConfig

    creator = DROIDEnv()
    creator.robot_ip = robot_ip
    creator.gripper_serial_number = gripper_serial
    cfg = creator.config()
    cfg.wrapper_cfg.home_on_reset = False
    cfg.wrapper_cfg.binary_gripper = True
    cameras = parse_camera_specs(args.cameras)
    cfg.camera_cfgs = (
        {
            "realsense": HardwareCameraCreatorConfig(
                camera_type_id="realsense",
                camera_cfgs={
                    name: common.BaseCameraConfig(
                        identifier=serial,
                        resolution_width=args.camera_width,
                        resolution_height=args.camera_height,
                        frame_rate=args.camera_fps,
                    )
                    for name, serial in cameras.items()
                },
                kwargs={
                    "enable_ir_emitter": False,
                    "enable_ir": False,
                    "enable_imu": False,
                    "align_depth_to_color": args.include_depth,
                },
            )
        }
        if cameras
        else None
    )
    cfg.wrapper_cfg.include_depth = args.include_depth
    # Hold a fixed Cartesian target between key presses. LAST_STEP rebases on
    # every measured pose, so a small tracking sag would be accepted as the
    # next target at 20 Hz and become continuous downward drift.
    cfg.relative_to = RelativeTo.CONFIGURED_ORIGIN
    cfg.max_relative_movement = (args.step, np.deg2rad(1.0))

    robot_cfg = cfg.robot_cfgs["right"]
    robot_cfg.tcp_offset = common.Pose(
        translation=np.asarray(args.tcp_translation, dtype=float),
        quaternion=np.asarray(args.tcp_quaternion, dtype=float),
    )
    robot_cfg.q_home = np.deg2rad(np.asarray(args.home_joints_deg, dtype=float))
    # This is a shared robot. Desk's stock Franka Hand mechanical data is
    # authoritative, so this process must never call libfranka setLoad().
    robot_cfg.load_parameters = None
    robot_cfg.speed_factor = args.speed_factor
    robot_cfg.policy_rate = args.frequency
    robot_cfg.ignore_realtime = False
    # Keep the repository's legacy asynchronous OSC startup behavior. Safety
    # gates stay outside the controller and do not alter its torque output.
    # Wait for the measured starting pose to settle before teleoperation.
    robot_cfg.blocking_move_on_start = True
    # Pinocchio's MJCF parser omits sites; this body has attachment_site's pose.
    robot_cfg.attachment_site = "attachment"

    gripper_cfg = cfg.gripper_cfgs["right"]
    gripper_cfg.serial_device = args.gripper_device
    gripper_cfg.speed = args.gripper_speed
    gripper_cfg.force = args.gripper_force
    return creator.create_env(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON defaults; explicit CLI arguments override them.")
    parser.add_argument("--robot-ip")
    parser.add_argument("--gripper-serial")
    parser.add_argument("--gripper-device", help="Explicit /dev/serial/by-id or /dev/ttyUSB path.")
    parser.add_argument("--instruction")
    parser.add_argument("--output", help="New absolute directory for Parquet fragments.")
    parser.add_argument("--workspace-min", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--workspace-max", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--tool-box-min",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Minimum tool-envelope corner relative to TCP, in metres.",
    )
    parser.add_argument(
        "--tool-box-max",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Maximum tool-envelope corner relative to TCP, in metres.",
    )
    parser.add_argument("--joint-limit-margin-deg", type=float, default=5.0)
    parser.add_argument(
        "--home-joints-deg",
        nargs=7,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Fixed startup home pose in joint degrees.",
    )
    parser.add_argument(
        "--tcp-translation",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Flange-to-TCP translation in metres.",
    )
    parser.add_argument(
        "--tcp-quaternion",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Flange-to-TCP quaternion in xyzw order.",
    )
    parser.add_argument("--step", type=float, default=MAX_STEP_METERS)
    parser.add_argument("--speed-factor", type=float, default=0.05)
    parser.add_argument("--frequency", type=int, default=20)
    parser.add_argument("--gripper-speed", type=float, default=20.0, help="mm/s; minimum driver value.")
    parser.add_argument("--gripper-force", type=float, default=20.0, help="N; minimum driver value.")
    parser.add_argument(
        "--camera",
        dest="cameras",
        action="append",
        metavar="NAME=SERIAL",
        help="RealSense name and serial; repeat for wrist and third_person.",
    )
    parser.add_argument(
        "--require-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse collection unless wrist and third_person cameras are configured.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--include-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm", required=True, help=f"Must equal '{CONFIRM_PREFIX} <robot-ip>'.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    preliminary, _ = config_parser.parse_known_args(argv)
    parser = build_parser()
    if preliminary.config is not None:
        try:
            config_data: dict[str, Any] = json.loads(preliminary.config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read JSON config {preliminary.config}: {error}")
        valid_keys = {action.dest for action in parser._actions} - {"help", "confirm", "config"}
        unknown = sorted(set(config_data) - valid_keys)
        if unknown:
            parser.error(f"unknown JSON config keys: {', '.join(unknown)}")
        parser.set_defaults(**config_data)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        robot_ip, output, workspace_min, workspace_max, tool_box_min, tool_box_max, cameras = (
            validate_safety_args(args)
        )
    except ValueError as error:
        raise SystemExit(f"Safety validation failed: {error}") from error

    print("REAL HARDWARE MODE")
    print(f"Panda: {robot_ip}; Robotiq serial: {args.gripper_serial}")
    print(f"Robotiq port: {args.gripper_device}")
    print(f"workspace: {workspace_min} .. {workspace_max}; step: {args.step} m")
    print(f"tool envelope relative to TCP: {tool_box_min} .. {tool_box_max}")
    print(f"RealSense cameras: {cameras or 'disabled'}; include depth: {args.include_depth}")
    for warning in realtime_host_warnings():
        print(f"REALTIME WARNING: {warning}")
    try:
        preflight_cameras(cameras)
    except RuntimeError as error:
        raise SystemExit(f"Camera preflight failed before hardware connection: {error}") from error
    print("Initialization performs Franka error recovery and OPENS the Robotiq gripper.")
    print(f"Configured home joints (deg): {np.asarray(args.home_joints_deg, dtype=float)}")
    print("A second confirmation is required before home motion. Keep the workspace clear.")
    if input(f"Type 'START {robot_ip}' to create the hardware environment: ").strip() != f"START {robot_ip}":
        raise SystemExit("Cancelled before opening either hardware driver.")

    try:
        from rcs.envs.storage_wrapper import StorageWrapper
    except ImportError as error:
        raise SystemExit("The local rcs package is not installed.") from error

    base_env = None
    recorder = None
    recording = False
    episode_finished = False
    try:
        base_env = create_hardware_env(robot_ip, args.gripper_serial, args)
        observation, info = base_env.reset()
        robot = base_env.get_wrapper_attr("robot")["right"]
        verify_franka_hand_defaults(robot.get_state().robot_state)
        joint_limits = np.asarray(robot.get_config().joint_limits, dtype=float)
        joint_margin = np.deg2rad(args.joint_limit_margin_deg)
        start_joints = np.asarray(observation["right"]["joints"], dtype=float)
        home_joints = np.deg2rad(np.asarray(args.home_joints_deg, dtype=float))
        if not joints_have_margin(home_joints, joint_limits, joint_margin):
            raise RuntimeError("Configured home pose is inside the joint-limit margin")
        initial_collisions = collision_report(observation)
        if initial_collisions:
            raise RuntimeError(f"Initial collision signal asserted: {initial_collisions}")
        kinematics = robot.get_ik()
        if kinematics is None:
            raise RuntimeError("Cannot preflight home path without Panda kinematics")
        if not home_path_within_workspace(
            kinematics,
            start_joints,
            home_joints,
            robot.get_config().tcp_offset,
            tool_box_min,
            tool_box_max,
            workspace_min,
            workspace_max,
        ):
            raise RuntimeError("Sampled path to home would move the tool envelope outside the workspace")
        expected_home_confirmation = f"{HOME_CONFIRM_PREFIX} {robot_ip}"
        home_confirmation = input(
            f"Type '{expected_home_confirmation}' to move to the fixed home pose: "
        ).strip()
        if home_confirmation != expected_home_confirmation:
            raise SystemExit("Cancelled before home motion.")
        robot.move_home()
        observation, info = base_env.reset()
        measured_home = np.asarray(observation["right"]["joints"], dtype=float)
        if not np.allclose(
            measured_home, home_joints, rtol=0.0, atol=np.deg2rad(HOME_TOLERANCE_DEG)
        ):
            raise RuntimeError(
                "Home motion did not reach the configured pose within "
                f"{HOME_TOLERANCE_DEG} deg; measured={np.rad2deg(measured_home)}"
            )
        current_tquat = np.asarray(observation["right"]["tquat"], dtype=float)
        current_tcp = current_tquat[:3]
        session_origin_tquat = current_tquat.copy()
        command_offset = np.zeros(3, dtype=float)
        if not tool_within_workspace(
            current_tquat, tool_box_min, tool_box_max, workspace_min, workspace_max
        ):
            raise RuntimeError("Initial Robotiq tool envelope is outside the configured workspace")
        if not joints_have_margin(observation["right"]["joints"], joint_limits, joint_margin):
            raise RuntimeError("Initial arm position is inside the configured joint-limit margin")
        initial_collisions = collision_report(observation)
        if initial_collisions:
            raise RuntimeError(f"Collision signal asserted after home: {initial_collisions}")

        output.mkdir(parents=True, exist_ok=False)
        metadata = {
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "robot_ip": robot_ip,
            "gripper_serial": args.gripper_serial,
            "gripper_device": args.gripper_device,
            "instruction": args.instruction,
            "frequency": args.frequency,
            "fixed_rate_keyboard_loop": True,
            "step_meters": args.step,
            "record_batch_seconds": RECORD_BATCH_SECONDS,
            "minimum_post_gripper_record_seconds": MIN_POST_GRIPPER_RECORD_SECONDS,
            "minimum_control_command_success_rate": MIN_CONTROL_COMMAND_SUCCESS_RATE,
            "maximum_low_control_seconds": MAX_LOW_CONTROL_SECONDS,
            "speed_factor": args.speed_factor,
            "relative_action_reference": "configured_session_origin",
            "workspace_min": workspace_min.tolist(),
            "workspace_max": workspace_max.tolist(),
            "tool_box_min_tcp_frame": tool_box_min.tolist(),
            "tool_box_max_tcp_frame": tool_box_max.tolist(),
            "joint_limit_margin_deg": args.joint_limit_margin_deg,
            "home_joints_deg": args.home_joints_deg,
            "desk_end_effector_preset": DESK_END_EFFECTOR_PRESET,
            "libfranka_set_load_called": False,
            "flange_to_tcp_translation_m": args.tcp_translation,
            "flange_to_tcp_quaternion_xyzw": args.tcp_quaternion,
            "cameras": cameras,
            "camera_resolution": [args.camera_width, args.camera_height],
            "camera_fps": args.camera_fps,
            "include_depth": args.include_depth,
            "camera_extrinsics": "uncalibrated/dummy",
        }
        (output / "_session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        recorder = StorageWrapper(
            base_env,
            str(output),
            args.instruction,
            # Five-second batches keep the asynchronous writer comfortably
            # ahead of image production. Tiny batches create many Parquet
            # files and can fill the bounded writer queue, stalling control.
            batch_size=max(1, args.frequency * RECORD_BATCH_SECONDS),
            max_rows_per_group=100,
            max_rows_per_file=1000,
        )

        gripper_state = 1
        period = 1.0 / args.frequency
        print("Initial TCP:", current_tcp)
        print("W/S x, A/D y, R/F z, Q open, E close")
        print(
            f"Each movement tick: {args.step * 1000:.1f} mm; "
            f"held-key maximum: {args.step * args.frequency * 1000:.1f} mm/s"
        )
        print("T start recording, Y finish SUCCESS, N finish FAILURE, ESC exit")
        with raw_terminal():
            next_tick = time.monotonic()
            last_status_print = 0.0
            last_gripper_change = float("-inf")
            low_control_ticks = 0
            while True:
                timeout = max(0.0, next_tick - time.monotonic())
                keys = read_pending_keys(sys.stdin.fileno(), timeout)
                # select() returns immediately when a key is buffered. Still
                # wait for the scheduled deadline so held keys cannot make the
                # control/data loop run at the keyboard repeat or camera rate.
                remaining = next_tick - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)

                if "\x1b" in keys:
                    break

                key = ""
                for event in keys:
                    if event in KEY_DELTAS:
                        key = event
                    elif event == "t":
                        if episode_finished:
                            print("Episode already finished; exit and use a new output directory.")
                        elif not recording:
                            recorder.start_record()
                            recording = True
                            print("RECORDING STARTED")
                    elif event in {"q", "e"}:
                        requested_gripper_state = 1 if event == "q" else 0
                        if requested_gripper_state != gripper_state:
                            gripper_state = requested_gripper_state
                            last_gripper_change = time.monotonic()
                        key = event
                    elif event == "y" and recording:
                        tail_seconds = time.monotonic() - last_gripper_change
                        if tail_seconds < MIN_POST_GRIPPER_RECORD_SECONDS:
                            print(
                                "WAIT before SUCCESS: record at least "
                                f"{MIN_POST_GRIPPER_RECORD_SECONDS:.1f} s after the last gripper command"
                            )
                        else:
                            recorder.success()
                            recorder.stop_record()
                            recording = False
                            episode_finished = True
                            print("RECORDING STOPPED: SUCCESS")
                    elif event == "n" and recording:
                        recorder.stop_record()
                        recording = False
                        episode_finished = True
                        print("RECORDING STOPPED: FAILURE")

                delta = delta_for_key(key, args.step)
                if delta is not None:
                    proposed_offset = accumulated_offset(command_offset, key, args.step)
                    target_tquat = session_origin_tquat.copy()
                    target_tquat[:3] += proposed_offset
                    if not tool_within_workspace(
                        target_tquat,
                        tool_box_min,
                        tool_box_max,
                        workspace_min,
                        workspace_max,
                    ):
                        print(f"BLOCKED by tool/workspace limit: TCP={current_tcp}, delta={delta}")
                    elif not joints_have_margin(
                        observation["right"]["joints"], joint_limits, joint_margin
                    ):
                        print("BLOCKED because a joint is inside the configured limit margin")
                    else:
                        command_offset = proposed_offset

                observation, reward, terminated, truncated, info = recorder.step(
                    make_action(command_offset, gripper_state)
                )
                current_tquat = np.asarray(observation["right"]["tquat"], dtype=float)
                current_tcp = current_tquat[:3]
                if not tool_within_workspace(
                    current_tquat,
                    tool_box_min,
                    tool_box_max,
                    workspace_min,
                    workspace_max,
                ):
                    raise RuntimeError("SAFETY STOP: measured tool envelope left the workspace")
                if not joints_have_margin(observation["right"]["joints"], joint_limits, joint_margin):
                    violations = joint_margin_violations(
                        observation["right"]["joints"], joint_limits, joint_margin
                    )
                    raise RuntimeError(
                        "SAFETY STOP: measured joint entered the configured limit margin: "
                        + "; ".join(violations)
                    )
                collisions = collision_report(observation)
                if collisions:
                    raise RuntimeError(f"SAFETY STOP: collision signal asserted: {collisions}")
                success_rate = control_command_success_rate(observation)
                if success_rate is not None and success_rate < MIN_CONTROL_COMMAND_SUCCESS_RATE:
                    low_control_ticks += 1
                else:
                    low_control_ticks = 0
                if low_control_ticks >= max(1, int(args.frequency * MAX_LOW_CONTROL_SECONDS)):
                    raise RuntimeError(
                        "SAFETY STOP: libfranka control command success rate stayed below "
                        f"{MIN_CONTROL_COMMAND_SUCCESS_RATE:.2f} for approximately "
                        f"{MAX_LOW_CONTROL_SECONDS:.1f} s (latest={success_rate:.3f})"
                    )
                now = time.monotonic()
                if (key in KEY_DELTAS or key in {"q", "e"}) and now - last_status_print >= 0.2:
                    print(
                        f"key={key} target_offset={command_offset} TCP={current_tcp} "
                        f"gripper={observation['right']['gripper']} "
                        f"recording={recording} reward={reward}"
                    )
                    last_status_print = now
                if terminated or truncated:
                    print("Environment stopped:", info)
                    break
                next_tick = max(next_tick + period, time.monotonic())
    finally:
        if recorder is not None:
            if recording:
                recorder.stop_record()
                print("Active episode flushed as FAILURE/unfinished.")
            recorder.close()
        if base_env is not None:
            base_env.close()
        print(f"Closed hardware environment. Dataset directory: {output}")


if __name__ == "__main__":
    main()
