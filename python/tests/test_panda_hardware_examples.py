import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "examples" / "panda" / "panda_hardware_keyboard_collect.py"
SPEC = importlib.util.spec_from_file_location("panda_hardware_keyboard_collect", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

DEVICE_SCRIPT = Path(__file__).parents[2] / "examples" / "panda" / "panda_hardware_device_check.py"
DEVICE_SPEC = importlib.util.spec_from_file_location("panda_hardware_device_check", DEVICE_SCRIPT)
assert DEVICE_SPEC is not None and DEVICE_SPEC.loader is not None
DEVICE_MODULE = importlib.util.module_from_spec(DEVICE_SPEC)
DEVICE_SPEC.loader.exec_module(DEVICE_MODULE)

WORKSPACE_SCRIPT = Path(__file__).parents[2] / "examples" / "panda" / "panda_workspace_from_pose.py"
WORKSPACE_SPEC = importlib.util.spec_from_file_location("panda_workspace_from_pose", WORKSPACE_SCRIPT)
assert WORKSPACE_SPEC is not None and WORKSPACE_SPEC.loader is not None
WORKSPACE_MODULE = importlib.util.module_from_spec(WORKSPACE_SPEC)
WORKSPACE_SPEC.loader.exec_module(WORKSPACE_MODULE)

MICRO_MOTION_SCRIPT = Path(__file__).parents[2] / "examples" / "panda" / "panda_hardware_micro_motion.py"
MICRO_MOTION_SPEC = importlib.util.spec_from_file_location("panda_hardware_micro_motion", MICRO_MOTION_SCRIPT)
assert MICRO_MOTION_SPEC is not None and MICRO_MOTION_SPEC.loader is not None
MICRO_MOTION_MODULE = importlib.util.module_from_spec(MICRO_MOTION_SPEC)
sys.modules[MICRO_MOTION_SPEC.name] = MICRO_MOTION_MODULE
MICRO_MOTION_SPEC.loader.exec_module(MICRO_MOTION_MODULE)

HOLD_SCRIPT = Path(__file__).parents[2] / "examples" / "panda" / "panda_hardware_cartesian_hold.py"
HOLD_SPEC = importlib.util.spec_from_file_location("panda_hardware_cartesian_hold", HOLD_SCRIPT)
assert HOLD_SPEC is not None and HOLD_SPEC.loader is not None
HOLD_MODULE = importlib.util.module_from_spec(HOLD_SPEC)
HOLD_SPEC.loader.exec_module(HOLD_MODULE)

REPLAY_SCRIPT = Path(__file__).parents[2] / "examples" / "panda" / "panda_hardware_replay.py"
REPLAY_SPEC = importlib.util.spec_from_file_location("panda_hardware_replay", REPLAY_SCRIPT)
assert REPLAY_SPEC is not None and REPLAY_SPEC.loader is not None
REPLAY_MODULE = importlib.util.module_from_spec(REPLAY_SPEC)
sys.modules[REPLAY_SPEC.name] = REPLAY_MODULE
REPLAY_SPEC.loader.exec_module(REPLAY_MODULE)

EXAMPLE_CONFIG = Path(__file__).parents[2] / "examples" / "panda" / "panda_hardware_session.example.json"
LAB_CONFIG = Path(__file__).parents[2] / "examples" / "panda" / "panda_hardware_session.lab.json"


def make_args(tmp_path: Path, **overrides):
    values = {
        "robot_ip": "192.168.4.100",
        "gripper_serial": "TEST-SERIAL",
        "gripper_device": "/dev/null",
        "instruction": "test instruction",
        "confirm": "ENABLE-REAL-PANDA 192.168.4.100",
        "step": 0.005,
        "speed_factor": 0.05,
        "frequency": 20,
        "gripper_speed": 20.0,
        "gripper_force": 20.0,
        "output": str(tmp_path / "new-session"),
        "workspace_min": [0.2, -0.3, 0.1],
        "workspace_max": [0.7, 0.3, 0.8],
        "tool_box_min": [-0.05, -0.05, -0.15],
        "tool_box_max": [0.05, 0.05, 0.02],
        "joint_limit_margin_deg": 5.0,
        "home_joints_deg": [0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 0.0],
        "tcp_translation": [0.0, 0.0, 0.15],
        "tcp_quaternion": [0.0, 0.0, 0.0, 1.0],
        "cameras": {"wrist": "WRIST-SERIAL", "third_person": "THIRD-SERIAL"},
        "require_cameras": True,
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 30,
        "include_depth": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_keyboard_mapping_and_action_shape():
    assert np.allclose(MODULE.delta_for_key("W"), [0.005, 0.0, 0.0])
    assert np.allclose(MODULE.delta_for_key("f"), [0.0, 0.0, -0.005])
    assert MODULE.delta_for_key("?") is None
    action = MODULE.make_action(np.array([0.005, 0.0, 0.0]), 1)
    assert action == {"right": {"tquat": [0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "gripper": [1]}}

    offset = MODULE.accumulated_offset(np.zeros(3), "w", 0.005)
    np.testing.assert_allclose(offset, [0.005, 0.0, 0.0])
    np.testing.assert_allclose(MODULE.accumulated_offset(offset, "a", 0.005), [0.005, 0.005, 0.0])
    np.testing.assert_allclose(MODULE.accumulated_offset(offset, "q", 0.005), offset)


def test_control_command_success_rate_handles_missing_and_nonfinite_values():
    assert MODULE.control_command_success_rate({}) is None
    observation = {"right": {"robot_state": {"control_command_success_rate": 0.97}}}
    assert MODULE.control_command_success_rate(observation) == pytest.approx(0.97)
    observation["right"]["robot_state"]["control_command_success_rate"] = np.nan
    assert MODULE.control_command_success_rate(observation) is None


def test_robotiq_hardware_creator_uses_configured_gripper_type_id():
    from rcs._core.common import GripperType
    from rcs_panda.creators import HARDWARE_GRIPPER_CREATORS

    assert GripperType("Robotiq2F85").id in HARDWARE_GRIPPER_CREATORS


def test_micro_motion_is_fixed_to_two_mm_and_returns():
    up, back = MICRO_MOTION_MODULE.movement_actions()
    np.testing.assert_allclose(up["tquat"], [0, 0, 0.002, 0, 0, 0, 1])
    np.testing.assert_allclose(back["tquat"], [0, 0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="exactly"):
        MICRO_MOTION_MODULE.movement_actions(0.005)


def test_cartesian_hold_action_commands_zero_motion():
    np.testing.assert_allclose(HOLD_MODULE.hold_action()["tquat"], [0, 0, 0, 0, 0, 0, 1])
    assert HOLD_MODULE.HOLD_SECONDS == 2.0
    assert HOLD_MODULE.MAX_DRIFT_METERS == 0.001


def test_hardware_replay_quaternion_angle_ignores_sign():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    assert REPLAY_MODULE.quaternion_angle_degrees(identity, -identity) == pytest.approx(0.0)
    quarter_turn = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    assert REPLAY_MODULE.quaternion_angle_degrees(identity, quarter_turn) == pytest.approx(90.0)


def test_hardware_replay_target_uses_configured_origin():
    origin = np.array([0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0])
    relative = np.array([0.01, 0.02, -0.03, 0.0, 0.0, 0.0, 1.0])
    target = REPLAY_MODULE.target_tquat_from_origin(origin, relative)
    np.testing.assert_allclose(target, [0.41, -0.08, 0.47, 0.0, 0.0, 0.0, 1.0])


def make_replay_episode(
    timestamps: list[float],
    *,
    gripper_change_step: int | None = None,
    success: bool = False,
    control_rates: list[float] | None = None,
):
    rates = control_rates or [1.0] * len(timestamps)
    steps = []
    for index, (timestamp, rate) in enumerate(zip(timestamps, rates)):
        gripper = 0 if gripper_change_step is not None and index >= gripper_change_step else 1
        steps.append(
            REPLAY_MODULE.ReplayStep(
                step=index,
                timestamp=timestamp,
                action={"right": {"tquat": [0, 0, 0, 0, 0, 0, 1], "gripper": [gripper]}},
                recorded_tquat=np.array([0, 0, 0, 0, 0, 0, 1], dtype=float),
                recorded_target_tquat=np.array([0, 0, 0, 0, 0, 0, 1], dtype=float),
                recorded_joints=np.zeros(7),
                joint_collision=np.zeros(7),
                cartesian_collision=np.zeros(6),
                success=success and index == len(timestamps) - 1,
                frame_timestamp=timestamp,
                camera_available=True,
                control_command_success_rate=rate,
            )
        )
    return REPLAY_MODULE.Episode("test", "pick", steps)


def test_hardware_replay_validates_real_timing_and_post_gripper_tail():
    metadata = {"frequency": 10, "cameras": {"third_person": "serial"}}
    valid = make_replay_episode([index * 0.1 for index in range(21)], gripper_change_step=5, success=True)
    REPLAY_MODULE.validate_episode_timing(valid, metadata)

    wrong_rate = make_replay_episode([index * 0.03 for index in range(21)])
    with pytest.raises(ValueError, match="does not match metadata"):
        REPLAY_MODULE.validate_episode_timing(wrong_rate, metadata)

    no_tail = make_replay_episode([index * 0.1 for index in range(21)], gripper_change_step=20, success=True)
    with pytest.raises(ValueError, match="after its last gripper command"):
        REPLAY_MODULE.validate_episode_timing(no_tail, metadata)


def test_hardware_replay_rejects_sustained_low_control_success_rate():
    metadata = {"frequency": 10, "cameras": {}}
    rates = [1.0] * 5 + [0.8] * 10 + [1.0] * 6
    episode = make_replay_episode([index * 0.1 for index in range(21)], control_rates=rates)
    with pytest.raises(ValueError, match="control command success rate"):
        REPLAY_MODULE.validate_episode_timing(episode, metadata)


def test_hardware_replay_requires_matching_session_metadata():
    config = {
        "robot_ip": "192.168.4.100",
        "gripper_serial": "TEST-SERIAL",
        "frequency": 20,
        "step": 0.001,
        "workspace_min": [0.1, -0.4, -0.05],
        "workspace_max": [0.85, 0.4, 0.95],
        "tool_box_min": [-0.13, -0.13, -0.16],
        "tool_box_max": [0.13, 0.13, 0.04],
        "home_joints_deg": [0.0, -24.0, 0.0, -142.0, 0.0, 122.0, 0.0],
        "tcp_translation": [0.0, 0.0, 0.1493],
        "tcp_quaternion": [0.0, 0.0, 0.0, 1.0],
    }
    metadata = {
        "robot_ip": "192.168.4.100",
        "gripper_serial": "TEST-SERIAL",
        "frequency": 20,
        "step_meters": 0.001,
        "workspace_min": [0.1, -0.4, -0.05],
        "workspace_max": [0.85, 0.4, 0.95],
        "tool_box_min_tcp_frame": [-0.13, -0.13, -0.16],
        "tool_box_max_tcp_frame": [0.13, 0.13, 0.04],
        "home_joints_deg": [0.0, -24.0, 0.0, -142.0, 0.0, 122.0, 0.0],
        "flange_to_tcp_translation_m": [0.0, 0.0, 0.1493],
        "flange_to_tcp_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    REPLAY_MODULE.validate_metadata(config, metadata)
    config["step"] = 0.002
    REPLAY_MODULE.validate_metadata(config, metadata)
    metadata["frequency"] = 30
    with pytest.raises(ValueError, match="frequency"):
        REPLAY_MODULE.validate_metadata(config, metadata)


def test_hardware_replay_rejects_unsafe_recorded_step():
    episode = make_replay_episode([index * 0.1 for index in range(21)])
    metadata = {
        "frequency": 10,
        "cameras": {},
        "workspace_min": [-1.0, -1.0, -1.0],
        "workspace_max": [1.0, 1.0, 1.0],
        "tool_box_min_tcp_frame": [-0.1, -0.1, -0.1],
        "tool_box_max_tcp_frame": [0.1, 0.1, 0.1],
        "step_meters": 0.006,
    }
    with pytest.raises(ValueError, match="recorded step_meters"):
        REPLAY_MODULE.validate_episode(episode, metadata)


def test_franka_hand_defaults_are_required_before_motion():
    state = SimpleNamespace(
        m_ee=MODULE.FRANKA_HAND_MASS_KG,
        F_x_Cee=MODULE.FRANKA_HAND_COM_M,
        I_ee=MODULE.FRANKA_HAND_INERTIA_KG_M2.reshape(-1, order="F"),
        F_T_EE=MODULE.FRANKA_HAND_F_T_EE.reshape(-1, order="F"),
        m_load=0.0,
    )
    MODULE.verify_franka_hand_defaults(state)
    state.m_ee = 0.921
    with pytest.raises(RuntimeError, match="Franka Hand"):
        MODULE.verify_franka_hand_defaults(state)


def test_parse_udev_properties_keeps_only_hardware_identifiers():
    output = "ID_VENDOR_ID=0403\nID_MODEL_ID=6015\nID_SERIAL_SHORT=ABC123\nDEVNAME=/dev/ttyUSB0\n"
    assert DEVICE_MODULE.parse_udev_properties(output) == {
        "ID_VENDOR_ID": "0403",
        "ID_MODEL_ID": "6015",
        "ID_SERIAL_SHORT": "ABC123",
    }


def test_workspace_guard():
    low = np.array([0.2, -0.3, 0.1])
    high = np.array([0.7, 0.3, 0.8])
    assert MODULE.workspace_allows(np.array([0.4, 0.0, 0.4]), np.array([0.005, 0.0, 0.0]), low, high)
    assert not MODULE.workspace_allows(np.array([0.699, 0.0, 0.4]), np.array([0.005, 0.0, 0.0]), low, high)


def test_tool_envelope_uses_tcp_orientation():
    low = np.array([-0.2, -0.2, -0.2])
    high = np.array([0.2, 0.2, 0.2])
    tool_low = np.array([-0.1, -0.02, -0.02])
    tool_high = np.array([0.1, 0.02, 0.02])
    identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert MODULE.tool_within_workspace(identity, tool_low, tool_high, low, high)

    # 90 degrees about z rotates the long tool dimension from x to y.
    rotated = np.array([0.0, 0.11, 0.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    assert not MODULE.tool_within_workspace(rotated, tool_low, tool_high, low, high)


def test_example_requires_measured_tool_box_for_attached_wrist_camera():
    config = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert config["tool_box_min"] is None
    assert config["tool_box_max"] is None
    assert config["require_cameras"] is True


def test_camera_specs_accept_json_mapping_and_cli_form():
    expected = {"wrist": "123", "third_person": "456"}
    assert MODULE.parse_camera_specs(expected) == expected
    assert MODULE.parse_camera_specs(["wrist=123", "third_person=456"]) == expected
    with pytest.raises(ValueError, match="NAME=SERIAL"):
        MODULE.parse_camera_specs(["invalid"])


def test_missing_camera_serials_reports_names():
    cameras = {"wrist": "123", "third_person": "456"}
    assert MODULE.missing_camera_serials(cameras, {"123"}) == {"third_person": "456"}


def test_missing_camera_serials_accepts_l515_udev_serial_form():
    assert MODULE.missing_camera_serials(
        {"third_person": "00000000ABC12345"}, {"abc12345"}
    ) == {}


def test_workspace_calculator_includes_oriented_tool_and_requested_travel():
    tcp = np.array([0.4, 0.0, 0.5, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    tool_low = np.array([-0.1, -0.02, -0.15])
    tool_high = np.array([0.1, 0.02, 0.03])
    negative = np.array([0.02, 0.03, 0.01])
    positive = np.array([0.04, 0.01, 0.02])
    low, high = WORKSPACE_MODULE.workspace_around_pose(
        tcp, tool_low, tool_high, negative, positive
    )
    corners = MODULE.tool_corners_in_base(tcp, tool_low, tool_high)

    assert np.allclose(low, corners.min(axis=0) - negative)
    assert np.allclose(high, corners.max(axis=0) + positive)


def test_workspace_calculator_rejects_negative_travel():
    with pytest.raises(ValueError, match="non-negative"):
        WORKSPACE_MODULE.workspace_around_pose(
            np.array([0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]),
            np.array([-0.1, -0.1, -0.15]),
            np.array([0.1, 0.1, 0.03]),
            np.array([0.02, 0.02, -0.01]),
            np.array([0.02, 0.02, 0.01]),
        )


def test_joint_margin_and_collision_report():
    limits = np.vstack((-np.ones(7), np.ones(7)))
    assert MODULE.joints_have_margin(np.zeros(7), limits, 0.1)
    assert not MODULE.joints_have_margin(np.array([0.95, 0, 0, 0, 0, 0, 0]), limits, 0.1)
    assert MODULE.joint_margin_violations(
        np.array([0.95, 0, 0, 0, 0, 0, -0.95]), limits, 0.1
    ) == [
        "J1=54.43 deg (safe -51.57..51.57 deg)",
        "J7=-54.43 deg (safe -51.57..51.57 deg)",
    ]
    obs = {
        "right": {
            "robot_state": {
                "joint_collision": np.array([0, 0, 1, 0, 0, 0, 0]),
                "cartesian_collision": np.zeros(6),
            }
        }
    }
    assert MODULE.collision_report(obs) == ["joint_collision=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]"]


def test_lab_home_tool_envelope_is_inside_workspace():
    import rcs
    from rcs import common
    from rcs._core.common import RobotType

    config = json.loads(LAB_CONFIG.read_text(encoding="utf-8"))
    home = np.deg2rad(config["home_joints_deg"])
    tcp_offset = common.Pose(
        translation=np.asarray(config["tcp_translation"], dtype=float),
        quaternion=np.asarray(config["tcp_quaternion"], dtype=float),
    )
    kinematics = common.Pin(rcs.ROBOTS[RobotType.Panda].mjcf_model_path, "attachment")
    home_pose = kinematics.forward(home, tcp_offset)
    np.testing.assert_allclose(home_pose.translation(), [0.405322, 0.0, 0.599659], atol=1e-5)
    # The hardware config currently reports a conservative J4 safe lower bound
    # of about -155.4 degrees after the configured two-degree margin.
    assert config["home_joints_deg"][3] > -145.0
    assert MODULE.home_path_within_workspace(
        kinematics,
        home,
        home,
        tcp_offset,
        np.asarray(config["tool_box_min"], dtype=float),
        np.asarray(config["tool_box_max"], dtype=float),
        np.asarray(config["workspace_min"], dtype=float),
        np.asarray(config["workspace_max"], dtype=float),
    )


def test_safety_arguments_accept_new_absolute_output(tmp_path):
    device = tmp_path / "ttyUSB1"
    device.touch()
    ip, output, low, high, tool_low, tool_high, cameras = MODULE.validate_safety_args(
        make_args(tmp_path, gripper_device=str(device))
    )
    assert ip == "192.168.4.100"
    assert output == tmp_path / "new-session"
    assert np.all(low < high)
    assert np.all(tool_low <= 0) and np.all(tool_high >= 0)
    assert set(cameras) == {"wrist", "third_person"}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"confirm": "yes"}, "confirmation"),
        ({"step": 0.006}, "--step"),
        ({"speed_factor": 0.2}, "--speed-factor"),
        ({"frequency": 31}, "--frequency"),
        ({"gripper_speed": 60.0}, "--gripper-speed"),
        ({"gripper_force": 60.0}, "--gripper-force"),
        ({"workspace_min": [0.8, 0, 0]}, "workspace minimum"),
        ({"tool_box_min": [0.01, -0.1, -0.1]}, "tool box"),
        ({"joint_limit_margin_deg": 21.0}, "joint-limit-margin"),
        ({"home_joints_deg": [0.0] * 6}, "home-joints-deg"),
        ({"cameras": {"wrist": "ONLY"}}, "third_person"),
        ({"camera_fps": 61}, "camera-fps"),
        ({"tcp_quaternion": [0, 0, 0, 2]}, "normalized"),
    ],
)
def test_safety_arguments_reject_unsafe_values(tmp_path, override, message):
    device = tmp_path / "ttyUSB1"
    device.touch()
    with pytest.raises(ValueError, match=message):
        MODULE.validate_safety_args(make_args(tmp_path, gripper_device=str(device), **override))


def test_safety_arguments_reject_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    device = tmp_path / "ttyUSB1"
    device.touch()
    with pytest.raises(ValueError, match="existing output"):
        MODULE.validate_safety_args(make_args(tmp_path, output=str(output), gripper_device=str(device)))


def test_json_config_is_loaded_and_cli_overrides_it(tmp_path):
    config = tmp_path / "session.json"
    config.write_text(
        """{
            "robot_ip": "192.168.178.12",
            "gripper_serial": "SERIAL",
            "gripper_device": "/dev/ttyUSB1",
            "instruction": "pick",
            "output": "/tmp/new-session",
            "workspace_min": [0.2, -0.3, 0.1],
            "workspace_max": [0.7, 0.3, 0.8],
            "tool_box_min": [-0.05, -0.05, -0.15],
            "tool_box_max": [0.05, 0.05, 0.02],
            "tcp_translation": [0.0, 0.0, 0.15],
            "tcp_quaternion": [0.0, 0.0, 0.0, 1.0],
            "cameras": {"wrist": "W", "third_person": "T"}
        }""",
        encoding="utf-8",
    )
    args = MODULE.parse_args(
        [
            "--config",
            str(config),
            "--frequency",
            "10",
            "--confirm",
            "ENABLE-REAL-PANDA 192.168.178.12",
        ]
    )
    assert args.robot_ip == "192.168.178.12"
    assert args.frequency == 10
    assert args.cameras == {"wrist": "W", "third_person": "T"}
