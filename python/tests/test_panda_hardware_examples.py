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

EXAMPLE_CONFIG = Path(__file__).parents[2] / "examples" / "panda" / "panda_hardware_session.example.json"


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
    obs = {
        "right": {
            "robot_state": {
                "joint_collision": np.array([0, 0, 1, 0, 0, 0, 0]),
                "cartesian_collision": np.zeros(6),
            }
        }
    }
    assert MODULE.collision_report(obs) == ["joint_collision=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]"]


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
