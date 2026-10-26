import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from rcs import common


EXAMPLE_DIR = Path(__file__).parents[2] / "examples" / "panda"
sys.path.insert(0, str(EXAMPLE_DIR))
SPEC = importlib.util.spec_from_file_location(
    "panda_pi05_inference", EXAMPLE_DIR / "panda_pi05_inference.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validate_action_chunk_requires_twenty_finite_eight_dimensional_actions():
    actions = np.zeros((20, 8), dtype=np.float32)
    assert MODULE.validate_action_chunk({"actions": actions}).shape == (20, 8)
    with pytest.raises(ValueError, match="shape"):
        MODULE.validate_action_chunk({"actions": np.zeros((19, 8))})
    actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        MODULE.validate_action_chunk({"actions": actions})


def test_bounded_joint_target_limits_every_finite_model_step():
    current = np.zeros(7)
    limits = np.array([[-2.0] * 7, [2.0] * 7])
    proposed = np.full(7, np.deg2rad(1.0))
    target = MODULE.bounded_joint_target(
        current,
        proposed,
        limits,
        margin_radians=0.1,
        max_step_radians=np.deg2rad(0.5),
    )
    np.testing.assert_allclose(target, np.deg2rad(0.5))
    large_target = MODULE.bounded_joint_target(
        current,
        np.full(7, np.deg2rad(60.0)),
        limits,
        margin_radians=0.1,
        max_step_radians=np.deg2rad(0.5),
    )
    np.testing.assert_allclose(large_target, np.deg2rad(0.5))


def test_bounded_joint_target_preserves_coordinated_joint_direction():
    current = np.zeros(7)
    proposed = np.deg2rad(np.array([1.0, -0.5, 0.25, 0.0, -0.75, 0.1, -0.2]))
    target = MODULE.bounded_joint_target(
        current,
        proposed,
        np.array([[-2.0] * 7, [2.0] * 7]),
        margin_radians=0.1,
        max_step_radians=np.deg2rad(0.1),
    )

    np.testing.assert_allclose(target, proposed * 0.1)


def test_configured_origin_tquat_action_matches_fixed_origin_semantics():
    origin = np.array([0.4, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0])
    half_angle = np.deg2rad(1.0) / 2.0
    target = np.array(
        [0.401, -0.102, 0.3005, 0.0, 0.0, np.sin(half_angle), np.cos(half_angle)]
    )

    action = MODULE.referenced_tquat_action(origin, target)

    np.testing.assert_allclose(action[:3], target[:3] - origin[:3])
    assert MODULE.quaternion_angle_degrees(action[3:], target[3:]) < 1e-6


def test_model_gripper_conversion_matches_training_convention():
    assert MODULE.model_gripper_to_binary(1.0) == 1
    assert MODULE.model_gripper_to_binary(0.0) == 1
    assert MODULE.model_gripper_to_binary(-0.01) == 0
    assert MODULE.model_gripper_to_binary(-1.0) == 0


def test_absolute_tcp_from_joint_target_uses_forward_kinematics():
    class FakeKinematics:
        def __init__(self):
            self.call = None

        def forward(self, joints):
            self.call = joints.copy()
            half_yaw = np.deg2rad(MODULE.PANDA_MJCF_ATTACHMENT_YAW_DEGREES) / 2.0
            return common.Pose(
                translation=np.array([0.4, -0.1, 0.3]),
                quaternion=np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)]),
            )

    kinematics = FakeKinematics()
    joints = np.arange(7, dtype=float) / 10.0
    tcp_offset = common.Pose()
    target = MODULE.absolute_tcp_from_joint_target(kinematics, joints, tcp_offset)

    np.testing.assert_allclose(target, [0.4, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(kinematics.call, joints)


def test_absolute_tcp_from_joint_target_rejects_invalid_quaternion():
    class FakePose:
        def __mul__(self, other):
            del other
            return self

        def translation(self):
            return np.zeros(3)

        def rotation_q(self):
            return np.zeros(4)

    class FakeKinematics:
        def forward(self, joints):
            del joints
            return FakePose()

    with pytest.raises(ValueError, match="non-unit quaternion"):
        MODULE.absolute_tcp_from_joint_target(FakeKinematics(), np.zeros(7), common.Pose())


def test_joint_to_tcp_matches_post_estop_hardware_snapshot():
    joints = np.array(
        [-0.00700954018, -0.272539328, 0.000162466713, -2.26667852, -0.0288918172, 1.87674086, 1.42791393]
    )
    measured_tcp = np.array(
        [0.43574548, -0.00927963, 0.31224843, 0.75553988, -0.65233097, -0.03564827, 0.04850762]
    )
    kinematics = common.Pin("assets/robots/panda/panda.xml", "attachment", urdf=False)
    tcp_offset = common.Pose(
        translation=np.array([0.0, 0.0, 0.1493]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
    )

    fk_tcp = MODULE.absolute_tcp_from_joint_target(kinematics, joints, tcp_offset)

    assert np.linalg.norm(fk_tcp[:3] - measured_tcp[:3]) < 5e-6
    assert MODULE.quaternion_angle_degrees(fk_tcp[3:], measured_tcp[3:]) < 0.001


def test_fk_validation_rejects_tcp_frame_mismatch():
    measured = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    bad_fk = measured.copy()
    bad_fk[2] += 0.3
    with pytest.raises(RuntimeError, match="FK TCP disagrees"):
        MODULE.validate_fk_matches_measured(bad_fk, measured)


def test_tcp_step_guard_rejects_large_translation_and_accepts_small_step():
    measured = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    target = measured.copy()
    target[2] += 0.002
    translation, rotation = MODULE.validate_tcp_step(measured, target, 0.005, 1.0)
    assert translation == pytest.approx(0.002)
    assert rotation == pytest.approx(0.0)

    target[2] += 0.01
    with pytest.raises(ValueError, match="Cartesian safety limit"):
        MODULE.validate_tcp_step(measured, target, 0.005, 1.0)


def test_cartesian_joint_scaling_preserves_direction_and_enforces_translation_limit():
    class LinearKinematics:
        def forward(self, joints):
            half_yaw = np.deg2rad(MODULE.PANDA_MJCF_ATTACHMENT_YAW_DEGREES) / 2.0
            return common.Pose(
                translation=np.array([0.02 * joints[0], 0.0, 0.0]),
                quaternion=np.array(
                    [0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)]
                ),
            )

    current = np.zeros(7)
    target = np.array([0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    measured_tcp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    scaled, _, scale, translation, rotation = MODULE.bound_joint_target_to_tcp_step(
        current,
        target,
        measured_tcp,
        LinearKinematics(),
        common.Pose(),
        0.005,
        1.0,
    )

    assert 0.49 < scale < 0.5
    np.testing.assert_allclose(scaled, target * scale)
    assert translation < 0.005
    assert rotation < 1e-6


def test_cartesian_outcome_keeps_absolute_motion_limits():
    start = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    measured = start.copy()
    measured[2] += 0.0051

    with pytest.raises(RuntimeError, match="absolute step limit"):
        MODULE.validate_cartesian_outcome(
            start,
            measured,
            max_achieved_translation_meters=0.005,
            max_achieved_rotation_degrees=1.0,
        )


def test_configured_origin_hold_repeats_zero_offset_and_accepts_stationary_robot(monkeypatch):
    class FakePose:
        def translation(self):
            return np.array([0.4, 0.0, 0.3])

        def rotation_q(self):
            return np.array([0.0, 0.0, 0.0, 1.0])

    class FakeRobotState:
        q = np.zeros(7)

    class FakeState:
        robot_state = FakeRobotState()

    class FakeRobot:
        def get_state(self):
            return FakeState()

        def get_cartesian_position(self):
            return FakePose()

    class FakeEnv:
        def __init__(self):
            self.actions = []

        def step(self, action):
            self.actions.append(action)
            return {}, 0.0, False, False, {}

    robot = FakeRobot()
    env = FakeEnv()
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    result = MODULE.run_configured_origin_hold_check(
        env, robot, gripper_state=1, frequency=20.0, duration_seconds=0.25
    )

    assert len(env.actions) == 5
    for action in env.actions:
        np.testing.assert_allclose(
            action["right"]["tquat"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )
        assert action["right"]["gripper"] == [1]
    assert result == pytest.approx((0.0, 0.0, 0.0))


def test_hold_check_only_returns_after_the_physical_check():
    source = (EXAMPLE_DIR / "panda_pi05_inference.py").read_text(encoding="utf-8")
    hold_block = source[source.index("hold_confirmation =") : source.index("client.reset()")]

    assert hold_block.index("run_configured_origin_hold_check(") < hold_block.index(
        "if args.hold_check_only:"
    )


def test_policy_step_waits_for_full_settle_before_outcome_audit():
    source = (EXAMPLE_DIR / "panda_pi05_inference.py").read_text(encoding="utf-8")
    step_block = source[source.index("observation, _, terminated") : source.index("collisions =")]

    assert step_block.index("time.sleep(remaining)") < step_block.index(
        "wait_for_velocity_settle("
    )
    assert step_block.index("wait_for_velocity_settle(") < step_block.index(
        "refresh_hardware_observation("
    )
    assert step_block.index("refresh_hardware_observation(") < step_block.index(
        "validate_cartesian_outcome("
    )


def test_velocity_settle_keeps_strict_threshold_with_faster_polling():
    source = (EXAMPLE_DIR / "panda_pi05_inference.py").read_text(encoding="utf-8")
    settle = source[
        source.index("def wait_for_velocity_settle(") : source.index(
            "def run_configured_origin_hold_check("
        )
    ]

    assert "velocity_tolerance: float = 0.01" in settle
    assert "stable_samples: int = 3" in settle
    assert "sample_interval_seconds: float = 0.025" in settle
    assert "time.sleep(sample_interval_seconds)" in settle


def test_policy_loop_prefetches_chunks_without_sync_inference():
    source = (EXAMPLE_DIR / "panda_pi05_inference.py").read_text(encoding="utf-8")
    loop = source[source.index("with ThreadPoolExecutor") : source.index("finally:")]

    assert "executor.submit(" in loop
    assert "next_actions.result()" in loop
    assert "client.infer(" not in loop


def test_hardware_policy_uses_absolute_joint_control_for_joint_checkpoint():
    source = (EXAMPLE_DIR / "panda_pi05_inference.py").read_text(encoding="utf-8")
    creator = source[source.index("def create_joint_hardware_env") : source.index("def absolute_tcp")]

    assert "cfg.control_mode = ControlMode.JOINTS" in creator
    assert "cfg.relative_to = RelativeTo.NONE" in creator
    assert '"joints": target' in source
    assert '"tquat": referenced_tcp_action' not in source


def test_hardware_policy_keeps_bounded_osc_stiffness_for_compatibility():
    source = (EXAMPLE_DIR / "panda_pi05_inference.py").read_text(encoding="utf-8")
    creator = source[source.index("def create_joint_hardware_env") : source.index("def absolute_tcp")]

    assert 'config.get("osc_translation_stiffness"' in creator
    assert "np.any(osc_translation_stiffness > 500.0)" in creator
    assert "robot_cfg.kp_p = osc_translation_stiffness" in creator


def test_openpi_msgpack_codec_round_trip_when_dependency_is_available():
    msgpack = pytest.importorskip("msgpack")
    del msgpack
    import pi05_msgpack_numpy

    payload = {"actions": np.zeros((20, 8), dtype=np.float32)}
    decoded = pi05_msgpack_numpy.unpackb(pi05_msgpack_numpy.packb(payload))
    np.testing.assert_array_equal(decoded["actions"], payload["actions"])
