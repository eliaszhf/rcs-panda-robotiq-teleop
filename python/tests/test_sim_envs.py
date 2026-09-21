import copy
import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
from rcs.camera.sim import SimCameraSet
from rcs.envs.base import (
    CameraSetWrapper,
    ControlMode,
    GripperDictType,
    GripperWrapper,
    JointsDictType,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
    SimEnv,
    TQuatDictType,
    TRPYDictType,
)
from rcs.envs.configs import EmptyWorldFR3, EmptyWorldPanda
from rcs.envs.sim import GripperWrapperSim, RobotSimWrapper

import rcs
from rcs import sim


def build_single_robot_env(
    control_mode: ControlMode,
    *,
    with_gripper: bool,
    with_camera: bool,
    max_relative_movement: float | tuple[float, float] | None,
) -> gym.Env:
    scene = EmptyWorldFR3()
    cfg = copy.deepcopy(scene.config())
    cfg.control_mode = control_mode
    cfg.headless = True
    cfg.sim_cfg.realtime = False
    cfg.sim_cfg.async_control = False
    if not with_gripper:
        cfg.gripper_cfgs = None
        cfg.gripper_offsets = None
    if not with_camera:
        cfg.camera_cfgs = None
        cfg.camera_adds = None
    cfg.max_relative_movement = max_relative_movement
    cfg.relative_to = RelativeTo.NONE if max_relative_movement is None else RelativeTo.LAST_STEP

    prefixed_cfg = scene.prefixed_cfg(cfg)
    robot_name = scene.lead_robot_name(prefixed_cfg)
    robot_cfg = prefixed_cfg.robot_cfgs[robot_name]
    mjmodel = scene.create_model(prefixed_cfg)
    simulation = sim.Sim(mjmodel, prefixed_cfg.sim_cfg)

    kinematic_model_path, attachment_site = scene.kinematics_cfg(prefixed_cfg)[robot_name]
    ik = rcs.common.Pin(
        kinematic_model_path,
        attachment_site,
    )

    env: gym.Env = SimEnv(simulation)
    robot = rcs.sim.SimRobot(simulation, ik, robot_cfg)
    env = RobotWrapper(env, robot, control_mode)

    if prefixed_cfg.gripper_cfgs is not None:
        gripper = sim.SimGripper(simulation, prefixed_cfg.gripper_cfgs[robot_name])
        env = GripperWrapper(env, gripper)

    env = RobotSimWrapper(env)

    if prefixed_cfg.gripper_cfgs is not None:
        env = GripperWrapperSim(env)

    if prefixed_cfg.camera_cfgs is not None:
        camera_set = SimCameraSet(simulation, prefixed_cfg.camera_cfgs, physical_units=True, render_on_demand=True)
        env = CameraSetWrapper(env, camera_set, include_depth=True)  # type: ignore[arg-type]

    if max_relative_movement is not None:
        env = RelativeActionSpace(env, max_mov=max_relative_movement, relative_to=RelativeTo.LAST_STEP)

    return env


class TestSimEnvs:
    """This class is for testing common sim env functionalities"""

    def assert_no_pose_change(self, info: dict, initial_obs: dict, final_obs: dict):
        assert info["ik_success"]
        out_pose = rcs.common.Pose(
            translation=np.array(final_obs["tquat"][:3]), quaternion=np.array(final_obs["tquat"][3:])
        )
        expected_pose = rcs.common.Pose(
            translation=np.array(initial_obs["tquat"][:3]), quaternion=np.array(initial_obs["tquat"][3:])
        )
        assert out_pose.is_close(expected_pose, 1e-1, 1e-2)

    def assert_collision(self, info: dict):
        assert info["ik_success"]
        assert info["collision"]


class TestSimEnvsTRPY(TestSimEnvs):
    """This class is for testing TRPY sim env functionalities"""

    def test_reset(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TRPY,
            with_gripper=True,
            with_camera=True,
            max_relative_movement=None,
        )
        env.reset()
        env.reset()

    def test_zero_action_trpy(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TRPY,
            with_gripper=False,
            with_camera=False,
            max_relative_movement=None,
        )
        obs_initial, _ = env.reset()
        zero_action = TRPYDictType(xyzrpy=obs_initial["xyzrpy"])
        obs, _, _, _, info = env.step(zero_action)
        self.assert_no_pose_change(info, obs_initial, obs)

    def test_non_zero_action_trpy(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TRPY,
            with_gripper=False,
            with_camera=False,
            max_relative_movement=None,
        )
        obs_initial, _ = env.reset()
        x_pos_change = 0.2
        initial_tquat = obs_initial["tquat"].copy()
        t = initial_tquat[:3]
        t[0] += x_pos_change
        q = initial_tquat[3:]
        initial_pose = rcs.common.Pose(translation=np.array(t), quaternion=np.array(q))
        xyzrpy = np.concatenate([t, initial_pose.rotation_rpy().as_vector()], axis=0)
        non_zero_action = TRPYDictType(xyzrpy=np.array(xyzrpy))
        non_zero_action.update(GripperDictType(gripper=np.array([0.0])))
        expected_obs = obs_initial.copy()
        expected_obs["tquat"][0] += x_pos_change
        obs, _, _, _, info = env.step(non_zero_action)
        self.assert_no_pose_change(info, expected_obs, obs)

    def test_relative_zero_action_trpy(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TRPY,
            with_gripper=True,
            with_camera=False,
            max_relative_movement=0.5,
        )
        obs_initial, _ = env.reset()
        zero_action = TRPYDictType(xyzrpy=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))  # type: ignore
        zero_action.update(GripperDictType(gripper=np.array([0.0])))
        obs, _, _, _, info = env.step(zero_action)
        self.assert_no_pose_change(info, obs_initial, obs)

    def test_relative_non_zero_action(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TRPY,
            with_gripper=True,
            with_camera=False,
            max_relative_movement=0.5,
        )
        obs_initial, _ = env.reset()
        x_pos_change = 0.2
        non_zero_action = TRPYDictType(xyzrpy=np.array([x_pos_change, 0, 0, 0, 0, 0]))  # type: ignore
        non_zero_action.update(GripperDictType(gripper=np.array([0.0])))
        expected_obs = obs_initial.copy()
        expected_obs["tquat"][0] += x_pos_change
        obs, _, _, _, info = env.step(non_zero_action)
        self.assert_no_pose_change(info, obs_initial, expected_obs)

    def test_collision_trpy(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TRPY,
            with_gripper=True,
            with_camera=False,
            max_relative_movement=None,
        )
        obs, _ = env.reset()
        obs["xyzrpy"][0] = 0.4
        obs["xyzrpy"][2] = -0.05
        collision_action = TRPYDictType(xyzrpy=obs["xyzrpy"])
        collision_action.update(GripperDictType(gripper=np.array([0.0])))
        _, _, _, _, info = env.step(collision_action)
        self.assert_collision(info)


class TestSimEnvsTquat(TestSimEnvs):
    """This class is for testing Tquat sim env functionalities"""

    def test_reset(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TQuat,
            with_gripper=True,
            with_camera=True,
            max_relative_movement=None,
        )
        env.reset()
        env.reset()

    def test_non_zero_action_tquat(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TQuat,
            with_gripper=False,
            with_camera=False,
            max_relative_movement=None,
        )
        obs_initial, _ = env.reset()
        t = obs_initial["tquat"][:3]
        q = obs_initial["tquat"][3:]
        x_pos_change = 0.3
        t[0] += x_pos_change
        non_zero_action = TQuatDictType(tquat=np.concatenate([t, q], axis=0))
        expected_obs = obs_initial.copy()
        expected_obs["tquat"][0] += x_pos_change
        _, _, _, _, info = env.step(non_zero_action)
        self.assert_no_pose_change(info, obs_initial, expected_obs)

    def test_zero_action_tquat(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TQuat,
            with_gripper=False,
            with_camera=False,
            max_relative_movement=None,
        )
        obs_initial, _ = env.reset()
        zero_action = TQuatDictType(tquat=obs_initial["tquat"])
        obs, _, _, _, info = env.step(zero_action)
        self.assert_no_pose_change(info, obs_initial, obs)

    def test_relative_zero_action_tquat(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TQuat,
            with_gripper=True,
            with_camera=False,
            max_relative_movement=0.5,
        )
        obs_initial, _ = env.reset()
        zero_rel_action = TQuatDictType(tquat=np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32))  # type: ignore
        zero_rel_action.update(GripperDictType(gripper=np.array([0.0])))
        obs, _, _, _, info = env.step(zero_rel_action)
        self.assert_no_pose_change(info, obs_initial, obs)

    def test_collision_tquat(self):
        env = build_single_robot_env(
            ControlMode.CARTESIAN_TQuat,
            with_gripper=True,
            with_camera=False,
            max_relative_movement=None,
        )
        obs, _ = env.reset()
        obs["tquat"][0] = 0.4
        obs["tquat"][2] = -0.05
        collision_action = TQuatDictType(tquat=obs["tquat"])
        collision_action.update(GripperDictType(gripper=np.array([0.0])))
        _, _, _, _, info = env.step(collision_action)
        self.assert_collision(info)


class TestSimEnvsJoints(TestSimEnvs):
    """This class is for testing Joints sim env functionalities"""

    def test_reset(self):
        env = build_single_robot_env(
            ControlMode.JOINTS,
            with_gripper=True,
            with_camera=True,
            max_relative_movement=None,
        )
        env.reset()
        env.reset()

    def test_zero_action_joints(self):
        env = build_single_robot_env(
            ControlMode.JOINTS,
            with_gripper=False,
            with_camera=False,
            max_relative_movement=None,
        )
        obs_initial, _ = env.reset()
        zero_action = JointsDictType(joints=np.array(obs_initial["joints"]))
        obs, _, _, _, info = env.step(zero_action)
        assert info["ik_success"]
        assert np.allclose(obs["joints"], obs_initial["joints"], atol=0.01, rtol=0)

    def test_non_zero_action_joints(self):
        env = build_single_robot_env(
            ControlMode.JOINTS,
            with_gripper=False,
            with_camera=False,
            max_relative_movement=None,
        )
        obs_initial, _ = env.reset()
        new_joint_vals = obs_initial["joints"] + np.array([0.1, 0.1, 0.1, 0.1, -0.1, -0.1, 0.1], dtype=np.float32)
        non_zero_action = JointsDictType(joints=new_joint_vals)
        obs, _, _, _, info = env.step(non_zero_action)
        assert info["ik_success"]
        assert np.allclose(obs["joints"], non_zero_action["joints"], atol=0.01, rtol=0)

    def test_collision_joints(self):
        env = build_single_robot_env(
            ControlMode.JOINTS,
            with_gripper=True,
            with_camera=False,
            max_relative_movement=None,
        )
        env.reset()
        collision_act = JointsDictType(joints=np.array([0, 1.78, 0, -1.45, 0, 0, 0], dtype=np.float32))
        collision_act.update(GripperDictType(gripper=np.array([1.0])))
        _, _, _, _, info = env.step(collision_act)
        self.assert_collision(info)

    def test_relative_zero_action_joints(self):
        env = build_single_robot_env(
            ControlMode.JOINTS,
            with_gripper=True,
            with_camera=False,
            max_relative_movement=0.5,
        )
        obs_initial, _ = env.reset()
        act = JointsDictType(joints=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        act.update(GripperDictType(gripper=np.array([1.0])))
        obs, _, _, _, info = env.step(act)
        self.assert_no_pose_change(info, obs_initial, obs)


def test_panda_robotiq_relative_cartesian_control():
    scene = EmptyWorldPanda()
    cfg = scene.config()
    cfg.headless = True
    cfg.sim_cfg.realtime = False
    env = scene.create_env(cfg)

    try:
        assert gym.spec("rcs/panda") is not None
        observation, info = env.reset()
        assert set(observation) == {"right"}
        assert info["right"]["robot_type"] == "Panda"
        assert info["right"]["gripper_type"] == "Robotiq2F85"
        assert set(env.action_space.spaces["right"].spaces) == {"tquat", "gripper"}

        moves = (
            (0.005, 0.0, 0.0),
            (-0.005, 0.0, 0.0),
            (0.0, 0.005, 0.0),
            (0.0, -0.005, 0.0),
            (0.0, 0.0, 0.005),
            (0.0, 0.0, -0.005),
        )
        robot = env.get_wrapper_attr("robot")["right"]
        for delta in moves:
            tcp_before = robot.get_cartesian_position().translation()
            action = {"right": {"tquat": [*delta, 0.0, 0.0, 0.0, 1.0], "gripper": [1]}}
            observation, _, terminated, truncated, info = env.step(action)
            tcp_after = robot.get_cartesian_position().translation()
            assert np.all(np.isfinite(observation["right"]["tquat"]))
            assert np.dot(tcp_after - tcp_before, np.asarray(delta)) > 0
            assert info["right"]["ik_success"]
            assert not terminated and not truncated

        for gripper_state in (0, 1):
            action = {
                "right": {
                    "tquat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "gripper": [gripper_state],
                }
            }
            observation, _, terminated, truncated, _ = env.step(action)
            assert observation["right"]["gripper"] == [float(gripper_state)]
            assert not terminated and not truncated
    finally:
        env.close()


def test_panda_keyboard_mapping():
    script = Path(__file__).parents[2] / "examples" / "panda" / "panda_keyboard_teleop.py"
    spec = importlib.util.spec_from_file_location("panda_keyboard_teleop", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    step_meters = module.STEP_METERS
    command_for_key = module.command_for_key

    expected_deltas = {
        "w": [step_meters, 0.0, 0.0],
        "s": [-step_meters, 0.0, 0.0],
        "a": [0.0, step_meters, 0.0],
        "d": [0.0, -step_meters, 0.0],
        "r": [0.0, 0.0, step_meters],
        "f": [0.0, 0.0, -step_meters],
    }
    for key, expected_delta in expected_deltas.items():
        action, state = command_for_key(key, 1)
        assert action is not None
        assert action["right"]["tquat"][:3] == expected_delta
        assert action["right"]["tquat"][3:] == [0.0, 0.0, 0.0, 1.0]
        assert state == 1

    open_action, open_state = command_for_key("q", 0)
    close_action, close_state = command_for_key("e", 1)
    ignored_action, ignored_state = command_for_key("?", close_state)
    assert open_action is not None and open_action["right"]["gripper"] == [1]
    assert open_state == 1
    assert close_action is not None and close_action["right"]["gripper"] == [0]
    assert close_state == 0
    assert ignored_action is None and ignored_state == 0
