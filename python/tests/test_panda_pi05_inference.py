import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


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


def test_bounded_joint_target_rejects_large_model_jump_and_limits_small_step():
    current = np.zeros(7)
    limits = np.array([[-2.0] * 7, [2.0] * 7])
    proposed = np.full(7, np.deg2rad(1.0))
    target = MODULE.bounded_joint_target(
        current,
        proposed,
        limits,
        margin_radians=0.1,
        max_step_radians=np.deg2rad(0.5),
        max_model_delta_radians=np.deg2rad(5.0),
    )
    np.testing.assert_allclose(target, np.deg2rad(0.5))
    with pytest.raises(ValueError, match="too far"):
        MODULE.bounded_joint_target(
            current,
            np.full(7, np.deg2rad(6.0)),
            limits,
            margin_radians=0.1,
            max_step_radians=np.deg2rad(0.5),
            max_model_delta_radians=np.deg2rad(5.0),
        )


def test_model_gripper_conversion_matches_training_convention():
    assert MODULE.model_gripper_to_binary(1.0) == 1
    assert MODULE.model_gripper_to_binary(0.0) == 1
    assert MODULE.model_gripper_to_binary(-1.0) == 0


def test_openpi_msgpack_codec_round_trip_when_dependency_is_available():
    msgpack = pytest.importorskip("msgpack")
    del msgpack
    import pi05_msgpack_numpy

    payload = {"actions": np.zeros((20, 8), dtype=np.float32)}
    decoded = pi05_msgpack_numpy.unpackb(pi05_msgpack_numpy.packb(payload))
    np.testing.assert_array_equal(decoded["actions"], payload["actions"])
