from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FRANKA_SOURCE = REPO_ROOT / "extensions" / "rcs_fr3" / "src" / "hw" / "Franka.cpp"


def test_osc_uses_starting_joint_state_as_nullspace_target() -> None:
    source = FRANKA_SOURCE.read_text(encoding="utf-8")

    assert "const franka::RobotState initial_state = this->curr_state.load();" in source
    assert "initial_state.q.data()" in source
    assert "static_q_task_ << 0.09017809387254755" not in source


def test_osc_adds_model_coriolis_before_safety_limiting() -> None:
    source = FRANKA_SOURCE.read_text(encoding="utf-8")

    coriolis_add = source.index("tau_d += coriolis;")
    torque_limit = source.index("TorqueSafetyGuardFn(tau_d_rate_limited, torque_limit);", coriolis_add)
    assert coriolis_add < torque_limit
