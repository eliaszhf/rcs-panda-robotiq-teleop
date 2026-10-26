from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FRANKA_SOURCE = REPO_ROOT / "extensions" / "rcs_fr3" / "src" / "hw" / "Franka.cpp"


def test_osc_uses_starting_joint_state_as_nullspace_target() -> None:
    source = FRANKA_SOURCE.read_text(encoding="utf-8")

    assert "const franka::RobotState initial_state = this->curr_state.load();" in source
    assert "initial_state.q.data()" in source
    assert "static_q_task_ << 0.09017809387254755" not in source


def test_osc_limits_feedback_around_model_feedforward() -> None:
    source = FRANKA_SOURCE.read_text(encoding="utf-8")

    osc_source = source[source.index("void Franka::osc()") : source.index("void Franka::joint_controller()")]
    feedforward = osc_source.index("feedforward_tau = coriolis")
    feedback_limit = osc_source.index("TorqueSafetyGuardFn(feedback_tau_array, torque_limit)")
    total_torque = osc_source.index("feedback_tau_array.data(), 7) + feedforward_tau")

    assert feedforward < feedback_limit < total_torque
    assert "TorqueSafetyGuardFn(tau_d_rate_limited, torque_limit)" not in osc_source


def test_osc_can_add_software_only_gravity_model_difference() -> None:
    source = FRANKA_SOURCE.read_text(encoding="utf-8")

    assert "gravity_compensation_total_mass" in source
    assert "gravity_compensation_total_com" in source
    assert "corrected_gravity - gravity" in source
    assert "robot.setLoad(" not in source[source.index("void Franka::osc()") :]


def test_realtime_callbacks_do_not_block_on_shared_state_or_interpolator() -> None:
    source = FRANKA_SOURCE.read_text(encoding="utf-8")
    realtime_source = source[
        source.index("void Franka::osc()") : source.index("void Franka::move_home()")
    ]

    assert realtime_source.count("try_store(robot_state)") == 3
    assert realtime_source.count("this->interpolator_mutex, std::try_to_lock") == 3
    assert "this->curr_state.store(robot_state);" not in realtime_source


def test_osc_reports_realtime_timing_after_control_stops() -> None:
    source = FRANKA_SOURCE.read_text(encoding="utf-8")
    osc_source = source[source.index("void Franka::osc()") : source.index("void Franka::joint_controller()")]

    assert 'PrintRealtimeDiagnostics("osc", diagnostics);' in osc_source
    assert "callback_over_1000us" in source
    assert "robot_period_over_1500us" in source
    assert "state_store_skips" in source
    assert "target_lock_skips" in source


def test_joint_stream_uses_repository_torque_controller() -> None:
    source = FRANKA_SOURCE.read_text(encoding="utf-8")
    joint_source = source[
        source.index("void Franka::joint_controller()") : source.index(
            "void Franka::zero_torque_guiding()"
        )
    ]

    assert "-> franka::Torques" in joint_source
    assert "Kp.cwiseProduct" in joint_source
    assert "feedforward_tau = coriolis" in joint_source
    assert "corrected_gravity - gravity" in joint_source
    assert "TorqueSafetyGuardFn(feedback_tau_array, torque_limit)" in joint_source
    assert "TorqueSafetyGuardFn(tau_d_rate_limited, torque_limit)" not in joint_source
    assert "-> franka::JointPositions" not in joint_source


def test_joint_interpolator_uses_minimum_jerk_time_scaling() -> None:
    source = (REPO_ROOT / "include" / "rcs" / "LinearPoseTrajInterpolator.h").read_text(
        encoding="utf-8"
    )
    joint_interpolator = source[source.index("class LinearJointPositionTrajInterpolator") :]

    assert "const double smooth_t = t3 * (10. + t * (-15. + 6. * t));" in joint_interpolator
    assert "last_q_t_ = q_start_ + smooth_t * (q_goal_ - q_start_);" in joint_interpolator
