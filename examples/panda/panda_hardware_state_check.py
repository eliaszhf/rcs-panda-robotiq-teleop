"""Connect to a Panda and print one state snapshot without commanding motion.

WARNING: constructing the repository's Franka driver writes its default
collision/impedance/guiding parameters to the controller.  It does not request
robot motion, but this is therefore not a byte-for-byte read-only operation.
"""

from __future__ import annotations

import argparse
import ipaddress

import numpy as np


CONFIRM_PREFIX = "CONNECT-NO-MOTION"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True, help="Panda FCI IP address.")
    parser.add_argument(
        "--confirm",
        required=True,
        help=f"Must equal '{CONFIRM_PREFIX} <robot-ip>'.",
    )
    args = parser.parse_args()
    robot_ip = str(ipaddress.ip_address(args.robot_ip))
    expected = f"{CONFIRM_PREFIX} {robot_ip}"
    if args.confirm != expected:
        raise SystemExit(f"Refusing connection. Pass exactly: --confirm '{expected}'")

    try:
        from rcs_panda._core import hw
    except ImportError as error:
        raise SystemExit("rcs_panda is not installed in this environment.") from error

    import rcs
    from rcs import common

    cfg = hw.PandaConfig(ip=robot_ip)
    # Match DROIDEnv/the collector: report the Robotiq TCP, not the bare flange.
    cfg.tcp_offset = rcs.GRIPPER_TCP_OFFSETS[common.GripperType("Robotiq2F85")]
    cfg.ignore_realtime = False
    robot = None
    try:
        robot = hw.Franka(cfg)
        state = robot.get_state().robot_state
        tcp = robot.get_cartesian_position()
        fields = {
            "q": np.asarray(state.q),
            "dq": np.asarray(state.dq),
            "tau_J": np.asarray(state.tau_J),
            "tau_ext_hat_filtered": np.asarray(state.tau_ext_hat_filtered),
            "joint_contact": np.asarray(state.joint_contact),
            "joint_collision": np.asarray(state.joint_collision),
            "cartesian_contact": np.asarray(state.cartesian_contact),
            "cartesian_collision": np.asarray(state.cartesian_collision),
            "control_command_success_rate": float(state.control_command_success_rate),
            "m_ee": float(state.m_ee),
            "F_x_Cee": np.asarray(state.F_x_Cee),
            "I_ee": np.asarray(state.I_ee).reshape(3, 3, order="F"),
            "F_T_EE": np.asarray(state.F_T_EE).reshape(4, 4, order="F"),
            "m_load": float(state.m_load),
            "F_x_Cload": np.asarray(state.F_x_Cload),
            "I_load": np.asarray(state.I_load).reshape(3, 3, order="F"),
            "m_total": float(state.m_total),
            "tcp_offset_translation": np.asarray(cfg.tcp_offset.translation()),
            "tcp_translation": np.asarray(tcp.translation()),
            "tcp_quaternion": np.asarray(tcp.rotation_q()),
        }
        for name, value in fields.items():
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"Non-finite values in {name}: {value}")
            print(f"{name}: {value}")
        print(f"robot_mode: {state.robot_mode}")
        print("State snapshot complete; no motion method was called.")
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    main()
