"""Terminal keyboard teleoperation for a simulated Panda and Robotiq 2F-85."""

import contextlib
import select
import sys
import termios
import time
import tty
from collections.abc import Iterator

from rcs._core.sim import SimConfig
from rcs.envs.configs import EmptyWorldPanda


STEP_METERS = 0.005
KEY_DELTAS = {
    "w": (STEP_METERS, 0.0, 0.0),
    "s": (-STEP_METERS, 0.0, 0.0),
    "a": (0.0, STEP_METERS, 0.0),
    "d": (0.0, -STEP_METERS, 0.0),
    "r": (0.0, 0.0, STEP_METERS),
    "f": (0.0, 0.0, -STEP_METERS),
}


def command_for_key(key: str, gripper_state: int) -> tuple[dict | None, int]:
    """Convert one key into an RCS relative action and updated gripper state."""
    key = key.lower()
    if key == "q":
        gripper_state = 1
    elif key == "e":
        gripper_state = 0
    elif key not in KEY_DELTAS:
        return None, gripper_state

    delta = KEY_DELTAS.get(key, (0.0, 0.0, 0.0))
    action = {
        "right": {
            "tquat": [*delta, 0.0, 0.0, 0.0, 1.0],
            "gripper": [gripper_state],
        }
    }
    return action, gripper_state


@contextlib.contextmanager
def raw_terminal() -> Iterator[None]:
    """Read single keys and always restore the terminal settings."""
    if not sys.stdin.isatty():
        raise RuntimeError("Keyboard teleoperation requires an interactive terminal.")
    file_descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        yield
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous)


def create_env():
    scene = EmptyWorldPanda()
    cfg = scene.config()
    cfg.headless = False
    cfg.sim_cfg = SimConfig(async_control=False, realtime=True, frequency=30, max_convergence_steps=2000)
    return scene.create_env(cfg)


def main() -> None:
    env = create_env()
    gripper_state = 1
    try:
        observation, info = env.reset()
        robot = env.get_wrapper_attr("robot")["right"]
        time.sleep(1.0)
        print("W/S: x  A/D: y  R/F: z  Q: open  E: close  ESC: quit")
        print("initial TCP:", robot.get_cartesian_position().translation())
        print("reset info:", info["right"])

        with raw_terminal():
            while True:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                key = sys.stdin.read(1)
                if key == "\x1b":
                    break

                action, gripper_state = command_for_key(key, gripper_state)
                if action is None:
                    continue
                observation, reward, terminated, truncated, info = env.step(action)
                print(
                    f"key={key.lower()} TCP={robot.get_cartesian_position().translation()} "
                    f"gripper={observation['right']['gripper']} reward={reward}"
                )
                if terminated or truncated:
                    print("environment stopped:", info)
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
