"""Small Cartesian-control demo for a simulated Panda and Robotiq 2F-85."""

import argparse

import numpy as np
from rcs._core.sim import SimConfig
from rcs.envs.configs import EmptyWorldPanda


STEP_METERS = 0.005
MOVES = (
    ("x+", (STEP_METERS, 0.0, 0.0)),
    ("x-", (-STEP_METERS, 0.0, 0.0)),
    ("y+", (0.0, STEP_METERS, 0.0)),
    ("y-", (0.0, -STEP_METERS, 0.0)),
    ("z+", (0.0, 0.0, STEP_METERS)),
    ("z-", (0.0, 0.0, -STEP_METERS)),
)


def create_env(*, headless: bool):
    scene = EmptyWorldPanda()
    cfg = scene.config()
    cfg.headless = headless
    cfg.sim_cfg = SimConfig(
        async_control=False,
        realtime=not headless,
        frequency=30,
        max_convergence_steps=2000,
    )
    return scene.create_env(cfg)


def make_action(delta: tuple[float, float, float], gripper: int) -> dict:
    return {
        "right": {
            "tquat": [*delta, 0.0, 0.0, 0.0, 1.0],
            "gripper": [gripper],
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    args = parser.parse_args()

    env = create_env(headless=args.headless)
    try:
        observation, info = env.reset()
        print("observation keys:", sorted(observation["right"]))
        print("reset info:", info["right"])

        robot = env.get_wrapper_attr("robot")["right"]
        print("initial TCP:", robot.get_cartesian_position().translation())

        for label, delta in MOVES:
            observation, reward, terminated, truncated, info = env.step(make_action(delta, gripper=1))
            tcp = robot.get_cartesian_position().translation()
            if not np.all(np.isfinite(observation["right"]["tquat"])):
                raise RuntimeError("Non-finite Panda observation received.")
            print(
                label,
                "TCP:",
                tcp,
                "reward:",
                reward,
                "converged:",
                info["right"]["is_sim_converged"],
                "terminated:",
                terminated,
                "truncated:",
                truncated,
            )
            if terminated or truncated:
                raise RuntimeError(f"Environment stopped during {label}: {info}")

        for label, gripper in (("close", 0), ("open", 1)):
            observation, reward, terminated, truncated, info = env.step(make_action((0.0, 0.0, 0.0), gripper))
            print("gripper", label, observation["right"]["gripper"], "reward:", reward)
            if terminated or truncated:
                raise RuntimeError(f"Environment stopped during gripper {label}: {info}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
