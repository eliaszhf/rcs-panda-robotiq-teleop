"""Calculate a small provisional workspace around a measured Robotiq TCP pose.

This is an offline calculation: it does not import a hardware extension, open a
driver, or command motion. The result is only valid after a person has verified
that the requested clearance is physically free in every direction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from panda_hardware_keyboard_collect import tool_corners_in_base


def workspace_around_pose(
    tcp_tquat: np.ndarray,
    tool_box_min: np.ndarray,
    tool_box_max: np.ndarray,
    negative_travel: np.ndarray,
    positive_travel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounds allowing the oriented tool the requested TCP travel."""
    negative_travel = np.asarray(negative_travel, dtype=float)
    positive_travel = np.asarray(positive_travel, dtype=float)
    if (
        negative_travel.shape != (3,)
        or positive_travel.shape != (3,)
        or not np.all(np.isfinite([negative_travel, positive_travel]))
        or np.any(negative_travel < 0)
        or np.any(positive_travel < 0)
    ):
        raise ValueError("travel values must be three finite, non-negative distances")

    corners = tool_corners_in_base(tcp_tquat, tool_box_min, tool_box_max)
    return corners.min(axis=0) - negative_travel, corners.max(axis=0) + positive_travel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Session JSON containing tool_box_min/max.")
    parser.add_argument(
        "--tcp-tquat",
        nargs=7,
        type=float,
        required=True,
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="Measured Robotiq TCP pose in the Panda base frame.",
    )
    parser.add_argument(
        "--negative-travel",
        nargs=3,
        type=float,
        default=(0.02, 0.02, 0.01),
        metavar=("X", "Y", "Z"),
        help="Allowed TCP travel in each negative direction (m).",
    )
    parser.add_argument(
        "--positive-travel",
        nargs=3,
        type=float,
        default=(0.02, 0.02, 0.01),
        metavar=("X", "Y", "Z"),
        help="Allowed TCP travel in each positive direction (m).",
    )
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        tool_box_min = np.asarray(config["tool_box_min"], dtype=float)
        tool_box_max = np.asarray(config["tool_box_max"], dtype=float)
        workspace_min, workspace_max = workspace_around_pose(
            np.asarray(args.tcp_tquat, dtype=float),
            tool_box_min,
            tool_box_max,
            np.asarray(args.negative_travel, dtype=float),
            np.asarray(args.positive_travel, dtype=float),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Cannot calculate workspace: {error}") from error

    print("PROVISIONAL ONLY: verify all requested travel and tool clearance physically.")
    print(json.dumps({"workspace_min": workspace_min.tolist(), "workspace_max": workspace_max.tolist()}, indent=2))


if __name__ == "__main__":
    main()
