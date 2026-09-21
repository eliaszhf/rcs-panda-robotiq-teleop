"""Read-only host/network/USB checks for a Panda + Robotiq workstation.

This script never imports a robot driver and never opens a serial port.  It is
safe to use before the hardware extensions are installed.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import platform
import resource
import shutil
import subprocess
from pathlib import Path


def run(command: list[str]) -> dict[str, object]:
    """Run a diagnostic command without a shell and return printable results."""
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def parse_udev_properties(output: str) -> dict[str, str]:
    wanted = {"ID_VENDOR_ID", "ID_MODEL_ID", "ID_SERIAL", "ID_SERIAL_SHORT", "ID_USB_DRIVER"}
    properties = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in wanted:
            properties[key] = value
    return properties


def serial_devices() -> list[dict[str, object]]:
    """List stable serial symlinks without opening any device."""
    by_id = Path("/dev/serial/by-id")
    if not by_id.is_dir():
        return []
    devices = []
    for link in sorted(by_id.iterdir()):
        try:
            target = link.resolve(strict=True)
        except FileNotFoundError:
            continue
        device: dict[str, object] = {
            "by_id": str(link),
            "device": str(target),
            "permissions": oct(target.stat().st_mode & 0o777),
        }
        if shutil.which("udevadm") is not None:
            udev = run(["udevadm", "info", "--query=property", "--name", str(target)])
            device["udev"] = parse_udev_properties(str(udev["stdout"]))
        devices.append(device)
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", help="Optional Panda FCI IP to ping four times.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    report: dict[str, object] = {
        "kernel": platform.release(),
        "realtime_marker": (
            Path("/sys/kernel/realtime").read_text().strip()
            if Path("/sys/kernel/realtime").is_file()
            else None
        ),
        "network": run(["ip", "-br", "addr"]),
        "routes": run(["ip", "route"]),
        "groups": run(["id", "-nG"]),
        "limits": {
            "rtprio": resource.getrlimit(resource.RLIMIT_RTPRIO),
            "memlock_bytes": resource.getrlimit(resource.RLIMIT_MEMLOCK),
        },
        "disk": run(["df", "-h", str(Path(__file__).resolve())]),
        "serial_devices": serial_devices(),
    }

    if args.robot_ip:
        robot_ip = str(ipaddress.ip_address(args.robot_ip))
        if shutil.which("ping") is None:
            report["ping"] = {"error": "ping executable not found"}
        else:
            report["ping"] = run(["ping", "-c", "4", "-W", "1", robot_ip])

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("Panda/Robotiq host diagnostic (no robot or serial driver opened)")
    for key, value in report.items():
        print(f"\n[{key}]")
        if isinstance(value, dict) and "stdout" in value:
            print(value["stdout"] or value["stderr"] or f"return code {value['returncode']}")
        elif key == "serial_devices" and not value:
            print("No /dev/serial/by-id devices found.")
        else:
            print(value)


if __name__ == "__main__":
    main()
