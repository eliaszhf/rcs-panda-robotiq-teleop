"""Identify a Robotiq 2F-85 port using a read-only Modbus status request.

This diagnostic reads input registers 2000--2002 with Modbus function code 4.
It never writes a register, activates, resets, or moves the gripper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import minimalmodbus
import serial


CONFIRMATION = "READ-ROBOTIQ-STATUS"


def resolve_serial_device(value: str) -> Path:
    device = Path(value).resolve(strict=True)
    if not device.name.startswith("ttyUSB"):
        raise ValueError(f"expected a /dev/ttyUSB device, got {device}")
    return device


def read_status_registers(device: Path, timeout: float) -> list[int]:
    instrument = minimalmodbus.Instrument(str(device), slaveaddress=9, mode=minimalmodbus.MODE_RTU)
    instrument.serial.baudrate = 115200
    instrument.serial.parity = serial.PARITY_NONE
    instrument.serial.bytesize = 8
    instrument.serial.stopbits = serial.STOPBITS_ONE
    instrument.serial.timeout = timeout
    try:
        return instrument.read_registers(registeraddress=2000, number_of_registers=3, functioncode=4)
    finally:
        instrument.serial.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("devices", nargs="+", help="One or more /dev/ttyUSB* or by-id paths.")
    parser.add_argument("--timeout", type=float, default=0.3)
    parser.add_argument("--confirm", required=True, help=f"Must equal {CONFIRMATION!r}.")
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"Refusing serial access. Pass exactly: --confirm {CONFIRMATION!r}")
    if not (0.05 <= args.timeout <= 2.0):
        raise SystemExit("--timeout must be between 0.05 and 2.0 seconds")

    for value in args.devices:
        try:
            device = resolve_serial_device(value)
            registers = read_status_registers(device, args.timeout)
        except (OSError, ValueError, minimalmodbus.ModbusException) as error:
            print(f"{value}: no valid Robotiq response ({error})")
        else:
            print(f"{value} -> {device}: Robotiq status registers={registers}")


if __name__ == "__main__":
    main()
