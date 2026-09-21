# RCS Robotiq2F85 Extension

This extension provides support for Robotiq 2F-85 Gripper in RCS.

## Installation

```shell
pip install rcs-robotiq2f85
```

For local development:

```shell
pip install -ve . --no-build-isolation
pip install -ve extensions/rcs_robotiq2f85
```

Get the serial number of the gripper with this command:
```shell
udevadm info -a -n /dev/ttyUSB0 | grep serial
```

Provide persistent device permissions through the serial-device group. Log out
and back in after adding the user:
```shell
sudo usermod -aG dialout "$USER"
```

## Usage
```python
from rcs_robotiq2f85 import RobotiQ2F85Gripper, RobotiQ2F85GripperConfig

gripper = RobotiQ2F85Gripper(RobotiQ2F85GripperConfig(serial_number="<YOUR_SERIAL_NUMBER>"))
gripper.reset()
gripper.shut()
print(gripper.get_normalized_width())
```
