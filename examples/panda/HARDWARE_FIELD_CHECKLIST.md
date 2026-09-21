# Panda + Robotiq laboratory information sheet

Do not put passwords or access tokens in this file or send them in chat.

## Robot and Desk

- Date/session ID:
- Exact robot model (Panda/FER):
- Arm serial number (optional/redacted):
- Robot System version:
- Server version reported by libfranka, if any:
- FCI feature installed (yes/no):
- FCI IP address:
- Desk reachable at `https://<FCI-IP>` (yes/no):
- Brakes and FCI state:
- Active Desk/Watchman safety rules:

## End effector and safety

- Mounted gripper/tool:
- Tool/TCP transform source and values:
- Payload mass (kg):
- Payload center of mass (m):
- Payload inertia:
- Safe Cartesian workspace minimum `[x, y, z]` in Panda base frame (m):
- Safe Cartesian workspace maximum `[x, y, z]` in Panda base frame (m):
- Conservative fully-open tool-box minimum `[x, y, z]` relative to TCP (m):
- Conservative fully-open tool-box maximum `[x, y, z]` relative to TCP (m):
- Tool box includes adapter, fingertips and attached sensors (yes/no):
- Initial TCP position `[x, y, z]` (m):
- Emergency stop tested and guarded (yes/no):
- Workspace clear and risk assessment complete (yes/no):

## Workstation and network

- Ethernet interface name:
- Workstation static IPv4/prefix:
- Direct gigabit Ethernet connection (yes/no):
- `ping` packet loss and min/avg/max/mdev:
- Kernel from `uname -a`:
- `/sys/kernel/realtime` value:
- `ulimit -r` and `ulimit -l`:
- User groups (must include the chosen realtime group and `dialout`):

## Robotiq 2F-85

- USB/RS-485 adapter model:
- `/dev/serial/by-id/...` path:
- Resolved `/dev/ttyUSB...` path:
- `ID_VENDOR_ID` / `ID_MODEL_ID`:
- `ID_SERIAL_SHORT` used by the driver:
- External gripper power confirmed (yes/no):

## Dataset

- Task instruction:
- Desired sampling rate (1--30 Hz; initial recommendation 20 Hz):
- Absolute output directory (must not already exist):
- Required cameras (none/RealSense/ZED/DIGIT):
- Camera names and serial numbers:
- RGB resolution/FPS:
- Depth required (yes/no):
- Target format (RCS Parquet/LeRobot/ROS bag):
- Available disk space and backup destination:

## Paste this message to Codex at the lab

```text
The robot is not authorized to move. Perform read-only host diagnostics only.
Robot model:
Robot System version:
FCI IP:
FCI feature installed:
Ethernet interface:
Robotiq /dev/serial/by-id path:
Robotiq ID_SERIAL_SHORT:
TCP/end-effector/payload configuration:
Workspace min/max in Panda base frame:
Camera model/serials:
Task and dataset format:
Emergency stop/workspace status:
```
