# Panda + Robotiq 2F-85 simulation

These examples use the RCS simulation API only. They do not import the Panda
hardware extension, connect to a robot, or send hardware commands.

Activate the environment created for this checkout:

```shell
source /home/haifeng/miniforge3/bin/activate rcs
```

Run the Cartesian smoke test without a viewer:

```shell
python examples/panda/panda_cartesian_control.py --headless
```

Run it with the MuJoCo viewer:

```shell
python examples/panda/panda_cartesian_control.py
```

Run terminal keyboard teleoperation while the MuJoCo viewer is open:

```shell
python examples/panda/panda_keyboard_teleop.py
```

Keep the terminal focused. W/S moves x, A/D moves y, R/F moves z, Q opens the
gripper, E closes it, and Escape exits. Each Cartesian key press requests a
0.005 m relative movement. RCS actions have this single-robot dictionary shape:

```python
{"right": {"tquat": [dx, dy, dz, 0, 0, 0, 1], "gripper": [1]}}
```

## Real-hardware preparation

The stable lab collection workflow is documented in
[`TELEOP_DATA_COLLECTION_ZH.md`](TELEOP_DATA_COLLECTION_ZH.md). The remote
pi05 policy-server and single-Panda deployment workflow is documented in
[`PI05_REMOTE_DEPLOYMENT_ZH.md`](PI05_REMOTE_DEPLOYMENT_ZH.md).

The hardware scripts are deliberately separate from the simulation examples:

- `panda_hardware_device_check.py` only inspects the host, network and USB
  enumeration. It never imports or opens a robot/gripper driver.
- `panda_hardware_state_check.py` requests one Panda state snapshot and never
  calls a motion method. Constructing this repository's driver does configure
  collision, impedance and guiding parameters, so it requires an explicit
  confirmation despite not requesting motion.
- `panda_hardware_keyboard_collect.py` is safety-gated real-hardware control.
  It disables arm homing on reset, enforces a 5 mm maximum step and caller-
  supplied Cartesian workspace, requires two confirmations, uses low initial
  speed/force values, and records one episode as Parquet. Creating/resetting
  its environment performs Franka error recovery and opens the Robotiq gripper.

Complete [`HARDWARE_FIELD_CHECKLIST.md`](HARDWARE_FIELD_CHECKLIST.md) at the
robot before using either driver script. Do not guess the TCP, payload,
workspace, Robot System/libfranka compatibility, or USB serial number.

Copy `panda_hardware_session.example.json` outside the examples directory and
fill every `null` field with measured/approved values. The workspace is an
axis-aligned safe volume in the Panda base frame. The tool box is a conservative
box around the fully open gripper, adapter, fingertips and any attached sensor,
expressed relative to the configured TCP. All eight oriented tool-box corners
must remain inside the workspace.

This shared robot must remain on Desk's stock **Franka Hand** mechanical-data
preset. The hardware scripts read back and verify its mass, centre of mass,
inertia and `F_T_EE` before the first motion command, and refuse to move on any
mismatch. They deliberately leave `robot_cfg.load_parameters=None`, so they do
not call libfranka `setLoad()` or overwrite the other group's setup. The
Robotiq TCP below is an RCS control/observation transform only; it is not written
to Desk. Do not change Desk end-effector data for these examples.

The flange-to-TCP translation and quaternion are mandatory as well. Use the
center between the fingertips at the chosen reference opening as the TCP, and
obtain the transform from the approved CAD/measurement for the actual adapter.
The repository's generic 0.1493 m Robotiq offset is not silently assumed by the
hardware collector.

The example leaves the tool box unset. The photographed hardware has a wrist
RealSense, bracket, custom adapter and exposed cabling, so the earlier stock
Robotiq estimate is not a valid collision envelope. Measure the complete moving
assembly in the TCP frame, with the fingers fully open, before filling these
values. Workspace values also remain unset because they depend on the actual
cell and must not be guessed.

For an initial low-range hardware test, first use
`panda_hardware_state_check.py` to read the Robotiq TCP pose. Then calculate a
provisional workspace around that pose without opening either hardware driver:

```shell
python examples/panda/panda_workspace_from_pose.py \
  --config examples/panda/panda_hardware_session.example.json \
  --tcp-tquat X Y Z QX QY QZ QW
```

The default requested TCP travel is only +/-20 mm in x/y and +/-10 mm in z.
Those defaults are not a claim that the space is clear: an operator must first
verify the full gripper envelope and the requested travel against the table,
objects, robot body, people, cables and overhead obstacles. Pass explicit
`--negative-travel X Y Z` and `--positive-travel X Y Z` values only after that
inspection. Copy the printed bounds into a separate session config; do not
treat these provisional bounds as the final task workspace.

Host-only diagnostics are safe to run without the extensions:

```shell
python examples/panda/panda_hardware_device_check.py --robot-ip <FCI-IP>
```

If a multi-port FTDI adapter exposes the same serial for several interfaces,
identify the Robotiq channel with a read-only status request after serial
permissions have been granted:

```shell
python examples/panda/panda_robotiq_status_check.py \
  /dev/ttyUSB0 /dev/ttyUSB1 --confirm READ-ROBOTIQ-STATUS
```

This reads Modbus input registers 2000--2002 (function code 4). It does not
write, activate, reset or move the gripper. Use the one port that returns three
registers; do not infer the port merely from enumeration order. Put its stable
by-id path in `gripper_device`; this checkout supports an explicit device path
because FT2232 interfaces can share one serial number.

The exact state-check and collection commands should be assembled only after
the checklist has been reviewed. The collector requires a new absolute output
directory and refuses to overwrite or append to an existing session. Keyboard
controls are W/S, A/D, R/F, Q/E, T=start recording, Y=finish success,
N=finish failure and Escape=exit. One process records one episode; start a new
process and output directory for the next episode.

The collector accepts repeated `--camera NAME=SERIAL` options or a `cameras`
mapping in JSON. By default it requires `wrist` and `third_person`, records RGB
and aligned depth at 640x480/30 Hz, and uses dummy (uncalibrated) extrinsics.
Run `python -m rcs_realsense serials` after connecting the cameras and copy the
reported serials into the reviewed config. Use `--no-include-depth` only when
RGB-only data is intentional. The result is one RCS Parquet dataset containing
robot/gripper observations, actions and camera frames; it is not a ROS bag.

After the completed configuration has been reviewed, the collector accepts it
as follows (the confirmation is deliberately never stored in JSON):

```shell
python examples/panda/panda_hardware_keyboard_collect.py \
  --config /absolute/path/to/reviewed-session.json \
  --confirm "ENABLE-REAL-PANDA <FCI-IP>"
```

## Future hardware transition

The Panda hardware implementation is in `extensions/rcs_panda`. Its local
installation requires the extension's Debian dependency and builds against the
libfranka 0.9.2 revision pinned by its CMake configuration. A real deployment
also needs the Panda in FCI mode, a wired low-latency connection, and host and
robot addresses in the same dedicated subnet. Real-time control should follow
Franka's recommended real-time kernel and network setup.

The Robotiq implementation is in `extensions/rcs_robotiq2f85`. It talks Modbus
RTU over a USB serial device and selects the port by serial number. Persistent
access should be granted through the `dialout` group instead of `chmod 777`.

This checkout's Robotiq extension has been updated for `robotiq2f==0.2.0` and
uses the actual `Robotiq2F85` driver API. Its public RCS wrapper class is
`RobotiQ2F85Gripper`.

For real hardware, replace `EmptyWorldPanda` with the Panda hardware creator,
provide the actual Panda IP and Robotiq serial number, and remove `SimConfig`
and viewer handling. `ControlMode.CARTESIAN_TQuat`, `RelativeTo.LAST_STEP`, and
the Cartesian/gripper action structure can remain the same.
