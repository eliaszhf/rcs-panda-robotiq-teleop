# Panda Cartesian controller recovery record

This directory records the exact repository version of the original RCS
asynchronous operational-space controller (OSC).  The original source remains
stored as an immutable Git object; it is not copied into the build tree and is
therefore never selected accidentally.

## Legacy controller snapshot

- Repository commit: `7b35f92e5c369a6c6ff21de40603552bb24cba1c`
- Source path: `extensions/rcs_fr3/src/hw/Franka.cpp`
- Git blob: `ca34468dda9f43ccf087a865a64b50803ef24f57`
- SHA-256 of the original file:
  `96752658e3244ea6182bb53c2fbfd0f3aed11b28570e5cad06146f0d259a3a35`

The snapshot can be inspected without changing the worktree:

```bash
git show 7b35f92e5c369a6c6ff21de40603552bb24cba1c:extensions/rcs_fr3/src/hw/Franka.cpp
```

Do not restore the entire file with `git checkout`: the working file also
contains independent safety, error-reporting, TCP-frame, and binding fixes.
If a real-robot test shows that the firmware Cartesian-pose path is unsuitable,
restore only the legacy `PInverse` helper and `Franka::osc()` implementation
from the snapshot, retain the independent fixes, then rebuild both extensions:

```bash
python -m pip install -ve extensions/rcs_fr3 --no-build-isolation
python -m pip install -ve extensions/rcs_panda --no-build-isolation
```

`rcs_panda` must be rebuilt second because its build materializes the shared
`rcs_fr3` C++ sources.  Never switch implementations while an FCI control
session is active.  After a rollback, repeat the zero-motion hold and 1 mm
single-axis tests before enabling keyboard teleoperation.

## Firmware Cartesian-pose alternative

The tested-to-compile alternative is stored in
`firmware_cartesian_pose_osc.cpp.txt`. Its full working-file SHA-256 after the
successful build was:

`d86927f5fe5b904c0871ccf1774164b6cf171018205c045866c5fc7cb5646228`

The active working source was switched back to the legacy OSC for the next
hardware test. The firmware snapshot is deliberately not part of the CMake
source tree.

## Minimal legacy OSC stabilization patch

The active legacy OSC now differs from the immutable snapshot in two narrowly
scoped ways:

- its nullspace target is captured from the measured joint state when OSC
  starts instead of pulling toward the repository's hard-coded posture;
- the model Coriolis term that was already calculated is added to the commanded
  torque, matching the libfranka Cartesian impedance example.

The existing per-joint 5 Nm torque limit, collision settings, Cartesian gains,
interpolator, TCP handling, and asynchronous controller architecture remain
unchanged.  Validate this patch first with a zero-motion hold and only then
with a single 2 mm Cartesian step; do not proceed directly to teleoperation.
