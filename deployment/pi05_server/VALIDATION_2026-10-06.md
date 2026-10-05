# Remote pi05 protocol validation — 2026-10-06

Validation used the selected smoke-test checkpoint without connecting to any
robot hardware.

- Server entry point: `teleop_pi05_pipeline/bin/04_serve_pi05.sh`
- Bind address: `127.0.0.1:8000`
- Transport: SSH local forwarding to robot-side port `18000`
- Input: zero-filled `256x256` base/wrist RGB, eight-dimensional zero state,
  and the fruit-to-tray task prompt
- Reset response: `reset_finished=True`
- Output: finite `(20, 8)` action chunk
- First request, including JAX compilation: `9.478 s`
- Warm requests: `0.290 s`, then `0.186 s`
- OpenPI msgpack compatibility: robot-to-server and server-to-robot passed

The smoke checkpoint is not approved for robot execution. Warm latency also
exceeds one 10 Hz control period, so initial hardware validation must use
observe-only mode. Action-chunk execution requires a separate supervised safety
review after a real checkpoint has been selected.

