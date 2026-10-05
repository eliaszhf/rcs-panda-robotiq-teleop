# π0.5 server workflow

These scripts are copied to
`/home/zhanghf/robomme/teleop_pi05_pipeline/bin` on `robomme-new`.
They assume the policy source snapshot is located at
`teleop_pi05_pipeline/code/robomme_policy_learning` and deliberately keep the
base π0.5 parameters and the existing Python virtual environment outside the
pipeline. `common.sh` sets `PYTHONPATH` so that execution uses the centralized
source snapshot rather than the original working tree.

Run them in order:

```bash
cd /home/zhanghf/robomme/teleop_pi05_pipeline
bin/01_convert_data.sh
bin/02_compute_norm_stats.sh
bin/03_train_pi05.sh
```

After evaluating checkpoints, write the selected absolute checkpoint path to
`03_training/SELECTED_CHECKPOINT.txt`, then serve it on loopback only:

```bash
bin/04_serve_pi05.sh
```

On the robot computer, create an SSH tunnel:

```bash
ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 8000:127.0.0.1:8000 robomme-new
```

The robot client then connects to `ws://127.0.0.1:8000`. Do not expose the
unauthenticated policy WebSocket directly to a public network.
