## 实验室快速启动：Panda 键盘遥操数据采集

> 仅用于真机。运行前必须确认：realtime 内核已激活、机械臂和夹爪周围净空、
> 急停可用、Robotiq 和配置中的 RealSense 已连接，以及 Panda 已解锁并激活 FCI。

Panda 示例和硬件准备说明见 [`examples/panda/README.md`](examples/panda/README.md)；服务器端
π0.5 转换、训练和服务流程见
[`deployment/pi05_server/README.md`](deployment/pi05_server/README.md)。

每次采集新 episode 时，只需先把 `TASK_INSTRUCTION` 改成本次的真实任务描述，再复制运行整段命令：

```bash
cd /home/haifeng/robomme/robot-control-stack
source /home/haifeng/miniforge3/etc/profile.d/conda.sh
conda activate rcs

# 每次重启后、首次采集前执行一次主机实时调优（需要 sudo 密码）
sudo examples/panda/panda_realtime_tune.sh 192.168.178.12

# 每次采集前修改这一行，使其与实际任务一致
TASK_INSTRUCTION="pick up the fruit and place it in the tray"

# 数据写到 VS Code 工作区之外，避免编辑器监听大量 Parquet 文件而卡顿；
# 时间戳目录也能避免覆盖上一条数据
RUN_DIR="/home/haifeng/robot-data/panda/episode_$(date +%Y%m%d_%H%M%S)"

python examples/panda/panda_hardware_keyboard_collect.py \
  --config examples/panda/panda_hardware_session.lab.json \
  --instruction "$TASK_INSTRUCTION" \
  --output "$RUN_DIR" \
  --confirm "ENABLE-REAL-PANDA 192.168.178.12"
```

程序首次提示后输入 `START 192.168.178.12`。硬件连接并完成路径预检后，程序会显示
固定 Home 的七个关节角；确认机械臂到 Home 的路径净空并且急停触手可及，再输入
`MOVE-HOME 192.168.178.12`。机械臂低速到达固定 Home 后：

实验室配置中的 Home TCP 约为 Panda 基座坐标 `(x=0.405, y=0.000, z=0.600) m`。
相比旧 Home，`x` 向前移动约 10 cm、`z` 降低约 5 cm；J4 约为 `-142.3°`，为向下运动
留出更多关节余量。首次使用该姿态时仍须目视确认整条 Home 路径净空。

- `T`：开始记录当前 episode。
- `Y`：结束记录并标记为成功，然后关闭相机和硬件环境。
- `N`：结束记录并标记为失败，然后关闭相机和硬件环境。
- `ESC`：安全退出并关闭硬件环境。

移动键可以长按：`W/S` 控制基座 `x`，`A/D` 控制基座 `y`，`R/F` 控制基座 `z`。
实验室配置每个控制周期移动 2 mm；控制循环默认 10 Hz，因此长按速度上限约为 20 mm/s。
这仍低于程序允许的 5 mm 单步上限，可以缩短大范围移动时间，但靠近水果、托盘和桌面时要
点按，避免越过目标。若某次任务需要更精细的 1 mm 操作，可在采集命令末尾临时加
`--step 0.001`；程序会把实际步长写入该 episode 的元数据，离线校验会按对应步长检查。
只要工具包围盒、关节余量和碰撞检查通过，连续按键目标不会因正常的控制跟踪延迟而被额外阻塞。

松开或闭合夹爪后，至少继续记录 1 秒再按 `Y`。如果过早按 `Y`，程序会显示
`WAIT before SUCCESS` 并继续记录，确保训练数据包含夹爪动作后的真实结果。键盘输入即使
持续缓冲也不会让采集循环超过配置的 10 Hz。若 libfranka 的命令成功率连续约 1 秒低于
0.90，程序会安全停止并把当前 episode 留作失败/未完成数据，不要用于训练。

如果 libfranka 因 `communication_constraints_violation` 中止，程序现在不会退出：当前录制会
立即作为失败/未完成 UUID 落盘，并停止接受移动键。确认机械臂到固定 Home 的整条路径净空后，
按一次 `R` 执行错误恢复并低速回到 Home；显示 `RECOVERY HOME COMPLETE` 和新 UUID 后，再按
`T` 会用该 UUID 开始全新录制，不会续接异常前的轨迹。正常遥操状态下 `R` 仍是基座 `z`
正方向；
只有通信中止等待恢复时才作为 Home 恢复键。碰撞、关节边界或工作空间等其他安全错误不会
启用快捷恢复，必须退出并排查原因。

恢复后的输出目录会同时包含失败 UUID 和新 UUID，因此离线校验时必须用 `--uuid`
指定终端显示的新 UUID；不带 `--uuid` 时，校验器会列出目录里的所有 UUID 并停止，避免误校验
失败轨迹。先把终端显示的新 UUID 填入第一行：

```bash
RECOVERED_UUID="把 RECOVERY HOME COMPLETE 后显示的新 UUID 填在这里"
python examples/panda/panda_hardware_replay.py \
  --dataset "$RUN_DIR" \
  --config examples/panda/panda_hardware_session.lab.json \
  --uuid "$RECOVERED_UUID"
```

每次运行必须使用新的输出目录；上面的时间戳命令会自动创建新目录。
采集状态最多每秒打印一次；VS Code 工作区同时限制了终端回滚行数，并排除了旧数据和构建目录
的文件监听。按 `Y` 或 `N` 后等待程序显示 `Closed hardware environment`，再启动下一条采集，
不要让上一条采集进程和相机在后台继续运行。

### 上传已校验的数据

只有离线校验显示 `VALID` 后，才把当前 episode 上传到服务器：

```bash
rsync -avh --partial --info=progress2 \
  "$RUN_DIR/" \
  robomme-new:/home/zhanghf/robomme/teleop_pi05_pipeline/01_raw_upload/incoming/"$(basename "$RUN_DIR")"/
```

首次使用 `robomme-new` 时，先运行
`ssh-copy-id -i ~/.ssh/id_ed25519.pub robomme-new` 安装本机公钥。上传后可用同一条
`rsync` 命令加上 `-n --itemize-changes` 做只读检查；没有输出即表示无需继续同步。
不要上传未通过校验或没有 Parquet 文件的目录，也不要在确认远端文件完整前删除本机数据。

实验室配置默认只记录 RGB，避免每步额外编码并写入约 1 MB 的深度 TIFF。确认 RGB-only
连续运行稳定后，如任务确实需要深度，再在命令末尾加 `--include-depth` 做短时测试。
L500/L515 的彩色流最低模式是 `960x540`，不要把该型号改成不支持的 `640x480` 彩色模式。

启动真机控制前应把 CPU governor 设为 `performance`；程序检测到 `schedutil` 等模式时会
打印 `REALTIME WARNING`。这类设置属于主机配置，不会由采集程序静默修改。

本实验室机器应在每次重启后、首次采集前运行以下临时调优（需要 sudo 密码；上面的完整
采集命令已经包含这一步）：

```bash
sudo examples/panda/panda_realtime_tune.sh 192.168.178.12
```

它通过内核 sysfs 把每个 CPU governor 切到 `performance`（不依赖与实时内核版本匹配的
`cpupower` 包），关闭通往 Panda 的有线网卡 GRO，并暂停 `irqbalance`、把 Panda 网卡中断
固定到本实验室主机负载较低的 CPU5，避免 1 kHz FCI 收包与 CPU0 housekeeping 竞争。
通常重启后恢复系统默认值；也可以用 `sudo systemctl start irqbalance` 恢复动态中断分配。
不要在机器人运动期间修改这些设置。其他主机应先检查 CPU 拓扑，再用第二个参数覆盖 IRQ
CPU，例如 `sudo examples/panda/panda_realtime_tune.sh 192.168.178.12 5`。

### 回放刚采集的一条数据

先做离线检查。这个命令只读 Parquet 和 `_session.json`，不会加载硬件驱动。若仍在刚才的
同一个终端，可直接使用上面完整流程中的 `--dataset "$RUN_DIR"`；也可以显式填写目录：

```bash
python examples/panda/panda_hardware_replay.py \
  --dataset "$RUN_DIR" \
  --config examples/panda/panda_hardware_session.lab.json
```

普通采集目录只有一个 UUID，不需要 `--uuid`；发生通信恢复后按上文指定新 UUID。

只有 dry-run 显示 `VALID` 后，重新摆好物体、确认整条轨迹净空，并保持急停可用，才可
显式开启真机回放：

```bash
python examples/panda/panda_hardware_replay.py \
  --dataset "$RUN_DIR" \
  --config examples/panda/panda_hardware_session.lab.json \
  --execute \
  --confirm "REPLAY-REAL-PANDA 192.168.178.12"
```

程序还会依次要求连接确认、移动到固定 Home 的确认，以及包含 episode UUID 的最终回放
确认。回放过程中按 `ESC` 可中止。真机轨迹回放只能复现记录的控制命令，不能保证物体
因摆放误差、抓取接触和相机延迟而产生完全相同的运动结果。

<div align="center">
  <img src="https://raw.githubusercontent.com/RobotControlStack/robotcontrolstack.github.io/refs/heads/master/static/images/rcs_logo_line.svg" alt="rcs logo" width="60%">

  ### A lean, ROS-free Sim-to-Real framework for training and deploying Vision-Language-Action (VLA) models and Reinforcement Learning (RL) agents.

  [![Documentation](https://img.shields.io/badge/docs-robotcontrolstack.org-blue.svg)](https://robotcontrolstack.org)
  [![Paper](https://img.shields.io/badge/paper-ICRA_2026-green.svg)](https://robotcontrolstack.github.io/)
  [![Release](https://img.shields.io/github/v/release/RobotControlStack/robot-control-stack?color=orange)](https://github.com/RobotControlStack/robot-control-stack/releases)
  [![License](https://img.shields.io/github/license/RobotControlStack/robot-control-stack?color=blueviolet)](https://github.com/RobotControlStack/robot-control-stack/blob/main/LICENSE)
  [![CI Status](https://github.com/RobotControlStack/robot-control-stack/actions/workflows/ci.yaml/badge.svg)](https://github.com/RobotControlStack/robot-control-stack/actions)
</div>

---

**Robot Control Stack (RCS)** is a flexible, native [Gymnasium](https://gymnasium.farama.org/) wrapper-based robot control interface designed specifically for modern robot learning and Vision-Language-Action (VLA) models. 

It completely unifies **MuJoCo simulation** and real-world physical robot control into a single, seamless API. Currently, RCS natively supports five robots out-of-the-box: **Franka FR3/Panda, xArm7, UR5e, SO101, and I2RT YAM.**

![RCS Demo](https://raw.githubusercontent.com/RobotControlStack/robotcontrolstack.github.io/refs/heads/master/static/videos/grid.webp)

## 🚀 Why use Robot Control Stack?

Traditional robotics middleware (like ROS/ROS2) and complex motion planning pipelines (like MoveIt or standard `ros2_control`) are built for asynchronous, distributed systems. This often becomes a massive bottleneck when attempting to train modern, synchronous machine learning models.

**RCS is built differently:**
* **Zero ROS Overhead:** No complex message-passing, middleware, or network configuration required. Run natively in Python with a lightweight C++ backend.
* **Frictionless Sim-to-Real:** Train your Reinforcement Learning or VLA policies in our MuJoCo Gymnasium wrapper, and deploy the *exact same code* directly to physical hardware.
* **Synchronous Execution:** Optimized specifically for the highly parallelized, synchronous data collection required by modern ML workflows.
* **Ready-to-Use Apps:** Ships with pre-built applications for data collection via teleoperation and remote model inference via [vlagents](https://github.com/RobotControlStack/vlagents). See the [teleoperation guide](examples/teleop/README.md), and [inference guide](examples/inference/README.md).

## 🧩 Wrapper-Based Architecture

RCS utilizes a highly modular, wrapper-based architecture, allowing you to easily stack capabilities (cameras, grippers, action spaces) as needed.

<img src="docs/_static/rcs_architecture_small.svg" alt="rcs architecture diagram" width="100%">

## 💻 Example: Composing your Environment

Flexibly compose your Gymnasium environment to fit your exact training needs. *For common environment compositions, factory functions such as `rcs.envs.creators.SimEnvCreator` are provided.*

```python
from time import sleep

import gymnasium as gym
import numpy as np
from rcs._core.sim import SimConfig
from rcs.camera.sim import SimCameraSet
from rcs.envs.base import (
    CameraSetWrapper,
    ControlMode,
    CoverWrapper,
    GripperWrapper,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
    SimEnv,
)
from rcs.envs.scenes import EmptyWorldFR3
from rcs.envs.sim import GripperWrapperSim, RobotSimWrapper

import rcs
from rcs import sim

if __name__ == "__main__":
    # default configs
    scene = EmptyWorldFR3()
    cfg = scene.prefixed_cfg(scene.config())
    fr3 = scene.lead_robot_name(cfg)

    robot_cfg = cfg.robot_cfgs[fr3]
    gripper_cfg = cfg.gripper_cfgs[fr3]  # type: ignore
    camera_cfgs = cfg.camera_cfgs
    sim_cfg = SimConfig(
        realtime=True,
        async_control=True,
        frequency=1,  # in Hz (1 sec delay)
    )
    mjmodel = scene.create_model(cfg)
    kinematic_model_path, attachment_site = scene.kinematics_cfg(cfg)[fr3]

    simulation = sim.Sim(mjmodel, sim_cfg)
    ik = rcs.common.Pin(
        kinematic_model_path,
        attachment_site,
    )

    # base env
    robot = rcs.sim.SimRobot(simulation, ik, robot_cfg)
    env: gym.Env = SimEnv(simulation)
    env = RobotWrapper(env, robot, ControlMode.CARTESIAN_TQuat)

    # gripper
    gripper = sim.SimGripper(simulation, gripper_cfg)
    env = GripperWrapper(env, gripper)

    env = RobotSimWrapper(env)
    env = GripperWrapperSim(env)

    # camera
    camera_set = SimCameraSet(simulation, camera_cfgs, physical_units=True, render_on_demand=True)  # type: ignore
    env = CameraSetWrapper(env, camera_set, include_depth=True)  # type: ignore

    # relative actions bounded by 10cm translation and 10 degree rotation
    env = RelativeActionSpace(env, max_mov=(0.1, np.deg2rad(10)), relative_to=RelativeTo.LAST_STEP)
    env = CoverWrapper(env)

    env.get_wrapper_attr("sim").open_gui()
    # wait for gui to open
    sleep(1)
    env.reset()

    # access low level robot api to get current cartesian position
    print(env.get_wrapper_attr("robot").get_cartesian_position())

    for _ in range(10):
        # move 1cm in x direction (forward) and close gripper
        act = {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]}
        obs, reward, terminated, truncated, info = env.step(act)
        print(obs)
```

> **Note:** This and other examples can be found in the [`examples/`]() folder.

## 🛠️ Installation
* *For Python >3.11: The `rcs_realsense` extension won't work due to the `pyrealsense2` version RCS utilizes.*
* *For Python >3.12: The `ompl` python module is currently not available on PyPI. If OMPL is not used, it is safe to remove this dependency in `pyproject.toml`.*
### Via PyPI/pip

```shell
pip install rcs-core
```

### From Source

Make sure that common build tools (i.e., `build-essential`), python headers and a C++ compiler like `gcc` or `clang` are installed on your system/conda/docker.

*RCS works best in Python 3.11, and all extensions have been tested to work in 3.11.*

```shell
# clone repository
git clone https://github.com/RobotControlStack/robot-control-stack.git
cd robot-control-stack

# setup environment
conda create -n rcs python=3.11
conda activate rcs
conda install -c conda-forge urdfdom urdfdom_headers glfw

# or sudo apt install $(cat debian_deps.txt)
pip install 'pip>=25.1'
pip install --group build_deps

# install rcs
pip install -ve . --no-build-isolation
```


### RCS Asset Cache

RCS resolves its asset directory from the `RCS_PREFIX` environment variable. When it is unset, RCS defaults to `~/.rcs`.

On import, RCS checks whether that path exists. If it does not, it downloads the matching asset archive from GitHub into that location automatically.

```shell
export RCS_PREFIX=/path/to/rcs-assets
```


## 🦾 Hardware Extensions

RCS supports various hardware extensions to seamlessly connect your policies to the real world (e.g., FR3, xArm7, YAM, RealSense). These are located in the `extensions` directory.

To install a specific robot extension (example for Franka FR3):

```shell
sudo apt install $(cat extensions/rcs_fr3/debian_deps.txt)
pip install rcs-fr3

# or install it locally
pip install -ve extensions/rcs_fr3
```

For a full list of extensions and detailed documentation, visit **[robotcontrolstack.org/extensions](https://robotcontrolstack.org/extensions)**.

## ⚠️ Troubleshooting & FAQ
* **License error or group argument not found during installation?** Make sure you are using a pip version `>=25.1` and setuptools version `>=45`.
* **Dependency error during installation?** Make sure you are using Python 3.11. RCS extensions currently do not support 3.12+ due to OMPL and RealSense dependencies.
* **Simulation is running too slow?** Check that you have enable on-demand rendering: `SimCameraSet(..., render_on_demand=True)` to render camera frames only once per step. Resolution and number of cameras in the scene has a large impact on simulation speed. Make sure to use a decent GPU when rendering is enabled.


## 📚 Documentation

For full documentation, including advanced installation, modular usage, and API references, please visit:
👉 **[robotcontrolstack.org](https://robotcontrolstack.org)**

Useful quick-reference pages:
- **[RCS Conventions](https://robotcontrolstack.org/user_guide/conventions)** for quaternion order, frames, Euler angles, and gripper semantics
- **[Sim Scene Configuration](https://robotcontrolstack.org/user_guide/scene_configuration)** for `SimEnvCreatorConfig`, scene frames, and example setup patterns
- **[Apps](https://robotcontrolstack.org/apps/index)** for the teleoperation and inference example entry points
- **[libfranka Version Info](https://robotcontrolstack.org/extensions/libfranka_versions)** for the currently pinned `rcs_fr3` and `rcs_panda` `libfranka` versions and local-install guidance

## 🤝 Contribution

We welcome contributions from the robotics and ML community! For contribution guidelines, please check out **[robotcontrolstack.org/contributing](https://robotcontrolstack.org/contributing)**.

## 📝 Citation

If you find RCS useful for your academic work please consider citing it:

```bibtex
@inproceedings{juelg2026robotcontrolstack,
  title={{Robot Control Stack}: {A} Lean Ecosystem for Robot Learning at Scale}, 
  author={Tobias J{\"u}lg and Pierre Krack and Seongjin Bien and Yannik Blei and Khaled Gamal and Ken Nakahara and Johannes Hechtl and Roberto Calandra and Wolfram Burgard and Florian Walter},
  year={2026},
  booktitle={Proc.~of the IEEE Int.~Conf.~on Robotics \& Automation (ICRA)},
  note={Accepted for publication.}
}
```

For more scientific information and supplementary videos, visit the **[paper website](https://robotcontrolstack.github.io/)**.

## License

The RCS source code is licensed under AGPL-3.0. A small subset of redistributed third-party robot and sensor assets under `assets/` keeps its original upstream license; the applicable notices are collected in [THIRD_PARTY_ASSET_LICENSES.md](THIRD_PARTY_ASSET_LICENSES.md).
