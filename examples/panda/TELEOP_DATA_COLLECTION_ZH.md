# Panda 遥操数据采集手册

本文档适用于实验室当前已经稳定运行的组合：Panda、Robotiq 2F-85、第三视角
RealSense 和键盘遥操。每次启动一个进程只采集一条 episode，数据保存为 RCS
Parquet 数据集。

> 真机运行前必须确认机械臂与夹爪周围净空、急停可用、Panda 已解锁并启用
> FCI、Robotiq 与相机已连接。机器人运动期间，操作员不得离开急停可触及范围。

## 1. 下载代码

项目 GitHub 地址：

<https://github.com/eliaszhf/rcs-panda-robotiq-teleop>

本文档和当前稳定采集代码位于 `haifeng-panda-teleop` 分支。第一次下载可执行：

```bash
git clone --branch haifeng-panda-teleop --single-branch \
  https://github.com/eliaszhf/rcs-panda-robotiq-teleop.git
cd rcs-panda-robotiq-teleop
```

已经克隆过仓库时，在没有未保存修改的前提下更新代码：

```bash
git switch haifeng-panda-teleop
git pull origin haifeng-panda-teleop
```

浏览器也可以直接打开该分支：

<https://github.com/eliaszhf/rcs-panda-robotiq-teleop/tree/haifeng-panda-teleop>

## 2. 当前固定配置

- Panda IP：`192.168.178.12`
- 配置文件：`examples/panda/panda_hardware_session.lab.json`
- 控制频率：10 Hz
- 平移步长：每个控制周期 2 mm
- 长按移动速度上限：约 20 mm/s
- 相机：第三视角 RealSense，`960x540 @ 30 Hz`
- 采集内容：RGB、机械臂/夹爪观测和控制动作；默认不采集深度
- Home TCP：Panda 基座坐标约 `(x=0.405, y=0.000, z=0.600) m`

不要随意修改实验室配置中的 TCP、工具包围盒、工作空间、Home 关节角或 Desk
机械参数。需要变更时应先重新完成现场安全检查。

## 3. 每次开机后的准备

打开一个终端，进入仓库并激活环境：

```bash
cd /home/haifeng/robomme/robot-control-stack
source /home/haifeng/miniforge3/etc/profile.d/conda.sh
conda activate rcs
```

主机每次重启后、第一次采集前运行一次实时调优：

```bash
sudo examples/panda/panda_realtime_tune.sh 192.168.178.12
```

该脚本会把 CPU governor 调为 `performance`、关闭 Panda 网卡 GRO、暂停
`irqbalance`，并把 Panda 网卡中断固定到 CPU5。这些设置通常会在主机重启后恢复。
不要在机器人运动期间运行调优脚本。

## 4. 采集一条 episode

先修改任务描述，再整段复制执行：

```bash
# 必须与这一条数据中实际执行的任务一致
TASK_INSTRUCTION="pick up the fruit and place it in the tray"

# 每条 episode 自动使用新的时间戳目录，防止覆盖已有数据
RUN_DIR="/home/haifeng/robot-data/panda/episode_$(date +%Y%m%d_%H%M%S)"

python examples/panda/panda_hardware_keyboard_collect.py \
  --config examples/panda/panda_hardware_session.lab.json \
  --instruction "$TASK_INSTRUCTION" \
  --output "$RUN_DIR" \
  --confirm "ENABLE-REAL-PANDA 192.168.178.12"
```

程序启动后会要求两次人工确认。它们是在程序提示出现后输入的内容，不是单独的
shell 命令：

1. 看到创建硬件环境的提示后，输入 `START 192.168.178.12`。
2. 程序完成硬件连接和 Home 路径预检后，再次确认路径净空并输入
   `MOVE-HOME 192.168.178.12`。

初始化会执行 Franka 错误恢复并打开 Robotiq 夹爪。第二次确认后，机械臂会低速
移动到固定 Home。到达 Home 且终端显示键盘说明后，才开始遥操。

## 5. 遥操按键

保持采集终端处于焦点：

| 按键 | 功能 |
| --- | --- |
| `W` / `S` | Panda 基座坐标系 `x` 正向 / 负向 |
| `A` / `D` | Panda 基座坐标系 `y` 正向 / 负向 |
| `R` / `F` | Panda 基座坐标系 `z` 正向 / 负向 |
| `Q` / `E` | 打开 / 闭合夹爪 |
| `T` | 开始记录当前 episode |
| `Y` | 结束记录并标记成功 |
| `N` | 结束记录并标记失败 |
| `Esc` | 安全退出并关闭硬件环境 |

推荐操作顺序：

1. 在尚未记录时，用移动键确认机器人响应正常并接近任务起点。
2. 按 `T`，看到 `RECORDING STARTED` 后再正式执行任务。
3. 靠近水果、托盘和桌面时使用点按，避免长按越过目标。
4. 完成夹爪动作后继续记录至少 1 秒，再按 `Y`。
5. 看到 `RECORDING STOPPED: SUCCESS` 或 `RECORDING STOPPED: FAILURE` 后按
   `Esc` 退出。

如果夹爪动作后过早按 `Y`，程序会显示 `WAIT before SUCCESS` 并继续记录；等待
满 1 秒后重新按 `Y`。一旦按下 `Y` 或 `N`，本进程中的 episode 就已结束。下一条
数据必须退出程序，并使用新的 `RUN_DIR` 重新启动。

## 6. 立即离线校验

采集程序退出后，在同一个终端运行：

```bash
python examples/panda/panda_hardware_replay.py \
  --dataset "$RUN_DIR" \
  --config examples/panda/panda_hardware_session.lab.json
```

这是只读 dry-run：它只检查 Parquet 和 `_session.json`，不会导入硬件驱动，也不会
向机器人发送命令。只有输出包含以下两行时，才把这条数据纳入后续数据集：

```text
VALID episode UUID: ...
DRY RUN ONLY: no hardware driver was imported and no command was sent.
```

校验会检查控制频率、时间戳、相机帧、控制命令成功率、碰撞信号、工作空间、关节
余量和配置一致性。保存 `RUN_DIR` 路径及校验输出，方便后续追踪。

## 7. 上传到数据服务器

只上传已经通过上一节离线校验的 episode。实验室数据服务器别名为
`robomme-new`，原始数据接收目录为：

```text
/home/zhanghf/robomme/teleop_pi05_pipeline/01_raw_upload/incoming
```

首次使用时安装本机 SSH 公钥。该命令会要求输入一次服务器密码，以后 `ssh` 和
`rsync` 可以直接使用 `robomme-new` 别名：

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub robomme-new
ssh -o BatchMode=yes robomme-new true
```

如果第二条命令没有输出且退出状态为 0，说明免密认证已配置成功。然后创建接收目录：

```bash
ssh robomme-new \
  'mkdir -p /home/zhanghf/robomme/teleop_pi05_pipeline/01_raw_upload/incoming'
```

上传刚刚采集并校验通过的 `$RUN_DIR`：

```bash
rsync -avh --partial --info=progress2 \
  "$RUN_DIR/" \
  robomme-new:/home/zhanghf/robomme/teleop_pi05_pipeline/01_raw_upload/incoming/"$(basename "$RUN_DIR")"/
```

末尾的斜杠必须保留。远端最终结构应为
`incoming/episode_日期_时间/_session.json` 和对应的 Parquet 文件。`--partial` 会保留
中断的临时传输；网络恢复后重复执行同一条命令即可续传，并跳过已经同步的文件。

上传结束后检查远端目录大小和文件列表：

```bash
REMOTE_EPISODE="/home/zhanghf/robomme/teleop_pi05_pipeline/01_raw_upload/incoming/$(basename "$RUN_DIR")"
ssh robomme-new "du -sh '$REMOTE_EPISODE' && find '$REMOTE_EPISODE' -maxdepth 1 -type f -printf '%f\n' | sort"
```

最后再做一次只读同步检查：

```bash
rsync -an --itemize-changes \
  "$RUN_DIR/" \
  robomme-new:/home/zhanghf/robomme/teleop_pi05_pipeline/01_raw_upload/incoming/"$(basename "$RUN_DIR")"/
```

这条命令没有输出，表示本地与远端没有待同步的文件。需要上传多条历史 episode 时，
应先逐条运行离线校验，再把所有通过 `VALID` 的目录作为 `rsync` 源参数；不要上传只有
`_session.json`、没有 Parquet 文件的未完成目录。

确认 `rsync` 成功且远端文件完整之前，不要删除本机原始数据。

## 8. 失败和异常处理

- 任务没有完成：按 `N`，退出后保留目录用于排查，但不要作为成功数据训练。
- 想放弃正在记录的数据：按 `Esc`。程序会刷新数据并将它保留为失败/未完成。
- 出现 `BLOCKED`：目标触及工具包围盒、工作空间或关节余量限制；不要反复强按。
- 出现 `SAFETY STOP`：立即停止本条采集，确认碰撞信号、关节位置和网络状态。
- 控制命令成功率连续约 1 秒低于 0.90：程序会安全停止；先排查实时调优和 Panda
  有线网络，不要直接重试任务。
- 相机预检失败：检查第三视角相机连接和序列号，不要绕过相机要求采集正式数据。
- 输出目录已存在：不要删除或覆盖旧数据，创建新的时间戳目录。

需要 1 mm 精细操作时，可以在采集命令末尾临时添加 `--step 0.001`。程序会把实际
步长写进本条 episode 的 `_session.json`，离线校验会使用记录值。正式采集默认保持
实验室配置中的 2 mm 步长。

## 9. 每条数据的检查清单

- [ ] 任务描述与实际任务一致
- [ ] 场地净空，急停可用且在手边
- [ ] 使用新的 `RUN_DIR`
- [ ] 到 Home 前确认整条运动路径净空
- [ ] 按 `T` 后才开始正式任务
- [ ] 夹爪最后动作后至少记录 1 秒
- [ ] 使用 `Y` 或 `N` 正确结束 episode
- [ ] 按 `Esc` 正常关闭硬件环境
- [ ] 离线校验输出 `VALID`
- [ ] 使用 `rsync` 上传，并核对远端目录

更底层的硬件准备和安全约束见
[`HARDWARE_FIELD_CHECKLIST.md`](HARDWARE_FIELD_CHECKLIST.md)；真机脚本的设计说明见
[`README.md`](README.md)。
