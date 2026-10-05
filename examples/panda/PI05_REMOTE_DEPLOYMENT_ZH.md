# Panda + 远程 π0.5 部署

本流程让 checkpoint 和 GPU 推理保留在 `robomme-new`，机器人电脑只运行硬件、
安全检查和 WebSocket 客户端。真机执行前必须先完成 observe-only 审计。

## 当前链路状态（2026-10-06）

远程推理通信链路已经打通：

```text
robomme-new 加载 checkpoint 和 norm stats
  → 127.0.0.1:8000 WebSocket 服务
  → SSH 隧道
  → 机器人电脑 OpenPI 兼容客户端
  → reset 成功
  → 返回有限的 (20, 8) 动作块
```

已完成的无机器人验证：

- `teleop_pi05_pipeline/bin/04_serve_pi05.sh` 能从集中源码启动服务。
- smoke checkpoint、π0.5 基础权重和对应 norm stats 均成功加载。
- 客户端收到服务器 metadata，`reset_finished=True`。
- 连续三次推理均返回 `(20, 8)`，且动作不含 NaN/Inf。
- 首次请求包含 JAX 编译，耗时约 9.48 秒；热启动请求约 0.29 秒和 0.19 秒。
- 测试结束后策略服务和 SSH 隧道均已正常关闭，没有遗留后台进程。

本次验证只使用全零合成图像和状态，完全没有连接或驱动 Panda。它证明网络、序列化、
checkpoint 加载和动作返回链路正常，但不证明策略动作适合真机。当前 smoke checkpoint
仍禁止进入 `--execute` 模式；必须训练并选择正式 checkpoint，再使用真实机器人观测完成
observe-only 审计。

## 1. 启动服务器策略

在 `robomme-new` 上：

```bash
cd /home/zhanghf/robomme/teleop_pi05_pipeline
bin/04_serve_pi05.sh
```

服务默认仅监听 `127.0.0.1:8000`。

当前 `SELECTED_CHECKPOINT.txt` 指向 smoke-test checkpoint，仅用于接口验证，不得直接
驱动机器人。2026-10-06 的合成观测测试中，首次 JAX 编译加推理约 9.48 秒，后续请求
约 0.19–0.29 秒；因此正式 checkpoint 上线后仍须重新测量延迟。初始真机验证只能使用
observe-only，不能因为接口测试成功就跳过动作审计。

## 2. 建立 SSH 隧道

机器人电脑打开单独终端并保持运行：

```bash
ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 8000:127.0.0.1:8000 robomme-new
```

另一个终端检查服务：

```bash
curl --fail http://127.0.0.1:8000/healthz
```

## 3. 安装轻量客户端依赖

```bash
source /home/haifeng/miniforge3/etc/profile.d/conda.sh
conda activate rcs
python -m pip install -r examples/panda/requirements_pi05.txt
```

## 4. 只观察审计

该模式连接 Panda、Robotiq 和相机，会执行 Franka error recovery 并打开夹爪，
但除此之外不会发送模型产生的机械臂或夹爪运动命令：

```bash
python examples/panda/panda_pi05_inference.py \
  --config examples/panda/panda_hardware_session.lab.json \
  --instruction "pick up the fruit and place it in the tray" \
  --confirm "OBSERVE-PI05-PANDA 192.168.178.12"
```

必须检查动作形状为 `(20, 8)`、没有 NaN/Inf、首步关节差值合理且推理延迟稳定。

## 5. 有界真机执行

只有 observe-only 多次稳定通过后才可使用：

```bash
python examples/panda/panda_pi05_inference.py \
  --config examples/panda/panda_hardware_session.lab.json \
  --instruction "pick up the fruit and place it in the tray" \
  --execute \
  --max-steps 100 \
  --actions-per-query 1 \
  --confirm "EXECUTE-PI05-PANDA 192.168.178.12"
```

程序还会要求输入 `START 192.168.178.12`、`MOVE-HOME 192.168.178.12` 和
`RUN-PI05 192.168.178.12`。执行期间按 `Esc` 停止。初次测试保持
`--actions-per-query 1`，每个关节每步默认最多变化 0.5°；模型目标相对实测状态超过
5°、工具包围盒越界、碰撞、关节余量不足、通信异常或持续低控制成功率都会停止。
