# Panda + 远程 π0.5 部署

本流程让 checkpoint 和 GPU 推理保留在 `robomme-new`，机器人电脑只运行硬件、
安全检查和 WebSocket 客户端。真机执行前必须先完成 observe-only 审计。

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
