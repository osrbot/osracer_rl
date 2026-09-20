# osracer_policy —— 实车策略节点

把仿真训练出的**仅传感器**策略（轮速 + 转向角 + 15 Hz 单线激光，无全局位置）
作为控制器接入 OSRacer 的 ROS 2 竞速链路。

```text
/scan ─────────────┐
/odometry/filtered ─┼─► osracer_policy_node ─► /race/raw_ackermann_cmd ─► speed_profile_node ─► /ackermann_cmd ─► osracer_base
/race/safety_stop ──┘                                                                                    │
                                                                                                 串口 v <speed> <steer_deg>
```

节点发到 `/race/raw_ackermann_cmd`，因此既有 `safety_node` 与 `speed_profile_node`
仍在链路里，不会被绕过。

## 安装

```bash
# 1. 车辆项目的策略代码（纯 numpy，提供 racing 包）
cd /home/osrbot/Desktop/osracer_work && pip install -e .        # 或把仓库路径填给 racing_root 参数

# 2. 把本包放进车辆工作区
cp -r /home/osrbot/Desktop/osracer_work/deploy/osracer_policy /home/osrbot/osracer_ws/src/
cd /home/osrbot/osracer_ws && colcon build --packages-select osracer_policy && source install/setup.bash
```

`ackermann_msgs` 是车辆端既有依赖（`osracer_base` 已使用），开发机上未安装时无法构建本包。

## 运行

阶段 0/1（**默认 shadow**：只算不发，用于离线回放与台架）：

```bash
ros2 launch osracer_policy policy.launch.py \
  checkpoint_path:=/home/osrbot/Desktop/osracer_work/output/racing/policies/frozen-candidates-g00c_v10c_lateral.json \
  racing_root:=/home/osrbot/Desktop/osracer_work \
  shadow_mode:=true max_speed_mps:=1.0 max_steering_rad:=0.3
```

或直接用参数文件：`ros2 run osracer_policy policy_node --ros-args --params-file config/policy.yaml`

确认无误后进入低速实车（限速由 `speed_profile_node` 再兜一层）：

```bash
ros2 launch osracer_policy policy.launch.py shadow_mode:=false max_speed_mps:=1.0
```

## 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `checkpoint_path` | 空 | 冻结检查点 JSON；为空即启动失败 |
| `racing_root` | 空 | 提供 `racing` 包的工程根目录（可留空，用已安装版本） |
| `shadow_mode` | `true` | `true` 只记录指令不发布 |
| `max_speed_mps` | `1.0` | 本节点自身的速度上限（与 `speed_profile_node` 互为兜底） |
| `max_steering_rad` | `0.3` | 转向角上限 |
| `wheel_radius_m` | `0.045` | 轮半径，用于轮速↔车速换算 |
| `scan_timeout_s` / `odom_timeout_s` | `0.2` / `0.3` | 输入超时即发 0 |

## 安全设计

- **哈希校验**：启动时比对检查点记录的 `actor_source_sha256` 与当前导入的控制器源码，
  不一致直接 fatal 退出，避免“跑的不是验收过的那份代码”。
- **失效即停**：`/race/safety_stop`、激光超时、里程计超时、输入缺失都会下发 0。
- **双重限幅**：本节点 `max_speed_mps`/`max_steering_rad` + 车辆 `speed_profile_node`
  + 底盘固件 `ForwardMaxMmps`/`SteeringMaxMdeg`。
- **默认 shadow**：任何情况下都不会“装完就动”。

## 已知限制

- 实车**没有逐轮编码器和转向关节反馈**，观测里的 `wheel_vel` 由 EKF 车速复现为四轮同值，
  `steer_pos` 用上一次下发的转角代替。`s` 遥测帧索引 6 的字段待确认。
- 底盘只接受**一个速度 + 一个转向角**，仿真里用于漂移的后轮增速没有对应执行器，
  实车上漂移段会退化为常规转向。
- 策略参数按 0.05 倍 RC 尺度赛道标定，实车策略参数需要重新标定；
  分级提速流程见 [../../docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md)。
