# 仿真器后端

仿真器采用参考 ASAP / HumanoidVerse 的分层思路（见[参考材料](REFERENCES.md)）：业务逻辑只依赖统一接口，后端实现彼此隔离，目标类和默认参数由独立配置选择。

```text
src/racing/
├── config/simulators/
│   ├── isaac.toml
│   └── mujoco.toml
└── simulators/
    ├── base.py
    ├── registry.py
    ├── isaac/environment.py
    └── mujoco/environment.py
```

`SimulatorBackend` 统一规定 `reset`、`step`、`states`、`render`、`close` 和时钟元数据。两个后端都保持 60 Hz 控制频率，物理频率必须是 60 的整数倍。调用方通过工厂创建后端：

```python
from racing.simulators import create_simulator

env = create_simulator("mujoco", track, render=False, physics_hz=480)
```

配置中的 `target` 使用 `module:Class` 形式。注册表延迟导入目标类，因此未选择 Isaac 时不会启动或导入 Isaac 原生运行时。新增后端时，应完成以下工作：

1. 在 `src/racing/simulators/<name>/` 中实现 `SimulatorBackend`。
2. 在 `src/racing/config/simulators/<name>.toml` 中声明目标类和默认参数。
3. 增加接口契约和原生物理测试；业务入口不增加新的后端判断分支。

`racing.isaac_env` 和 `racing.mujoco_env` 兼容层已移除；统一使用上面的工厂入口。旧脚本迁移与历史复现见 [LEGACY.md](LEGACY.md)。
