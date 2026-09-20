# Suzuka RC 分层桥梁：几何与接口

版本 `rc-r2-bridge` 保留原有 0.05 倍水平路径和 1.5 m 路宽，增加真正供物理接触使用的桥面与坡道。源 GeoJSON 未提供高程；桥的高度、坡长、厚度和其余平地均是 RC 训练假设，不是实测或历史赛季复原。

[Honda 官方赛道介绍](https://global.honda/en/F1/race/2026/Japan/preview/)
确认后直道跨过前半段后进入 130R；
[Formula 1 的 Suzuka 介绍](https://www.formula1.com/en/latest/article/need-to-know-japan.5UJu7JbjnqEqCgCMG8u02O.5UJu7JbjnqEqCgCMG8u02O)
描述 Degner Two / 130R 的交叉关系。这些来源只支持上下层关系，不为下列训练尺寸背书。

| 项目 | RC 几何 |
| --- | --- |
| 下层交点水平弧长 | 约 116.115474 m |
| 上层交点水平弧长 | 约 234.625941 m |
| 桥面上表面 / 物理厚度 | 0.90 m / 0.08 m |
| 交叉处净空 | 0.82 m，超过含反射盒车辆的 0.65 m 高度包络 |
| 平台 | 8 m，约 s=229.125941–237.125941 m |
| 上下坡 | 各 20 m，约 s=209.125941–257.125941 m 之间 |
| 最大解析坡度 | 0.0675，即 6.75%，约 3.86° |

平台中心位于上层交点前 1.5 m，使下坡结束于后续紧弯之前。交点前后仍有平坦的完整跨线空间。坡高按 cubic smoothstep 变化，实际导出为采样的分段线性三角面；坡度和曲面法线来自相同网格。平地段使用稳健的多边形并集剖分，桥段单独生成三角面，避免二维交点合并或紧弯偏移带反面。

## 数据与接口

`track.json` 新增以下字段，旧的 `centerline` 仍为 N×2，`length` 仍为**水平弧长**，不能当作完整三维表面长度：

- `elevation`：N 个路面高程，与中心线点一一对应。
- `road_vertices`、`road_faces`：共享接触/显示三角网格。
- `boundary_segments_3d`：每条边界两端的世界坐标，N×2×3。
- `bridge`：桥厚、平台范围、起降范围、交叉点和假设说明。
- `has_elevation`、`requires_3d_backend`：提醒消费者使用三维场景。

`Track` 的方法：

```python
track.has_elevation                         # bool
track.at3d(s, offset=0)                      # (xyz, yaw, grade)
track.elevation_at(s)                       # z in metres
track.grade_at(s)                           # dz / d(horizontal s), not radians
track.road_triangles_3d()                   # (K, 3, 3)
track.boundary_segments_3d()                # (M, 2, 3)
track.project(xy, s_hint=previous_s, z=z)    # (s, signed_cte, yaw)
```

`project` 的 `s_hint` 将搜索限制在前后默认 10 m；`z` 或三元素坐标可区分两层。坐标使用车辆底盘高度也可接近正确层，但精准路面查询应使用路面高程。已知历史进度时应继续使用 `s_hint`。单独调用旧二维投影仍有交点歧义，不能用它宣称已解决桥梁定位。

两引擎须消费共享三角网格，并将护栏底部放在三维边界端点上。MuJoCo 可将高程大于零的三角面向下挤出 0.08 m，构成独立凸棱柱；不应将整座桥合成巨大凸包封住桥下通道。Isaac 可使用静态三角网格接触。底部平地、坡道、桥面和车间碰撞必须由原生引擎积分；仅抬高可视网格不符合此接口。

激光必须使用真实三维挂载位姿，对相同高程的护栏、桥面和桥底测距。重置、进度、越界及漂移评估也必须采用正确层；元数据中的“支持计分”是几何可用性声明，不是双引擎驾驶已经通过的证据。

## 可复现与保留内容

运行 `python -m racing.build_tracks --track suzuka` 可离线重建。旧二维 RC 版本归档在 `output/racing/track_revision1/suzuka/`；此前结果不能直接作为新桥验收。原 `source_metric.json` 和 `full_scale/` 均保持原字节内容；全尺度版本仍是未分层的二维参考资产，不支持桥梁竞速。

赛道测试检查跨线处两层表面存在、0.82 m 净空、坡度上限、前后高度连续、定位不跳支路、平地 API 兼容，以及 DAE/OBJ 的独立解析和面朝向。实际通桥、桥下双车相遇和全圈表现需要原生引擎结果另行证明。
