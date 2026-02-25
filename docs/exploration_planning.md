# 地震后室内无人机自动巡检 — 探索规划设计记录

> 记录日期：2026-02-25

---

## 一、项目现状

### 已完成模块

| 模块 | 文件 | 状态 |
|------|------|------|
| 仿真环境 | `main.py` + `CtrlAviary` | 可运行，原地悬停 |
| 公寓场景构建 | `env/build_arena.py` | 完整（两室一厅/三室两厅，含门洞、障碍、目标） |
| 传感器感知层 | `env/sensors.py` | 完整（22根射线、碰撞检测、目标检测、噪声注入） |
| 目标管理 | `env/targets.py` | 完整，但未接入 main.py |
| PID控制 | `control/pid.py` | 可用 |
| 正交实验设计 | `experiments/scenarios.py` | 完整（easy/med/hard × l0-l3） |
| 导航框架骨架 | `navigation/base.py`, `mission_manager.py` | 接口已定义 |

### 待实现模块

| 模块 | 文件 | 状态 |
|------|------|------|
| 势场避障 | `navigation/avoidance.py` | 空壳（接口已定义） |
| 搜索任务 | `navigation/search_mission.py` | 空文件 |
| 环境扰动 | `env/disturbances.py` | 空文件 |
| 评估指标 | `utils/metrics.py` | 空文件 |

---

## 二、无地图自主探索方法对比

| 方法 | 优点 | 缺点 | 适合场景 |
|------|------|------|----------|
| 随机游走 | 极简 | 覆盖率低、重复扫描 | 演示用途 |
| 势场法（Potential Field） | 实现简单、实时性好 | 容易陷入局部极小值 | 避障层，非主探索 |
| 沿墙探索（Wall-following） | 简单，能覆盖外围 | 漏掉内部区域、无法主动进门 | 简单凸环境 |
| **占据栅格 + Frontier 探索** | **系统性全覆盖、学术成熟、与射线传感器契合** | **需维护栅格，有一定开销** | **室内多房间环境 ✓** |
| RRT-based Exploration | 处理复杂几何、路径平滑 | 实现复杂，实时性要求高 | 大规模3D空间 |

### 推荐方案：Frontier-Based Exploration + 势场法避障

理由：
- 22根360°射线天然适合实时更新占据栅格
- 公寓户型（有墙有门）需要系统性覆盖，随机游走不够用
- 地震巡检场景对覆盖率要求高于速度
- 已有 `TargetManager`，发现目标后可立即切换巡检模式

---

## 三、系统架构

```
main.py
  └── MissionManager（已有）
        ├── SearchMission          ← 任务4
        │     ├── FrontierPlanner  ← 任务3
        │     │     └── OccupancyGrid  ← 任务2
        │     └── TargetManager   （已有，接入即可）
        └── AvoidanceLayer         ← 任务1
              └── PIDController   （已有）
```

### SearchMission 状态机

```
TAKEOFF
  │ 到达飞行高度
  ▼
EXPLORE ──── 检测到新目标 ──► GOTO_TARGET
  │                               │
  │ frontier全部探索完             │ 到达目标附近
  ▼                               ▼
DONE（返航）                   INSPECT
                                   │
                                   │ 悬停 2 秒完成巡检
                                   ▼
                              还有目标? → GOTO_TARGET
                              还有frontier? → EXPLORE
                              全部完成? → DONE
```

---

## 四、任务清单（按实现顺序）

### 任务 1 — 势场法避障层
**文件：** `navigation/avoidance.py`（填充已有空壳）

实现内容：
- 读取射线数据，对距离较近的障碍施加排斥力
- 计算指向 `target_pos` 的吸引力
- 合力叠加，输出修正后的安全 `target_pos`

关键接口：
```python
def filter_target(self, state, sensors: dict, target_pos: np.ndarray) -> np.ndarray
# sensors 中有: ray_dists, ray_dirs_body, min_dist, collision
```

验收标准：给定一个靠近墙的 `target_pos`，返回值被推离墙壁。

---

### 任务 2 — 占据栅格
**文件：** `navigation/occupancy_grid.py`（新建）

实现内容：
- 初始化覆盖整个公寓的 2D 网格（每格约 0.1m），所有格子标记为 `unknown`
- 每帧根据射线数据更新：沿射线路径 → `free`；击中点 → `occupied`；超出范围 → 保持 `unknown`
- 提供 `get_frontiers()` 方法

关键接口：
```python
class OccupancyGrid:
    def update(self, drone_pos, ray_dists, ray_dirs_world)
    def get_frontiers(self) -> list[np.ndarray]
    def is_free(self, pos) -> bool
```

验收标准：matplotlib 可视化地图，能看到墙体 `occupied`、飞过区域 `free`、未到达区域 `unknown`。

---

### 任务 3 — Frontier 规划器
**文件：** `navigation/frontier_planner.py`（新建）

实现内容：
- 调用 `OccupancyGrid.get_frontiers()` 获取 frontier 点
- 对 frontier 点聚类（相邻格子合并为一个区域）
- 选出最优 frontier（先用最近策略）作为下一航点
- 无 frontier 时返回 `None`（探索完毕）

关键接口：
```python
class FrontierPlanner:
    def update(self, drone_pos, ray_dists, ray_dirs_world)
    def get_next_waypoint(self, drone_pos) -> np.ndarray | None
    def is_exploration_done(self) -> bool
```

验收标准：打印每次选出的 frontier 点坐标，观察无人机朝未探索方向移动。

---

### 任务 4 — 搜索任务状态机
**文件：** `navigation/search_mission.py`（完整实现）

实现内容：
- 继承 `BaseMission`，实现 `reset()` 和 `update()`
- 内嵌状态机：TAKEOFF → EXPLORE → GOTO_TARGET → INSPECT → DONE
- EXPLORE 状态调用 `FrontierPlanner`
- GOTO_TARGET / INSPECT 状态调用 `TargetManager`

关键接口：
```python
class SearchMission(BaseMission):
    def reset(self, state: State)
    def update(self, state: State, sensors: dict) -> Command
```

验收标准：无人机能自主飞完整个公寓，找到并巡检所有目标，最终返回起飞点。

---

### 任务 5 — 更新 main.py，打通完整流程
**文件：** `main.py`

修改内容：
- 接入 `TargetManager`
- 创建 `SearchMission` + `AvoidanceLayer` + `MissionManager`
- 替换悬停循环为完整导航循环
- 添加终止条件（`cmd.finished == True` 或超时）

修改后主循环结构：
```python
manager = MissionManager(
    mission=SearchMission(frontier_planner, target_manager, ...),
    avoidance_layer=AvoidanceLayer()
)

for i in range(steps):
    obs, _, _, _, _ = env.step(action)
    pkt = sensor.sense(obs=obs[0], step=i, t=t)
    target_manager.update(pkt)

    state = State(xyz=pkt["pos"], vel=pkt["vel"], step=i, t=t)
    cmd = manager.update(state, pkt)

    if cmd.finished:
        break

    action[0, :] = pid.compute(..., target_pos=cmd.target_pos, ...)
```

验收标准：`python -m my_project.main` 能运行到底，无人机完成探索并巡检所有目标。

---

### 任务 6（可选）— 环境扰动注入
**文件：** `env/disturbances.py`

实现内容：
- 读取 `scenario` 的风扰参数（`wind_std`, `gust_prob` 等）
- 每步对无人机施加 `p.applyExternalForce()`

验收标准：在 l2/l3 退化等级下，无人机能感受到偏移但仍能完成任务。

---

### 任务 7（可选）— 评估指标
**文件：** `utils/metrics.py`

实现内容：
- 记录每次实验：巡检覆盖率、任务完成时间、碰撞次数、飞行总路程
- 支持批量实验，用于正交实验对比

---

## 五、依赖关系与并行分工

### 依赖图

```
任务1（势场避障）        ──────────────────────────────────┐
                                                          ├──► 任务5（main.py）
任务2（占据栅格）──► 任务3（Frontier规划器）──► 任务4（搜索状态机）──┘

任务6（扰动注入）   完全独立
任务7（评估指标）   完全独立
```

硬依赖：
- 任务3 需要任务2的 `OccupancyGrid`
- 任务4 需要任务3的 `FrontierPlanner`
- 任务5 需要任务4的 `SearchMission` 和任务1的 `AvoidanceLayer`

### 推荐分工

**2人团队：**

| 人员 | 负责 |
|------|------|
| 甲 | 任务1 → 任务6/7（等待期）→ 任务5 |
| 乙 | 任务2 → 任务3 → 任务4 |

**3人团队：**

| 人员 | 负责 |
|------|------|
| 甲 | 任务1 + 任务6 |
| 乙 | 任务2 → 任务3 |
| 丙 | 任务7 → 任务4（等乙完成任务3后） |

**4人团队：**

| 人员 | 负责 |
|------|------|
| 甲 | 任务1 |
| 乙 | 任务2 |
| 丙 | 任务3（等乙定好接口后立即开始） |
| 丁 | 任务4 → 任务5 |

### 并行开发技巧

无论几人，建议开发前统一约定所有模块的接口签名（方法名、参数、返回值类型），各自先用空壳占位，再填充实现，最后在任务5集成。

---

## 六、占据栅格更新逻辑（伪代码参考）

```python
for i, (dist, dir_body) in enumerate(zip(ray_dists, ray_dirs_body)):
    dir_world = R @ dir_body  # 旋转到世界坐标

    # 沿射线路径，到距离 dist 之前的格子 → free
    for d in linspace(0, dist - margin, steps):
        mark_free(drone_pos + d * dir_world)

    # 击中点的格子 → occupied（如果 dist < ray_length）
    if dist < ray_length - eps:
        mark_occupied(drone_pos + dist * dir_world)

    # 超出射线范围的格子 → 保持 unknown
```

---

## 七、Frontier 选取策略

| 策略 | 描述 | 复杂度 |
|------|------|--------|
| 最近 frontier | 飞向距当前位置最近的 frontier 中心（greedy） | 低，推荐入手 |
| 最大 frontier | 飞向规模最大的 frontier 区域 | 中 |
| 信息增益最大 | 飞向能新探索最多未知格子的位置 | 高，效果最优 |

推荐先实现"最近 frontier"，验证流程跑通后再优化为更优策略。
