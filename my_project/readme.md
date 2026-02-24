my_project/
  main.py                 # 主入口：初始化环境，执行 sense→plan→control 主循环，统一退出与保存结果
  config.py               # 全局配置：频率/场地/难度/阈值/终止条件等参数集中管理

  experiments/
    scenarios.py          # 难度/随机种子/批量实验配置：保证可复现对比（Easy/Medium/Hard）

  env/
    build_arena.py        # 搭建室内场景：墙/随机障碍/no-fly/巡检目标生成
    sensors.py            # 感知接口：自状态读取 + raycast 距离 + 碰撞/接触 + 目标GT/相机接口
    targets.py            # 目标管理：保存 target_ids、到点判定、巡检完成记录、已完成状态维护
    disturbances.py       # 扰动注入：风/噪声/载荷变化，用于鲁棒性评估与自适应调参

  navigation/
    base.py               # 统一接口：State/Command 数据结构 + 规划模块抽象基类
    wall_follow.py        # 验证任务：直飞→碰墙→绕墙→回家，用于验证闭环pipeline跑通
    search_mission.py     # 任务级搜索：网格/候选点探索、覆盖率维护、下一目标选择（未知目标/未知路线）
    avoidance.py          # 局部避障：势场/短视距安全航行，输出局部安全航点
    mission_manager.py    # 任务状态机：takeoff→search→inspect→resume→finish，并统一终止条件

  control/
    pid.py                # 底层控制：封装 DSLPIDControl，目标点→RPM，实现稳定跟踪与悬停
    adaptive_tuning.py    # 自适应调参：基于指标评估在扰动环境下调整PID（可先规则/离线版本）

  utils/
    geometry.py           # 数学/几何工具：距离、限幅、向量运算等
    rate.py               # 频率工具：print/render 降频与节流
    metrics.py            # 评估指标：耗时/碰撞/覆盖率/成功率等，支持实验对比
    logging.py            # 统一日志输出：保存轨迹、配置、指标到CSV/JSON，便于写报告画图
