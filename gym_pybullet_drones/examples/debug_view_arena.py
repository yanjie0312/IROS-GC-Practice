# gym_pybullet_drones/examples/debug_view_arena.py
# -*- coding: utf-8 -*-

"""
debug_view_arena.py

用途：
- 可视化查看 build_arena.py 生成的户型(栅格地图版) + 随机障碍/目标/no-fly
- 输出当前 scenario 信息（难度/seed/layout）
- 做一组“自检验证”：
  1) ray 必命中墙（墙可被 ray 命中）
  2) 从上往下 ray 命中每个 obstacle/nofly（它们有 collision）
  3) 同 seed 重建 max |Δpos| = 0.0（严格可复现）

注意：
- 不改 sensors，不依赖 sensors
- 只验证 arena_handle 的正确性
"""

import time
import numpy as np
import pybullet as p
import pybullet_data

from my_project.experiments.scenarios import make_scenario
from my_project.env.build_arena import create_arena


def print_arena_handle(arena: dict):
    """打印 arena_handle 的数量概览"""
    print("arena_handle:")
    for k in ["wall_ids", "obstacle_ids", "nofly_ids", "target_ids", "ray_blocker_ids", "los_blocker_ids"]:
        v = arena.get(k, [])
        print(f"{k}: {len(v)} -> {v[:20]}{'...' if len(v) > 20 else ''}")


def ray_hit(cid: int, start, end):
    """一条 ray，返回 (hit_body_id, hitFraction, hitPos)"""
    hit = p.rayTest(start, end, physicsClientId=cid)[0]
    return hit[0], hit[2], hit[3]


def ray_down_to_body(cid: int, body_id: int):
    """从该 body 的 AABB 中心上方往下打一条 ray，看命中谁"""
    aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=cid)
    cx = (aabb_min[0] + aabb_max[0]) / 2
    cy = (aabb_min[1] + aabb_max[1]) / 2
    start = [cx, cy, 5.0]
    end = [cx, cy, -1.0]
    hit_id, frac, pos = ray_hit(cid, start, end)
    return hit_id, pos


def snapshot_xy(cid: int, ids):
    """记录一组 body 的 (x,y) 位置，便于复现性验证"""
    out = []
    for bid in ids:
        pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=cid)
        out.append([pos[0], pos[1]])
    return np.array(out, dtype=float)


def main():
    # -----------------------------
    # 1) 选择难度 + 自动 seed（每次运行不同）
    # -----------------------------
    map_level = "med"     # easy / med / hard
    deg_level = "l2"      # l0 / l1 / l2 / l3
    seed = int(time.time()) % 100000

    scenario = make_scenario(
        map_level=map_level,
        deg_level=deg_level,
        seed=seed
    )

    print(f"[Scenario] diff={scenario.name} seed={scenario.seed} layout={scenario.layout_name} "
          f"obs={scenario.num_obstacles} nofly={scenario.num_nofly} tgt={scenario.num_targets}")

    # -----------------------------
    # 2) 启动 PyBullet GUI
    # -----------------------------
    cid = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=18,
        cameraYaw=90,
        cameraPitch=-89,
        cameraTargetPosition=[4.0, 0.0, 0.0]
    )

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(physicsClientId=cid)
    p.setGravity(0, 0, -9.8, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)

    # -----------------------------
    # 3) 生成 arena
    # -----------------------------
    arena = create_arena(scenario, client_id=cid)

    # -----------------------------
    # 4) 设置相机：俯视 + 看全户型（像平面图）
    # -----------------------------
    p.resetDebugVisualizerCamera(
        cameraDistance=14,
        cameraYaw=0,
        cameraPitch=-89,
        cameraTargetPosition=[0, 0, 0.5],
        physicsClientId=cid
    )

    # 也可以在 GUI 上显示当前配置（可选）
    p.addUserDebugText(
        f"diff={scenario.name} seed={scenario.seed}\nlayout={scenario.layout_name}",
        textPosition=[-2, -2, 1.5],
        textColorRGB=[1, 1, 0],
        textSize=1.3,
        lifeTime=0,
        physicsClientId=cid
    )

    # -----------------------------
    # 5) 自检 1：ray 打墙
    # -----------------------------
    # 朝 +X 打：命中某面墙
    hit_id, frac, pos = ray_hit(cid, [0, 0, 0.2], [10, 0, 0.2])
    print("[Ray +X] hit_id:", hit_id, "frac:", frac, "pos:", pos,
          "is_wall:", hit_id in arena["wall_ids"])

    # 朝 +Y 打：命中某面墙
    hit_id, frac, pos = ray_hit(cid, [0, 0, 0.2], [0, 10, 0.2])
    print("[Ray +Y] hit_id:", hit_id, "frac:", frac, "pos:", pos,
          "is_wall:", hit_id in arena["wall_ids"])

    # -----------------------------
    # 6) 自检 2：ray 从上往下命中 obstacle/nofly
    # -----------------------------
    for oid in arena["obstacle_ids"]:
        hid, hpos = ray_down_to_body(cid, oid)
        print(f"[Ray obstacle {oid}] hit={hid} ok={hid==oid} hitPos={hpos}")

    for nid in arena["nofly_ids"]:
        hid, hpos = ray_down_to_body(cid, nid)
        print(f"[Ray nofly {nid}] hit={hid} ok={hid==nid} hitPos={hpos}")

    # -----------------------------
    # 7) 自检 3：targets 不应在 blocker 中（防未来改坏）
    # -----------------------------
    assert set(arena["target_ids"]).isdisjoint(set(arena["ray_blocker_ids"])), "Targets are in ray_blocker_ids!"
    assert set(arena["target_ids"]).isdisjoint(set(arena["los_blocker_ids"])), "Targets are in los_blocker_ids!"
    print("[OK] targets not in blockers")

    # -----------------------------
    # 8) 自检 4：同 seed 重建可复现（max |Δpos| = 0.0）
    # -----------------------------
    ids = arena["obstacle_ids"] + arena["nofly_ids"] + arena["target_ids"]
    snap1 = snapshot_xy(cid, ids)

    # 重置并用同一个 scenario/seed 重建
    p.resetSimulation(physicsClientId=cid)
    p.setGravity(0, 0, -9.8, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)

    arena2 = create_arena(scenario, client_id=cid)
    ids2 = arena2["obstacle_ids"] + arena2["nofly_ids"] + arena2["target_ids"]
    snap2 = snapshot_xy(cid, ids2)

    print("[Repro] counts same:", len(ids) == len(ids2))
    if len(snap1) == len(snap2) and len(snap1) > 0:
        print("[Repro] max |Δpos|:", float(np.max(np.abs(snap1 - snap2))))
    else:
        print("[Repro] max |Δpos|: N/A (no objects)")

    # 重建后再次设置相机（因为 resetSimulation 会清掉可视化状态）
    p.resetDebugVisualizerCamera(
        cameraDistance=14,
        cameraYaw=0,
        cameraPitch=-89,
        cameraTargetPosition=[0, 0, 0.5],
        physicsClientId=cid
    )
    p.addUserDebugText(
        f"diff={scenario.name} seed={scenario.seed}\nlayout={scenario.layout_name}",
        textPosition=[-2, -2, 1.5],
        textColorRGB=[1, 1, 0],
        textSize=1.3,
        lifeTime=0,
        physicsClientId=cid
    )

    # 打印 arena_handle 概览（注意墙很多，所以只显示前 20 个）
    print_arena_handle(arena2)

    # -----------------------------
    # 9) 保持窗口不退出
    # -----------------------------
    print("\n[INFO] PyBullet GUI running. Close the window or Ctrl+C to exit.")
    while True:
        p.stepSimulation(physicsClientId=cid)
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
