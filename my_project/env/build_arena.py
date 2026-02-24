# my_project/env/build_arena.py
# -*- coding: utf-8 -*-


from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import pybullet as p


# ============================================================
# 1) 户型模板池
# ============================================================
def get_layout(layout_name: str) -> Dict:
    """
    返回 dict：
      {
        "W": 半宽, "H": 半高,
        "outer": [segments...],
        "inner": [segments...],
        "gaps":  [("outer"/"inner", idx, center, width), ...],
        "rooms": {room_name: (xmin,xmax,ymin,ymax), ...},  # 用于“只在房间里”随机刷障碍/目标
        "spawn_xy": (x,y)  # 推荐起飞点（入户门内侧）
      }

    约定：
    - segment: (x1,y1,x2,y2)
    - gaps:
        竖墙（x1==x2）: center 表示 y_center
        横墙（y1==y2）: center 表示 x_center
    """
    layout_name = layout_name.strip()

    # -------------------------
    # EASY：两室一厅（2B1L）
    # -------------------------
    if layout_name == "apt_2room1hall":
        W, H = 7.0, 7.0  # 外框约 7m x 14m

        outer = [
            (0, -H, +W, -H),    # 0：下外墙（横）
            (0, +H, +W, +H),    # 1：上外墙（横）
            (0, -H, 0, +H),     # 2：左外墙（竖）
            (+W, -H, +W, +H),   # 3：右外墙（竖）
        ]

        # 墙(x1, y1, x2, y2)
        inner = [
            (4.0, 2.0, 4.0, +H),         # 0 厨房与次卧分割（竖）
            (4.0, 0.0, 4.0, -H),         # 1 客厅与主卧分隔（竖）
            (5.0, 0.0, 5.0, 2.0),       # 2 卫生间与餐厅分隔（竖）
            (1.0, 4.0, 1.0, +H),       # 3 阳台与厨房分隔（竖）
            (0.0, 4.0, 4.0, 4.0),         # 4 餐厅与阳台厨房分隔（横）
            (4.0, 2.0, +W, 2.0),       # 5 次卧与卫生间分隔（横）
            (4.0, 0.0, +W, 0.0),       # 6 主卧与卫生间分隔（横）
        ]

        # 门(which, idx, center, width)
        gaps = [
            ("outer", 2, 0.0, 1.2),  # 入户门
            ("inner", 4, 0.5, 0.9),   # 餐厅->阳台
            ("inner", 4, 2.5, 1),   # 餐厅->厨房
            ("inner", 5, 4.5, 1),   # 餐厅->次卧
            ("inner", 2, 1, 1),   # 餐厅->卫生间
            ("inner", 6, 4.5, 1),  # 餐厅->主卧
        ]

        rooms = {
            # 左下：客厅（x:0..4, y:-7..0）
            "living": (0.35, 3.75,  -6.65, -0.35),

            # 左中：餐厅/过道（x:0..4, y:0..4）
            # 这里是连接各房间的核心区域：target 可以刷，障碍建议少刷
            "dining": (0.35, 3.75,   0.35, 3.65),

            # 左上：厨房（x:1..4, y:4..7）——因为你有 x=1 的竖墙把阳台分出去了
            "kitchen": (1.25, 3.75,  4.35, 6.65),

            # 左上：阳台（x:0..1, y:4..7）
            # 不想刷东西到阳台就删掉这个 key
            "balcony": (0.35, 0.85,  4.35, 6.65),

            # 右上：次卧（x:4..7, y:2..7）
            "bed2": (4.25, 6.65,  2.35, 6.65),

            # 右下：主卧（x:4..7, y:-7..0）
            "bed1": (4.25, 6.65, -6.65, -0.35),

            # 右中：卫生间（大致 x:5..7, y:0..2）
            # 如果你不想刷东西进厕所，就只保留给 target 或者直接删掉
            "bath": (5.25, 6.65,  0.25, 1.75),

            # （可选）右中靠左的小走廊/缓冲区（x:4..5, y:0..2）
            # 这块是通行区，建议不给障碍；如果你要让 target 偶尔出现在走廊可以保留
            "corr": (4.25, 4.85,  0.25, 1.75),
        }


        spawn_xy = (-3.4, 2.6)  # 入户门内侧

        return {"W": W, "H": H, "outer": outer, "inner": inner, "gaps": gaps, "rooms": rooms, "spawn_xy": spawn_xy}

    # -------------------------
    # MED：三室两厅（3B2L）
    # -------------------------
    if layout_name == "apt_3room2hall":
        W, H = 8.0, 8.0  # 外框约 8m x 16m

        outer = [
            (0, -H, +W, -H),    # 0：下外墙（横）
            (0, +H, +W, +H),    # 1：上外墙（横）
            (0, -H, 0, +H),     # 2：左外墙（竖）
            (+W, -H, +W, +H),   # 3：右外墙（竖）
        ]

        #墙(x1, y1, x2, y2)
        inner = [
            (4.2, -H, 4.2, +H),     # 0 左区与右区主隔断（竖）
            (0, 4.5, 4.2, 4.5),     # 1 左上厨房卫生间与餐厅分隔（横）
            (2.4, 4.5, 2.4, +H),    # 2 左上厨房与卫生间分隔（竖）
            (0, -6, 4.2, -6),       # 3 起居厅与阳台分隔（横）
            (4.2, 7.0, +W, 7.0),    # 4 右上卧室与阳台分隔（横）
            (4.2, 3.0, +W, 3.0),    # 5 右上卧室与儿童房分隔（横）
            (4.2, 0.0, +W, 0.0),    # 6 儿童房与右下主卧分隔（横）
            (5.2, -2.0, +W, -2.0),  # 7 右下主卧与卫生间分隔（横）
            (5.2, 0.0, 5.2, -2.0),  # 8 右下主卧与卫生间分隔（竖）
        ]

        #门(which, idx, center, width)
        gaps = [
            ("outer", 2, 0, 1.2),    # 入户门
            ("inner", 1, 1.2, 1),    # 厨房<->餐厅
            ("inner", 1, 2.9, 1),    # 卫生间<->餐厅
            ("inner", 3, 2.1, 1.5),  # 起居厅<->阳台
            ("inner", 4, 6.1, 1.5),  # 右上卧室<->阳台
            ("inner", 0, 3.5, 1.0),  # 右上卧室<->餐厅
            ("inner", 0, 2.5, 1.0),  # 儿童房<->餐厅
            ("inner", 0, -0.5, 1),   # 主卧<->起居厅
            ("inner", 8, -0.5, 1),   # 主卧<->卫生间
        ]

        rooms = {
            # -------------------------
            # 左区（0..4.2）
            # -------------------------

            # 左下：起居厅（y:-6..0 左半边）
            # 不要贴墙，留 0.25~0.4 的边距，避免刷到墙体碰撞边缘
            "living":  (0.35, 4.0,  -5.7, -0.35),

            # 左上：餐厅/过道区（y:0..4.5）
            # 这里是连接右侧各房间的核心区域，可以放目标，障碍建议少放
            "dining":  (0.35, 4.0,   0.35, 4.15),

            # 左上角：厨房（被 x=2.4 竖墙切开，厨房在 x:0..2.4 且 y:4.5..8）
            "kitchen": (0.35, 2.15,  4.75, 7.65),

            # 厨房右侧：卫生间（x:2.4..4.2 且 y:4.5..8）
            "bathL":   (2.55, 4.0,   4.75, 7.65),

            # 左下：阳台（y:-8..-6）
            # 如果你不想刷东西到阳台就删掉这个 key
            "balconyL":(0.35, 4.0,  -7.65, -6.35),

            # -------------------------
            # 右区（4.2..8）
            # -------------------------

            # 右上卧室（y:3..7）
            "bedR_top": (4.45, 7.65,  3.25, 6.75),

            # 右中：儿童房（y:0..3）
            "bedR_mid": (4.45, 7.65,  0.25, 2.75),

            # 右下：主卧（y:-2..0，注意右侧有 x=5.2 的竖墙，x>5.2 是卫浴）
            "bedR_master": (4.45, 5.0, -1.75, -0.25),

            # 右下卫浴（x:5.2..8 且 y:-2..0）
            "bathR":   (5.35, 7.65,  -1.75, -0.25),

            # 右侧下厅（y:-8..-2，右区走廊/储物/次厅区域）
            # 如果你觉得这里不该放障碍，就只让 target 在这里采样（后面我教你加权）
            "hallR":   (4.45, 7.65,  -7.65, -2.25),

            # 右上阳台（y:7..8）
            "balconyR":(4.45, 7.65,   7.25, 7.65),
        }


        spawn_xy = (4.6, -2.8)  # 入户门内侧
        return {"W": W, "H": H, "outer": outer, "inner": inner, "gaps": gaps, "rooms": rooms, "spawn_xy": spawn_xy}

    raise ValueError(f"Unknown layout: {layout_name}")


# ============================================================
# 2) 造墙：segment -> box（支持门洞：把墙段切开）
# ============================================================
def _segment_is_vertical(x1, y1, x2, y2) -> bool:
    return abs(x2 - x1) < 1e-6


def _segment_is_horizontal(x1, y1, x2, y2) -> bool:
    return abs(y2 - y1) < 1e-6


def _split_segment_with_gap(x1, y1, x2, y2, gap_center: float, gap_len: float):
    segs = []

    if _segment_is_vertical(x1, y1, x2, y2):
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        g1 = gap_center - gap_len / 2
        g2 = gap_center + gap_len / 2

        # ✅ 关键：gap 不与该段相交 -> 不切
        if g2 <= y1 or g1 >= y2:
            return [(x1, y1, x2, y2)]

        # clamp 到段内
        g1c = max(g1, y1)
        g2c = min(g2, y2)

        eps = 1e-6
        if g1c - y1 > eps:
            segs.append((x1, y1, x2, g1c))
        if y2 - g2c > eps:
            segs.append((x1, g2c, x2, y2))
        return segs

    if _segment_is_horizontal(x1, y1, x2, y2):
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        g1 = gap_center - gap_len / 2
        g2 = gap_center + gap_len / 2

        # ✅ 关键：gap 不与该段相交 -> 不切
        if g2 <= x1 or g1 >= x2:
            return [(x1, y1, x2, y2)]

        g1c = max(g1, x1)
        g2c = min(g2, x2)

        eps = 1e-6
        if g1c - x1 > eps:
            segs.append((x1, y1, g1c, y2))
        if x2 - g2c > eps:
            segs.append((g2c, y1, x2, y2))
        return segs

    return [(x1, y1, x2, y2)]



def _create_wall_segment(
    x1: float, y1: float, x2: float, y2: float,
    thickness: float,
    height: float,
    rgba: Tuple[float, float, float, float],
    client_id: int,
) -> int:
    """
    把墙中心线 (x1,y1)->(x2,y2) 变成一个 box 墙体：
    - box 的长度沿 segment 方向
    - thickness 是墙厚（建议 0.03~0.06）
    - height 是墙高（1.0 左右够）
    """
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = (x2 - x1), (y2 - y1)
    length = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)

    half_ext = (length / 2.0, thickness / 2.0, height / 2.0)
    quat = p.getQuaternionFromEuler([0, 0, yaw])

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext, physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=rgba, physicsClientId=client_id)

    bid = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=(cx, cy, height / 2.0),
        baseOrientation=quat,
        physicsClientId=client_id
    )
    return int(bid)


# ============================================================
# 3) 障碍/目标/no-fly 的几何创建
# ============================================================
def _create_box_body(
    half_extents: Tuple[float, float, float],
    base_pos: Tuple[float, float, float],
    rgba: Tuple[float, float, float, float],
    mass: float,
    client_id: int,
) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=client_id)
    bid = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=base_pos,
        physicsClientId=client_id
    )
    return int(bid)


def _create_cylinder_body(
    radius: float,
    height: float,
    base_pos: Tuple[float, float, float],
    rgba: Tuple[float, float, float, float],
    mass: float,
    client_id: int,
) -> int:
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba, physicsClientId=client_id)
    bid = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=base_pos,
        physicsClientId=client_id
    )
    return int(bid)


# ============================================================
# 4) 主函数：create_arena（接口不变）
# ============================================================
def create_arena(scenario, client_id: Optional[int] = None) -> Dict[str, List[int]]:
    """
    根据 scenario 自动生成：
    - 户型墙（外墙 + 内墙 + 门洞）
    - 随机障碍（优先刷在 rooms 里）
    - no-fly（可选）
    - 目标（优先刷在 rooms 里）

    返回 arena_handle（供 sensors 使用）
    """
    cid = int(client_id) if client_id is not None else 0
    rng = np.random.default_rng(int(scenario.seed))

    # ---- 4.1 读取户型 ----
    layout = get_layout(scenario.layout_name)
    outer = list(layout["outer"])
    inner = list(layout["inner"])
    gaps = list(layout["gaps"])
    rooms = dict(layout.get("rooms", {}))

    wall_thickness = float(getattr(scenario, "wall_thickness", 0.05))
    wall_height = float(getattr(scenario, "wall_height", 1.0))

    # ---- 4.2 建 gap_map：gap_map["inner"][idx] = (center,width) ----
    # 允许一面墙多个门：gap_map["inner"][idx] = [(center1,width1), (center2,width2), ...]
    gap_map: Dict[str, Dict[int, List[Tuple[float, float]]]] = {"outer": {}, "inner": {}}
    for which, idx, center, width in gaps:
        which = str(which)
        idx = int(idx)
        center = float(center)
        width = float(width)
        gap_map[which].setdefault(idx, []).append((center, width))


    def build_walls(seg_list: List[Tuple[float, float, float, float]], which: str) -> List[int]:
        ids: List[int] = []
        for i, (x1, y1, x2, y2) in enumerate(seg_list):
            parts = [(x1, y1, x2, y2)]

            # 如果该墙有多个门洞，就依次挖洞：每挖一次都会把当前 parts 进一步切分
            if i in gap_map[which]:
                for (center, width) in gap_map[which][i]:
                    new_parts = []
                    for seg in parts:
                        new_parts.extend(_split_segment_with_gap(*seg, center, width))
                    parts = new_parts
            if i in gap_map[which]:
                print(f"[GAP] {which}[{i}] seg={(x1,y1,x2,y2)} gaps={gap_map[which][i]} -> parts={parts}")


            for (a, b, c, d) in parts:
                wid = _create_wall_segment(
                    a, b, c, d,
                    thickness=wall_thickness,
                    height=wall_height,
                    rgba=(0.70, 0.70, 0.70, 1.0),
                    client_id=cid
                )
                ids.append(wid)
        return ids

    wall_ids: List[int] = []
    wall_ids += build_walls(outer, "outer")
    wall_ids += build_walls(inner, "inner")

    # ---- 4.3 随机采样：只在 rooms 里刷（更像真实户型）----
    # placed: (x,y,r) 做圆形近似避碰，防止障碍/目标重叠
    placed: List[Tuple[float, float, float]] = []

    room_keys = list(rooms.keys())
    if len(room_keys) == 0:
        # 如果你没给 rooms，就退化为整个外框范围采样（不推荐）
        W = float(layout["W"])
        H = float(layout["H"])
        rooms = {"all": (-W, W, -H, H)}
        room_keys = ["all"]

    def sample_xy_from_rooms(r_approx: float, max_tries: int) -> Tuple[float, float]:
        """
        从 rooms 的矩形里采样一个点，并通过 placed 做避碰。
        """
        for _ in range(max_tries):
            rk = room_keys[int(rng.integers(0, len(room_keys)))]
            xmin, xmax, ymin, ymax = rooms[rk]

            # 防止采样到房间边界外
            if xmax - xmin < 2 * r_approx or ymax - ymin < 2 * r_approx:
                continue

            x = float(rng.uniform(xmin + r_approx, xmax - r_approx))
            y = float(rng.uniform(ymin + r_approx, ymax - r_approx))

            ok = True
            for px, py, pr in placed:
                if (x - px) ** 2 + (y - py) ** 2 < (r_approx + pr) ** 2:
                    ok = False
                    break
            if ok:
                return x, y

        # fallback：实在放不下，就返回 0,0
        return 0.0, 0.0

    # ---- 4.4 生成随机障碍 ----
    obstacle_ids: List[int] = []
    for _ in range(int(scenario.num_obstacles)):
        size = float(rng.uniform(float(scenario.obstacle_min_size), float(scenario.obstacle_max_size)))
        height = float(rng.uniform(float(scenario.obstacle_min_height), float(scenario.obstacle_max_height)))

        hx, hy, hz = size / 2.0, size / 2.0, height / 2.0
        r_approx = max(hx, hy) * 1.4

        x, y = sample_xy_from_rooms(r_approx, int(scenario.max_sample_tries))
        placed.append((x, y, r_approx))

        oid = _create_box_body(
            half_extents=(hx, hy, hz),
            base_pos=(x, y, float(scenario.spawn_z) + hz),
            rgba=(0.25, 0.25, 0.25, 1.0),
            mass=0.0,
            client_id=cid
        )
        obstacle_ids.append(oid)

    # ---- 4.5 生成 no-fly（实体薄板，可选）----
    nofly_ids: List[int] = []
    if bool(getattr(scenario, "nofly_is_solid", True)):
        for _ in range(int(scenario.num_nofly)):
            size = float(rng.uniform(float(scenario.nofly_min_size), float(scenario.nofly_max_size)))
            hx, hy = size / 2.0, size / 2.0
            hz = 0.02  # half height

            r_approx = max(hx, hy) * 1.4
            x, y = sample_xy_from_rooms(r_approx, int(scenario.max_sample_tries))
            placed.append((x, y, r_approx))

            nid = _create_box_body(
                half_extents=(hx, hy, hz),
                base_pos=(x, y, hz),
                rgba=(1.0, 0.2, 0.2, 0.35),
                mass=0.0,
                client_id=cid
            )
            nofly_ids.append(nid)
    else:
        nofly_ids = []

    # ---- 4.6 生成 targets（圆柱，建议有 visual，便于 camera/seg）----
    target_ids: List[int] = []
    for _ in range(int(scenario.num_targets)):
        r0 = float(scenario.target_radius)
        h0 = float(scenario.target_height)
        r_approx = r0 * 2.2

        # 目标更倾向刷在“客厅/卧室/餐厅”里：可以加权（可选）
        x, y = sample_xy_from_rooms(r_approx, int(scenario.max_sample_tries))
        placed.append((x, y, r_approx))

        tid = _create_cylinder_body(
            radius=r0,
            height=h0,
            base_pos=(x, y, float(scenario.spawn_z) + h0 / 2.0),
            rgba=(0.10, 0.85, 0.20, 1.0),
            mass=0.0,
            client_id=cid
        )
        target_ids.append(tid)

    # ---- 4.7 blocker 语义：默认 wall+obstacle+nofly，不把 target 放进去 ----
    ray_blocker_ids = list(wall_ids) + list(obstacle_ids) + list(nofly_ids)
    los_blocker_ids = list(wall_ids) + list(obstacle_ids) + list(nofly_ids)

    arena_handle: Dict[str, List[int]] = {
        "wall_ids": wall_ids,
        "obstacle_ids": obstacle_ids,
        "nofly_ids": nofly_ids,
        "target_ids": target_ids,
        "ray_blocker_ids": ray_blocker_ids,
        "los_blocker_ids": los_blocker_ids,
    }
    return arena_handle


# ============================================================
# 5) 兼容接口：简单四面墙（供 main.py 使用）
# ============================================================
def create_four_walls(
    pyb_client_id: int,
    half_size: float = 1.5,
    wall_thickness: float = 0.005,
    wall_height: float = 0.8,
) -> List[int]:
    """
    创建以原点为中心的正方形四面墙，返回墙体 body ID 列表。
    """
    cid = int(pyb_client_id)
    s = float(half_size)
    t = float(wall_thickness)
    h = float(wall_height)

    segments = [
        (-s, -s, +s, -s),  # 下墙
        (-s, +s, +s, +s),  # 上墙
        (-s, -s, -s, +s),  # 左墙
        (+s, -s, +s, +s),  # 右墙
    ]

    ids: List[int] = []
    for x1, y1, x2, y2 in segments:
        wid = _create_wall_segment(x1, y1, x2, y2,
                                   thickness=t, height=h,
                                   rgba=(0.70, 0.70, 0.70, 1.0),
                                   client_id=cid)
        ids.append(wid)
    return ids
