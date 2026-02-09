# my_project/env/build_arena.py
import pybullet as p

def create_four_walls(pyb_client_id: int,
                      half_size: float,
                      wall_thickness: float,
                      wall_height: float):
    """
    四面墙围成正方形，中心在原点。
    返回 wall_ids：用于碰撞检测 / 最近距离 / ray test
    """
    walls = []
    wall_z_center = wall_height / 2.0

    def _create_box(pos, half_extents):
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=pyb_client_id
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.8, 0.8, 0.8, 1.0],
            physicsClientId=pyb_client_id
        )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=pyb_client_id
        )
        return bid

    # +X
    walls.append(_create_box(
        pos=[half_size + wall_thickness, 0.0, wall_z_center],
        half_extents=[wall_thickness, half_size + wall_thickness, wall_height/2.0]
    ))
    # -X
    walls.append(_create_box(
        pos=[-half_size - wall_thickness, 0.0, wall_z_center],
        half_extents=[wall_thickness, half_size + wall_thickness, wall_height/2.0]
    ))
    # +Y
    walls.append(_create_box(
        pos=[0.0, half_size + wall_thickness, wall_z_center],
        half_extents=[half_size + wall_thickness, wall_thickness, wall_height/2.0]
    ))
    # -Y
    walls.append(_create_box(
        pos=[0.0, -half_size - wall_thickness, wall_z_center],
        half_extents=[half_size + wall_thickness, wall_thickness, wall_height/2.0]
    ))

    return walls
