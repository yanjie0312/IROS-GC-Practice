# my_project/env/sensors.py
import pybullet as p

def hit_any_wall(pyb_client_id: int, drone_id: int, wall_ids) -> bool:
    """用 contact points 判断是否撞墙"""
    for wid in wall_ids:
        contacts = p.getContactPoints(bodyA=drone_id, bodyB=wid, physicsClientId=pyb_client_id)
        if len(contacts) > 0:
            return True
    return False
