# my_project/utils/geometry.py
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def dist2(xy, target_xy):
    dx = xy[0] - target_xy[0]
    dy = xy[1] - target_xy[1]
    return dx*dx + dy*dy
