# my_project/utils/rate.py
class RateLimiter:
    """
    用 step 计数做降频：例如 1 秒一次 => every_steps = int(1.0 * ctrl_freq)
    """
    def __init__(self, every_steps: int):
        self.every_steps = max(1, int(every_steps))

    def hit(self, step: int) -> bool:
        return (step % self.every_steps) == 0
