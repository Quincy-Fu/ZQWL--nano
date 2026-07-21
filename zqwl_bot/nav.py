"""全向底盘点到点导航.

main 设定目标点 (gx,gy,gθ), 本模块独立线程固定频率算 (vx,vy,w) 发给下位机.
- 全向底盘 (麦轮/omni), vx/vy/w 为本体坐标系速度
- P 控制: 位置误差世界系→本体系 乘增益得 (vx,vy); 朝向误差乘增益得 w
- 多点依次经过, 到达判定 (位置+朝向都在容差内) 弹出下一目标
- 全局单例: init() / goto() / set_goals() / is_idle() / shutdown()
"""

import logging
import math
import threading
import time

log = logging.getLogger("zqwl.nav")

DEFAULTS = dict(
    freq=50.0,      # Hz 控制频率
    max_v=0.5,      # m/s 线速度上限
    max_w=2.0,      # rad/s 角速度上限
    kp_v=1.5,       # 位置增益
    kp_w=2.0,       # 朝向增益
    pos_tol=0.02,   # m 位置到达容差
    ang_tol=0.05,   # rad 朝向到达容差
)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sat(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


class _Nav:
    def __init__(self, **cfg):
        self.cfg = {**DEFAULTS, **cfg}
        self._goals: list[tuple[float, float, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._idle = True
        self._ser = None  # 绑 serial 模块, init 时注入

    def start(self, ser_mod=None) -> None:
        if ser_mod is None:
            from . import serial as ser_mod
        self._ser = ser_mod
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="nav", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._ser:
            try:
                self._ser.send_velocity(0.0, 0.0, 0.0)
            except Exception:
                pass

    def set_goals(self, goals) -> None:
        with self._lock:
            self._goals = list(goals)
            self._idle = (not self._goals)

    def goto(self, x: float, y: float, theta: float) -> None:
        self.set_goals([(x, y, theta)])

    def is_idle(self) -> bool:
        with self._lock:
            return self._idle

    def _step(self) -> None:
        c = self.cfg
        s = self._ser
        pose = s.get_pose()
        with self._lock:
            goals = self._goals
        if pose is None or not goals:
            try:
                s.send_velocity(0.0, 0.0, 0.0)
            except Exception as exc:
                log.warning("send_velocity failed: %s", exc)
            return
        x, y, th = pose
        gx, gy, gth = goals[0]
        dxw, dyw = gx - x, gy - y
        cos_t, sin_t = math.cos(th), math.sin(th)
        # 世界系误差 -> 本体系 (R(-θ))
        dxb = cos_t * dxw + sin_t * dyw
        dyb = -sin_t * dxw + cos_t * dyw
        dth = _wrap(gth - th)
        vx = _sat(c["kp_v"] * dxb, c["max_v"])
        vy = _sat(c["kp_v"] * dyb, c["max_v"])
        w = _sat(c["kp_w"] * dth, c["max_w"])
        if math.hypot(dxb, dyb) < c["pos_tol"] and abs(dth) < c["ang_tol"]:
            with self._lock:
                if self._goals:
                    self._goals.pop(0)
                self._idle = (not self._goals)
            vx = vy = w = 0.0
        try:
            s.send_velocity(vx, vy, w)
        except Exception as exc:
            log.warning("send_velocity failed: %s", exc)

    def _loop(self) -> None:
        dt = 1.0 / self.cfg["freq"]
        while not self._stop.is_set():
            self._step()
            time.sleep(dt)


# --- 模块级单例 ---

_nav: _Nav | None = None


def init(**cfg) -> None:
    global _nav
    if _nav is not None:
        _nav.stop()
    _nav = _Nav(**cfg)
    _nav.start()


def shutdown() -> None:
    global _nav
    if _nav is not None:
        _nav.stop()
        _nav = None


def set_goals(goals) -> None:
    if _nav is None:
        raise RuntimeError("nav not initialized, call init() first")
    _nav.set_goals(goals)


def goto(x: float, y: float, theta: float) -> None:
    set_goals([(x, y, theta)])


def is_idle() -> bool:
    if _nav is None:
        return True
    return _nav.is_idle()


# --- self-check (不接硬件) ---

class _FakeSerial:
    """模拟下位机: 本体系 (vx,vy,w) 推演世界系位姿."""

    def __init__(self):
        self.pose = (0.0, 0.0, 0.0)

    def get_pose(self):
        return self.pose

    def send_velocity(self, vx, vy, w) -> None:
        if self.pose is None:
            return
        x, y, th = self.pose
        c, s = math.cos(th), math.sin(th)
        dt = 0.02  # 50Hz
        dx = (c * vx - s * vy) * dt
        dy = (s * vx + c * vy) * dt
        self.pose = (x + dx, y + dy, _wrap(th + w * dt))


def _self_check() -> None:
    fake = _FakeSerial()
    n = _Nav()
    n.start(ser_mod=fake)
    try:
        # 1. 无 pose 时发零速不崩
        fake.pose = None
        n.goto(1.0, 0.0, 0.0)
        n._step()
        assert n.is_idle() is False

        # 2. 单点收敛
        fake.pose = (0.0, 0.0, 0.0)
        n.goto(1.0, 0.0, 0.0)
        for _ in range(2000):
            n._step()
            if n.is_idle():
                break
        x, y, th = fake.pose
        assert abs(x - 1.0) < 0.05 and abs(y) < 0.05 and abs(th) < 0.1, fake.pose
        assert n.is_idle()

        # 3. 多点序列按顺序到达
        fake.pose = (0.0, 0.0, 0.0)
        n.set_goals([(1.0, 0.0, 0.0), (1.0, 1.0, math.pi / 2), (0.0, 1.0, math.pi)])
        for _ in range(10000):
            n._step()
            if n.is_idle():
                break
        x, y, th = fake.pose
        assert abs(x) < 0.05 and abs(y - 1.0) < 0.05 and abs(th - math.pi) < 0.1, fake.pose
        assert n.is_idle()
    finally:
        n.stop()

    print("self-check OK")


if __name__ == "__main__":
    _self_check()
