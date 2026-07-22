"""全向底盘导航模块 — PID 位置环 + 直线/圆弧路径跟踪.

上位机 (Jetson Nano) 以固定频率计算 (vx, vy, w) 发给下位机 (STM32F407),
下位机持续回传里程计位姿 (x, y, θ).

功能:
  1. 点到点 — goto(x, y, θ), 航点队列 set_goals([(x,y,θ), ...])
  2. 路径跟踪 — follow_path([PathLine(...), PathArc(...), ...])
  3. PID 控制 — 位置 PID(含积分抗饱和) + 朝向 PID(微分在测量值上)
  4. 辅助 — 轨迹记录 / 段超时保护 / 事件回调

坐标系: 世界系右手, θ 从 +X 逆时针; 本体系 X 前 Y 左.
控制频率: 50 Hz (可配).  麦轮/omni 全向底盘, vx/vy/w 解耦.

API:
    import zqwl_bot.nav as nav
    nav.init()
    nav.goto(1.0, 0.5, 0.0)                 # 单点
    nav.set_goals([(1,0,0),(1,1,pi/2)])     # 多点依次
    nav.follow_path([PathLine(0,0,1,0), PathArc(1,0,1,1,ccw=True)])
    while not nav.is_idle(): time.sleep(0.1)
    nav.shutdown()
"""

import logging
import math
import threading
import time

log = logging.getLogger("zqwl.nav")


# ═══════════════ helpers ═══════════════

def _wrap(a: float) -> float:
    """角度归一化到 [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sat(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


# ═══════════════ PID 控制器 ═══════════════

class PID:
    """单轴 PID, 积分抗饱和(clamping) + 微分在测量值(避免 setpoint 阶跃冲击).

    输出 = Kp·e + Ki·∫e + Kd·(-d(measurement)/dt)
    """

    __slots__ = ("kp", "ki", "kd", "_integral", "_prev_meas")

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_meas: float | None = None

    def compute(self, err: float, meas: float, dt: float,
                out_lim: float = float("inf")) -> float:
        """
        err  : 当前误差 (setpoint − measurement)
        meas : 当前测量值 (用于微分项)
        dt   : 控制周期 (s)
        out_lim : 输出限幅 (用于积分抗饱和)
        """
        if dt <= 0.0:
            return 0.0

        # — 比例 —
        p = self.kp * err

        # — 积分 (clamping 抗饱和) —
        self._integral += err * dt
        if self.ki > 0.0:
            max_int = out_lim / self.ki
            self._integral = _sat(self._integral, max_int)
        i = self.ki * self._integral

        # — 微分 (在测量值上, 避免 setpoint 跳变引起的 kick) —
        d = 0.0
        if self._prev_meas is not None and dt > 0.0:
            d = -self.kd * (meas - self._prev_meas) / dt
        self._prev_meas = meas

        return _sat(p + i + d, out_lim)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_meas = None


class _PosPID2D:
    """本体系 X/Y 双轴独立 PID (全向底盘位置控制)."""

    __slots__ = ("_px", "_py")

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0):
        self._px = PID(kp, ki, kd)
        self._py = PID(kp, ki, kd)

    def compute(self, ex: float, ey: float,
                mx: float, my: float,
                dt: float, lim: float) -> tuple[float, float]:
        vx = self._px.compute(ex, mx, dt, lim)
        vy = self._py.compute(ey, my, dt, lim)
        return vx, vy

    def reset(self) -> None:
        self._px.reset()
        self._py.reset()


# ═══════════════ 路径段 ═══════════════

class PathLine:
    """直线段: (x0,y0) → (x1,y1).

    Parameters
    ----------
    x0, y0 : 起点坐标 (m)
    x1, y1 : 终点坐标 (m)
    heading : 目标朝向 (rad). None = 自动取行进方向.
    """

    __slots__ = ("x0", "y0", "x1", "y1", "_heading",
                 "_dx", "_dy", "_len", "_ux", "_uy")

    def __init__(self, x0: float, y0: float,
                 x1: float, y1: float, heading: float | None = None):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1
        self._heading = heading
        self._dx = x1 - x0
        self._dy = y1 - y0
        self._len = math.hypot(self._dx, self._dy)
        if self._len > 1e-6:
            self._ux = self._dx / self._len
            self._uy = self._dy / self._len
        else:
            self._ux = self._uy = 0.0

    def nearest(self, px: float, py: float, lookahead: float
                ) -> tuple[float, float, float | None, bool]:
        """前视点查询.  Returns (tx, ty, target_heading, done)."""
        if self._len < 1e-6:
            # 零长度段: 视为原地转向
            h = self._heading if self._heading is not None else None
            return self.x1, self.y1, h, True

        # 投影参数 t ∈ [0, 1]
        t = ((px - self.x0) * self._ux + (py - self.y0) * self._uy) / self._len
        t = max(0.0, min(1.0, t))

        # 沿路径前进 lookahead
        s = t * self._len + lookahead
        if s >= self._len:
            # 已过终点
            h = self._heading if self._heading is not None else math.atan2(self._dy, self._dx)
            return self.x1, self.y1, h, True
        t2 = s / self._len
        tx = self.x0 + t2 * self._dx
        ty = self.y0 + t2 * self._dy
        h = self._heading if self._heading is not None else math.atan2(self._dy, self._dx)
        return tx, ty, h, False


class PathArc:
    """圆弧段: 绕 center(cx,cy) 从 start_pos 旋转.

    半径 = dist(center, start_pos) 自动算.
    CCW (ccw=True): θ 递增;  CW (ccw=False): θ 递减.
    切向朝向 = 行进方向切线 (可通过 heading 参数覆盖, 暂未实现).

    Parameters
    ----------
    cx, cy : 圆心坐标 (m)
    sx, sy : 圆弧起点坐标 (m)
    ex, ey : 圆弧终点坐标 (m)
    ccw : True=逆时针, False=顺时针
    """

    __slots__ = ("cx", "cy", "r", "_a0", "_a1", "_sweep",
                 "_ccw", "_total_arc")

    def __init__(self, cx: float, cy: float,
                 sx: float, sy: float,
                 ex: float, ey: float, ccw: bool = True):
        self.cx, self.cy = cx, cy
        self.r = math.hypot(sx - cx, sy - cy)
        self._a0 = math.atan2(sy - cy, sx - cx)
        a_end = math.atan2(ey - cy, ex - cx)
        if ccw:
            self._sweep = _wrap(a_end - self._a0)
            if self._sweep < 0:
                self._sweep += 2 * math.pi
        else:
            self._sweep = _wrap(a_end - self._a0)
            if self._sweep > 0:
                self._sweep -= 2 * math.pi
        self._a1 = self._a0 + self._sweep
        self._ccw = ccw
        self._total_arc = self.r * abs(self._sweep)

    def nearest(self, px: float, py: float, lookahead: float
                ) -> tuple[float, float, float | None, bool]:
        """前视点查询.  Returns (tx, ty, target_heading, done)."""
        if self.r < 1e-6:
            return self.cx, self.cy, None, True

        # 当前位置在圆上的投影角度
        alpha = math.atan2(py - self.cy, px - self.cx)

        # 归一化到弧的角度范围
        d = _wrap(alpha - self._a0)
        if self._ccw:
            if d < 0:
                d = 0.0
            elif d > self._sweep:
                d = self._sweep
        else:
            if d > 0:
                d = 0.0
            elif d < self._sweep:
                d = self._sweep

        cur_angle = self._a0 + d
        arc_covered = abs(d) * self.r
        arc_remain = self._total_arc - arc_covered

        # 前视
        la_angle = lookahead / self.r if self.r > 1e-6 else 0.0
        if arc_remain <= lookahead:
            # 将到达或已过终点
            tx = self.cx + self.r * math.cos(self._a1)
            ty = self.cy + self.r * math.sin(self._a1)
            if self._ccw:
                h = _wrap(self._a1 + math.pi / 2)
            else:
                h = _wrap(self._a1 - math.pi / 2)
            return tx, ty, h, True

        if self._ccw:
            tgt_angle = cur_angle + la_angle
            h = _wrap(tgt_angle + math.pi / 2)   # CCW 切向 = 径向左转 90°
        else:
            tgt_angle = cur_angle - la_angle
            h = _wrap(tgt_angle - math.pi / 2)   # CW 切向 = 径向右转 90°

        tx = self.cx + self.r * math.cos(tgt_angle)
        ty = self.cy + self.r * math.sin(tgt_angle)
        return tx, ty, h, False


# ═══════════════ 导航器 ═══════════════

# 事件类型字符串
EV_WAYPOINT = "waypoint_reached"
EV_SEG_DONE = "segment_done"
EV_TIMEOUT = "timeout"
EV_PATH_DONE = "path_done"
EV_ERROR = "error"


class Navigator:
    """全向底盘导航器.

    Parameters
    ----------
    freq     : 控制频率 Hz (默认 50)
    max_v    : 线速度上限 m/s (默认 0.5)
    max_w    : 角速度上限 rad/s (默认 2.0)
    kp_v     : 位置比例增益 (默认 1.5)
    ki_v     : 位置积分增益 (默认 0.1, 消除稳态误差)
    kd_v     : 位置微分增益 (默认 0.05, 抑制超调)
    kp_w     : 朝向比例增益 (默认 2.0)
    ki_w     : 朝向积分增益 (默认 0.0)
    kd_w     : 朝向微分增益 (默认 0.1)
    pos_tol  : 位置到达容差 m (默认 0.02)
    ang_tol  : 朝向到达容差 rad (默认 0.05)
    lookahead: 路径前视距离 m (默认 0.10)
    v_cruise : 路径巡航速度 m/s (默认 0.3)
    seg_timeout : 单段超时 s (0=不启用, 默认 10)
    wp_tol   : 航点模式相邻点最大间距 m, 超出报警 (默认 3.0)
    """

    def __init__(self, **cfg):
        # 控制参数
        self._freq = cfg.get("freq", 50.0)
        self._max_v = cfg.get("max_v", 0.5)
        self._max_w = cfg.get("max_w", 2.0)
        self._pos_tol = cfg.get("pos_tol", 0.02)
        self._ang_tol = cfg.get("ang_tol", 0.05)
        self._lookahead = cfg.get("lookahead", 0.10)
        self._v_cruise = cfg.get("v_cruise", 0.3)
        self._seg_timeout = cfg.get("seg_timeout", 10.0)
        self._wp_tol = cfg.get("wp_tol", 3.0)

        # PID
        self._pos_pid = _PosPID2D(
            kp=cfg.get("kp_v", 1.5),
            ki=cfg.get("ki_v", 0.1),
            kd=cfg.get("kd_v", 0.05),
        )
        self._ang_pid = PID(
            kp=cfg.get("kp_w", 2.0),
            ki=cfg.get("ki_w", 0.0),
            kd=cfg.get("kd_w", 0.1),
        )

        # 状态
        self._goals: list[tuple[float, float, float]] = []
        self._path: list[PathLine | PathArc] | None = None
        self._seg_idx = 0
        self._idle = True
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ser = None

        # 轨迹
        self._traj: list[tuple[float, float, float, float]] = []
        self._traj_on = False

        # 回调
        self._cbs: dict[str, list] = {}

        # 计时
        self._seg_t0 = 0.0

    # ──────────── lifecycle ────────────

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

    # ──────────── target API ────────────

    def set_goals(self, goals: list[tuple[float, float, float]]) -> None:
        """设航点队列 [(x,y,θ), ...], 清空路径."""
        with self._lock:
            self._goals = list(goals)
            self._path = None
            self._idle = not self._goals
            self._reset_pids()
            self._seg_t0 = time.monotonic()
        self._check_wp_dist(goals)

    def goto(self, x: float, y: float, theta: float) -> None:
        """单航点."""
        self.set_goals([(x, y, theta)])

    def follow_path(self, segments: list[PathLine | PathArc]) -> None:
        """路径跟踪: 直线 + 圆弧组合. 路径段应首尾相接."""
        with self._lock:
            self._path = list(segments)
            self._goals = []
            self._seg_idx = 0
            self._idle = not self._path
            self._reset_pids()
            self._seg_t0 = time.monotonic()

    def cancel(self) -> None:
        """取消当前导航, 发零速."""
        with self._lock:
            self._goals = []
            self._path = None
            self._idle = True
        self._send_vel(0.0, 0.0, 0.0)

    def is_idle(self) -> bool:
        with self._lock:
            return self._idle

    # ──────────── callback API ────────────

    def on(self, event: str, fn) -> None:
        """注册回调.  event: waypoint_reached / segment_done / timeout / path_done / error."""
        self._cbs.setdefault(event, []).append(fn)

    def off(self, event: str, fn) -> None:
        lst = self._cbs.get(event, [])
        if fn in lst:
            lst.remove(fn)

    def _emit(self, event: str, **kw) -> None:
        for fn in self._cbs.get(event, []):
            try:
                fn(**kw)
            except Exception as exc:
                log.warning("callback error [%s]: %s", event, exc)

    # ──────────── trajectory API ────────────

    def start_log(self) -> None:
        self._traj = []
        self._traj_on = True

    def stop_log(self) -> None:
        self._traj_on = False

    def get_traj(self) -> list[tuple[float, float, float, float]]:
        return list(self._traj)

    # ──────────── internal: helpers ────────────

    def _reset_pids(self) -> None:
        self._pos_pid.reset()
        self._ang_pid.reset()

    def _send_vel(self, vx: float, vy: float, w: float) -> None:
        try:
            self._ser.send_velocity(vx, vy, w)
        except Exception as exc:
            log.warning("send_velocity failed: %s", exc)
            self._emit(EV_ERROR, error=str(exc))

    def _check_wp_dist(self, goals: list) -> None:
        """检查相邻航点间距."""
        pose = self._ser.get_pose()
        prev = (pose[0], pose[1]) if pose else (0.0, 0.0)
        for g in goals:
            d = math.hypot(g[0] - prev[0], g[1] - prev[1])
            if d > self._wp_tol:
                log.warning("waypoint gap %.2fm (>%.1f): (%.2f,%.2f) → (%.2f,%.2f)",
                            d, self._wp_tol, prev[0], prev[1], g[0], g[1])
                self._emit(EV_ERROR,
                           error=f"waypoint gap {d:.2f}m",
                           distance=d, goal=g)
            prev = (g[0], g[1])

    def _get_path_endpoint(self, segs) -> tuple[float, float, float] | None:
        """取路径末段终点坐标."""
        if not segs:
            return None
        last = segs[-1]
        if isinstance(last, PathLine):
            return (last.x1, last.y1, last._heading or 0.0)
        elif isinstance(last, PathArc):
            ex = last.cx + last.r * math.cos(last._a1)
            ey = last.cy + last.r * math.sin(last._a1)
            if last._ccw:
                h = _wrap(last._a1 + math.pi / 2)
            else:
                h = _wrap(last._a1 - math.pi / 2)
            return (ex, ey, h)
        return None

    def _at_point(self, x, y, th, target) -> bool:
        """判断是否在目标点容差内."""
        return math.hypot(target[0] - x, target[1] - y) < self._pos_tol

    # ──────────── internal: step logic ────────────

    def _step(self) -> None:
        """单次控制周期.  path 优先于 goals."""
        pose = self._ser.get_pose()
        if pose is None:
            self._send_vel(0.0, 0.0, 0.0)
            return

        x, y, th = pose
        now = time.monotonic()

        # 轨迹记录
        if self._traj_on:
            self._traj.append((now, x, y, th))

        with self._lock:
            has_path = self._path is not None and len(self._path) > 0
            has_goals = bool(self._goals)

        if not has_path and not has_goals:
            self._send_vel(0.0, 0.0, 0.0)
            return

        if has_path:
            self._step_path(x, y, th, now)
        else:
            self._step_p2p(x, y, th, now)

    def _step_p2p(self, x: float, y: float, th: float, now: float) -> None:
        """点到点 (PID)."""
        with self._lock:
            if not self._goals:
                return
            gx, gy, gth = self._goals[0]

        # 超时检测
        if self._seg_timeout > 0 and (now - self._seg_t0) > self._seg_timeout:
            log.warning("p2p timeout → (%.2f, %.2f)", gx, gy)
            self._emit(EV_TIMEOUT, goal=(gx, gy, gth))
            with self._lock:
                self._goals.pop(0)
                self._idle = not self._goals
                self._reset_pids()
                self._seg_t0 = now
            self._send_vel(0.0, 0.0, 0.0)
            return

        # 世界系 → 本体系
        dxw, dyw = gx - x, gy - y
        ct, st = math.cos(th), math.sin(th)
        dxb = ct * dxw + st * dyw
        dyb = -st * dxw + ct * dyw
        dth = _wrap(gth - th)

        dt = 1.0 / self._freq
        vx, vy = self._pos_pid.compute(dxb, dyb, x, y, dt, self._max_v)
        w = self._ang_pid.compute(dth, th, dt, self._max_w)

        # 到达判定
        if math.hypot(dxb, dyb) < self._pos_tol and abs(dth) < self._ang_tol:
            self._emit(EV_WAYPOINT, goal=(gx, gy, gth))
            with self._lock:
                self._goals.pop(0)
                self._idle = not self._goals
                self._reset_pids()
                self._seg_t0 = now
            self._send_vel(0.0, 0.0, 0.0)
            return

        self._send_vel(vx, vy, w)

    def _step_path(self, x: float, y: float, th: float, now: float) -> None:
        """路径跟踪 (PID + 前视点)."""
        with self._lock:
            segs = self._path
            idx = self._seg_idx

        if segs is None or idx >= len(segs):
            # 路径段全部走完 → settle 到末端
            endpoint = self._get_path_endpoint(segs)
            with self._lock:
                self._path = None
                if endpoint and not self._at_point(x, y, th, endpoint):
                    # 切到点到点收敛
                    self._goals = [endpoint]
                    self._idle = False
                    self._reset_pids()
                    self._seg_t0 = now
                else:
                    self._emit(EV_PATH_DONE)
                    self._idle = True
            self._send_vel(0.0, 0.0, 0.0)
            return

        seg = segs[idx]

        # 超时
        if self._seg_timeout > 0 and (now - self._seg_t0) > self._seg_timeout:
            log.warning("path segment %d timeout", idx)
            self._emit(EV_TIMEOUT, segment_index=idx)
            with self._lock:
                self._seg_idx += 1
                if self._seg_idx >= len(segs):
                    self._path = None
                    self._idle = True
                self._reset_pids()
                self._seg_t0 = now
            self._send_vel(0.0, 0.0, 0.0)
            return

        # 前视点查询
        tx, ty, t_h, seg_done = seg.nearest(x, y, self._lookahead)

        # 目标朝向
        if t_h is None:
            dxh, dyh = tx - x, ty - y
            if math.hypot(dxh, dyh) > 0.01:
                t_h = math.atan2(dyh, dxh)
            else:
                t_h = th  # 原地保持

        # 位置误差 → 本体系
        dxw, dyw = tx - x, ty - y
        ct, st = math.cos(th), math.sin(th)
        dxb = ct * dxw + st * dyw
        dyb = -st * dxw + ct * dyw

        # 朝向误差
        dth = _wrap(t_h - th)

        # PID 计算
        dt = 1.0 / self._freq
        vx, vy = self._pos_pid.compute(dxb, dyb, x, y, dt, self._max_v)
        w = self._ang_pid.compute(dth, th, dt, self._max_w)

        # 速度缩放: 路径巡航速度
        spd = math.hypot(vx, vy)
        if spd > self._v_cruise:
            r = self._v_cruise / spd
            vx *= r
            vy *= r

        # 段完成 → 切换下一段
        if seg_done:
            self._emit(EV_SEG_DONE, segment_index=idx)
            with self._lock:
                self._seg_idx += 1
                self._reset_pids()
                self._seg_t0 = now
            # 末段完成会在下次 _step_path 走 settle 分支

        self._send_vel(vx, vy, w)

    # ──────────── loop ────────────

    def _loop(self) -> None:
        dt = 1.0 / self._freq
        while not self._stop.is_set():
            try:
                self._step()
            except Exception as exc:
                log.error("nav step error: %s", exc, exc_info=True)
                self._emit(EV_ERROR, error=str(exc))
                self._send_vel(0.0, 0.0, 0.0)
            time.sleep(dt)


# ═══════════════ 模块级 API (单例) ═══════════════

_nav: Navigator | None = None


def init(**cfg) -> None:
    global _nav
    if _nav is not None:
        _nav.stop()
    _nav = Navigator(**cfg)
    _nav.start()


def shutdown() -> None:
    global _nav
    if _nav is not None:
        _nav.stop()
        _nav = None


def set_goals(goals: list[tuple[float, float, float]]) -> None:
    if _nav is None:
        raise RuntimeError("nav not initialized")
    _nav.set_goals(goals)


def goto(x: float, y: float, theta: float) -> None:
    set_goals([(x, y, theta)])


def follow_path(segments: list) -> None:
    if _nav is None:
        raise RuntimeError("nav not initialized")
    _nav.follow_path(segments)


def cancel() -> None:
    if _nav is not None:
        _nav.cancel()


def is_idle() -> bool:
    if _nav is None:
        return True
    return _nav.is_idle()


def get_navigator() -> Navigator | None:
    """获取单例实例 (注册回调 / 轨迹记录)."""
    return _nav


# ═══════════════ self-check (不接硬件) ═══════════════

class _FakeSerial:
    """模拟下位机: 全向运动学积分 (vx,vy,w) → (x,y,θ)."""

    def __init__(self):
        self.pose = (0.0, 0.0, 0.0)
        self.dt = 0.02  # 50 Hz

    def get_pose(self):
        return self.pose

    def send_velocity(self, vx, vy, w) -> None:
        if self.pose is None:
            return
        x, y, th = self.pose
        c, s = math.cos(th), math.sin(th)
        dx = (c * vx - s * vy) * self.dt
        dy = (s * vx + c * vy) * self.dt
        self.pose = (x + dx, y + dy, _wrap(th + w * self.dt))


def _run_steps(nav: Navigator, fake: _FakeSerial, max_steps: int,
               check_idle: bool = True) -> int:
    """执行 nav._step() 直到 idle 或超限. 返回实际步数."""
    for i in range(max_steps):
        nav._step()
        if check_idle and nav.is_idle():
            return i + 1
    return max_steps


def _self_check() -> None:
    print("nav self-check starting ...")

    # ── 1. PID 收敛 ──
    pid = PID(kp=1.5, ki=0.1, kd=0.05)
    meas = 0.0
    for _ in range(500):
        err = 1.0 - meas
        out = pid.compute(err, meas, 0.02, 1.0)
        meas += out * 0.02
    assert abs(meas - 1.0) < 0.05, f"PID converge: {meas:.4f}"
    print("  [1/10] PID convergence OK")

    # ── 2. 点到点收敛 ──
    fake = _FakeSerial()
    fake.pose = (0.0, 0.0, 0.0)
    n = Navigator()
    n.start(ser_mod=fake)
    try:
        n.goto(1.0, 0.0, 0.0)
        _run_steps(n, fake, 3000)
        x, y, th = fake.pose
        assert abs(x - 1.0) < 0.05 and abs(y) < 0.05, f"p2p pos: ({x:.3f},{y:.3f})"
        assert n.is_idle(), "p2p idle"
        print("  [2/10] point-to-point OK")

        # ── 3. 直线路径跟踪 ──
        fake.pose = (0.0, 0.1, 0.0)   # 起点有 0.1m 横向偏移
        n.follow_path([PathLine(0.0, 0.0, 1.0, 0.0)])
        _run_steps(n, fake, 3000)
        x, y, th = fake.pose
        assert abs(x - 1.0) < 0.08 and abs(y) < 0.08, f"line: ({x:.3f},{y:.3f})"
        assert n.is_idle(), "line idle"
        print("  [3/10] straight-line path OK")

        # ── 4. 圆弧路径跟踪 ──
        fake.pose = (1.0, 0.0, math.pi / 2)
        arc = PathArc(cx=0.0, cy=0.0, sx=1.0, sy=0.0,
                      ex=0.0, ey=1.0, ccw=True)
        n.follow_path([arc])
        _run_steps(n, fake, 5000)
        x, y, th = fake.pose
        assert abs(x) < 0.15 and abs(y - 1.0) < 0.15, f"arc: ({x:.3f},{y:.3f})"
        assert n.is_idle(), "arc idle"
        print("  [4/10] arc path OK")

        # ── 5. 多段路径 (直线+直线) ──
        fake.pose = (0.0, 0.0, 0.0)
        n.follow_path([
            PathLine(0.0, 0.0, 1.0, 0.0),
            PathLine(1.0, 0.0, 1.0, 1.0),
        ])
        _run_steps(n, fake, 6000)
        x, y, th = fake.pose
        assert abs(x - 1.0) < 0.12 and abs(y - 1.0) < 0.12, f"multi: ({x:.3f},{y:.3f})"
        assert n.is_idle(), "multi idle"
        print("  [5/10] multi-segment path OK")

        # ── 6. 超时保护 ──
        fake.pose = (0.0, 0.0, 0.0)
        n2 = Navigator(seg_timeout=0.1)
        n2._ser = fake  # 手动注入, 不启线程, 避免 race
        timeout_fired = False
        n2.on(EV_TIMEOUT, lambda **kw: None)
        n2.goto(5.0, 0.0, 0.0)
        for _ in range(300):
            time.sleep(0.025)  # 模拟真实 50Hz 节奏
            n2._step()
            if n2.is_idle():
                timeout_fired = True
                break
        assert timeout_fired, "timeout must fire"
        assert n2.is_idle(), "idle after timeout"
        print("  [6/10] timeout protection OK")

        # ── 7. 轨迹记录 ──
        fake.pose = (0.0, 0.0, 0.0)
        n.start_log()
        n.goto(0.5, 0.0, 0.0)
        _run_steps(n, fake, 1000)
        traj = n.get_traj()
        n.stop_log()
        assert len(traj) > 10, f"traj len: {len(traj)}"
        assert len(traj[0]) == 4, "traj tuple format"
        print(f"  [7/10] trajectory log OK ({len(traj)} pts)")

        # ── 8. 事件回调 ──
        fake.pose = (0.0, 0.0, 0.0)
        events = []
        n.on(EV_WAYPOINT, lambda **kw: events.append(("wp", kw)))
        n.on(EV_PATH_DONE, lambda **kw: events.append(("done", kw)))
        n.goto(0.3, 0.0, 0.0)
        _run_steps(n, fake, 1000)
        assert any(e[0] == "wp" for e in events), "waypoint callback"
        n.off(EV_WAYPOINT, events.append)
        n.off(EV_PATH_DONE, events.append)
        print(f"  [8/10] callbacks OK ({len(events)} events)")

        # ── 9. 航点间距警告 ──
        fake.pose = (0.0, 0.0, 0.0)
        warnings = []
        n3 = Navigator(wp_tol=1.0)
        n3._ser = fake  # 手动注入, 不启线程
        n3.on(EV_ERROR, lambda **kw: warnings.append(kw))
        n3.set_goals([(0.5, 0.0, 0.0), (5.0, 0.0, 0.0)])
        assert len(warnings) > 0, "gap warning must fire"
        print(f"  [9/10] waypoint gap warning OK ({len(warnings)} warnings)")

        # ── 10. 取消导航 ──
        fake.pose = (0.0, 0.0, 0.0)
        n.goto(2.0, 0.0, 0.0)
        for _ in range(10):
            n._step()
        n.cancel()
        assert n.is_idle(), "cancel → idle"
        print("  [10/10] cancel OK")

    finally:
        n.stop()

    print("self-check OK — all 10 tests passed")


if __name__ == "__main__":
    _self_check()
