"""导航命令调度器 — 发送命令、等待MCU响应、逐段执行.

上位机不再做 50Hz PID 闭环, 而是把路径拆成一系列导航命令,
依次发给下位机 (STM32 NavTask), 等待每条命令执行完毕再发下一条.

MCU 侧内置了:
  - 两阶段纠偏 (全速+低速)
  - 轴锁定运动 (TOX/TOY 防麦轮漂移)
  - 段间航向校准
  - 圆弧切线跟踪

坐标系: Blu3场坐标, +X=右, +Y=前, CW正, 单位 m / deg.

命令格式 (dict):
    {"cmd": "tox", "x": 0.295}
    {"cmd": "toy", "y": -0.662}
    {"cmd": "turnto", "yaw": 90.0}
    {"cmd": "goto", "x": 1.0, "y": 0.5}
    {"cmd": "arc", "cx": 1.055, "cy": -0.575, "r": 0.949, "a0": -5.2, "a1": 93.9}
    {"cmd": "sync", "x": 0.0, "y": 0.0}
    {"cmd": "fine", "dx": 5.0, "dy": -3.0}
    可选字段: "vision": True (段后暂停视觉纠偏), "label": "段1: HOME→(0,295)"

API:
    import zqwl_bot.nav as nav
    nav.init()
    nav.run(commands)          # 阻塞直到全部执行完
    nav.cancel()               # 中途取消
    nav.shutdown()
"""

import logging
import threading
import time

log = logging.getLogger("zqwl.nav")

# 事件类型
EV_SEG_DONE  = "segment_done"
EV_TIMEOUT   = "timeout"
EV_PATH_DONE = "path_done"
EV_ERROR     = "error"
EV_VISION    = "vision"


class PathRunner:
    """路径命令调度器.

    Parameters
    ----------
    seg_timeout : 单段超时 s (默认 30, 圆弧可设更大)
    """

    def __init__(self, seg_timeout: float = 30.0):
        self._seg_timeout = seg_timeout
        self._ser = None
        self._idle = True
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._cbs: dict[str, list] = {}
        self._vision_handler = None  # callable(index, cmd) -> None

    # ──────────── lifecycle ────────────

    def start(self, ser_mod=None) -> None:
        if ser_mod is None:
            from . import comm as ser_mod
        self._ser = ser_mod

    def stop(self) -> None:
        self._cancel.set()

    # ──────────── callback API ────────────

    def on(self, event: str, fn) -> None:
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

    # ──────────── vision ────────────

    def set_vision_handler(self, handler) -> None:
        """设置视觉纠偏回调: handler(index, cmd) -> None."""
        self._vision_handler = handler

    # ──────────── target API ────────────

    def run(self, commands: list[dict]) -> bool:
        """依次执行命令列表, 阻塞直到全部完成或取消.

        Returns: True = 全部成功, False = 有超时或被取消.
        """
        self._cancel.clear()
        all_ok = True

        with self._lock:
            self._idle = False

        for i, cmd in enumerate(commands):
            if self._cancel.is_set():
                log.info("navigation cancelled at segment %d", i)
                all_ok = False
                break

            label = cmd.get("label", f"seg {i}")
            log.info("[%d/%d] %s", i + 1, len(commands), label)

            # 选择超时: arc 段给更多时间
            timeout = self._seg_timeout
            if cmd["cmd"] == "arc":
                timeout = max(timeout, 60.0)

            ok = self._execute_one(cmd, timeout)

            if ok:
                self._emit(EV_SEG_DONE, index=i, label=label)
                # 视觉纠偏
                if cmd.get("vision") and self._vision_handler:
                    self._emit(EV_VISION, index=i)
                    try:
                        self._vision_handler(i, cmd)
                    except Exception as exc:
                        log.warning("vision handler error: %s", exc)
            else:
                log.warning("[%d/%d] TIMEOUT or FAIL: %s", i + 1, len(commands), label)
                self._emit(EV_TIMEOUT, index=i, label=label)
                all_ok = False
                # 超时不终止后续段, 继续执行

        with self._lock:
            self._idle = True

        if all_ok:
            self._emit(EV_PATH_DONE)
        return all_ok

    def cancel(self) -> None:
        self._cancel.set()

    def is_idle(self) -> bool:
        with self._lock:
            return self._idle

    # ──────────── internal ────────────

    # 命令 → 期望响应类型 (与 comm.py 常量一致, 必须核对类型防陈旧响应顶替)
    _EXPECT = {
        "tox":    0x13,   # TYPE_CMD_TOX_RESP
        "toy":    0x15,   # TYPE_CMD_TOY_RESP
        "turnto": 0x17,   # TYPE_CMD_TURNTO_RESP
        "goto":   0x11,   # TYPE_CMD_GOTO_RESP
        "arc":    0x1D,   # TYPE_CMD_ARC_RESP
        "sync":   0x1B,   # TYPE_CMD_SYNC_RESP
        "fine":   0x19,   # TYPE_CMD_FINE_RESP
    }

    def _execute_one(self, cmd: dict, timeout: float) -> bool:
        """发送一条导航命令并等待对应类型的响应. Returns True=成功."""
        c = cmd["cmd"]
        try:
            if c == "tox":
                self._ser.send_tox(cmd["x"])
            elif c == "toy":
                self._ser.send_toy(cmd["y"])
            elif c == "turnto":
                self._ser.send_turnto(cmd["yaw"])
            elif c == "goto":
                self._ser.send_goto(cmd["x"], cmd["y"])
            elif c == "arc":
                self._ser.send_arc(cmd["cx"], cmd["cy"], cmd["r"],
                                   cmd["a0"], cmd["a1"])
            elif c == "sync":
                self._ser.send_sync_pose(cmd["x"], cmd["y"])
            elif c == "fine":
                self._ser.send_fine_move(cmd["dx"], cmd["dy"])
            else:
                log.error("unknown nav cmd: %s", c)
                self._emit(EV_ERROR, error=f"unknown cmd: {c}")
                return False
        except Exception as exc:
            log.error("send failed: %s", exc)
            self._emit(EV_ERROR, error=str(exc))
            return False

        # 等待响应: 必须核对类型 (wait_for 会跳过陈旧/错类型响应)
        return self._ser.wait_for(self._EXPECT[c], timeout)


# ═══════════════ 模块级 API (单例) ═══════════════

_nav: PathRunner | None = None


def init(seg_timeout: float = 30.0, ser_mod=None) -> None:
    global _nav
    if _nav is not None:
        _nav.stop()
    _nav = PathRunner(seg_timeout=seg_timeout)
    _nav.start(ser_mod=ser_mod)


def shutdown() -> None:
    global _nav
    if _nav is not None:
        _nav.stop()
        _nav = None


def run(commands: list[dict]) -> bool:
    if _nav is None:
        raise RuntimeError("nav not initialized")
    return _nav.run(commands)


def cancel() -> None:
    if _nav is not None:
        _nav.cancel()


def is_idle() -> bool:
    if _nav is None:
        return True
    return _nav.is_idle()


def get_runner() -> PathRunner | None:
    """获取单例实例 (注册回调 / 设置视觉handler)."""
    return _nav


def on(event: str, fn) -> None:
    if _nav is not None:
        _nav.on(event, fn)


def off(event: str, fn) -> None:
    if _nav is not None:
        _nav.off(event, fn)


def set_vision_handler(handler) -> None:
    if _nav is not None:
        _nav.set_vision_handler(handler)


# ═══════════════ self-check ═══════════════

class _FakeSerial:
    """模拟下位机: 立即返回成功响应."""

    def __init__(self, always_ok: bool = True):
        self._ok = always_ok
        self.sent: list[tuple] = []

    def send_tox(self, x):      self.sent.append(("tox", x))
    def send_toy(self, y):      self.sent.append(("toy", y))
    def send_turnto(self, yaw): self.sent.append(("turnto", yaw))
    def send_goto(self, x, y):  self.sent.append(("goto", x, y))
    def send_arc(self, cx, cy, r, a0, a1):
        self.sent.append(("arc", cx, cy, r, a0, a1))
    def send_sync_pose(self, x, y):
        self.sent.append(("sync", x, y))
    def send_fine_move(self, dx, dy):
        self.sent.append(("fine", dx, dy))

    def wait_for(self, expect: int, timeout: float = 30.0) -> bool:
        return self._ok


def _self_check() -> None:
    print("nav self-check starting ...")

    fake = _FakeSerial(always_ok=True)
    runner = PathRunner(seg_timeout=5.0)
    runner.start(ser_mod=fake)

    # 1. 基本命令序列
    cmds = [
        {"cmd": "tox", "x": 0.295, "label": "seg1"},
        {"cmd": "toy", "y": -0.662, "label": "seg2"},
        {"cmd": "turnto", "yaw": 90.0, "label": "seg3"},
    ]
    events = []
    runner.on(EV_SEG_DONE, lambda **kw: events.append(kw))
    ok = runner.run(cmds)
    assert ok, "run should succeed"
    assert len(events) == 3, f"expected 3 seg_done events, got {len(events)}"
    assert len(fake.sent) == 3, f"expected 3 sends, got {len(fake.sent)}"
    assert fake.sent[0] == ("tox", 0.295)
    assert fake.sent[1] == ("toy", -0.662)
    assert fake.sent[2] == ("turnto", 90.0)
    print("  [1/4] basic command sequence OK")

    # 2. 圆弧命令
    fake.sent.clear()
    arc_cmd = {"cmd": "arc", "cx": 1.055, "cy": -0.575,
               "r": 0.949, "a0": -5.2, "a1": 93.9, "label": "arc1"}
    ok = runner.run([arc_cmd])
    assert ok
    assert fake.sent[0][0] == "arc"
    assert abs(fake.sent[0][3] - 0.949) < 0.001
    print("  [2/4] arc command OK")

    # 3. 超时处理
    fake2 = _FakeSerial(always_ok=False)  # 永远返回 None (模拟超时)
    runner2 = PathRunner(seg_timeout=0.1)
    runner2.start(ser_mod=fake2)
    timeout_fired = False
    runner2.on(EV_TIMEOUT, lambda **kw: None)
    ok = runner2.run([{"cmd": "tox", "x": 1.0}])
    assert not ok, "run should fail on timeout"
    print("  [3/4] timeout handling OK")

    # 4. 取消 (run 完成后 cancel, 验证 idle 状态)
    fake3 = _FakeSerial(always_ok=True)
    runner3 = PathRunner()
    runner3.start(ser_mod=fake3)
    ok = runner3.run([{"cmd": "tox", "x": 1.0}])
    assert ok, "first run should succeed"
    runner3.cancel()  # cancel after completion
    assert runner3.is_idle(), "should be idle after cancel"
    print("  [4/4] cancel/idle OK")

    print("self-check OK — all 4 tests passed")


if __name__ == "__main__":
    _self_check()
