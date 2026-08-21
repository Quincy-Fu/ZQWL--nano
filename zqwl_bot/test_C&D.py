#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_C&D.py - C/D 区新流程测试。

流程按实车调试顺序写死：先同步位姿，二维码识别得到 A/B/C/D/E 目标颜色，
再转到 -90° 准备机械臂/补光灯/USB，按实车经过顺序识别 5 个物块并同步收纳，
随后按 A/B/D/C/E 顺序到点，先按颜色切转盘，再调用 ring 完成定位放置。
放置段只走横移+直行，不走斜线；C 点按用户要求先走 Y 再走 X。
"""

import math
import os
import sys
import threading
import time
from typing import Optional

# 必须在任何视觉模块导入 cv2 前设置，避免 USB 异常时 V4L2 select 默认等待约 10 秒。
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")
os.environ.setdefault("OPENCV_VIDEOIO_V4L_READ_ATTEMPTS", "1")

import comm
import qr1
import block
import ring


BAUDRATE = 115200
RUN_START_ARGS = {"run", "--run", "wait-run", "--wait-run"}
RUN_GATE_TIMEOUT = 5.0
QR1_RECOGNIZE_TIMEOUT_S = 18.0

INIT_X = 0.0
INIT_Y = 0.0
INIT_YAW = 90.0

KEY_PATH_SPEED = 0.50
KEY_PATH_MEDIUM_SPEED = 0.40
KEY_PATH_MEDIUM_SEG_M = 0.60
KEY_PATH_LONG_SPEED = 0.35
KEY_PATH_LONG_SEG_M = 0.80
KEY_PATH_XY_EPS = 1e-4
KEY_PATH_SEG_TIMEOUT_MIN = 25.0
KEY_PATH_TURN_TIMEOUT = 25.0
KEY_PATH_ROTATE_TIMEOUT = 12.0
AXIS_MOVE_EPS = 1e-4

CD_ARC_START = (-0.900, 0.250)
CD_ARC_START_YAW = -69.0
CD_ARC_RADIUS = 0.869
CD_ARC_DIR = 1
CD_ARC_SWEEP_DEG = 130.0
# C/D圆弧速度预设：需要快速切回0.30时，只改 CD_ARC_PROFILE = "stable_030"。
CD_ARC_PROFILE = "stable_030"
CD_ARC_PROFILES = {
    # 0.45备份版：圆弧提速，识别整体后移；第5个物块同样在圆弧后、转180前切状态5。
    "fast_045": {
        "speed": 0.45,
        "recognize_offset_s": 1.20,
        "rotate_delay_s": 0.30,
        "per_point_rotate_extra_delay_s": {},
        "state5_before_end_s": 0.10,
        "per_point_extra_delay_s": {4: 0.30, 5: 0.30},
        "allow_post_arc_windows": True,
        "post_monitor_grace_s": 2.50,
    },
    # 备份稳定版：原0.30m/s配置；第5个物块不在圆弧中定时转盘，圆弧后转180前切状态5。
    "stable_030": {
        "speed": 0.30,
        "recognize_offset_s": 1.00,
        "rotate_delay_s": 0.30,
        "per_point_rotate_extra_delay_s": {},
        "state5_before_end_s": 0.10,
        "per_point_extra_delay_s": {},
        "allow_post_arc_windows": True,
        "post_monitor_grace_s": 2.50,
    },
}
CD_ARC_CFG = CD_ARC_PROFILES[CD_ARC_PROFILE]
CD_ARC_SPEED = CD_ARC_CFG["speed"]
CD_USE_FIXED_ARC = False
CD_FIXED_ARC_TIMEOUT_S = 18.0
CD_FIXED_PRE_PHASE_S = 1.20
CD_SLOT_COUNT = 5
CD_LOAD_ROTATE_LIMIT = 4
CD_ARC_POSE_POLL_S = 0.02
CD_ARM_READY_SETTLE_S = 0.80
CD_FIRST_BLOCK_STABLE_OBSERVE_S = 0.35
CD_DISPLAY_ONCE_WINDOW_S = 0.80
CD_BLOCK_READY_TIMEOUT_S = 0.7
CD_BLOCK_RESTART_TIMEOUT_S = 0.9
CD_ARC_RECOGNIZE_TIME_OFFSET_S = CD_ARC_CFG["recognize_offset_s"]
CD_ARC_PER_POINT_EXTRA_DELAY_S = CD_ARC_CFG["per_point_extra_delay_s"]
CD_ARC_PER_POINT_ROTATE_EXTRA_DELAY_S = CD_ARC_CFG.get("per_point_rotate_extra_delay_s", {})
CD_ARC_TIME_WINDOW_HALF_S = 0.90
# 转盘切槽时机：识别中心时间后固定延迟。
# 当前值由 CD_ARC_PROFILE 选择，避免0.30/0.45两套时间参数混在一起。
CD_ARC_ROTATE_AFTER_CENTER_DELAY_S = CD_ARC_CFG["rotate_delay_s"]
# 旧圆弧内转盘钳位逻辑保留参数；当前第5个状态5在 turn_180_after_arc() 中下发。
CD_ARC_STATE5_BEFORE_END_S = CD_ARC_CFG["state5_before_end_s"]
CD_ARC_ALLOW_POST_ARC_WINDOWS = CD_ARC_CFG["allow_post_arc_windows"]
CD_ARC_POST_MONITOR_GRACE_S = CD_ARC_CFG["post_monitor_grace_s"]
CD_ARC_ACTIVE_EXCLUDE_COLORS = set()
CD_ARC_COLOR_POINTS = [
    ("第2个物块", -1.1184, 0.3751),
    ("第3个物块", -1.4294, 0.7839),
    ("第4个物块", -1.3818, 1.2153),
    ("第5个物块", -1.1385, 1.5952),
]

KEY_PATH_POINTS = [
    (-0.662, 0.250, -90.0),
    (-0.900, 0.250, -90.0),
    (-0.900, 0.250, -55.0),
    (-1.222, 0.480, -55.0),
    (-1.222, 0.480, -25.0),
    (-1.412, 0.883, -25.0),
    (-1.412, 0.883,   0.0),
    (-1.415, 1.329,   0.0),
    (-1.415, 1.329,  30.0),
    (-1.160, 1.755,  30.0),
]
KEY_PATH_ROTATE_SLOTS = [1, 2, 3, 4]

TARGET_POINTS = {
    "A": (-0.630, 1.630),
    "B": (-0.430, 1.340),
    "C": ( 0.210, 1.620),
    "D": ( 0.330, 1.340),
    "E": ( 0.820, 0.600),
}
RING_PUSH_FORWARD_M = 0.098
CD_RING_BACK_SETTLE_S = 0.15
CD_RING_BACK_ZERO_VEL_SETTLE_S = 0.08
CD_RING_BODY_BACK_TIMEOUT_S = 45.0
# BODY_POS 完成后给电机驱动器留出额外稳定时间，再同步理论坐标。
BODY_POS_POST_SETTLE_S = 0.30
RING_BACK_MM_BY_POINT = {
    "A": 200.0,
    "B": 200.0,
    "C": 200.0,
    "D": 395.0,
    "E": 460.0,
}
PLACE_BACK_TARGET_POINTS = {
    "A": (-0.630, 1.720),
    "B": (-0.420, 1.440),
    "C": ( 0.220, 1.700),
    # D 点后退后落到 C 点的 Y 坐标，下一步只需要走 X。
    "D": ( 0.330, TARGET_POINTS["C"][1]),
    # E 点朝向 0°，前推后直退 460mm，最终接近 C/D 结束区。
    "E": ( 0.820, TARGET_POINTS["E"][1] + RING_PUSH_FORWARD_M - 0.460),
}
PLACE_SYNC_POINTS = {
    # ring 前推后的理论坐标，仅保留作调试参考；主流程现在后退后再同步到 PLACE_BACK_TARGET_POINTS。
    "A": (-0.630, PLACE_BACK_TARGET_POINTS["A"][1] - 0.200),
    "B": (-0.420, PLACE_BACK_TARGET_POINTS["B"][1] - 0.200),
    "C": ( 0.220, PLACE_BACK_TARGET_POINTS["C"][1] - 0.200),
    "D": ( 0.330, PLACE_BACK_TARGET_POINTS["D"][1] - 0.395),
    "E": ( 0.820, TARGET_POINTS["E"][1] + RING_PUSH_FORWARD_M),
}
E_EXIT_X_SHIFT_M = -0.240
E_EXIT_YAW = 93.0
QR_TARGET_ORDER = ["A", "B", "C", "D", "E"]
ROTATE_SPECIAL_STATE = 5

POST_PATH_AFTER_ARM0 = [
    (0.600, 0.200),
]

_CUR_X = 0.0
_CUR_Y = 0.0


def _update_cur(x: float, y: float) -> None:
    global _CUR_X, _CUR_Y
    _CUR_X, _CUR_Y = float(x), float(y)


def _start_by_run_protocol() -> bool:
    """是否要求先通过 RUN 协议确认，再启动整套 C/D 流程。"""
    return any(arg.lower() in RUN_START_ARGS for arg in sys.argv[1:])


def _port_arg() -> Optional[str]:
    """命令行里除 run/wait-run 之外的第一个参数视为串口。"""
    for arg in sys.argv[1:]:
        if arg.lower() in RUN_START_ARGS:
            continue
        return arg
    return None


def serial_port_config() -> Optional[str]:
    """返回命令行串口覆盖；没有覆盖时交给 comm.py 自动匹配。"""
    return _port_arg()


def _timed(label: str, func, *args, **kwargs):
    t0 = time.monotonic()
    print(f"[TIMING] {label} start", flush=True)
    result = func(*args, **kwargs)
    print(f"[TIMING] {label} done {time.monotonic() - t0:.3f}s -> {result}", flush=True)
    return result


def _timing_mark(label: str, t0: float) -> float:
    """打印阶段内耗时标记，并返回新的计时起点。"""
    now = time.monotonic()
    print(f"[TIMING] {label} +{now - t0:.3f}s", flush=True)
    return now


def _require(ok: bool, label: str) -> None:
    if not ok:
        raise RuntimeError(f"{label} failed")


def wait_run_start_if_requested() -> None:
    """命令行带 run 时，先通过已有 RUN 协议确认启动，再执行整套流程。"""
    if not _start_by_run_protocol():
        return

    print("\n=== RUN 协议启动门 ===")
    print(f"  发送 RUN，等待 RUN_RESP，timeout={RUN_GATE_TIMEOUT:.1f}s")
    _require(comm.run(RUN_GATE_TIMEOUT), "RUN start gate")
    print("  RUN_RESP OK，开始执行 C/D 全流程")
    time.sleep(0.2)


def sync_pose(x: float, y: float, yaw_deg: Optional[float] = None) -> bool:
    yaw_text = "keep" if yaw_deg is None else f"{yaw_deg:.1f}"
    ok = _timed(
        f"SYNC pose ({x:.3f}, {y:.3f}, yaw={yaw_text})",
        comm.sync_pose,
        x,
        y,
        yaw_deg,
        timeout=5.0,
    )
    if ok:
        _update_cur(x, y)
    return ok


def go_to(x: float, y: float, timeout: float = 40.0) -> bool:
    print(f"  GOTO ({x:.4f}, {y:.4f})")
    ok = _timed(f"GOTO ({x:.4f}, {y:.4f})", comm.goto, x, y, timeout=timeout)
    if ok:
        _update_cur(x, y)
    return ok


def _current_yaw(default: float = 0.0) -> float:
    """读取当前yaw；path只用它保持当前朝向，读不到时用保守默认值。"""
    pose = comm.get_pose(max_age=1.0)
    if pose is None:
        return default
    return float(pose[2])


def _key_path_points(points: list[tuple[float, float, float]],
                     label: str,
                     speed: float = KEY_PATH_SPEED) -> bool:
    """用下位机 key_path 串联短分段，减少多条GOTO之间的停顿。"""
    compact: list[tuple[float, float, float]] = []
    for p in points:
        if compact:
            last = compact[-1]
            if (abs(last[0] - p[0]) <= AXIS_MOVE_EPS and
                    abs(last[1] - p[1]) <= AXIS_MOVE_EPS and
                    abs(last[2] - p[2]) <= 0.1):
                continue
        compact.append(p)
    points = compact
    if len(points) < 2:
        return True
    print(f"  KEY_PATH {label}: {len(points)} pts, speed={speed:.2f}")
    ok = _timed(f"KEY_PATH {label}", comm.key_path, points,
                speed=speed, prepend_current=False)
    if ok:
        _update_cur(points[-1][0], points[-1][1])
    return ok


def _max_move_segment_len(points: list[tuple[float, float, float]]) -> float:
    """计算path中最长平移段，忽略重复坐标的原地转角点。"""
    max_len = 0.0
    for a, b in zip(points, points[1:]):
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg_len > max_len:
            max_len = seg_len
    return max_len


def _axis_path_speed(points: list[tuple[float, float, float]]) -> float:
    """中长平移段需要更长制动距离；分档降速换末端稳定。"""
    max_seg = _max_move_segment_len(points)
    if max_seg >= KEY_PATH_LONG_SEG_M:
        return KEY_PATH_LONG_SPEED
    if max_seg >= KEY_PATH_MEDIUM_SEG_M:
        return KEY_PATH_MEDIUM_SPEED
    return KEY_PATH_SPEED


def _validate_axis_chain(points: list[tuple[float, float, float]], label: str) -> bool:
    """打印并校验轴向路径：相邻平移点只能改 X 或只改 Y。"""
    print(f"  AXIS_CHAIN {label}:")
    for idx, (x, y, yaw) in enumerate(points):
        print(f"    p{idx}=({x:.4f}, {y:.4f}, yaw={yaw:.1f}°)")
    for idx, (a, b) in enumerate(zip(points, points[1:]), start=1):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        move_x = abs(dx) > AXIS_MOVE_EPS
        move_y = abs(dy) > AXIS_MOVE_EPS
        if move_x and move_y:
            print(
                f"  [ERROR] AXIS_CHAIN {label} 第{idx}段不是纯轴向: "
                f"dx={dx:+.4f}, dy={dy:+.4f}"
            )
            return False
        kind = "TURN" if not move_x and not move_y else ("X" if move_x else "Y")
        print(f"    seg{idx}: {kind} dx={dx:+.4f}, dy={dy:+.4f}")
    return True


def _axis_path(points: list[tuple[float, float, float]],
               label: str,
               fallback) -> bool:
    """优先path；若path失败，回退到原分段GOTO，保证测试不中断。"""
    if not _validate_axis_chain(points, label):
        return False
    speed = _axis_path_speed(points)
    if _key_path_points(points, label, speed=speed):
        return True
    print(f"  [WARN] KEY_PATH {label}失败，回退原分段GOTO")
    refresh_cur_from_pose(label=f"{label} fallback")
    return fallback()


def lock_y_to(y: float, timeout: float = 40.0) -> bool:
    """只沿 Y 方向后退/前进：下位机 TOY 会锁住当前 X，避免普通 GOTO 的横向修正。"""
    print(f"  LOCK_Y_TO {y:.4f} (keep X={_CUR_X:.4f})")
    ok = _timed(f"LOCK_Y_TO {y:.4f}", comm.lock_axis, "y", y, timeout=timeout)
    if ok:
        _update_cur(_CUR_X, y)
    return ok


def go_axis_x_then_y(x: float, y: float, label: str = "axis move") -> bool:
    """只走横移+直行：先改 X，再改 Y，避免斜走。"""
    print(f"  AXIS_MOVE {label}: target=({x:.4f}, {y:.4f})")
    sx, sy = _CUR_X, _CUR_Y
    yaw = _current_yaw()
    pts = [(sx, sy, yaw)]
    if abs(sx - x) > AXIS_MOVE_EPS:
        pts.append((x, sy, yaw))
    if abs(sy - y) > AXIS_MOVE_EPS:
        pts.append((x, y, yaw))

    def fallback() -> bool:
        if abs(_CUR_X - x) > AXIS_MOVE_EPS:
            if not go_to(x, _CUR_Y):
                return False
        if abs(_CUR_Y - y) > AXIS_MOVE_EPS:
            if not go_to(x, y):
                return False
        return True

    return _axis_path(pts, label, fallback)


def go_axis_y_then_x(x: float, y: float, label: str = "axis move") -> bool:
    """只走直行+横移：先改 Y，再改 X，用于必须先后退/前进的点。"""
    print(f"  AXIS_MOVE_YX {label}: target=({x:.4f}, {y:.4f})")
    sx, sy = _CUR_X, _CUR_Y
    yaw = _current_yaw()
    pts = [(sx, sy, yaw)]
    if abs(sy - y) > AXIS_MOVE_EPS:
        pts.append((sx, y, yaw))
    if abs(sx - x) > AXIS_MOVE_EPS:
        pts.append((x, y, yaw))

    def fallback() -> bool:
        if abs(_CUR_Y - y) > AXIS_MOVE_EPS:
            if not go_to(_CUR_X, y):
                return False
        if abs(_CUR_X - x) > AXIS_MOVE_EPS:
            if not go_to(x, y):
                return False
        return True

    return _axis_path(pts, label, fallback)


def go_axis_x_turn0_then_y(x: float, y: float, label: str = "axis move") -> bool:
    """只给 E 点使用：分段执行 X 方向 GOTO、转角、Y 方向 GOTO。"""
    print(f"  AXIS_MOVE_X_TURN0_Y {label}: target=({x:.4f}, {y:.4f})")

    # 不再强行把当前坐标和航向同步成 180°，避免理想坐标覆盖实车当前位置。
    # 先完成纯 X 方向移动，避免把 X/Y 合成为斜线。
    if abs(_CUR_X - x) > AXIS_MOVE_EPS:
        if not go_to(x, _CUR_Y):
            return False

    # 保留原先的大角度分段转向：先转到 90°，再转到 0°。
    if not turn_to(90.0):
        return False
    if not turn_to(0.0):
        return False

    # 转角完成后再执行纯 Y 方向移动。
    if abs(_CUR_Y - y) > AXIS_MOVE_EPS:
        if not go_to(x, y):
            return False
    return True


def turn_to(deg: float, timeout: float = 30.0) -> bool:
    print(f"  TURNTO {deg:.1f}°")
    return _timed(f"TURNTO {deg:.1f}", comm.turnto, deg, timeout=timeout)


def arm(state: int, timeout: float = 6.0) -> bool:
    print(f"  ARM {state}")
    return _timed(f"ARM {state}", comm.arm, state, timeout=timeout)


def rotate(pos: int, timeout: float = 12.0) -> bool:
    label = "state 5 (324°)" if pos == ROTATE_SPECIAL_STATE else f"slot {pos}"
    print(f"  ROTATE {label}")
    return _timed(f"ROTATE {label}", comm.rotate, pos, timeout=timeout)


def turn_180_after_arc() -> bool:
    """圆弧后先把转盘切到状态5，再转车体到180°。"""
    if not rotate(ROTATE_SPECIAL_STATE):
        print("  !! TURNTO 180 前状态5转盘失败")
        return False
    print("  TURNTO 180.0° after arc")
    seen = comm.response_seq()
    comm.send_turnto(180.0)
    turn_ok = comm.wait_for_after(comm.TYPE_CMD_TURNTO_RESP, seen, 30.0)
    if not turn_ok:
        print("  !! TURNTO 180 响应超时或失败")
    return turn_ok


def rotate_async(pos: int, label: str = "") -> bool:
    """只下发转盘槽位命令，不等待响应；用于和转向/圆弧并行动作。"""
    suffix = f" {label}" if label else ""
    print(f"  ROTATE slot {pos} (不等待，双发防丢帧{suffix})")
    comm.send_rotate(pos, repeats=2, repeat_delay_s=0.08)
    return True


def rotate_state5_async(label: str = "") -> bool:
    """备用：不等待响应切到状态5。当前C/D主流程在转180前同步切状态5。"""
    suffix = f" {label}" if label else ""
    print(f"  ROTATE state 5 (324°) (不等待，双发防丢帧{suffix})")
    comm.send_rotate(ROTATE_SPECIAL_STATE, repeats=2, repeat_delay_s=0.08)
    return True


def advance_turntable_after_color(slot_state: dict[str, int], label: str) -> int:
    """识别到一个物块后，把转盘切到下一槽位用于收纳/准备下一个物块。"""
    rotate_count = slot_state.get("load_rotate_count", 0)
    if rotate_count >= CD_LOAD_ROTATE_LIMIT:
        print(f"  ROTATE skip: 收纳阶段已完成 {CD_LOAD_ROTATE_LIMIT} 次转盘，不再转盘 ({label})")
        return slot_state["current"]
    next_slot = (slot_state["current"] + 1) % CD_SLOT_COUNT
    rotate_count += 1
    rotate_async(next_slot, label=f"收纳转盘第{rotate_count}/{CD_LOAD_ROTATE_LIMIT}次 {label}")
    slot_state["current"] = next_slot
    slot_state["load_rotate_count"] = rotate_count
    return next_slot


def recognize_current_slot_color(loaded_slot_colors: dict[int, str],
                                 slot_state: dict[str, int],
                                 label: str,
                                 stable: bool,
                                 once_timeout: Optional[float] = None) -> Optional[str]:
    """识别当前经过的物块颜色，并记录到当前转盘槽位。"""
    slot = slot_state["current"]
    block.set_status(f"正在识别{label} -> 槽位 {slot}")
    used_colors = set(loaded_slot_colors.values())

    if once_timeout is not None:
        color = _timed(f"BLOCK display-once {label} slot {slot}",
                       block.wait_for_display_color,
                       timeout_s=once_timeout,
                       window_s=CD_DISPLAY_ONCE_WINDOW_S,
                       exclude_colors=used_colors)
        if color is not None:
            loaded_slot_colors[slot] = color
            block.clear_recognition_cache()
            block.set_status(f"{label} -> 槽位 {slot}: {color}")
            print(f"  {label}: slot {slot} <- color {color} (viewer显示一次确认)")
            return color

    if stable:
        color = _timed(f"BLOCK recognize {label} slot {slot}", block.recognize_stable,
                       frames=10, timeout=3.0)
    else:
        debug = block.recent_motion_debug(window_s=0.45, exclude_colors=used_colors)
        print(
            f"  [{label}] recent block samples={debug.get('samples')}, "
            f"votes={debug.get('votes')}, best_pct={debug.get('best_pct')}"
        )
        color = _timed(f"BLOCK recognize {label} slot {slot}", block.recognize_motion,
                       window_s=0.45,
                       min_hits=1,
                       exclude_colors=used_colors)
    if color == "白":
        display_color = block.get_recent_display_color(
            window_s=max(CD_DISPLAY_ONCE_WINDOW_S, 0.60),
            exclude_colors=used_colors,
        )
        if display_color and display_color != "白":
            print(f"  {label}: 回退识别为白，但 viewer 即时颜色为 {display_color}，改用即时颜色")
            color = display_color
        elif display_color != "白":
            block.set_status(f"{label} 白色未被 viewer 即时确认，按未识别处理")
            block.clear_recognition_cache()
            print(f"  {label}: 回退识别为白但即时判定未确认，按未识别处理")
            return None
    if color is None:
        block.set_status(f"{label} 识别失败")
        block.clear_recognition_cache()
        print(f"  {label}: slot {slot} 颜色未识别，先按缺失处理")
        return None
    if not stable and color in loaded_slot_colors.values():
        # C/D 五个物块颜色应唯一；重复颜色更可能是背景/上一块误判，留给排除法处理。
        block.set_status(f"{label} 重复识别为 {color}，按未识别处理")
        block.clear_recognition_cache()
        print(f"  {label}: slot {slot} 重复颜色 {color}，按未识别处理")
        return None
    loaded_slot_colors[slot] = color
    block.clear_recognition_cache()
    block.set_status(f"{label} -> 槽位 {slot}: {color}")
    print(f"  {label}: slot {slot} <- color {color}")
    return color


def start_first_block_monitor(loaded_slot_colors: dict[int, str],
                              slot_state: dict[str, int]) -> tuple[threading.Thread, threading.Event, dict]:
    """ARM1 稳定后在当前位置监听第1个物块；使用和圆弧段一致的窗口证据判色。"""
    stop_event = threading.Event()
    state = {"color": None, "samples": 0, "active_samples": 0, "analysis": None}

    def record_color(color: str, suffix: str) -> None:
        slot = slot_state["current"]
        loaded_slot_colors[slot] = color
        state["color"] = color
        block.set_status(f"第1个物块 -> 槽位 {slot}: {color}")
        print(f"  第1个物块: slot {slot} <- color {color} ({suffix})")

    def force_first_block_color(analysis: dict) -> tuple[str, str]:
        """第1个物块必须给结果：彩色优先；没有明显彩色时黑/白二选一。"""
        color = analysis.get("color")
        if color in ("红", "绿", "蓝"):
            return color, "彩色优先判定"

        max_pct = analysis.get("max_pct") or {}
        avg_pct = analysis.get("avg_pct") or {}
        hits = analysis.get("hits") or {}
        max_black = float(max_pct.get("黑", 0.0))
        avg_black = float(avg_pct.get("黑", 0.0))
        max_white = float(max_pct.get("白", 0.0))
        avg_white = float(avg_pct.get("白", 0.0))
        hit_black = int(hits.get("黑", 0) or 0)

        # 静止第1块：如果白色面积明显占优，黑色多半来自黑线/阴影，必须判白。
        white_dominates = (
            max_white >= 0.70
            and avg_white >= avg_black + 0.25
            and max_white >= max_black + 0.25
        )
        if white_dominates:
            analysis["first_block_forced_bw"] = "白"
            return "白", (
                "无明显彩色，白色面积占优强制判白 "
                f"max黑/白={max_black:.3f}/{max_white:.3f}, "
                f"avg黑/白={avg_black:.3f}/{avg_white:.3f}"
            )

        # 只有黑色证据足够强，才判黑；否则剩余情况统一判白，保证第1个不为空。
        black_strong = (
            max_black >= 0.30
            or avg_black >= 0.16
            or (hit_black >= 3 and max_black >= 0.22)
        )
        forced = "黑" if black_strong else "白"
        analysis["first_block_forced_bw"] = forced
        return forced, (
            "无明显彩色，黑白强制二选一 "
            f"max黑/白={max_black:.3f}/{max_white:.3f}, "
            f"avg黑/白={avg_black:.3f}/{avg_white:.3f}, hits黑={hit_black}"
        )

    def worker() -> None:
        block.set_status("第1个物块监听中: ARM1 已到位，在当前位置窗口判定")
        window_start_wall = time.time()
        while not stop_event.is_set() and state["color"] is None:
            if block.sample_motion_frame():
                state["active_samples"] += 1
            time.sleep(0.02)

        if state["color"] is None:
            analysis = block.analyze_arc_interval_color(
                start_time=window_start_wall,
                end_time=time.time(),
                allowed_colors={"黑", "白", "红", "绿", "蓝"},
            )
            state["analysis"] = analysis
            state["samples"] = analysis.get("samples", 0)
            color, reason = force_first_block_color(analysis)
            record_color(
                color,
                f"当前位置窗口强制判定 samples={analysis.get('samples')}, {reason}, "
                f"black_score={analysis.get('black_score', 0.0):.3f}",
            )
        if state["color"] is None:
            state["debug"] = block.recent_motion_debug(window_s=CD_DISPLAY_ONCE_WINDOW_S)

    th = threading.Thread(target=worker, name="cd-first-block-monitor", daemon=True)
    block.clear_recognition_cache()
    th.start()
    return th, stop_event, state


def start_arc_color_monitor(loaded_slot_colors: dict[int, str],
                            slot_state: dict[str, int],
                            target_colors: dict[str, str],
                            start_delay_s: float = 0.0) -> tuple[threading.Thread, threading.Event, dict]:
    """圆弧期间按预计经过时间开窗口识别；转盘也按时间触发。"""
    stop_event = threading.Event()
    state = {"done": 0, "failed": [], "missing_colors": [], "last_pose": None,
             "schedule": [], "pending_bw": []}
    expected_order = list(target_colors.values())
    expected_colors = set(expected_order)

    def remaining_expected_colors() -> set[str]:
        """二维码给定颜色中，当前还没有装入转盘槽位的颜色。"""
        return expected_colors - set(loaded_slot_colors.values()) - CD_ARC_ACTIVE_EXCLUDE_COLORS

    def assign_window_color(slot: int, label: str, color: str, suffix: str) -> None:
        loaded_slot_colors[slot] = color
        block.set_status(f"{label} -> 槽位 {slot}: {color}")
        print(f"  {label}: slot {slot} <- color {color} ({suffix})")

    def resolve_pending_black_white() -> None:
        """黑白都在圆弧后段时，用窗口黑色证据强弱排序，避免白地面误判。"""
        pending = [p for p in state["pending_bw"] if p["slot"] not in loaded_slot_colors]
        if not pending:
            return
        missing = [c for c in expected_order if c not in set(loaded_slot_colors.values())]
        bw_missing = [c for c in ("黑", "白") if c in missing]
        if not bw_missing:
            return

        ranked = sorted(pending, key=lambda p: p["analysis"].get("black_score", 0.0), reverse=True)
        if set(bw_missing) == {"黑", "白"} and len(ranked) >= 2:
            black_item = ranked[0]
            white_item = ranked[-1]
            assign_window_color(black_item["slot"], black_item["label"], "黑",
                                f"黑白延后判定 black_score={black_item['analysis'].get('black_score', 0.0):.3f}")
            assign_window_color(white_item["slot"], white_item["label"], "白",
                                f"黑白延后判定 black_score={white_item['analysis'].get('black_score', 0.0):.3f}")
            return

        if len(bw_missing) == 1 and len(ranked) == 1:
            item = ranked[0]
            assign_window_color(item["slot"], item["label"], bw_missing[0], "单一剩余黑白颜色排除")

    def arc_point_time_s(x: float, y: float) -> tuple[float, float]:
        """把近似坐标投影到当前圆弧上，返回预计经过时间和扫过角度。"""
        yaw = math.radians(CD_ARC_START_YAW)
        right_x = math.cos(yaw)
        right_y = -math.sin(yaw)
        side = 1.0 if CD_ARC_DIR >= 0 else -1.0
        cx = CD_ARC_START[0] + side * right_x * CD_ARC_RADIUS
        cy = CD_ARC_START[1] + side * right_y * CD_ARC_RADIUS
        start_ang = math.atan2(CD_ARC_START[1] - cy, CD_ARC_START[0] - cx)
        point_ang = math.atan2(y - cy, x - cx)
        if CD_ARC_DIR >= 0:
            progress_rad = (start_ang - point_ang) % (2.0 * math.pi)
        else:
            progress_rad = (point_ang - start_ang) % (2.0 * math.pi)
        progress_deg = math.degrees(progress_rad)
        return progress_rad * CD_ARC_RADIUS / max(CD_ARC_SPEED, 0.01), progress_deg

    schedule = []
    total_t = math.radians(CD_ARC_SWEEP_DEG) * CD_ARC_RADIUS / max(CD_ARC_SPEED, 0.01)
    for idx, (label, tx, ty) in enumerate(CD_ARC_COLOR_POINTS):
        raw_center_t, progress_deg = arc_point_time_s(tx, ty)
        point_no = idx + 2
        extra_delay_s = float(CD_ARC_PER_POINT_EXTRA_DELAY_S.get(point_no, 0.0))
        planned_center_t = raw_center_t + CD_ARC_RECOGNIZE_TIME_OFFSET_S + extra_delay_s
        center_t = planned_center_t if CD_ARC_ALLOW_POST_ARC_WINDOWS else min(total_t, planned_center_t)
        window_start = max(0.0, center_t - CD_ARC_TIME_WINDOW_HALF_S)
        planned_window_end = center_t + CD_ARC_TIME_WINDOW_HALF_S
        window_end = planned_window_end if CD_ARC_ALLOW_POST_ARC_WINDOWS else min(total_t, planned_window_end)
        is_last_arc_point = idx == len(CD_ARC_COLOR_POINTS) - 1
        rotate_action = None if is_last_arc_point else "load_next"
        rotate_extra_delay_s = float(CD_ARC_PER_POINT_ROTATE_EXTRA_DELAY_S.get(point_no, 0.0))
        rotate_delay_s = CD_ARC_ROTATE_AFTER_CENTER_DELAY_S + rotate_extra_delay_s
        if rotate_action is None:
            rotate_t = None
        elif CD_ARC_ALLOW_POST_ARC_WINDOWS:
            rotate_t = center_t + rotate_delay_s
        else:
            rotate_t = min(center_t + rotate_delay_s,
                           max(window_start, total_t - CD_ARC_STATE5_BEFORE_END_S))
        schedule.append({
            "label": label,
            "raw_center_t": raw_center_t,
            "center_t": center_t,
            "extra_delay_s": extra_delay_s,
            "rotate_extra_delay_s": rotate_extra_delay_s,
            "progress_deg": progress_deg,
            "window_start": window_start,
            "window_end": window_end,
            "rotate_t": rotate_t,
            "rotate_action": rotate_action,
        })
    state["schedule"] = schedule

    print("  C/D 圆弧时间窗口:")
    if start_delay_s > 0.0:
        print(f"    固定过渡段延时: {start_delay_s:.2f}s 后进入圆弧时间轴")
    for item in schedule:
        rotate_text = (
            f", rotate_t={item['rotate_t']:.2f}s action={item['rotate_action']}"
            if item["rotate_t"] is not None else f", no rotate action={item['rotate_action']}"
        )
        print(
            f"    {item['label']}: raw={item['raw_center_t']:.2f}s center={item['center_t']:.2f}s "
            f"deg={item['progress_deg']:.1f}° extra={item['extra_delay_s']:.2f}s "
            f"rotate_extra={item['rotate_extra_delay_s']:.2f}s "
            f"window={item['window_start']:.2f}-{item['window_end']:.2f}s{rotate_text}"
        )

    def worker() -> None:
        next_idx = 0
        started_at = time.monotonic()
        while not stop_event.is_set() and next_idx < len(schedule):
            item = schedule[next_idx]
            label = item["label"]
            elapsed = time.monotonic() - started_at - start_delay_s
            if elapsed < item["window_start"]:
                time.sleep(CD_ARC_POSE_POLL_S)
                continue

            print(f"  [{label}] 时间窗口开始 t={elapsed:.2f}s slot={slot_state['current']}")
            if not block.has_fresh_frame(max_age_s=0.5):
                print(f"  [{label}] USB 当前无新鲜帧，窗口内尝试重开: {block.camera_health()}")
                block.ensure_ready_for_use(
                    f"{label}时间窗口",
                    timeout_s=0.25,
                    restart_timeout_s=0.55,
                )
            slot_at_window = slot_state["current"]
            color = None
            rotated = False
            block.clear_recognition_cache()
            window_start_wall = time.time()
            active_samples = 0
            while not stop_event.is_set():
                elapsed = time.monotonic() - started_at - start_delay_s

                if block.sample_motion_frame():
                    active_samples += 1

                if item["rotate_t"] is not None and not rotated and elapsed >= item["rotate_t"]:
                    if item.get("rotate_action") == "state5":
                        rotate_state5_async(f"{label}按时间{item['rotate_t']:.2f}s提前切状态5")
                    else:
                        advance_turntable_after_color(slot_state, f"{label}按时间{item['rotate_t']:.2f}s收纳")
                    rotated = True

                if elapsed > item["window_end"] and (rotated or item["rotate_t"] is None):
                    break
                time.sleep(CD_ARC_POSE_POLL_S)

            window_end_wall = time.time()
            used_colors = set(loaded_slot_colors.values()) | CD_ARC_ACTIVE_EXCLUDE_COLORS
            allowed_colors = remaining_expected_colors()
            analysis = block.analyze_arc_interval_color(
                start_time=window_start_wall,
                end_time=window_end_wall,
                exclude_colors=used_colors,
                allowed_colors=allowed_colors,
            )
            analysis["active_samples"] = active_samples
            color = analysis.get("color")
            bw_both_remaining = "黑" in allowed_colors and "白" in allowed_colors
            if color in ("黑", "白") and bw_both_remaining and not analysis.get("black_confirmed"):
                state["pending_bw"].append({
                    "slot": slot_at_window,
                    "label": label,
                    "analysis": analysis,
                })
                print(
                    f"  {label}: 黑白暂不立即判定，slot={slot_at_window}, "
                    f"samples={analysis.get('samples')}, active={analysis.get('active_samples')}, "
                    f"black_score={analysis.get('black_score', 0.0):.3f}, "
                    f"max_pct={analysis.get('max_pct')}, avg_pct={analysis.get('avg_pct')}"
                )
                color = None
            if color is not None:
                assign_window_color(slot_at_window, label, color, "时间窗口区间判定")

            if color is None:
                debug = block.recent_motion_debug(
                    window_s=CD_DISPLAY_ONCE_WINDOW_S,
                    exclude_colors=used_colors,
                )
                state["missing_colors"].append(label)
                print(
                    f"  {label}: 时间窗口未识别到有效颜色，后续用排除法; "
                    f"samples={debug.get('samples')}, active={active_samples}, votes={debug.get('votes')}, "
                    f"best_pct={debug.get('best_pct')}"
                )
            state["done"] += 1
            next_idx += 1
            block.clear_recognition_cache()

        if next_idx < len(schedule):
            state["failed"] = [item["label"] for item in schedule[next_idx:]]
        resolve_pending_black_white()

    th = threading.Thread(target=worker, name="cd-arc-color-monitor", daemon=True)
    th.start()
    return th, stop_event, state


def refresh_cur_from_pose(label: str = "pose") -> bool:
    """用下位机上报位姿刷新本地坐标，供后续横移+直行分段使用。"""
    for _ in range(20):
        pose = comm.get_pose(max_age=0.5)
        if pose is not None:
            x, y, yaw = pose
            _update_cur(x, y)
            print(f"  POSE {label}: ({x:.4f}, {y:.4f}, yaw={yaw:.1f}°)")
            return True
        time.sleep(0.05)
    print(f"  !! {label} 位姿刷新失败")
    return False


def run_cd_arc(loaded_slot_colors: dict[int, str], slot_state: dict[str, int],
               target_colors: dict[str, str]) -> bool:
    """按用户指定参数走 C/D 圆弧，并在近似坐标处识别颜色、按定时切转盘。"""
    timeout = (math.radians(CD_ARC_SWEEP_DEG) * CD_ARC_RADIUS /
               max(CD_ARC_SPEED, 0.01) * 2.5 + 15.0)
    monitor_thread, monitor_stop, monitor_state = start_arc_color_monitor(
        loaded_slot_colors,
        slot_state,
        target_colors,
    )
    ok = False
    try:
        ok = _timed(
            f"ARC r={CD_ARC_RADIUS:.3f} dir={CD_ARC_DIR} sweep={CD_ARC_SWEEP_DEG:.1f} v={CD_ARC_SPEED:.2f}",
            comm.arc,
            CD_ARC_RADIUS,
            CD_ARC_DIR,
            CD_ARC_SWEEP_DEG,
            speed=CD_ARC_SPEED,
            timeout=timeout,
        )
        if ok and monitor_thread.is_alive() and CD_ARC_POST_MONITOR_GRACE_S > 0.0:
            print(f"  圆弧命令完成，继续等待后段识别窗口 {CD_ARC_POST_MONITOR_GRACE_S:.2f}s")
            monitor_thread.join(timeout=CD_ARC_POST_MONITOR_GRACE_S)
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=1.0)
    if monitor_state["done"] != len(CD_ARC_COLOR_POINTS):
        print(
            f"  [WARN] 圆弧颜色识别未完成: done={monitor_state['done']}, "
            f"missing={monitor_state['failed']}；继续后续任务，颜色表后面兜底补齐"
        )
    if monitor_state.get("missing_colors"):
        print(f"  圆弧中有颜色未识别，后续尝试排除法补齐: {monitor_state['missing_colors']}")
    if ok:
        return refresh_cur_from_pose("after C/D arc")
    print("  [WARN] ARC 响应失败/超时，但不因颜色兜底策略停止；继续刷新位姿后走后续任务")
    return refresh_cur_from_pose("after C/D arc after warn")


def run_cd_fixed_arc(loaded_slot_colors: dict[int, str], slot_state: dict[str, int],
                     target_colors: dict[str, str]) -> bool:
    """下位机写死的 C/D 固定连续段：过渡段+圆弧期间继续识别颜色。"""
    print(
        "  CD_FIXED_ARC 下位机固定连续段 "
        f"(pre {CD_ARC_START} yaw {CD_ARC_START_YAW:.1f}°, "
        f"arc r={CD_ARC_RADIUS:.3f} sweep={CD_ARC_SWEEP_DEG:.1f})"
    )

    monitor_thread, monitor_stop, monitor_state = start_arc_color_monitor(
        loaded_slot_colors,
        slot_state,
        target_colors,
        start_delay_s=CD_FIXED_PRE_PHASE_S,
    )

    def first_rotate_after_pre() -> None:
        time.sleep(CD_FIXED_PRE_PHASE_S)
        if not monitor_stop.is_set():
            advance_turntable_after_color(slot_state, "固定过渡段结束后第1个物块收纳")

    first_rotate_thread = threading.Thread(
        target=first_rotate_after_pre,
        name="cd-fixed-first-rotate",
        daemon=True,
    )
    first_rotate_thread.start()

    try:
        ok = _timed("CD_FIXED_ARC", comm.cd_fixed_arc, timeout=CD_FIXED_ARC_TIMEOUT_S)
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=1.0)
        first_rotate_thread.join(timeout=0.2)

    if monitor_state["done"] != len(CD_ARC_COLOR_POINTS):
        print(
            f"  [WARN] 固定圆弧颜色识别未完成: done={monitor_state['done']}, "
            f"missing={monitor_state['failed']}；继续后续任务，颜色表后面兜底补齐"
        )
    if monitor_state.get("missing_colors"):
        print(f"  固定圆弧中有颜色未识别，后续尝试排除法补齐: {monitor_state['missing_colors']}")
    if ok:
        return refresh_cur_from_pose("after CD_FIXED_ARC")
    print("  [WARN] CD_FIXED_ARC 响应失败/超时，但不停止；继续刷新位姿后走后续任务")
    return refresh_cur_from_pose("after CD_FIXED_ARC after warn")


def prepare_block_scan(arm_state: int, light_id: int) -> bool:
    """机械臂、补光灯和 USB 摄像头尽量同时准备。"""
    def run() -> bool:
        t_stage = time.monotonic()
        seen = comm.response_seq()
        comm.send_arm(arm_state)
        comm.send_light(light_id, True)
        t_stage = _timing_mark("ARM/LIGHT send", t_stage)
        block.CONFIG["show_window"] = os.environ.get("ZQWL_HEADLESS") not in {"1", "true", "yes", "on"}
        block.set_status("准备物块识别: ARM 1 + LIGHT 4 + USB")
        block.start_viewer()
        block.clear_recognition_cache()
        t_stage = _timing_mark("block viewer start/cache clear", t_stage)
        usb_ok = block.ensure_ready_for_use(
            "C/D block准备",
            timeout_s=CD_BLOCK_READY_TIMEOUT_S,
            restart_timeout_s=CD_BLOCK_RESTART_TIMEOUT_S,
        )
        t_stage = _timing_mark(f"block USB ready -> {usb_ok}", t_stage)
        arm_ok = comm.wait_for_after(comm.TYPE_ARM_RESP, seen, 6.0)
        t_stage = _timing_mark(f"ARM_RESP wait -> {arm_ok}", t_stage)
        light_ok = comm.wait_for_after(comm.TYPE_LIGHT_RESP, seen, 5.0)
        _timing_mark(f"LIGHT_RESP wait -> {light_ok}", t_stage)
        if not usb_ok:
            print(f"  !! USB 摄像头无有效画面: {block.camera_health()}")
        if not arm_ok:
            print("  !! 机械臂响应超时或失败")
        if not light_ok:
            print("  !! 补光灯响应超时或失败")
        if usb_ok and arm_ok and light_ok:
            # ARM 1 到位前 viewer 可能已经看到机械臂运动过程，必须清掉这段缓存。
            block.clear_recognition_cache()
            block.set_status("ARM 1 已到位，等待第1个物块稳定识别")
        return usb_ok and arm_ok and light_ok

    print(f"  ARM {arm_state} + LIGHT {light_id} ON + USB CAMERA START")
    return _timed(f"ARM {arm_state} + LIGHT {light_id} ON + USB CAMERA START", run)


def recognize_qr1_targets() -> dict[str, str]:
    seq = _timed("QR1 recognize", qr1.recognize, timeout=QR1_RECOGNIZE_TIMEOUT_S)
    color_seq = qr1.TASK1_PLANS.get(seq)
    if color_seq is None:
        raise RuntimeError(f"QR1 result {seq!r} not in TASK1_PLANS")
    target_colors = dict(zip(QR_TARGET_ORDER, color_seq))
    print(f"  QR1: {seq} -> {target_colors}")
    return target_colors


def recognize_block_for_slot(slot: int) -> str:
    block.set_status(f"正在识别转盘槽位 {slot}")
    color = _timed(f"BLOCK recognize block for slot {slot}", block.recognize_stable,
                   frames=10, timeout=3.0)
    if color is None:
        block.set_status(f"槽位 {slot} 识别失败")
        raise RuntimeError(f"block color for slot {slot} recognize failed")
    block.set_status(f"槽位 {slot} 识别结果: {color}")
    print(f"  loaded slot {slot} <- block color {color}")
    return color


def scan_turntable_slots() -> dict[int, str]:
    """按槽位 0→4 顺序扫描转盘颜色表。"""
    print("\n=== 转盘槽位颜色顺序扫描 ===")
    loaded_slot_colors: dict[int, str] = {}
    _require(rotate(0), "rotate slot 0 before slot scan")
    for slot in range(5):
        loaded_slot_colors[slot] = recognize_block_for_slot(slot)
        if slot < 4:
            _require(rotate(slot + 1), f"rotate slot {slot + 1} during slot scan")
    return loaded_slot_colors


def invert_loaded_slot_colors(loaded_slot_colors: dict[int, str]) -> dict[str, int]:
    if len(loaded_slot_colors) != 5:
        print(f"  [WARN] loaded slot color count invalid but continue: {loaded_slot_colors}")
    color_to_slot: dict[str, int] = {}
    for slot, color in loaded_slot_colors.items():
        if color in color_to_slot:
            print(
                f"  [WARN] duplicate color {color}: keep slot {color_to_slot[color]}, "
                f"ignore slot {slot} for target mapping"
            )
            continue
        color_to_slot[color] = slot
    print(f"  color -> slot: {color_to_slot}")
    return color_to_slot


def complete_loaded_slot_colors_by_exclusion(loaded_slot_colors: dict[int, str],
                                             target_colors: dict[str, str]) -> dict[int, str]:
    """用二维码给出的五种目标颜色补齐槽位；多缺/重复也不停止，直接兜底蒙上。"""
    expected_order = list(target_colors.values())
    expected_colors = set(expected_order)
    if len(expected_colors) != CD_SLOT_COUNT:
        print(f"  [WARN] 二维码目标颜色不是5个唯一颜色，仍继续兜底: {target_colors}")
        expected_order = []
        for color in list(target_colors.values()) + ["黑", "白", "红", "绿", "蓝"]:
            if color not in expected_order:
                expected_order.append(color)
            if len(expected_order) >= CD_SLOT_COUNT:
                break
        expected_colors = set(expected_order)

    invalid_slots = [slot for slot, color in loaded_slot_colors.items()
                     if slot < 0 or slot >= CD_SLOT_COUNT or color not in expected_colors]
    for slot in invalid_slots:
        print(f"  slot {slot} 颜色 {loaded_slot_colors[slot]} 不在有效槽位/二维码目标颜色内，按未识别处理")
        loaded_slot_colors.pop(slot, None)

    seen_colors: set[str] = set()
    duplicate_slots: list[int] = []
    for slot in sorted(loaded_slot_colors):
        color = loaded_slot_colors[slot]
        if color in seen_colors:
            duplicate_slots.append(slot)
        else:
            seen_colors.add(color)
    for slot in duplicate_slots:
        print(f"  slot {slot} 颜色 {loaded_slot_colors[slot]} 重复，按未识别处理并兜底补齐")
        loaded_slot_colors.pop(slot, None)

    missing_slots = sorted(set(range(CD_SLOT_COUNT)) - set(loaded_slot_colors.keys()))
    used_colors = set(loaded_slot_colors.values())
    missing_colors = [color for color in expected_order if color not in used_colors]

    if not missing_slots and not missing_colors:
        return loaded_slot_colors
    if len(missing_colors) < len(missing_slots):
        fallback_colors = expected_order or ["黑", "白", "红", "绿", "蓝"]
        while len(missing_colors) < len(missing_slots):
            missing_colors.append(fallback_colors[len(missing_colors) % len(fallback_colors)])
    if len(missing_colors) > len(missing_slots):
        print(
            f"  [WARN] 剩余颜色多于空槽，无法完全唯一映射，保留现有槽位并继续: "
            f"missing_slots={missing_slots}, missing_colors={missing_colors}"
        )
    for slot, color in zip(missing_slots, missing_colors):
        loaded_slot_colors[slot] = color
        print(f"  兜底补齐: slot {slot} <- {color}")
    if len(missing_slots) > 1:
        print(
            f"  [WARN] 多个槽位未识别，已按二维码剩余颜色顺序兜底，不停止: "
            f"loaded={loaded_slot_colors}"
        )
    return loaded_slot_colors


def target_slot_for_target(name: str, target_colors: dict[str, str], color_to_slot: dict[str, int]) -> tuple[str, int]:
    color = target_colors[name]
    if color not in color_to_slot:
        if color_to_slot:
            fallback_color, fallback_slot = sorted(color_to_slot.items(), key=lambda item: item[1])[0]
            print(
                f"  [WARN] PLACE {name} 需要 {color}，但颜色表缺失；"
                f"兜底使用 slot {fallback_slot}({fallback_color})，不中断"
            )
            return color, fallback_slot
        print(f"  [WARN] PLACE {name} 颜色表为空；兜底使用 slot 0，不中断")
        return color, 0
    slot = color_to_slot[color]
    return color, slot


def wait_parallel_rotate(slot: int, seen_seq: int, label: str, timeout: float = 12.0) -> bool:
    """等待并行转盘响应；失败时重发同一绝对槽位一次，避免物块前推到错误槽位。"""
    if comm.wait_for_after(comm.TYPE_ROTATE_RESP, seen_seq, timeout):
        return True
    print(f"  !! {label} 转盘响应超时或失败，前推前重发 slot {slot}")
    return comm.rotate(slot, timeout=timeout)


def ring_place_and_sync(name: str,
                        push_sync_x: float,
                        push_sync_y: float,
                        back_sync_x: float,
                        back_sync_y: float,
                        back_mm: float,
                        pre_push_wait=None) -> bool:
    """ring 前推后先用车体相对后退脱离接触，再同步到后退后的真实坐标。"""
    print(f"\n=== {name} 点 ring 定位放置 ===")
    print(
        f"  ring 前推理论点=({push_sync_x:.3f}, {push_sync_y:.3f})；"
        f"BODY_POS 后退 {back_mm:.0f}mm 后同步到 ({back_sync_x:.3f}, {back_sync_y:.3f})"
    )
    if not ring.align_checked_then_forward(verbose=True, back_mm=0.0, pre_push_wait=pre_push_wait):
        return False
    if back_mm <= 0.0:
        return sync_pose(push_sync_x, push_sync_y, None)
    back_ok = comm.body_pos_move(0.0, -back_mm, timeout=CD_RING_BODY_BACK_TIMEOUT_S)
    if not back_ok:
        print(f"  [WARN] {name} BODY_POS 后退未确认完成；不重发，继续同步到后退坐标")
    print(f"  BODY_POS 后退完成，等待 {BODY_POS_POST_SETTLE_S:.2f}s 后同步坐标")
    time.sleep(BODY_POS_POST_SETTLE_S)
    return sync_pose(back_sync_x, back_sync_y, None)


def release_block_camera_before_ring() -> None:
    """物块颜色已经完成后，释放 block 摄像头，避免 ring 打开 USB 摄像头失败。"""
    print("\n=== 物块颜色识别完成，释放 block 摄像头，切换到 ring 定位 ===")
    block.close()
    ring.close()
    time.sleep(0.6)
    ring.prepare_camera_async(verbose=True)


def place_target_with_ring(name: str,
                           target_colors: dict[str, str],
                           color_to_slot: dict[str, int],
                           move_order: str = "xy") -> bool:
    """到达放置点后先切转盘，再调用 ring 放置，最后同步该点后的坐标。"""
    x, y = TARGET_POINTS[name]
    sx, sy = PLACE_SYNC_POINTS[name]
    back_x, back_y = PLACE_BACK_TARGET_POINTS[name]
    back_mm = RING_BACK_MM_BY_POINT[name]
    if move_order == "x_turn0_y":
        if not go_axis_x_turn0_then_y(x, y, label=f"move to {name}"):
            return False
    elif move_order == "yx":
        if not go_axis_y_then_x(x, y, label=f"move to {name}"):
            return False
    else:
        if not go_axis_x_then_y(x, y, label=f"move to {name}"):
            return False
    color, slot = target_slot_for_target(name, target_colors, color_to_slot)
    print(f"  PLACE {name}: need {color}, rotate slot {slot} (与 ring 微调并行，前推前确认)")
    rotate_seen = comm.send_rotate(slot, repeats=2, repeat_delay_s=0.08)
    return ring_place_and_sync(
        name, sx, sy, back_x, back_y, back_mm,
        pre_push_wait=lambda: wait_parallel_rotate(slot, rotate_seen, f"PLACE {name}")
    )


def finish_cd_after_e() -> bool:
    """E 点完成后只做 X 左移 240mm，再转到结束角度，结束 C/D 段。"""
    exit_x = PLACE_BACK_TARGET_POINTS["E"][0] + E_EXIT_X_SHIFT_M
    sx, sy = _CUR_X, _CUR_Y
    yaw = _current_yaw(default=0.0)
    pts = [(sx, sy, yaw)]
    if abs(exit_x - sx) > AXIS_MOVE_EPS:
        pts.append((exit_x, sy, yaw))
    pts.append((exit_x, sy, E_EXIT_YAW))

    def fallback() -> bool:
        if not go_to(exit_x, _CUR_Y):
            return False
        return turn_to(E_EXIT_YAW)

    return _axis_path(pts, "finish C/D after E", fallback)


def _same_xy(a, b) -> bool:
    return abs(a[0] - b[0]) <= KEY_PATH_XY_EPS and abs(a[1] - b[1]) <= KEY_PATH_XY_EPS


def _segment_timeout(a, b) -> float:
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    return max(KEY_PATH_SEG_TIMEOUT_MIN, dist / max(KEY_PATH_SPEED, 0.01) * 6.0 + 6.0)


def _send_key_path_segment(points) -> int:
    comm.send_path_begin(KEY_PATH_SPEED, len(points))
    for x, y, yaw in points:
        comm.send_path_point(x, y, yaw, comm.PATH_MODE_KEY)
    return comm.send_path_exec()


def _run_key_path_segment(a, b, idx: int) -> bool:
    timeout = _segment_timeout(a, b)
    ok = _timed(
        f"KEY_PATH move p{idx - 1}->p{idx} v={KEY_PATH_SPEED:.2f}m/s",
        comm.key_path,
        [a, b],
        speed=KEY_PATH_SPEED,
        timeout=timeout,
    )
    if ok:
        _update_cur(b[0], b[1])
    return ok


def run_cd_arc_entry_key_path() -> bool:
    """用稳定的一段一段 key_path 进入 C/D 圆弧起点。"""
    start = (_CUR_X, _CUR_Y, -90.0)
    move_point = (CD_ARC_START[0], CD_ARC_START[1], -90.0)
    turn_point = (CD_ARC_START[0], CD_ARC_START[1], CD_ARC_START_YAW)
    timeout = max(
        KEY_PATH_SEG_TIMEOUT_MIN,
        math.hypot(move_point[0] - start[0], move_point[1] - start[1])
        / max(KEY_PATH_SPEED, 0.01) * 6.0 + 10.0,
    ) + KEY_PATH_TURN_TIMEOUT
    ok = _timed(
        f"KEY_PATH C/D arc entry {CD_ARC_START} yaw={CD_ARC_START_YAW:.1f}",
        comm.key_path,
        [start, move_point, turn_point],
        speed=KEY_PATH_SPEED,
        timeout=timeout,
    )
    if ok:
        _update_cur(CD_ARC_START[0], CD_ARC_START[1])
    return ok


def _run_key_turn_with_rotate(a, b, idx: int, slot: int) -> bool:
    def run() -> bool:
        path_seen = _send_key_path_segment([a, b])
        rotate_seen = comm.send_rotate(slot)

        path_ok = comm.wait_for_after(comm.TYPE_CMD_PATH_RESP, path_seen, KEY_PATH_TURN_TIMEOUT)
        rotate_ok = comm.wait_for_after(comm.TYPE_ROTATE_RESP, rotate_seen, KEY_PATH_ROTATE_TIMEOUT)
        if not path_ok:
            print("  !! 原地转角 PATH_RESP 超时或失败")
        if not rotate_ok:
            print("  !! 转盘 ROTATE_RESP 超时或失败")
            rotate_ok = comm.rotate(slot, timeout=KEY_PATH_ROTATE_TIMEOUT)
        return path_ok and rotate_ok

    return _timed(
        f"KEY_PATH turn p{idx - 1}->p{idx}, rotate slot {slot}",
        run,
    )


def run_key_path_with_rotate() -> Optional[dict[int, str]]:
    """执行固定关键点轨迹，并记录进入转盘 0-4 槽位的物块颜色。

    约定进入该函数前转盘已经在槽位 0。路径中每个重复坐标点会先识别
    即将进入当前槽位的物块颜色，再一边原地转角一边切换到下一个槽位。
    """
    print("\n=== 固定关键点轨迹 + 物块入槽颜色识别 + 转角同步转盘 ===")
    loaded_slot_colors: dict[int, str] = {}
    current_slot = 0
    rotate_idx = 0
    for idx in range(1, len(KEY_PATH_POINTS)):
        a = KEY_PATH_POINTS[idx - 1]
        b = KEY_PATH_POINTS[idx]
        if _same_xy(a, b):
            if rotate_idx >= len(KEY_PATH_ROTATE_SLOTS):
                print("  !! 转盘槽位数量少于原地转角数量")
                return None
            loaded_slot_colors[current_slot] = recognize_block_for_slot(current_slot)
            slot = KEY_PATH_ROTATE_SLOTS[rotate_idx]
            rotate_idx += 1
            if not _run_key_turn_with_rotate(a, b, idx, slot):
                return None
            current_slot = slot
        else:
            if not _run_key_path_segment(a, b, idx):
                return None

    if rotate_idx != len(KEY_PATH_ROTATE_SLOTS):
        print("  !! 转盘槽位数量多于原地转角数量")
        return None
    _update_cur(KEY_PATH_POINTS[-1][0], KEY_PATH_POINTS[-1][1])
    loaded_slot_colors[current_slot] = recognize_block_for_slot(current_slot)
    return loaded_slot_colors


def run_task_cd() -> None:
    print("\n=== test_C&D 新流程 ===")

    _require(comm.use_encoder_yaw(timeout=2.0), "set yaw source to encoder for C/D")
    _require(sync_pose(INIT_X, INIT_Y, INIT_YAW), "SYNC initial pose")
    _require(go_to(0.0, 0.25), "go to (0.000, 0.250)")
    _require(go_to(-0.662, 0.25), "go to (-0.662, 0.250)")

    target_colors = recognize_qr1_targets()

    _require(turn_to(-90.0), "turn to -90 before C/D arc")
    _require(prepare_block_scan(1, 4), "arm 1, light 4 and USB camera ready")
    t_cd_block = time.monotonic()
    time.sleep(CD_ARM_READY_SETTLE_S)
    t_cd_block = _timing_mark(f"ARM ready fixed settle {CD_ARM_READY_SETTLE_S:.2f}s", t_cd_block)
    _require(block.ensure_ready_for_use(
        "第1个物块识别前",
        timeout_s=0.4,
        restart_timeout_s=CD_BLOCK_RESTART_TIMEOUT_S,
    ), "USB camera fresh frame before first block")
    t_cd_block = _timing_mark("first block USB confirm", t_cd_block)
    block.clear_recognition_cache()
    loaded_slot_colors: dict[int, str] = {}
    slot_state = {"current": 0, "load_rotate_count": 0}

    first_thread, first_stop, first_state = start_first_block_monitor(
        loaded_slot_colors,
        slot_state,
    )
    first_monitor_start = time.monotonic()
    first_monitor_closed = False

    def finish_first_block_monitor(label: str) -> None:
        nonlocal first_monitor_closed, t_cd_block
        if first_monitor_closed:
            return
        # 第1个物块的识别窗口：从开始识别，到即将下发去(-0.9,0.25)圆弧起点命令前。
        # 同时保留最短稳定观察时间，避免机械臂刚到位的瞬间帧误判。
        remain_s = CD_FIRST_BLOCK_STABLE_OBSERVE_S - (time.monotonic() - first_monitor_start)
        if remain_s > 0.0:
            time.sleep(remain_s)
        first_stop.set()
        first_thread.join(timeout=0.8)
        first_monitor_closed = True
        t_cd_block = _timing_mark(
            f"first block observe until {label}, color={first_state.get('color')}",
            t_cd_block,
        )

    def report_first_block_missing() -> None:
        if first_state["color"] is not None:
            return
        debug = first_state.get("debug") or block.recent_motion_debug(window_s=CD_DISPLAY_ONCE_WINDOW_S)
        print(
            "  第1个物块颜色未识别，继续收纳，最终只缺一个时用排除法补齐; "
            f"active={first_state.get('active_samples')}, samples={debug.get('samples')}, "
            f"votes={debug.get('votes')}, best_pct={debug.get('best_pct')}"
        )
    if CD_USE_FIXED_ARC:
        _require(block.ensure_ready_for_use(
            "C/D固定连续段识别前",
            timeout_s=0.4,
            restart_timeout_s=CD_BLOCK_RESTART_TIMEOUT_S,
        ), "USB camera fresh frame before C/D fixed arc")
        finish_first_block_monitor("C/D fixed arc command")
        report_first_block_missing()
        _require(run_cd_fixed_arc(loaded_slot_colors, slot_state, target_colors), "C/D fixed arc")
    else:
        finish_first_block_monitor("key_path to (-0.9,0.25)")
        report_first_block_missing()
        _require(run_cd_arc_entry_key_path(), "key_path to C/D arc entry")
        advance_turntable_after_color(slot_state, "第1个物块到圆弧起点后收纳")
        _require(block.ensure_ready_for_use(
            "C/D圆弧识别前",
            timeout_s=0.4,
            restart_timeout_s=CD_BLOCK_RESTART_TIMEOUT_S,
        ), "USB camera fresh frame before C/D arc")
        _require(run_cd_arc(loaded_slot_colors, slot_state, target_colors), "C/D arc")
    complete_loaded_slot_colors_by_exclusion(loaded_slot_colors, target_colors)
    color_to_slot = invert_loaded_slot_colors(loaded_slot_colors)
    release_block_camera_before_ring()
    _require(turn_180_after_arc(), "turn to 180 after arc")
    _require(comm.use_encoder_yaw(timeout=2.0), "keep encoder yaw after turn 180")
    print("  C/D 180°后继续使用编码器 yaw")

    _require(place_target_with_ring("A", target_colors, color_to_slot, move_order="xy"),
             "place A with ring")
    _require(place_target_with_ring("B", target_colors, color_to_slot, move_order="xy"),
             "place B with ring")
    _require(place_target_with_ring("D", target_colors, color_to_slot, move_order="xy"),
             "place D with ring")
    _require(place_target_with_ring("C", target_colors, color_to_slot, move_order="yx"),
             "place C with ring")
    _require(place_target_with_ring("E", target_colors, color_to_slot, move_order="x_turn0_y"),
             "place E with ring")
    _require(finish_cd_after_e(), "finish C/D after E")
    if not comm.use_encoder_yaw(timeout=2.0):
        print("  [WARN] C/D 结束后保持编码器 yaw 失败，A/B 入口会再次强制切换")

    print("\n=== test_C&D 流程完成 ===")


def main() -> None:
    port = serial_port_config()
    print(f"  SERIAL {port or 'AUTO'} @ {BAUDRATE}")
    comm.init(port, BAUDRATE)
    ring.configure({"comm_port": port, "comm_baud": BAUDRATE})
    time.sleep(1.0)
    try:
        wait_run_start_if_requested()
        run_task_cd()
    except RuntimeError as e:
        print(f"\n[兜底] 任务中断: {e}")
    except KeyboardInterrupt:
        print("\n[用户中断]")
    finally:
        try:
            comm.light(4, False, timeout=2.0)
        except Exception as e:
            print(f"[清理] light 4 off failed: {e}")
        try:
            block.close()
        except Exception as e:
            print(f"[清理] block camera close failed: {e}")
        try:
            ring.close()
        except Exception as e:
            print(f"[清理] ring camera close failed: {e}")
        comm.shutdown()


if __name__ == "__main__":
    main()
