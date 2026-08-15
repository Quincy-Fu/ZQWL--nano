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
import sys
import threading
import time
from typing import Optional

import comm
import qr1
import block
import ring


BAUDRATE = 115200
RUN_START_ARGS = {"run", "--run", "wait-run", "--wait-run"}
RUN_GATE_TIMEOUT = 5.0

INIT_X = 0.0
INIT_Y = 0.0
INIT_YAW = 90.0

KEY_PATH_SPEED = 0.60
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
CD_ARC_SPEED = 0.30
CD_SLOT_COUNT = 5
CD_LOAD_ROTATE_LIMIT = 4
CD_ARC_POSE_POLL_S = 0.02
CD_ARM_READY_SETTLE_S = 0.80
CD_DISPLAY_ONCE_WINDOW_S = 0.80
CD_ARC_RECOGNIZE_TIME_OFFSET_S = 1.00
CD_ARC_TIME_WINDOW_HALF_S = 0.90
# 转盘切槽时机按圆弧行驶距离算，而不是写死时间。
# 旧参数 0.18m/s * 0.50s ≈ 0.09m；提速后保持同样空间位置触发。
CD_ARC_ROTATE_AFTER_CENTER_DIST_M = 0.09
# 圆弧运动中白色很容易被白底误触发，默认排除；只在白色成为最后一个缺失颜色时放开。
CD_ARC_ACTIVE_EXCLUDE_COLORS = {"白"}
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
    "A": (-0.630, 1.600),
    "B": (-0.430, 1.340),
    "C": ( 0.210, 1.620),
    "D": ( 0.330, 1.340),
    "E": ( 0.820, 1.350),
}
PLACE_SYNC_POINTS = {
    "A": (-0.630, 1.720),
    "B": (-0.420, 1.440),
    "C": ( 0.220, 1.700),
    "D": ( 0.330, 1.425),
    "E": ( 0.820, 1.460),
}
QR_TARGET_ORDER = ["A", "B", "C", "D", "E"]
ROTATE_SPECIAL_STATE = 5

POST_PATH_AFTER_ARM0 = [
    (0.600, 0.200),
    (0.550, 0.200),
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


def go_axis_x_then_y(x: float, y: float, label: str = "axis move") -> bool:
    """只走横移+直行：先改 X，再改 Y，避免斜走。"""
    print(f"  AXIS_MOVE {label}: target=({x:.4f}, {y:.4f})")
    if abs(_CUR_X - x) > AXIS_MOVE_EPS:
        if not go_to(x, _CUR_Y):
            return False
    if abs(_CUR_Y - y) > AXIS_MOVE_EPS:
        if not go_to(x, y):
            return False
    return True


def go_axis_y_then_x(x: float, y: float, label: str = "axis move") -> bool:
    """只走直行+横移：先改 Y，再改 X，用于必须先后退/前进的点。"""
    print(f"  AXIS_MOVE_YX {label}: target=({x:.4f}, {y:.4f})")
    if abs(_CUR_Y - y) > AXIS_MOVE_EPS:
        if not go_to(_CUR_X, y):
            return False
    if abs(_CUR_X - x) > AXIS_MOVE_EPS:
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
    label = "state 5 (36°)" if pos == ROTATE_SPECIAL_STATE else f"slot {pos}"
    print(f"  ROTATE {label}")
    return _timed(f"ROTATE {label}", comm.rotate, pos, timeout=timeout)


def rotate_async(pos: int, label: str = "") -> bool:
    """只下发转盘槽位命令，不等待响应；用于和转向/圆弧并行动作。"""
    suffix = f" {label}" if label else ""
    print(f"  ROTATE slot {pos} (不等待{suffix})")
    comm.send_rotate(pos)
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
    """第1个物块从 ARM1+USB 就绪后开始看，直到车走到圆弧起点。"""
    stop_event = threading.Event()
    state = {"color": None, "white_seen": False, "samples": 0}

    def record_color(color: str, suffix: str) -> None:
        slot = slot_state["current"]
        loaded_slot_colors[slot] = color
        state["color"] = color
        block.set_status(f"第1个物块 -> 槽位 {slot}: {color}")
        print(f"  第1个物块: slot {slot} <- color {color} ({suffix})")

    def worker() -> None:
        block.set_status("第1个物块监听中: ARM1 已到位，行进到圆弧起点")
        while not stop_event.is_set() and state["color"] is None:
            color = block.wait_for_display_color(
                timeout_s=0.05,
                window_s=CD_DISPLAY_ONCE_WINDOW_S,
                exclude_colors=set(),
            )
            if color is None:
                time.sleep(0.01)
                continue
            state["samples"] += 1
            if color == "白":
                # 白底会持续存在，先记候选，窗口结束时若没有非白色再确认白色。
                state["white_seen"] = True
                continue
            record_color(color, "行进中显示确认")

        if state["color"] is None and state["white_seen"]:
            record_color("白", "窗口结束白色确认")

    th = threading.Thread(target=worker, name="cd-first-block-monitor", daemon=True)
    block.clear_recognition_cache()
    th.start()
    return th, stop_event, state


def start_arc_color_monitor(loaded_slot_colors: dict[int, str],
                            slot_state: dict[str, int],
                            target_colors: dict[str, str]) -> tuple[threading.Thread, threading.Event, dict]:
    """圆弧期间按预计经过时间开窗口识别；转盘也按时间触发。"""
    stop_event = threading.Event()
    state = {"done": 0, "failed": [], "missing_colors": [], "last_pose": None, "schedule": []}

    def active_exclude_colors() -> set[str]:
        expected = list(target_colors.values())
        used = set(loaded_slot_colors.values())
        missing = [color for color in expected if color not in used]
        exclude = set(CD_ARC_ACTIVE_EXCLUDE_COLORS)
        if missing == ["白"]:
            exclude.discard("白")
        return exclude

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
        center_t = min(total_t, raw_center_t + CD_ARC_RECOGNIZE_TIME_OFFSET_S)
        window_start = max(0.0, center_t - CD_ARC_TIME_WINDOW_HALF_S)
        window_end = min(total_t, center_t + CD_ARC_TIME_WINDOW_HALF_S)
        rotate_delay_s = CD_ARC_ROTATE_AFTER_CENTER_DIST_M / max(CD_ARC_SPEED, 0.01)
        rotate_t = center_t + rotate_delay_s if idx < len(CD_ARC_COLOR_POINTS) - 1 else None
        schedule.append({
            "label": label,
            "raw_center_t": raw_center_t,
            "center_t": center_t,
            "progress_deg": progress_deg,
            "window_start": window_start,
            "window_end": window_end,
            "rotate_t": rotate_t,
        })
    state["schedule"] = schedule

    print("  C/D 圆弧时间窗口:")
    for item in schedule:
        rotate_text = f", rotate_t={item['rotate_t']:.2f}s" if item["rotate_t"] is not None else ", no rotate"
        print(
            f"    {item['label']}: raw={item['raw_center_t']:.2f}s center={item['center_t']:.2f}s "
            f"deg={item['progress_deg']:.1f}° window={item['window_start']:.2f}-{item['window_end']:.2f}s{rotate_text}"
        )

    def worker() -> None:
        next_idx = 0
        started_at = time.monotonic()
        while not stop_event.is_set() and next_idx < len(schedule):
            item = schedule[next_idx]
            label = item["label"]
            elapsed = time.monotonic() - started_at
            if elapsed < item["window_start"]:
                time.sleep(CD_ARC_POSE_POLL_S)
                continue

            print(f"  [{label}] 时间窗口开始 t={elapsed:.2f}s slot={slot_state['current']}")
            color = None
            rotated = False
            while not stop_event.is_set():
                elapsed = time.monotonic() - started_at

                if color is None and elapsed <= item["window_end"]:
                    used_colors = set(loaded_slot_colors.values()) | active_exclude_colors()
                    color = block.wait_for_display_color(
                        timeout_s=0.03,
                        window_s=CD_DISPLAY_ONCE_WINDOW_S,
                        exclude_colors=used_colors,
                    )
                    if color is not None:
                        if color in loaded_slot_colors.values():
                            print(f"  {label}: 重复颜色 {color}，按未识别处理")
                            color = None
                        else:
                            slot = slot_state["current"]
                            loaded_slot_colors[slot] = color
                            block.set_status(f"{label} -> 槽位 {slot}: {color}")
                            print(f"  {label}: slot {slot} <- color {color} (时间窗口显示确认)")

                if item["rotate_t"] is not None and not rotated and elapsed >= item["rotate_t"]:
                    advance_turntable_after_color(slot_state, f"{label}按时间{item['rotate_t']:.2f}s收纳")
                    rotated = True

                if elapsed > item["window_end"] and (rotated or item["rotate_t"] is None):
                    break
                time.sleep(CD_ARC_POSE_POLL_S)

            if color is None:
                used_colors = set(loaded_slot_colors.values()) | active_exclude_colors()
                debug = block.recent_motion_debug(
                    window_s=CD_DISPLAY_ONCE_WINDOW_S,
                    exclude_colors=used_colors,
                )
                state["missing_colors"].append(label)
                print(
                    f"  {label}: 时间窗口未识别到有效颜色，后续用排除法; "
                    f"samples={debug.get('samples')}, votes={debug.get('votes')}, "
                    f"best_pct={debug.get('best_pct')}"
                )
            state["done"] += 1
            next_idx += 1
            block.clear_recognition_cache()

        if next_idx < len(schedule):
            state["failed"] = [item["label"] for item in schedule[next_idx:]]

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


def run_cd_arc(loaded_slot_colors: dict[int, str],
               slot_state: dict[str, int],
               target_colors: dict[str, str]) -> bool:
    """按用户指定参数走 C/D 圆弧，并在近似坐标处识别颜色、按定时切转盘。"""
    timeout = (math.radians(CD_ARC_SWEEP_DEG) * CD_ARC_RADIUS /
               max(CD_ARC_SPEED, 0.01) * 2.5 + 15.0)
    monitor_thread, monitor_stop, monitor_state = start_arc_color_monitor(
        loaded_slot_colors,
        slot_state,
        target_colors,
    )
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
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=1.0)
    if monitor_state["done"] != len(CD_ARC_COLOR_POINTS):
        print(f"  !! 圆弧颜色识别未完成: done={monitor_state['done']}, missing={monitor_state['failed']}")
        return False
    if monitor_state.get("missing_colors"):
        print(f"  圆弧中有颜色未识别，后续尝试排除法补齐: {monitor_state['missing_colors']}")
    if ok:
        return refresh_cur_from_pose("after C/D arc")
    return False


def prepare_block_scan(arm_state: int, light_id: int) -> bool:
    """机械臂、补光灯和 USB 摄像头尽量同时准备。"""
    def run() -> bool:
        seen = comm.response_seq()
        comm.send_arm(arm_state)
        comm.send_light(light_id, True)
        block.CONFIG["show_window"] = True
        block.set_status("准备物块识别: ARM 1 + LIGHT 4 + USB")
        block.start_viewer()
        block.clear_recognition_cache()
        arm_ok = comm.wait_for_after(comm.TYPE_ARM_RESP, seen, 6.0)
        light_ok = comm.wait_for_after(comm.TYPE_LIGHT_RESP, seen, 5.0)
        if not arm_ok:
            print("  !! 机械臂响应超时或失败")
        if not light_ok:
            print("  !! 补光灯响应超时或失败")
        if arm_ok and light_ok:
            # ARM 1 到位前 viewer 可能已经看到机械臂运动过程，必须清掉这段缓存。
            block.clear_recognition_cache()
            block.set_status("ARM 1 已到位，等待第1个物块稳定识别")
        return arm_ok and light_ok

    print(f"  ARM {arm_state} + LIGHT {light_id} ON + USB CAMERA START")
    return _timed(f"ARM {arm_state} + LIGHT {light_id} ON + USB CAMERA START", run)


def recognize_qr1_targets() -> dict[str, str]:
    seq = _timed("QR1 recognize", qr1.recognize)
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
        raise RuntimeError(f"loaded slot color count invalid: {loaded_slot_colors}")
    color_to_slot: dict[str, int] = {}
    for slot, color in loaded_slot_colors.items():
        if color in color_to_slot:
            raise RuntimeError(
                f"duplicate color {color}: slot {color_to_slot[color]} and slot {slot}"
            )
        color_to_slot[color] = slot
    print(f"  color -> slot: {color_to_slot}")
    return color_to_slot


def complete_loaded_slot_colors_by_exclusion(loaded_slot_colors: dict[int, str],
                                             target_colors: dict[str, str]) -> dict[int, str]:
    """只缺一个槽位颜色时，用二维码给出的五种目标颜色做排除法补齐。"""
    expected_order = list(target_colors.values())
    expected_colors = set(expected_order)
    if len(expected_colors) != CD_SLOT_COUNT:
        raise RuntimeError(f"target colors are not unique: {target_colors}")

    invalid_slots = [slot for slot, color in loaded_slot_colors.items()
                     if color not in expected_colors]
    for slot in invalid_slots:
        print(f"  slot {slot} 颜色 {loaded_slot_colors[slot]} 不在二维码目标颜色内，按未识别处理")
        loaded_slot_colors.pop(slot, None)

    missing_slots = sorted(set(range(CD_SLOT_COUNT)) - set(loaded_slot_colors.keys()))
    used_colors = set(loaded_slot_colors.values())
    missing_colors = [color for color in expected_order if color not in used_colors]

    if not missing_slots and not missing_colors:
        return loaded_slot_colors
    if len(missing_slots) == 1 and len(missing_colors) == 1:
        slot = missing_slots[0]
        color = missing_colors[0]
        loaded_slot_colors[slot] = color
        print(f"  排除法补齐: slot {slot} <- {color}")
        return loaded_slot_colors

    raise RuntimeError(
        f"cannot infer slot colors: loaded={loaded_slot_colors}, "
        f"missing_slots={missing_slots}, missing_colors={missing_colors}"
    )


def rotate_for_target(name: str, target_colors: dict[str, str], color_to_slot: dict[str, int]) -> bool:
    color = target_colors[name]
    if color not in color_to_slot:
        raise RuntimeError(f"target {name} needs color {color}, but slot map is {color_to_slot}")
    slot = color_to_slot[color]
    print(f"  PLACE {name}: need {color}, rotate slot {slot}")
    return rotate(slot)


def ring_place_and_sync(name: str, sync_x: float, sync_y: float) -> bool:
    """调用圆环视觉完成定位/推送/回退，然后按实测点重置下位机XY。"""
    print(f"\n=== {name} 点 ring 定位放置 ===")
    if not ring.align_checked_then_forward(verbose=True):
        return False
    return sync_pose(sync_x, sync_y, None)


def release_block_camera_before_ring() -> None:
    """物块颜色已经完成后，释放 block 摄像头，避免 ring 打开 USB 摄像头失败。"""
    print("\n=== 物块颜色识别完成，释放 block 摄像头，切换到 ring 定位 ===")
    block.close()


def place_target_with_ring(name: str,
                           target_colors: dict[str, str],
                           color_to_slot: dict[str, int],
                           move_order: str = "xy") -> bool:
    """到达放置点后先切转盘，再调用 ring 放置，最后同步该点后的坐标。"""
    x, y = TARGET_POINTS[name]
    sx, sy = PLACE_SYNC_POINTS[name]
    if move_order == "yx":
        if not go_axis_y_then_x(x, y, label=f"move to {name}"):
            return False
    else:
        if not go_axis_x_then_y(x, y, label=f"move to {name}"):
            return False
    if not rotate_for_target(name, target_colors, color_to_slot):
        return False
    return ring_place_and_sync(name, sx, sy)


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

    _require(sync_pose(INIT_X, INIT_Y, INIT_YAW), "SYNC initial pose")
    _require(go_to(-0.662, 0.25), "go to (-0.662, 0.25)")

    target_colors = recognize_qr1_targets()

    _require(turn_to(-90.0), "turn to -90 before C/D arc")
    _require(prepare_block_scan(1, 4), "arm 1, light 4 and USB camera ready")
    time.sleep(CD_ARM_READY_SETTLE_S)
    block.clear_recognition_cache()
    loaded_slot_colors: dict[int, str] = {}
    slot_state = {"current": 0, "load_rotate_count": 0}

    first_thread, first_stop, first_state = start_first_block_monitor(
        loaded_slot_colors,
        slot_state,
    )
    try:
        _require(go_to(*CD_ARC_START), "go to C/D first block static point")
    finally:
        first_stop.set()
        first_thread.join(timeout=0.8)

    if first_state["color"] is None:
        print("  第1个物块颜色未识别，继续收纳，最终只缺一个时用排除法补齐")

    advance_turntable_after_color(slot_state, "第1个物块到圆弧起点后收纳")
    _require(turn_to(CD_ARC_START_YAW), "turn to C/D arc yaw")
    _require(run_cd_arc(loaded_slot_colors, slot_state, target_colors), "C/D arc")
    complete_loaded_slot_colors_by_exclusion(loaded_slot_colors, target_colors)
    color_to_slot = invert_loaded_slot_colors(loaded_slot_colors)
    release_block_camera_before_ring()
    _require(rotate(ROTATE_SPECIAL_STATE), "rotate special state 5 after arc")
    _require(turn_to(180.0), "turn to 180")

    _require(place_target_with_ring("A", target_colors, color_to_slot, move_order="xy"),
             "place A with ring")
    _require(place_target_with_ring("B", target_colors, color_to_slot, move_order="xy"),
             "place B with ring")
    _require(place_target_with_ring("D", target_colors, color_to_slot, move_order="xy"),
             "place D with ring")
    _require(place_target_with_ring("C", target_colors, color_to_slot, move_order="yx"),
             "place C with ring")
    _require(place_target_with_ring("E", target_colors, color_to_slot, move_order="xy"),
             "place E with ring")

    _require(arm(0), "arm 0")
    for x, y in POST_PATH_AFTER_ARM0:
        _require(go_axis_x_then_y(x, y, label=f"exit ({x:.3f}, {y:.3f})"),
                 f"exit to ({x:.3f}, {y:.3f})")
    _require(turn_to(90.0), "turn to 90 final")

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
