#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_C&D.py - C/D 区新流程测试。

流程按实车调试顺序写死：先同步位姿，二维码识别得到 A/B/C/D/E 目标颜色，
再转到 -90° 准备机械臂/补光灯/USB，按实车经过顺序识别 5 个物块并同步收纳，
随后按 A/B/D/C/E 顺序到点、按颜色切转盘并后退 5cm。放置段只走横移+直行，不走斜线。
"""

import math
import threading
import time
from typing import Optional

import comm
import qr1
import block


PORT = "/dev/ttyCH341USB0"
BAUDRATE = 115200

INIT_X = 0.0
INIT_Y = 0.0
INIT_YAW = 90.0

KEY_PATH_SPEED = 0.60
KEY_PATH_XY_EPS = 1e-4
KEY_PATH_SEG_TIMEOUT_MIN = 25.0
KEY_PATH_TURN_TIMEOUT = 25.0
KEY_PATH_ROTATE_TIMEOUT = 12.0
AXIS_MOVE_EPS = 1e-4
PLACE_BACKUP_Y = 0.05

CD_ARC_START = (-0.900, 0.250)
CD_ARC_START_YAW = -69.0
CD_ARC_RADIUS = 0.869
CD_ARC_DIR = 1
CD_ARC_SWEEP_DEG = 130.0
CD_ARC_SPEED = 0.18
CD_SLOT_COUNT = 5
CD_ARC_TRIGGER_RADIUS_M = 0.38
CD_ARC_POSE_POLL_S = 0.02
CD_ARC_RECOGNIZE_FRAMES = 2
CD_ARC_RECOGNIZE_TIMEOUT_S = 0.18
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
    "A": (-0.620, 1.650),
    "B": (-0.400, 1.370),
    "C": ( 0.290, 1.600),
    "D": ( 0.400, 1.310),
    "E": ( 0.900, 1.390),
}
QR_TARGET_ORDER = ["A", "B", "C", "D", "E"]
PLACE_ORDER = ["A", "B", "D", "C", "E"]
FIRST_PLACE_APPROACH = (-0.620, 1.751)

POST_PATH_AFTER_ARM0 = [
    (0.600, 1.390),
    (0.600, 0.250),
    (0.100, 0.250),
    (0.100, -0.050),
]

_CUR_X = 0.0
_CUR_Y = 0.0


def _update_cur(x: float, y: float) -> None:
    global _CUR_X, _CUR_Y
    _CUR_X, _CUR_Y = float(x), float(y)


def _timed(label: str, func, *args, **kwargs):
    t0 = time.monotonic()
    print(f"[TIMING] {label} start", flush=True)
    result = func(*args, **kwargs)
    print(f"[TIMING] {label} done {time.monotonic() - t0:.3f}s -> {result}", flush=True)
    return result


def _require(ok: bool, label: str) -> None:
    if not ok:
        raise RuntimeError(f"{label} failed")


def sync_pose(x: float, y: float, yaw_deg: float) -> bool:
    ok = _timed(
        f"SYNC pose ({x:.3f}, {y:.3f}, yaw={yaw_deg:.1f})",
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


def backup_y_after_place(name: str) -> bool:
    """放置点动作后，沿 +Y 后退 5cm。"""
    backup_y = _CUR_Y + PLACE_BACKUP_Y
    print(f"  BACKUP {name}: y + {PLACE_BACKUP_Y:.3f}m -> ({_CUR_X:.4f}, {backup_y:.4f})")
    return go_to(_CUR_X, backup_y)


def turn_to(deg: float, timeout: float = 30.0) -> bool:
    print(f"  TURNTO {deg:.1f}°")
    return _timed(f"TURNTO {deg:.1f}", comm.turnto, deg, timeout=timeout)


def arm(state: int, timeout: float = 6.0) -> bool:
    print(f"  ARM {state}")
    return _timed(f"ARM {state}", comm.arm, state, timeout=timeout)


def rotate(pos: int, timeout: float = 12.0) -> bool:
    print(f"  ROTATE slot {pos}")
    return _timed(f"ROTATE slot {pos}", comm.rotate, pos, timeout=timeout)


def rotate_async(pos: int, label: str = "") -> bool:
    """只下发转盘槽位命令，不等待响应；用于和转向/圆弧并行动作。"""
    suffix = f" {label}" if label else ""
    print(f"  ROTATE slot {pos} (不等待{suffix})")
    comm.send_rotate(pos)
    return True


def advance_turntable_after_color(slot_state: dict[str, int], label: str) -> int:
    """识别到一个物块后，把转盘切到下一槽位用于收纳/准备下一个物块。"""
    next_slot = (slot_state["current"] + 1) % CD_SLOT_COUNT
    rotate_async(next_slot, label=label)
    slot_state["current"] = next_slot
    return next_slot


def recognize_current_slot_color(loaded_slot_colors: dict[int, str],
                                 slot_state: dict[str, int],
                                 label: str,
                                 stable: bool) -> Optional[str]:
    """识别当前经过的物块颜色，并记录到当前转盘槽位。"""
    slot = slot_state["current"]
    block.set_status(f"正在识别{label} -> 槽位 {slot}")
    if stable:
        color = _timed(f"BLOCK recognize {label} slot {slot}", block.recognize_stable,
                       frames=10, timeout=3.0)
    else:
        color = _timed(f"BLOCK recognize {label} slot {slot}", block.recognize_motion,
                       window_s=0.28,
                       min_hits=2)
    if color is None:
        debug = block.recent_motion_debug(window_s=0.45)
        print(
            f"  !! {label} 运动识别失败: samples={debug['samples']}, "
            f"age={debug['newest_age']}, votes={debug['votes']}, "
            f"best_pct={debug['best_pct']}",
            flush=True,
        )
        block.set_status(f"{label} 识别失败")
        return None
    loaded_slot_colors[slot] = color
    block.clear_recent_colors()
    block.set_status(f"{label} -> 槽位 {slot}: {color}")
    print(f"  {label}: slot {slot} <- color {color}")
    return color


def start_arc_color_monitor(loaded_slot_colors: dict[int, str],
                            slot_state: dict[str, int]) -> tuple[threading.Thread, threading.Event, dict]:
    """圆弧期间按近似坐标触发颜色识别，成功后立即切下一槽位。"""
    stop_event = threading.Event()
    state = {"done": 0, "failed": [], "last_pose": None}

    def worker() -> None:
        next_idx = 0
        while not stop_event.is_set() and next_idx < len(CD_ARC_COLOR_POINTS):
            pose = comm.get_pose(max_age=0.5)
            if pose is None:
                time.sleep(CD_ARC_POSE_POLL_S)
                continue
            x, y, yaw = pose
            state["last_pose"] = pose
            label, tx, ty = CD_ARC_COLOR_POINTS[next_idx]
            dist = math.hypot(x - tx, y - ty)
            if dist <= CD_ARC_TRIGGER_RADIUS_M:
                print(f"  [{label}] 进入识别范围 dist={dist:.3f}m, pose=({x:.4f},{y:.4f},yaw={yaw:.1f}°)")
                debug = block.recent_motion_debug(window_s=0.45)
                print(
                    f"  [{label}] recent block samples={debug['samples']}, "
                    f"votes={debug['votes']}, best_pct={debug['best_pct']}",
                    flush=True,
                )
                color = recognize_current_slot_color(loaded_slot_colors, slot_state,
                                                     label, stable=False)
                if color is not None:
                    advance_turntable_after_color(slot_state, f"{label}识别后收纳")
                    state["done"] += 1
                    next_idx += 1
                else:
                    # 识别失败时短暂重试；如果还在范围内，下一轮会继续尝试。
                    time.sleep(0.05)
            else:
                time.sleep(CD_ARC_POSE_POLL_S)
        if next_idx < len(CD_ARC_COLOR_POINTS):
            state["failed"] = [p[0] for p in CD_ARC_COLOR_POINTS[next_idx:]]

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


def run_cd_arc(loaded_slot_colors: dict[int, str], slot_state: dict[str, int]) -> bool:
    """按用户指定参数走 C/D 圆弧，并在近似坐标处识别颜色、切转盘。"""
    timeout = (math.radians(CD_ARC_SWEEP_DEG) * CD_ARC_RADIUS /
               max(CD_ARC_SPEED, 0.01) * 2.5 + 15.0)
    monitor_thread, monitor_stop, monitor_state = start_arc_color_monitor(
        loaded_slot_colors,
        slot_state,
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
        block.clear_recent_colors()
        arm_ok = comm.wait_for_after(comm.TYPE_ARM_RESP, seen, 6.0)
        light_ok = comm.wait_for_after(comm.TYPE_LIGHT_RESP, seen, 5.0)
        if not arm_ok:
            print("  !! 机械臂响应超时或失败")
        if not light_ok:
            print("  !! 补光灯响应超时或失败")
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


def rotate_for_target(name: str, target_colors: dict[str, str], color_to_slot: dict[str, int]) -> bool:
    color = target_colors[name]
    if color not in color_to_slot:
        raise RuntimeError(f"target {name} needs color {color}, but slot map is {color_to_slot}")
    slot = color_to_slot[color]
    print(f"  PLACE {name}: need {color}, rotate slot {slot}")
    return rotate(slot)


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
    _require(go_to(0.0, 0.25), "go to (0, 0.25)")
    _require(go_to(-0.662, 0.25), "go to (-0.662, 0.25)")

    target_colors = recognize_qr1_targets()

    _require(turn_to(-90.0), "turn to -90 before C/D arc")
    _require(prepare_block_scan(1, 4), "arm 1, light 4 and USB camera ready")
    loaded_slot_colors: dict[int, str] = {}
    slot_state = {"current": 0}
    _require(
        recognize_current_slot_color(loaded_slot_colors, slot_state,
                                     "第1个物块", stable=True) is not None,
        "recognize first block color",
    )

    _require(go_to(*CD_ARC_START), "go to C/D arc start")
    advance_turntable_after_color(slot_state, "第1个物块到圆弧起点后收纳")
    _require(turn_to(CD_ARC_START_YAW), "turn to C/D arc yaw")
    _require(run_cd_arc(loaded_slot_colors, slot_state), "C/D arc")
    color_to_slot = invert_loaded_slot_colors(loaded_slot_colors)
    _require(turn_to(180.0), "turn to 180")

    _require(go_axis_x_then_y(*FIRST_PLACE_APPROACH, label="first approach before A"),
             "first approach before A")
    for name in PLACE_ORDER:
        x, y = TARGET_POINTS[name]
        _require(go_axis_x_then_y(x, y, label=f"move to {name}"), f"move to {name}")
        _require(rotate_for_target(name, target_colors, color_to_slot), f"rotate for {name}")
        _require(backup_y_after_place(name), f"backup after {name}")

    _require(arm(0), "arm 0")
    for x, y in POST_PATH_AFTER_ARM0:
        _require(go_axis_x_then_y(x, y, label=f"exit ({x:.3f}, {y:.3f})"),
                 f"exit to ({x:.3f}, {y:.3f})")

    print("\n=== test_C&D 流程完成 ===")


def main() -> None:
    comm.init(PORT, BAUDRATE)
    time.sleep(1.0)
    try:
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
        comm.shutdown()


if __name__ == "__main__":
    main()
