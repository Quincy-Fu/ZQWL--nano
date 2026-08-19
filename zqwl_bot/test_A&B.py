#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B 阶段测试流程：二维码、圆弧收纳和三个圆环放置。"""

import math
import os
import sys
import threading
import time

# 摄像头异常时缩短 V4L2 默认等待，避免 USB 打开失败后长时间卡住。
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")
os.environ.setdefault("OPENCV_VIDEOIO_V4L_READ_ATTEMPTS", "1")

import comm
import qr2
import block
import ring


# ============== 配置 ==============
ARC_SPEED_MM_S = 180
ARC_PRELOAD_POSITION = 3
# 圆弧实际经过的三个物块位置顺序：3 → 2 → 1。
ARC_COLLECT_ORDER = (3, 2, 1)
ARC_SWITCH_BY_DEG = [(50.0, 2), (80.0, 1)]
ARC_END_UNUSED_SLOT = 4
ARC_SPEED_MPS = ARC_SPEED_MM_S / 1000.0
ARC_RADIUS_M = 0.84
ARC_DIR = -1
ARC_SWEEP_DEG = 130.0

INIT_YAW_D = 90.0
HEADING_D = 0.0
# 兼容旧版测试辅助函数的默认后退距离。
BACKUP_M = 0.05
PLACE_BACKUP_M = 0.10
SPLIT_THRESHOLD = 0.03
AB_PATH_EPS = 1e-4
AB_KEY_PATH_SPEED = 0.40
QR2_RECOGNIZE_TIMEOUT_S = 18.0

RANK_RING_BACK_MM = 200.0
RANK_RING_BACK_TIMEOUT_S = 45.0
# BODY_POS 完成后给电机驱动器留出额外稳定时间，再同步坐标。
BODY_POS_POST_SETTLE_S = 0.30

# 冠军、亚军、季军固定对应 A、B、C。
RANK_PLACE_POINTS = {
    "亚军": ("B", 0.25, 1.779, 3, 0.25, 1.67),
    "冠军": ("A", 0.00, 1.779, 4, 0.00, 1.66),
    "季军": ("C", -0.10, 1.73, 1, -0.25, 1.67),
}

# 当前转盘映射：0=B，1=A，2=C；槽位4作为圆弧结束后的空槽。
ALPHA_TO_POS = {"B": 0, "A": 1, "C": 2}

_CUR_X, _CUR_Y = 0.0, 0.0


def _update_cur(x, y):
    global _CUR_X, _CUR_Y
    _CUR_X, _CUR_Y = float(x), float(y)


def refresh_cur_from_pose():
    """从下位机读取新鲜位姿，作为后续分段路径的起点。"""
    pose = comm.get_pose(max_age=1.0)
    if pose is None:
        print("  !! 没有新鲜 POSE，本地分段起点沿用上一次记录")
        return False
    _update_cur(pose[0], pose[1])
    print(f"  当前位姿刷新: ({pose[0]:.4f}, {pose[1]:.4f}), yaw={pose[2]:.1f}°")
    return True


def _current_yaw(default=HEADING_D):
    pose = comm.get_pose(max_age=1.0)
    return float(pose[2]) if pose is not None else float(default)


def go_to(x, y):
    print(f"  GOTO ({x:.4f}, {y:.4f})")
    ok = comm.goto(x, y, timeout=40.0)
    if ok:
        _update_cur(x, y)
    return ok


def _key_path_points(points, label, speed=AB_KEY_PATH_SPEED):
    """用短 key_path 发送连续轴向段，失败时由调用方回退 GOTO。"""
    compact = []
    for point in points:
        if compact:
            last = compact[-1]
            if (abs(last[0] - point[0]) <= AB_PATH_EPS and
                    abs(last[1] - point[1]) <= AB_PATH_EPS and
                    abs(last[2] - point[2]) <= 0.1):
                continue
        compact.append(tuple(point))
    if len(compact) < 2:
        return True
    print(f"  KEY_PATH {label}: {len(compact)} pts, speed={speed:.2f}")
    ok = comm.key_path(compact, speed=speed, prepend_current=False)
    if ok:
        _update_cur(compact[-1][0], compact[-1][1])
    return ok


def path_to(x, y, label="axis move", x_first=False):
    """按纯横移/直行规划到目标点，不发送斜线段。"""
    sx, sy = _CUR_X, _CUR_Y
    yaw = _current_yaw()
    points = [(sx, sy, yaw)]
    if x_first:
        if abs(x - sx) > AB_PATH_EPS:
            points.append((x, sy, yaw))
        if abs(y - sy) > AB_PATH_EPS:
            points.append((x, y, yaw))
    else:
        if abs(y - sy) > AB_PATH_EPS:
            points.append((sx, y, yaw))
        if abs(x - sx) > AB_PATH_EPS:
            points.append((x, y, yaw))
    if len(points) < 2:
        return True
    if _key_path_points(points, label):
        return True
    print(f"  [WARN] KEY_PATH {label}失败，回退单轴GOTO")
    refresh_cur_from_pose()
    if x_first:
        if abs(x - _CUR_X) > AB_PATH_EPS and not go_to(x, _CUR_Y):
            return False
        if abs(y - _CUR_Y) > AB_PATH_EPS and not go_to(x, y):
            return False
    else:
        if abs(y - _CUR_Y) > AB_PATH_EPS and not go_to(_CUR_X, y):
            return False
        if abs(x - _CUR_X) > AB_PATH_EPS and not go_to(x, _CUR_Y):
            return False
    return True


def go_to_split(x, y, x_first=False):
    """兼容旧调用名，统一转到纯轴向 path。"""
    return path_to(x, y, label="分轴移动", x_first=x_first)


def move_first_rank_y_then_x(rank_name):
    """圆弧结束后先直行到名次点 Y，再横移到 X。"""
    _letter, target_x, target_y, _arm_state, _sync_x, _sync_y = RANK_PLACE_POINTS[rank_name]
    print(
        f"  [AB] {rank_name}前置实走: "
        f"先 Y {_CUR_Y:.4f}->{target_y:.4f}, 再 X {_CUR_X:.4f}->{target_x:.4f}"
    )
    return path_to(target_x, target_y, label=f"{rank_name}前置Y->X", x_first=False)


def turn_to(deg):
    print(f"  TURNTO {deg:.1f}°")
    return comm.turnto(deg, timeout=30.0)


def rotate(pos):
    print(f"  ROTATE {pos}")
    return comm.rotate(pos, timeout=12.0)


def arm(state):
    print(f"  ARM {state}")
    return comm.arm(state, timeout=6.0)


def arm_light(state, light_id=4, light_on=True):
    """同时下发机械臂和补光灯，分别等待响应。"""
    print(f"  ARM {state} + LIGHT {light_id} {'ON' if light_on else 'OFF'}")
    seen = comm.response_seq()
    comm.send_arm(state)
    comm.send_light(light_id, light_on)
    arm_ok = comm.wait_for_after(comm.TYPE_ARM_RESP, seen, 6.0)
    light_ok = comm.wait_for_after(comm.TYPE_LIGHT_RESP, seen, 5.0)
    return bool(arm_ok and light_ok)


def arm_and_rotate_async(arm_state, slot):
    """圆弧开始前同时下发机械臂和转盘，不等待响应。"""
    print(f"  ARM {arm_state} + ROTATE {slot} (不等待，圆弧同步开始)")
    comm.send_arm(arm_state)
    comm.send_rotate(slot)
    return True


def arm_and_rotate(arm_state, slot):
    print(f"  ARM {arm_state} + ROTATE {slot} (等待到位)")
    seen = comm.response_seq()
    comm.send_arm(arm_state)
    comm.send_rotate(slot)
    return bool(
        comm.wait_for_after(comm.TYPE_ARM_RESP, seen, 6.0)
        and comm.wait_for_after(comm.TYPE_ROTATE_RESP, seen, 12.0)
    )


def arm_async(state):
    print(f"  ARM {state} (不等待)")
    comm.send_arm(state)
    return True


def sync_pose(x, y, yaw):
    yaw_text = "保持当前" if yaw is None else f"{yaw:.1f}°"
    print(f"  SYNC POSE ({x:.4f}, {y:.4f}, yaw={yaw_text})")
    ok = comm.sync_pose(x, y, yaw, timeout=5.0)
    if ok:
        _update_cur(x, y)
    return ok


def arc_with_waypoints(r, dir_, sweep_deg, on_waypoints):
    """走圆弧 + 定时回调. dir=-1 左(逆时针), +1 右(顺时针)."""
    def timer():
        t0 = time.time()
        for t, cb in on_waypoints:
            wait = t - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
            cb()
    th = threading.Thread(target=timer, daemon=True)
    th.start()
    dname = "左转(CCW)" if dir_ < 0 else "右转(CW)"
    print(f"  ARC r={r:.3f}m {dname} sweep={sweep_deg:.1f}°")
    ok = comm.arc(r, dir_, sweep_deg)
    th.join()
    return ok


def place_and_backup(x, y, rotate_pos, heading=HEADING_D, backup=BACKUP_M):
    """朝 heading 走到 (x,y) → 调转盘 → 放物 → 反方向后移 backup 米"""
    turn_to(heading)
    comm.goto(x, y, timeout=40.0)
    comm.rotate(rotate_pos, timeout=12.0)
    # TODO: 实际放物 (无机械臂)
    rad = math.radians(heading)
    bx = x - backup * math.sin(rad)
    by = y - backup * math.cos(rad)
    print(f"  后移到 ({bx:.4f}, {by:.4f})")
    comm.goto(bx, by, timeout=10.0)


def place_rank(rank_name, letter, x, y, arm_state, after_arm_state):
    """到达名次点后切机械臂、切转盘、后退 10cm，并同步切后续机械臂状态。"""
    slot = ALPHA_TO_POS[letter]
    print(f"  {rank_name}: {letter} -> 转盘槽位 {slot}, 目标=({x:.4f}, {y:.4f})")
    if not go_to_split(x, y):
        return False
    if not arm_and_rotate(arm_state, slot):
        return False
    bx, by = x, y - PLACE_BACKUP_M
    print(f"  {rank_name}: 后退 10cm 到 ({bx:.4f}, {by:.4f}), 同步切机械臂状态 {after_arm_state}")
    arm_async(after_arm_state)
    return go_to_split(bx, by)


def place_rank_with_ring(rank_name, move_to_target=True):
    """到名次点后先等转盘到位，再切机械臂/补光灯，调用 ring 放置。"""
    letter, x, y, arm_state, sync_x, sync_y = RANK_PLACE_POINTS[rank_name]
    slot = ALPHA_TO_POS[letter]
    print(f"\n=== {rank_name}: {letter} -> 槽位 {slot}, 目标=({x:.4f}, {y:.4f}) ===")
    if move_to_target:
        if not go_to_split(x, y):
            return False
    elif abs(x - _CUR_X) > SPLIT_THRESHOLD or abs(y - _CUR_Y) > SPLIT_THRESHOLD:
        print(f"  !! {rank_name}: 预走后仍不在目标附近，补走到目标点")
        if not go_to_split(x, y):
            return False
    t0 = time.monotonic()
    rotate_ok = rotate(slot)
    print(f"  [TIMING] {rank_name} rotate slot {slot}: {time.monotonic() - t0:.3f}s -> {rotate_ok}")
    if not rotate_ok:
        return False
    t0 = time.monotonic()
    arm_ok = arm_light(arm_state, light_id=4, light_on=True)
    print(f"  [TIMING] {rank_name} arm {arm_state} + light 4: {time.monotonic() - t0:.3f}s -> {arm_ok}")
    if not arm_ok:
        return False
    if not ring.align_checked_then_forward(verbose=True, back_mm=0.0):
        return False
    print(
        f"  {rank_name}: BODY_POS 后退 {RANK_RING_BACK_MM:.0f}mm 脱离接触，"
        f"完成后同步到 ({sync_x:.3f}, {sync_y:.3f})"
    )
    back_ok = comm.body_pos_move(0.0, -RANK_RING_BACK_MM, timeout=RANK_RING_BACK_TIMEOUT_S)
    if not back_ok:
        print(f"  [WARN] {rank_name} BODY_POS 后退未确认完成；不重发，继续同步到后退坐标")
    print(f"  BODY_POS 后退完成，等待 {BODY_POS_POST_SETTLE_S:.2f}s 后同步坐标")
    time.sleep(BODY_POS_POST_SETTLE_S)
    return sync_pose(sync_x, sync_y, None)


def return_home_strict() -> bool:
    """AB回原点走path，但路径点强制拆成横移、直行、横移，禁止斜线直连。"""
    print("  回原点严格PATH: 横移到 X=0 -> 直行到 Y=0 -> 横移到 X=-0.05")
    sx, sy = _CUR_X, _CUR_Y
    yaw = _current_yaw(default=HEADING_D)
    pts = [(sx, sy, yaw)]
    if abs(sx - 0.0) > AB_PATH_EPS:
        pts.append((0.0, sy, yaw))
    if abs(sy - 0.0) > AB_PATH_EPS:
        pts.append((0.0, 0.0, yaw))
    pts.append((-0.05, 0.0, yaw))
    if _key_path_points(pts, "回原点严格分段", speed=AB_KEY_PATH_SPEED):
        return True
    print("  [WARN] 回原点KEY_PATH失败，回退单轴GOTO分段")
    refresh_cur_from_pose()
    if abs(_CUR_X - 0.0) > AB_PATH_EPS and not go_to(0.0, _CUR_Y):
        return False
    if abs(_CUR_Y - 0.0) > AB_PATH_EPS and not go_to(_CUR_X, 0.0):
        return False
    if abs(_CUR_X - (-0.05)) > AB_PATH_EPS and not go_to(-0.05, _CUR_Y):
        return False
    return True


# ============== 阶段 A + B ==============
def run_task_ab():
    print("\n=== 阶段 A+B: 按指定路线测试 ===")

    if not comm.use_encoder_yaw(timeout=2.0):
        raise RuntimeError("A/B yaw源切到编码器失败")

    # 1. 起步直接把当前位置同步为扫码点，不做前置 10cm 移动。
    if not sync_pose(0.7, 0.25, INIT_YAW_D):
        raise RuntimeError("初始位姿同步失败")

    # 2. QR2 识别 1->2->3 三个位置上的 ABC 摆放顺序
    seq2 = qr2.recognize(timeout=QR2_RECOGNIZE_TIMEOUT_S)  # 例: "CAB"
    pos_to_slot = {
        pos_no: ALPHA_TO_POS[letter]
        for pos_no, letter in zip((1, 2, 3), seq2)
    }
    print(f"  QR2: {seq2} -> 1/2/3位置转盘槽位 {pos_to_slot}")
    collect_slots = [pos_to_slot[pos_no] for pos_no in ARC_COLLECT_ORDER]
    collect_letters = [seq2[pos_no - 1] for pos_no in ARC_COLLECT_ORDER]
    print(
        f"  圆弧经过顺序 3→2→1: 字母 {collect_letters} -> 槽位 {collect_slots} "
        f"(B=0, A=1, C=2)"
    )

    # 3. 扫码完成后先同步进入取放状态，并让转盘切到第一个经过的位置 3；随后前进 5cm 再进圆弧。
    first_slot = collect_slots[0]
    first_letter = collect_letters[0]
    print(f"  [圆弧预置位置 {ARC_PRELOAD_POSITION}] 先下发ARM1+转盘槽位 {first_slot} ({first_letter})，同时前进5cm")
    arm_and_rotate_async(1, first_slot)

    # 扫码完成后按当前初始朝向 90° 前进 5cm：即 X 增加 0.05m，再开始 A/B 圆弧。
    if not go_to(0.75, 0.25):
        raise RuntimeError("扫码后前进 5cm 失败")

    arc_r = 0.86
    arc_dir = -1
    arc_sweep_deg = 130.0
    arc_speed = ARC_SPEED_MM_S / 1000.0
    arc_triggers = [
        (ARC_SWITCH_BY_DEG[0][0], collect_slots[1]),
        (ARC_SWITCH_BY_DEG[1][0], collect_slots[2]),
        (arc_sweep_deg, ARC_END_UNUSED_SLOT),
    ]
    arc_timeout = math.radians(arc_sweep_deg) * arc_r / arc_speed * 2.5 + 15.0
    print(f"  ARC_ROTATE triggers={arc_triggers}, timeout={arc_timeout:.1f}s")
    print("  [圆弧] 已下发复合命令；等待下位机 0x30 响应。若车不动，优先检查下位机是否已烧录新固件。")
    if not comm.arc_rotate(arc_r, arc_dir, arc_sweep_deg, arc_speed, arc_triggers,
                           timeout=arc_timeout):
        raise RuntimeError("arc_rotate failed: 未收到 0x30 响应；请确认下位机已烧录支持 ARC_ROTATE 的新固件")

    # 4. 圆弧后进入冠亚季放置流程。
    refresh_cur_from_pose()
    turn_to(0)
    arm(2)
    if not move_first_rank_y_then_x("季军"):
        raise RuntimeError("圆弧后首个名次点 Y->X 实走失败")
    print("  [AB] 放置顺序: 季军 -> 冠军 -> 亚军")

    if not place_rank_with_ring("季军", move_to_target=False):
        raise RuntimeError("季军 放置流程失败")
    arm(2)
    if not path_to(0.0, 1.67, label="季军后到冠军前置点", x_first=True):
        raise RuntimeError("季军后横移到冠军前置点失败")

    if not place_rank_with_ring("冠军"):
        raise RuntimeError("冠军 放置流程失败")
    arm(2)
    if not path_to(0.25, 1.67, label="冠军后到亚军前置点", x_first=True):
        raise RuntimeError("冠军后横移到亚军前置点失败")

    if not place_rank_with_ring("亚军"):
        raise RuntimeError("亚军 放置流程失败")
    arm(0)
    if not turn_to(0):
        print("  [WARN] 回程转到 0° 失败，继续执行回程")
    if not return_home_strict():
        raise RuntimeError("回到 (-0.05, 0.0) 失败")

    print("\n[完成] 阶段 A + B 测试结束")


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"  SERIAL {port or 'AUTO'} @ 115200")
    comm.init(port, 115200)
    ring.configure({"comm_port": port, "comm_baud": 115200})
    time.sleep(1.0)
    try:
        run_task_ab()
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
            ring.close()
        except Exception as e:
            print(f"[清理] ring camera close failed: {e}")
        comm.shutdown()


if __name__ == "__main__":
    main()
