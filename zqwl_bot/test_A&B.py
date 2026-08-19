#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_A_B.py - 阶段 A 装载 + 阶段 B 圆环放置 测试
1. 阶段 A: 走逆时针圆弧, 装载到转盘
2. 阶段 B: 3 个圆环放置，调用 ring 完成微调放置
"""
import math
import os
import sys
import time
import threading

# 必须在任何视觉模块导入 cv2 前设置，避免 USB 异常时 V4L2 select 默认等待约 10 秒。
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")
os.environ.setdefault("OPENCV_VIDEOIO_V4L_READ_ATTEMPTS", "1")

import comm
import qr2, block, ring


# ============== 配置 ==============
ARC_SPEED_MM_S = 300
WAYPOINT_DELAY_S = 1.0
BACKUP_M = 0.05
PLACE_BACKUP_M = 0.10
HEADING_D = 0                  # 阶段 B 放物车头朝向 (车体坐标)
INIT_YAW_D = 90.0
SPLIT_THRESHOLD = 0.03
AB_KEY_PATH_SPEED = 0.60
AB_PATH_EPS = 1e-4
QR2_RECOGNIZE_TIMEOUT_S = 18.0
RANK_RING_BACK_MM = 200.0
RANK_RING_BACK_TIMEOUT_S = 45.0

# 冠亚季点位。按用户说明: 冠/亚/季 对应 A/B/C。
RANK_PLACE_POINTS = {
    "亚军": ("B", 0.25, 1.779, 3, 0.25, 1.67),
    "冠军": ("A", 0.00, 1.779, 4, 0.00, 1.66),
    "季军": ("C", -0.15, 1.730, 1, -0.25, 1.67),
}

# QR2 返回的 3 个字母对应位置 1、2、3 上的物块顺序；圆弧实际经过顺序固定为 3→2→1。
ARC_COLLECT_ORDER = (3, 2, 1)
ARC_PRELOAD_POSITION = ARC_COLLECT_ORDER[0]
ARC_SWITCH_BY_DEG = [(50.0, ARC_COLLECT_ORDER[1]), (80.0, ARC_COLLECT_ORDER[2])]
ARC_END_UNUSED_SLOT = 4

# 阶段 B 3 个目标点
TARGETS_B = [
    ( 0.27017, 1.77938, 1),    # 1号圆环, 转盘 1
    (-0.27017, 1.77938, 2),    # 2号圆环, 转盘 2
    ( 0,         1.77938, 0),   # 3号圆环, 转盘 0
]

ALPHA_TO_POS = {"B": 0, "A": 1, "C": 2}

_CUR_X, _CUR_Y = 0.0, 0.0


# ============== 角度工具 (算 sweep) ==============
def user_deg(p, c):
    dx, dy = p[0] - c[0], p[1] - c[1]
    return (90 - math.degrees(math.atan2(dy, dx))) % 360

def arc_sweep(start, end, center, ccw):
    a_s = user_deg(start, center)
    a_e = user_deg(end, center)
    if ccw:
        return (a_s - a_e) % 360
    return (a_e - a_s) % 360


# ============== 串口高层 ==============
def _update_cur(x, y):
    global _CUR_X, _CUR_Y
    _CUR_X, _CUR_Y = x, y


def refresh_cur_from_pose():
    """圆弧后用下位机上报位姿刷新本地分段移动起点。"""
    pose = comm.get_pose(max_age=1.0)
    if pose is None:
        print("  !! 没有新鲜 POSE，本地分段起点沿用上一次记录")
        return False
    _update_cur(pose[0], pose[1])
    print(f"  当前位姿刷新: ({pose[0]:.4f}, {pose[1]:.4f}), yaw={pose[2]:.1f}°")
    return True


def go_to(x, y):
    print(f"  GOTO ({x:.4f}, {y:.4f})")
    ok = comm.goto(x, y, timeout=40.0)
    if ok:
        _update_cur(x, y)
    return ok


def _current_yaw(default=0.0):
    """读取当前yaw；path段用它保持现有朝向。"""
    pose = comm.get_pose(max_age=1.0)
    if pose is None:
        return default
    return float(pose[2])


def _key_path_points(points, label, speed=AB_KEY_PATH_SPEED):
    """A/B放置阶段短分段优先用key_path，减少多条GOTO之间的停顿。"""
    compact = []
    for p in points:
        if compact:
            last = compact[-1]
            if (abs(last[0] - p[0]) <= AB_PATH_EPS and
                    abs(last[1] - p[1]) <= AB_PATH_EPS and
                    abs(last[2] - p[2]) <= 0.1):
                continue
        compact.append(p)
    points = compact
    if len(points) < 2:
        return True
    print(f"  KEY_PATH {label}: {len(points)} pts, speed={speed:.2f}")
    ok = comm.key_path(points, speed=speed, prepend_current=False)
    if ok:
        _update_cur(points[-1][0], points[-1][1])
    return ok


def path_to(x, y, label="path move", x_first=False):
    """优先用key_path走到目标；失败则按原分轴顺序回退GOTO。"""
    sx, sy = _CUR_X, _CUR_Y
    yaw = _current_yaw(default=HEADING_D)
    pts = [(sx, sy, yaw)]
    dx = abs(x - sx)
    dy = abs(y - sy)
    if dx > SPLIT_THRESHOLD and dy > SPLIT_THRESHOLD:
        if x_first:
            pts.append((x, sy, yaw))
            pts.append((x, y, yaw))
        else:
            pts.append((sx, y, yaw))
            pts.append((x, y, yaw))
    elif dx > AB_PATH_EPS or dy > AB_PATH_EPS:
        pts.append((x, y, yaw))
    else:
        return True
    if _key_path_points(pts, label):
        return True
    print(f"  [WARN] KEY_PATH {label}失败，回退原分轴GOTO")
    refresh_cur_from_pose()
    cur_dx = abs(x - _CUR_X)
    cur_dy = abs(y - _CUR_Y)
    if cur_dx > SPLIT_THRESHOLD and cur_dy > SPLIT_THRESHOLD:
        if x_first:
            if not go_to(x, _CUR_Y):
                return False
            return go_to(x, y)
        if not go_to(_CUR_X, y):
            return False
        return go_to(x, y)
    return go_to(x, y)


def lock_y_to(y, timeout=40.0):
    """只沿 Y 方向后退/前进：复用 CD 的 TOY 锁轴模式，比普通 GOTO 更直接。"""
    print(f"  LOCK_Y_TO {y:.4f} (keep X={_CUR_X:.4f})")
    ok = comm.lock_axis("y", y, timeout=timeout)
    if ok:
        _update_cur(_CUR_X, y)
    return ok


def go_to_split(x, y, x_first=False):
    """两轴差距都较大时拆成横移和直行两段，减少斜移。"""
    return path_to(x, y, label=f"split move to ({x:.3f},{y:.3f})", x_first=x_first)

def turn_to(deg):
    print(f"  TURNTO {deg:.1f}° (车体)")
    return comm.turnto(deg, timeout=30.0)


def rotate(pos):
    print(f"  ROTATE {pos}")
    return comm.rotate(pos, timeout=12.0)


def rotate_async(pos):
    """只下发转盘切换命令，不等待估算到位响应，用于圆弧过程中提前切槽。"""
    print(f"  ROTATE {pos} (不等待，双发防丢帧)")
    comm.send_rotate(pos, repeats=2, repeat_delay_s=0.08)
    return True


def arm(state):
    print(f"  ARM {state}")
    return comm.arm(state, timeout=6.0)


def arm_and_rotate_async(arm_state, slot):
    """机械臂和转盘同时下发，不等待响应；圆弧立即开始。"""
    print(f"  ARM {arm_state} + ROTATE {slot} (不等待，转盘双发，圆弧同步开始)")
    comm.send_arm(arm_state)
    time.sleep(0.05)
    comm.send_rotate(slot, repeats=2, repeat_delay_s=0.08)
    return True


def arm_and_rotate(arm_state, slot):
    """机械臂和转盘同时下发，两个都确认后才继续。"""
    print(f"  ARM {arm_state} + ROTATE {slot} (等待到位)")
    seen = comm.response_seq()
    comm.send_arm(arm_state)
    time.sleep(0.05)
    comm.send_rotate(slot)
    arm_ok = comm.wait_for_after(comm.TYPE_ARM_RESP, seen, 6.0)
    rotate_ok = comm.wait_for_after(comm.TYPE_ROTATE_RESP, seen, 12.0)
    if not arm_ok:
        print("  !! 机械臂响应超时或失败")
    if not rotate_ok:
        print("  !! 转盘响应超时或失败")
        rotate_ok = comm.rotate(slot, timeout=12.0)
    return arm_ok and rotate_ok


def arm_light(arm_state, light_id=4, light_on=True):
    """机械臂和补光灯同时下发，全部确认后再进入 ring 放置。"""
    print(f"  ARM {arm_state} + LIGHT {light_id} {'ON' if light_on else 'OFF'} (等待到位)")
    seen = comm.response_seq()
    comm.send_arm(arm_state)
    time.sleep(0.05)
    comm.send_light(light_id, light_on)
    arm_ok = comm.wait_for_after(comm.TYPE_ARM_RESP, seen, 6.0)
    light_ok = comm.wait_for_after(comm.TYPE_LIGHT_RESP, seen, 5.0)
    if not arm_ok:
        print("  !! 机械臂响应超时或失败")
    if not light_ok:
        print("  !! 补光灯响应超时或失败")
    return arm_ok and light_ok


def arm_light_rotate(arm_state, slot, light_id=4, light_on=True):
    """机械臂、补光灯和转盘同时下发，全部确认后再进入 ring 放置。"""
    print(f"  ARM {arm_state} + LIGHT {light_id} {'ON' if light_on else 'OFF'} + ROTATE {slot} (等待到位)")
    seen = comm.response_seq()
    comm.send_arm(arm_state)
    time.sleep(0.05)
    comm.send_light(light_id, light_on)
    time.sleep(0.05)
    comm.send_rotate(slot)
    arm_ok = comm.wait_for_after(comm.TYPE_ARM_RESP, seen, 6.0)
    light_ok = comm.wait_for_after(comm.TYPE_LIGHT_RESP, seen, 5.0)
    rotate_ok = comm.wait_for_after(comm.TYPE_ROTATE_RESP, seen, 12.0)
    if not arm_ok:
        print("  !! 机械臂响应超时或失败")
    if not light_ok:
        print("  !! 补光灯响应超时或失败")
    if not rotate_ok:
        print("  !! 转盘响应超时或失败")
        rotate_ok = comm.rotate(slot, timeout=12.0)
    return arm_ok and light_ok and rotate_ok


def arm_async(state):
    """发送机械臂命令但不等待响应，用于后退时同步切状态。"""
    print(f"  ARM {state} (不等待, 与后退动作衔接)")
    comm.send_arm(state)
    return True


def sync_pose(x, y, yaw):
    if yaw is None:
        print(f"  SYNC POSE ({x:.4f}, {y:.4f}, yaw=保持当前)")
    else:
        print(f"  SYNC POSE ({x:.4f}, {y:.4f}, yaw={yaw:.1f}°)")
    ok = comm.sync_pose(x, y, yaw, timeout=5.0)
    if ok:
        _update_cur(x, y)
    return ok


def move_first_rank_y_then_x(rank_name="季军"):
    """圆弧后首个名次点衔接：先真实走到目标 Y，再横移到目标 X。"""
    _letter, target_x, target_y, _arm_state, _sync_x, _sync_y = RANK_PLACE_POINTS[rank_name]
    print(
        f"  [AB] {rank_name}前置实走: "
        f"先 Y {_CUR_Y:.4f}->{target_y:.4f}, 再 X {_CUR_X:.4f}->{target_x:.4f}"
    )
    return path_to(target_x, target_y, label=f"{rank_name}前置Y->X", x_first=False)


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

    # 1. 起步直接把当前位置同步为扫码点，减少前置移动等待。
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

    # 3. 扫码完成后先同步进入取放状态，并让转盘切到第一个经过的位置 3；随后前进10cm再进圆弧。
    first_slot = collect_slots[0]
    first_letter = collect_letters[0]
    print(f"  [圆弧预置位置 {ARC_PRELOAD_POSITION}] 先下发ARM1+转盘槽位 {first_slot} ({first_letter})，同时前进10cm")
    arm_and_rotate_async(1, first_slot)

    # 扫码完成后按当前初始朝向 90° 前进 10cm：即 X 增加 0.10m，再开始 A/B 圆弧。
    if not go_to(0.80, 0.25):
        raise RuntimeError("扫码后前进 10cm 失败")

    arc_r = 0.84
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
