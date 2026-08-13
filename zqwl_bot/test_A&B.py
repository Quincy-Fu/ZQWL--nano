#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_A_B.py - 阶段 A 装载 + 阶段 B 圆环放置 测试
1. 阶段 A: 走逆时针圆弧, 装载到转盘
2. 阶段 B: 3 个圆环放置 (简化, 不调 ring)
"""
import math
import time
import threading

import comm
import qr2, block


# ============== 配置 ==============
ARC_SPEED_MM_S = 200
WAYPOINT_DELAY_S = 1.0
BACKUP_M = 0.05
PLACE_BACKUP_M = 0.10
HEADING_D = 0                  # 阶段 B 放物车头朝向 (车体坐标)
INIT_YAW_D = 90.0
SPLIT_THRESHOLD = 0.03

# 冠亚季点位。按用户说明: 冠/亚/季 对应 A/B/C。
RANK_TARGETS = [
    ("亚军", "B", 0.25, 1.779, 3, 2),
    ("冠军", "A", -0.02, 1.779, 4, 2),
    ("季军", "C", -0.30, 1.779, 1, 0),
]
# 季军放置后回起点路径：显式拆成直行/横移路点，避免斜移。
RETURN_POINTS = [(-0.30, 0.10), (-0.15, 0.10), (-0.05, 0.10), (0.0, 0.10), (0.0, 0.0)]

# QR2 返回的 3 个字母对应位置 1、2、3 上的物块顺序。
# 圆弧过程中实际依次经过 3、2、1，切换触发点按已转过角度定义。
ARC_PRELOAD_POSITION = 3
ARC_SWITCH_BY_DEG = [(40.0, 2), (70.0, 1)]
ARC_END_UNUSED_SLOT = 4

# 阶段 B 3 个目标点
TARGETS_B = [
    ( 0.27017, 1.77938, 1),    # 1号圆环, 转盘 1
    (-0.27017, 1.77938, 2),    # 2号圆环, 转盘 2
    ( 0,         1.77938, 0),   # 3号圆环, 转盘 0
]

ALPHA_TO_POS = {"A": 0, "B": 1, "C": 2}

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


def go_to_split(x, y, x_first=False):
    """两轴差距都较大时拆成横移和直行两段，减少斜移。"""
    global _CUR_X, _CUR_Y
    dx = abs(x - _CUR_X)
    dy = abs(y - _CUR_Y)
    if dx > SPLIT_THRESHOLD and dy > SPLIT_THRESHOLD:
        if x_first:
            if not go_to(x, _CUR_Y):
                return False
            return go_to(x, y)
        if not go_to(_CUR_X, y):
            return False
        return go_to(x, y)
    return go_to(x, y)

def turn_to(deg):
    print(f"  TURNTO {deg:.1f}° (车体)")
    return comm.turnto(deg, timeout=30.0)

def rotate(pos):
    print(f"  ROTATE {pos}")
    return comm.rotate(pos, timeout=12.0)


def rotate_async(pos):
    """只下发转盘切换命令，不等待估算到位响应，用于圆弧过程中提前切槽。"""
    print(f"  ROTATE {pos} (不等待)")
    comm.send_rotate(pos)
    return True

def arm(state):
    print(f"  ARM {state}")
    return comm.arm(state, timeout=6.0)


def arm_and_rotate_async(arm_state, slot):
    """机械臂和转盘同时下发；只等待机械臂响应，转盘让它后台转。"""
    print(f"  ARM {arm_state} + ROTATE {slot} (转盘不等待)")
    seen = comm.response_seq()
    comm.send_arm(arm_state)
    comm.send_rotate(slot)
    ok = comm.wait_for_after(comm.TYPE_ARM_RESP, seen, 6.0)
    if not ok:
        print("  !! 机械臂响应超时或失败")
    return ok


def arm_and_rotate(arm_state, slot):
    """机械臂和转盘同时下发，两个都确认后才继续。"""
    print(f"  ARM {arm_state} + ROTATE {slot} (等待到位)")
    seen = comm.response_seq()
    comm.send_arm(arm_state)
    comm.send_rotate(slot)
    arm_ok = comm.wait_for_after(comm.TYPE_ARM_RESP, seen, 6.0)
    rotate_ok = comm.wait_for_after(comm.TYPE_ROTATE_RESP, seen, 12.0)
    if not arm_ok:
        print("  !! 机械臂响应超时或失败")
    if not rotate_ok:
        print("  !! 转盘响应超时或失败")
    return arm_ok and rotate_ok


def arm_async(state):
    """发送机械臂命令但不等待响应，用于后退时同步切状态。"""
    print(f"  ARM {state} (不等待, 与后退动作衔接)")
    comm.send_arm(state)
    return True


def sync_pose(x, y, yaw):
    print(f"  SYNC POSE ({x:.4f}, {y:.4f}, yaw={yaw:.1f}°)")
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


# ============== 阶段 A + B ==============
def run_task_ab():
    print("\n=== 阶段 A+B: 按指定路线测试 ===")

    # 1. 起点位姿基准
    if not sync_pose(0.0, 0.0, INIT_YAW_D):
        raise RuntimeError("初始位姿同步失败")

    # 2. 扫码前移动
    go_to(0.0, 0.25)
    go_to(0.7, 0.25)

    # 3. QR2 识别 1->2->3 三个位置上的 ABC 摆放顺序
    seq2 = qr2.recognize()                       # 例: "CAB"
    pos_to_slot = {
        pos_no: ALPHA_TO_POS[letter]
        for pos_no, letter in zip((1, 2, 3), seq2)
    }
    print(f"  QR2: {seq2} -> 1/2/3位置转盘槽位 {pos_to_slot}")

    # 4. 进入圆弧前先进入取放状态，同时转盘切到第一个经过的位置 3
    first_slot = pos_to_slot[ARC_PRELOAD_POSITION]
    first_letter = seq2[ARC_PRELOAD_POSITION - 1]
    print(f"  [圆弧预置位置 {ARC_PRELOAD_POSITION}] 转盘切到槽位 {first_slot} ({first_letter})")
    arm_and_rotate_async(1, first_slot)

    arc_r = 0.84
    arc_dir = -1
    arc_sweep_deg = 130.0
    arc_time = math.radians(arc_sweep_deg) * arc_r / (ARC_SPEED_MM_S / 1000)

    def on_arc_switch(pos_no):
        time.sleep(WAYPOINT_DELAY_S)
        slot = pos_to_slot[pos_no]
        letter = seq2[pos_no - 1]
        print(f"  [圆弧位置 {pos_no}] 转盘切到槽位 {slot} ({letter})")
        rotate_async(slot)

    arc_with_waypoints(arc_r, arc_dir, arc_sweep_deg, [
        (arc_time * trigger_deg / arc_sweep_deg,
         lambda pos_no=pos_no: on_arc_switch(pos_no))
        for trigger_deg, pos_no in ARC_SWITCH_BY_DEG
    ])
    print(f"  [圆弧结束] 转盘切到未使用槽位 {ARC_END_UNUSED_SLOT}")
    rotate_async(ARC_END_UNUSED_SLOT)

    # 5. 圆弧后进入冠亚季放置流程
    refresh_cur_from_pose()
    turn_to(0)
    arm(2)
    for rank_name, letter, x, y, arm_state, after_arm_state in RANK_TARGETS:
        if not place_rank(rank_name, letter, x, y, arm_state, after_arm_state):
            raise RuntimeError(f"{rank_name} 放置流程失败")

    # 6. 后续回退点
    for x, y in RETURN_POINTS:
        if not go_to_split(x, y):
            raise RuntimeError("回退路径失败")

    print("\n[完成] 阶段 A + B 测试结束")


def main():
    comm.init("/dev/ttyCH341USB0", 115200)
    time.sleep(1.0)
    try:
        run_task_ab()
    except RuntimeError as e:
        print(f"\n[兜底] 任务中断: {e}")
    except KeyboardInterrupt:
        print("\n[用户中断]")
    finally:
        comm.shutdown()


if __name__ == "__main__":
    main()
