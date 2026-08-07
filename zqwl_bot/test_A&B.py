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
HEADING_D = 0                  # 阶段 B 放物车头朝向 (车体坐标)

# 阶段 B 3 个目标点
TARGETS_B = [
    ( 0.27017, 1.77938, 1),    # 1号圆环, 转盘 1
    (-0.27017, 1.77938, 2),    # 2号圆环, 转盘 2
    ( 0,         1.77938, 0),   # 3号圆环, 转盘 0
]

ALPHA_TO_POS = {"A": 0, "B": 1, "C": 2}


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
def go_to(x, y):
    print(f"  GOTO ({x:.4f}, {y:.4f})")
    return comm.goto(x, y, timeout=40.0)

def turn_to(deg):
    print(f"  TURNTO {deg:.1f}° (车体)")
    return comm.turnto(deg, timeout=30.0)

def rotate(pos):
    print(f"  ROTATE {pos}")
    return comm.rotate(pos, timeout=12.0)

def arm(state):
    print(f"  ARM {state}")
    return comm.arm(state, timeout=6.0)


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


# ============== 阶段 A + B ==============
def run_task_ab():
    # ===== 阶段 A: 右半区装载 =====
    print("\n=== 阶段 A: 右半区装载 ===")

    # 1. HOME (0,0) -> (0, 0.295), 朝 0°
    turn_to(0)
    go_to(0, 0.295)

    # 2. 朝 90° -> (0.63679, 0.29568)
    turn_to(90)
    go_to(0.63679, 0.29568)

    # 3-4. QR2 识别字母顺序 -> 转盘位置方案
    seq2 = qr2.recognize()                       # "CAB"
    plan = [ALPHA_TO_POS[c] for c in seq2]       # 例 [2, 0, 1]
    print(f"  QR2: {seq2} -> 方案 {plan}")

    # 5. 平移到 (0.63679, 0.12996)
    go_to(0.63679, 0.26)

    # 6. 准备: 调转盘到 plan[0] + 机械臂 1 号位
    rotate(plan[0])
    arm(1)

    # 7. 走逆时针圆弧 (dir=-1) + 2 过点切转盘
    start_a = (0.63679, 0.26)
    end_a = (0.575, 2.024)
    center_a = (0.575, 1.055)
    r = 0.939

    sweep_a = arc_sweep(start_a, end_a, center_a, ccw=True)
    print(f"  走逆时针圆弧, sweep={sweep_a:.1f}°")

    arc_len_a = math.radians(sweep_a) * r
    arc_time_a = arc_len_a / (ARC_SPEED_MM_S / 1000)
    wp1_deg = user_deg((1.46721, 0.65758), center_a)
    wp2_deg = user_deg((1.564, 1.075), center_a)
    a_s_a = user_deg(start_a, center_a)
    t1 = arc_time_a * ((a_s_a - wp1_deg) % 360) / sweep_a
    t2 = arc_time_a * ((a_s_a - wp2_deg) % 360) / sweep_a

    def on_wp_a(idx):
        # 过点 idx+1 1s 后切到 plan[idx+1]
        time.sleep(WAYPOINT_DELAY_S)
        rotate(plan[idx + 1])

    arc_with_waypoints(r, -1, sweep_a, [
        (t1, lambda: on_wp_a(0)),
        (t2, lambda: on_wp_a(1)),
    ])

    # ===== 阶段 B: 右半区圆环放置 =====
    print("\n=== 阶段 B: 3 个圆环放置 (简化版) ===")
    turn_to(0)

    # TODO: 阶段 B 之前应该 arm(3), 然后 3 个 place_and_backup
    # 这里只做 3 个放置, 转盘位置写死
    for i, (tx, ty, pos) in enumerate(TARGETS_B):
        print(f"  第 {i+1} 个: 目标=({tx:.4f}, {ty:.4f}), 转盘={pos}")
        # 调机械臂 (按你之前的规则: 3, 2, 4)
        arm_states = [3, 2, 4]
        arm(arm_states[i])
        place_and_backup(tx, ty, pos)

    # 13. 回到 (0, 0.295)
    go_to(0, 0.295)

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