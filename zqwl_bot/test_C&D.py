#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_task_c.py - 阶段 C 测试 (恢复原版起点, 接入 block 多帧)
"""
import math
import time
import threading

import comm
import qr1, block


# ============== 配置 ==============
ARC_SPEED_MM_S = 200
AFTER_RECOGNIZE_DELAY_S = 1.0
HEADING_D = 180
BACKUP_M = 0.05
SPLIT_THRESHOLD = 0.03
INIT_YAW_D = -90.0

_CUR_X, _CUR_Y = 0.0, 0.0

TARGETS = [
    (-0.63203, 1.45514),
    ( 0.22038, 1.45564),
    (-0.42737, 0.71556),
    ( 0.33466, 0.71464),
    ( 0.83621, 0.71518),
]


# ============== 角度工具 ==============
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


def _timed(label, func, *args, **kwargs):
    t0 = time.monotonic()
    print(f"[TIMING] {label} start", flush=True)
    result = func(*args, **kwargs)
    print(f"[TIMING] {label} done {time.monotonic() - t0:.3f}s -> {result}", flush=True)
    return result


def go_to(x, y, x_first=False):
    """分段走到 (x,y)：两轴差距都较大时拆成平移/直行两段。"""
    global _CUR_X, _CUR_Y
    print(f"  GOTO ({x:.4f}, {y:.4f})")
    dx = abs(x - _CUR_X)
    dy = abs(y - _CUR_Y)
    if dx > SPLIT_THRESHOLD and dy > SPLIT_THRESHOLD:
        if x_first:
            print(f"    -> 拆段: ({x:.4f}, {_CUR_Y:.4f}) -> ({x:.4f}, {y:.4f})")
            ok = _timed(f"GOTO seg1 ({x:.4f}, {_CUR_Y:.4f})", comm.goto, x, _CUR_Y, timeout=40.0)
            if not ok:
                return ok
            ok = _timed(f"GOTO seg2 ({x:.4f}, {y:.4f})", comm.goto, x, y, timeout=40.0)
        else:
            print(f"    -> 拆段: ({_CUR_X:.4f}, {y:.4f}) -> ({x:.4f}, {y:.4f})")
            ok = _timed(f"GOTO seg1 ({_CUR_X:.4f}, {y:.4f})", comm.goto, _CUR_X, y, timeout=40.0)
            if not ok:
                return ok
            ok = _timed(f"GOTO seg2 ({x:.4f}, {y:.4f})", comm.goto, x, y, timeout=40.0)
    else:
        ok = _timed(f"GOTO direct ({x:.4f}, {y:.4f})", comm.goto, x, y, timeout=40.0)
    if ok:
        _update_cur(x, y)
    return ok


def turn_to(deg):
    """原地转到指定车体角度。"""
    print(f"  TURNTO {deg:.1f}° (车体)")
    return _timed(f"TURNTO {deg:.1f}", comm.turnto, deg, timeout=30.0)

def rotate(pos):
    print(f"  ROTATE {pos}")
    return _timed(f"ROTATE {pos}", comm.rotate, pos, timeout=12.0)


def arc_with_waypoints(r, dir_, sweep_deg, on_waypoints):
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
    turn_to(heading)
    go_to(x, y)
    rotate(rotate_pos)
    rad = math.radians(heading)
    bx = x - backup * math.sin(rad)
    by = y - backup * math.cos(rad)
    print(f"  后移到 ({bx:.4f}, {by:.4f})")
    go_to(bx, by)

def arm(state):
    print(f"  ARM {state}")
    return _timed(f"ARM {state}", comm.arm, state, timeout=6.0)


def sync_initial_pose():
    """任务开始前把下位机当前位置基准重置为 (0,0,-90°)。"""
    _update_cur(0.0, 0.0)
    ok = _timed("SYNC initial pose (0,0,-90deg)", comm.sync_pose,
                0.0, 0.0, INIT_YAW_D, timeout=5.0)
    if not ok:
        raise RuntimeError("初始位姿同步失败")
    return ok


def run_task_c():
    print("\n=== 阶段 C: 走弧 + 颜色识别 ===")

    # 14
    turn_to(90)
    go_to(-0.66158, 0.29568)

    # 15. QR1
    seq1 = _timed("QR1 recognize", qr1.recognize)
    color_seq = qr1.TASK1_PLANS[seq1]
    print(f"  QR1: {seq1} -> {color_seq}")

    # 16. 恢复原版起点 (-0.6, 0.2)，但用连续 path 到位并转向
    go_to(-0.6, 0.2)
    turn_to(-89)
    arm(1)

    # 17
    rotate(0)

    # 18-19. ★ start_c 跟 go_to 一致, 用 (-0.6, 0.2)
    start_c = (-0.6, 0.2)
    end_c = (-0.63203, 2.02231)
    center_c = (-0.575, 1.055)
    r = 0.910

    sweep_c = arc_sweep(start_c, end_c, center_c, ccw=False)  # 顺时针
    print(f"  走顺时针圆弧, sweep={sweep_c:.1f}°")

    arc_len_c = math.radians(sweep_c) * r
    arc_time_c = arc_len_c / (ARC_SPEED_MM_S / 1000)
    wp_deg_c = [
        user_deg((-1.05824, 0.25825), center_c),
        user_deg((-1.39938, 0.60489), center_c),
        user_deg((-1.524, 1.075), center_c),
        user_deg((-1.39938, 1.54511), center_c),
    ]
    a_s_c = user_deg(start_c, center_c)
    wp_times_c = [arc_time_c * ((w - a_s_c) % 360) / sweep_c for w in wp_deg_c]

    color_at_pos = [None] * 5

    def on_wp_c(idx):
        # ★ 5 帧众数 (快速移动中)
        color = block.recognize(frames=5, timeout=1.5)
        color_at_pos[idx] = color
        print(f"  [过点 {idx}] 颜色 = {color}")
        time.sleep(AFTER_RECOGNIZE_DELAY_S)
        rotate(idx + 1)

    # 走圆弧前开补光灯 4
    print("  LIGHT 4 ON")
    comm.send_light(4, True)
    time.sleep(0.05)

    arc_with_waypoints(r, 1, sweep_c, [
        (wp_times_c[0], lambda: on_wp_c(0)),
        (wp_times_c[1], lambda: on_wp_c(1)),
        (wp_times_c[2], lambda: on_wp_c(2)),
        (wp_times_c[3], lambda: on_wp_c(3)),
    ])
    _update_cur(end_c[0], end_c[1])

    color_at_pos[4] = block.recognize(frames=5, timeout=1.5)
    print(f"  [终点 4] 颜色 = {color_at_pos[4]}")

    color_to_pos = {c: i for i, c in enumerate(color_at_pos) if c}
    print(f"  颜色->位置: {color_to_pos}")

    # === 阶段 D ===
    print("\n=== 阶段 D: 5 个圆环放置 ===")
    for i, (tx, ty) in enumerate(TARGETS):
        color = color_seq[i]
        if color not in color_to_pos:
            print(f"  [!] 第 {i+1}: 颜色 {color} 不在, 跳过")
            continue
        pos = color_to_pos[color]
        print(f"  第 {i+1}: 颜色={color} -> 转盘={pos}, 目标=({tx:.4f}, {ty:.4f})")
        place_and_backup(tx, ty, pos)

    print("\n=== 回 HOME ===")
    print("  ARM 0")
    comm.send_arm(0)
    time.sleep(0.05)
    go_to(0, 0, x_first=True)
    turn_to(0)


def main():
    comm.init("/dev/ttyCH341USB0", 115200)
    time.sleep(1.0)
    try:
        sync_initial_pose()
        run_task_c()
    except RuntimeError as e:
        print(f"\n[兜底] 任务中断: {e}")
    except KeyboardInterrupt:
        print("\n[用户中断]")
    finally:
        comm.shutdown()


if __name__ == "__main__":
    main()
