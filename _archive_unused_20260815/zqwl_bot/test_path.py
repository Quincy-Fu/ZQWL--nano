#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上位机主控制器 (基于 comm API, test_comm.py 同包)
25 步流程: 右半区装载 → 圆环 → 左半区装载 → 圆环 → 回 HOME
"""

import math
import time
import threading

import comm
import qr1, qr2, ring, block


# ============== 配置 ==============
ARC_SPEED_MM_S = 200
WAYPOINT_DELAY_S = 1.0
AFTER_RECOGNIZE_DELAY_S = 1.0
VISION_MAX_CORRECTIONS = 3
ALPHA_TO_POS = {"A": 0, "B": 1, "C": 2}

# TODO: 编号 1-5 对应的颜色
BLOCK_NUM_TO_COLOR = {1: "黑", 2: "白", 3: "红", 4: "绿", 5: "蓝"}


# ============== 角度工具 (用户角度: 0=北, CW+) ==============
def user_deg(p, c):
    dx, dy = p[0] - c[0], p[1] - c[1]
    a = math.degrees(math.atan2(dy, dx))
    return (90 - a) % 360

def arc_sweep(start, end, center, ccw):
    a_s = user_deg(start, center)
    a_e = user_deg(end, center)
    if ccw:
        return (a_s - a_e) % 360
    return (a_e - a_s) % 360

def arc_heading(start, center, dir_):
    theta = user_deg(start, center)
    if dir_ < 0:
        return (theta + 90) % 360
    return (theta - 90) % 360


# ============== 串口高层 ==============
def go_to(x, y):
    print(f"  GOTO ({x:.4f}, {y:.4f})")
    return comm.goto(x, y, timeout=40.0)

def turn_to(deg):
    print(f"  TURNTO {deg:.1f}°")
    return comm.turnto(deg, timeout=30.0)

def rotate(pos):
    print(f"  ROTATE {pos}")
    return comm.rotate(pos, timeout=12.0)

def sync_pose(x, y):
    print(f"  SYNC ({x:.4f}, {y:.4f})")
    comm.send_sync_pose(x, y)
    return comm.wait_for(comm.TYPE_CMD_SYNC_RESP, 5.0)


def arc_with_waypoints(r, dir_, sweep_deg, on_waypoints):
    """走圆弧 + 按时间触发回调. on_waypoints=[(time_s, callback), ...]"""
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


# ============== 圆环放置 (无机械臂) ==============
def place_at_ring(x, y, rotate_pos):
    """粗移到 (x,y) → ring 检测圆心 → 闭环 → 调转盘 → 放物"""
    go_to(x, y)
    corrected = False
    for attempt in range(VISION_MAX_CORRECTIONS + 1):
        measurement = ring.detect_offset()
        print(
            f"  RING pixel={measurement.pixel_offset_px} "
            f"body_mm={measurement.body_offset_mm} "
            f"scale={measurement.scale_mm_per_px}"
        )
        if measurement.aligned:
            if not corrected:
                # Already aligned: only synchronize odometry to the known ring.
                comm.send_sync_pose(x, y)
                if not comm.wait_for(comm.TYPE_CMD_SYNC_RESP, 5.0):
                    raise RuntimeError("ring pose sync failed")
            break
        if attempt == VISION_MAX_CORRECTIONS:
            raise RuntimeError("ring alignment did not converge")

        pose = comm.get_pose(max_age=1.0)
        if pose is None:
            raise RuntimeError("ring alignment requires a fresh STM32 pose")
        dx_mm, dy_mm = measurement.world_correction_mm(pose[2])
        if not comm.vision_correct(dx_mm, dy_mm, x, y, timeout=5.0):
            raise RuntimeError(
                f"ring correction failed: dx={dx_mm:.2f}mm dy={dy_mm:.2f}mm"
            )
        corrected = True
    rotate(rotate_pos)
    # TODO: 实际抓放动作 (无机械臂, 放物逻辑待加)


# ============== 主流程 ==============
def main():
    comm.init("/dev/ttyCH341USB0", 115200)
    time.sleep(1.0)  # 等 POSE 帧
    try:
        run_task()
    except RuntimeError as e:
        # 全局兜底: 视觉/串口识别失败时停下
        print(f"\n[兜底] 任务中断: {e}")
        print("[兜底] 车保留当前位置, 串口关闭")
    except KeyboardInterrupt:
        print("\n[用户中断]")
    finally:
        comm.shutdown()


def run_task():
    # ========== 阶段 A: 右半区装载 ==========
    print("\n=== 阶段 A: 右半区装载 ===")

    turn_to(0)
    go_to(0, 0.295)

    turn_to(90)
    go_to(0.63679, 0.29568)

    seq2 = qr2.recognize()                       # "CAB"
    plan = [ALPHA_TO_POS[c] for c in seq2]       # 例 [2, 0, 1]
    print(f"  QR2: {seq2} -> 方案 {plan}")

    go_to(0.63679, 0.12996)

    rotate(plan[0])

    # 走逆时针圆弧 (车 (0.63679, 0.12996), 圆心 (0.575, 1.055), r=0.949)
    start_a = (0.63679, 0.12996)
    end_a = (0.575, 2.024)
    center_a = (0.575, 1.055)
    r = 0.949

    sweep_a = arc_sweep(start_a, end_a, center_a, ccw=True)
    head_a = arc_heading(start_a, center_a, dir_=-1)
    turn_to(head_a)                             # 车头朝西偏南 (266.3°)

    arc_len_a = math.radians(sweep_a) * r
    arc_time_a = arc_len_a / (ARC_SPEED_MM_S / 1000)
    wp1_deg = user_deg((1.46721, 0.65758), center_a)
    wp2_deg = user_deg((1.564, 1.075), center_a)
    a_s_a = user_deg(start_a, center_a)
    t1 = arc_time_a * ((a_s_a - wp1_deg) % 360) / sweep_a
    t2 = arc_time_a * ((a_s_a - wp2_deg) % 360) / sweep_a

    arc_with_waypoints(r, -1, sweep_a, [
        (t1, lambda: (time.sleep(WAYPOINT_DELAY_S), rotate(plan[1]))),
        (t2, lambda: (time.sleep(WAYPOINT_DELAY_S), rotate(plan[2]))),
    ])

    # ========== 阶段 B: 右半区圆环放置 ==========
    print("\n=== 阶段 B: 右半区圆环放置 ===")
    turn_to(0)
    place_at_ring(0.27017, 1.77938, 1)
    place_at_ring(-0.27017, 1.77938, 2)
    place_at_ring(0, 1.77938, 0)
    go_to(0, 0.295)

    # ========== 阶段 C: 左半区装载 ==========
    print("\n=== 阶段 C: 左半区装载 ===")

    turn_to(-90)
    go_to(-0.66158, 0.29568)

    seq1 = qr1.recognize()                                # "12345"
    color_seq = [BLOCK_NUM_TO_COLOR[int(c)] for c in seq1]
    print(f"  QR1: {seq1} -> 颜色顺序 {color_seq}")

    go_to(-0.66158, 0.12996)

    rotate(0)

    # 走顺时针圆弧 (车 (-0.66158, 0.12996), 圆心 (-0.575, 1.055), r=0.949)
    start_c = (-0.66158, 0.12996)
    end_c = (-0.63203, 2.02231)
    center_c = (-0.575, 1.055)

    sweep_c = arc_sweep(start_c, end_c, center_c, ccw=False)
    head_c = arc_heading(start_c, center_c, dir_=1)
    turn_to(head_c)                              # 车头朝东偏北 (95.4°)

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
        # 单帧快速识别, 避免走弧过程中车移动导致识别失效
        color = block.recognize(frames=1, timeout=0.3)
        color_at_pos[idx] = color
        time.sleep(AFTER_RECOGNIZE_DELAY_S)
        rotate(idx + 1)

    arc_with_waypoints(r, 1, sweep_c, [
        (wp_times_c[0], lambda: on_wp_c(0)),
        (wp_times_c[1], lambda: on_wp_c(1)),
        (wp_times_c[2], lambda: on_wp_c(2)),
        (wp_times_c[3], lambda: on_wp_c(3)),
    ])

    # 起点位 0 的颜色 (走弧前, 车已 rotate(0))
    color_at_pos[0] = block.recognize(frames=1, timeout=0.3)
    color_to_pos = {c: i for i, c in enumerate(color_at_pos) if c}
    print(f"  颜色->位置: {color_to_pos}")

    # ========== 阶段 D: 左半区圆环放置 ==========
    print("\n=== 阶段 D: 左半区圆环放置 ===")
    turn_to(180)

    place_at_ring(-0.63203, 1.45514, color_to_pos[color_seq[0]])
    place_at_ring(0.22038, 1.45564, color_to_pos[color_seq[1]])
    go_to(0, 1.45564)
    place_at_ring(-0.42737, 0.71556, color_to_pos[color_seq[2]])
    place_at_ring(0.33466, 0.71464, color_to_pos[color_seq[3]])
    place_at_ring(0.83621, 0.71518, color_to_pos[color_seq[4]])

    go_to(0, 0)


if __name__ == "__main__":
    main()
