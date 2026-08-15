"""`
按 25 步执行: 右半区装载 → 圆环放置 → 左半区装载 → 圆环放置 → 回 HOME
"""
import math
import time
import threading
# 项目模块
import serial as stm32   # 串口 (你贴的 zqwl_bot/serial.py)
import qr1               # 左半区二维码 (返回 "12345")
import qr2               # 右半区二维码 (返回 "CAB")
import ring              # 圆环检测
import block             # 物块识别
# ============== 配置 ==============
ARC_SPEED_MM_S = 200          # 圆弧速度 (mm/s), 实测标定
WAYPOINT_DELAY_S = 1.0        # 过点后延时
ALPHA_TO_POS = {"A": 0, "B": 1, "C": 2}  # 字母固定映射
# TODO: 编号 1-5 对应的颜色 (按规则填)
# 或者改成让 qr1.recognize() 直接返回 "黑白红绿蓝" 这种
BLOCK_NUM_TO_COLOR = {1: "黑", 2: "白", 3: "红", 4: "绿", 5: "蓝"}
# ============== 角度工具 ==============
def user_deg(p, c):
    """点 P 相对圆心 C 的用户角度 (0=北, 顺时针正)"""
    dx, dy = p[0] - c[0], p[1] - c[1]
    a = math.degrees(math.atan2(dy, dx))   # atan2, -180~180
    return (90 - a) % 360
def arc_time(a_from, a_to, ccw, radius, wp_deg, speed):
    """从 a_from 沿指定方向到 wp_deg 的时间 (秒)"""
    if ccw:
        d_total = (a_to - a_from) % 360
        d_wp = (wp_deg - a_from) % 360
    else:
        d_total = (a_from - a_to) % 360
        d_wp = (a_from - wp_deg) % 360
    if d_wp > d_total:
        raise ValueError(f"过点 {wp_deg} 不在弧段内")
    return math.radians(d_wp) * radius / speed
# ============== 串口高层 ==============
def _ok(resp, what):
    if resp is None or resp[1] != 1:
        raise RuntimeError(f"{what} failed: {resp}")
def go_to(x, y):
    stm32.send_goto(x, y)
    _ok(stm32.wait_nav_response(), f"goto ({x}, {y})")
def turn_to(deg):
    stm32.send_turnto(deg)
    _ok(stm32.wait_nav_response(), f"turnto {deg}")
def fine_move(dx, dy):
    stm32.send_fine_move(dx, dy)
    _ok(stm32.wait_nav_response(), f"fine_move ({dx}, {dy})")
def sync_pose(x, y):
    stm32.send_sync_pose(x, y)
    _ok(stm32.wait_nav_response(), f"sync_pose ({x}, {y})")
def rotate(pos):
    stm32.send_rotate(pos)
def arm(state):
    stm32.send_arm(state)
def run_arc(cx, cy, r, a_start, a_end, ccw, on_waypoints):
    """走圆弧, on_waypoints=[(wp_deg, callback), ...]
    callback 在过点时间触发 (主线程同时 wait_nav_response)
    """
    # 预算每个过点时间
    items = [(arc_time(a_start, a_end, ccw, r, wp, ARC_SPEED_MM_S), cb)
             for wp, cb in on_waypoints]
    def timer():
        t0 = time.time()
        for t, cb in items:
            wait = t - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
            cb()
    th = threading.Thread(target=timer, daemon=True)
    th.start()
    stm32.send_arc(cx, cy, r, a_start, a_end)
    _ok(stm32.wait_nav_response(), f"arc ({cx}, {cy}, {r}, {a_start}, {a_end})")
    th.join()
# ============== 圆环放置 ==============
def place_at_ring(x, y, rotate_pos):
    """粗移到 (x,y) → 调 ring 检测圆心 → 闭环 → 调转盘 → 放物"""
    go_to(x, y)
    centers = ring.detect_centers()  # TODO: 确认接口签名
    cx, cy = min(centers, key=lambda c: (c[0]-x)**2 + (c[1]-y)**2)
    fine_move(cx - x, cy - y)
    rotate(rotate_pos)
    # TODO: 实际抓放动作 (机械臂放物)
# ============== 主流程 ==============
def main():
    stm32.init(port="/dev/ttyCH341USB0", baudrate=115200)
    try:
        run_task()
    finally:
        stm32.shutdown()
def run_task():
    # ========== 阶段 A: 右半区装载 ==========
    # 1. HOME -> (0, 295), 朝 0°
    turn_to(0)
    go_to(0, 295)
    # 2. 朝 90° -> (636.79, 295.68)
    turn_to(90)
    go_to(636.79, 295.68)
    # 3-4. QR2 识别, 字母->位置方案
    seq2 = qr2.recognize()                       # "CAB"
    plan = [ALPHA_TO_POS[c] for c in seq2]       # 例: [2, 0, 1]
    # 5. 平移
    go_to(636.79, 129.96)
    # 6. 准备 (走弧前)
    rotate(plan[0])
    arm(1)
    # 7. 走逆时针圆弧 + 2 个过点切转盘
    C = (575, 1055)
    a_s = user_deg((636.79, 129.96), C)
    a_e = user_deg((575, 2024), C)
    wp1 = user_deg((1467.21, 657.58), C)
    wp2 = user_deg((1564, 1075), C)
    run_arc(C[0], C[1], 949, a_s, a_e, ccw=True, on_waypoints=[
        (wp1, lambda: (time.sleep(WAYPOINT_DELAY_S), rotate(plan[1]))),
        (wp2, lambda: (time.sleep(WAYPOINT_DELAY_S), rotate(plan[2]))),
    ])
    # ========== 阶段 B: 右半区圆环放置 ==========
    turn_to(0)
    # 10-12. 3 个圆环放置
    arm(3)
    place_at_ring(270.17, 1779.38, 1)
    arm(2)
    place_at_ring(-270.17, 1779.38, 2)
    arm(4)
    place_at_ring(0, 1779.38, 0)
    # 13. 回到 (0, 295)
    go_to(0, 295)
    # ========== 阶段 C: 左半区装载 ==========
    # 14. 朝 -90° -> (-661.58, 295.68)
    turn_to(-90)
    go_to(-661.58, 295.68)
    # 15. QR1 识别
    seq1 = qr1.recognize()                                # "12345"
    color_seq = [BLOCK_NUM_TO_COLOR[int(c)] for c in seq1]  # ["黑","白","红","绿","蓝"]
    # 16. 平移
    go_to(-661.58, 129.96)
    # 17. 准备 (走弧前)
    rotate(0)
    arm(5)
    # 18-19. 走顺时针圆弧 + 4 个过点 block 识别
    color_at_pos = [None] * 5   # pos -> 颜色
    def on_wp(idx):
        color = block.recognize()  # TODO: 确认接口
        color_at_pos[idx] = color
        time.sleep(WAYPOINT_DELAY_S)
        rotate(idx + 1)
    C2 = (-575, 1055)
    a_s = user_deg((-661.58, 129.96), C2)
    a_e = user_deg((-632.03, 2022.31), C2)
    wps = [
        user_deg((-1058.24, 258.25), C2),
        user_deg((-1399.38, 604.89), C2),
        user_deg((-1524, 1075), C2),
        user_deg((-1399.38, 1545.11), C2),
    ]
    run_arc(C2[0], C2[1], 949, a_s, a_e, ccw=False, on_waypoints=[
        (wps[0], lambda: on_wp(0)),
        (wps[1], lambda: on_wp(1)),
        (wps[2], lambda: on_wp(2)),
        (wps[3], lambda: on_wp(3)),
    ])
    # 起点位 0 的颜色 (走弧开始时已设 rotate(0), 实际需在走弧前检测)
    color_at_pos[0] = block.recognize()  # TODO: 也可能要在过点 0 前检测
    color_to_pos = {c: i for i, c in enumerate(color_at_pos) if c}
    # ========== 阶段 D: 左半区圆环放置 ==========
    turn_to(180)
    arm(6)
    # 20-24. 5 个圆环放置
    place_at_ring(-632.03, 1455.14, color_to_pos[color_seq[0]])
    place_at_ring(220.38, 1455.64, color_to_pos[color_seq[1]])
    go_to(0, 1455.64)
    place_at_ring(-427.37, 715.56, color_to_pos[color_seq[2]])
    place_at_ring(334.66, 714.64, color_to_pos[color_seq[3]])
    place_at_ring(836.21, 715.18, color_to_pos[color_seq[4]])
    # 25. 回 HOME
    go_to(0, 0)
    arm(0)
if __name__ == "__main__":
    main()

'''
需要你确认/填的几个 TODO：
1.
qr1.recognize() / qr2.recognize() 接口：返回字符串就行对吧？比如 "12345" 和 "CAB"。
2.
ring.detect_centers()：返回 5 个圆心坐标 list[(x, y)]（mm 单位）？还是要传当前坐标作为参考？
3.
block.recognize()：返回当前看到的物块颜色字符串 "黑/白/红/绿/蓝"？需要传摄像头帧吗？
4.
ARC_SPEED_MM_S = 200：先默认 200 mm/s，你后面按实测调
5.
BLOCK_NUM_TO_COLOR 表：编号 1-5 对应的颜色对吗？规则文档里有没有这表？

几个我自己拿不准的地方：
机械臂抓放动作：place_at_ring 里我留了 TODO，因为你只说"对准圆环中心后放置"，具体怎么下放物块的指令没给（是不是 arm(state) 切到某个状态 = 放下？）——你确认下机械臂 0-7 各自代表什么动作
起点位 0 的颜色：圆弧走完后 color_at_pos[0] 是空的，我用 block.recognize() 补了一次，但可能要在走弧前就检测好（这逻辑可能不对）
过点时机：用 ARC_SPEED_MM_S 估算时间，假设 STM32 是匀速走的。如果不是匀速（比如加减速），时间会偏。我先按匀速来，你测试时再校准

没写到的地方（按你说"不要加东西"，我没加，但你可能需要）：
错误重试
紧急停止
串口断开重连
调试日志
进度显示
'''