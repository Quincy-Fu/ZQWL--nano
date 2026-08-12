"""上位机综合测试脚本 (交互式, Jetson Nano 上位机, 经 USB-TTL 连 STM32).

启动一次, 然后在提示符下输入命令即可, 类似 Windows 时的 nav_test.py.

用法:
    python3 test_comm.py                    # 默认 TTL: /dev/ttyCH341USB0
    python3 test_comm.py /dev/ttyUSB0       # 换串口

命令:
    r <pos>            转盘槽位 0-4           例: r 2
    r all              转盘依次走完 5 个槽位
    a <state>          机械臂姿态 0-7 (0=默认) 例: a 1
    l <id> on|off      补光灯 (0=全部, 1-4)    例: l 1 on
    run                启动按键 (PD15 500ms 脉冲)
    x <val>            锁轴: X 移到 val m      例: x 0.3
    y <val>            锁轴: Y 移到 val m      例: y 0.3
    g <x> <y>          位置环 GOTO (m)         例: g 0.3 0.3
    fd <dist> [yaw]    沿当前/指定朝向前进 dist m 例: fd 0.3 或 fd 0.3 -60
    t <deg>            角度环 TURNTO (CW+)     例: t 90
    arc r dir sweep    圆弧 (半径m, dir: 1=右转/-1=左转, 扫过角度°, 圆心自动算)
                       例: arc 0.5 1 180  (右转半圆)
    kp [nosync]         固定关键点连续路径测试, 默认先同步到起点
    s <x> <y> [yaw]    重置坐标/可选朝向        例: s 0 0 或 s 0 0 -90
    sy <yaw>           保持当前坐标, 重置朝向    例: sy -90
    p                  打印当前位姿 (下位机50Hz上报)
    n <f|b|l|r|s>      视觉微调 (体坐标系, 慢速) 例: n f
                       f=前进 b=后退 l=左 r=右 s=停止+锁死
    fm <dx> <dy>       诊断: 发送 FINE_MOVE 偏移(mm) 例: fm 30 0
    vc <dx> <dy> [x y] 诊断: 发送 VISION_CORRECT 偏移(mm), 可选同步坐标(m)
                       例: vc 30 0        或 vc 30 0 0.5 0.3
    all                综合测试 (全部子系统按安全顺序跑一遍, 最后回原点)
    h                  显示帮助
    q                  退出

坐标系: +X=右, +Y=前, 角度 CW 为正, 单位 m / deg. MCU 上电 odom 归零.
注意: 转盘响应是"估算移动时间到"而非电机真实到位; TOX/TOY 恒回成功(时序握手).
"""

import math
import sys
import time

try:
    from . import comm          # 作为包内模块
except ImportError:
    import comm                 # 作为脚本直接运行


KEY_PATH_SPEED = 0.60
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


def timed(label: str, fn) -> bool:
    """执行一条阻塞命令, 打印耗时与结果."""
    print(f"  >> {label}")
    t0 = time.perf_counter()
    try:
        ok = bool(fn())
    except ValueError as e:
        print(f"  !! 参数错误: {e}")
        return False
    dt = time.perf_counter() - t0
    print(f"  << {'OK  ' if ok else 'FAIL'}  ({dt:.2f}s)")
    return ok


# ── 各命令 ──

def do_run():
    return timed("RUN 启动按键", lambda: comm.run(5.0))

def do_light(lid: int, on: bool):
    name = f"灯{lid}" if lid else "全部灯"
    return timed(f"LIGHT {name} {'开' if on else '关'}",
                 lambda: comm.light(lid, on, 5.0))

def do_arm(state: int):
    return timed(f"ARM 姿态 {state} (五次多项式缓动)", lambda: comm.arm(state, 6.0))

def do_rotate(pos) -> bool:
    if pos == "all":
        ok = True
        for p in range(5):
            ok &= timed(f"ROTATE 槽位 {p}", lambda p=p: comm.rotate(p, 12.0))
        return ok
    return timed(f"ROTATE 槽位 {pos}", lambda: comm.rotate(pos, 12.0))

def do_axis(axis: str, target: float):
    name = "X (锁Y)" if axis == 'x' else "Y (锁X)"
    return timed(f"锁轴 {name} → {target:.3f}m",
                 lambda: comm.lock_axis(axis, target, 40.0))

def do_goto(x: float, y: float):
    return timed(f"GOTO ({x:.3f}, {y:.3f})", lambda: comm.goto(x, y, timeout=40.0))


def do_forward(dist_m: float, yaw_deg: float | None = None):
    """按当前或指定朝向前进一段距离, 自动换算成 GOTO 目标坐标。"""
    pose = comm.get_pose(max_age=1.0)
    if pose is None:
        print("  !! 没有新鲜 POSE，不能计算起点；请先用 p 确认位姿上报")
        return False
    x, y, pose_yaw = pose
    yaw = pose_yaw if yaw_deg is None else yaw_deg
    rad = math.radians(yaw)
    tx = x + dist_m * math.sin(rad)
    ty = y + dist_m * math.cos(rad)
    print(f"  .. 起点=({x:.4f}, {y:.4f}) yaw={yaw:.1f}° dist={dist_m:.3f}m")
    print(f"  .. 目标=({tx:.4f}, {ty:.4f})")
    return do_goto(tx, ty)

def do_turn(deg: float):
    return timed(f"TURNTO {deg:.1f}°", lambda: comm.turnto(deg, 30.0))

def do_arc(r: float, dir_: int, sweep: float):
    dname = "右转" if dir_ > 0 else "左转"
    return timed(f"ARC r={r:.3f} {dname} {abs(sweep):.1f}° (车头沿切线)",
                 lambda: comm.arc(r, dir_, sweep))


def do_key_path(sync_start: bool = True):
    """固定关键点连续路径测试: 重复坐标点表示先转角, 再走下一段。"""
    start = KEY_PATH_POINTS[0]
    print("  .. 固定关键点路径:")
    for idx, (x, y, yaw) in enumerate(KEY_PATH_POINTS):
        print(f"     p{idx}: x={x:.3f} y={y:.3f} yaw={yaw:.1f}°")

    if sync_start:
        print("  .. 默认先同步下位机里程计到 p0；实车也应放在该起点附近。")
        if not do_sync(start[0], start[1], start[2]):
            return False
    else:
        print("  .. nosync: 不改里程计, 直接按当前下位机位姿执行。")

    return timed(
        f"KEY_PATH {len(KEY_PATH_POINTS)}点 v={KEY_PATH_SPEED:.2f}m/s",
        lambda: comm.key_path(KEY_PATH_POINTS, speed=KEY_PATH_SPEED, timeout=120.0),
    )

def do_sync(x: float, y: float, yaw_deg: float | None = None):
    if yaw_deg is None:
        print(f"  >> SYNC 坐标重置为 ({x:.3f}, {y:.3f})")
    else:
        print(f"  >> SYNC 位姿重置为 ({x:.3f}, {y:.3f}, yaw={yaw_deg:.1f}°)")
    t0 = time.perf_counter()
    comm.send_sync_pose(x, y, yaw_deg)
    ok = comm.wait_for(comm.TYPE_CMD_SYNC_RESP, 5.0)
    dt = time.perf_counter() - t0
    print(f"  << {'OK  ' if ok else 'FAIL'}  ({dt:.2f}s)")
    return bool(ok)


def do_sync_yaw(yaw_deg: float):
    """保持当前 x/y 不变, 只重置下位机当前朝向。"""
    pose = comm.get_pose(max_age=1.0)
    if pose is None:
        print("  !! 没有新鲜 POSE，不能安全保持当前坐标；请先确认通信，或用 s <x> <y> <yaw>")
        return False
    x, y, _ = pose
    return do_sync(x, y, yaw_deg)


def do_pose() -> bool:
    """打印下位机当前位姿 (50Hz 常驻上报, 直接读缓存, 不发请求)."""
    pose = comm.get_pose(max_age=1.0)
    if pose is None:
        age = comm.pose_age()
        if age == float('inf'):
            print("  !! 未收到 POSE 帧: 确认 MCU 已上电且串口连通")
        else:
            print(f"  !! POSE 帧已 {age:.1f}s 未更新, 通信可能中断")
        return False
    x, y, yaw_deg = pose
    age = comm.pose_age()
    print(f"  坐标: x={x:.4f}m  y={y:.4f}m  yaw={yaw_deg:.1f}° (CW+)  "
          f"(帧龄 {age*1000:.0f}ms)")
    return True


_NUDGE_MAP = {
    "f": (comm.NUDGE_FORWARD,  "前进"),
    "b": (comm.NUDGE_BACKWARD, "后退"),
    "l": (comm.NUDGE_LEFT,     "左移"),
    "r": (comm.NUDGE_RIGHT,    "右移"),
    "s": (comm.NUDGE_STOP,     "停止+锁死"),
}

def do_nudge(sub: str):
    if sub not in _NUDGE_MAP:
        print("用法: n <f|b|l|r|s>  (f=前进 b=后退 l=左 r=右 s=停止+锁死)")
        return
    code, name = _NUDGE_MAP[sub]
    return timed(f"NUDGE {name}", lambda: comm.vision_nudge(code, 3.0))


def do_fine_move(dx_mm: float, dy_mm: float):
    """诊断 0x18: 只做物理偏移，不同步坐标。"""
    def run():
        comm.send_fine_move(dx_mm, dy_mm)
        return comm.wait_for(comm.TYPE_CMD_FINE_RESP, 5.0)
    return timed(f"FINE_MOVE dx={dx_mm:.1f}mm dy={dy_mm:.1f}mm", run)


def do_vision_correct(dx_mm: float, dy_mm: float,
                      target_x: float | None = None,
                      target_y: float | None = None):
    """诊断 0x2B: 物理偏移成功后同步坐标。"""
    if target_x is None or target_y is None:
        pose = comm.get_pose(max_age=1.0)
        if pose is None:
            print("  !! 没有新鲜 POSE，vc 需要显式给 x y，例如: vc 30 0 0 0")
            return False
        target_x, target_y = pose[0], pose[1]
        print(f"  .. 未指定同步坐标，使用当前 POSE: x={target_x:.4f}m y={target_y:.4f}m")
    return timed(
        f"VISION_CORRECT dx={dx_mm:.1f}mm dy={dy_mm:.1f}mm sync=({target_x:.3f},{target_y:.3f})",
        lambda: comm.vision_correct(dx_mm, dy_mm, target_x, target_y, timeout=5.0),
    )


def do_all() -> bool:
    steps = [
        ("第1步: 坐标重置",            lambda: do_sync(0.0, 0.0)),
        ("第2步: RUN",                 do_run),
        ("第3步: 灯1 开",              lambda: do_light(1, True)),
        ("第4步: 灯1 关",              lambda: do_light(1, False)),
        ("第5步: 机械臂姿态1",          lambda: do_arm(1)),
        ("第6步: 机械臂回默认位",       lambda: do_arm(0)),
        ("第7步: 转盘槽位1",           lambda: do_rotate(1)),
        ("第8步: 转盘回槽位0",          lambda: do_rotate(0)),
        ("第9步: 锁轴 X→0.3",          lambda: do_axis('x', 0.3)),
        ("第10步: 锁轴 Y→0.3",         lambda: do_axis('y', 0.3)),
        ("第11步: TURNTO 90°",         lambda: do_turn(90.0)),
        ("第12步: TURNTO 0°",          lambda: do_turn(0.0)),
        ("第13步: GOTO 回原点",        lambda: do_goto(0.0, 0.0)),
        ("第14步: 1/4圆弧 (右转 r=0.3, 90° → (0.3,0.3))",
                                       lambda: do_arc(0.3, 1, 90.0)),
        ("第15步: GOTO 回原点",        lambda: do_goto(0.0, 0.0)),
    ]
    n_ok = 0
    for label, fn in steps:
        print(f"── {label} ──")
        if fn():
            n_ok += 1
        else:
            print("  !! 本步失败, 继续后续步骤 (Ctrl+C 可中断)")
    print(f"\n综合测试结果: {n_ok}/{len(steps)} 通过")
    return n_ok == len(steps)


HELP = """命令:
    r <pos>            转盘槽位 0-4           例: r 2
    r all              转盘依次走完 5 个槽位
    a <state>          机械臂姿态 0-7 (0=默认) 例: a 1
    l <id> on|off      补光灯 (0=全部, 1-4)    例: l 1 on
    run                启动按键 (PD15 脉冲)
    x <val>            锁轴 X 移到 val m
    y <val>            锁轴 Y 移到 val m
    g <x> <y>          位置环 GOTO
    fd <dist> [yaw]    沿当前/指定朝向前进 dist m, 例: fd 0.3 或 fd 0.3 -60
    t <deg>            角度环 TURNTO (CW+)
    arc r dir sweep    圆弧 (dir: 1=右转 -1=左转, 扫过角度°)  例: arc 0.5 1 180
    kp [nosync]         固定关键点连续路径测试, 默认先同步到起点
    s <x> <y> [yaw]    重置坐标/可选朝向
    sy <yaw>           保持当前坐标, 重置朝向
    p                  打印当前位姿
    n <f|b|l|r|s>      视觉微调 (f=前 b=后 l=左 r=右 s=停止+锁死)
    fm <dx> <dy>       诊断 FINE_MOVE 偏移(mm), 例: fm 30 0
    vc <dx> <dy> [x y] 诊断 VISION_CORRECT 偏移(mm), 可选同步坐标(m)
    all                综合测试
    q                  退出"""


# ── 主循环 ──

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyCH341USB0"

    print(f"连接 {port} @ 115200 ...")
    try:
        comm.init(port, 115200)
    except Exception as e:
        print(f"串口打开失败: {e}")
        sys.exit(1)
    time.sleep(1.0)   # 等 POSE 帧建立通信
    print("已连接\n")
    print("═══  综合测试 (输入 h 查看命令)  ═══\n")

    try:
        while True:
            try:
                line = input("> ").strip().lower()
            except EOFError:
                break
            if not line:
                continue

            parts = line.split()
            if len(parts) == 1 and len(parts[0]) >= 2 and parts[0][0] == "r":
                parts = ["r", parts[0][1:]]
            cmd = parts[0]

            try:
                if cmd == "q":
                    break
                elif cmd == "h":
                    print(HELP)
                elif cmd == "run":
                    do_run()
                elif cmd == "r":
                    if len(parts) < 2:
                        print("用法: r <0-4> 或 r all")
                        continue
                    pos = parts[1]
                    do_rotate("all" if pos == "all" else int(pos))
                elif cmd == "a":
                    if len(parts) < 2:
                        print("用法: a <0-7>")
                        continue
                    do_arm(int(parts[1]))
                elif cmd == "l":
                    if len(parts) < 3:
                        print("用法: l <0-4> on|off")
                        continue
                    do_light(int(parts[1]), parts[2] in ("on", "1", "true"))
                elif cmd == "x":
                    if len(parts) < 2:
                        print("用法: x <米>")
                        continue
                    do_axis('x', float(parts[1]))
                elif cmd == "y":
                    if len(parts) < 2:
                        print("用法: y <米>")
                        continue
                    do_axis('y', float(parts[1]))
                elif cmd == "g":
                    if len(parts) < 3:
                        print("用法: g <x> <y>")
                        continue
                    do_goto(float(parts[1]), float(parts[2]))
                elif cmd in ("fd", "forward"):
                    if len(parts) not in (2, 3):
                        print("用法: fd <距离m> [yaw角度]  例: fd 0.3 或 fd 0.3 -60")
                        continue
                    if len(parts) == 3:
                        do_forward(float(parts[1]), float(parts[2]))
                    else:
                        do_forward(float(parts[1]))
                elif cmd == "t":
                    if len(parts) < 2:
                        print("用法: t <角度>")
                        continue
                    do_turn(float(parts[1]))
                elif cmd == "arc":
                    if len(parts) < 4:
                        print("用法: arc <半径m> <方向: 1右转/-1左转> <扫过角度°>  例: arc 0.5 1 180")
                        continue
                    do_arc(float(parts[1]), int(float(parts[2])), float(parts[3]))
                elif cmd in ("kp", "keypath"):
                    if len(parts) > 2 or (len(parts) == 2 and parts[1] != "nosync"):
                        print("用法: kp [nosync]")
                        continue
                    do_key_path(sync_start=(len(parts) == 1))
                elif cmd == "s":
                    if len(parts) not in (3, 4):
                        print("用法: s <x> <y> [yaw]  例: s 0 0 或 s 0 0 -90")
                        continue
                    if len(parts) == 4:
                        do_sync(float(parts[1]), float(parts[2]), float(parts[3]))
                    else:
                        do_sync(float(parts[1]), float(parts[2]))
                elif cmd == "sy":
                    if len(parts) != 2:
                        print("用法: sy <yaw>  例: sy -90")
                        continue
                    do_sync_yaw(float(parts[1]))
                elif cmd == "p":
                    do_pose()
                elif cmd == "n":
                    if len(parts) < 2:
                        print("用法: n <f|b|l|r|s>  (f=前进 b=后退 l=左 r=右 s=停止+锁死)")
                        continue
                    do_nudge(parts[1])
                elif cmd == "fm":
                    if len(parts) < 3:
                        print("用法: fm <dx_mm> <dy_mm>  例: fm 30 0")
                        continue
                    do_fine_move(float(parts[1]), float(parts[2]))
                elif cmd == "vc":
                    if len(parts) not in (3, 5):
                        print("用法: vc <dx_mm> <dy_mm> [target_x_m target_y_m]  例: vc 30 0 或 vc 30 0 0.5 0.3")
                        continue
                    if len(parts) == 5:
                        do_vision_correct(float(parts[1]), float(parts[2]),
                                          float(parts[3]), float(parts[4]))
                    else:
                        do_vision_correct(float(parts[1]), float(parts[2]))
                elif cmd == "all":
                    do_all()
                else:
                    print(f"未知命令: {cmd} (输入 h 查看帮助)")
            except ValueError:
                print("参数格式错误, 请输入数字")

    except KeyboardInterrupt:
        pass

    comm.shutdown()
    print("\n串口已关闭")


if __name__ == "__main__":
    main()
