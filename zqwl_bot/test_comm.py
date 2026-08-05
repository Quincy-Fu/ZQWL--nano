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
    t <deg>            角度环 TURNTO (CW+)     例: t 90
    arc r dir sweep    圆弧 (半径m, dir: 1=右转/-1=左转, 扫过角度°, 圆心自动算)
                       例: arc 0.5 1 180  (右转半圆)
    s <x> <y>          重置坐标                例: s 0 0
    p                  打印当前位姿 (下位机50Hz上报)
    n <f|b|l|r|s>      视觉微调 (体坐标系, 慢速) 例: n f
                       f=前进 b=后退 l=左 r=右 s=停止+锁死
    all                综合测试 (全部子系统按安全顺序跑一遍, 最后回原点)
    h                  显示帮助
    q                  退出

坐标系: +X=右, +Y=前, 角度 CW 为正, 单位 m / deg. MCU 上电 odom 归零.
注意: 转盘响应是"估算移动时间到"而非电机真实到位; TOX/TOY 恒回成功(时序握手).
"""

import sys
import time

try:
    from . import comm          # 作为包内模块
except ImportError:
    import comm                 # 作为脚本直接运行


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

def do_turn(deg: float):
    return timed(f"TURNTO {deg:.1f}°", lambda: comm.turnto(deg, 30.0))

def do_arc(r: float, dir_: int, sweep: float):
    dname = "右转" if dir_ > 0 else "左转"
    return timed(f"ARC r={r:.3f} {dname} {abs(sweep):.1f}° (车头沿切线)",
                 lambda: comm.arc(r, dir_, sweep))

def do_sync(x: float, y: float):
    print(f"  >> SYNC 坐标重置为 ({x:.3f}, {y:.3f})")
    t0 = time.perf_counter()
    comm.send_sync_pose(x, y)
    ok = comm.wait_for(comm.TYPE_CMD_SYNC_RESP, 5.0)
    dt = time.perf_counter() - t0
    print(f"  << {'OK  ' if ok else 'FAIL'}  ({dt:.2f}s)")
    return bool(ok)


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
    print(f"  坐标: x={x:.4f}m  y={y:.4f}m  yaw={yaw_deg:.1f}° (CCW+)  "
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
    t <deg>            角度环 TURNTO (CW+)
    arc r dir sweep    圆弧 (dir: 1=右转 -1=左转, 扫过角度°)  例: arc 0.5 1 180
    s <x> <y>          重置坐标
    p                  打印当前位姿
    n <f|b|l|r|s>      视觉微调 (f=前 b=后 l=左 r=右 s=停止+锁死)
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
                elif cmd == "s":
                    if len(parts) < 3:
                        print("用法: s <x> <y>")
                        continue
                    do_sync(float(parts[1]), float(parts[2]))
                elif cmd == "p":
                    do_pose()
                elif cmd == "n":
                    if len(parts) < 2:
                        print("用法: n <f|b|l|r|s>  (f=前进 b=后退 l=左 r=右 s=停止+锁死)")
                        continue
                    do_nudge(parts[1])
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
