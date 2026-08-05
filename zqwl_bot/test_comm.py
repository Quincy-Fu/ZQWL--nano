"""上位机综合测试脚本 (Jetson Nano 上位机, 经 USB-TTL 连 STM32).

基于 comm.py 的高层阻塞 API, 逐条命令发送并等待下位机真实响应,
覆盖: 启动按键 / 补光灯 / 机械臂 / 转盘 / 位置环 / 角度环 / 圆弧 / 坐标重置.

TTL 设备名固定为 /dev/ttyCH341USB0 (与 comm.py 默认一致), 无需传串口:

    # 单项测试 (在 zqwl_bot 目录下运行)
    python3 test_comm.py run                   # 启动按键 (PD15 脉冲)
    python3 test_comm.py light 1 on            # 灯1 开 (id: 0=全部, 1-4)
    python3 test_comm.py light 0 off           # 全部关
    python3 test_comm.py arm 1                 # 机械臂姿态1 (0-7, 0=默认)
    python3 test_comm.py rotate 2              # 转盘切到槽位2 (0-4)
    python3 test_comm.py rotate all            # 依次走完 5 个槽位
    python3 test_comm.py axis x 0.3            # 锁轴: X 移到 0.3m
    python3 test_comm.py pos 0.3 0.3           # 位置环: GOTO (0.3, 0.3)
    python3 test_comm.py turn 90               # 角度环: 转到 90° (CW+)
    python3 test_comm.py arc 0 0.3 0.3 -90 0   # 圆弧: 圆心/半径/起止角(度)
    python3 test_comm.py sync 0 0              # 重置坐标为 (0, 0)

    # 综合测试: 按安全顺序跑一遍全部子系统, 最后回原点
    python3 test_comm.py all

    # 可选参数 (换串口/超时/波特率)
    python3 test_comm.py --port /dev/ttyUSB0 turn 90   # 换串口
    python3 test_comm.py pos 0.5 0.5 --timeout 60 --baud 115200

坐标系: +X=右, +Y=前, 角度 CW 为正, 单位 m / deg. MCU 上电 odom 归零.
注意: 转盘响应是"估算移动时间到"而非电机真实到位; TOX/TOY 恒回成功(时序握手).
"""

import argparse
import sys
import time

try:
    from . import comm          # 作为包内模块
except ImportError:
    import comm                 # 作为脚本直接运行


# ── 工具 ──

def _timed(label: str, fn, timeout: float) -> bool:
    """执行一条阻塞命令, 打印耗时与结果."""
    print(f"  >> {label}")
    t0 = time.perf_counter()
    ok = fn()
    dt = time.perf_counter() - t0
    print(f"  << {'OK  ' if ok else 'FAIL'}  {label}   ({dt:.2f}s)")
    return bool(ok)


class Tally:
    def __init__(self):
        self.ok = 0
        self.fail = 0

    def add(self, ok: bool):
        if ok:
            self.ok += 1
        else:
            self.fail += 1

    def report(self):
        total = self.ok + self.fail
        print("\n" + "═" * 40)
        print(f"结果: {self.ok}/{total} 通过" + (f", {self.fail} 失败" if self.fail else ""))
        print("═" * 40)
        return self.fail == 0


# ── 单项测试 ──

def do_run(t: Tally, timeout: float):
    t.add(_timed("RUN 启动按键 (PD15 500ms 脉冲)", lambda: comm.run(timeout), timeout))


def do_light(t: Tally, light_id: int, on: bool, timeout: float):
    state = "开" if on else "关"
    name = f"灯{light_id}" if light_id else "全部灯"
    t.add(_timed(f"LIGHT {name} {state}", lambda: comm.light(light_id, on, timeout), timeout))


def do_arm(t: Tally, state: int, timeout: float):
    t.add(_timed(f"ARM 姿态 {state} (五次多项式缓动)", lambda: comm.arm(state, timeout), timeout))


def do_rotate(t: Tally, pos, timeout: float):
    if pos == "all":
        for p in range(5):
            t.add(_timed(f"ROTATE 槽位 {p}", lambda p=p: comm.rotate(p, timeout), timeout))
    else:
        t.add(_timed(f"ROTATE 槽位 {pos}", lambda: comm.rotate(pos, timeout), timeout))


def do_axis(t: Tally, axis: str, target: float, timeout: float):
    name = "X (锁Y)" if axis == 'x' else "Y (锁X)"
    t.add(_timed(f"锁轴 {name} → {target:.3f}m",
                 lambda: comm.lock_axis(axis, target, timeout), timeout))


def do_pos(t: Tally, x: float, y: float, timeout: float):
    t.add(_timed(f"GOTO ({x:.3f}, {y:.3f})", lambda: comm.goto(x, y, timeout=timeout), timeout))


def do_turn(t: Tally, deg: float, timeout: float):
    t.add(_timed(f"TURNTO {deg:.1f}°", lambda: comm.turnto(deg, timeout), timeout))


def do_arc(t: Tally, cx: float, cy: float, r: float,
           a0: float, a1: float, timeout: float):
    t.add(_timed(f"ARC 圆心({cx:.3f},{cy:.3f}) r={r:.3f} {a0:.1f}°→{a1:.1f}°",
                 lambda: comm.arc(cx, cy, r, a0, a1, timeout), timeout))


def do_sync(t: Tally, x: float, y: float, timeout: float = 5.0):
    # sync 没有高层封装, 直接用底层 send + wait_for
    print(f"  >> SYNC 坐标重置为 ({x:.3f}, {y:.3f})")
    t0 = time.perf_counter()
    comm.send_sync_pose(x, y)
    ok = comm.wait_for(comm.TYPE_CMD_SYNC_RESP, timeout)
    dt = time.perf_counter() - t0
    print(f"  << {'OK  ' if ok else 'FAIL'}  SYNC   ({dt:.2f}s)")
    t.add(bool(ok))


# ── 综合测试 ──

def do_all(t: Tally, timeout: float):
    print("── 第1步: 坐标重置 ──")
    do_sync(t, 0.0, 0.0)

    print("── 第2步: 原地设备 (启动/灯/机械臂/转盘) ──")
    do_run(t, 5.0)
    do_light(t, 1, True, 5.0)
    do_light(t, 1, False, 5.0)
    do_arm(t, 1, 6.0)
    do_arm(t, 0, 6.0)            # 回默认位
    do_rotate(t, 1, 12.0)
    do_rotate(t, 0, 12.0)        # 回槽位0

    print("── 第3步: 位置环 (锁轴) ──")
    do_axis(t, 'x', 0.3, timeout)
    do_axis(t, 'y', 0.3, timeout)

    print("── 第4步: 角度环 ──")
    do_turn(t, 90.0, 30.0)
    do_turn(t, 0.0, 30.0)

    print("── 第5步: 位置环 (走点回原点) ──")
    do_pos(t, 0.0, 0.0, timeout)

    print("── 第6步: 圆弧 (从原点出发, 1/4 弧到 (0.3, 0.3)) ──")
    do_arc(t, 0.0, 0.3, 0.3, -90.0, 0.0, 120.0)
    do_pos(t, 0.0, 0.0, timeout)   # 收尾回原点


# ── 主入口 ──

def main():
    ap = argparse.ArgumentParser(
        description="ZQWL 上位机综合测试 (基于 comm.py 阻塞 API)")
    ap.add_argument("cmd", choices=[
        "run", "light", "arm", "rotate", "axis", "pos",
        "turn", "arc", "sync", "all"])
    ap.add_argument("args", nargs="*", help="命令参数 (见文件头用法)")
    ap.add_argument("--port", default="/dev/ttyCH341USB0",
                    help="TTL 串口设备名 (默认 /dev/ttyCH341USB0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=40.0,
                    help="单条命令超时 s (默认 40)")
    ns = ap.parse_args()

    print(f"连接 {ns.port} @ {ns.baud} ...")
    try:
        comm.init(ns.port, ns.baud)
    except Exception as e:
        print(f"串口打开失败: {e}")
        sys.exit(1)

    # 等 POSE 帧建立通信 (最多 2s, 仅观察, 不影响后续)
    time.sleep(1.0)
    print("已连接\n")

    t = Tally()
    try:
        a = ns.args
        if ns.cmd == "run":
            do_run(t, 5.0)
        elif ns.cmd == "light":
            lid = int(a[0]) if a else 0
            on = (a[1].lower() in ("on", "1", "true")) if len(a) > 1 else True
            do_light(t, lid, on, 5.0)
        elif ns.cmd == "arm":
            do_arm(t, int(a[0]) if a else 0, 6.0)
        elif ns.cmd == "rotate":
            pos = a[0] if a else "all"
            pos = "all" if pos == "all" else int(pos)
            do_rotate(t, pos, 12.0)
        elif ns.cmd == "axis":
            do_axis(t, a[0], float(a[1]), ns.timeout)
        elif ns.cmd == "pos":
            do_pos(t, float(a[0]), float(a[1]), ns.timeout)
        elif ns.cmd == "turn":
            do_turn(t, float(a[0]), 30.0)
        elif ns.cmd == "arc":
            do_arc(t, float(a[0]), float(a[1]), float(a[2]),
                   float(a[3]), float(a[4]), 120.0)
        elif ns.cmd == "sync":
            do_sync(t, float(a[0]) if a else 0.0,
                    float(a[1]) if len(a) > 1 else 0.0)
        elif ns.cmd == "all":
            do_all(t, ns.timeout)
    except ValueError as e:
        print(f"参数错误: {e}")
        comm.shutdown()
        sys.exit(2)
    except IndexError:
        print("缺少参数, 用法见文件头注释")
        comm.shutdown()
        sys.exit(2)

    ok = t.report()
    comm.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
