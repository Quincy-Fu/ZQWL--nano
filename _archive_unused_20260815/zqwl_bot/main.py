"""PID 调参仿真 — 实时可调参数, Foxglove 看波形."""

import threading
import time
from foxglove_streamer import FoxgloveStreamer
from nav import Navigator, _FakeSerial


def _show_status(nav, fake):
    p = fake.get_pose()
    kp = nav._pos_pid._px.kp
    ki = nav._pos_pid._px.ki
    kd = nav._pos_pid._px.kd
    kpw = nav._ang_pid.kp
    kiw = nav._ang_pid.ki
    kdw = nav._ang_pid.kd
    idle = "空闲" if nav.is_idle() else "移动中"
    loop = "循环中" if _loop_running else "停止"
    print(f"[{idle}] [{loop}] 位置:({p[0]:.3f},{p[1]:.3f}) θ:{p[2]:.3f} | "
          f"位置PID: kp={kp} ki={ki} kd={kd} | 朝向PID: kp={kpw} ki={kiw} kd={kdw}")


_loop_running = False
_loop_stop = threading.Event()


def _loop_worker(nav, tx, ty):
    """后台线程: 在 (0,0) 和 (tx,ty) 之间来回跑, 波形一直动."""
    while not _loop_stop.is_set():
        nav.goto(tx, ty, 0.0)
        while not nav.is_idle() and not _loop_stop.is_set():
            time.sleep(0.05)
        if _loop_stop.is_set():
            break
        nav.goto(0.0, 0.0, 0.0)
        while not nav.is_idle() and not _loop_stop.is_set():
            time.sleep(0.05)


def main():
    global _loop_running, _loop_stop

    streamer = FoxgloveStreamer(port=8765)
    fake = _FakeSerial()
    nav = Navigator(streamer=streamer, max_v=1, kp_v=2.0, ki_v=0.02, kd_v=0.2, pos_tol=0.005)
    nav.start(ser_mod=fake)

    print("=" * 55)
    print("PID 调参仿真 — Foxglove → ws://<IP>:8765 → robot_debug")
    print()
    print("  loop x y    来回循环跑 (0,0)↔(x,y), 波形不间断")
    print("  stop        停止循环")
    print("  goto x y    单次目标点")
    print("  kp/kd/ki    改位置环 (运行时直接改, 立刻生效)")
    print("  kpw/kdw/kiw 改朝向环")
    print("  maxv 值     改最大线速度 m/s")
    print("  maxw 值     改最大角速度 rad/s")
    print("  s           查看状态")
    print("  q           退出")
    print("=" * 55)

    try:
        while True:
            cmd = input("\n> ").strip()
            if not cmd:
                continue

            parts = cmd.split()
            head = parts[0].lower()

            if head == 'q':
                _loop_stop.set()
                break

            if head == 's':
                _show_status(nav, fake)
                continue

            # ── 循环来回跑 ──
            if head == 'loop':
                if len(parts) != 3:
                    print("用法: loop x y")
                    continue
                try:
                    tx, ty = float(parts[1]), float(parts[2])
                except ValueError:
                    print("请输入数字")
                    continue
                _loop_stop.set()  # 先停旧的
                time.sleep(0.1)
                _loop_stop.clear()
                _loop_running = True
                t = threading.Thread(target=_loop_worker, args=(nav, tx, ty), daemon=True)
                t.start()
                print(f"🔄 循环: (0,0) ↔ ({tx:.2f}, {ty:.2f})  只管调参数, 看波形变化!")
                continue

            if head == 'stop':
                _loop_stop.set()
                _loop_running = False
                nav.cancel()
                print("⏹ 循环已停止")
                continue

            # ── PID 参数 ──
            if head in ('kp', 'ki', 'kd', 'kpw', 'kiw', 'kdw', 'maxv', 'maxw'):
                if len(parts) != 2:
                    print("用法: kp 2.0")
                    continue
                try:
                    val = float(parts[1])
                except ValueError:
                    print("请输入数字")
                    continue

                if head == 'kp':
                    nav._pos_pid._px.kp = nav._pos_pid._py.kp = val
                elif head == 'ki':
                    nav._pos_pid._px.ki = nav._pos_pid._py.ki = val
                elif head == 'kd':
                    nav._pos_pid._px.kd = nav._pos_pid._py.kd = val
                elif head == 'kpw':
                    nav._ang_pid.kp = val
                elif head == 'kiw':
                    nav._ang_pid.ki = val
                elif head == 'kdw':
                    nav._ang_pid.kd = val
                elif head == 'maxv':
                    nav._max_v = val
                elif head == 'maxw':
                    nav._max_w = val
                if head not in ('maxv', 'maxw'):
                    nav._reset_pids()
                print(f"✓ {head} = {val}")
                continue

            # ── 单次目标 ──
            if head == 'goto':
                args = parts[1:]
                try:
                    if len(args) == 2:
                        x, y = float(args[0]), float(args[1])
                        th = 0.0
                    elif len(args) == 3:
                        x, y, th = float(args[0]), float(args[1]), float(args[2])
                    else:
                        print("用法: goto x y [theta]")
                        continue
                except ValueError:
                    print("请输入数字")
                    continue
                nav.goto(x, y, th)
                print(f"📍 前往 ({x:.2f}, {y:.2f}, θ={th:.2f})")
                continue

            print(f"未知命令: {head}")

    except KeyboardInterrupt:
        pass
    finally:
        _loop_stop.set()
        nav.stop()
        print("退出")


if __name__ == "__main__":
    main()
