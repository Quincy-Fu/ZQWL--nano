"""转盘零点设置脚本 (一次性标定).

功能: 把转盘当前位置存为零点并写入 Emm_V5.0 驱动器 flash, 掉电不丢.
以后每次上电, 驱动器会自动回零 (Emm_V5_Origin_Trigger_Return nearest),
不需要再手动摆正.

使用方法:
  1. 上电, 等 MCU 启动完成 (~3s)
  2. 手动把转盘转到 slot 0 的位置 (你要作为零点的位置)
  3. 运行本脚本:
       python3 test_set_zero.py                 # 默认串口
       python3 test_set_zero.py /dev/ttyUSB0    # 指定串口
  4. 看到 "OK" 表示零点已保存

验证: 关电 → 手动转一下转盘 → 重新上电 → 转盘应自动回到零点

注意: 这是一次性操作, 存一次就行. 以后换场地需要重新标定时再跑一次.
"""

import sys
import time

try:
    from . import comm
except ImportError:
    import comm


def main() -> int:
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyCH341USB*"
    baud = 115200

    print(f"[set_zero] 串口={port} 波特率={baud}")
    print("[set_zero] 正在连接 MCU...")
    comm.init(port, baud)
    time.sleep(0.5)

    # 等 MCU 确认在线 (收到位姿帧)
    pose = comm.get_pose(max_age=2.0)
    if pose is None:
        print("[set_zero] 警告: 未收到位姿帧, MCU 可能未启动. 仍继续发送命令.")
    else:
        x, y, yaw = pose
        print(f"[set_zero] MCU 在线: x={x:.3f} y={y:.3f} yaw={yaw:.1f}")

    print("[set_zero] 发送零点设置命令...")
    t0 = time.perf_counter()
    ok = comm.set_zero(timeout=5.0)
    dt = time.perf_counter() - t0

    if ok:
        print(f"[set_zero] OK - 零点已保存到 flash ({dt:.2f}s)")
        print("[set_zero] 以后每次上电转盘会自动回零, 不需手动摆正.")
        return 0
    else:
        print(f"[set_zero] FAIL - 超时未收到响应 ({dt:.2f}s)")
        print("[set_zero] 检查: MCU 是否已启动? 串口是否正确? 驱动器是否已使能?")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[set_zero] 已取消")
        sys.exit(1)
    finally:
        comm.shutdown()
