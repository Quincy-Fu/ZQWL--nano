"""IMU 零偏校准触发脚本。

用法:
  python3 test_imu_calib.py                 # 默认串口
  python3 test_imu_calib.py /dev/ttyUSB0    # 指定串口

注意:
  运行前和运行期间必须让底盘完全静止。下位机会发送 IMU 0x70 校准命令，
  等待 IMU 完成/恢复姿态帧后返回结果，通常需要 7~9 秒。
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

    print(f"[imu_calib] 串口={port} 波特率={baud}")
    print("[imu_calib] 正在连接 MCU...")
    comm.init(port, baud)
    time.sleep(0.5)

    pose = comm.get_pose(max_age=2.0)
    if pose is None:
        print("[imu_calib] 警告: 未收到位姿帧，仍继续发送校准命令。")
    else:
        x, y, yaw = pose
        print(f"[imu_calib] MCU 在线: x={x:.3f} y={y:.3f} yaw={yaw:.1f}°")

    print("[imu_calib] 保持底盘完全静止，开始 IMU 零偏校准...")
    t0 = time.perf_counter()
    ok = comm.imu_calib(timeout=12.0)
    dt = time.perf_counter() - t0

    if ok:
        print(f"[imu_calib] OK - IMU 帧已恢复 ({dt:.2f}s)")
        pose = comm.get_pose(max_age=2.0)
        if pose is not None:
            x, y, yaw = pose
            print(f"[imu_calib] 当前位姿: x={x:.3f} y={y:.3f} yaw={yaw:.1f}°")
        return 0

    print(f"[imu_calib] FAIL - 超时或 IMU 帧未恢复 ({dt:.2f}s)")
    print("[imu_calib] 检查: IMU 串口是否在线、校准时是否保持静止、下位机是否已烧录新版。")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[imu_calib] 已取消")
        sys.exit(1)
    finally:
        comm.shutdown()
