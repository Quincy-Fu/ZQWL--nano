"""坐标约定验证脚本 — 逐条测试每个轴方向.

在 Jetson Nano 上运行:
    cd /path/to/ZQWL--nano/zqwl_bot
    python ../test/test_coord.py

每个测试前会提示你观察车的动作, 按回车继续.
串口默认 /dev/ttyCH341USB0, 可命令行改:
    python ../test/test_coord.py /dev/ttyUSB0
"""

import os
import sys
import time

# 确保能导入 zqwl_bot 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "zqwl_bot"))

from comm import init, shutdown, send_sync_pose, send_tox, send_toy
from comm import send_turnto, send_goto, wait_nav_response


def wait(msg: str):
    print(f"\n  >>> {msg}")
    input("  观察车的动作, 按回车继续...")


def run_test(port: str):
    print("=" * 50)
    print("  坐标约定上车验证")
    print("  +X=右, +Y=前, CW正(右转正)")
    print("=" * 50)

    init(port)
    time.sleep(0.5)

    # 复位坐标
    print("\n[0] 复位坐标 (sync_pose 0,0)")
    send_sync_pose(0.0, 0.0)
    r = wait_nav_response(timeout=5)
    print(f"  响应: {r}")

    # --- 测试1: 前进 ---
    print("\n[1] TOY y=0.3  (期望: 车往前移动30cm)")
    send_toy(0.3)
    wait("车往前走30cm了吗?")
    r = wait_nav_response(timeout=10)
    print(f"  响应: {r}")

    # --- 测试2: 后退 ---
    print("\n[2] TOY y=0.0  (期望: 车往后退回到Y=0)")
    send_toy(0.0)
    wait("车退回原位了吗?")
    r = wait_nav_response(timeout=10)
    print(f"  响应: {r}")

    # --- 测试3: 右移 ---
    print("\n[3] TOX x=0.1  (期望: 车往右横移10cm)")
    send_tox(0.1)
    wait("车往右横移10cm了吗?")
    r = wait_nav_response(timeout=10)
    print(f"  响应: {r}")

    # --- 测试4: 左移 ---
    print("\n[4] TOX x=0.0  (期望: 车往左横移回到X=0)")
    send_tox(0.0)
    wait("车左移回原位了吗?")
    r = wait_nav_response(timeout=10)
    print(f"  响应: {r}")

    # --- 测试5: 右转 ---
    print("\n[5] TURNTO 45  (期望: 车原地右转45度)")
    send_turnto(45.0)
    wait("车往右(顺时针)转了45度吗?")
    r = wait_nav_response(timeout=10)
    print(f"  响应: {r}")

    # --- 测试6: 左转 ---
    print("\n[6] TURNTO -45  (期望: 车从45度左转90度到-45度)")
    send_turnto(-45.0)
    wait("车往左(逆时针)转到了-45度吗?")
    r = wait_nav_response(timeout=10)
    print(f"  响应: {r}")

    # --- 测试7: 回正 ---
    print("\n[7] TURNTO 0  (期望: 车转回0度朝前)")
    send_turnto(0.0)
    wait("车转回初始朝向了吗?")
    r = wait_nav_response(timeout=10)
    print(f"  响应: {r}")

    # --- 测试8: GOTO 斜移 ---
    print("\n[8] GOTO x=0.1, y=0.3  (期望: 车斜移到右前方)")
    send_goto(0.1, 0.3)
    wait("车到了右前方(x=右10cm, y=前30cm)吗?")
    r = wait_nav_response(timeout=15)
    print(f"  响应: {r}")

    # --- 测试9: 回原点 ---
    print("\n[9] GOTO x=0, y=0  (期望: 车回到原点)")
    send_goto(0.0, 0.0)
    wait("车回到出发点了吗?")
    r = wait_nav_response(timeout=15)
    print(f"  响应: {r}")

    print("\n" + "=" * 50)
    print("  验证完成!")
    print("  如果所有方向都对, 坐标约定OK.")
    print("  如果某个方向反了, 告诉我具体哪个测试.")
    print("=" * 50)

    shutdown()


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyCH341USB0"
    print(f"串口: {port}")
    run_test(port)
