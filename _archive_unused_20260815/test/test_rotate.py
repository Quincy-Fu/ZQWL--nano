"""测试转盘通信: 依次发送位置 0-4, 间隔 2 秒.
用法: python3 test_rotate.py [/dev/ttyCH341USB0]
"""
import sys
import time

from comm import init, send_rotate

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyCH341USB0"

print(f"连接 {PORT} @ 115200...")
init(PORT, 115200)
print("已连接, 开始测试转盘 5 个位置 (Ctrl+C 退出)\n")

try:
    for pos in range(5):
        print(f">>> 发送转盘位置 {pos}")
        send_rotate(pos)
        time.sleep(2)
except KeyboardInterrupt:
    print("\n测试结束")
