import time
from foxglove_streamer import FoxgloveStreamer

# 1. 启动 Foxglove 数据推送服务
fg_streamer = FoxgloveStreamer(port=8765)

# 假设这是你的上位机位置环主循环
target_x = 100.0
current_x = 0.0

print("🤖 主循环开始，等待 Foxglove 连入...")

while True:
    # 模拟传感器读取与 PID 模拟计算 (实际替换为你自己的真实坐标和 PID 计算逻辑)
    err = target_x - current_x
    vx_cmd = err * 0.2
    current_x += vx_cmd * 0.1  # 模拟小车移动
    
    # 🌟 核心：一行代码把想要看波形的数据喂给 Foxglove！
    fg_streamer.send_data({
        "target_x": target_x,
        "current_x": current_x,
        "vx_cmd": vx_cmd
    })
    
    time.sleep(0.03) # 30Hz 控制频率