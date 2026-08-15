"""
Foxglove WebSocket 数据推送 — PID 调参波形可视化.

电脑端 Foxglove 连接 ws://<Jetson_IP>:8765, subscribe topic "robot_debug".

═══ 12 个数据字段 ═══

┌─ 目标值 (想让机器人去哪) ─────────────────────────────┐
│ target_x     目标点 X 坐标 (m)                         │
│ target_y     目标点 Y 坐标 (m)                         │
│ target_theta 目标朝向角 (rad, 0=正X方向, 1.57=正Y)     │
└───────────────────────────────────────────────────────┘

┌─ 当前值 (机器人实际在哪) ─────────────────────────────┐
│ current_x      实际 X 坐标 (m)                         │
│ current_y      实际 Y 坐标 (m)                         │
│ current_theta  实际朝向角 (rad)                        │
└───────────────────────────────────────────────────────┘

┌─ 误差 (调 PID 主要看这个, 越快归零越好) ──────────────┐
│ error_x    本体系前后方向剩余距离 (m)                   │
│ error_y    本体系左右方向剩余距离 (m)                   │
│ error_theta朝向还差多少 (rad)                           │
└───────────────────────────────────────────────────────┘

┌─ 速度指令 (PID 算出的输出) ───────────────────────────┐
│ vx_cmd 前后速度 (m/s)                                  │
│ vy_cmd 左右速度 (m/s)                                  │
│ w_cmd  旋转角速度 (rad/s)                              │
└───────────────────────────────────────────────────────┘

═══ Foxglove 面板配置 (3 张图) ═══

图1 — 位置跟踪: 看目标 vs 实际是否重合
  Y轴: target_x, current_x, target_y, current_y

图2 — 误差曲线 (调参核心):
  Y轴: error_x, error_y, error_theta
  判断: 快速归零=好 / 反复震荡=P太大或D太小 / 差一截不动=加I

图3 — 速度指令: 看 PID 输出是否平滑
  Y轴: vx_cmd, vy_cmd, w_cmd
  如果到了目标还在正负乱跳 → D太小或P太大

═══ 调参命令 (main.py 终端交互) ═══

  goto 1.0 0.5     设定目标, 看波形
  kp 2.0            改位置P (大了冲过头, 小了到得慢)
  ki 0.05           改位置I (消静差, 大了震荡)
  kd 0.1            改位置D (抑超调, 大了发抖)
  kpw 3.0           改朝向P
  kdw 0.2           改朝向D
  s                 查看当前位姿和所有PID值
"""

import asyncio
import json
import threading
import time
from foxglove_websocket.server import FoxgloveServer

class FoxgloveStreamer:
    def __init__(self, host="0.0.0.0", port=8765):
        """
        初始化 Foxglove WebSocket 服务器 (运行在后台线程，绝不阻塞主循环)
        """
        self.host = host
        self.port = port
        self.loop = asyncio.new_event_loop()
        self.server = None
        self.channel_id = None
        
        # 启动后台线程跑 WebSocket 事件循环
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        
        # 等待服务器启动完成
        time.sleep(0.5)
        print(f"🚀 Foxglove 服务已启动！请在电脑端连接 ws://<Jetson_IP>:{self.port}")

    def _run_server(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start_async_server())

    async def _start_async_server(self):
        self.server = FoxgloveServer(self.host, self.port, "Python-Robot-Server")
        
        # 注册一个 JSON 数据通道 (用来发送你的坐标、速度、PID等波形数据)
        self.channel_id = await self.server.add_channel({
            "topic": "robot_debug",
            "encoding": "json",
            "schemaName": "RobotDebugData",
            "schema": json.dumps({
                "type": "object",
                "properties": {
                    "target_x":    {"type": "number"},
                    "target_y":    {"type": "number"},
                    "target_theta":{"type": "number"},
                    "current_x":    {"type": "number"},
                    "current_y":    {"type": "number"},
                    "current_theta":{"type": "number"},
                    "error_x":     {"type": "number"},
                    "error_y":     {"type": "number"},
                    "error_theta": {"type": "number"},
                    "vx_cmd":      {"type": "number"},
                    "vy_cmd":      {"type": "number"},
                    "w_cmd":       {"type": "number"}
                }
            })
        })
        self.server.start()  # start() 不是 async, 不 await
        await asyncio.Event().wait()

    def send_data(self, data_dict: dict):
        """
        在 main.py 主循环里调用此函数，传入你想绘制波形的数据字典
        例如: streamer.send_data({"target_x": 100, "current_x": 85.2, "vx_cmd": 20.5})
        """
        if self.server and self.channel_id is not None:
            msg = json.dumps(data_dict).encode('utf-8')
            now_ns = int(time.time() * 1e9) # 当前纳秒级时间戳
            
            # 跨线程投递异步发送任务
            asyncio.run_coroutine_threadsafe(
                self.server.send_message(self.channel_id, now_ns, msg),
                self.loop
            )