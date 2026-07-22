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
                    "target_x": {"type": "number"},
                    "current_x": {"type": "number"},
                    "target_y": {"type": "number"},
                    "current_y": {"type": "number"},
                    "vx_cmd": {"type": "number"},
                    "vy_cmd": {"type": "number"}
                }
            })
        })
        await self.server.start()
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