import cv2
import numpy as np
import threading
import time
import platform
import os
import glob
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingTCPServer

# ==========================================
# 功能一：网页双路视频流显示类
# ==========================================

class MJPEGHandler(BaseHTTPRequestHandler):
    """自定义 HTTP 请求处理器，用于分发 HTML 页面和 MJPEG 视频流"""
    streamer = None  # 指向 WebStreamer 实例

    def log_message(self, format, *args):
        """静默服务器控制台的频繁日志输出"""
        pass

    def do_GET(self):
        # 1. 根目录：返回监控主页 HTML
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>xbhdcc 视觉监控面板</title>
                <style>
                    body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #f1f5f9; text-align: center; margin: 0; padding: 20px; }
                    h1 { color: #38bdf8; margin-bottom: 5px; font-weight: 800; }
                    .subtitle { color: #94a3b8; font-size: 14px; margin-bottom: 25px; }
                    
                    /* 核心改进：强制左右并排的 Grid 布局 */
                    .container { 
                        display: grid; 
                        grid-template-columns: repeat(2, 1fr); 
                        gap: 20px; 
                        max-width: 1400px; 
                        margin: 0 auto; 
                        padding: 10px;
                    }
                    
                    /* 手机等窄屏下自动退化为上下堆叠 */
                    @media (max-width: 900px) {
                        .container { grid-template-columns: 1fr; }
                    }
                    
                    .stream-box { 
                        background: #1e293b; 
                        padding: 15px; 
                        border-radius: 12px; 
                        box-shadow: 0 10px 25px rgba(0,0,0,0.5); 
                        border: 1px solid #334155; 
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }
                    .stream-box h2 { margin-top: 0; color: #38bdf8; font-size: 18px; border-bottom: 1px solid #334155; padding-bottom: 8px; width: 100%; }
                    img { width: 100%; max-width: 640px; height: auto; border-radius: 6px; background: #000; }
                    .footer { margin-top: 40px; color: #64748b; font-size: 12px; }
                </style>
            </head>
            <body>
                <h1>🎬 xbhdcc 实时视频流监视器</h1>
                <div class="subtitle">支持局域网多设备同时访问 | 左右双路画面对比</div>
                <div class="container">
                    <div class="stream-box">
                        <h2>📺 视频流 1</h2>
                        <img src="/stream/0" />
                    </div>
                    <div class="stream-box">
                        <h2>📺 视频流 2</h2>
                        <img src="/stream/1" />
                    </div>
                </div>
                <div class="footer">Powered by OpenCV & Python BaseHTTPServer</div>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
            
        # 2. 视频流路径：/stream/0 或 /stream/1
        elif self.path.startswith('/stream/'):
            try:
                stream_id = int(self.path.split('/')[-1])
            except ValueError:
                self.send_error(400, "Invalid stream ID")
                return

            if stream_id not in (0, 1):
                self.send_error(404, "Stream not found")
                return

            # 设置 MJPEG 流的 HTTP 响应头
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            try:
                while True:
                    # 从 WebStreamer 获取对应通道的图像
                    frame = self.streamer.get_frame(stream_id)
                    
                    # 编码为 JPG (支持 3 通道彩色和 1 通道灰度)
                    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b'\r\n')
                    
                    # 控制推流帧率，约 30 FPS
                    time.sleep(0.03)
            except (ConnectionResetError, BrokenPipeError):
                # 客户端关闭网页时静默退出
                pass
            except Exception as e:
                print(f"[WebStreamer] 推流异常: {e}")
        else:
            self.send_error(404)


class WebStreamer:
    """网页视频流服务器类，最多支持 2 路 OpenCV 图像同时显示"""
    def __init__(self, port=8080):
        self.port = port
        self.frames = {0: None, 1: None}
        self.lock = threading.Lock()
        self.server = None
        self.server_thread = None
        self._start_server()

    def _generate_placeholder(self, stream_id):
        """当用户还没有传入图像时，生成一个好看的等待占位图"""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:] = (30, 30, 30)
        cv2.rectangle(placeholder, (15, 15), (625, 465), (100, 100, 100), 2)
        cv2.putText(placeholder, f"Waiting for Stream {stream_id}...", (130, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return placeholder

    def _start_server(self):
        MJPEGHandler.streamer = self

        class ThreadedHTTPServer(ThreadingTCPServer, HTTPServer):
            allow_reuse_address = True

        self.server = ThreadedHTTPServer(('0.0.0.0', self.port), MJPEGHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"\n[WebStreamer] 网页服务已启动！")
        print(f"本地预览: http://localhost:{self.port}")
        print(f"局域网预览: http://<你的电脑IP>:{self.port}\n")

    def update_frame(self, stream_id, frame):
        """更新指定通道的图像 (stream_id 只能是 0 或 1)"""
        if stream_id not in (0, 1):
            raise ValueError("stream_id 必须是 0 或者 1")
        with self.lock:
            if frame is not None:
                self.frames[stream_id] = frame.copy()
            else:
                self.frames[stream_id] = None

    def get_frame(self, stream_id):
        """获取指定通道的图像，若为空则返回占位图"""
        with self.lock:
            if self.frames[stream_id] is None:
                return self._generate_placeholder(stream_id)
            return self.frames[stream_id]

    def stop(self):
        """停止服务器"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("[WebStreamer] 服务器已安全关闭。")


# ==========================================
# 功能二：系统摄像头探测函数 (完美支持 Win/Linux)
# ==========================================

def detect_cameras(max_to_test=15):
    """
    探测当前系统连接的所有摄像头，获取它们的 OpenCV 编号、真实硬件名称及支持的分辨率。
    支持 Windows (DSHOW 快速扫描) 和 Linux (sysfs 硬件树分析)
    """
    system_name = platform.system()
    print("=" * 60)
    print(f" 🔍 开始探测系统摄像头 (当前系统: {system_name})")
    print("=" * 60)

    available_cameras = []
    test_resolutions = [(1920, 1080), (1280, 720), (640, 480), (320, 240)]
    candidates = []

    if system_name == "Linux":
        # Linux 专属高级扫描：读取 V4L2 硬件树并过滤虚拟节点
        video_paths = glob.glob('/sys/class/video4linux/video*')
        if video_paths:
            sorted_paths = sorted(video_paths, key=lambda x: int(os.path.basename(x).replace('video', '')))
            for path in sorted_paths:
                dev_name = os.path.basename(path)
                idx = int(dev_name.replace('video', ''))
                
                friendly_name = "未知摄像头"
                name_file = os.path.join(path, 'name')
                if os.path.exists(name_file):
                    try:
                        with open(name_file, 'r', encoding='utf-8') as f:
                            friendly_name = f.read().strip()
                    except Exception:
                        pass
                
                # 过滤掉无法成像的虚拟节点
                ignore_keywords = ["metadata", "association", "statistics", "params", "meta"]
                if any(kw in friendly_name.lower() for kw in ignore_keywords):
                    continue
                
                candidates.append((idx, friendly_name))
        
        if not candidates:
            candidates = [(i, f"Camera {i}") for i in range(max_to_test)]
    else:
        # Windows / macOS 默认顺序扫描
        candidates = [(i, f"Camera {i}") for i in range(max_to_test)]

    # 遍历设备获取详细参数
    for index, name in candidates:
        if system_name == "Windows":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        elif system_name == "Linux":
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(index)

        if cap.isOpened():
            default_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            default_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            default_fps = cap.get(cv2.CAP_PROP_FPS)

            supported_res = []
            for w, h in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                res_str = f"{act_w}x{act_h}"
                if res_str not in supported_res:
                    supported_res.append(res_str)

            # 恢复默认值
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, default_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, default_h)

            fps_str = f"{default_fps:.1f}" if default_fps > 0 else "未知/动态"
            cam_info = {
                "index": index,
                "name": name,
                "default_res": f"{default_w}x{default_h}",
                "default_fps": fps_str,
                "supported_resolutions": supported_res
            }
            available_cameras.append(cam_info)

            print(f"\n[+] 发现摄像头 [编号: {index}]")
            print(f"    - 设备名称: {name}")
            print(f"    - 默认启动分辨率: {cam_info['default_res']}")
            print(f"    - 默认帧率 (FPS) : {cam_info['default_fps']}")
            print(f"    - 硬件支持分辨率: {', '.join(supported_res)}")
            
            if system_name == "Windows":
                print(f"    -OpenCV 启动代码建议: cv2.VideoCapture({index}, cv2.CAP_DSHOW)")
            elif system_name == "Linux":
                print(f"    -OpenCV 启动代码建议: cv2.VideoCapture({index}, cv2.CAP_V4L2)")
            else:
                print(f"    -OpenCV 启动代码建议: cv2.VideoCapture({index})")

            cap.release()

    print("\n" + "=" * 60)
    if not available_cameras:
        print("未检测到任何可用的摄像头设备！")
    else:
        print(f"探测完成！共发现 {len(available_cameras)} 个可用摄像头。")
    print("=" * 60)
    
    return available_cameras


# ==========================================
# 极简测试运行入口
# ==========================================
if __name__ == '__main__':
    # 1. 探测摄像头
    cams = detect_cameras()
    if not cams:
        print("错误：未检测到任何摄像头，无法启动演示！")
        exit()

    # 2. 启动网页服务器
    streamer = WebStreamer(port=8080)

    # 3. 打开第一个检测到的摄像头
    target_idx = cams[0]['index']
    sys_name = platform.system()
    if sys_name == "Windows":
        cap = cv2.VideoCapture(target_idx, cv2.CAP_DSHOW)
    elif sys_name == "Linux":
        cap = cv2.VideoCapture(target_idx, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(target_idx)
    
    print(f"\n[演示] 正在读取摄像头 {target_idx} 并推流到网页...")
    print("请在浏览器中打开: http://localhost:8080")
    print("按下键盘 'q' 键或在终端按 Ctrl+C 退出。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 【通道 0】：直接推送原始彩色画面
            streamer.update_frame(0, frame)

            # 【通道 1】：直接转成灰度图并推送（无需手动合并通道，imencode 会自动处理）
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            streamer.update_frame(1, gray_frame)

            # 控制主循环速度，避免 CPU 空转
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        streamer.stop()
