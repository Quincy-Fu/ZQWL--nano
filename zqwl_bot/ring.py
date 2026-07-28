import cv2
import numpy as np

# 配置
CONFIG = {
    "usb_device": 0,
    "width": 640,
    "height": 480,
    "fps": 30,

    # 检测参数
    "black_v_max": 100,        # 黑色的 V 上限
    "min_radius_px": 20,       # 最小圆半径(像素)
    "max_radius_px": 400,      # 最大圆半径(像素)
    "min_circularity": 0.75,   # 圆形度阈值(越接近1越圆)
    "min_area": 200,           # 最小面积(像素²)
}


# ============ USB 摄像头 ============
class USBCamera:
    def __init__(self, device=0, width=640, height=480, fps=30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开 USB 摄像头")
        print(f"[Camera] {self.width}x{self.height}")

    def read(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        if self.cap:
            self.cap.release()


# ============ 黑色圆环检测器 ============
class BlackRingDetector:
    """检测黑色印刷的同心圆环（不匹配固定半径）"""

    def __init__(self, config):
        self.config = config

    def detect(self, frame):
        """检测所有黑色圆环，返回 [(cx, cy, r), ...]"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, self.config["black_v_max"]])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config["min_area"]:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.config["min_circularity"]:
                continue

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < self.config["min_radius_px"] or r > self.config["max_radius_px"]:
                continue

            results.append((int(cx), int(cy), int(r), round(circularity, 2)))

        return results

    def draw(self, frame, results):
        display = frame.copy()
        for i, (cx, cy, r, circ) in enumerate(results):
            color = (0, 255, 0)
            cv2.circle(display, (cx, cy), r, color, 2)
            cv2.line(display, (cx - 20, cy), (cx + 20, cy), color, 2)
            cv2.line(display, (cx, cy - 20), (cx, cy + 20), color, 2)
            cv2.circle(display, (cx, cy), 4, color, -1)
            label = f"#{i+1} ({cx},{cy}) r={r} c={circ}"
            cv2.putText(display, label, (cx + 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return display


# ============ 主程序 ============
def main():
    print("=" * 50)
    print("黑色圆环检测")
    print("=" * 50)
    print("按 'q' 退出，按 's' 保存截图")
    print("按 '+' 提高黑色阈值，按 '-' 降低黑色阈值")

    cam = USBCamera(CONFIG["usb_device"], CONFIG["width"], CONFIG["height"], CONFIG["fps"])
    cam.start()

    detector = BlackRingDetector(CONFIG)

    cv2.namedWindow("Black Ring Detector", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            results = detector.detect(frame)
            display = detector.draw(frame, results)

            info = f"检测到 {len(results)} 个圆环   V_max={CONFIG['black_v_max']}"
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            for i, (cx, cy, r, circ) in enumerate(results):
                txt = f"  #{i+1}: 中心({cx},{cy}) 半径={r} 圆度={circ}"
                cv2.putText(display, txt, (10, 60 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Black Ring Detector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n中心坐标汇总:")
                for i, (cx, cy, r, circ) in enumerate(results):
                    print(f"  #{i+1}: ({cx}, {cy})  r={r}  c={circ}")
                if not results:
                    print("  (未检测到圆环)")
                break
            elif key == ord('s'):
                cv2.imwrite("rings_snapshot.png", display)
                print("[保存] rings_snapshot.png")
            elif key == ord('+') or key == ord('='):
                CONFIG["black_v_max"] = min(255, CONFIG["black_v_max"] + 5)
                print(f"[阈值] V_max = {CONFIG['black_v_max']}")
            elif key == ord('-') or key == ord('_'):
                CONFIG["black_v_max"] = max(0, CONFIG["black_v_max"] - 5)
                print(f"[阈值] V_max = {CONFIG['black_v_max']}")

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()