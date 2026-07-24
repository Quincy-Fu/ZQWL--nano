import cv2
import numpy as np


# 配置
CONFIG = {
    "usb_device": 0,
    "width": 640,
    "height": 480,
    "fps": 30,
    
    # 圆环实际尺寸 (mm)
    "ring_radii_mm": [25, 45, 65, 85, 105],
    "ring_count": 5,            # 期望5个圆
    
    # 像素到mm的转换 (需要实测标定)
    "px_per_mm": 3.0,           # 1mm = 3像素 (估)
    
    # 检测参数
    "canny_low": 50,
    "canny_high": 150,
    "hough_param2": 25,
    "min_radius": 30,           # 最小半径(像素)
    "max_radius": 400,
    "center_tolerance": 20,     # 同心圆容差
}


# ============ USB摄像头 ============
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
            raise RuntimeError("无法打开USB摄像头")
        print(f"[Camera] {self.width}x{self.height}")
    
    def read(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def stop(self):
        if self.cap:
            self.cap.release()


# ============ 黑色同心圆环检测器 ============
class BlackRingDetector:
    """检测黑色印刷的同心圆环"""
    
    def __init__(self, config):
        self.config = config
        self.ring_radii_px = [int(r * config["px_per_mm"]) for r in config["ring_radii_mm"]]
        print(f"[检测器] 期望圆环半径(像素): {self.ring_radii_px}")
    
    def detect_circles(self, frame):
        """检测所有圆"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        edges = cv2.Canny(binary, 50, 150)
        
        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT,
            dp=1, minDist=30,
            param1=50, param2=self.config["hough_param2"],
            minRadius=self.config["min_radius"],
            maxRadius=self.config["max_radius"]
        )
        
        if circles is None:
            return []
        
        return [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]
    
    def group_concentric(self, circles):
        """把同心圆分组"""
        groups = []
        
        for cx, cy, r in circles:
            found = False
            for g in groups:
                gx, gy, radii = g
                if abs(cx - gx) <= self.config["center_tolerance"] and \
                   abs(cy - gy) <= self.config["center_tolerance"]:
                    n = len(radii)
                    g[0] = int((gx * n + cx) / (n + 1))
                    g[1] = int((gy * n + cy) / (n + 1))
                    radii.append(r)
                    found = True
                    break
            
            if not found:
                groups.append([cx, cy, [r]])
        
        return groups
    
    def match_known_rings(self, group_radii):
        """和已知5个半径匹配, 返回哪些圈被检测到"""
        detected = []
        for known_r in self.ring_radii_px:
            for r in group_radii:
                if abs(r - known_r) <= known_r * 0.15:
                    detected.append((known_r, r))
                    break
        return detected
    
    def detect(self, frame):
        """主检测"""
        circles = self.detect_circles(frame)
        groups = self.group_concentric(circles)
        
        results = []
        for cx, cy, radii in groups:
            radii_sorted = sorted(radii)
            matched = self.match_known_rings(radii_sorted)
            results.append({
                "center": (cx, cy),
                "rings": radii_sorted,
                "matched": matched,
                "ring_count": len(matched),
            })
        
        return results
    
    def draw(self, frame, results):
        """绘制"""
        display = frame.copy()
        h, w = frame.shape[:2]
        
        colors = [
            (0, 255, 0),    # 绿
            (255, 128, 0),  # 橙
            (255, 0, 255),  # 紫
            (0, 255, 255),  # 黄
            (128, 0, 255),  # 紫蓝
        ]
        
        for i, ring in enumerate(results):
            color = colors[i % len(colors)]
            cx, cy = ring["center"]
            radii = ring["rings"]
            
            for r in radii:
                cv2.circle(display, (cx, cy), r, color, 2)
            
            cv2.line(display, (cx - 20, cy), (cx + 20, cy), color, 2)
            cv2.line(display, (cx, cy - 20), (cx, cy + 20), color, 2)
            cv2.circle(display, (cx, cy), 5, color, -1)
            
            label = f"({cx}, {cy}) {ring['ring_count']}圈"
            cv2.putText(display, label, (cx + 25, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return display


# ============ 主程序 ============
def main():
    print("=" * 50)
    print("黑色同心圆环中心检测")
    print("=" * 50)
    print("圆环参数: 半径 25/45/65/85/105mm (5圈)")
    print("按 'q' 退出并打印中心坐标")
    print("按 's' 保存截图")
    
    cam = USBCamera(CONFIG["usb_device"], CONFIG["width"], CONFIG["height"], CONFIG["fps"])
    cam.start()
    
    detector = BlackRingDetector(CONFIG)
    all_centers = []
    
    cv2.namedWindow("Black Ring Detector", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            
            results = detector.detect(frame)
            display = detector.draw(frame, results)
            
            h, w = frame.shape[:2]
            info = f"检测到 {len(results)} 个同心圆环"
            cv2.putText(display, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            for i, ring in enumerate(results):
                cx, cy = ring["center"]
                txt = f"  #{i+1}: 中心({cx}, {cy}), {ring['ring_count']}圈"
                cv2.putText(display, txt, (10, 60 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Black Ring Detector", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n" + "=" * 50)
                print("中心坐标汇总:")
                print("=" * 50)
                for i, ring in enumerate(results):
                    print(f"  圆环 #{i+1}: 中心=({ring['center'][0]}, {ring['center'][1]}), "
                          f"圈数={ring['ring_count']}, 半径={ring['rings']}")
                if not results:
                    print("  (未检测到圆环)")
                break
            elif key == ord('s'):
                cv2.imwrite("rings_snapshot.png", display)
                print("[保存] rings_snapshot.png")
    
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()