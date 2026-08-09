""""
block.py - 物块颜色识别 (无圆度检测, 3 帧快速版)
- 黑色: 宽高比 ≤ 2.5
- 红/绿/蓝: HSV + 宽高比 (不用圆度)
- 白色: 4 色都没识别到时推断
- 接口: recognize(frames=3, timeout=0.3) -> str
"""
import os
import json
import time
import cv2
import numpy as np
from collections import Counter


CONFIG = {
    # The USB camera can enumerate as /dev/video0 or /dev/video1.
    "usb_devices": (0, 1),
    "width": 640,
    "height": 480,
    "fps": 30,
    "thresholds_file": "hsv_thresholds.json",
    "roi_size": 200,

    # 黑色专有参数
    "black_min_area": 500,
    "black_max_aspect": 2.5,

    # 其他颜色参数 (无圆度, 只用宽高比)
    "color_min_area": 800,
    "color_max_area": 80000,
    "color_max_aspect": 2.5,
}

DEFAULT_HSV = {
    "黑": {"lower": [0, 0, 0],    "upper": [180, 255, 100]},
    "白": {"lower": [0, 0, 160],  "upper": [180, 50, 255]},
    "红": {"lower": [0, 80, 80],  "upper": [10, 255, 255],
          "lower2": [155, 80, 80], "upper2": [180, 255, 255]},
    "绿": {"lower": [30, 40, 40], "upper": [90, 255, 255]},
    "蓝": {"lower": [95, 80, 80], "upper": [135, 255, 255]},
}


# ============== 摄像头 ==============
class USBCamera:
    def __init__(self, device=0, width=640, height=480, fps=30):
        if isinstance(device, (tuple, list)):
            self.devices = tuple(dict.fromkeys(int(item) for item in device))
        else:
            self.devices = (int(device),)
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self):
        for device in self.devices:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.cap = cap
                self.device = device
                return
            cap.release()
        attempted = ", ".join(str(device) for device in self.devices)
        raise RuntimeError(f"Block: USB 摄像头打开失败 (tried devices {attempted})")

    def read(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        if self.cap:
            self.cap.release()


# ============== 阈值加载 ==============
_thresholds = None
def _get_thresholds():
    global _thresholds
    if _thresholds is None:
        if os.path.exists(CONFIG["thresholds_file"]):
            try:
                with open(CONFIG["thresholds_file"], "r") as f:
                    _thresholds = json.load(f)
                return _thresholds
            except Exception:
                pass
        _thresholds = DEFAULT_HSV
    return _thresholds


# ============== 黑色检测 (只宽高比) ==============
def find_black_block(mask, min_area, max_aspect):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect:
            continue
        if area > best_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                best = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                best_area = area
    return best, best_area


# ============== 其他颜色检测 (只宽高比, 无圆度) ==============
def find_color_block(mask, min_area, max_area, max_aspect):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect:
            continue
        if area > best_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                best = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                best_area = area
    return best, best_area


# ============== 单帧分类 ==============
def _classify_frame(frame, thresholds, cfg):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi, (cx, cy), cfg["roi_size"], 255, -1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.bitwise_and(hsv, hsv, mask=roi)

    detected = {}

    # 黑色 (宽高比)
    thr = thresholds["黑"]
    bmask = cv2.inRange(hsv_roi, tuple(thr["lower"]), tuple(thr["upper"]))
    bmask = cv2.bitwise_and(bmask, roi)
    center, area = find_black_block(bmask, cfg["black_min_area"], cfg["black_max_aspect"])
    detected["黑"] = area if center else 0

    # 红/绿/蓝 (HSV + 宽高比, 不用圆度)
    for color in ["红", "绿", "蓝"]:
        thr = thresholds[color]
        mask = cv2.inRange(hsv_roi, tuple(thr["lower"]), tuple(thr["upper"]))
        if "lower2" in thr:
            mask2 = cv2.inRange(hsv_roi, tuple(thr["lower2"]), tuple(thr["upper2"]))
            mask = cv2.bitwise_or(mask, mask2)
        mask = cv2.bitwise_and(mask, roi)
        center, area = find_color_block(mask, cfg["color_min_area"], cfg["color_max_area"],
                                        cfg["color_max_aspect"])
        detected[color] = area if center else 0

    best = max(detected, key=detected.get)
    if detected[best] >= 500:
        return best

    # 4 色都没识别到, 推断白
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.bitwise_and(gray, gray, mask=roi)
    edges = cv2.Canny(gray_roi, 30, 100)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            return "白"
    return None


# ============== 识别入口 ==============
def recognize(frames: int = 3, timeout: float = 5.0) -> str:
    """识别物块颜色.

    frames=3, timeout=0.3: 3 帧快速 (走弧用)
    frames=3, timeout=5.0: 3 帧 5 秒众数 (静止用)
    """
    thresholds = _get_thresholds()
    cap = USBCamera(CONFIG["usb_devices"], CONFIG["width"],
                    CONFIG["height"], CONFIG["fps"])
    cap.start()
    try:
        t0 = time.time()
        results = []
        while time.time() - t0 < timeout and len(results) < frames:
            frame = cap.read()
            if frame is None:
                continue
            color = _classify_frame(frame, thresholds, CONFIG)
            if color is not None:
                results.append(color)
        if not results:
            raise RuntimeError("Block: 未识别到任何物块")
        return Counter(results).most_common(1)[0][0]
    finally:
        cap.stop()


if __name__ == "__main__":
    print("Block (3 帧众数):", recognize(frames=3, timeout=5.0))
