#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物块识别 V7 - 黑色用宽高比
- 黑色: 不卡圆度, 只卡宽高比(排除引导线)
- 引导线: 长宽比很大 → 排除
- 物块: 长宽比接近1 → 识别
"""

import cv2
import numpy as np
import json
import os


CONFIG = {
    "usb_device": 0,
    "width": 640,
    "height": 480,
    "fps": 30,
    "thresholds_file": "hsv_thresholds.json",
    
    "roi_size": 200,
    
    # 黑色专有参数
    "black_min_area": 500,        # 最小黑色面积
    "black_max_aspect": 2.5,      # 长宽比上限(引导线>2.5, 物块<2.5)
    
    # 其他颜色参数
    "color_min_area": 800,
    "color_max_area": 80000,
    "color_min_circularity": 0.3,
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

COLOR_BGR = {
    "黑": (0, 0, 0),
    "白": (200, 200, 200),
    "红": (0, 0, 255),
    "绿": (0, 255, 0),
    "蓝": (255, 0, 0),
}

KNOWN_COLORS = ["黑", "红", "绿", "蓝"]


def load_thresholds(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_HSV


# ============ 摄像头 ============
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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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


# ============ 黑色专用检测 ============
def find_black_block(mask, min_area, max_aspect):
    """
    找黑色物块
    不卡圆度(黑色圆度难算), 只卡宽高比
    引导线长宽比>2.5, 物块长宽比<=2.5
    """
    # 形态学: 闭运算填补黑色物块内部小洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 不做开运算, 怕把物块边缘削弱
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / max(min(w, h), 1)
        
        # 引导线: 长宽比很大(细长条)
        if aspect > max_aspect:
            continue
        
        if area > best_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                best = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                best_area = area
    
    return best, best_area


# ============ 其他颜色检测 ============
def find_color_block(mask, min_area, max_area, min_circ, max_aspect):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < min_circ:
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


# ============ 识别器 ============
class BlockRecognizer:
    def __init__(self, thresholds, config):
        self.thresholds = thresholds
        self.config = config
        self.roi_center = (config["width"] // 2, config["height"] // 2)
        self.roi_radius = config["roi_size"]
    
    def _make_roi_mask(self, shape):
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, self.roi_center, self.roi_radius, 255, -1)
        return mask
    
    def recognize(self, frame):
        roi_mask = self._make_roi_mask(frame.shape)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_roi = cv2.bitwise_and(hsv, hsv, mask=roi_mask)
        
        detected = {}
        block_positions = {}
        
        # === 黑色: 用宽高比判断 ===
        black_thr = self.thresholds["黑"]
        black_mask = cv2.inRange(hsv_roi,
                                tuple(black_thr["lower"]),
                                tuple(black_thr["upper"]))
        # 应用ROI(已经有roi的mask, 这里再叠一层)
        black_mask = cv2.bitwise_and(black_mask, roi_mask)
        
        black_center, black_area = find_black_block(
            black_mask,
            self.config["black_min_area"],
            self.config["black_max_aspect"]
        )
        if black_center:
            detected["黑"] = black_area
            block_positions["黑"] = black_center
        else:
            detected["黑"] = 0
        
        # === 其他颜色 ===
        for color in ["红", "绿", "蓝"]:
            thr = self.thresholds[color]
            mask = cv2.inRange(hsv_roi,
                              tuple(thr["lower"]),
                              tuple(thr["upper"]))
            if "lower2" in thr:
                mask2 = cv2.inRange(hsv_roi,
                                   tuple(thr["lower2"]),
                                   tuple(thr["upper2"]))
                mask = cv2.bitwise_or(mask, mask2)
            mask = cv2.bitwise_and(mask, roi_mask)
            
            center, area = find_color_block(
                mask,
                self.config["color_min_area"],
                self.config["color_max_area"],
                self.config["color_min_circularity"],
                self.config["color_max_aspect"]
            )
            if center:
                detected[color] = area
                block_positions[color] = center
            else:
                detected[color] = 0
        
        # 找最大
        best_color = max(detected, key=detected.get)
        max_pixels = detected[best_color]
        
        if max_pixels >= 500:
            return best_color, {
                "detected": detected,
                "positions": block_positions,
            }
        
        # 4种都没识别到, 推断白
        has_anything = self._check_has_anything(frame, roi_mask)
        if has_anything:
            return "白", {
                "detected": detected,
                "positions": {"白": self.roi_center},
            }
        
        return None, {"detected": detected, "positions": {}}
    
    def _check_has_anything(self, frame, roi_mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.bitwise_and(gray, gray, mask=roi_mask)
        edges = cv2.Canny(gray_roi, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                return True
        return False
    
    def draw(self, frame, color, info):
        display = frame.copy()
        h, w = frame.shape[:2]
        
        cv2.circle(display, self.roi_center, self.roi_radius, (255, 0, 255), 2)
        
        for c, (cx, cy) in info.get("positions", {}).items():
            if c == "白":
                cv2.circle(display, (cx, cy), 25, (200, 200, 200), 3)
                cv2.putText(display, "白(推断)", (cx - 30, cy + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            else:
                bgr = COLOR_BGR.get(c, (255, 255, 255))
                cv2.circle(display, (cx, cy), 25, bgr, 3)
                cv2.putText(display, c, (cx - 20, cy + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
        
        if color:
            if color == "白":
                cv2.rectangle(display, (0, 0), (w, 50), (200, 200, 200), -1)
                cv2.putText(display, "白色 (4色都没识别到)", (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                bgr = COLOR_BGR.get(color, (255, 255, 255))
                cv2.rectangle(display, (0, 0), (w, 50), bgr, -1)
                cv2.putText(display, f"识别: {color}", (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            cv2.rectangle(display, (0, 0), (w, 50), (50, 50, 50), -1)
            cv2.putText(display, "无物块", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        detected = info.get("detected", {})
        y = 70
        for c in KNOWN_COLORS:
            px = detected.get(c, 0)
            bgr = COLOR_BGR.get(c, (255, 255, 255))
            cv2.rectangle(display, (w - 110, y - 15), (w, y + 5), bgr, -1)
            cv2.putText(display, f"{px}", (w - 60, y - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y += 25
        
        return display


def main():
    print("=" * 50)
    print("物块识别 V7 - 黑色用宽高比")
    print("=" * 50)
    print("黑色: 卡宽高比<=2.5 (排除引导线)")
    print("其他颜色: 圆度+宽高比")
    print()
    print("操作:")
    print("  1-5  选序号")
    print("  空格 保存")
    print("  r    重置当前")
    print("  R    重置所有")
    print("  q    退出")
    
    thresholds = load_thresholds(CONFIG["thresholds_file"])
    cam = USBCamera(CONFIG["usb_device"], CONFIG["width"],
                    CONFIG["height"], CONFIG["fps"])
    cam.start()
    
    recognizer = BlockRecognizer(thresholds, CONFIG)
    slots = [None] * 5
    current_pos = None
    
    cv2.namedWindow("Block V7", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            
            color, info = recognizer.recognize(frame)
            display = recognizer.draw(frame, color, info)
            
            h, w = frame.shape[:2]
            
            slot_y = h - 60
            for i in range(5):
                x = 10 + i * (w - 20) // 5
                slot_w = (w - 20) // 5 - 5
                pos_num = i + 1
                if current_pos == pos_num:
                    cv2.rectangle(display, (x - 2, slot_y - 2),
                                 (x + slot_w + 2, slot_y + 42), (255, 255, 0), 3)
                s = slots[i]
                if s is None:
                    cv2.rectangle(display, (x, slot_y),
                                 (x + slot_w, slot_y + 40), (80, 80, 80), -1)
                    cv2.putText(display, f"{pos_num}.空", (x + 5, slot_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    bgr = COLOR_BGR.get(s, (100, 100, 100))
                    cv2.rectangle(display, (x, slot_y),
                                 (x + slot_w, slot_y + 40), bgr, -1)
                    cv2.putText(display, f"{pos_num}.{s}", (x + 5, slot_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            if current_pos is None:
                hint = "请按 1-5 选序号"
            else:
                if color is None:
                    hint = f"{current_pos}号位 - 等待物块"
                elif color == "白":
                    hint = f"{current_pos}号位 - 推断为白, 按空格保存"
                else:
                    hint = f"{current_pos}号位 - 识别到 {color}, 按空格保存"
            cv2.putText(display, hint, (10, h - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow("Block V7", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif ord('1') <= key <= ord('5'):
                current_pos = key - ord('0')
                print(f"\n>>> {current_pos}号位")
            elif key == ord(' '):
                if current_pos is None:
                    print("  先选序号")
                elif color is None:
                    print(f"  {current_pos}号位: 没物块")
                else:
                    slots[current_pos - 1] = color
                    print(f"  保存 {current_pos}号 = {color}")
                    for j in range(5):
                        if slots[j] is None:
                            current_pos = j + 1
                            break
            elif key == ord('r'):
                if current_pos:
                    slots[current_pos - 1] = None
            elif key == ord('R'):
                slots = [None] * 5
                current_pos = None
    
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("\n最终:")
        for i, c in enumerate(slots):
            print(f"  {i+1}号: {c}")


if __name__ == "__main__":
    main()