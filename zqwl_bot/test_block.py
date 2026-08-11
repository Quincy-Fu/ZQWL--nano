#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物块识别 V23 - ROI 圆框内 HSV 投票
- 黑色: V<阈值 (固定)
- 红/绿/蓝: H+S 范围
- 白色: V 高 + S 低
"""

import cv2
import numpy as np
import json
import os


CONFIG = {
    "usb_device": 1,
    "width": 640,
    "height": 480,
    "fps": 30,
    "thresholds_file": "hsv_thresholds.json",
    "roi_radius": 200,

    "vote_min_pct": 0.25,

    # 黑色: V<60 算黑 (反光黑块暗的部分)
    "black_v_max": 60,
    # 白色: V>150 + S<80 算白
    "white_v_min": 150,
    "white_s_max": 80,
}


def load_thresholds(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "红": {"lower": [0, 80, 80],   "upper": [10, 255, 255],
              "lower2": [170, 80, 80], "upper2": [180, 255, 255]},
        "绿": {"lower": [30, 40, 40],  "upper": [90, 255, 255]},
        "蓝": {"lower": [95, 80, 80],  "upper": [135, 255, 255]},
    }


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

    def read(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        if self.cap:
            self.cap.release()


def recognize(frame, cfg, thresholds):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    S = hsv[:, :, 1]
    H = hsv[:, :, 0]

    # ROI 圆框 (画面正中心)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (w // 2, h // 2), cfg["roi_radius"], 255, -1)

    roi_total = int(np.sum(roi_mask == 255))
    if roi_total == 0:
        return None, {"pct": {}, "roi_mask": roi_mask}

    # === 黑色: V 低于阈值 ===
    black_mask = (V < cfg["black_v_max"]).astype(np.uint8) * 255
    black_mask = cv2.bitwise_and(black_mask, roi_mask)

    # === 红/绿/蓝: H+S ===
    color_masks = {"黑": black_mask}

    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    color_masks["红"] = cv2.bitwise_and(cv2.bitwise_or(red1, red2), roi_mask)

    color_masks["绿"] = cv2.bitwise_and(
        cv2.inRange(hsv, (30, 40, 40), (90, 255, 255)), roi_mask)

    color_masks["蓝"] = cv2.bitwise_and(
        cv2.inRange(hsv, (95, 80, 80), (135, 255, 255)), roi_mask)

    # === 白色: V 高 + S 低 ===
    white_mask = ((V > cfg["white_v_min"]) & (S < cfg["white_s_max"])).astype(np.uint8) * 255
    color_masks["白"] = cv2.bitwise_and(white_mask, roi_mask)

    pct = {}
    for color, mask in color_masks.items():
        pct[color] = float(np.sum(mask == 255)) / roi_total

    best_color = max(pct, key=lambda k: pct[k])
    best_pct = pct[best_color]

    info = {
        "pct": pct,
        "best": best_color,
        "best_pct": best_pct,
        "roi_mask": roi_mask,
    }

    if best_pct < cfg["vote_min_pct"]:
        return None, info

    return best_color, info


def main():
    print("Block V23 - ROI circle, vote")
    print("1-5 slot, SPACE save, q quit")

    thresholds = load_thresholds(CONFIG["thresholds_file"])
    cam = USBCamera(CONFIG["usb_device"], CONFIG["width"],
                    CONFIG["height"], CONFIG["fps"])
    cam.start()

    slots = [None] * 5
    current_pos = None
    cv2.namedWindow("Block V23", cv2.WINDOW_NORMAL)

    COLOR_BGR = {"黑": (0, 0, 0), "白": (200, 200, 200),
                 "红": (0, 0, 255), "绿": (0, 255, 0), "蓝": (255, 0, 0)}

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            color, info = recognize(frame, CONFIG, thresholds)
            pct = info.get("pct", {})
            best = info.get("best", "?")
            best_pct = info.get("best_pct", 0)

            display = frame.copy()
            h, w = frame.shape[:2]

            # 画 ROI 圆框
            cv2.circle(display, (w // 2, h // 2), CONFIG["roi_radius"],
                      (255, 0, 255), 2)

            # 顶部
            if color:
                bgr = COLOR_BGR.get(color, (255, 255, 255))
                cv2.rectangle(display, (0, 0), (w, 50), bgr, -1)
                cv2.putText(display, f"[{color}] {best_pct*100:.0f}%",
                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            else:
                cv2.rectangle(display, (0, 0), (w, 50), (50, 50, 50), -1)
                cv2.putText(display, f"NO  max={best} {best_pct*100:.0f}%",
                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 右侧
            y = 70
            for c in ["黑", "白", "红", "绿", "蓝"]:
                p = pct.get(c, 0)
                bgr = COLOR_BGR.get(c, (255, 255, 255))
                cv2.rectangle(display, (w - 130, y - 15), (w, y + 5), bgr, -1)
                if c == best:
                    cv2.rectangle(display, (w - 130, y - 15), (w, y + 5), (0, 255, 255), 2)
                cv2.putText(display, f"{c} {p*100:.0f}%",
                           (w - 120, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           (0, 0, 0), 1)
                y += 22

            # 底部
            slot_y = h - 60
            for i in range(5):
                x = 10 + i * (w - 20) // 5
                slot_w = (w - 20) // 5 - 5
                if current_pos == i + 1:
                    cv2.rectangle(display, (x - 2, slot_y - 2),
                                 (x + slot_w + 2, slot_y + 42), (255, 255, 0), 3)
                s = slots[i]
                if s is None:
                    cv2.rectangle(display, (x, slot_y), (x + slot_w, slot_y + 40),
                                 (80, 80, 80), -1)
                    cv2.putText(display, f"{i+1}.--", (x + 5, slot_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    bgr = COLOR_BGR.get(s, (100, 100, 100))
                    cv2.rectangle(display, (x, slot_y), (x + slot_w, slot_y + 40),
                                 bgr, -1)
                    cv2.putText(display, f"{i+1}.{s}", (x + 5, slot_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Block V23", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif ord('1') <= key <= ord('5'):
                current_pos = key - ord('0')
            elif key == ord(' '):
                if current_pos is not None and color:
                    slots[current_pos - 1] = color
                    print(f"SAVE {current_pos} = {color}")
                    for j in range(5):
                        if slots[j] is None:
                            current_pos = j + 1
                            break
            elif key in (ord('r'), ord('R')):
                slots = [None] * 5
                current_pos = None

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()