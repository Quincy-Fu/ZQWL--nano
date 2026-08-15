#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物块识别 V25 - 输出车体坐标系下物块位置 + 车当前位姿
"""
import cv2
import numpy as np
import json
import os
import time
import math
import comm


CONFIG = {
    "usb_device": 1,
    "width": 640,
    "height": 480,
    "fps": 30,
    "thresholds_file": "hsv_thresholds.json",
    "roi_radius": 140,
    "vote_min_pct": 0.25,
    "black_v_max": 60,
    "white_v_min": 150,
    "white_s_max": 80,

    "confirm_frames": 3,
    "confirm_pos_diff": 30,
    "log_file": "block_log.txt",
    "print_console": True,

    # ★ 物块实际直径 (用于标定, 改这个)
    "block_diameter_m": 0.05,      # 5cm 物块
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
    def __init__(self, device=1, width=640, height=480, fps=30):
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
    """返回 (color, cx, cy, r_px, info)"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    S = hsv[:, :, 1]
    H = hsv[:, :, 0]

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (w // 2, h // 2), cfg["roi_radius"], 255, -1)
    roi_total = int(np.sum(roi_mask == 255))
    if roi_total == 0:
        return None, None, None, None, {"pct": {}, "roi_mask": roi_mask}

    color_masks = {}
    black_mask = (V < cfg["black_v_max"]).astype(np.uint8) * 255
    color_masks["黑"] = cv2.bitwise_and(black_mask, roi_mask)

    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    color_masks["红"] = cv2.bitwise_and(cv2.bitwise_or(red1, red2), roi_mask)
    color_masks["绿"] = cv2.bitwise_and(
        cv2.inRange(hsv, (30, 40, 40), (90, 255, 255)), roi_mask)
    color_masks["蓝"] = cv2.bitwise_and(
        cv2.inRange(hsv, (95, 80, 80), (135, 255, 255)), roi_mask)
    white_mask = ((V > cfg["white_v_min"]) & (S < cfg["white_s_max"])).astype(np.uint8) * 255
    color_masks["白"] = cv2.bitwise_and(white_mask, roi_mask)

    # ★ 每种颜色的中心 + 半径 (moments)
    centers = {}
    radii = {}
    for color, mask in color_masks.items():
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers[color] = (cx, cy)
            # 等效半径 (假设圆形): r = sqrt(area / pi)
            radii[color] = (float(mask.sum() / 255) / 3.14159) ** 0.5

    pct = {}
    for color, mask in color_masks.items():
        pct[color] = float(np.sum(mask == 255)) / roi_total

    best_color = max(pct, key=lambda k: pct[k])
    best_pct = pct[best_color]

    info = {
        "pct": pct,
        "best": best_color,
        "best_pct": best_pct,
        "centers": centers,
        "radii": radii,
        "roi_mask": roi_mask,
    }

    if best_pct < cfg["vote_min_pct"]:
        return None, None, None, None, info

    cx, cy = centers.get(best_color, (None, None))
    r_px = radii.get(best_color, None)
    return best_color, cx, cy, r_px, info


# ============ ★ 车体坐标系转换 ============
def block_in_body_frame(cx, cy, r_px, cfg):
    """
    算物块在车体坐标系下的位置
    +X = 车体右, +Y = 车体前
    摄像头反装: 图像 X 反, 图像 Y 反
    """
    if r_px is None or r_px <= 0:
        return None, None

    h, w = cfg["height"], cfg["width"]
    dx_px = cx - w // 2
    dy_px = cy - h // 2

    # 标定: 物块直径 D, 半径 r_px → k = D / (2*r_px) m/px
    k_m_per_px = cfg["block_diameter_m"] / (2 * r_px)

    # 摄像头反装 (x y 都取反)
    bx = -dx_px * k_m_per_px   # 车体 +X (右)
    by = -dy_px * k_m_per_px   # 车体 +Y (前)

    return bx, by


def block_to_world(car_x, car_y, car_yaw, bx, by):
    """车体坐标 → 世界坐标"""
    if bx is None or by is None:
        return None, None
    # 车体 yaw (compass) → math 角
    yaw_math = math.radians(90 - car_yaw)
    wx = car_x + bx * math.cos(yaw_math) - by * math.sin(yaw_math)
    wy = car_y + bx * math.sin(yaw_math) + by * math.cos(yaw_math)
    return wx, wy


def log_print(msg, cfg):
    if cfg["print_console"]:
        print(msg, flush=True)
    try:
        with open(cfg["log_file"], "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except:
        pass


def main():
    print("Block V25 - ROI 140 + 车体坐标 + 车位姿")
    print(f"  日志: {CONFIG['log_file']}")
    print("  q 退出")

    try:
        open(CONFIG["log_file"], "w").close()
    except:
        pass

    thresholds = load_thresholds(CONFIG["thresholds_file"])
    cam = USBCamera(CONFIG["usb_device"], CONFIG["width"],
                    CONFIG["height"], CONFIG["fps"])
    cam.start()

    cv2.namedWindow("Block V25", cv2.WINDOW_NORMAL)

    COLOR_BGR = {"黑": (0, 0, 0), "白": (200, 200, 200),
                 "红": (0, 0, 255), "绿": (0, 255, 0), "蓝": (255, 0, 0)}

    history = []
    last_log = None
    confirm_n = CONFIG["confirm_frames"]

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            color, cx, cy, r_px, info = recognize(frame, CONFIG, thresholds)
            pct = info.get("pct", {})
            best = info.get("best", "?")
            best_pct = info.get("best_pct", 0)
            centers = info.get("centers", {})
            radii = info.get("radii", {})

            history.append(color)
            if len(history) > confirm_n:
                history.pop(0)

            if (len(history) == confirm_n and
                len(set(history)) == 1 and
                history[0] is not None):
                confirmed = history[0]

                # ★ 算车体坐标
                bx, by = block_in_body_frame(cx, cy, r_px, CONFIG)

                # ★ 读车当前位姿
                car_x, car_y, car_yaw = None, None, None
                try:
                    pose = comm.get_pose(max_age=0.5)
                    if pose:
                        car_x, car_y, car_yaw = pose
                except:
                    pass

                # ★ 物块世界坐标
                wx, wy = block_to_world(car_x or 0, car_y or 0,
                                        car_yaw or 0, bx, by)

                # 位置变化判断
                need_log = False
                if last_log is None:
                    need_log = True
                elif last_log[0] != confirmed:
                    need_log = True
                elif (abs(last_log[1] - bx) > CONFIG["block_diameter_m"] or
                      abs(last_log[2] - by) > CONFIG["block_diameter_m"]):
                    need_log = True

                if need_log and bx is not None:
                    msg = (f"[3-FRAME] {time.strftime('%H:%M:%S')}  "
                           f"color={confirmed}  "
                           f"body=({bx:+.3f}, {by:+.3f}) m  "
                           f"r={r_px:.1f}px  pct={best_pct*100:.1f}%")
                    if car_x is not None:
                        msg += (f"  car=({car_x:.3f}, {car_y:.3f}, "
                                f"{car_yaw:.1f}°)")
                    if wx is not None:
                        msg += f"  world=({wx:.3f}, {wy:.3f})"
                    log_print(msg, CONFIG)
                    last_log = (confirmed, bx, by)

                history = []

            # === 显示 ===
            display = frame.copy()
            h, w = frame.shape[:2]
            cv2.circle(display, (w // 2, h // 2), CONFIG["roi_radius"],
                      (255, 0, 255), 2)

            for c, (ccx, ccy) in centers.items():
                bgr = COLOR_BGR.get(c, (255, 255, 255))
                cv2.circle(display, (ccx, ccy), 5, bgr, -1)
                cv2.putText(display, c, (ccx + 8, ccy - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 1)

            if color:
                bgr = COLOR_BGR.get(color, (255, 255, 255))
                cv2.rectangle(display, (0, 0), (w, 50), bgr, -1)
                cv2.putText(display, f"[{color}] {best_pct*100:.0f}%",
                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            else:
                cv2.rectangle(display, (0, 0), (w, 50), (50, 50, 50), -1)
                cv2.putText(display, f"NO  max={best} {best_pct*100:.0f}%",
                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

            cv2.putText(display,
                       f"history={history}  need={confirm_n}",
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (255, 255, 0), 1)

            cv2.imshow("Block V25", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print(f"\n日志: {CONFIG['log_file']}")


if __name__ == "__main__":
    main()
