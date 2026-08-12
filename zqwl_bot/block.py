#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
block.py - 物块颜色识别 (实时显示 + 5/10 帧众数)
"""

import cv2
import numpy as np
import json
import os
import time
import threading
import comm


CONFIG = {
    "usb_device": 1,
    "width": 640,
    "height": 480,
    "fps": 30,
    "thresholds_file": "hsv_thresholds.json",
    "roi_radius": 200,

    "vote_min_pct": 0.20,

    "black_v_max": 60,
    "white_v_min": 150,
    "white_s_max": 80,

    "stable_frames": 10,
    "stable_timeout_s": 3.0,
    "stable_min_hits": 6,

    "fast_frames": 5,
    "fast_timeout_s": 1.5,
    "fast_min_hits": 3,

    "last_known_max_age_s": 3.0,

    # ★ 实时显示
    "show_window": True,            # 实时显示开关
    "show_window_name": "Block Live",
    "show_draw_pct": True,          # 画面上画占比条
    "show_fps": 25,                 # 显示限速 fps
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


# ============ 摄像头单例 ============
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
            self.cap = None


_cam_cache = {"cam": None}
_last_known = {"result": None, "time": 0.0}
_latest_frame = None           # ★ viewer 线程共享的最近帧
_frame_lock = threading.Lock()
_viewer_started = False


def _get_cam():
    if _cam_cache["cam"] is None:
        cam = USBCamera(CONFIG["usb_device"], CONFIG["width"],
                       CONFIG["height"], CONFIG["fps"])
        cam.start()
        _cam_cache["cam"] = cam
    return _cam_cache["cam"]


# ============ ★ 实时显示后台线程 ============
def _draw_pct_bars(display, h, w, pcts, best, color):
    """画面右下角画 5 条占比条"""
    if not pcts:
        return
    y = 70
    COLOR_BGR = {"黑": (0, 0, 0), "白": (200, 200, 200),
                 "红": (0, 0, 255), "绿": (0, 255, 0), "蓝": (255, 0, 0)}
    for c in ["黑", "白", "红", "绿", "蓝"]:
        p = pcts.get(c, 0)
        bgr = COLOR_BGR.get(c, (255, 255, 255))
        cv2.rectangle(display, (w - 160, y - 15), (w, y + 5), bgr, -1)
        if c == best:
            cv2.rectangle(display, (w - 160, y - 15), (w, y + 5), (0, 255, 255), 2)
        cv2.putText(display, f"{c} {p*100:4.1f}%",
                   (w - 150, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   (0, 0, 0), 1)
        y += 22


def _viewer_loop():
    """后台线程: 持续 read 摄像头 + 显示 (实时画面)"""
    cam = _get_cam()
    cv2.namedWindow(CONFIG["show_window_name"], cv2.WINDOW_NORMAL)

    # 启动时如果阈值文件没就绪
    thresholds = load_thresholds(CONFIG["thresholds_file"])

    frame_interval = 1.0 / CONFIG["show_fps"]
    h, w = CONFIG["height"], CONFIG["width"]
    COLOR_BGR = {"黑": (0, 0, 0), "白": (200, 200, 200),
                 "红": (0, 0, 255), "绿": (0, 255, 0), "蓝": (255, 0, 0)}

    while True:
        t0 = time.time()
        frame = cam.read()
        if frame is not None:
            # ★ 缓存最近帧 (给 recognizer 用)
            global _latest_frame
            with _frame_lock:
                _latest_frame = frame.copy()

            # 实时显示
            if CONFIG["show_window"]:
                display = frame.copy()

                # 画 ROI
                cv2.circle(display, (w // 2, h // 2), CONFIG["roi_radius"],
                          (255, 0, 255), 2)

                # 单帧识别结果 (快速反馈)
                color, pcts = _single_frame_color(frame, thresholds)
                if color:
                    bgr = COLOR_BGR.get(color, (255, 255, 255))
                    cv2.rectangle(display, (0, 0), (w, 50), bgr, -1)
                    cv2.putText(display, f"[{color}]", (10, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    cv2.rectangle(display, (0, 0), (w, 50), (50, 50, 50), -1)
                    cv2.putText(display, "NO BLOCK", (10, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if CONFIG["show_draw_pct"]:
                    _draw_pct_bars(display, h, w, pcts,
                                    max(pcts, key=pcts.get) if pcts else None,
                                    color)

                cv2.imshow(CONFIG["show_window_name"], display)

        # 限速
        elapsed = time.time() - t0
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        # waitKey 让窗口响应
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def start_viewer():
    """启动实时显示线程 (一次性)"""
    global _viewer_started
    if _viewer_started:
        return
    th = threading.Thread(target=_viewer_loop, daemon=True)
    th.start()
    _viewer_started = True
    print("[block] 实时显示线程已启动")


def _read_latest_frame():
    """从 viewer 线程的缓存拿最近帧 (线程安全)"""
    with _frame_lock:
        return _latest_frame


# ============ 单帧识别 (供 recognizer + viewer 共享) ============
def _single_frame_color(frame, thresholds):
    """单帧识别, 返回 (color, pct_dict) 或 (None, {})"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    S = hsv[:, :, 1]

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (w // 2, h // 2), CONFIG["roi_radius"], 255, -1)
    roi_total = int(np.sum(roi_mask == 255))
    if roi_total == 0:
        return None, {}

    color_masks = {}

    black_mask = (V < CONFIG["black_v_max"]).astype(np.uint8) * 255
    color_masks["黑"] = cv2.bitwise_and(black_mask, roi_mask)

    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    color_masks["红"] = cv2.bitwise_and(cv2.bitwise_or(red1, red2), roi_mask)

    color_masks["绿"] = cv2.bitwise_and(
        cv2.inRange(hsv, (30, 40, 40), (90, 255, 255)), roi_mask)

    color_masks["蓝"] = cv2.bitwise_and(
        cv2.inRange(hsv, (95, 80, 80), (135, 255, 255)), roi_mask)

    white_mask = ((V > CONFIG["white_v_min"]) &
                  (S < CONFIG["white_s_max"])).astype(np.uint8) * 255
    color_masks["白"] = cv2.bitwise_and(white_mask, roi_mask)

    pct = {c: float(np.sum(m == 255)) / roi_total
           for c, m in color_masks.items()}
    best = max(pct, key=lambda k: pct[k])
    if pct[best] < CONFIG["vote_min_pct"]:
        return None, pct
    return best, pct


# ============ 公共 API ============
def _recognize_multi(frames, min_hits, timeout_s):
    """多帧投票, 优先用 viewer 缓存帧, 没启动 viewer 则自己读"""
    thresholds = load_thresholds(CONFIG["thresholds_file"])
    votes = {"黑": 0, "白": 0, "红": 0, "绿": 0, "蓝": 0, None: 0}
    start = time.time()
    count = 0

    while count < frames and (time.time() - start) < timeout_s:
        # 优先用 viewer 线程的缓存
        frame = _read_latest_frame()
        if frame is None:
            # 没启动 viewer 或还没读到, 自己读
            cam = _get_cam()
            frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        color, _ = _single_frame_color(frame, thresholds)
        votes[color if color else None] += 1
        count += 1

    if count == 0:
        return None, 0

    color_votes = {c: v for c, v in votes.items() if c is not None}
    if not color_votes:
        return None, 0

    winner = max(color_votes, key=color_votes.get)
    if color_votes[winner] < min_hits:
        return None, color_votes[winner]
    return winner, color_votes[winner]


def recognize(frames=None, timeout=None):
    """快速识别: 5帧众数, 1.5s 超时"""
    # ★ 第一次调用时启动 viewer (实时画面)
    start_viewer()

    if frames is None:
        frames = CONFIG["fast_frames"]
    if timeout is None:
        timeout = CONFIG["fast_timeout_s"]
    min_hits = CONFIG["fast_min_hits"]

    result, hits = _recognize_multi(frames, min_hits, timeout)

    if result is None:
        if _last_known["result"] is not None:
            age = time.time() - _last_known["time"]
            if age < CONFIG["last_known_max_age_s"]:
                return _last_known["result"]
        return None

    _last_known["result"] = result
    _last_known["time"] = time.time()
    return result


def recognize_stable(frames=None, timeout=None):
    """稳定识别: 10帧众数, 3s 超时 (车停下后用)"""
    start_viewer()

    if frames is None:
        frames = CONFIG["stable_frames"]
    if timeout is None:
        timeout = CONFIG["stable_timeout_s"]
    min_hits = CONFIG["stable_min_hits"]

    result, hits = _recognize_multi(frames, min_hits, timeout)

    if result is None:
        return None

    _last_known["result"] = result
    _last_known["time"] = time.time()
    return result


def close():
    """关闭摄像头 (main 退出时调用)"""
    if _cam_cache["cam"] is not None:
        _cam_cache["cam"].stop()
        _cam_cache["cam"] = None
    cv2.destroyAllWindows()
