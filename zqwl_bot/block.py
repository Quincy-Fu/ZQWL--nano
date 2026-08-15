#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
block.py - 物块颜色识别 (实时显示 + 5/10 帧众数)
"""

import cv2
import numpy as np
import json
import os
import re
import time
import threading
import comm


CONFIG = {
    "usb_device": 1,
    "usb_devices": [1, 0],
    "width": 640,
    "height": 480,
    "fps": 30,
    "thresholds_file": "hsv_thresholds.json",
    "roi_radius": 180,

    "vote_min_pct": 0.20,
    "motion_vote_min_pct": 0.08,
    "motion_chroma_min_pct": 0.04,
    "motion_black_min_pct": 0.16,
    "motion_white_min_pct": 0.82,
    "instant_recent_window_s": 0.80,
    "instant_chroma_min_pct": 0.02,
    "instant_black_min_pct": 0.12,
    "instant_white_min_pct": 0.90,

    "black_v_max": 75,
    "white_v_min": 150,
    "white_s_max": 80,

    "stable_frames": 10,
    "stable_timeout_s": 3.0,
    "stable_min_hits": 6,

    "fast_frames": 5,
    "fast_timeout_s": 1.5,
    "fast_min_hits": 3,

    "motion_recent_window_s": 0.25,
    "motion_recent_min_hits": 2,
    "motion_fallback_frames": 2,
    "motion_fallback_timeout_s": 0.18,
    "motion_fallback_min_hits": 1,

    "last_known_max_age_s": 3.0,
    "latest_frame_max_age_s": 0.30,
    "device_retry_cooldown_s": 20.0,

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
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"无法打开USB摄像头 device={self.device}")

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
_cam_lock = threading.RLock()
_bad_devices = {}
_last_known = {"result": None, "time": 0.0}
_latest_frame = None           # ★ viewer 线程共享的最近帧
_latest_frame_time = 0.0
_frame_lock = threading.Lock()
_recent_color_samples = []     # 圆弧运动中复用 viewer 连续识别结果: (time, color, pct)
_recent_color_lock = threading.Lock()
_viewer_started = False
_viewer_thread = None
_viewer_stop = threading.Event()
_status_text = "等待识别"
_status_lock = threading.Lock()


def _parse_device_list(value):
    """解析 USB 设备号列表，支持 "1,0" / "1 0" 这类写法。"""
    if value is None or value == "":
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    items = re.split(r"[,;\s]+", str(value).strip())
    return [int(v) for v in items if v != ""]


def configure_usb(device=None, devices=None):
    """配置 USB 摄像头候选顺序；默认优先 1，再回退 0。"""
    ordered = []
    if device is not None:
        ordered.extend(_parse_device_list(device))
    if devices is not None:
        ordered.extend(_parse_device_list(devices))
    ordered.extend([1, 0])

    unique = []
    for dev in ordered:
        if dev not in unique:
            unique.append(dev)
    CONFIG["usb_device"] = unique[0]
    CONFIG["usb_devices"] = unique
    return unique


def apply_usb_env(prefix="ZQWL_BLOCK"):
    """从环境变量覆盖 USB 设备号；专用变量优先，通用变量兜底。"""
    device = os.environ.get(f"{prefix}_USB_DEVICE", os.environ.get("ZQWL_USB_DEVICE"))
    devices = os.environ.get(f"{prefix}_USB_DEVICES", os.environ.get("ZQWL_USB_DEVICES"))
    if device is not None or devices is not None:
        return configure_usb(device=device, devices=devices)
    return configure_usb(CONFIG.get("usb_device", 1), CONFIG.get("usb_devices", [1, 0]))


def set_status(text):
    """设置实时画面左上角状态文字。"""
    global _status_text
    with _status_lock:
        _status_text = str(text)


def _get_status():
    with _status_lock:
        return _status_text


def _ordered_devices():
    """生成 USB 摄像头尝试顺序；实车优先 1，0 只作为回退。"""
    return apply_usb_env()


def _available_devices():
    """过滤近期读帧失败的设备；如果全被过滤，则清空冷却重新尝试。"""
    now = time.time()
    ordered = _ordered_devices()
    available = []
    for dev in ordered:
        until = _bad_devices.get(dev, 0.0)
        if until <= now:
            _bad_devices.pop(dev, None)
            available.append(dev)
    if not available:
        _bad_devices.clear()
        available = ordered
    return available


def _clear_frame_cache_locked():
    global _latest_frame, _latest_frame_time
    with _frame_lock:
        _latest_frame = None
        _latest_frame_time = 0.0
    with _recent_color_lock:
        _recent_color_samples.clear()
    _last_known["result"] = None
    _last_known["time"] = 0.0


def _drop_cam_locked(cam, reason):
    """丢弃当前摄像头，下一次读帧会尝试下一个 USB 设备。"""
    if cam is None:
        return
    print(f"[block] USB摄像头 device={cam.device} 异常: {reason}，切换下一个设备", flush=True)
    _bad_devices[cam.device] = time.time() + CONFIG["device_retry_cooldown_s"]
    if _cam_cache["cam"] is cam:
        try:
            cam.stop()
        finally:
            _cam_cache["cam"] = None
            _clear_frame_cache_locked()


def _get_cam():
    with _cam_lock:
        if _cam_cache["cam"] is None:
            ordered_devices = _available_devices()
            last_error = None
            for dev in ordered_devices:
                cam = USBCamera(dev, CONFIG["width"], CONFIG["height"], CONFIG["fps"])
                try:
                    cam.start()
                except RuntimeError as e:
                    last_error = e
                    continue
                CONFIG["usb_device"] = dev
                _cam_cache["cam"] = cam
                print(f"[block] USB摄像头已打开: device={dev}", flush=True)
                break

            if _cam_cache["cam"] is None:
                raise RuntimeError(f"无法打开USB摄像头，已尝试 {ordered_devices}: {last_error}")
        return _cam_cache["cam"]


def _cache_frame(frame):
    """缓存最新有效帧，识别线程只使用短时间内的新帧。"""
    global _latest_frame, _latest_frame_time
    with _frame_lock:
        _latest_frame = frame.copy()
        _latest_frame_time = time.time()


def _read_camera_frame():
    """统一摄像头读帧入口；读失败时自动切换 0/1 设备。"""
    attempts = max(1, len(_ordered_devices()))
    for _ in range(attempts):
        with _cam_lock:
            cam = _get_cam()
            frame = cam.read()
            if frame is not None:
                _cache_frame(frame)
                return frame
            _drop_cam_locked(cam, "读帧失败")
        time.sleep(0.05)
    return None


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
    if CONFIG["show_window"]:
        cv2.namedWindow(CONFIG["show_window_name"], cv2.WINDOW_NORMAL)

    # 启动时如果阈值文件没就绪
    thresholds = load_thresholds(CONFIG["thresholds_file"])

    frame_interval = 1.0 / CONFIG["show_fps"]
    h, w = CONFIG["height"], CONFIG["width"]
    COLOR_BGR = {"黑": (0, 0, 0), "白": (200, 200, 200),
                 "红": (0, 0, 255), "绿": (0, 255, 0), "蓝": (255, 0, 0)}

    while not _viewer_stop.is_set():
        t0 = time.time()
        frame = _read_camera_frame()
        if frame is not None:
            # 单帧识别结果持续写入缓存；显示窗口只是消费这个结果。
            color, pcts = _single_frame_color(
                frame,
                thresholds,
                vote_min_pct=CONFIG["motion_vote_min_pct"],
            )
            _record_color_sample(color, pcts)

            # 实时显示
            if CONFIG["show_window"]:
                display = frame.copy()

                # 画 ROI
                cv2.circle(display, (w // 2, h // 2), CONFIG["roi_radius"],
                          (255, 0, 255), 2)

                status = _get_status()
                cv2.rectangle(display, (0, h - 38), (w, h), (30, 30, 30), -1)
                cv2.putText(display, status, (10, h - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 单帧识别结果 (快速反馈)
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
        if CONFIG["show_window"] and (cv2.waitKey(1) & 0xFF == ord('q')):
            _viewer_stop.set()
            break


def start_viewer():
    """启动实时显示线程 (一次性)"""
    global _viewer_started, _viewer_thread
    if _viewer_thread is not None and _viewer_thread.is_alive():
        return
    _viewer_stop.clear()
    th = threading.Thread(target=_viewer_loop, daemon=True)
    th.start()
    _viewer_thread = th
    _viewer_started = True
    print("[block] 实时显示线程已启动")


def _read_latest_frame():
    """从 viewer 线程的缓存拿最近帧 (线程安全)"""
    with _frame_lock:
        if _latest_frame is None:
            return None
        if (time.time() - _latest_frame_time) > CONFIG["latest_frame_max_age_s"]:
            return None
        return _latest_frame.copy()


def clear_recent_colors():
    """清空最近颜色缓存，避免上一个物块的结果串到下一个点。"""
    with _recent_color_lock:
        _recent_color_samples.clear()


def clear_recognition_cache():
    """清空颜色识别缓存；用于机械臂/车体调整后重新从当前画面开始判断。"""
    clear_recent_colors()
    _last_known["result"] = None
    _last_known["time"] = 0.0


def _record_color_sample(color, pct=None):
    """记录 viewer 每帧颜色结果，供圆弧运动中快速读取。"""
    now = time.time()
    keep_s = max(CONFIG["motion_recent_window_s"], CONFIG["instant_recent_window_s"], 1.0)
    with _recent_color_lock:
        _recent_color_samples.append((now, color, pct or {}))
        cutoff = now - keep_s
        while _recent_color_samples and _recent_color_samples[0][0] < cutoff:
            _recent_color_samples.pop(0)


def _motion_color_from_pct(pct, exclude_colors=None):
    """运动中从 ROI 占比判断物块颜色，避免白/黑背景按最大面积误胜出。"""
    exclude = set(exclude_colors or [])

    chroma = [c for c in ("红", "绿", "蓝") if c not in exclude]
    if chroma:
        best_chroma = max(chroma, key=lambda c: pct.get(c, 0.0))
        if pct.get(best_chroma, 0.0) >= CONFIG["motion_chroma_min_pct"]:
            return best_chroma

    if "黑" not in exclude and pct.get("黑", 0.0) >= CONFIG["motion_black_min_pct"]:
        return "黑"
    if "白" not in exclude and pct.get("白", 0.0) >= CONFIG["motion_white_min_pct"]:
        return "白"
    return None


def _instant_color_from_sample(sample_color, pct, exclude_colors=None):
    """从 viewer 已显示过的一帧里取色；彩色优先，白色背景最后才接受。"""
    exclude = set(exclude_colors or [])

    chroma = [c for c in ("红", "绿", "蓝") if c not in exclude]
    if chroma:
        best_chroma = max(chroma, key=lambda c: pct.get(c, 0.0))
        if pct.get(best_chroma, 0.0) >= CONFIG["instant_chroma_min_pct"]:
            return best_chroma

    if "黑" not in exclude and pct.get("黑", 0.0) >= CONFIG["instant_black_min_pct"]:
        return "黑"

    if sample_color in ("红", "绿", "蓝") and sample_color not in exclude:
        return sample_color
    if sample_color == "黑" and sample_color not in exclude:
        return "黑"

    # 白色最容易被地面/反光误触发，只有没有明显彩色/黑色证据时才接受。
    max_chroma = max((pct.get(c, 0.0) for c in ("红", "绿", "蓝")), default=0.0)
    if (
        "白" not in exclude
        and sample_color == "白"
        and pct.get("白", 0.0) >= CONFIG["instant_white_min_pct"]
        and max_chroma < CONFIG["instant_chroma_min_pct"]
        and pct.get("黑", 0.0) < CONFIG["instant_black_min_pct"]
    ):
        return "白"

    return None


def get_recent_display_color(window_s=None, exclude_colors=None):
    """读取 viewer 最近显示过的一次有效颜色；用于“看到一次就算到位”的场景。"""
    if window_s is None:
        window_s = CONFIG["instant_recent_window_s"]
    now = time.time()
    with _recent_color_lock:
        samples = [(t, c, pct) for t, c, pct in _recent_color_samples
                   if (now - t) <= window_s]
    for _t, color, pct in reversed(samples):
        instant_color = _instant_color_from_sample(color, pct or {}, exclude_colors=exclude_colors)
        if instant_color is not None:
            return instant_color
    return None


def wait_for_display_color(timeout_s=0.5, window_s=None, exclude_colors=None):
    """在 timeout 内只要 viewer 显示过一次有效颜色就返回。"""
    start = time.time()
    while (time.time() - start) <= timeout_s:
        color = get_recent_display_color(window_s=window_s, exclude_colors=exclude_colors)
        if color is not None:
            return color
        time.sleep(0.02)
    return None


def get_recent_motion_color(window_s=None, min_hits=None, exclude_colors=None):
    """读取最近连续识别缓存；不使用 last_known 兜底，适合运动中触发点。"""
    if window_s is None:
        window_s = CONFIG["motion_recent_window_s"]
    if min_hits is None:
        min_hits = CONFIG["motion_recent_min_hits"]
    now = time.time()
    votes = {"黑": 0, "白": 0, "红": 0, "绿": 0, "蓝": 0}
    with _recent_color_lock:
        samples = [(t, pct) for t, _c, pct in _recent_color_samples
                   if (now - t) <= window_s]
    for _, pct in samples:
        color = _motion_color_from_pct(pct or {}, exclude_colors=exclude_colors)
        if color in votes:
            votes[color] += 1
    winner = max(votes, key=votes.get)
    if votes[winner] >= min_hits:
        return winner
    return None


def recent_motion_debug(window_s=None, exclude_colors=None):
    """返回最近运动识别窗口的统计，判断是没赶上还是 ROI 比例不够。"""
    if window_s is None:
        window_s = CONFIG["motion_recent_window_s"]
    now = time.time()
    votes = {"黑": 0, "白": 0, "红": 0, "绿": 0, "蓝": 0, None: 0}
    best_pct = {"黑": 0.0, "白": 0.0, "红": 0.0, "绿": 0.0, "蓝": 0.0}
    ages = []
    with _recent_color_lock:
        samples = [(t, c, pct) for t, c, pct in _recent_color_samples
                   if (now - t) <= window_s]
    for t, _color, pct in samples:
        color = _motion_color_from_pct(pct or {}, exclude_colors=exclude_colors)
        votes[color if color else None] = votes.get(color if color else None, 0) + 1
        ages.append(now - t)
        for c, v in (pct or {}).items():
            if c in best_pct and v > best_pct[c]:
                best_pct[c] = float(v)
    return {
        "samples": len(samples),
        "newest_age": min(ages) if ages else None,
        "oldest_age": max(ages) if ages else None,
        "votes": votes,
        "best_pct": best_pct,
    }


# ============ 单帧识别 (供 recognizer + viewer 共享) ============
def _single_frame_color(frame, thresholds, vote_min_pct=None):
    """单帧识别, 返回 (color, pct_dict) 或 (None, {})"""
    if vote_min_pct is None:
        vote_min_pct = CONFIG["vote_min_pct"]
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
    # 不再用“整个 ROI 最大占比”直接判色；白色地面/黑色边缘面积通常更大。
    # 先按物块颜色规则判断，彩色小块优先，白/黑必须达到更高占比才算。
    object_color = _motion_color_from_pct(pct)
    if object_color is not None:
        return object_color, pct

    best = max(pct, key=lambda k: pct[k])
    if best in ("红", "绿", "蓝") and pct[best] >= vote_min_pct:
        return best, pct
    return None, pct


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
            # 没启动 viewer 或缓存过旧时，走统一读帧入口；避免多线程同时读同一个 VideoCapture。
            frame = _read_camera_frame()
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


def recognize_no_fallback(frames=None, timeout=None, min_hits=None):
    """快速识别但不使用 last_known 兜底，避免运动中把上一个物块串到下一个槽位。"""
    start_viewer()

    if frames is None:
        frames = CONFIG["fast_frames"]
    if timeout is None:
        timeout = CONFIG["fast_timeout_s"]
    if min_hits is None:
        min_hits = max(1, min(int(frames), CONFIG["fast_min_hits"]))

    result, _hits = _recognize_multi(frames, min_hits, timeout)
    return result


def recognize_motion(window_s=None, min_hits=None, exclude_colors=None):
    """圆弧运动中取色：优先用 viewer 连续缓存，失败再极短补采。"""
    start_viewer()
    color = get_recent_motion_color(window_s=window_s,
                                    min_hits=min_hits,
                                    exclude_colors=exclude_colors)
    if color is not None:
        return color

    thresholds = load_thresholds(CONFIG["thresholds_file"])
    votes = {"黑": 0, "白": 0, "红": 0, "绿": 0, "蓝": 0}
    start = time.time()
    count = 0
    while count < CONFIG["motion_fallback_frames"] and \
          (time.time() - start) < CONFIG["motion_fallback_timeout_s"]:
        frame = _read_latest_frame()
        if frame is None:
            frame = _read_camera_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        _plain_color, pct = _single_frame_color(
            frame,
            thresholds,
            vote_min_pct=CONFIG["motion_vote_min_pct"],
        )
        motion_color = _motion_color_from_pct(pct, exclude_colors=exclude_colors)
        if motion_color in votes:
            votes[motion_color] += 1
        count += 1

    if count == 0:
        return None
    winner = max(votes, key=votes.get)
    if votes[winner] >= CONFIG["motion_fallback_min_hits"]:
        return winner
    return None


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
        debug = recent_motion_debug(window_s=0.5)
        print(
            f"[block] 稳定识别失败: samples={debug['samples']}, "
            f"votes={debug['votes']}, best_pct={debug['best_pct']}",
            flush=True,
        )
        return None

    _last_known["result"] = result
    _last_known["time"] = time.time()
    return result


def close():
    """关闭摄像头 (main 退出时调用)"""
    global _viewer_started, _viewer_thread
    _viewer_stop.set()
    th = _viewer_thread
    if th is not None and th.is_alive():
        th.join(timeout=2.0)
        if th.is_alive():
            print("[block] 实时显示线程仍在读帧，继续执行摄像头释放", flush=True)

    with _cam_lock:
        if _cam_cache["cam"] is not None:
            _cam_cache["cam"].stop()
            _cam_cache["cam"] = None
        _clear_frame_cache_locked()

    _viewer_started = False
    _viewer_thread = None
    try:
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"[block] destroyAllWindows failed: {e}", flush=True)
