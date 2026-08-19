#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
block.py - 物块颜色识别 (实时显示 + 5/10 帧众数)
"""

import json
import os
import re
import time
import threading

# 必须在 cv2 首次导入前设置；否则 V4L2 异常读帧会沿用 OpenCV 默认约 10 秒 select 超时。
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")
os.environ.setdefault("OPENCV_VIDEOIO_V4L_READ_ATTEMPTS", "1")

import cv2
import numpy as np
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
    "motion_bw_margin_pct": 0.06,
    "arc_chroma_min_pct": 0.015,
    "instant_recent_window_s": 0.80,
    "instant_chroma_min_pct": 0.02,
    "instant_black_min_pct": 0.12,
    "instant_white_min_pct": 0.90,
    "instant_bw_margin_pct": 0.05,

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
    "arc_recent_window_s": 3.0,
    "arc_black_hit_min_pct": 0.13,
    "arc_black_peak_min_pct": 0.17,
    "arc_black_avg_min_pct": 0.10,
    "arc_black_min_hits": 2,
    "motion_recent_min_hits": 2,
    "motion_fallback_frames": 2,
    "motion_fallback_timeout_s": 0.18,
    "motion_fallback_min_hits": 1,

    "last_known_max_age_s": 3.0,
    "latest_frame_max_age_s": 0.30,
    "usb_open_verify_frames": 1,
    "usb_open_verify_timeout_s": 1.2,
    "usb_black_mean_min": 5.0,
    "usb_dark_uniform_mean_max": 25.0,
    "usb_dark_uniform_std_min": 1.0,
    "usb_ready_timeout_s": 3.0,
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

        valid = None
        deadline = time.time() + float(CONFIG.get("usb_open_verify_timeout_s", 1.2))
        for _ in range(int(CONFIG.get("usb_open_verify_frames", 1))):
            if time.time() >= deadline:
                break
            ret, frame = self.cap.read()
            if ret and _frame_is_valid(frame):
                valid = frame
                break
            time.sleep(0.03)
        if valid is None:
            self.stop()
            raise RuntimeError(f"USB摄像头 device={self.device} 打开后无有效画面/疑似黑屏")

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
_camera_health = {"ok": False, "last_ok_time": 0.0, "last_error": "not started", "last_device": None}
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
_thresholds_cache = {"path": None, "mtime": None, "value": None}


def _frame_is_valid(frame):
    """判断 USB 帧是否有效；防止 isOpened=True 但实际黑屏/空帧。"""
    if frame is None:
        return False
    if getattr(frame, "size", 0) <= 0:
        return False
    try:
        mean_v = float(np.mean(frame))
        std_v = float(np.std(frame))
    except Exception:
        return False
    if mean_v < float(CONFIG.get("usb_black_mean_min", 5.0)):
        return False
    # 黑屏有时不是全 0，而是低亮度、低纹理的均匀帧；这种帧不能进入识别缓存。
    if (
        mean_v < float(CONFIG.get("usb_dark_uniform_mean_max", 25.0))
        and std_v < float(CONFIG.get("usb_dark_uniform_std_min", 1.0))
    ):
        return False
    return True


def _set_camera_health(ok, reason=None, device=None):
    """记录摄像头健康状态，供主流程判断是否允许继续运动。"""
    _camera_health["ok"] = bool(ok)
    if ok:
        _camera_health["last_ok_time"] = time.time()
        _camera_health["last_error"] = None
    elif reason is not None:
        _camera_health["last_error"] = str(reason)
    if device is not None:
        _camera_health["last_device"] = device


def camera_health():
    """返回 USB 摄像头最近状态；仅用于日志和主流程准入检查。"""
    now = time.time()
    info = dict(_camera_health)
    last_ok = float(info.get("last_ok_time") or 0.0)
    info["last_ok_age_s"] = now - last_ok if last_ok > 0.0 else None
    return info


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


def _load_thresholds_cached():
    """缓存 HSV 阈值，避免圆弧窗口主动采样时反复读文件。"""
    path = CONFIG["thresholds_file"]
    try:
        mtime = os.path.getmtime(path) if os.path.exists(path) else None
    except OSError:
        mtime = None
    if (
        _thresholds_cache["value"] is None
        or _thresholds_cache["path"] != path
        or _thresholds_cache["mtime"] != mtime
    ):
        _thresholds_cache["path"] = path
        _thresholds_cache["mtime"] = mtime
        _thresholds_cache["value"] = load_thresholds(path)
    return _thresholds_cache["value"]


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
    _set_camera_health(False, reason=reason, device=cam.device)
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
                    _set_camera_health(False, reason=e, device=dev)
                    _bad_devices[dev] = time.time() + CONFIG["device_retry_cooldown_s"]
                    continue
                CONFIG["usb_device"] = dev
                _cam_cache["cam"] = cam
                print(f"[block] USB摄像头已打开: device={dev}", flush=True)
                break

            if _cam_cache["cam"] is None:
                _set_camera_health(False, reason=last_error)
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
        try:
            with _cam_lock:
                cam = _get_cam()
                frame = cam.read()
                if _frame_is_valid(frame):
                    _cache_frame(frame)
                    _set_camera_health(True, device=cam.device)
                    return frame
                _drop_cam_locked(cam, "读帧失败或黑屏")
        except RuntimeError as e:
            _set_camera_health(False, reason=e)
        time.sleep(0.05)
    return None


def has_fresh_frame(max_age_s=None):
    """判断当前是否有新鲜有效帧。"""
    if max_age_s is None:
        max_age_s = CONFIG["latest_frame_max_age_s"]
    with _frame_lock:
        if _latest_frame is None:
            return False
        return (time.time() - _latest_frame_time) <= float(max_age_s)


def _viewer_alive():
    """判断后台取流线程是否仍在运行。"""
    return _viewer_thread is not None and _viewer_thread.is_alive()


def wait_until_ready(timeout_s=None):
    """等待 USB 摄像头拿到新鲜有效帧；失败时返回 False，不让主流程盲走。"""
    if timeout_s is None:
        timeout_s = CONFIG.get("usb_ready_timeout_s", 3.0)
    start_viewer()
    start = time.time()
    while (time.time() - start) <= float(timeout_s):
        if has_fresh_frame():
            return True
        if _read_camera_frame() is not None:
            return True
        time.sleep(0.05)
    info = camera_health()
    print(f"[block] USB摄像头未就绪: {info}", flush=True)
    return False


def ensure_ready_for_use(label="block", timeout_s=0.7, restart_timeout_s=0.9,
                         max_age_s=0.6, verbose=True):
    """正式使用前确认 USB 摄像头可立即取帧；失败则快速重启一次。

    这个函数不依赖 RUN 前预检结果：预检只负责提前热机，正式阶段仍必须
    现场确认，避免摄像头线程已死、帧已过期或设备号切换后仍继续盲等。
    """
    start_viewer()
    if _viewer_alive() and has_fresh_frame(max_age_s=max_age_s):
        return True

    start = time.time()
    while (time.time() - start) <= float(timeout_s):
        if _read_camera_frame() is not None:
            return True
        time.sleep(0.03)

    if verbose:
        print(f"[block] {label} 使用前无新鲜帧，快速重启 USB 摄像头: {camera_health()}", flush=True)
    close()
    time.sleep(0.08)
    start_viewer()

    start = time.time()
    while (time.time() - start) <= float(restart_timeout_s):
        if _read_camera_frame() is not None:
            if verbose:
                print(f"[block] {label} USB 摄像头重启后可用: {camera_health()}", flush=True)
            return True
        time.sleep(0.03)

    if verbose:
        print(f"[block] {label} USB 摄像头重启后仍不可用: {camera_health()}", flush=True)
    return False


def keep_warm(timeout_s=0.25):
    """RUN 等待期间轻量保活；不输出大量日志，不长时间阻塞。"""
    return ensure_ready_for_use(
        label="RUN前保活",
        timeout_s=timeout_s,
        restart_timeout_s=max(timeout_s, 0.35),
        max_age_s=0.8,
        verbose=False,
    )



def sample_motion_frame():
    """主动读取并记录一帧运动判色样本；用于圆弧窗口，避免只依赖后台 viewer 缓存。"""
    start_viewer()
    frame = _read_camera_frame()
    if frame is None:
        frame = _read_latest_frame()
    if frame is None:
        return False
    color, pcts = _single_frame_color(
        frame,
        _load_thresholds_cached(),
        vote_min_pct=CONFIG["motion_vote_min_pct"],
    )
    _record_color_sample(color, pcts)
    return True


def _show_camera_error_frame(reason):
    """显示摄像头异常提示，避免把黑窗口误认为正常画面。"""
    if not CONFIG["show_window"]:
        return
    h, w = CONFIG["height"], CONFIG["width"]
    display = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(display, "USB CAMERA NO VALID FRAME", (25, h // 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    text = str(reason or "retrying 1/0")[:70]
    cv2.putText(display, text, (25, h // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.imshow(CONFIG["show_window_name"], display)


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
    thresholds = _load_thresholds_cached()

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
        else:
            _show_camera_error_frame(camera_health().get("last_error"))

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
    keep_s = max(
        CONFIG["motion_recent_window_s"],
        CONFIG["instant_recent_window_s"],
        CONFIG.get("arc_recent_window_s", 1.0),
        1.0,
    )
    with _recent_color_lock:
        _recent_color_samples.append((now, color, pct or {}))
        cutoff = now - keep_s
        while _recent_color_samples and _recent_color_samples[0][0] < cutoff:
            _recent_color_samples.pop(0)


def analyze_arc_interval_color(start_time=None, end_time=None, window_s=None,
                               exclude_colors=None, allowed_colors=None):
    """分析圆弧时间窗口颜色证据，返回判色和黑/白证据。

    白色地面会让白色占比长期偏大，所以圆弧中不能只看白色最大。
    彩色按短暂出现判定；黑色看窗口内黑色面积峰值、均值和命中帧数；
    白色通常留给“没有彩色/黑色证据”或上层排除法处理。
    """
    exclude = set(exclude_colors or [])
    if allowed_colors is None:
        allowed = {"黑", "白", "红", "绿", "蓝"} - exclude
    else:
        allowed = set(allowed_colors) - exclude
    now = time.time()
    if end_time is None:
        end_time = now
    if start_time is None:
        if window_s is None:
            window_s = CONFIG["arc_recent_window_s"]
        start_time = end_time - float(window_s)

    with _recent_color_lock:
        samples = [(t, c, pct) for t, c, pct in _recent_color_samples
                   if start_time <= t <= end_time]
    result = {
        "color": None,
        "samples": len(samples),
        "max_pct": {c: 0.0 for c in ("黑", "白", "红", "绿", "蓝")},
        "avg_pct": {c: 0.0 for c in ("黑", "白", "红", "绿", "蓝")},
        "hits": {c: 0 for c in ("黑", "白", "红", "绿", "蓝")},
        "black_score": 0.0,
        "black_confirmed": False,
        "allowed_colors": sorted(allowed),
    }
    if not samples:
        return result

    max_pct = result["max_pct"]
    avg_pct = result["avg_pct"]
    hits = result["hits"]
    chroma_min = float(CONFIG.get("arc_chroma_min_pct", CONFIG["motion_chroma_min_pct"]))
    black_hit_min = float(CONFIG.get("arc_black_hit_min_pct", CONFIG["motion_black_min_pct"]))
    for _t, _color, pct in samples:
        pct = pct or {}
        for c in max_pct:
            v = float(pct.get(c, 0.0))
            avg_pct[c] += v
            if v > max_pct[c]:
                max_pct[c] = v
        for c in ("红", "绿", "蓝"):
            if c in allowed and float(pct.get(c, 0.0)) >= chroma_min:
                hits[c] += 1
        if "黑" in allowed and float(pct.get("黑", 0.0)) >= black_hit_min:
            hits["黑"] += 1

    for c in avg_pct:
        avg_pct[c] /= float(len(samples))

    chroma = [c for c in ("红", "绿", "蓝") if c in allowed and hits[c] > 0]
    if chroma:
        result["color"] = max(chroma, key=lambda c: (hits[c], max_pct[c]))
        return result

    black_peak_min = float(CONFIG.get("arc_black_peak_min_pct", CONFIG["motion_black_min_pct"]))
    black_avg_min = float(CONFIG.get("arc_black_avg_min_pct", 0.10))
    black_min_hits = int(CONFIG.get("arc_black_min_hits", 2))
    black_score = max_pct["黑"] + 0.6 * avg_pct["黑"] + 0.015 * hits["黑"]
    result["black_score"] = float(black_score)
    result["black_confirmed"] = (
        "黑" in allowed
        and (
            (hits["黑"] >= black_min_hits and max_pct["黑"] >= black_peak_min)
            or avg_pct["黑"] >= black_avg_min
        )
    )

    if result["black_confirmed"]:
        result["color"] = "黑"
        return result

    if "白" in allowed:
        result["color"] = "白"
        return result
    return result


def classify_arc_interval_color(start_time=None, end_time=None, window_s=None,
                                exclude_colors=None, allowed_colors=None):
    """圆弧时间区间判色：兼容旧接口，只返回颜色。"""
    return analyze_arc_interval_color(
        start_time=start_time,
        end_time=end_time,
        window_s=window_s,
        exclude_colors=exclude_colors,
        allowed_colors=allowed_colors,
    )["color"]


def _motion_color_from_pct(pct, exclude_colors=None):
    """运动中从 ROI 占比判断物块颜色：彩色优先，黑/白按面积主导。"""
    exclude = set(exclude_colors or [])

    chroma = [c for c in ("红", "绿", "蓝") if c not in exclude]
    if chroma:
        best_chroma = max(chroma, key=lambda c: pct.get(c, 0.0))
        if pct.get(best_chroma, 0.0) >= CONFIG["motion_chroma_min_pct"]:
            return best_chroma

    black_pct = pct.get("黑", 0.0)
    white_pct = pct.get("白", 0.0)
    if (
        "白" not in exclude
        and white_pct >= CONFIG["motion_white_min_pct"]
        and white_pct >= black_pct + CONFIG["motion_bw_margin_pct"]
    ):
        return "白"
    if (
        "黑" not in exclude
        and black_pct >= CONFIG["motion_black_min_pct"]
        and black_pct >= white_pct + CONFIG["motion_bw_margin_pct"]
    ):
        return "黑"
    if "白" not in exclude and white_pct >= CONFIG["motion_white_min_pct"]:
        return "白"
    return None


def _instant_color_from_sample(sample_color, pct, exclude_colors=None):
    """从 viewer 已显示过的一帧里取色；彩色优先，黑/白按面积主导。"""
    exclude = set(exclude_colors or [])

    chroma = [c for c in ("红", "绿", "蓝") if c not in exclude]
    if chroma:
        best_chroma = max(chroma, key=lambda c: pct.get(c, 0.0))
        if pct.get(best_chroma, 0.0) >= CONFIG["instant_chroma_min_pct"]:
            return best_chroma

    black_pct = pct.get("黑", 0.0)
    white_pct = pct.get("白", 0.0)
    if (
        "白" not in exclude
        and white_pct >= CONFIG["instant_white_min_pct"]
        and white_pct >= black_pct + CONFIG["instant_bw_margin_pct"]
    ):
        return "白"

    if (
        "黑" not in exclude
        and black_pct >= CONFIG["instant_black_min_pct"]
        and black_pct >= white_pct + CONFIG["instant_bw_margin_pct"]
    ):
        return "黑"

    if sample_color in ("红", "绿", "蓝") and sample_color not in exclude:
        return sample_color
    if sample_color == "黑" and sample_color not in exclude:
        return "黑"

    # 白色面积足够大时接受；黑线只占少量时不会压过白色。
    max_chroma = max((pct.get(c, 0.0) for c in ("红", "绿", "蓝")), default=0.0)
    if (
        "白" not in exclude
        and sample_color == "白"
        and white_pct >= CONFIG["instant_white_min_pct"]
        and max_chroma < CONFIG["instant_chroma_min_pct"]
    ):
        return "白"

    return None


def get_recent_display_color(window_s=None, exclude_colors=None, prefer_non_white=False):
    """读取 viewer 最近显示过的颜色；可让非白色证据优先于白色背景。"""
    if window_s is None:
        window_s = CONFIG["instant_recent_window_s"]
    now = time.time()
    with _recent_color_lock:
        samples = [(t, c, pct) for t, c, pct in _recent_color_samples
                   if (now - t) <= window_s]
    white_candidate = None
    for _t, color, pct in reversed(samples):
        instant_color = _instant_color_from_sample(color, pct or {}, exclude_colors=exclude_colors)
        if instant_color is None:
            continue
        if prefer_non_white and instant_color == "白":
            white_candidate = "白"
            continue
        if instant_color is not None:
            return instant_color
    return white_candidate


def wait_for_display_color(timeout_s=0.5, window_s=None, exclude_colors=None,
                           prefer_non_white=False):
    """在 timeout 内只要 viewer 显示过一次有效颜色就返回。"""
    start = time.time()
    white_candidate = None
    while (time.time() - start) <= timeout_s:
        color = get_recent_display_color(
            window_s=window_s,
            exclude_colors=exclude_colors,
            prefer_non_white=prefer_non_white,
        )
        if prefer_non_white and color == "白":
            white_candidate = "白"
        elif color is not None:
            return color
        time.sleep(0.02)
    return white_candidate


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
