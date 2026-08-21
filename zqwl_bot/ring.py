#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ring.py - 同心圆环识别 + 自动像素比例估算 + dx/dy 定位

main 里用法:
    import comm
    import ring
    comm.init("/dev/ttyCH341USB*", 115200)
    ring.align_once_to_ring()     # 到附近后: 找同心圆圆心 → 自动估算 mm/px → dx/dy 定位
    ring.align_to_ring()          # 兼容旧 3 步，内部统一使用 FINE_MOVE
    ring.close()                  # 退出时
"""

import glob
import math
import os
import re
import sys
import threading
import time
import comm

# 必须在 cv2 首次导入前设置；否则 USB 摄像头异常时 V4L2 select 可能一次卡约 10 秒。
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")
os.environ.setdefault("OPENCV_VIDEOIO_V4L_READ_ATTEMPTS", "1")

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
try:
    import numpy as np
except ModuleNotFoundError:
    np = None


CONFIG = {
    "usb_device": 1,
    "usb_devices": [1, 0],
    "width": 640,
    "height": 480,
    "fps": 30,
    "usb_open_verify_frames": 1,
    "usb_black_mean_min": 5.0,
    "usb_dark_uniform_mean_max": 25.0,
    "usb_dark_uniform_std_min": 1.0,
    "device_retry_cooldown_s": 8.0,
    "camera_read_timeout_s": 0.12,
    "camera_stale_timeout_s": 1.0,
    "camera_no_frame_reopen_s": 1.2,

    # 黑线不是靠单一黑度阈值抠出来，而是靠“局部暗线 + 边缘 + 同心几何”联合判断。
    "evidence_percentile": 88,
    "evidence_min_thresh": 32,
    "blackhat_kernel_px": 21,
    "blackhat_weight": 0.62,
    "edge_weight": 0.28,
    "dark_weight": 0.10,
    "edge_dilate_px": 1,
    "known_ring_diameters_mm": [50, 90, 130, 170, 210],
    "single_arc_default_diameter_mm": 210,
    "use_manual_scale": True,
    "manual_mm_per_px": 0.25,
    "manual_center_min_diam_ratio": 0.38,
    "manual_center_min_support": 2,
    "manual_center_refine_enable": False,
    "manual_center_refine_search_px": 32,
    "manual_center_refine_step_px": 3,
    "manual_center_refine_angles": 72,
    "manual_center_refine_ring_width_px": 5,
    "manual_center_refine_hit_thresh": 0.18,
    "manual_center_refine_min_score": 0.08,
    "manual_center_refine_min_hits": 2,
    "manual_center_min_visible_ratio": 0.15,
    "min_candidate_diameter_px": 18,
    "max_candidate_diameter_px": 900,
    "min_arc_len_px": 45,
    "min_arc_coverage": 0.06,
    "min_eccentricity": 0.30,
    "min_contour_points": 5,
    "center_cluster_px": 80,
    "diam_merge_abs_px": 28,
    "diam_merge_rel": 0.075,
    "match_abs_tol_px": 18,
    "match_rel_tol": 0.18,
    "min_scale_matches": 2,
    "center_reliable_min_diameter_mm": 90,
    "center_consistency_abs_px": 35,
    "center_consistency_rel": 0.12,

    # RANSAC 圆弧兜底：用于黑环断裂、只看到一段圆弧、字母干扰较重的情况。
    "enable_ransac": False,
    "ransac_iterations": 120,
    "ransac_max_points": 700,
    "ransac_max_circles": 3,
    "ransac_inlier_tol_px": 4.5,
    "ransac_remove_tol_px": 7.0,
    "ransac_min_inliers": 55,
    "ransac_min_coverage": 0.045,
    "ransac_edge_low": 40,
    "ransac_edge_high": 120,

    # 径向扫描仅保留作调试兜底；默认不用它选结果，避免把某个圆环误当 5 环比例。
    "enable_radial_scan": False,
    "radial_trigger_confidence": 0.45,
    "radial_trigger_offset_mm": 30.0,
    "radial_trigger_match_count": 4,
    "radial_prefer_confidence_below": 0.70,
    "radial_prefer_match_count_below": 4,
    "radial_prefer_min_score": 0.14,
    "radial_prefer_min_hits": 3,
    "radial_center_search_px": 100,
    "radial_center_step_px": 20,
    "radial_refine_step_px": 4,
    "radial_outer_radius_min_px": 45,
    "radial_outer_radius_max_px": 360,
    "radial_outer_radius_step_px": 8,
    "radial_refine_radius_step_px": 2,
    "radial_angles": 48,
    "radial_ring_width_px": 5,
    "radial_hit_thresh": 0.18,
    "radial_ring_min_score": 0.07,
    "radial_min_ring_hits": 2,
    "radial_min_score": 0.12,

    "ring_actual_radius_m": 0.075,  # 仅作为自动比例失败时的旧兜底
    "calib_step": 0.0,

    "detect_frames": 5,
    "detect_min_hits": 2,
    "detect_timeout_s": 1.2,
    "otsu_smooth_n": 3,
    "consistency_max_diff": 80,
    "last_known_max_age_s": 3.0,
    "camera_warmup_frames": 5,
    "camera_warmup_timeout_s": 1.2,

    "dead_zone_px": 0,
    "align_tolerance_mm": 3.0,
    "align_tolerance_x_mm": 2.0,
    "align_tolerance_y_mm": 2.0,
    "max_align_iterations": 10,
    "checked_align_max_moves": 6,
    "align_detect_retry_count": 4,
    "align_detect_retry_sleep_s": 0.15,
    "align_confirm_frames": 2,
    "align_confirm_sleep_s": 0.12,
    "post_fine_move_settle_s": 0.25,
    "correct_timeout_s": 20.0,
    "push_move_timeout_s": 35.0,
    "post_push_quiesce_s": 0.0,
    "back_move_timeout_s": 45.0,
    "fine_move_max_step_mm": 30.0,
    "auto_forward_after_align": False,
    "camera_dx_sign": -1.0,
    "camera_dy_sign": 1.0,
    "camera_swap_xy": False,

    "pre_cmd_sleep_s": 0.3,
    "comm_port": "/dev/ttyCH341USB*",
    "comm_baud": 115200,

    "recover_pose_age_max": 5.0,
    "recover_max_retry": 3,

    "step1_to_step2_pause_s": 2.0,
    # 摄像头安装补偿：视觉圆心对准后，实测车身偏右，因此前推时额外向左 2mm。
    # 车体坐标约定：+dx=右移，-dx=左移；该补偿只作用在对准后的固定前推动作。
    "post_align_trim_dx_mm": -2.0,
    "post_align_trim_dy_mm": 0.0, 
    "post_align_offset_y_mm": 94,
    "post_align_back_y_mm": 200,
    "step2_to_step3_pause_s": 2.0,
    "step3_back_offset_px": 500,

    "show_debug_window": True,
    "debug_window_name": "Ring Debug",
    "debug_mask_window_name": "Ring Evidence",
    "debug_max_candidates": 25,
    "show_candidate_circles": False,
    "draw_selected_ellipse": False,
    "draw_manual_reference_rings": False,
    "draw_observed_fit_rings": True,
    "preview_detect_interval_s": 0.25,
    "preview_result_max_age_s": 1.00,
    "enable_debug_trackbars": True,
    "trackbar_window_name": "Ring Tune",
}


def _port_sort_key(path):
    """串口候选排序：CH341 优先，同类设备优先尝试编号较大的新设备。"""
    name = os.path.basename(str(path))
    if name.startswith("ttyCH341USB"):
        group = 0
    elif name.startswith("ttyUSB"):
        group = 1
    elif name.startswith("ttyACM"):
        group = 2
    else:
        group = 9
    match = re.search(r"(\d+)$", name)
    number = int(match.group(1)) if match else -1
    return group, -number, str(path)


def resolve_comm_port(port=None):
    """把串口配置解析成真实设备名；统一使用 comm.py 的候选和环境变量逻辑。"""
    env_override = any(os.environ.get(k) for k in (
        "ZQWL_COMM_PORTS", "ZQWL_SERIAL_PORTS", "ZQWL_COMM_PORT", "ZQWL_SERIAL_PORT"
    ))
    use_default = port is None or port == "" or (env_override and str(port) == "/dev/ttyCH341USB*")
    candidates = comm.resolve_port_candidates(None if use_default else port)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "没有找到可用串口；已尝试 " + ", ".join(candidates) +
        "。请先执行: ls /dev/ttyCH341USB* /dev/ttyUSB* /dev/ttyACM*"
    )


def validate_config():
    """校验圆环定位关键配置，避免现场调参时出现隐蔽错误。"""
    diameters = [float(v) for v in CONFIG.get("known_ring_diameters_mm", [])]
    if len(diameters) < 2:
        raise ValueError("known_ring_diameters_mm 至少需要 2 个已知直径")
    if any(v <= 0.0 for v in diameters):
        raise ValueError("known_ring_diameters_mm 必须全部为正数")
    if any(b <= a for a, b in zip(diameters, diameters[1:])):
        raise ValueError("known_ring_diameters_mm 必须按从小到大排列")
    if float(CONFIG.get("single_arc_default_diameter_mm", 0.0)) <= 0.0:
        raise ValueError("single_arc_default_diameter_mm 必须为正数")
    if CONFIG.get("use_manual_scale", True) and float(CONFIG.get("manual_mm_per_px", 0.0)) <= 0.0:
        raise ValueError("manual_mm_per_px 必须为正数")
    ratio = float(CONFIG.get("manual_center_min_diam_ratio", 0.38))
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("manual_center_min_diam_ratio 必须在 0~1 之间")
    if int(CONFIG.get("manual_center_min_support", 1)) < 1:
        raise ValueError("manual_center_min_support 至少为 1")
    if int(CONFIG.get("manual_center_refine_search_px", 1)) < 1:
        raise ValueError("manual_center_refine_search_px 必须为正数")
    if int(CONFIG.get("manual_center_refine_step_px", 1)) < 1:
        raise ValueError("manual_center_refine_step_px 必须为正数")
    if int(CONFIG.get("manual_center_refine_angles", 8)) < 8:
        raise ValueError("manual_center_refine_angles 至少为 8")
    visible_ratio = float(CONFIG.get("manual_center_min_visible_ratio", 0.15))
    if visible_ratio <= 0.0 or visible_ratio > 1.0:
        raise ValueError("manual_center_min_visible_ratio 必须在 0~1 之间")
    if int(CONFIG.get("detect_min_hits", 1)) > int(CONFIG.get("detect_frames", 1)):
        raise ValueError("detect_min_hits 不能大于 detect_frames")
    if int(CONFIG.get("checked_align_max_moves", 1)) < 1:
        raise ValueError("checked_align_max_moves 至少为 1")
    if float(CONFIG.get("align_tolerance_x_mm", CONFIG.get("align_tolerance_mm", 0.0))) <= 0.0:
        raise ValueError("align_tolerance_x_mm 必须为正数")
    if float(CONFIG.get("align_tolerance_y_mm", CONFIG.get("align_tolerance_mm", 0.0))) <= 0.0:
        raise ValueError("align_tolerance_y_mm 必须为正数")
    if float(CONFIG.get("post_align_offset_y_mm", 0.0)) <= 0.0:
        raise ValueError("post_align_offset_y_mm 必须为正数")
    if float(CONFIG.get("post_align_back_y_mm", 0.0)) < 0.0:
        raise ValueError("post_align_back_y_mm 不能为负数")
    if float(CONFIG.get("min_candidate_diameter_px", 0.0)) >= float(CONFIG.get("max_candidate_diameter_px", 0.0)):
        raise ValueError("min_candidate_diameter_px 必须小于 max_candidate_diameter_px")
    for key in ("camera_dx_sign", "camera_dy_sign"):
        value = float(CONFIG.get(key, 0.0))
        if value not in (-1.0, 1.0):
            raise ValueError(f"{key} 只能是 +1.0 或 -1.0")
    if float(CONFIG.get("fine_move_max_step_mm", 0.0)) <= 0.0:
        raise ValueError("fine_move_max_step_mm 必须为正数")
    return True


def configure(overrides=None, **kwargs):
    """统一更新圆环定位配置；测试脚本和主流程都从这里进入。"""
    if overrides:
        CONFIG.update(overrides)
    if kwargs:
        CONFIG.update(kwargs)
    validate_config()
    return CONFIG


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


def apply_usb_env(prefix="ZQWL_RING"):
    """从环境变量覆盖 USB 设备号；专用变量优先，通用变量兜底。"""
    device = os.environ.get(f"{prefix}_USB_DEVICE", os.environ.get("ZQWL_USB_DEVICE"))
    devices = os.environ.get(f"{prefix}_USB_DEVICES", os.environ.get("ZQWL_USB_DEVICES"))
    if device is not None or devices is not None:
        return configure_usb(device=device, devices=devices)
    return configure_usb(CONFIG.get("usb_device", 1), CONFIG.get("usb_devices", [1, 0]))


def set_camera_mapping(dx_sign=None, dy_sign=None, swap_xy=None):
    """设置图像偏差到车体 dx/dy 的映射；dx=右移，dy=前进。"""
    if dx_sign is not None:
        CONFIG["camera_dx_sign"] = 1.0 if float(dx_sign) >= 0.0 else -1.0
    if dy_sign is not None:
        CONFIG["camera_dy_sign"] = 1.0 if float(dy_sign) >= 0.0 else -1.0
    if swap_xy is not None:
        CONFIG["camera_swap_xy"] = bool(swap_xy)
    validate_config()
    return {
        "camera_dx_sign": CONFIG["camera_dx_sign"],
        "camera_dy_sign": CONFIG["camera_dy_sign"],
        "camera_swap_xy": CONFIG["camera_swap_xy"],
    }


def pixel_offset_to_body_mm(dx_px, dy_px, mm_per_px):
    """把图像像素偏差转换成下位机 FINE_MOVE 使用的车体 dx/dy。"""
    raw_dx_mm = float(dx_px) * float(mm_per_px)
    raw_dy_mm = float(dy_px) * float(mm_per_px)
    if CONFIG.get("camera_swap_xy", False):
        raw_dx_mm, raw_dy_mm = raw_dy_mm, raw_dx_mm
    return {
        "dx_mm": raw_dx_mm * float(CONFIG["camera_dx_sign"]),
        "dy_mm": raw_dy_mm * float(CONFIG["camera_dy_sign"]),
        "raw_dx_mm": raw_dx_mm,
        "raw_dy_mm": raw_dy_mm,
    }


def _require_cv2():
    """需要摄像头/图像处理时再检查视觉依赖，方便无硬件配置自检。"""
    if cv2 is None:
        raise RuntimeError("当前 Python 环境缺少 OpenCV(cv2)，只能做 mapping/selftest，不能打开视觉识别")
    if np is None:
        raise RuntimeError("当前 Python 环境缺少 numpy，只能做 mapping/selftest，不能打开视觉识别")


# ============ 摄像头 ============
def _frame_is_valid(frame):
    """判断摄像头帧是否有效；防止 isOpened=True 但实际黑屏/空帧。"""
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
    return not (
        mean_v < float(CONFIG.get("usb_dark_uniform_mean_max", 25.0))
        and std_v < float(CONFIG.get("usb_dark_uniform_std_min", 1.0))
    )


class USBCamera:
    def __init__(self, device=1, width=640, height=480, fps=30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._latest_frame = None
        self._latest_time = 0.0
        self._started_time = 0.0
        self._read_fail_count = 0

    def start(self):
        _require_cv2()
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.stop()
            raise RuntimeError(f"无法打开USB摄像头 device={self.device}")

        self._started_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop,
                                        name=f"ring-usb-camera-{self.device}",
                                        daemon=True)
        self._thread.start()

        verify_need = max(0, int(CONFIG.get("usb_open_verify_frames", 0)))
        if verify_need > 0:
            got = 0
            verify_timeout = max(0.5, float(CONFIG.get("camera_read_timeout_s", 0.35)) * 3.0)
            deadline = time.time() + verify_timeout
            while got < verify_need and time.time() < deadline:
                frame = self.read(timeout=0.05)
                if frame is not None:
                    got += 1
                else:
                    time.sleep(0.02)
            if got <= 0:
                print(f"[ring] USB摄像头 device={self.device} 已打开，但首帧尚未就绪，后台继续预热", flush=True)

    def _reader_loop(self):
        while not self._stop_event.is_set():
            cap = self.cap
            if cap is None:
                break
            ret, frame = cap.read()
            if ret and _frame_is_valid(frame):
                with self._lock:
                    self._latest_frame = frame
                    self._latest_time = time.time()
                    self._read_fail_count = 0
            else:
                with self._lock:
                    self._read_fail_count += 1
                time.sleep(0.02)

    def read(self, timeout=None):
        if timeout is None:
            timeout = float(CONFIG.get("camera_read_timeout_s", 0.35))
        deadline = time.time() + max(0.0, float(timeout))
        while True:
            with self._lock:
                frame = self._latest_frame
                age = time.time() - self._latest_time if self._latest_time > 0.0 else float("inf")
                if frame is not None and age <= float(CONFIG.get("camera_stale_timeout_s", 1.0)):
                    return frame.copy()
            if time.time() >= deadline:
                return None
            time.sleep(0.01)

    def stop(self):
        self._stop_event.set()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        with self._lock:
            self._latest_frame = None
            self._latest_time = 0.0
            self._started_time = 0.0


# ============ 圆环检测 ============
def _normalize_u8(src):
    """把灰度证据图鲁棒归一化到 0~255，避免强反光把动态范围吃掉。"""
    arr = src.astype(np.float32)
    lo, hi = np.percentile(arr, (2, 98))
    if hi <= lo + 1e-6:
        return np.zeros_like(src, dtype=np.uint8)
    arr = (arr - lo) * (255.0 / (hi - lo))
    return np.clip(arr, 0, 255).astype(np.uint8)


def _odd_kernel_size(value, minimum=3):
    """OpenCV 形态学核需要正奇数。"""
    k = max(int(value), int(minimum))
    return k if k % 2 == 1 else k + 1


def _build_ring_evidence(frame):
    """构建黑色圆环证据图。

    这里不追求把 mask 调到“只剩黑线”。黑环的第一性原理是：它在局部邻域里
    是一条暗线，并且两侧有边缘；最终是否为圆环交给几何一致性判断。
    """
    _require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # black-hat 会增强“比周围暗的细线”，比绝对黑度更抗补光灯反光和阴影。
    k = _odd_kernel_size(CONFIG["blackhat_kernel_px"], minimum=9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    blackhat = cv2.subtract(closed, blur)
    blackhat_norm = _normalize_u8(blackhat)

    # 边缘只作为弱证据，避免把所有背景纹理都当成圆。
    edges = cv2.Canny(blur, CONFIG["ransac_edge_low"], CONFIG["ransac_edge_high"])
    dilate_px = int(CONFIG.get("edge_dilate_px", 1))
    if dilate_px > 0:
        edge_kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        edges = cv2.dilate(edges, edge_kernel)

    # 绝对暗度只给很小权重，不再作为核心门槛。
    inv_dark = _normalize_u8(255 - blur)
    evidence = (
        blackhat_norm.astype(np.float32) * float(CONFIG["blackhat_weight"]) +
        edges.astype(np.float32) * float(CONFIG["edge_weight"]) +
        inv_dark.astype(np.float32) * float(CONFIG["dark_weight"])
    )
    evidence = np.clip(evidence, 0, 255).astype(np.uint8)

    q = float(CONFIG["evidence_percentile"])
    thresh = max(float(CONFIG["evidence_min_thresh"]), float(np.percentile(evidence, q)))
    _, mask = cv2.threshold(evidence, thresh, 255, cv2.THRESH_BINARY)
    small_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_kernel)
    return mask, evidence, edges


def _threshold_dark(frame):
    """兼容旧调用：返回黑环证据二值图，而不是纯黑度阈值图。"""
    _require_cv2()
    mask, _, _ = _build_ring_evidence(frame)
    return mask


def _angle_coverage(points, cx, cy, bins=72):
    """估算一段圆弧覆盖了多少角度，允许只看到部分圆弧。"""
    pts = points.reshape(-1, 2).astype(np.float32)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    idx = ((angles + np.pi) / (2.0 * np.pi) * bins).astype(np.int32)
    idx = np.clip(idx, 0, bins - 1)
    return len(np.unique(idx)) / float(bins)


def _extract_ring_candidates(frame, dark=None):
    """从黑色 mask 中提取可能属于同心圆环的椭圆/圆弧候选。"""
    if dark is None:
        dark = _threshold_dark(frame)
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], dark

    h, w = frame.shape[:2]
    candidates = []
    for cnt in contours:
        if len(cnt) < CONFIG["min_contour_points"]:
            continue

        arc_len = cv2.arcLength(cnt, False)
        if arc_len < CONFIG["min_arc_len_px"]:
            continue

        try:
            (cx, cy), (ew, eh), angle = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        axis_major = max(float(ew), float(eh))
        axis_minor = min(float(ew), float(eh))
        if axis_major <= 1.0 or axis_minor <= 1.0:
            continue

        eccentricity = axis_minor / axis_major
        if eccentricity < CONFIG["min_eccentricity"]:
            continue

        diameter_px = (axis_major + axis_minor) / 2.0
        if diameter_px < CONFIG["min_candidate_diameter_px"]:
            continue
        if diameter_px > CONFIG["max_candidate_diameter_px"]:
            continue

        # 允许圆心略在画面外，适配只看到外圈一段圆弧的情况。
        if cx < -0.35 * w or cx > 1.35 * w or cy < -0.35 * h or cy > 1.35 * h:
            continue

        coverage = _angle_coverage(cnt, cx, cy)
        if coverage < CONFIG["min_arc_coverage"]:
            continue

        # 以椭圆中心为参考估算径向离散度，过滤字母/杂散黑块。
        pts = cnt.reshape(-1, 2).astype(np.float32)
        rr = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        radial_std = float(np.std(rr)) if len(rr) else 999.0
        score = float(len(cnt)) * coverage * eccentricity / (1.0 + radial_std / max(diameter_px, 1.0))
        candidates.append({
            "cx": float(cx),
            "cy": float(cy),
            "diam_px": float(diameter_px),
            "radius_px": float(diameter_px) / 2.0,
            "ellipse_w": float(ew),
            "ellipse_h": float(eh),
            "ellipse_angle": float(angle),
            "axis_major": axis_major,
            "axis_minor": axis_minor,
            "eccentricity": float(eccentricity),
            "coverage": float(coverage),
            "arc_len": float(arc_len),
            "radial_std": radial_std,
            "score": score,
        })
    return candidates, dark


def _cluster_by_center(candidates):
    """把圆心接近的候选聚成同心圆组。"""
    clusters = []
    for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
        assigned = False
        for cluster in clusters:
            dx = cand["cx"] - cluster["cx"]
            dy = cand["cy"] - cluster["cy"]
            if (dx * dx + dy * dy) ** 0.5 <= CONFIG["center_cluster_px"]:
                cluster["items"].append(cand)
                weight_sum = sum(i["score"] for i in cluster["items"])
                cluster["cx"] = sum(i["cx"] * i["score"] for i in cluster["items"]) / weight_sum
                cluster["cy"] = sum(i["cy"] * i["score"] for i in cluster["items"]) / weight_sum
                assigned = True
                break
        if not assigned:
            clusters.append({"cx": cand["cx"], "cy": cand["cy"], "items": [cand]})
    return clusters


def _merge_similar_diameters(items):
    """合并同一实际圆环的内外边缘/重复候选。"""
    merged = []
    for cand in sorted(items, key=lambda c: c["diam_px"]):
        if not merged:
            merged.append({"items": [cand]})
            continue
        prev_items = merged[-1]["items"]
        prev_d = sum(i["diam_px"] * i["score"] for i in prev_items) / sum(i["score"] for i in prev_items)
        tol = max(CONFIG["diam_merge_abs_px"], prev_d * CONFIG["diam_merge_rel"])
        if abs(cand["diam_px"] - prev_d) <= tol:
            prev_items.append(cand)
        else:
            merged.append({"items": [cand]})

    out = []
    for group in merged:
        group_items = group["items"]
        weight_sum = sum(i["score"] for i in group_items)
        best_item = max(group_items, key=lambda i: i["score"])
        out.append({
            "cx": sum(i["cx"] * i["score"] for i in group_items) / weight_sum,
            "cy": sum(i["cy"] * i["score"] for i in group_items) / weight_sum,
            "diam_px": sum(i["diam_px"] * i["score"] for i in group_items) / weight_sum,
            "ellipse_w": float(best_item.get("ellipse_w", best_item.get("diam_px", 0.0))),
            "ellipse_h": float(best_item.get("ellipse_h", best_item.get("diam_px", 0.0))),
            "ellipse_angle": float(best_item.get("ellipse_angle", 0.0)),
            "score": weight_sum,
            "coverage": max(i["coverage"] for i in group_items),
            "count": len(group_items),
        })
    return out


def _weighted_match_center(matches):
    """用大直径和高分候选加权估计圆心，降低中心字母闭合轮廓的影响。"""
    weight_sum = 0.0
    sx = 0.0
    sy = 0.0
    for m in matches:
        diameter_mm = max(float(m.get("known_mm", 1.0)), 1.0)
        weight = max(float(m.get("score", 1.0)), 1.0) * diameter_mm * diameter_mm
        weight_sum += weight
        sx += float(m["cx"]) * weight
        sy += float(m["cy"]) * weight
    if weight_sum <= 0.0:
        return None
    return sx / weight_sum, sy / weight_sum


def _filter_center_consistent_matches(matches, px_per_mm):
    """剔除中心明显偏离大圆中心的小圆候选，典型来源是圆环内部字母。"""
    if not matches:
        return [], []

    min_reliable = float(CONFIG.get("center_reliable_min_diameter_mm", 90.0))
    reliable = [m for m in matches if float(m.get("known_mm", 0.0)) >= min_reliable]
    base_center = _weighted_match_center(reliable)
    if base_center is None:
        return matches, []

    bx, by = base_center
    kept = []
    rejected = []
    for m in matches:
        known_mm = float(m.get("known_mm", 0.0))
        dev = ((float(m["cx"]) - bx) ** 2 + (float(m["cy"]) - by) ** 2) ** 0.5
        diam_px = known_mm * float(px_per_mm)
        tol = max(float(CONFIG.get("center_consistency_abs_px", 35.0)),
                  diam_px * float(CONFIG.get("center_consistency_rel", 0.12)))
        if known_mm < min_reliable and dev > tol:
            bad = dict(m)
            bad["center_dev_px"] = float(dev)
            bad["center_tol_px"] = float(tol)
            rejected.append(bad)
            continue
        kept.append(m)

    return kept if kept else reliable, rejected


def _fit_scale_for_cluster(cluster):
    """用 50/90/130/170/210mm 已知直径，自动匹配 px/mm 比例。"""
    known = [float(v) for v in CONFIG["known_ring_diameters_mm"]]
    observed = _merge_similar_diameters(cluster["items"])
    if not observed:
        return None

    best = None
    for obs in observed:
        for known_mm in known:
            px_per_mm = obs["diam_px"] / known_mm
            if px_per_mm <= 0:
                continue

            matches_by_known = {}
            score = 0.0
            for obs2 in observed:
                best_known = None
                best_err = None
                best_tol = None
                for km in known:
                    expected_px = km * px_per_mm
                    err = abs(obs2["diam_px"] - expected_px)
                    tol = max(CONFIG["match_abs_tol_px"], expected_px * CONFIG["match_rel_tol"])
                    if best_err is None or err < best_err:
                        best_known, best_err, best_tol = km, err, tol
                if best_err is not None and best_err <= best_tol:
                    quality = max(0.0, 1.0 - best_err / best_tol)
                    item_score = obs2["score"] * quality
                    old = matches_by_known.get(best_known)
                    if old is None or item_score > old["score"]:
                        matches_by_known[best_known] = {
                            "known_mm": best_known,
                            "diam_px": obs2["diam_px"],
                            "err_px": best_err,
                            "tol_px": best_tol,
                            "score": item_score,
                            "cx": obs2["cx"],
                            "cy": obs2["cy"],
                            "ellipse_w": obs2.get("ellipse_w", obs2["diam_px"]),
                            "ellipse_h": obs2.get("ellipse_h", obs2["diam_px"]),
                            "ellipse_angle": obs2.get("ellipse_angle", 0.0),
                        }
                    score += item_score

            raw_matches = list(matches_by_known.values())
            trusted_matches, rejected_matches = _filter_center_consistent_matches(raw_matches, px_per_mm)
            match_count = len(trusted_matches)
            trusted_score = sum(m["score"] for m in trusted_matches)
            total_score = match_count * 100000.0 + trusted_score
            if best is None or total_score > best["total_score"]:
                best = {
                    "px_per_mm": px_per_mm,
                    "mm_per_px": 1.0 / px_per_mm,
                    "matches": trusted_matches,
                    "rejected_matches": rejected_matches,
                    "raw_match_count": len(raw_matches),
                    "match_count": match_count,
                    "total_score": total_score,
                    "observed": observed,
                }

    # 单段圆弧兜底：用户实测说明单段通常按最外圈 210mm 处理。
    if best is None or best["match_count"] < CONFIG["min_scale_matches"]:
        largest = max(observed, key=lambda o: o["diam_px"])
        default_mm = float(CONFIG["single_arc_default_diameter_mm"])
        px_per_mm = largest["diam_px"] / default_mm
        best = {
            "px_per_mm": px_per_mm,
            "mm_per_px": 1.0 / px_per_mm,
            "matches": [{
                "known_mm": default_mm,
                "diam_px": largest["diam_px"],
                "err_px": 0.0,
                "tol_px": 0.0,
                "score": largest["score"],
                "cx": largest["cx"],
                "cy": largest["cy"],
                "ellipse_w": largest.get("ellipse_w", largest["diam_px"]),
                "ellipse_h": largest.get("ellipse_h", largest["diam_px"]),
                "ellipse_angle": largest.get("ellipse_angle", 0.0),
            }],
            "match_count": 1,
            "total_score": largest["score"],
            "observed": observed,
            "rejected_matches": [],
            "raw_match_count": 1,
            "fallback_single": True,
        }
    else:
        best["fallback_single"] = False

    matched = best["matches"]
    if matched:
        center = _weighted_match_center(matched)
        if center is None:
            cx, cy = cluster["cx"], cluster["cy"]
        else:
            cx, cy = center
        outer = max(matched, key=lambda m: m["known_mm"])
        radius_px = outer["diam_px"] / 2.0
        debug_ellipse = {
            "cx": float(outer.get("cx", cx)),
            "cy": float(outer.get("cy", cy)),
            "w": float(outer.get("ellipse_w", outer["diam_px"])),
            "h": float(outer.get("ellipse_h", outer["diam_px"])),
            "angle": float(outer.get("ellipse_angle", 0.0)),
        }
    else:
        cx, cy = cluster["cx"], cluster["cy"]
        radius_px = max(o["diam_px"] for o in observed) / 2.0
        debug_ellipse = None

    best.update({
        "cx": float(cx),
        "cy": float(cy),
        "radius_px": float(radius_px),
        "debug_ellipse": debug_ellipse,
        "confidence": min(1.0, best["match_count"] / float(len(known)))
    })
    return best


def _ellipse_shape_from_support(support):
    """从大圆候选估计透视下的椭圆形状；只估形状，不用它决定圆心。"""
    if not support:
        return 1.0, 1.0, 0.0

    best = max(support, key=lambda o: float(o.get("score", 0.0)) * float(o.get("diam_px", 0.0)))
    ew = max(2.0, float(best.get("ellipse_w", best.get("diam_px", 1.0))))
    eh = max(2.0, float(best.get("ellipse_h", best.get("diam_px", 1.0))))
    major = max(ew, eh)
    if major <= 2.0:
        return 1.0, 1.0, 0.0
    return ew / major, eh / major, float(best.get("ellipse_angle", 0.0))


def _robust_observed_center(support):
    """只根据图像中实际拟合到的圆/圆弧估计圆心，不使用物理直径和像素比例。"""
    if not support:
        return None, []

    items = list(support)
    if len(items) <= 2:
        kept = items
    else:
        def item_weight(item):
            diam = max(float(item.get("diam_px", 1.0)), 1.0)
            score = max(float(item.get("score", 1.0)), 1.0)
            coverage = max(float(item.get("coverage", 0.05)), 0.05)
            return score * diam * diam * coverage

        weight_sum = sum(item_weight(item) for item in items)
        seed_x = sum(float(item["cx"]) * item_weight(item) for item in items) / max(weight_sum, 1e-6)
        seed_y = sum(float(item["cy"]) * item_weight(item) for item in items) / max(weight_sum, 1e-6)
        dists = [((float(item["cx"]) - seed_x) ** 2 + (float(item["cy"]) - seed_y) ** 2) ** 0.5 for item in items]
        med = float(np.median(np.array(dists, dtype=np.float32)))
        mad = float(np.median(np.abs(np.array(dists, dtype=np.float32) - med)))
        tol = max(18.0, med + 2.5 * max(mad, 1.0), float(CONFIG.get("center_cluster_px", 80)) * 0.45)
        kept = [item for item, dist in zip(items, dists) if dist <= tol]
        if len(kept) < 2:
            kept = sorted(items, key=item_weight, reverse=True)[:2]

    weight_sum = 0.0
    sx = 0.0
    sy = 0.0
    for item in kept:
        diam = max(float(item.get("diam_px", 1.0)), 1.0)
        score = max(float(item.get("score", 1.0)), 1.0)
        coverage = max(float(item.get("coverage", 0.05)), 0.05)
        weight = score * diam * diam * coverage
        weight_sum += weight
        sx += float(item["cx"]) * weight
        sy += float(item["cy"]) * weight
    if weight_sum <= 0.0:
        return None, []
    return (sx / weight_sum, sy / weight_sum), kept


def _score_fixed_scale_center(mask_norm, cx, cy, radii_px, cos_a, sin_a, ellipse_shape=(1.0, 1.0, 0.0)):
    """固定半径下给圆心打分；大半径权重更高，中心字母影响自然降低。"""
    h, w = mask_norm.shape[:2]
    ring_width = max(1.0, float(CONFIG.get("manual_center_refine_ring_width_px", 5)))
    hit_thresh = float(CONFIG.get("manual_center_refine_hit_thresh", 0.18))
    min_visible_ratio = float(CONFIG.get("manual_center_min_visible_ratio", 0.15))
    offsets = np.array([-ring_width, 0.0, ring_width], dtype=np.float32)
    axis_w_scale, axis_h_scale, angle_deg = ellipse_shape
    axis_w_scale = max(0.20, float(axis_w_scale))
    axis_h_scale = max(0.20, float(axis_h_scale))
    angle_rad = np.deg2rad(float(angle_deg))
    rot_c = float(np.cos(angle_rad))
    rot_s = float(np.sin(angle_rad))

    total_weight = 0.0
    total_score = 0.0
    ring_scores = []
    visible_flags = []
    visible_count = 0
    hit_count = 0
    for radius in radii_px:
        r = float(radius)
        if r <= 1.0:
            ring_scores.append(0.0)
            visible_flags.append(False)
            continue

        per_offset_vals = []
        valid_count = 0
        total_count = 0
        for off in offsets:
            rx = max(1.0, r * axis_w_scale + float(off))
            ry = max(1.0, r * axis_h_scale + float(off))
            local_x = rx * cos_a
            local_y = ry * sin_a
            xs = np.rint(cx + local_x * rot_c - local_y * rot_s).astype(np.int32)
            ys = np.rint(cy + local_x * rot_s + local_y * rot_c).astype(np.int32)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            vals = np.zeros_like(cos_a, dtype=np.float32)
            vals[valid] = mask_norm[ys[valid], xs[valid]]
            per_offset_vals.append(vals)
            valid_count += int(np.count_nonzero(valid))
            total_count += int(len(valid))

        valid_ratio = valid_count / float(max(total_count, 1))
        if valid_ratio < min_visible_ratio:
            ring_scores.append(0.0)
            visible_flags.append(False)
            continue
        visible_flags.append(True)
        visible_count += 1

        vals = np.max(np.vstack(per_offset_vals), axis=0)
        mean_dark = float(np.mean(vals))
        hit_ratio = float(np.mean(vals >= hit_thresh))
        ring_score = (0.60 * mean_dark + 0.40 * hit_ratio) * valid_ratio
        ring_scores.append(ring_score)
        if ring_score >= float(CONFIG.get("manual_center_refine_min_score", 0.08)):
            hit_count += 1

        weight = r * r
        total_weight += weight
        total_score += ring_score * weight

    if total_weight <= 0.0:
        return 0.0, ring_scores, hit_count, visible_count, visible_flags
    return total_score / total_weight, ring_scores, hit_count, visible_count, visible_flags


def _refine_manual_center_by_fixed_radii(mask, cx0, cy0, ellipse_shape=(1.0, 1.0, 0.0)):
    """手动比例模式下，只优化圆心；物理半径由已知圆环直径和 manual_mm_per_px 固定。"""
    if not CONFIG.get("manual_center_refine_enable", True):
        return float(cx0), float(cy0), 0.0, [], 0, 0, []

    manual_scale = float(CONFIG["manual_mm_per_px"])
    if manual_scale <= 0.0:
        return float(cx0), float(cy0), 0.0, [], 0, 0, []

    known = np.array(CONFIG["known_ring_diameters_mm"], dtype=np.float32)
    radii_px = known / manual_scale / 2.0
    mask_norm = mask.astype(np.float32) / 255.0
    h, w = mask.shape[:2]
    search = int(CONFIG.get("manual_center_refine_search_px", 32))
    step = max(1, int(CONFIG.get("manual_center_refine_step_px", 3)))
    angle_count = int(CONFIG.get("manual_center_refine_angles", 72))
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)

    best = {
        "cx": float(cx0),
        "cy": float(cy0),
        "score": -1.0,
        "ring_scores": [],
        "hits": 0,
        "visible_count": 0,
        "visible_flags": [],
    }

    x_min = max(-int(0.2 * w), int(round(cx0 - search)))
    x_max = min(int(1.2 * w), int(round(cx0 + search)))
    y_min = max(-int(0.2 * h), int(round(cy0 - search)))
    y_max = min(int(1.2 * h), int(round(cy0 + search)))
    for yy in range(y_min, y_max + 1, step):
        for xx in range(x_min, x_max + 1, step):
            score, ring_scores, hits, visible_count, visible_flags = _score_fixed_scale_center(
                mask_norm, float(xx), float(yy), radii_px, cos_a, sin_a, ellipse_shape
            )
            if score > best["score"]:
                best = {
                    "cx": float(xx),
                    "cy": float(yy),
                    "score": float(score),
                    "ring_scores": ring_scores,
                    "hits": int(hits),
                    "visible_count": int(visible_count),
                    "visible_flags": visible_flags,
                }

    refine_step = 1
    local = best.copy()
    for yy in range(int(best["cy"] - step), int(best["cy"] + step) + 1, refine_step):
        for xx in range(int(best["cx"] - step), int(best["cx"] + step) + 1, refine_step):
            score, ring_scores, hits, visible_count, visible_flags = _score_fixed_scale_center(
                mask_norm, float(xx), float(yy), radii_px, cos_a, sin_a, ellipse_shape
            )
            if score > local["score"]:
                local = {
                    "cx": float(xx),
                    "cy": float(yy),
                    "score": float(score),
                    "ring_scores": ring_scores,
                    "hits": int(hits),
                    "visible_count": int(visible_count),
                    "visible_flags": visible_flags,
                }

    return (local["cx"], local["cy"], local["score"], local["ring_scores"], local["hits"],
            local["visible_count"], local["visible_flags"])


def _fit_manual_center_for_cluster(cluster, mask=None):
    """只拟合圆心，不自动判断物理圆环编号；像素比例由 manual_mm_per_px 给定。"""
    observed = _merge_similar_diameters(cluster["items"])
    if not observed:
        return None

    largest = max(observed, key=lambda o: o["diam_px"])
    largest_diam = float(largest["diam_px"])
    min_ratio = float(CONFIG.get("manual_center_min_diam_ratio", 0.38))
    min_support = int(CONFIG.get("manual_center_min_support", 2))

    # 第一性原理：同心圆中心由大半径圆环约束更强，中心字母/小闭合块半径小，不能参与圆心加权。
    support = [o for o in observed if float(o["diam_px"]) >= largest_diam * min_ratio]
    if len(support) < min_support:
        support = sorted(observed, key=lambda o: o["score"] * o["diam_px"] * o["diam_px"], reverse=True)[:min_support]

    center, center_support = _robust_observed_center(support)
    if center is None:
        return None
    cx, cy = center
    if center_support:
        support = center_support

    fixed_score = 0.0
    fixed_ring_scores = []
    fixed_hits = 0
    fixed_visible_count = 0
    fixed_visible_flags = []
    if mask is not None:
        ellipse_shape = _ellipse_shape_from_support(support)
        (rcx, rcy, fixed_score, fixed_ring_scores, fixed_hits,
         fixed_visible_count, fixed_visible_flags) = _refine_manual_center_by_fixed_radii(
            mask, cx, cy, ellipse_shape
        )
        required_hits = min(int(CONFIG.get("manual_center_refine_min_hits", 2)), fixed_visible_count)
        if (fixed_visible_count >= 2 and fixed_hits >= required_hits and
                fixed_score >= float(CONFIG.get("manual_center_refine_min_score", 0.08))):
            cx, cy = rcx, rcy
    outer = max(support, key=lambda o: o["diam_px"])
    support_score = sum(float(o.get("score", 0.0)) for o in support)
    avg_coverage = sum(float(o.get("coverage", 0.0)) for o in support) / max(len(support), 1)
    confidence = min(1.0, 0.20 + len(support) * 0.18 + avg_coverage * 0.45 + fixed_score * 0.40)
    manual_scale = float(CONFIG["manual_mm_per_px"])

    matches = []
    for item in support:
        matches.append({
            "diam_px": float(item["diam_px"]),
            "score": float(item.get("score", 0.0)),
            "cx": float(item["cx"]),
            "cy": float(item["cy"]),
            "ellipse_w": float(item.get("ellipse_w", item["diam_px"])),
            "ellipse_h": float(item.get("ellipse_h", item["diam_px"])),
            "ellipse_angle": float(item.get("ellipse_angle", 0.0)),
            "coverage": float(item.get("coverage", 0.0)),
        })

    return {
        "px_per_mm": 1.0 / manual_scale,
        "mm_per_px": manual_scale,
        "scale_source": "manual",
        "matches": matches,
        "match_count": len(matches),
        "raw_match_count": len(observed),
        "observed_count": len(observed),
        "rejected_matches": [],
        "fallback_single": False,
        "total_score": len(matches) * 100000.0 + support_score + largest_diam * 1000.0,
        "observed": observed,
        "cx": float(cx),
        "cy": float(cy),
        "radius_px": float(outer["diam_px"]) / 2.0,
        "fixed_center_score": float(fixed_score),
        "fixed_center_hits": int(fixed_hits),
        "fixed_ring_visible_count": int(fixed_visible_count),
        "fixed_ring_visible_flags": fixed_visible_flags,
        "fixed_ring_scores": fixed_ring_scores,
        "debug_ellipse": {
            "cx": float(cx),
            "cy": float(cy),
            "w": float(outer.get("ellipse_w", outer["diam_px"])),
            "h": float(outer.get("ellipse_h", outer["diam_px"])),
            "angle": float(outer.get("ellipse_angle", 0.0)),
        },
        "confidence": float(confidence),
    }


def _best_fit_from_candidates(candidates, dark, method):
    """从一组候选圆/圆弧中选出最可信的同心圆结果。"""
    if not candidates:
        return None

    best = None
    for cluster in _cluster_by_center(candidates):
        if CONFIG.get("use_manual_scale", True):
            fit = _fit_manual_center_for_cluster(cluster, dark)
        else:
            fit = _fit_scale_for_cluster(cluster)
        if fit is None:
            continue
        cluster_score = fit["total_score"] + len(cluster["items"]) * 1000.0
        fit.update({
            "method": method,
            "cluster_score": cluster_score,
            "candidate_count": len(cluster["items"]),
            "candidates": candidates,
            "mask": dark,
        })
        if best is None or cluster_score > best["cluster_score"]:
            best = fit
    return best


def _circle_from_3_points(p1, p2, p3):
    """三点定圆；共线或数值不稳定时返回 None。"""
    mat = np.array([
        [p1[0], p1[1], 1.0],
        [p2[0], p2[1], 1.0],
        [p3[0], p3[1], 1.0],
    ], dtype=np.float64)
    rhs = -np.array([
        p1[0] * p1[0] + p1[1] * p1[1],
        p2[0] * p2[0] + p2[1] * p2[1],
        p3[0] * p3[0] + p3[1] * p3[1],
    ], dtype=np.float64)
    try:
        a, b, c = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return None
    cx = -a * 0.5
    cy = -b * 0.5
    r2 = cx * cx + cy * cy - c
    if not np.isfinite(r2) or r2 <= 1.0:
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def _refine_circle_least_squares(points):
    """用内点做最小二乘圆拟合，降低 RANSAC 随机三点带来的抖动。"""
    if points is None or len(points) < 5:
        return None
    pts = points.astype(np.float64)
    mat = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    rhs = -(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    try:
        a, b, c = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx = -a * 0.5
    cy = -b * 0.5
    r2 = cx * cx + cy * cy - c
    if not np.isfinite(r2) or r2 <= 1.0:
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def _circle_is_in_reasonable_range(cx, cy, r, shape):
    """过滤明显不可能的圆，避免 RANSAC 被远处杂点带跑。"""
    h, w = shape[:2]
    diam = 2.0 * r
    if diam < CONFIG["min_candidate_diameter_px"]:
        return False
    if diam > CONFIG["max_candidate_diameter_px"]:
        return False
    if cx < -0.45 * w or cx > 1.45 * w or cy < -0.45 * h or cy > 1.45 * h:
        return False
    return True


def _ransac_one_circle(points, rng, shape):
    """在边缘点中用 RANSAC 找一条圆弧。"""
    n = len(points)
    if n < 3:
        return None, None

    tol = float(CONFIG["ransac_inlier_tol_px"])
    best_mask = None
    best_circle = None
    best_count = 0
    for _ in range(int(CONFIG["ransac_iterations"])):
        ids = rng.choice(n, 3, replace=False)
        circle = _circle_from_3_points(points[ids[0]], points[ids[1]], points[ids[2]])
        if circle is None:
            continue
        cx, cy, r = circle
        if not _circle_is_in_reasonable_range(cx, cy, r, shape):
            continue
        dist = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        mask = np.abs(dist - r) <= tol
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_mask = mask
            best_circle = circle

    if best_mask is None or best_count < CONFIG["ransac_min_inliers"]:
        return None, None

    inliers = points[best_mask]
    refined = _refine_circle_least_squares(inliers) or best_circle
    cx, cy, r = refined
    if not _circle_is_in_reasonable_range(cx, cy, r, shape):
        return None, None

    dist = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    mask = np.abs(dist - r) <= tol
    inliers = points[mask]
    if len(inliers) < CONFIG["ransac_min_inliers"]:
        return None, None

    contour_like = inliers.reshape(-1, 1, 2).astype(np.float32)
    coverage = _angle_coverage(contour_like, cx, cy)
    if coverage < CONFIG["ransac_min_coverage"]:
        return None, None

    radial_std = float(np.std(np.abs(dist[mask] - r))) if len(inliers) else 999.0
    score = float(len(inliers)) * coverage / (1.0 + radial_std / max(tol, 1.0))
    cand = {
        "cx": float(cx),
        "cy": float(cy),
        "diam_px": float(2.0 * r),
        "radius_px": float(r),
        "axis_major": float(2.0 * r),
        "axis_minor": float(2.0 * r),
        "eccentricity": 1.0,
        "coverage": float(coverage),
        "arc_len": float(len(inliers)),
        "radial_std": radial_std,
        "score": score,
        "method": "ransac",
    }
    return cand, mask


def _extract_ransac_candidates(evidence):
    """从连续证据图边缘中反复 RANSAC，提取多个圆环候选。"""
    edges = cv2.Canny(evidence, CONFIG["ransac_edge_low"], CONFIG["ransac_edge_high"])
    ys, xs = np.nonzero(edges)
    if len(xs) < CONFIG["ransac_min_inliers"]:
        return [], edges

    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) > CONFIG["ransac_max_points"]:
        idx = np.linspace(0, len(points) - 1, CONFIG["ransac_max_points"]).astype(np.int32)
        points = points[idx]

    rng = np.random.default_rng()
    remaining = points
    candidates = []
    for _ in range(int(CONFIG["ransac_max_circles"])):
        cand, _ = _ransac_one_circle(remaining, rng, evidence.shape)
        if cand is None:
            break
        candidates.append(cand)

        # 已识别圆弧附近的点移除，继续找其它半径的同心圆。
        dist = np.sqrt((remaining[:, 0] - cand["cx"]) ** 2 +
                       (remaining[:, 1] - cand["cy"]) ** 2)
        keep = np.abs(dist - cand["radius_px"]) > CONFIG["ransac_remove_tol_px"]
        remaining = remaining[keep]
        if len(remaining) < CONFIG["ransac_min_inliers"]:
            break
    return candidates, edges


def _score_radial_pattern(mask_norm, cx, cy, outer_radius_px, cos_a, sin_a):
    """以一个圆心和外圈半径为假设，统计五个同心半径上的黑色命中率。"""
    known = np.array(CONFIG["known_ring_diameters_mm"], dtype=np.float32)
    outer_mm = float(max(known))
    radii = outer_radius_px * (known / outer_mm)
    h, w = mask_norm.shape[:2]
    ring_width = max(1.0, float(CONFIG["radial_ring_width_px"]))
    offsets = np.array([-ring_width, 0.0, ring_width], dtype=np.float32)

    ring_scores = []
    ring_valids = []
    for r in radii:
        per_offset_vals = []
        valid_count = 0
        total_count = 0
        for off in offsets:
            rr = max(1.0, float(r + off))
            xs = np.rint(cx + rr * cos_a).astype(np.int32)
            ys = np.rint(cy + rr * sin_a).astype(np.int32)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            vals = np.zeros_like(cos_a, dtype=np.float32)
            vals[valid] = mask_norm[ys[valid], xs[valid]]
            per_offset_vals.append(vals)
            valid_count += int(np.count_nonzero(valid))
            total_count += int(len(valid))

        if total_count <= 0 or valid_count / float(total_count) < 0.25:
            ring_scores.append(0.0)
            ring_valids.append(0.0)
            continue
        vals = np.max(np.vstack(per_offset_vals), axis=0)
        mean_dark = float(np.mean(vals))
        hit_ratio = float(np.mean(vals >= CONFIG["radial_hit_thresh"]))
        ring_scores.append(0.65 * mean_dark + 0.35 * hit_ratio)
        ring_valids.append(valid_count / float(total_count))

    ring_scores = np.array(ring_scores, dtype=np.float32)
    strong = ring_scores >= CONFIG["radial_ring_min_score"]
    strong_count = int(np.count_nonzero(strong))
    if strong_count < CONFIG["radial_min_ring_hits"]:
        return 0.0, ring_scores.tolist(), strong_count

    score = float(np.mean(ring_scores[strong])) * min(1.0, strong_count / 3.0)
    score *= float(np.mean(ring_valids)) if ring_valids else 0.0
    return score, ring_scores.tolist(), strong_count


def _radial_scan_result(dark, seed_candidates=None):
    """在画面中心附近搜索同心比例，作为椭圆/RANSAC 都不稳时的兜底。"""
    mask_norm = dark.astype(np.float32) / 255.0
    h, w = dark.shape[:2]
    angles = np.linspace(0.0, 2.0 * np.pi, int(CONFIG["radial_angles"]), endpoint=False)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)

    search = int(CONFIG["radial_center_search_px"])
    step = max(2, int(CONFIG["radial_center_step_px"]))
    bases = [(w * 0.5, h * 0.5)]
    if seed_candidates:
        for cand in sorted(seed_candidates, key=lambda c: c.get("score", 0.0), reverse=True)[:3]:
            bases.append((float(cand["cx"]), float(cand["cy"])))

    center_set = set()
    centers = []
    for bx, by in bases:
        local_search = search if (bx, by) == bases[0] else max(step * 3, search // 2)
        for yy in range(int(by - local_search), int(by + local_search) + 1, step):
            for xx in range(int(bx - local_search), int(bx + local_search) + 1, step):
                if xx < -0.2 * w or xx > 1.2 * w or yy < -0.2 * h or yy > 1.2 * h:
                    continue
                key = (int(round(xx / step)), int(round(yy / step)))
                if key not in center_set:
                    center_set.add(key)
                    centers.append((float(xx), float(yy)))

    min_r = int(CONFIG["radial_outer_radius_min_px"])
    max_r = int(min(CONFIG["radial_outer_radius_max_px"], max(w, h)))
    r_step = max(2, int(CONFIG["radial_outer_radius_step_px"]))
    radii = list(range(min_r, max_r + 1, r_step))

    def eval_grid(center_list, radius_list):
        best_local = None
        for cx, cy in center_list:
            for outer_r in radius_list:
                score, ring_scores, ring_hits = _score_radial_pattern(
                    mask_norm, cx, cy, float(outer_r), cos_a, sin_a)
                if best_local is None or score > best_local["radial_score"]:
                    best_local = {
                        "cx": float(cx),
                        "cy": float(cy),
                        "outer_radius_px": float(outer_r),
                        "radial_score": float(score),
                        "ring_scores": ring_scores,
                        "ring_hits": int(ring_hits),
                    }
        return best_local

    best = eval_grid(centers, radii)
    if best is None or best["radial_score"] <= 0:
        return None

    # 粗搜后在最优点附近细搜，减少网格步长造成的圆心抖动。
    refine_step = max(1, int(CONFIG["radial_refine_step_px"]))
    refine_r_step = max(1, int(CONFIG["radial_refine_radius_step_px"]))
    refine_centers = []
    for yy in range(int(best["cy"] - step), int(best["cy"] + step) + 1, refine_step):
        for xx in range(int(best["cx"] - step), int(best["cx"] + step) + 1, refine_step):
            refine_centers.append((float(xx), float(yy)))
    refine_radii = list(range(int(best["outer_radius_px"] - r_step),
                              int(best["outer_radius_px"] + r_step) + 1,
                              refine_r_step))
    refine_radii = [r for r in refine_radii if min_r <= r <= max_r]
    refined = eval_grid(refine_centers, refine_radii)
    if refined is not None and refined["radial_score"] >= best["radial_score"]:
        best = refined

    if best["radial_score"] < CONFIG["radial_min_score"]:
        return None

    known = [float(v) for v in CONFIG["known_ring_diameters_mm"]]
    outer_mm = max(known)
    px_per_mm = (2.0 * best["outer_radius_px"]) / outer_mm
    mm_per_px = 1.0 / px_per_mm
    matches = []
    for km, ring_score in zip(known, best["ring_scores"]):
        matches.append({
            "known_mm": km,
            "diam_px": km * px_per_mm,
            "err_px": 0.0,
            "tol_px": 0.0,
            "score": float(ring_score),
            "cx": best["cx"],
            "cy": best["cy"],
        })

    confidence = min(1.0, best["radial_score"] * 2.0 + best["ring_hits"] * 0.08)
    total_score = best["radial_score"] * 1000000.0 + best["ring_hits"] * 100000.0
    return {
        "cx": best["cx"],
        "cy": best["cy"],
        "radius_px": best["outer_radius_px"],
        "px_per_mm": px_per_mm,
        "mm_per_px": mm_per_px,
        "matches": matches,
        "match_count": int(best["ring_hits"]),
        "total_score": total_score,
        "cluster_score": total_score,
        "observed": [],
        "confidence": confidence,
        "fallback_single": False,
        "method": "radial",
        "candidate_count": 0,
        "candidates": seed_candidates or [],
        "mask": dark,
        "radial_score": best["radial_score"],
        "ring_scores": best["ring_scores"],
    }


def _rank_result(result):
    """统一比较不同算法的结果，优先选择多圆匹配、非兜底、置信度高的结果。"""
    method_bonus = {"ellipse": 0.05, "ransac": 0.03, "radial": 0.00}.get(result.get("method"), 0.0)
    fallback_penalty = -0.18 if result.get("fallback_single") else 0.0
    confidence = float(result.get("confidence", 0.0))
    match_bonus = min(int(result.get("match_count", 0)), 5) * 0.035
    return confidence + match_bonus + method_bonus + fallback_penalty, float(result.get("cluster_score", 0.0))


def _result_offset_mm(result, frame_shape):
    """估算当前结果相对画面中心的偏差距离，用于决定是否启用慢速校验。"""
    h, w = frame_shape[:2]
    dx_px = float(result.get("cx", w * 0.5)) - w * 0.5
    dy_px = float(result.get("cy", h * 0.5)) - h * 0.5
    return ((dx_px * dx_px + dy_px * dy_px) ** 0.5) * float(result.get("mm_per_px", 0.0))


def _should_verify_with_radial(result, frame_shape):
    """字母干扰/远偏时，椭圆拟合可能看似有结果但中心不稳，需要径向同心校验。"""
    if result is None:
        return True
    if result.get("fallback_single"):
        return True
    if float(result.get("confidence", 0.0)) < float(CONFIG.get("radial_trigger_confidence", 0.45)):
        return True
    if int(result.get("match_count", 0)) < int(CONFIG.get("radial_trigger_match_count", 4)):
        return True
    return _result_offset_mm(result, frame_shape) >= float(CONFIG.get("radial_trigger_offset_mm", 30.0))


def _select_best_result(results):
    """在普通候选和径向同心校验之间选最终结果。"""
    if not results:
        return None

    ranked_best = max(results, key=_rank_result)
    radial_results = [r for r in results if r.get("method") == "radial"]
    if not radial_results:
        return ranked_best

    radial_best = max(radial_results, key=_rank_result)
    non_radial = [r for r in results if r.get("method") != "radial"]
    normal_best = max(non_radial, key=_rank_result) if non_radial else None
    if normal_best is None:
        return radial_best

    normal_weak = (
        normal_best.get("fallback_single") or
        float(normal_best.get("confidence", 0.0)) < float(CONFIG.get("radial_prefer_confidence_below", 0.70)) or
        int(normal_best.get("match_count", 0)) < int(CONFIG.get("radial_prefer_match_count_below", 4))
    )
    radial_good = (
        float(radial_best.get("radial_score", 0.0)) >= float(CONFIG.get("radial_prefer_min_score", 0.14)) and
        int(radial_best.get("match_count", 0)) >= int(CONFIG.get("radial_prefer_min_hits", 3))
    )
    if normal_weak and radial_good:
        return radial_best

    return ranked_best


def find_ring_center(frame, dark=None, evidence=None):
    """返回同心圆中心和自动比例估算结果。

    识别顺序：轮廓椭圆拟合 → RANSAC 圆弧拟合 → 径向同心比例扫描。
    """
    _require_cv2()
    if dark is None or evidence is None:
        built_mask, built_evidence, _ = _build_ring_evidence(frame)
        if dark is None:
            dark = built_mask
        if evidence is None:
            evidence = built_evidence

    candidates, dark = _extract_ring_candidates(frame, dark=dark)
    all_candidates = list(candidates)
    results = []

    ellipse_best = _best_fit_from_candidates(candidates, dark, "ellipse")
    if ellipse_best is not None:
        results.append(ellipse_best)

    best_now = _select_best_result(results)
    far_offset = (
        best_now is not None and
        _result_offset_mm(best_now, frame.shape) >= float(CONFIG.get("radial_trigger_offset_mm", 30.0))
    )
    need_ransac = CONFIG.get("enable_ransac", True) and (
        best_now is None or best_now.get("fallback_single") or
        best_now.get("confidence", 0.0) < 0.65 or far_offset
    )
    if need_ransac:
        ransac_candidates, _ = _extract_ransac_candidates(evidence)
        if ransac_candidates:
            all_candidates.extend(ransac_candidates)
            ransac_best = _best_fit_from_candidates(ransac_candidates, dark, "ransac")
            if ransac_best is not None:
                results.append(ransac_best)

    best_now = _select_best_result(results)
    need_radial = CONFIG.get("enable_radial_scan", True) and _should_verify_with_radial(best_now, frame.shape)
    if need_radial:
        radial_best = _radial_scan_result(evidence, seed_candidates=all_candidates)
        if radial_best is not None:
            results.append(radial_best)

    if not results:
        return None

    best = _select_best_result(results)
    best["candidates"] = all_candidates
    best["mask"] = evidence
    return best


def _reference_ring_axes(result, diam_mm, mm_per_px):
    """按最终圆心画固定物理直径参考环；有椭圆候选时沿用现场透视形状。"""
    if mm_per_px <= 0.0:
        return None

    nominal_diam_px = float(diam_mm) / float(mm_per_px)
    if nominal_diam_px <= 2.0:
        return None

    debug_ellipse = result.get("debug_ellipse") if result else None
    if debug_ellipse:
        ew = max(2.0, float(debug_ellipse.get("w", nominal_diam_px)))
        eh = max(2.0, float(debug_ellipse.get("h", nominal_diam_px)))
        major = max(ew, eh)
        if major > 2.0:
            axis_w = nominal_diam_px * ew / major / 2.0
            axis_h = nominal_diam_px * eh / major / 2.0
            return max(2, int(round(axis_w))), max(2, int(round(axis_h))), float(debug_ellipse.get("angle", 0.0))

    r = max(2, int(round(nominal_diam_px / 2.0)))
    return r, r, 0.0


def _draw_observed_fit_rings(display, result):
    """画图像里真实拟合到的圆/椭圆候选；绿色线只代表视觉拟合，不代表物理比例。"""
    if not CONFIG.get("draw_observed_fit_rings", True):
        return False

    matches = result.get("matches", []) if result else []
    if not matches:
        return False

    for item in sorted(matches, key=lambda m: float(m.get("diam_px", 0.0))):
        cx = int(round(float(item.get("cx", result["cx"]))))
        cy = int(round(float(item.get("cy", result["cy"]))))
        ew = max(2, int(round(float(item.get("ellipse_w", item.get("diam_px", 0.0))) / 2.0)))
        eh = max(2, int(round(float(item.get("ellipse_h", item.get("diam_px", 0.0))) / 2.0)))
        angle = float(item.get("ellipse_angle", 0.0))
        cv2.ellipse(display, (cx, cy), (ew, eh), angle, 0, 360, (0, 255, 0), 2)
        cv2.circle(display, (cx, cy), 2, (0, 200, 0), -1)
    return True


def _draw_manual_reference_rings(display, result):
    """只用于调试判断：以最终圆心为中心画 5 个已知直径参考环，不代表候选匹配数量。"""
    if not CONFIG.get("draw_manual_reference_rings", True):
        return
    if not result or result.get("scale_source") != "manual":
        return

    mm_per_px = float(result.get("mm_per_px", CONFIG.get("manual_mm_per_px", 0.0)))
    if mm_per_px <= 0.0:
        return

    cx = int(round(result["cx"]))
    cy = int(round(result["cy"]))
    scores = list(result.get("fixed_ring_scores", []))
    visible_flags = list(result.get("fixed_ring_visible_flags", []))
    min_score = float(CONFIG.get("manual_center_refine_min_score", 0.08))

    for idx, diam_mm in enumerate(CONFIG.get("known_ring_diameters_mm", [])):
        if idx < len(visible_flags) and not visible_flags[idx]:
            continue
        axes = _reference_ring_axes(result, float(diam_mm), mm_per_px)
        if axes is None:
            continue
        ax, ay, angle = axes
        score = scores[idx] if idx < len(scores) else 0.0
        color = (0, 255, 0) if score >= min_score else (0, 165, 255)
        thickness = 2 if score >= min_score else 1
        cv2.ellipse(display, (cx, cy), (ax, ay), angle, 0, 360, color, thickness)

        label_x = min(max(cx + ax + 4, 0), display.shape[1] - 52)
        label_y = min(max(cy - ay + 14, 14), display.shape[0] - 6)
        cv2.putText(display, f"{int(diam_mm)}", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def _draw_debug(frame, result=None, candidates=None):
    """绘制同心圆调试画面。"""
    display = frame.copy()
    h, w = display.shape[:2]
    cv2.line(display, (w // 2 - 18, h // 2), (w // 2 + 18, h // 2), (255, 255, 0), 1)
    cv2.line(display, (w // 2, h // 2 - 18), (w // 2, h // 2 + 18), (255, 255, 0), 1)
    cv2.circle(display, (w // 2, h // 2), 5, (255, 255, 0), -1)

    if candidates and CONFIG.get("show_candidate_circles", False):
        for cand in sorted(candidates, key=lambda c: c["score"], reverse=True)[:CONFIG["debug_max_candidates"]]:
            cx = int(round(cand["cx"]))
            cy = int(round(cand["cy"]))
            r = int(round(cand["radius_px"]))
            cv2.circle(display, (cx, cy), max(2, r), (80, 120, 255), 1)
            cv2.circle(display, (cx, cy), 2, (80, 120, 255), -1)

    if result:
        cx = int(round(result["cx"]))
        cy = int(round(result["cy"]))
        r = int(round(result["radius_px"]))
        method = result.get("method", "unknown")
        debug_ellipse = result.get("debug_ellipse")
        if result.get("scale_source") == "manual" and _draw_observed_fit_rings(display, result):
            pass
        elif result.get("scale_source") == "manual" and CONFIG.get("draw_manual_reference_rings", True):
            _draw_manual_reference_rings(display, result)
        elif CONFIG.get("draw_selected_ellipse", True) and debug_ellipse:
            ex = int(round(debug_ellipse["cx"]))
            ey = int(round(debug_ellipse["cy"]))
            ew = max(2, int(round(debug_ellipse["w"] / 2.0)))
            eh = max(2, int(round(debug_ellipse["h"] / 2.0)))
            cv2.ellipse(display, (ex, ey), (ew, eh), float(debug_ellipse["angle"]),
                        0, 360, (0, 255, 0), 2)
        elif result.get("scale_source") != "manual":
            cv2.circle(display, (cx, cy), max(2, r), (0, 255, 0), 2)
        cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)
        cv2.line(display, (w // 2, h // 2), (cx, cy), (0, 255, 255), 2)
        offset = _offset_from_result(result)
        cv2.putText(display, f"dx={offset['dx_mm']:+.1f}mm dy={offset['dy_mm']:+.1f}mm",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if offset.get("scale_source") == "manual":
            support_text = ",".join(f"{m['diam_px']:.0f}px" for m in sorted(offset["matches"], key=lambda x: x["diam_px"]))
            cv2.putText(display, f"scale={offset['mm_per_px']:.3f} mm/px convert-only",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            diam_text = ",".join(str(int(m["known_mm"])) for m in sorted(offset["matches"], key=lambda x: x["known_mm"]))
            cv2.putText(display, f"scale={offset['mm_per_px']:.3f} mm/px rings={offset['match_count']}",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if offset.get("scale_source") == "manual":
            cv2.putText(display, f"method={method} conf={offset['confidence']:.2f} support=[{support_text}]",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(display, f"method={method} conf={offset['confidence']:.2f} [{diam_text}]",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if offset["fallback_single"]:
            cv2.putText(display, "fallback: 210mm outer arc",
                        (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
        elif offset.get("scale_source") == "manual":
            if offset.get("fixed_ring_visible_count", 0) > 0:
                cv2.putText(display, f"center={offset.get('fixed_center_score', 0.0):.2f} hits={offset.get('fixed_center_hits', 0)}/{offset.get('fixed_ring_visible_count', 0)}",
                            (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
            else:
                cv2.putText(display, f"fit support={offset.get('match_count', 0)}",
                            (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
        elif method == "radial":
            cv2.putText(display, f"radial={result.get('radial_score', 0.0):.2f}",
                        (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    else:
        cv2.putText(display, "NO RING", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return display


# ============ 摄像头单例 ============
_cam_cache = {"cam": None}
_cam_lock = threading.RLock()
_bad_devices = {}
_last_known = {"result": None, "time": 0.0}
_prepare_thread = None


def _available_usb_devices():
    """返回当前可尝试的 USB 设备；近期黑屏/读帧失败的设备短暂冷却。"""
    now = time.time()
    ordered = apply_usb_env()
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


def _drop_cam_locked(cam, reason):
    """释放异常 USB 摄像头，下一次读帧会重新枚举 1/0。"""
    if cam is None:
        return
    print(f"[ring] USB摄像头 device={cam.device} 异常: {reason}，准备重开/切换", flush=True)
    _bad_devices[cam.device] = time.time() + float(CONFIG.get("device_retry_cooldown_s", 8.0))
    if _cam_cache.get("cam") is cam:
        try:
            cam.stop()
        finally:
            _cam_cache["cam"] = None
            _last_known["result"] = None
            _last_known["time"] = 0.0


def _get_cam():
    with _cam_lock:
        if _cam_cache["cam"] is None:
            ordered_devices = _available_usb_devices()

            last_error = None
            for dev in ordered_devices:
                cam = USBCamera(dev, CONFIG["width"], CONFIG["height"], CONFIG["fps"])
                try:
                    cam.start()
                except Exception as e:
                    last_error = e
                    _bad_devices[dev] = time.time() + float(CONFIG.get("device_retry_cooldown_s", 8.0))
                    continue
                CONFIG["usb_device"] = dev
                _cam_cache["cam"] = cam
                print(f"[ring] USB摄像头已打开: device={dev}", flush=True)
                break

            if _cam_cache["cam"] is None:
                raise RuntimeError(f"无法打开USB摄像头，已尝试 {ordered_devices}: {last_error}")
        return _cam_cache["cam"]


def _read_usb_frame():
    """统一 USB 读帧入口；黑屏/读帧失败时释放并重新尝试 1/0。"""
    attempts = max(1, len(apply_usb_env()))
    for _ in range(attempts):
        try:
            with _cam_lock:
                cam = _get_cam()
                frame = cam.read()
                if _frame_is_valid(frame):
                    return frame
                with cam._lock:
                    no_frame_age = time.time() - cam._latest_time if cam._latest_time > 0.0 else time.time() - getattr(cam, "_started_time", time.time())
                if no_frame_age < float(CONFIG.get("camera_no_frame_reopen_s", 20.0)):
                    return None
                _drop_cam_locked(cam, "读帧失败或黑屏")
        except RuntimeError as exc:
            print(f"[ring] USB摄像头读帧失败: {exc}", flush=True)
        time.sleep(0.05)
    return None


def _warmup_usb_camera(verbose=False):
    """定位前预读几帧，丢掉刚打开/刚切换 USB 后的黑屏、旧帧和曝光波动。"""
    need = max(0, int(CONFIG.get("camera_warmup_frames", 0)))
    if need <= 0:
        return True
    timeout = max(0.1, float(CONFIG.get("camera_warmup_timeout_s", 1.2)))
    start = time.time()
    got = 0
    while got < need and (time.time() - start) < timeout:
        frame = _read_usb_frame()
        if frame is not None:
            got += 1
        else:
            time.sleep(0.03)
    if verbose and got < need:
        print(f"  [WARN] ring 摄像头热身帧不足: {got}/{need}")
    return got > 0


def prepare_camera_async(verbose=True):
    """后台预热 ring 摄像头，避免到放置点才打开 USB 导致等待。"""
    global _prepare_thread

    def worker():
        try:
            ok = _warmup_usb_camera(verbose=verbose)
            if verbose:
                print(f"[ring] 后台预热 {'OK' if ok else '未拿到足够帧，后续继续尝试'}", flush=True)
        except Exception as exc:
            if verbose:
                print(f"[ring] 后台预热失败: {exc}", flush=True)

    if _prepare_thread is not None and _prepare_thread.is_alive():
        return
    _prepare_thread = threading.Thread(target=worker, name="ring-camera-prewarm", daemon=True)
    _prepare_thread.start()


def detect():
    hits = []
    start = time.time()
    frames = CONFIG["detect_frames"]
    timeout = CONFIG["detect_timeout_s"]

    while len(hits) < frames and (time.time() - start) < timeout:
        frame = _read_usb_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        result = find_ring_center(frame)
        if result:
            hits.append(result)

    if len(hits) < CONFIG["detect_min_hits"]:
        if _last_known["result"] is not None:
            age = time.time() - _last_known["time"]
            if age < CONFIG["last_known_max_age_s"]:
                return _last_known["result"]
        return None

    xs = sorted([h["cx"] for h in hits])
    ys = sorted([h["cy"] for h in hits])
    rs = sorted([h["radius_px"] for h in hits])
    mm_per_pxs = sorted([h["mm_per_px"] for h in hits])
    confidences = sorted([h["confidence"] for h in hits])
    n = len(hits)
    cx = xs[n // 2]
    cy = ys[n // 2]
    r = rs[n // 2]
    mm_per_px = mm_per_pxs[n // 2]
    confidence = confidences[n // 2]

    if len(hits) >= 3:
        recent_xs = [h["cx"] for h in hits[-3:]]
        if max(recent_xs) - min(recent_xs) > CONFIG["consistency_max_diff"]:
            if _last_known["result"] is not None and \
               time.time() - _last_known["time"] < CONFIG["last_known_max_age_s"]:
                return _last_known["result"]
            return None
        recent_ys = [h["cy"] for h in hits[-3:]]
        if max(recent_ys) - min(recent_ys) > CONFIG["consistency_max_diff"]:
            if _last_known["result"] is not None and \
               time.time() - _last_known["time"] < CONFIG["last_known_max_age_s"]:
                return _last_known["result"]
            return None

    # 取与中位圆心最接近的一帧作为匹配明细来源，避免 P 打印的 hits/可见环数和实际圆心错帧。
    detail = min(
        hits,
        key=lambda h: (
            (float(h["cx"]) - float(cx)) ** 2 +
            (float(h["cy"]) - float(cy)) ** 2 +
            0.05 * (float(h["radius_px"]) - float(r)) ** 2
        )
    )
    result = {
        "cx": int(round(cx)),
        "cy": int(round(cy)),
        "radius_px": float(r),
        "mm_per_px": float(mm_per_px),
        "confidence": float(confidence),
        "method": detail.get("method", "unknown"),
        "scale_source": detail.get("scale_source", "auto"),
        "match_count": int(detail.get("match_count", 0)),
        "matches": detail.get("matches", []),
        "rejected_matches": detail.get("rejected_matches", []),
        "raw_match_count": int(detail.get("raw_match_count", detail.get("match_count", 0))),
        "observed_count": int(detail.get("observed_count", detail.get("raw_match_count", detail.get("match_count", 0)))),
        "fixed_center_score": float(detail.get("fixed_center_score", 0.0)),
        "fixed_center_hits": int(detail.get("fixed_center_hits", 0)),
        "fixed_ring_visible_count": int(detail.get("fixed_ring_visible_count", 0)),
        "fixed_ring_visible_flags": detail.get("fixed_ring_visible_flags", []),
        "fixed_ring_scores": detail.get("fixed_ring_scores", []),
        "fallback_single": bool(detail.get("fallback_single", False)),
        "candidate_count": int(detail.get("candidate_count", 0)),
        "radial_score": float(detail.get("radial_score", 0.0)),
        "ring_scores": detail.get("ring_scores", []),
        "debug_ellipse": detail.get("debug_ellipse"),
    }
    _last_known["result"] = result
    _last_known["time"] = time.time()
    return result


# ============ 自愈 ============
def _is_comm_alive():
    try:
        if hasattr(comm, "is_started") and comm.is_started():
            return True
    except:
        pass
    try:
        pose = comm.get_pose(max_age=CONFIG["recover_pose_age_max"])
        return pose is not None
    except:
        return False


def _recover_comm(verbose=True):
    if verbose:
        print("  [recover] 关闭 comm...")
    try:
        comm.shutdown()
    except:
        pass
    time.sleep(0.5)

    if verbose:
        print("  [recover] 重开 comm...")
    try:
        port = resolve_comm_port(CONFIG.get("comm_port"))
        CONFIG["comm_port"] = port
        if verbose:
            print(f"  [recover] 串口 = {port}")
        comm.init(port, CONFIG["comm_baud"])
    except Exception as e:
        if verbose:
            print(f"  [recover] init err: {e}")
        return False
    time.sleep(1.0)

    if verbose:
        print("  [recover] 发启动脉冲...")
    try:
        comm.run(5.0)
    except:
        pass
    time.sleep(1.0)

    for i in range(5):
        if _is_comm_alive():
            if verbose:
                print(f"  [recover] 恢复成功 ({i+1}/5)")
            return True
        time.sleep(0.3)

    return False


def ensure_comm_alive(verbose=True):
    if _is_comm_alive():
        return True
    if verbose:
        print("  [comm 死了, 自动恢复]")
    for attempt in range(CONFIG["recover_max_retry"]):
        if verbose:
            print(f"  [恢复 {attempt+1}/{CONFIG['recover_max_retry']}]")
        if _recover_comm(verbose=verbose):
            return True
        time.sleep(1.0)
    return False


def _safe_goto(target_x, target_y, timeout=20.0, max_retry=2):
    for attempt in range(max_retry):
        try:
            ok = comm.goto(target_x, target_y, timeout=timeout)
            if ok:
                return True
        except Exception as e:
            print(f"  [goto {attempt+1}] err: {e}")
        time.sleep(0.3)

    if not ensure_comm_alive(verbose=True):
        return False
    time.sleep(0.5)

    try:
        return comm.goto(target_x, target_y, timeout=timeout)
    except Exception as e:
        print(f"  [goto] 恢复后仍失败: {e}")
        return False


def _safe_fine_move(dx_mm, dy_mm, timeout=20.0, max_retry=1, allow_resend=False):
    """发送下位机已有 dx/dy 微调命令，并等待到位反馈。

    FINE_MOVE 是真实运动命令，超时后下位机可能仍在执行。
    默认不重发，避免同一段前进/后退被执行两次。
    """
    if not ensure_comm_alive(verbose=True):
        return False

    attempts = max(1, int(max_retry)) if allow_resend else 1
    for attempt in range(attempts):
        try:
            seen_seq = comm.response_seq()
            comm.send_fine_move(dx_mm, dy_mm)
            if hasattr(comm, "wait_status_after"):
                status = comm.wait_status_after(comm.TYPE_CMD_FINE_RESP, seen_seq, timeout)
                if status == 1:
                    return True
                if status == 0:
                    print(f"  [fine_move {attempt+1}] 收到 FINE_RESP=0，下位机判定本次微调失败/超时")
                else:
                    print(f"  [fine_move {attempt+1}] 未收到 FINE_RESP，等待超时")
            else:
                ok = comm.wait_for_after(comm.TYPE_CMD_FINE_RESP, seen_seq, timeout)
                if ok:
                    return True
                print(f"  [fine_move {attempt+1}] 下位机返回失败或等待超时")
        except Exception as e:
            print(f"  [fine_move {attempt+1}] err: {e}")
            if not allow_resend:
                return False
        if not allow_resend:
            print("  [fine_move] 已下发过运动命令，不重发，避免重复运动")
            return False
        time.sleep(0.3)

    if not allow_resend:
        return False

    if not ensure_comm_alive(verbose=True):
        return False
    time.sleep(0.5)

    try:
        seen_seq = comm.response_seq()
        comm.send_fine_move(dx_mm, dy_mm)
        return comm.wait_for_after(comm.TYPE_CMD_FINE_RESP, seen_seq, timeout)
    except Exception as e:
        print(f"  [fine_move] 恢复后仍失败: {e}")
        return False


def _safe_body_pos_move(dx_mm, dy_mm, timeout=3.0):
    """使用下位机 BODY_POS_MOVE：四轮 Emm 位置模式开环相对位移。"""
    if not ensure_comm_alive(verbose=True):
        return False
    try:
        ok = comm.body_pos_move(dx_mm, dy_mm, timeout=timeout)
        if not ok:
            print("  [body_pos_move] 下位机返回失败或等待超时")
        return bool(ok)
    except Exception as e:
        print(f"  [body_pos_move] err: {e}")
        return False


def _fresh_pose_after_wait(max_age=0.5, wait_s=0.25):
    """等待一小段时间，读取开环位置模式更新后的最新 POSE。"""
    deadline = time.monotonic() + max(0.0, float(wait_s))
    pose = comm.get_pose(max_age=max_age)
    while pose is None and time.monotonic() < deadline:
        time.sleep(0.03)
        pose = comm.get_pose(max_age=max_age)
    return pose


def _body_delta_target_from_pose(pose, dx_body_m, dy_body_m):
    """把车体相对位移换算成地图目标点；yaw 使用下位机 CW+ 约定。"""
    x, y, yaw_cw_deg = pose
    yaw_ccw_rad = math.radians(-float(yaw_cw_deg))
    cy = math.cos(yaw_ccw_rad)
    sy = math.sin(yaw_ccw_rad)
    target_x = float(x) + float(dx_body_m) * cy - float(dy_body_m) * sy
    target_y = float(y) + float(dx_body_m) * sy + float(dy_body_m) * cy
    return target_x, target_y


def _safe_body_back_goto(back_mm, timeout=20.0, verbose=True):
    """后退使用普通 GOTO：按当前 yaw 将车体 dy 负方向换算成地图目标点。"""
    pose = _fresh_pose_after_wait(max_age=0.8, wait_s=0.35)
    if pose is None:
        print("  [WARN] 没有新鲜 POSE，后退回退为 FINE_MOVE，避免用错误地图点")
        return _safe_fine_move(0.0, -back_mm, timeout=timeout)
    target_x, target_y = _body_delta_target_from_pose(pose, 0.0, -float(back_mm) / 1000.0)
    if verbose:
        print(f"  后退 GOTO 目标 = ({target_x:.4f}, {target_y:.4f}), pose=({pose[0]:.4f}, {pose[1]:.4f}, yaw={pose[2]:.1f}°)")
    return _safe_goto(target_x, target_y, timeout=timeout, max_retry=1)


def _post_align_forward_command(forward_mm):
    """返回对准后前推 BODY_POS 指令；包含摄像头安装左右补偿。"""
    trim_dx = float(CONFIG.get("post_align_trim_dx_mm", 0.0))
    trim_dy = float(CONFIG.get("post_align_trim_dy_mm", 0.0))
    return trim_dx, float(forward_mm) + trim_dy, trim_dx, trim_dy


def _wait(seconds, label, verbose=True):
    if verbose:
        print(f"  === {label} (等 {seconds}s) ===")
    for i in range(int(seconds)):
        time.sleep(1.0)
        if verbose:
            remaining = int(seconds) - i - 1
            print(f"  等待中... 剩余 {remaining} 秒")


def _offset_from_result(result):
    """把检测结果转换为以画面中心为参考的 dx/dy。"""
    h, w = CONFIG["height"], CONFIG["width"]
    cx_target, cy_target = w // 2, h // 2

    cx = int(result["cx"])
    cy = int(result["cy"])
    dx_px = cx - cx_target
    dy_px = cy - cy_target
    mm_per_px = float(result["mm_per_px"])

    # 约定: 下位机 FINE_MOVE 使用车体坐标，dx=右移，dy=前进。
    # 摄像头安装方向未标定时，现场可用 X/Y/S 热键翻转符号或交换轴。
    mapped = pixel_offset_to_body_mm(dx_px, dy_px, mm_per_px)
    return {
        "cx": cx,
        "cy": cy,
        "radius_px": float(result["radius_px"]),
        "dx_px": int(dx_px),
        "dy_px": int(dy_px),
        "dx_mm": float(mapped["dx_mm"]),
        "dy_mm": float(mapped["dy_mm"]),
        "raw_dx_mm": float(mapped["raw_dx_mm"]),
        "raw_dy_mm": float(mapped["raw_dy_mm"]),
        "mm_per_px": mm_per_px,
        "confidence": float(result.get("confidence", 0.0)),
        "method": result.get("method", "unknown"),
        "scale_source": result.get("scale_source", "auto"),
        "match_count": int(result.get("match_count", 0)),
        "matches": result.get("matches", []),
        "rejected_matches": result.get("rejected_matches", []),
        "raw_match_count": int(result.get("raw_match_count", result.get("match_count", 0))),
        "observed_count": int(result.get("observed_count", result.get("raw_match_count", result.get("match_count", 0)))),
        "fixed_center_score": float(result.get("fixed_center_score", 0.0)),
        "fixed_center_hits": int(result.get("fixed_center_hits", 0)),
        "fixed_ring_visible_count": int(result.get("fixed_ring_visible_count", 0)),
        "fixed_ring_visible_flags": result.get("fixed_ring_visible_flags", []),
        "fixed_ring_scores": result.get("fixed_ring_scores", []),
        "fallback_single": bool(result.get("fallback_single", False)),
        "radial_score": float(result.get("radial_score", 0.0)),
        "ring_scores": result.get("ring_scores", []),
    }


def detect_offset(verbose=False):
    """检测同心圆圆心，返回 dx/dy(mm) 和自动估算比例。"""
    result = detect()
    if result is None:
        return None
    offset = _offset_from_result(result)
    if verbose:
        _print_detection(offset)
    return offset


def _offset_distance_mm(offset):
    """返回当前圆心偏差距离，单位 mm。"""
    if offset is None:
        return float("inf")
    return float((offset["dx_mm"] * offset["dx_mm"] + offset["dy_mm"] * offset["dy_mm"]) ** 0.5)


def _alignment_tolerances_mm():
    """返回 X/Y 分轴到位容差；缺省时兼容旧的总容差配置。"""
    base = float(CONFIG.get("align_tolerance_mm", 3.0))
    tol_x = float(CONFIG.get("align_tolerance_x_mm", base))
    tol_y = float(CONFIG.get("align_tolerance_y_mm", base))
    return tol_x, tol_y


def _alignment_error_text(offset):
    tol_x, tol_y = _alignment_tolerances_mm()
    return (f"dx={offset['dx_mm']:+.1f}/{tol_x:.1f}mm, "
            f"dy={offset['dy_mm']:+.1f}/{tol_y:.1f}mm, "
            f"dist={_offset_distance_mm(offset):.1f}mm")


def _detect_offset_with_retries(verbose=False, retries=None, label=""):
    """识别同心圆；偶发没识别到时重试几次，避免单帧/短时卡顿直接中断流程。"""
    if retries is None:
        retries = int(CONFIG.get("align_detect_retry_count", 4))
    attempts = max(1, int(retries))
    sleep_s = float(CONFIG.get("align_detect_retry_sleep_s", 0.15))
    prefix = f" {label}" if label else ""
    for attempt in range(attempts):
        offset = detect_offset(verbose=verbose)
        if offset is not None:
            if verbose and attempt > 0:
                print(f"  [OK] 同心圆重试识别成功{prefix}: {attempt + 1}/{attempts}")
            return offset
        if verbose:
            print(f"  [WARN] 未检测到可靠同心圆{prefix}: {attempt + 1}/{attempts}")
        if attempt + 1 < attempts and sleep_s > 0.0:
            time.sleep(sleep_s)
    return None


def _limited_dxdy(dx_mm, dy_mm):
    """限制单次视觉微调距离，防止未标定时一次跑偏过大。"""
    limit = float(CONFIG.get("fine_move_max_step_mm", 30.0))
    dist = float((dx_mm * dx_mm + dy_mm * dy_mm) ** 0.5)
    if limit <= 0.0 or dist <= limit:
        return float(dx_mm), float(dy_mm), 1.0
    scale = limit / max(dist, 1e-6)
    return float(dx_mm * scale), float(dy_mm * scale), float(scale)


def _offset_is_aligned(offset):
    """判断圆心偏差是否已经足够小；X/Y 分轴容差避免左右误差被前后容差放宽。"""
    dead_px = int(CONFIG.get("dead_zone_px", 0))
    if dead_px > 0 and abs(offset["dx_px"]) <= dead_px and abs(offset["dy_px"]) <= dead_px:
        return True

    tol_x, tol_y = _alignment_tolerances_mm()
    if tol_x <= 0.0 or tol_y <= 0.0:
        return False
    return abs(float(offset["dx_mm"])) <= tol_x and abs(float(offset["dy_mm"])) <= tol_y


def _fine_adjust_by_offset(offset, verbose=True):
    """只按检测得到的 dx/dy 做一次微调，不附带前进动作。"""
    if _offset_is_aligned(offset):
        if verbose:
            print(f"  [OK] 已在允许误差内，无需移动: {_alignment_error_text(offset)}")
        return True

    if not ensure_comm_alive(verbose=verbose):
        return False

    if verbose:
        print("\n  === 单次同心圆 dx/dy 微调 ===")
    cmd_dx, cmd_dy, scale = _limited_dxdy(offset["dx_mm"], offset["dy_mm"])
    if verbose:
        print(f"  计算微调    = dx {offset['dx_mm']:+.1f} mm, dy {offset['dy_mm']:+.1f} mm")
        if scale < 0.999:
            print(f"  限幅后下发  = dx {cmd_dx:+.1f} mm, dy {cmd_dy:+.1f} mm")
        else:
            print(f"  下发微调    = dx {cmd_dx:+.1f} mm, dy {cmd_dy:+.1f} mm")

    align_ok = _safe_fine_move(cmd_dx, cmd_dy,
                               timeout=CONFIG["correct_timeout_s"])
    if verbose:
        print(f"  [fine move] {'OK' if align_ok else 'FAIL'}")
    return bool(align_ok)


def _confirm_aligned_twice(first_offset, verbose=True):
    """进容差后再复检一次；连续两次都在容差内才放行前推。"""
    confirm_frames = max(1, int(CONFIG.get("align_confirm_frames", 1)))
    if confirm_frames <= 1:
        return True, first_offset

    last_offset = first_offset
    sleep_s = float(CONFIG.get("align_confirm_sleep_s", 0.12))
    for idx in range(2, confirm_frames + 1):
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        offset = _detect_offset_with_retries(verbose=verbose, label=f"对准确认{idx}/{confirm_frames}")
        if offset is None:
            if verbose:
                print(f"  [WARN] 对准确认 {idx}/{confirm_frames} 未检测到可靠同心圆，继续微调流程")
            return False, last_offset
        last_offset = offset
        if not _offset_is_aligned(offset):
            if verbose:
                print(f"  [WARN] 对准确认 {idx}/{confirm_frames} 未进容差: "
                      f"{_alignment_error_text(offset)}，继续微调流程")
            return False, last_offset
        if verbose:
            print(f"  [OK] 对准确认 {idx}/{confirm_frames} 通过: {_alignment_error_text(offset)}")
    return True, last_offset


def _move_by_offset(offset, verbose=True):
    """按检测得到的 dx/dy 微调；是否前进由配置控制。"""
    if not _fine_adjust_by_offset(offset, verbose=verbose):
        return False

    forward_mm = float(CONFIG["post_align_offset_y_mm"])
    if not CONFIG.get("auto_forward_after_align", False):
        if verbose:
            print(f"  [safe] 自动 dy+{forward_mm:.0f} 已关闭；需要前进时按 F")
        return True

    if not ensure_comm_alive(verbose=verbose):
        return False
    if verbose:
        print(f"\n  === 定位后前进 dy +{forward_mm:.0f}mm ===")
    ok = _safe_fine_move(0.0, forward_mm,
                         timeout=float(CONFIG.get("push_move_timeout_s", CONFIG["correct_timeout_s"])))
    if verbose:
        print(f"  [forward dy] {'OK' if ok else 'FAIL'}")
    return ok


def align_iterative_to_ring(max_iterations=None, verbose=True):
    """多帧识别 + 限幅微调 + 再识别，适合到达三环附近后的最终定位。"""
    if max_iterations is None:
        max_iterations = int(CONFIG.get("max_align_iterations", 10))
    max_iterations = max(1, int(max_iterations))
    _warmup_usb_camera(verbose=verbose)

    for i in range(max_iterations):
        if verbose:
            print(f"\n[ring] 视觉定位迭代 {i + 1}/{max_iterations}")
        offset = _detect_offset_with_retries(verbose=verbose, label=f"迭代{i + 1}")
        if offset is None:
            if verbose:
                print("  [WARN] 本轮未检测到可靠同心圆，继续下一轮")
            continue
        if _offset_is_aligned(offset):
            if verbose:
                print("  [OK] 圆心已经对齐")
            return True
        if not _fine_adjust_by_offset(offset, verbose=verbose):
            return False
        time.sleep(float(CONFIG.get("post_fine_move_settle_s", 0.25)))

    if verbose:
        print("\n[ring] 末次复查")
    offset = _detect_offset_with_retries(verbose=verbose, label="末次复查")
    if offset is None:
        return False
    ok = _offset_is_aligned(offset)
    if verbose:
        print(f"  [final check] {'OK' if ok else '仍有偏差，建议再次执行'}")
    return ok


def _push_forward_then_back(verbose=True, back_mm=None):
    """对准后执行推送：前推走电机位置模式，默认后退仍走 FINE_MOVE。"""
    forward_mm = float(CONFIG["post_align_offset_y_mm"])
    if back_mm is None:
        back_mm = float(CONFIG.get("post_align_back_y_mm", 0.0))
    else:
        back_mm = float(back_mm)
    push_timeout = float(CONFIG.get("push_move_timeout_s", CONFIG["correct_timeout_s"]))
    back_timeout = float(CONFIG.get("back_move_timeout_s", CONFIG["correct_timeout_s"]))
    if not ensure_comm_alive(verbose=verbose):
        return False
    cmd_dx, cmd_dy, trim_dx, trim_dy = _post_align_forward_command(forward_mm)
    if verbose:
        print(f"\n  === 对准完成，车体前推 +{forward_mm:.0f}mm（BODY_POS 位置模式） ===")
        if abs(trim_dx) > 1e-6 or abs(trim_dy) > 1e-6:
            print(f"  安装补偿    = dx {trim_dx:+.1f}mm, dy {trim_dy:+.1f}mm；实际下发 dx {cmd_dx:+.1f}mm, dy {cmd_dy:+.1f}mm")
    ok = _safe_body_pos_move(cmd_dx, cmd_dy, timeout=push_timeout)
    if verbose:
        print(f"  [forward body_pos] {'OK' if ok else 'FAIL'}")
    if not ok:
        return False
    settle_s = float(CONFIG.get("post_push_quiesce_s", 0.0))
    if settle_s > 0.0:
        try:
            comm.send_velocity(0.0, 0.0, 0.0)
        except Exception as exc:
            if verbose:
                print(f"  [WARN] 前推后零速度帧发送失败，继续后续流程: {exc}")
        if verbose:
            print(f"  前推后停稳等待 {settle_s:.2f}s，避免微调/位置模式残留影响后退")
        time.sleep(settle_s)
    if back_mm <= 0.0:
        return True
    if verbose:
        print(f"\n  === 前进完成，后退 dy -{back_mm:.0f}mm（FINE_MOVE） ===")
    ok = _safe_fine_move(0.0, -back_mm, timeout=back_timeout)
    if verbose:
        print(f"  [backward fine_move] {'OK' if ok else 'FAIL'}")
    if not ok:
        # 固定后退属于放置后的安全退出动作；下位机可能实际已移动但因内部闭环超时返回0。
        # 不重发，避免二次后退；继续交给调用方按实测点 sync_pose 修正坐标。
        print("  [WARN] 后退未确认完成；不重发，继续主流程并依赖后续 sync_pose 修正坐标")
        return True
    return True


def align_checked_then_forward(max_moves=None, verbose=True, back_mm=None, pre_push_wait=None):
    """A 键流程：先微调；前推前可等待并行动作确认，避免槽位未到位就推出。"""
    if max_moves is None:
        max_moves = int(CONFIG.get("checked_align_max_moves", 10))
    max_moves = max(1, int(max_moves))
    settle_s = float(CONFIG.get("post_fine_move_settle_s", 0.25))

    time.sleep(float(CONFIG.get("pre_cmd_sleep_s", 0.0)))
    _warmup_usb_camera(verbose=verbose)
    moves_done = 0
    final_offset = None
    best_offset = None
    best_dist = float("inf")
    check_idx = 0
    max_check_rounds = max_moves + max(2, int(CONFIG.get("align_detect_retry_count", 4)))

    while moves_done <= max_moves and check_idx < max_check_rounds:
        check_idx += 1
        if verbose:
            print(f"\n[ring] A 键定位复检 {check_idx}/{max_check_rounds}，已微调 {moves_done}/{max_moves} 次")
        offset = _detect_offset_with_retries(verbose=verbose, label=f"A键复检{check_idx}")
        if offset is None:
            if verbose:
                print("  [WARN] 当前没有可靠同心圆检测，先不退出，继续尝试")
            continue

        final_offset = offset
        dist_mm = _offset_distance_mm(offset)
        if dist_mm < best_dist:
            best_dist = dist_mm
            best_offset = dict(offset)

        if _offset_is_aligned(offset):
            confirmed, confirm_offset = _confirm_aligned_twice(offset, verbose=verbose)
            final_offset = confirm_offset
            if confirmed:
                if verbose:
                    print("  [OK] 双帧确认通过，圆心已对准")
                break
            offset = confirm_offset
            dist_mm = _offset_distance_mm(offset)
            if dist_mm < best_dist:
                best_dist = dist_mm
                best_offset = dict(offset)
        if moves_done >= max_moves:
            if verbose:
                print("  [WARN] 已达到微调次数上限，仍未进容差；按当前最小误差继续前推/后退")
            break
        if not _fine_adjust_by_offset(offset, verbose=verbose):
            if verbose:
                print("  [WARN] 本次微调失败；停止继续微调，但仍执行前推/后退")
            break
        moves_done += 1
        if settle_s > 0.0:
            time.sleep(settle_s)

    if final_offset is None:
        if verbose:
            print("  [WARN] 本轮始终未检测到可靠同心圆；按流程仍执行前进/后退")
    elif not _offset_is_aligned(final_offset):
        if verbose:
            if best_offset is not None:
                print(f"  [WARN] 未进分轴容差；当前误差 {_alignment_error_text(final_offset)}，"
                      f"过程最小总误差 {best_dist:.1f}mm，继续前推/后退")
            else:
                print("  [WARN] 最终未确认完全对准，但继续前进/后退")
    if pre_push_wait is not None:
        if verbose:
            print("\n  === 前推前确认并行动作完成 ===")
        try:
            ready = bool(pre_push_wait())
        except Exception as exc:
            if verbose:
                print(f"  [parallel wait] FAIL: {exc}")
            return False
        if verbose:
            print(f"  [parallel wait] {'OK' if ready else 'FAIL'}")
        if not ready:
            return False

    return _push_forward_then_back(verbose=verbose, back_mm=back_mm)


_trackbars_ready = False


def _noop_trackbar(_value):
    pass


def _setup_debug_trackbars():
    """创建现场调参滑条；只影响本次运行，不写回文件。"""
    global _trackbars_ready
    if _trackbars_ready or not CONFIG.get("enable_debug_trackbars", True):
        return
    win = CONFIG["trackbar_window_name"]
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 520, 260)
    cv2.createTrackbar("evidPct", win, int(CONFIG["evidence_percentile"]), 99, _noop_trackbar)
    cv2.createTrackbar("blackHatK", win, int(CONFIG["blackhat_kernel_px"]), 99, _noop_trackbar)
    cv2.createTrackbar("minCov%", win, int(CONFIG["min_arc_coverage"] * 100), 50, _noop_trackbar)
    cv2.createTrackbar("matchTol%", win, int(CONFIG["match_rel_tol"] * 100), 45, _noop_trackbar)
    cv2.createTrackbar("ransacTol x10", win, int(CONFIG["ransac_inlier_tol_px"] * 10), 120, _noop_trackbar)
    cv2.createTrackbar("radialMin%", win, int(CONFIG["radial_min_score"] * 100), 80, _noop_trackbar)
    _trackbars_ready = True


def _read_debug_trackbars():
    """读取滑条并更新 CONFIG，便于现场直接试阈值。"""
    if not _trackbars_ready or not CONFIG.get("enable_debug_trackbars", True):
        return
    win = CONFIG["trackbar_window_name"]
    try:
        CONFIG["evidence_percentile"] = min(99, max(50, cv2.getTrackbarPos("evidPct", win)))
        CONFIG["blackhat_kernel_px"] = max(9, cv2.getTrackbarPos("blackHatK", win))
        CONFIG["min_arc_coverage"] = max(0.01, cv2.getTrackbarPos("minCov%", win) / 100.0)
        CONFIG["match_rel_tol"] = max(0.05, cv2.getTrackbarPos("matchTol%", win) / 100.0)
        CONFIG["ransac_inlier_tol_px"] = max(1.0, cv2.getTrackbarPos("ransacTol x10", win) / 10.0)
        CONFIG["radial_min_score"] = max(0.01, cv2.getTrackbarPos("radialMin%", win) / 100.0)
    except cv2.error:
        pass


def _print_mapping():
    """打印当前图像偏差到车体 dx/dy 的映射。"""
    print(
        "[ring] 映射: "
        f"dx_sign={CONFIG['camera_dx_sign']:+.0f}, "
        f"dy_sign={CONFIG['camera_dy_sign']:+.0f}, "
        f"swap_xy={CONFIG['camera_swap_xy']}, "
        f"单次限幅={CONFIG['fine_move_max_step_mm']:.0f}mm, "
        f"ransac={CONFIG['enable_ransac']}, radial={CONFIG['enable_radial_scan']}, "
        f"候选圆={'显示' if CONFIG.get('show_candidate_circles', False) else '隐藏'}"
    )


def _forward_after_align(verbose=True):
    """沿车体 dy 正方向单独前进固定距离，使用电机位置模式。"""
    if not ensure_comm_alive(verbose=verbose):
        return False
    forward_mm = float(CONFIG["post_align_offset_y_mm"])
    cmd_dx, cmd_dy, trim_dx, trim_dy = _post_align_forward_command(forward_mm)
    if verbose:
        print(f"\n  === 手动前进 dy +{forward_mm:.0f}mm（BODY_POS 位置模式） ===")
        if abs(trim_dx) > 1e-6 or abs(trim_dy) > 1e-6:
            print(f"  安装补偿    = dx {trim_dx:+.1f}mm, dy {trim_dy:+.1f}mm；实际下发 dx {cmd_dx:+.1f}mm, dy {cmd_dy:+.1f}mm")
    ok = _safe_body_pos_move(cmd_dx, cmd_dy,
                             timeout=float(CONFIG.get("push_move_timeout_s", CONFIG["correct_timeout_s"])))
    if verbose:
        print(f"  [forward body_pos] {'OK' if ok else 'FAIL'}")
    return ok


def preview():
    """实时显示圆环识别画面，并提供现场校准热键。"""
    _require_cv2()
    cv2.namedWindow(CONFIG["debug_window_name"], cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONFIG["debug_mask_window_name"], cv2.WINDOW_NORMAL)
    _setup_debug_trackbars()
    print("[ring] 预览已启动: q退出, p打印, A微调到位+前进105mm+后退200mm, M多次定位不前进, F前进105mm, G单次微调+前进")
    print("[ring] 调参热键: X/Y翻符号, S交换轴, C显示/隐藏候选圆, R/D开关兜底")
    print("[ring] Ring Evidence 是局部暗线证据图，不要求画面只剩黑线")
    _print_mapping()
    last_result = None
    last_candidates = []
    last_result_time = 0.0
    next_detect_time = 0.0
    last_evidence = None
    while True:
        _read_debug_trackbars()
        frame = _read_usb_frame()
        if frame is None:
            time.sleep(0.03)
            continue
        now = time.time()
        result = None
        candidates = last_candidates
        if now >= next_detect_time:
            mask, evidence, _ = _build_ring_evidence(frame)
            last_evidence = evidence
            result = find_ring_center(frame, dark=mask, evidence=evidence)
            next_detect_time = now + float(CONFIG["preview_detect_interval_s"])
            if result:
                last_evidence = result.get("mask", evidence)
                candidates = result.get("candidates", [])
                last_result = result
                last_candidates = candidates
                last_result_time = now
            else:
                candidates = []
                last_candidates = []

        display_result = last_result if (now - last_result_time) <= CONFIG["preview_result_max_age_s"] else None
        if result:
            last_result = result
        display = _draw_debug(frame, display_result, candidates)
        cv2.imshow(CONFIG["debug_window_name"], display)
        if last_evidence is not None:
            cv2.imshow(CONFIG["debug_mask_window_name"], last_evidence)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in (ord('p'), ord('P')):
            if display_result is None:
                print("[ring] 当前没有有效检测")
            else:
                _print_detection(_offset_from_result(display_result))
                _print_mapping()
        if key in (ord('x'), ord('X')):
            set_camera_mapping(dx_sign=-float(CONFIG["camera_dx_sign"]))
            _print_mapping()
        if key in (ord('y'), ord('Y')):
            set_camera_mapping(dy_sign=-float(CONFIG["camera_dy_sign"]))
            _print_mapping()
        if key in (ord('s'), ord('S')):
            set_camera_mapping(swap_xy=not CONFIG.get("camera_swap_xy", False))
            _print_mapping()
        if key in (ord('c'), ord('C')):
            CONFIG["show_candidate_circles"] = not CONFIG.get("show_candidate_circles", False)
            _print_mapping()
        if key in (ord('r'), ord('R')):
            CONFIG["enable_ransac"] = not CONFIG.get("enable_ransac", False)
            _print_mapping()
        if key in (ord('d'), ord('D')):
            CONFIG["enable_radial_scan"] = not CONFIG.get("enable_radial_scan", False)
            _print_mapping()
        if key in (ord('f'), ord('F')):
            ok = _forward_after_align(verbose=True)
            print(f"[ring] F键前进 {'OK' if ok else 'FAIL'}")
        if key in (ord('m'), ord('M')):
            ok = align_iterative_to_ring(verbose=True)
            print(f"[ring] M键多次定位 {'OK' if ok else 'FAIL'}")
        if key in (ord('a'), ord('A')):
            ok = align_checked_then_forward(verbose=True)
            print(f"[ring] A键微调到位+前进+后退 {'OK' if ok else 'FAIL'}")
        if key in (ord('g'), ord('G')):
            offset = detect_offset(verbose=True)
            if offset is None:
                print("[ring] 当前没有可靠检测，不能移动")
            else:
                old_auto = CONFIG.get("auto_forward_after_align", False)
                CONFIG["auto_forward_after_align"] = True
                try:
                    ok = _move_by_offset(offset, verbose=True)
                finally:
                    CONFIG["auto_forward_after_align"] = old_auto
                print(f"[ring] G键微调+前进 {'OK' if ok else 'FAIL'}")
    cv2.destroyWindow(CONFIG["debug_window_name"])
    cv2.destroyWindow(CONFIG["debug_mask_window_name"])
    if _trackbars_ready:
        cv2.destroyWindow(CONFIG["trackbar_window_name"])


def _print_detection(offset):
    """打印检测结果，方便现场调试比例是否匹配到正确圆。"""
    print(f"  识别方法    = {offset.get('method', 'unknown')}")
    print(f"  ring center = ({offset['cx']}, {offset['cy']})")
    print(f"  ring radius = {offset['radius_px']:.1f} px")
    print(f"  像素偏差    = dx {offset['dx_px']:+d} px, dy {offset['dy_px']:+d} px")
    if offset.get("scale_source") == "manual":
        print(f"  手动比例    = {offset['mm_per_px']:.3f} mm/px，仅用于 dx/dy 换算，未参与圆心拟合")
        if offset.get("matches"):
            desc = ", ".join(f"{m['diam_px']:.1f}px" for m in sorted(offset["matches"], key=lambda x: x["diam_px"]))
            print(f"  图像拟合支撑 = {offset['match_count']}/{offset.get('observed_count', offset['raw_match_count'])} 个候选: {desc}")
        if offset.get("fixed_ring_visible_count", 0) > 0 and offset.get("fixed_ring_scores"):
            scores = ", ".join(f"{s:.2f}" for s in offset["fixed_ring_scores"])
            print(
                f"  固定比例兜底 = score {offset.get('fixed_center_score', 0.0):.2f}, "
                f"hits {offset.get('fixed_center_hits', 0)}/{offset.get('fixed_ring_visible_count', 0)}, "
                f"rings [{scores}]"
            )
    else:
        print(f"  自动比例    = {offset['mm_per_px']:.3f} mm/px, confidence={offset['confidence']:.2f}")
    if offset["fallback_single"]:
        print(f"  比例来源    = 单段圆弧兜底，按 {CONFIG['single_arc_default_diameter_mm']}mm 外圈处理")
    elif offset["matches"] and offset.get("scale_source") != "manual":
        desc = []
        for m in sorted(offset["matches"], key=lambda x: x["known_mm"]):
            desc.append(f"{int(m['known_mm'])}mm≈{m['diam_px']:.1f}px")
        print("  直径匹配    = " + ", ".join(desc))
        if offset.get("raw_match_count", offset["match_count"]) != offset["match_count"]:
            print(f"  有效匹配    = {offset['match_count']}/{offset['raw_match_count']} 个直径")
    if offset.get("rejected_matches"):
        desc = []
        for m in sorted(offset["rejected_matches"], key=lambda x: x["known_mm"]):
            desc.append(
                f"{int(m['known_mm'])}mm(dev={m.get('center_dev_px', 0.0):.1f}px>"
                f"{m.get('center_tol_px', 0.0):.1f}px)"
            )
        print("  剔除候选    = " + ", ".join(desc) + "，疑似内部字母/假圆")
    if offset.get("method") == "radial" and offset.get("ring_scores"):
        scores = ", ".join(f"{s:.2f}" for s in offset["ring_scores"])
        print(f"  径向得分    = {offset.get('radial_score', 0.0):.2f} [{scores}]")


def _print_manual_scale_trials(offset):
    """按候选像素直径列出手动比例试算值；只辅助人工标定，不参与自动控制。"""
    matches = offset.get("matches") or []
    if not matches:
        return
    known = [float(v) for v in CONFIG.get("known_ring_diameters_mm", [])]
    if not known:
        return

    print("  比例试算    = 肉眼确认该候选是哪一圈后，取对应 mm/px")
    for item in sorted(matches, key=lambda x: x.get("diam_px", 0.0)):
        diam_px = float(item.get("diam_px", 0.0))
        if diam_px <= 1.0:
            continue
        trials = []
        for known_mm in known:
            trials.append(f"{int(known_mm)}mm:{known_mm / diam_px:.3f}")
        print(f"    {diam_px:.1f}px -> " + ", ".join(trials))


def align_once_to_ring(verbose=True):
    """到达圆环附近后，只根据当前画面圆心偏差执行一次 dx/dy 定位。"""
    time.sleep(CONFIG["pre_cmd_sleep_s"])
    offset = _detect_offset_with_retries(verbose=verbose, label="单次定位")
    if offset is None:
        if verbose:
            print("  [FAIL] 没找到同心圆环")
        return False
    return _move_by_offset(offset, verbose=verbose)


# ============ ★ 公共 API: 3 步对齐 ============
def align_to_ring(verbose=True):
    """
    兼容旧 3 步对齐流程；内部统一使用下位机 FINE_MOVE 微调。
      1. 检测圆环，按画面中心偏差做一次限幅 dx/dy 对准
      2. 等 2 秒，沿车体 dy 正方向前进固定距离
      3. 等 2 秒，再沿车体 dy 负方向退回指定像素对应距离

    返回 True=全部成功, False=某步失败
    """
    if not ensure_comm_alive(verbose=verbose):
        return False

    time.sleep(CONFIG["pre_cmd_sleep_s"])
    offset = _detect_offset_with_retries(verbose=verbose, label="旧3步流程")
    if offset is None:
        if verbose:
            print("  [WARN] 没找到同心圆环，跳过步骤1微调，仍继续前进/后退")
        k_mm_per_px = float(CONFIG.get("manual_mm_per_px", 0.25))
    else:
        if verbose:
            print("\n  === 步骤 1: 视觉 dx/dy 对准 ===")
        if not _move_by_offset(offset, verbose=verbose):
            if verbose:
                print("  [step 1] WARN: 微调失败，仍继续前进/后退")
        else:
            if verbose:
                print("  [step 1] OK")
        k_mm_per_px = float(offset.get("mm_per_px", CONFIG.get("manual_mm_per_px", 0.25)))
        if k_mm_per_px <= 0.0:
            k_mm_per_px = float(CONFIG.get("manual_mm_per_px", 0.25))

    # === 等 1→2 ===
    _wait(CONFIG["step1_to_step2_pause_s"], "步骤1→2 等待", verbose)

    # === 步骤 2: y 正方向前进固定距离 ===
    forward_mm = float(CONFIG["post_align_offset_y_mm"])

    if verbose:
        print(f"\n  === 步骤 2: 车体 dy +{forward_mm:.0f}mm ===")

    time.sleep(CONFIG["pre_cmd_sleep_s"])
    ok = _safe_fine_move(0.0, forward_mm,
                         timeout=float(CONFIG.get("push_move_timeout_s", CONFIG["correct_timeout_s"])))
    if not ok:
        if verbose:
            print("  [step 2] FAIL")
        return False
    if verbose:
        print("  [step 2] OK")

    # === 等 2→3 ===
    _wait(CONFIG["step2_to_step3_pause_s"], "步骤2→3 等待", verbose)

    # === 步骤 3: 退后 500 像素 ===
    back_offset_mm = CONFIG["step3_back_offset_px"] * k_mm_per_px

    if verbose:
        print(f"\n  === 步骤 3: 退后 {CONFIG['step3_back_offset_px']} 像素 ===")
        print(f"  {CONFIG['step3_back_offset_px']}px 实际  = {back_offset_mm:.1f} mm")

    time.sleep(CONFIG["pre_cmd_sleep_s"])
    ok = _safe_fine_move(0.0, -back_offset_mm,
                         timeout=float(CONFIG.get("back_move_timeout_s", CONFIG["correct_timeout_s"])))
    if verbose:
        print(f"  [step 3] {'OK' if ok else 'FAIL'}")
    return ok


def close():
    """关闭摄像头 (main 退出时调用)"""
    with _cam_lock:
        if _cam_cache["cam"] is not None:
            _cam_cache["cam"].stop()
            _cam_cache["cam"] = None
        _bad_devices.clear()


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1].lower() in ("once", "detect"):
            offset = detect_offset(verbose=True)
            if offset is None:
                print("[ring] 未检测到同心圆环")
            else:
                print(f"[ring] dx={offset['dx_mm']:+.1f}mm, dy={offset['dy_mm']:+.1f}mm")
        else:
            preview()
    finally:
        close()
