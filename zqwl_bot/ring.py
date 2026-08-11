#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""黑色圆环识别 + 一次性对齐 (自愈版: comm 死了自动重启)"""
import cv2
import numpy as np
import time
import comm


CONFIG = {
    "usb_device": 1,
    "width": 640,
    "height": 480,
    "fps": 30,

    "black_thresh": 140,
    "min_area": 2000,
    "max_area": 200000,
    "min_circularity": 0.6,
    "min_eccentricity": 0.4,
    "min_contour_points": 5,

    "ring_actual_radius_m": 0.025,
    "calib_step": 0.005,

    "detect_frames": 5,
    "detect_min_hits": 2,
    "detect_timeout_s": 1.5,
    "otsu_smooth_n": 3,
    "consistency_max_diff": 80,
    "last_known_max_age_s": 3.0,

    "dead_zone_px": 10,
    "correct_timeout_s": 20.0,

    "pre_cmd_sleep_s": 0.3,
    "comm_port": "/dev/ttyCH341USB0",
    "comm_baud": 115200,

    # 自愈参数
    "recover_pose_age_max": 1.0,     # POSE 帧超过这秒数没更新就认为死了
    "recover_max_retry": 3,          # 自愈重试次数
}


# ============ 摄像头 ============
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


# ============ 圆环检测 ============
def find_ring_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(blur, CONFIG["black_thresh"], 255,
                            cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = 0
    for cnt in contours:
        if len(cnt) < CONFIG["min_contour_points"]:
            continue

        area = cv2.contourArea(cnt)
        if area < CONFIG["min_area"] or area > CONFIG["max_area"]:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < CONFIG["min_circularity"]:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        (cx, cy), (w, h), angle = ellipse
        axis_major = max(w, h)
        axis_minor = min(w, h)
        if axis_major == 0:
            continue

        eccentricity = axis_minor / axis_major
        if eccentricity < CONFIG["min_eccentricity"]:
            continue

        r_avg = (axis_major + axis_minor) / 2.0

        score = area * eccentricity
        if score > best_score:
            best = (int(cx), int(cy), int(r_avg), int(area),
                   int(axis_major), int(axis_minor), float(eccentricity))
            best_score = score
    return best


# ============ 摄像头单例 ============
_cam_cache = {"cam": None}
_last_known = {"result": None, "time": 0.0}


def _get_cam():
    if _cam_cache["cam"] is None:
        cam = USBCamera(CONFIG["usb_device"], CONFIG["width"],
                       CONFIG["height"], CONFIG["fps"])
        cam.start()
        _cam_cache["cam"] = cam
    return _cam_cache["cam"]


def detect():
    cam = _get_cam()
    hits = []
    start = time.time()
    frames = CONFIG["detect_frames"]
    timeout = CONFIG["detect_timeout_s"]

    while len(hits) < frames and (time.time() - start) < timeout:
        frame = cam.read()
        if frame is None:
            continue
        result = find_ring_center(frame)
        if result:
            hits.append((result[0], result[1], result[2]))

    if len(hits) < CONFIG["detect_min_hits"]:
        if _last_known["result"] is not None:
            age = time.time() - _last_known["time"]
            if age < CONFIG["last_known_max_age_s"]:
                return _last_known["result"]
        return None

    xs = sorted([h[0] for h in hits])
    ys = sorted([h[1] for h in hits])
    rs = sorted([h[2] for h in hits])
    n = len(hits)
    cx = xs[n // 2]
    cy = ys[n // 2]
    r = rs[n // 2]

    if len(hits) >= 3:
        recent_xs = [h[0] for h in hits[-3:]]
        if max(recent_xs) - min(recent_xs) > CONFIG["consistency_max_diff"]:
            if _last_known["result"] is not None and \
               time.time() - _last_known["time"] < CONFIG["last_known_max_age_s"]:
                return _last_known["result"]
            return None
        recent_ys = [h[1] for h in hits[-3:]]
        if max(recent_ys) - min(recent_ys) > CONFIG["consistency_max_diff"]:
            if _last_known["result"] is not None and \
               time.time() - _last_known["time"] < CONFIG["last_known_max_age_s"]:
                return _last_known["result"]
            return None

    result = (cx, cy, r)
    _last_known["result"] = result
    _last_known["time"] = time.time()
    return result


# ============ ★ 自愈: comm 死了自动重启 ============
def _is_comm_alive():
    """检查 comm 是否活着: POSE 帧有更新 + 能拿到"""
    try:
        pose = comm.get_pose(max_age=CONFIG["recover_pose_age_max"])
        return pose is not None
    except:
        return False


def _recover_comm(verbose=True):
    """
    ★ 自动恢复 comm:
    1. comm.shutdown() 关
    2. 等 0.5s
    3. comm.init() 重开
    4. 等 1s
    5. comm.run() 发启动脉冲 (PD15 500ms) - 等于按物理 RUN 键
    6. 等 1s
    7. 检查 POSE 帧是否恢复
    """
    if verbose:
        print("  [recover] 关闭 comm...")

    try:
        comm.shutdown()
    except Exception as e:
        if verbose:
            print(f"  [recover] shutdown err: {e}")
    time.sleep(0.5)

    if verbose:
        print("  [recover] 重开 comm...")

    try:
        comm.init(CONFIG["comm_port"], CONFIG["comm_baud"])
    except Exception as e:
        if verbose:
            print(f"  [recover] init err: {e}")
        return False
    time.sleep(1.0)

    if verbose:
        print("  [recover] 发启动脉冲...")

    try:
        comm.run(5.0)
    except Exception as e:
        if verbose:
            print(f"  [recover] run err: {e}")
    time.sleep(1.0)

    if verbose:
        print("  [recover] 检查 POSE 帧...")

    for i in range(5):
        if _is_comm_alive():
            if verbose:
                print(f"  [recover] 恢复成功 (尝试 {i+1}/5)")
            return True
        time.sleep(0.3)

    if verbose:
        print("  [recover] 失败, POSE 帧没恢复")
    return False


def ensure_comm_alive(verbose=True):
    """确保 comm 活着, 死了就自动恢复"""
    if _is_comm_alive():
        return True

    if verbose:
        print("  [comm 死了, 自动恢复]")

    for attempt in range(CONFIG["recover_max_retry"]):
        if verbose:
            print(f"  [恢复尝试 {attempt+1}/{CONFIG['recover_max_retry']}]")
        if _recover_comm(verbose=verbose):
            return True
        time.sleep(1.0)

    return False


# ============ safe_goto 带重试 ============
def _safe_goto(target_x, target_y, timeout=20.0, max_retry=2):
    """comm.goto 带重试, 失败时尝试 comm 重连"""
    for attempt in range(max_retry):
        try:
            ok = comm.goto(target_x, target_y, timeout=timeout)
            if ok:
                return True
        except Exception as e:
            print(f"  [goto attempt {attempt+1}] err: {e}")
        time.sleep(0.3)

    # 失败, 自动恢复
    print("  [goto] 失败, 自动恢复 comm...")
    if not ensure_comm_alive(verbose=True):
        return False
    time.sleep(0.5)

    try:
        ok = comm.goto(target_x, target_y, timeout=timeout)
        return ok
    except Exception as e:
        print(f"  [goto] 恢复后仍失败: {e}")
        return False


# ============ 对齐 ============
def align_to_ring(verbose=True):
    # ★ 1. 自检 comm, 死了自动恢复
    if not ensure_comm_alive(verbose=verbose):
        if verbose:
            print("  [FAIL] comm 没救活, 放弃")
        return False

    # 发命令前等
    time.sleep(CONFIG["pre_cmd_sleep_s"])

    # ★ 2. 找圆环, 失败也试着重连
    result = detect()
    if result is None:
        if verbose:
            print("  [圆环没找到, 试着重连再找]")
        if not ensure_comm_alive(verbose=False):
            return False
        result = detect()
        if result is None:
            if verbose:
                print("  [FAIL] 重连后还没找到圆环")
            return False

    h, w = CONFIG["height"], CONFIG["width"]
    cx_target, cy_target = w // 2, h // 2

    cx, cy, r_px = result
    dx_px = cx - cx_target
    dy_px = cy - cy_target

    if verbose:
        print(f"  ring center = ({cx}, {cy})")
        print(f"  ring radius = {r_px} px")
        print(f"  像素偏差    = dx {dx_px:+d} px, dy {dy_px:+d} px")

    if abs(dx_px) < CONFIG["dead_zone_px"] and \
       abs(dy_px) < CONFIG["dead_zone_px"]:
        if verbose:
            print(f"  [OK] 已在死区内 ({CONFIG['dead_zone_px']}px), 不动")
        return True

    if r_px <= 0:
        return False

    k_m_per_px = CONFIG["ring_actual_radius_m"] / r_px
    k_mm_per_px = k_m_per_px * 1000

    # ★ x 取反, y 不取反
    dx_mm = -dx_px * k_mm_per_px
    dy_mm = dy_px * k_mm_per_px
    dx_m = dx_mm / 1000.0
    dy_m = dy_mm / 1000.0

    if verbose:
        print(f"  标定系数    = {k_mm_per_px:.3f} mm/px "
              f"(calib={CONFIG['ring_actual_radius_m']*1000:.1f}mm)")
        print(f"  实际移动    = dx {dx_mm:+.1f} mm, dy {dy_mm:+.1f} mm")

    pose = comm.get_pose(max_age=1.0)
    if pose is None:
        if verbose:
            print("  [FAIL] 没拿到当前位姿 (comm 又死了?)")
        return False

    x_now, y_now, _ = pose
    target_x = x_now + dx_m
    target_y = y_now + dy_m

    if verbose:
        print(f"  当前位置    = ({x_now:.3f}, {y_now:.3f}) m")
        print(f"  目标位置    = ({target_x:.3f}, {target_y:.3f}) m")

    ok = _safe_goto(target_x, target_y, timeout=CONFIG["correct_timeout_s"])
    if verbose:
        print(f"  [align] {'OK' if ok else 'FAIL'}")
    return ok


# ============ 重连（手动） ============
def reconnect():
    print("[manual reconnect]")
    _recover_comm(verbose=True)


def close():
    if _cam_cache["cam"] is not None:
        _cam_cache["cam"].stop()
        _cam_cache["cam"] = None


# ============ 主循环 ============
def main():
    print("=" * 50)
    print("Black ring one-shot align (auto-recover)")
    print("=" * 50)
    print("q=quit  a=align  +/-=calib  r=manual reconnect")

    comm.init(CONFIG["comm_port"], CONFIG["comm_baud"])
    time.sleep(1.0)

    cam = _get_cam()
    cv2.namedWindow("Ring", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            result = find_ring_center(frame)
            display = frame.copy()
            h, w = frame.shape[:2]

            cv2.line(display, (w//2 - 30, h//2), (w//2 + 30, h//2),
                    (0, 255, 255), 1)
            cv2.line(display, (w//2, h//2 - 30), (w//2, h//2 + 30),
                    (0, 255, 255), 1)

            if result:
                cx, cy, r, area, major, minor, ecc = result
                cv2.ellipse(display, (cx, cy), (major // 2, minor // 2),
                           0, 0, 360, (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)
                dx = cx - w // 2
                dy = cy - h // 2
                cv2.putText(display,
                           f"({cx},{cy}) r={r}  dx={dx:+d} dy={dy:+d}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 0), 2)
            else:
                cv2.putText(display, "NO RING", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display, "q=quit  a=align  +/-=calib  r=reconnect",
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 0), 1)

            cv2.imshow("Ring", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                print("\n[align one-shot]")
                align_to_ring(verbose=True)
                print()
            elif key in (ord('+'), ord('=')):
                CONFIG["ring_actual_radius_m"] += CONFIG["calib_step"]
                print(f"  calib radius = {CONFIG['ring_actual_radius_m']*1000:.1f}mm")
            elif key in (ord('-'), ord('_')):
                CONFIG["ring_actual_radius_m"] = max(0.005,
                    CONFIG["ring_actual_radius_m"] - CONFIG["calib_step"])
                print(f"  calib radius = {CONFIG['ring_actual_radius_m']*1000:.1f}mm")
            elif key == ord('r'):
                reconnect()

    except KeyboardInterrupt:
        pass
    finally:
        close()
        cv2.destroyAllWindows()
        try:
            comm.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()