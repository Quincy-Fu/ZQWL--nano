#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ring.py — 黑色圆环识别 + 3步对齐（自包含完整版）
========================================================
- 50/90mm 两种尺寸（±15% 容差）
- 圆弧 + 整圆通用（minEnclosingCircle，不用 fitEllipse）
- 双圆环显示：主=金框实线粗，副=虚线细圈
- ←/→ 切 primary，a 对当前 primary
- 摄像头反装：x 取反，y 不取反
- ★ calib 按识别尺寸（不再固定 75mm）
- 串口自愈 + 重连

部署：/home/fuzr/Desktop/ZQWL--nano/zqwl_bot/test_ring.py
依赖：comm.py（同目录）
"""

import cv2
import numpy as np
import time
import comm


# ============================================================
# 配置
# ============================================================
CONFIG = {
    "comm_port": "/dev/ttyCH341USB0",
    "comm_baud": 115200,

    "usb_device": 1,
    "width": 640,
    "height": 480,
    "fps": 30,

    "black_thresh": 140,
    "min_area": 200,
    "max_area": 200000,
    "min_circularity": 0.20,
    "min_eccentricity": 0.0,
    "min_contour_points": 5,

    # ★ 不再固定 ring_actual_radius_m；calib 用 size_mm 算

    "detect_frames": 5,
    "detect_min_hits": 2,
    "detect_timeout_s": 1.5,
    "consistency_max_diff": 80,
    "last_known_max_age_s": 3.0,

    "dead_zone_px": 0,
    "correct_timeout_s": 20.0,
    "pre_cmd_sleep_s": 0.3,
    "step1_to_step2_pause_s": 2.0,
    "step2_to_step3_pause_s": 2.0,
    "post_align_offset_y_mm": 102,
    "step3_back_offset_px": 500,

    "recover_pose_age_max": 1.0,
    "recover_max_retry": 3,

    "ring_real_sizes_mm": [50, 90],
    "size_tolerance_pct": 15,
}

# 自标定（50mm = 207px）
K_MM_PER_PX = 50.0 / 207.0   # ≈ 0.2415


def _r_px_range(size_mm, tol=0.15):
    nom = size_mm / K_MM_PER_PX
    return (nom * (1 - tol), nom * (1 + tol), nom)


# ============================================================
# 摄像头
# ============================================================
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


def _get_cam():
    if _cam_cache["cam"] is None:
        cam = USBCamera(CONFIG["usb_device"], CONFIG["width"], CONFIG["height"], CONFIG["fps"])
        cam.start()
        _cam_cache["cam"] = cam
    return _cam_cache["cam"]


# ============================================================
# ★ 找所有圆环（圆弧 + 整圆通用）
# ============================================================
def find_all_rings(frame):
    if frame is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(blur, CONFIG["black_thresh"], 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
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

        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < 30 or r > 400:
            continue

        candidates.append({
            "cx": int(cx),
            "cy": int(cy),
            "r": int(r),
            "area": int(area),
            "circularity": round(circularity, 3),
        })

    candidates.sort(key=lambda x: x["r"], reverse=True)
    return candidates


def classify_rings(rings):
    """按 r_px 范围匹配 50/90mm"""
    classified = []
    for r in rings:
        r_px = r["r"]
        size_mm = None
        for s in CONFIG["ring_real_sizes_mm"]:
            mn, mx, _ = _r_px_range(s, CONFIG["size_tolerance_pct"] / 100.0)
            if mn <= r_px <= mx:
                size_mm = s
                break
        if size_mm is None:
            continue
        classified.append({**r, "size_mm": size_mm})
    return classified


def find_ring_center(frame):
    rings = find_all_rings(frame)
    classified = classify_rings(rings)
    if not classified:
        return None
    r = classified[0]
    return (r["cx"], r["cy"], r["r"], r["area"], r["circularity"])


# ============================================================
# 多帧众数
# ============================================================
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


# ============================================================
# 自愈
# ============================================================
def _is_comm_alive():
    try:
        pose = comm.get_pose(max_age=CONFIG["recover_pose_age_max"])
        return pose is not None
    except Exception:
        return False


def _recover_comm(verbose=True):
    if verbose:
        print("  [recover] 关闭 comm...")
    try:
        comm.shutdown()
    except Exception:
        pass
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
    except Exception:
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


def _wait(seconds, label, verbose=True):
    if verbose:
        print(f"  === {label} (等 {seconds}s) ===")
    for i in range(int(seconds)):
        time.sleep(1.0)
        if verbose:
            remaining = int(seconds) - i - 1
            print(f"  等待中... 剩余 {remaining} 秒")


# ============================================================
# 3 步对齐（calib 按识别尺寸，不固定 75mm）
# ============================================================
def align_to_ring(primary=None, verbose=True):
    """
    primary: dict {cx, cy, r, size_mm} 或 None（默认 detect 最大）
    """
    if not ensure_comm_alive(verbose=verbose):
        return False
    time.sleep(CONFIG["pre_cmd_sleep_s"])

    if primary is not None:
        cx, cy, r_px = primary["cx"], primary["cy"], primary["r"]
        size_mm = primary.get("size_mm", 0)
        if verbose:
            print(f"  [target] primary={size_mm}mm r={r_px}px @ ({cx},{cy})")
    else:
        result = detect()
        if result is None:
            if verbose:
                print("  [圆环没找到, 试着重连]")
            if not ensure_comm_alive(verbose=False):
                return False
            result = detect()
            if result is None:
                if verbose:
                    print("  [FAIL] 没找到圆环")
                return False
        cx, cy, r_px = result
        size_mm = 0
        if verbose:
            print(f"  [target] detect r={r_px}px @ ({cx},{cy})")

    h, w = CONFIG["height"], CONFIG["width"]
    cx_target, cy_target = w // 2, h // 2
    dx_px = cx - cx_target
    dy_px = cy - cy_target

    if verbose:
        print(f"  ring center = ({cx}, {cy})")
        print(f"  ring radius = {r_px} px")
        print(f"  像素偏差    = dx {dx_px:+d} px, dy {dy_px:+d} px")

    if r_px <= 0:
        return False

    # ★ calib 按 size_mm 来（不再固定 75mm）
    # 50mm 圆环 → calib = 50mm
    # 90mm 圆环 → calib = 90mm
    # default 没识别到尺寸 → 用 50mm 兜底
    calib_mm = size_mm if size_mm > 0 else 50
    k_m_per_px = calib_mm / 1000.0 / r_px
    k_mm_per_px = k_m_per_px * 1000

    # === 步骤 1: 对准 ===
    # 摄像头反装：x 取反，y 不取反
    dx_mm = -dx_px * k_mm_per_px
    dy_mm = dy_px * k_mm_per_px
    dx_m = dx_mm / 1000.0
    dy_m_align = dy_mm / 1000.0

    pose = comm.get_pose(max_age=1.0)
    if pose is None:
        if verbose:
            print("  [FAIL] 没拿到位姿")
        return False

    x_now, y_now, _ = pose
    target_x_align = x_now + dx_m
    target_y_align = y_now + dy_m_align

    if verbose:
        print(f"\n  === 步骤 1: 对准 ===")
        print(f"  calib       = {calib_mm}mm（按识别尺寸）")
        print(f"  标定系数    = {k_mm_per_px:.3f} mm/px")
        print(f"  移动        = dx {dx_mm:+.1f} mm, dy {dy_mm:+.1f} mm")
        print(f"  目标位置    = ({target_x_align:.3f}, {target_y_align:.3f}) m")

    ok = _safe_goto(target_x_align, target_y_align, timeout=CONFIG["correct_timeout_s"])
    if not ok:
        if verbose:
            print("  [step 1] FAIL")
        return False
    if verbose:
        print(f"  [step 1] OK")

    _wait(CONFIG["step1_to_step2_pause_s"], "步骤1→2 等待", verbose)

    # === 步骤 2: y +100mm ===
    target_y_step2 = target_y_align + CONFIG["post_align_offset_y_mm"] / 1000.0
    if verbose:
        print(f"\n  === 步骤 2: y +{CONFIG['post_align_offset_y_mm']}mm ===")
        print(f"  目标位置    = ({target_x_align:.3f}, {target_y_step2:.3f}) m")
    time.sleep(CONFIG["pre_cmd_sleep_s"])
    ok = _safe_goto(target_x_align, target_y_step2, timeout=CONFIG["correct_timeout_s"])
    if not ok:
        if verbose:
            print("  [step 2] FAIL")
        return False
    if verbose:
        print(f"  [step 2] OK")
    _wait(CONFIG["step2_to_step3_pause_s"], "步骤2→3 等待", verbose)

    # === 步骤 3: 退后 500 像素 ===
    back_offset_mm = CONFIG["step3_back_offset_px"] * k_mm_per_px
    target_y_step3 = target_y_step2 - back_offset_mm / 1000.0
    if verbose:
        print(f"\n  === 步骤 3: 退后 {CONFIG['step3_back_offset_px']} 像素 ===")
        print(f"  {CONFIG['step3_back_offset_px']}px 实际  = {back_offset_mm:.1f} mm")
        print(f"  目标位置    = ({target_x_align:.3f}, {target_y_step3:.3f}) m")
    time.sleep(CONFIG["pre_cmd_sleep_s"])
    ok = _safe_goto(target_x_align, target_y_step3, timeout=CONFIG["correct_timeout_s"])
    if verbose:
        print(f"  [step 3] {'OK' if ok else 'FAIL'}")
    return ok


def reconnect():
    print("[manual reconnect]")
    _recover_comm(verbose=True)


def close():
    if _cam_cache["cam"] is not None:
        _cam_cache["cam"].stop()
        _cam_cache["cam"] = None


# ============================================================
# main: 双圆环显示 + 切换 primary
# ============================================================
def main():
    print("=" * 60)
    print("Black ring align 3-step (50/90mm, 圆弧+整圆)")
    print("  1. 对准 (r 最大的圆环)")
    print("  2. 等2s, y+100mm")
    print("  3. 等2s, 退后 500 像素")
    print("=" * 60)
    print("q=quit  a=align  r=reconnect  ←/→ 切primary")

    comm.init(CONFIG["comm_port"], CONFIG["comm_baud"])
    time.sleep(1.0)
    cam = _get_cam()
    cv2.namedWindow("Ring", cv2.WINDOW_NORMAL)

    primary_idx = 0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            display = frame.copy()
            h, w = frame.shape[:2]

            # 中心十字
            cv2.line(display, (w//2 - 30, h//2), (w//2 + 30, h//2),
                     (0, 255, 255), 1)
            cv2.line(display, (w//2, h//2 - 30), (w//2, h//2 + 30),
                     (0, 255, 255), 1)

            # 找所有圆环 + 分类
            rings = find_all_rings(frame)
            classified = classify_rings(rings)

            if classified:
                primary_idx = primary_idx % len(classified)
            else:
                primary_idx = 0

            # 按尺寸上色
            size_to_color = {
                50: (255, 200, 0),    # 50mm 青色
                90: (0, 165, 255),    # 90mm 橙色
            }
            for i, r in enumerate(classified):
                base_color = size_to_color.get(r["size_mm"], (200, 200, 200))
                is_primary = (i == primary_idx)

                if is_primary:
                    cv2.circle(display, (r["cx"], r["cy"]), r["r"], base_color, 3, cv2.LINE_AA)
                    cv2.rectangle(display,
                                  (r["cx"] - r["r"] - 8, r["cy"] - r["r"] - 8),
                                  (r["cx"] + r["r"] + 8, r["cy"] + r["r"] + 8),
                                  (0, 255, 255), 2)
                    cv2.circle(display, (r["cx"], r["cy"]), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(display, (r["cx"], r["cy"]), r["r"], base_color, 1, cv2.LINE_4)
                    cv2.circle(display, (r["cx"], r["cy"]), 2, base_color, -1)

                prefix = "★" if is_primary else " "
                label = f"{prefix}#{i} {r['size_mm']}mm r={r['r']}"
                label_color = (0, 255, 255) if is_primary else base_color
                cv2.putText(display, label,
                            (r["cx"] + r["r"] + 5, r["cy"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

            # 顶部
            if classified:
                primary = classified[primary_idx]
                dx = primary["cx"] - w // 2
                dy = primary["cy"] - h // 2
                # ★ calib 按当前 primary 尺寸（实时显示）
                calib_mm = primary.get("size_mm", 50)
                k_mm_per_px = calib_mm / primary["r"]
                cv2.putText(display,
                           f"PRIMARY #{primary_idx} {primary['size_mm']}mm r={primary['r']}  "
                           f"dx={dx:+d} dy={dy:+d}  K={k_mm_per_px:.3f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                           (0, 255, 0), 2)
                if len(classified) > 1:
                    sub = "副: " + " | ".join(
                        f"#{i} {r['size_mm']}mm(r={r['r']})"
                        for i, r in enumerate(classified) if i != primary_idx
                    )
                    cv2.putText(display, sub,
                                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (200, 200, 200), 1)
            else:
                cv2.putText(display, "NO RING",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 0, 255), 2)

            # 底部
            cv2.putText(display, "q=quit  a=align  r=reconnect  ←/→ 切primary",
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 0), 1)
            # ★ 改成 calib=AUTO
            cv2.putText(display, "calib=AUTO (按识别尺寸)",
                       (w - 290, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                       (255, 0, 255), 2)

            cv2.imshow("Ring", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                if not classified:
                    print("\n[!] 没有圆环可对齐")
                else:
                    p = classified[primary_idx]
                    print(f"\n[align 3-step] primary=#{primary_idx} "
                          f"{p['size_mm']}mm r={p['r']} center=({p['cx']},{p['cy']})")
                    align_to_ring(primary=p, verbose=True)
                print()
            elif key == ord('r'):
                reconnect()
            elif key in (81, 2):       # ←
                if classified:
                    primary_idx = (primary_idx - 1) % len(classified)
                    p = classified[primary_idx]
                    print(f"[primary ←] #{primary_idx} {p['size_mm']}mm r={p['r']}")
            elif key in (83, 3):       # →
                if classified:
                    primary_idx = (primary_idx + 1) % len(classified)
                    p = classified[primary_idx]
                    print(f"[primary →] #{primary_idx} {p['size_mm']}mm r={p['r']}")

    except KeyboardInterrupt:
        pass
    finally:
        close()
        cv2.destroyAllWindows()
        try:
            comm.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()