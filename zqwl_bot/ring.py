#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ring.py - 同心圆环识别 + 自动像素比例估算 + dx/dy 定位

main 里用法:
    import comm
    import ring
    comm.init("/dev/ttyCH341USB0", 115200)
    ring.align_once_to_ring()     # 到附近后: 找同心圆圆心 → 自动估算 mm/px → dx/dy 定位
    ring.align_to_ring()          # 兼容旧 3 步: 对准 → +100mm → 退后 500px
    ring.close()                  # 退出时
"""

import cv2
import numpy as np
import time
import comm


CONFIG = {
    "usb_device": 0,
    "usb_devices": [0, 1],
    "width": 640,
    "height": 480,
    "fps": 30,

    "black_thresh": 140,
    "known_ring_diameters_mm": [50, 90, 130, 170, 210],
    "single_arc_default_diameter_mm": 210,
    "min_candidate_diameter_px": 18,
    "max_candidate_diameter_px": 900,
    "min_arc_len_px": 45,
    "min_arc_coverage": 0.06,
    "min_eccentricity": 0.30,
    "min_contour_points": 5,
    "center_cluster_px": 80,
    "diam_merge_abs_px": 10,
    "diam_merge_rel": 0.055,
    "match_abs_tol_px": 18,
    "match_rel_tol": 0.18,
    "min_scale_matches": 2,

    "ring_actual_radius_m": 0.075,  # 仅作为自动比例失败时的旧兜底
    "calib_step": 0.0,

    "detect_frames": 5,
    "detect_min_hits": 2,
    "detect_timeout_s": 1.5,
    "otsu_smooth_n": 3,
    "consistency_max_diff": 80,
    "last_known_max_age_s": 3.0,

    "dead_zone_px": 0,
    "correct_timeout_s": 20.0,

    "pre_cmd_sleep_s": 0.3,
    "comm_port": "/dev/ttyCH341USB0",
    "comm_baud": 115200,

    "recover_pose_age_max": 1.0,
    "recover_max_retry": 3,

    "step1_to_step2_pause_s": 2.0,
    "post_align_offset_y_mm": 100,
    "step2_to_step3_pause_s": 2.0,
    "step3_back_offset_px": 500,
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
def _threshold_dark(frame):
    """提取黑色圆环/圆弧区域，固定阈值为主，Otsu 只用于补强暗目标。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = min(CONFIG["black_thresh"], int(otsu_thr) + 20)
    _, dark = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    return dark


def _angle_coverage(points, cx, cy, bins=72):
    """估算一段圆弧覆盖了多少角度，允许只看到部分圆弧。"""
    pts = points.reshape(-1, 2).astype(np.float32)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    idx = ((angles + np.pi) / (2.0 * np.pi) * bins).astype(np.int32)
    idx = np.clip(idx, 0, bins - 1)
    return len(np.unique(idx)) / float(bins)


def _extract_ring_candidates(frame):
    """从黑色 mask 中提取可能属于同心圆环的椭圆/圆弧候选。"""
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
        out.append({
            "cx": sum(i["cx"] * i["score"] for i in group_items) / weight_sum,
            "cy": sum(i["cy"] * i["score"] for i in group_items) / weight_sum,
            "diam_px": sum(i["diam_px"] * i["score"] for i in group_items) / weight_sum,
            "score": weight_sum,
            "coverage": max(i["coverage"] for i in group_items),
            "count": len(group_items),
        })
    return out


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
                        }
                    score += item_score

            match_count = len(matches_by_known)
            total_score = match_count * 100000.0 + score
            if best is None or total_score > best["total_score"]:
                best = {
                    "px_per_mm": px_per_mm,
                    "mm_per_px": 1.0 / px_per_mm,
                    "matches": list(matches_by_known.values()),
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
            }],
            "match_count": 1,
            "total_score": largest["score"],
            "observed": observed,
            "fallback_single": True,
        }
    else:
        best["fallback_single"] = False

    matched = best["matches"]
    if matched:
        weight_sum = sum(max(m["score"], 1.0) for m in matched)
        cx = sum(m["cx"] * max(m["score"], 1.0) for m in matched) / weight_sum
        cy = sum(m["cy"] * max(m["score"], 1.0) for m in matched) / weight_sum
        outer = max(matched, key=lambda m: m["known_mm"])
        radius_px = outer["diam_px"] / 2.0
    else:
        cx, cy = cluster["cx"], cluster["cy"]
        radius_px = max(o["diam_px"] for o in observed) / 2.0

    best.update({
        "cx": float(cx),
        "cy": float(cy),
        "radius_px": float(radius_px),
        "confidence": min(1.0, best["match_count"] / float(len(known)))
    })
    return best


def find_ring_center(frame):
    """返回同心圆中心和自动比例估算结果。"""
    candidates, dark = _extract_ring_candidates(frame)
    if not candidates:
        return None

    best = None
    for cluster in _cluster_by_center(candidates):
        fit = _fit_scale_for_cluster(cluster)
        if fit is None:
            continue
        cluster_score = fit["total_score"] + len(cluster["items"]) * 1000.0
        if best is None or cluster_score > best["cluster_score"]:
            fit["cluster_score"] = cluster_score
            fit["candidate_count"] = len(cluster["items"])
            best = fit
    return best


# ============ 摄像头单例 ============
_cam_cache = {"cam": None}
_last_known = {"result": None, "time": 0.0}


def _get_cam():
    if _cam_cache["cam"] is None:
        devices = CONFIG.get("usb_devices") or [CONFIG.get("usb_device", 0)]
        ordered_devices = []
        for dev in [CONFIG.get("usb_device", 0), *devices, 0, 1]:
            if dev not in ordered_devices:
                ordered_devices.append(dev)

        last_error = None
        for dev in ordered_devices:
            cam = USBCamera(dev, CONFIG["width"], CONFIG["height"], CONFIG["fps"])
            try:
                cam.start()
            except Exception as e:
                last_error = e
                continue
            CONFIG["usb_device"] = dev
            _cam_cache["cam"] = cam
            print(f"[ring] USB摄像头已打开: device={dev}")
            break

        if _cam_cache["cam"] is None:
            raise RuntimeError(f"无法打开USB摄像头，已尝试 {ordered_devices}: {last_error}")
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

    # 取与中位数 mm_per_px 最接近的一帧作为匹配明细来源，便于调试输出。
    detail = min(hits, key=lambda h: abs(h["mm_per_px"] - mm_per_px))
    result = {
        "cx": int(round(cx)),
        "cy": int(round(cy)),
        "radius_px": float(r),
        "mm_per_px": float(mm_per_px),
        "confidence": float(confidence),
        "match_count": int(detail.get("match_count", 0)),
        "matches": detail.get("matches", []),
        "fallback_single": bool(detail.get("fallback_single", False)),
        "candidate_count": int(detail.get("candidate_count", 0)),
    }
    _last_known["result"] = result
    _last_known["time"] = time.time()
    return result


# ============ 自愈 ============
def _is_comm_alive():
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

    # 约定: 画面圆心偏右 → 车需要向左(dx为负); 圆心偏下 → 车向前(dy为正)。
    dx_mm = -dx_px * mm_per_px
    dy_mm = dy_px * mm_per_px
    return {
        "cx": cx,
        "cy": cy,
        "radius_px": float(result["radius_px"]),
        "dx_px": int(dx_px),
        "dy_px": int(dy_px),
        "dx_mm": float(dx_mm),
        "dy_mm": float(dy_mm),
        "mm_per_px": mm_per_px,
        "confidence": float(result.get("confidence", 0.0)),
        "match_count": int(result.get("match_count", 0)),
        "matches": result.get("matches", []),
        "fallback_single": bool(result.get("fallback_single", False)),
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


def _print_detection(offset):
    """打印检测结果，方便现场调试比例是否匹配到正确圆。"""
    print(f"  ring center = ({offset['cx']}, {offset['cy']})")
    print(f"  ring radius = {offset['radius_px']:.1f} px")
    print(f"  像素偏差    = dx {offset['dx_px']:+d} px, dy {offset['dy_px']:+d} px")
    print(f"  自动比例    = {offset['mm_per_px']:.3f} mm/px, confidence={offset['confidence']:.2f}")
    if offset["fallback_single"]:
        print(f"  比例来源    = 单段圆弧兜底，按 {CONFIG['single_arc_default_diameter_mm']}mm 外圈处理")
    elif offset["matches"]:
        desc = []
        for m in sorted(offset["matches"], key=lambda x: x["known_mm"]):
            desc.append(f"{int(m['known_mm'])}mm≈{m['diam_px']:.1f}px")
        print("  直径匹配    = " + ", ".join(desc))


def align_once_to_ring(verbose=True):
    """到达圆环附近后，只根据当前画面圆心偏差执行一次 dx/dy 定位。"""
    if not ensure_comm_alive(verbose=verbose):
        return False

    time.sleep(CONFIG["pre_cmd_sleep_s"])
    offset = detect_offset(verbose=verbose)
    if offset is None:
        if verbose:
            print("  [FAIL] 没找到同心圆环")
        return False

    if abs(offset["dx_px"]) <= CONFIG["dead_zone_px"] and \
       abs(offset["dy_px"]) <= CONFIG["dead_zone_px"]:
        if verbose:
            print("  [OK] 已在死区内，无需移动")
        return True

    pose = comm.get_pose(max_age=1.0)
    if pose is None:
        if verbose:
            print("  [FAIL] 没拿到位姿")
        return False

    x_now, y_now, _ = pose
    target_x = x_now + offset["dx_mm"] / 1000.0
    target_y = y_now + offset["dy_mm"] / 1000.0
    if verbose:
        print("\n  === 单次同心圆 dx/dy 定位 ===")
        print(f"  移动        = dx {offset['dx_mm']:+.1f} mm, dy {offset['dy_mm']:+.1f} mm")
        print(f"  目标位置    = ({target_x:.3f}, {target_y:.3f}) m")

    ok = _safe_goto(target_x, target_y, timeout=CONFIG["correct_timeout_s"])
    if verbose:
        print(f"  [align once] {'OK' if ok else 'FAIL'}")
    return ok


# ============ ★ 公共 API: 3 步对齐 ============
def align_to_ring(verbose=True):
    """
    3 步对齐流程:
      1. 检测圆环, 一次性 goto 对准画面中心
      2. 等 2 秒, y +100mm
      3. 等 2 秒, 退后 500 像素

    返回 True=全部成功, False=某步失败
    """
    if not ensure_comm_alive(verbose=verbose):
        return False

    time.sleep(CONFIG["pre_cmd_sleep_s"])

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

    offset = _offset_from_result(result)
    if verbose:
        _print_detection(offset)

    if offset["radius_px"] <= 0 or offset["mm_per_px"] <= 0:
        return False

    k_mm_per_px = offset["mm_per_px"]

    # === 步骤 1: 对准 ===
    dx_mm = offset["dx_mm"]
    dy_mm = offset["dy_mm"]
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
        print(f"  标定系数    = {k_mm_per_px:.3f} mm/px (auto diameter match)")
        print(f"  移动        = dx {dx_mm:+.1f} mm, dy {dy_mm:+.1f} mm")
        print(f"  目标位置    = ({target_x_align:.3f}, {target_y_align:.3f}) m")

    ok = _safe_goto(target_x_align, target_y_align,
                    timeout=CONFIG["correct_timeout_s"])
    if not ok:
        if verbose:
            print("  [step 1] FAIL")
        return False
    if verbose:
        print(f"  [step 1] OK")

    # === 等 1→2 ===
    _wait(CONFIG["step1_to_step2_pause_s"], "步骤1→2 等待", verbose)

    # === 步骤 2: y +100mm ===
    target_y_step2 = target_y_align + CONFIG["post_align_offset_y_mm"] / 1000.0

    if verbose:
        print(f"\n  === 步骤 2: y +{CONFIG['post_align_offset_y_mm']}mm ===")
        print(f"  目标位置    = ({target_x_align:.3f}, {target_y_step2:.3f}) m")

    time.sleep(CONFIG["pre_cmd_sleep_s"])
    ok = _safe_goto(target_x_align, target_y_step2,
                    timeout=CONFIG["correct_timeout_s"])
    if not ok:
        if verbose:
            print("  [step 2] FAIL")
        return False
    if verbose:
        print(f"  [step 2] OK")

    # === 等 2→3 ===
    _wait(CONFIG["step2_to_step3_pause_s"], "步骤2→3 等待", verbose)

    # === 步骤 3: 退后 500 像素 ===
    back_offset_mm = CONFIG["step3_back_offset_px"] * k_mm_per_px
    target_y_step3 = target_y_step2 - back_offset_mm / 1000.0

    if verbose:
        print(f"\n  === 步骤 3: 退后 {CONFIG['step3_back_offset_px']} 像素 ===")
        print(f"  {CONFIG['step3_back_offset_px']}px 实际  = {back_offset_mm:.1f} mm")
        print(f"  目标位置    = ({target_x_align:.3f}, {target_y_step3:.3f}) m")

    time.sleep(CONFIG["pre_cmd_sleep_s"])
    ok = _safe_goto(target_x_align, target_y_step3,
                    timeout=CONFIG["correct_timeout_s"])
    if verbose:
        print(f"  [step 3] {'OK' if ok else 'FAIL'}")
    return ok


def close():
    """关闭摄像头 (main 退出时调用)"""
    if _cam_cache["cam"] is not None:
        _cam_cache["cam"].stop()
        _cam_cache["cam"] = None


if __name__ == "__main__":
    try:
        offset = detect_offset(verbose=True)
        if offset is None:
            print("[ring] 未检测到同心圆环")
        else:
            print(f"[ring] dx={offset['dx_mm']:+.1f}mm, dy={offset['dy_mm']:+.1f}mm")
    finally:
        close()
