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
import sys
import time
import comm


CONFIG = {
    "usb_device": 0,
    "usb_devices": [0, 1],
    "width": 640,
    "height": 480,
    "fps": 30,

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

    # 径向扫描兜底：直接利用 50/90/130/170/210mm 同心比例搜索圆心。
    "enable_radial_scan": False,
    "radial_trigger_confidence": 0.45,
    "radial_center_search_px": 100,
    "radial_center_step_px": 20,
    "radial_refine_step_px": 4,
    "radial_outer_radius_min_px": 45,
    "radial_outer_radius_max_px": 260,
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

    "detect_frames": 3,
    "detect_min_hits": 1,
    "detect_timeout_s": 0.8,
    "otsu_smooth_n": 3,
    "consistency_max_diff": 80,
    "last_known_max_age_s": 3.0,

    "dead_zone_px": 0,
    "correct_timeout_s": 20.0,
    "fine_move_max_step_mm": 30.0,
    "auto_forward_after_align": False,
    "camera_dx_sign": -1.0,
    "camera_dy_sign": -1.0,
    "camera_swap_xy": False,

    "pre_cmd_sleep_s": 0.3,
    "comm_port": "/dev/ttyCH341USB0",
    "comm_baud": 115200,

    "recover_pose_age_max": 1.0,
    "recover_max_retry": 3,

    "step1_to_step2_pause_s": 2.0,
    "post_align_offset_y_mm": 100,
    "step2_to_step3_pause_s": 2.0,
    "step3_back_offset_px": 500,

    "show_debug_window": True,
    "debug_window_name": "Ring Debug",
    "debug_mask_window_name": "Ring Evidence",
    "debug_max_candidates": 25,
    "preview_detect_interval_s": 0.25,
    "preview_result_max_age_s": 1.00,
    "enable_debug_trackbars": True,
    "trackbar_window_name": "Ring Tune",
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


def _best_fit_from_candidates(candidates, dark, method):
    """从一组候选圆/圆弧中选出最可信的同心圆结果。"""
    if not candidates:
        return None

    best = None
    for cluster in _cluster_by_center(candidates):
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


def find_ring_center(frame, dark=None, evidence=None):
    """返回同心圆中心和自动比例估算结果。

    识别顺序：轮廓椭圆拟合 → RANSAC 圆弧拟合 → 径向同心比例扫描。
    """
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

    best_now = max(results, key=_rank_result) if results else None
    need_ransac = CONFIG.get("enable_ransac", True) and (
        best_now is None or best_now.get("fallback_single") or
        best_now.get("confidence", 0.0) < 0.65
    )
    if need_ransac:
        ransac_candidates, _ = _extract_ransac_candidates(evidence)
        if ransac_candidates:
            all_candidates.extend(ransac_candidates)
            ransac_best = _best_fit_from_candidates(ransac_candidates, dark, "ransac")
            if ransac_best is not None:
                results.append(ransac_best)

    best_now = max(results, key=_rank_result) if results else None
    need_radial = CONFIG.get("enable_radial_scan", True) and (
        best_now is None or best_now.get("fallback_single") or
        best_now.get("confidence", 0.0) < CONFIG["radial_trigger_confidence"]
    )
    if need_radial:
        radial_best = _radial_scan_result(evidence, seed_candidates=all_candidates)
        if radial_best is not None:
            results.append(radial_best)

    if not results:
        return None

    best = max(results, key=_rank_result)
    best["candidates"] = all_candidates
    best["mask"] = evidence
    return best


def _draw_debug(frame, result=None, candidates=None):
    """绘制同心圆调试画面。"""
    display = frame.copy()
    h, w = display.shape[:2]
    cv2.line(display, (w // 2 - 18, h // 2), (w // 2 + 18, h // 2), (255, 255, 0), 1)
    cv2.line(display, (w // 2, h // 2 - 18), (w // 2, h // 2 + 18), (255, 255, 0), 1)
    cv2.circle(display, (w // 2, h // 2), 5, (255, 255, 0), -1)

    if candidates:
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
        cv2.circle(display, (cx, cy), max(2, r), (0, 255, 0), 2)
        cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)
        cv2.line(display, (w // 2, h // 2), (cx, cy), (0, 255, 255), 2)
        offset = _offset_from_result(result)
        cv2.putText(display, f"dx={offset['dx_mm']:+.1f}mm dy={offset['dy_mm']:+.1f}mm",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f"scale={offset['mm_per_px']:.3f} mm/px match={offset['match_count']}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, f"method={method} conf={offset['confidence']:.2f}",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if offset["fallback_single"]:
            cv2.putText(display, "fallback: 210mm outer arc",
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
        "method": detail.get("method", "unknown"),
        "match_count": int(detail.get("match_count", 0)),
        "matches": detail.get("matches", []),
        "fallback_single": bool(detail.get("fallback_single", False)),
        "candidate_count": int(detail.get("candidate_count", 0)),
        "radial_score": float(detail.get("radial_score", 0.0)),
        "ring_scores": detail.get("ring_scores", []),
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


def _safe_fine_move(dx_mm, dy_mm, timeout=20.0, max_retry=2):
    """发送下位机已有 dx/dy 微调命令，并等待到位反馈。"""
    for attempt in range(max_retry):
        try:
            seen_seq = comm.response_seq()
            comm.send_fine_move(dx_mm, dy_mm)
            ok = comm.wait_for_after(comm.TYPE_CMD_FINE_RESP, seen_seq, timeout)
            if ok:
                return True
            print(f"  [fine_move {attempt+1}] 下位机返回失败或等待超时")
        except Exception as e:
            print(f"  [fine_move {attempt+1}] err: {e}")
        time.sleep(0.3)

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

    raw_dx_mm = dx_px * mm_per_px
    raw_dy_mm = dy_px * mm_per_px
    if CONFIG.get("camera_swap_xy", False):
        raw_dx_mm, raw_dy_mm = raw_dy_mm, raw_dx_mm

    # 约定: 下位机 FINE_MOVE 使用体坐标，dx=右移，dy=前进。
    # 摄像头安装方向未标定时，现场可用 X/Y/S 热键翻转符号或交换轴。
    dx_mm = raw_dx_mm * float(CONFIG["camera_dx_sign"])
    dy_mm = raw_dy_mm * float(CONFIG["camera_dy_sign"])
    return {
        "cx": cx,
        "cy": cy,
        "radius_px": float(result["radius_px"]),
        "dx_px": int(dx_px),
        "dy_px": int(dy_px),
        "dx_mm": float(dx_mm),
        "dy_mm": float(dy_mm),
        "raw_dx_mm": float(raw_dx_mm),
        "raw_dy_mm": float(raw_dy_mm),
        "mm_per_px": mm_per_px,
        "confidence": float(result.get("confidence", 0.0)),
        "method": result.get("method", "unknown"),
        "match_count": int(result.get("match_count", 0)),
        "matches": result.get("matches", []),
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


def _limited_dxdy(dx_mm, dy_mm):
    """限制单次视觉微调距离，防止未标定时一次跑偏过大。"""
    limit = float(CONFIG.get("fine_move_max_step_mm", 30.0))
    dist = float((dx_mm * dx_mm + dy_mm * dy_mm) ** 0.5)
    if limit <= 0.0 or dist <= limit:
        return float(dx_mm), float(dy_mm), 1.0
    scale = limit / max(dist, 1e-6)
    return float(dx_mm * scale), float(dy_mm * scale), float(scale)


def _move_by_offset(offset, verbose=True):
    """按检测得到的 dx/dy 微调；是否前进由配置控制。"""
    align_ok = True
    if abs(offset["dx_px"]) <= CONFIG["dead_zone_px"] and \
       abs(offset["dy_px"]) <= CONFIG["dead_zone_px"]:
        if verbose:
            print("  [OK] 已在死区内，无需移动")
    else:
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
        if not align_ok:
            return False

    if not CONFIG.get("auto_forward_after_align", False):
        if verbose:
            print("  [safe] 自动 dy+100 已关闭；需要前进时按 F")
        return True

    forward_mm = float(CONFIG["post_align_offset_y_mm"])
    if not ensure_comm_alive(verbose=verbose):
        return False
    if verbose:
        print(f"\n  === 定位后前进 dy +{forward_mm:.0f}mm ===")
    ok = _safe_fine_move(0.0, forward_mm, timeout=CONFIG["correct_timeout_s"])
    if verbose:
        print(f"  [forward dy] {'OK' if ok else 'FAIL'}")
    return ok


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
        f"ransac={CONFIG['enable_ransac']}, radial={CONFIG['enable_radial_scan']}"
    )


def _forward_after_align(verbose=True):
    """沿车体 dy 正方向单独前进固定距离。"""
    if not ensure_comm_alive(verbose=verbose):
        return False
    forward_mm = float(CONFIG["post_align_offset_y_mm"])
    if verbose:
        print(f"\n  === 手动前进 dy +{forward_mm:.0f}mm ===")
    ok = _safe_fine_move(0.0, forward_mm, timeout=CONFIG["correct_timeout_s"])
    if verbose:
        print(f"  [forward dy] {'OK' if ok else 'FAIL'}")
    return ok


def preview():
    """实时显示圆环识别画面，并提供现场校准热键。"""
    cam = _get_cam()
    cv2.namedWindow(CONFIG["debug_window_name"], cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONFIG["debug_mask_window_name"], cv2.WINDOW_NORMAL)
    _setup_debug_trackbars()
    print("[ring] 预览已启动: q退出, p打印, A限幅微调, F前进100mm, G微调+前进, X/Y翻符号, S交换轴, R/D开慢兜底")
    print("[ring] Ring Evidence 是局部暗线证据图，不要求画面只剩黑线")
    _print_mapping()
    last_result = None
    last_candidates = []
    last_result_time = 0.0
    next_detect_time = 0.0
    last_evidence = None
    while True:
        _read_debug_trackbars()
        frame = cam.read()
        if frame is None:
            time.sleep(0.02)
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
            CONFIG["camera_dx_sign"] *= -1.0
            _print_mapping()
        if key in (ord('y'), ord('Y')):
            CONFIG["camera_dy_sign"] *= -1.0
            _print_mapping()
        if key in (ord('s'), ord('S')):
            CONFIG["camera_swap_xy"] = not CONFIG.get("camera_swap_xy", False)
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
        if key in (ord('a'), ord('A')):
            # 按键动作必须用当前画面重新识别一次，避免用过期圆心移动。
            fresh_mask, fresh_evidence, _ = _build_ring_evidence(frame)
            fresh_result = find_ring_center(frame, dark=fresh_mask, evidence=fresh_evidence)
            if fresh_result is None:
                print("[ring] 当前没有有效检测，不能移动")
            else:
                last_result = fresh_result
                last_candidates = fresh_result.get("candidates", [])
                last_result_time = time.time()
                offset = _offset_from_result(last_result)
                _print_detection(offset)
                ok = _move_by_offset(offset, verbose=True)
                print(f"[ring] A键微调 {'OK' if ok else 'FAIL'}")
        if key in (ord('g'), ord('G')):
            fresh_mask, fresh_evidence, _ = _build_ring_evidence(frame)
            fresh_result = find_ring_center(frame, dark=fresh_mask, evidence=fresh_evidence)
            if fresh_result is None:
                print("[ring] 当前没有有效检测，不能移动")
            else:
                last_result = fresh_result
                last_candidates = fresh_result.get("candidates", [])
                last_result_time = time.time()
                offset = _offset_from_result(last_result)
                _print_detection(offset)
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
    print(f"  自动比例    = {offset['mm_per_px']:.3f} mm/px, confidence={offset['confidence']:.2f}")
    if offset["fallback_single"]:
        print(f"  比例来源    = 单段圆弧兜底，按 {CONFIG['single_arc_default_diameter_mm']}mm 外圈处理")
    elif offset["matches"]:
        desc = []
        for m in sorted(offset["matches"], key=lambda x: x["known_mm"]):
            desc.append(f"{int(m['known_mm'])}mm≈{m['diam_px']:.1f}px")
        print("  直径匹配    = " + ", ".join(desc))
    if offset.get("method") == "radial" and offset.get("ring_scores"):
        scores = ", ".join(f"{s:.2f}" for s in offset["ring_scores"])
        print(f"  径向得分    = {offset.get('radial_score', 0.0):.2f} [{scores}]")


def align_once_to_ring(verbose=True):
    """到达圆环附近后，只根据当前画面圆心偏差执行一次 dx/dy 定位。"""
    time.sleep(CONFIG["pre_cmd_sleep_s"])
    offset = detect_offset(verbose=verbose)
    if offset is None:
        if verbose:
            print("  [FAIL] 没找到同心圆环")
        return False
    return _move_by_offset(offset, verbose=verbose)


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
