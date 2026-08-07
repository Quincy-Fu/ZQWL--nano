"""
Ring - 同心圆环中心检测
返回 5 个圆心坐标 (mm) [(x, y), ...]
"""
import cv2
import numpy as np
import time

CONFIG = {
    "usb_device": 0,
    "width": 640,
    "height": 480,
    "fps": 30,
    "px_per_mm": 3.0,                 # 需实测标定
    "black_v_max": 100,
    "min_radius_px": 20,
    "max_radius_px": 400,
    "min_circularity": 0.4,
    "min_area": 200,
    "center_tolerance_px": 20,        # 同心圆容差
}


def _open_usb():
    cap = cv2.VideoCapture(CONFIG["usb_device"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["height"])
    cap.set(cv2.CAP_PROP_FPS, CONFIG["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Ring: USB 摄像头打开失败")
    return cap


def _detect_circles(frame):
    """从单帧找所有候选圆"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, CONFIG["black_v_max"]]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < CONFIG["min_area"]:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ = 4 * np.pi * area / (perim * perim)
        if circ < CONFIG["min_circularity"]:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < CONFIG["min_radius_px"] or r > CONFIG["max_radius_px"]:
            continue
        circles.append((cx, cy, r))
    return circles


def _group_concentric(circles):
    """把候选圆按圆心位置分组 (同心圆)"""
    groups = []
    for cx, cy, r in circles:
        found = False
        for g in groups:
            gx, gy, radii = g
            if abs(cx - gx) < CONFIG["center_tolerance_px"] and abs(cy - gy) < CONFIG["center_tolerance_px"]:
                n = len(radii)
                g[0] = (gx * n + cx) / (n + 1)
                g[1] = (gy * n + cy) / (n + 1)
                radii.append(r)
                found = True
                break
        if not found:
            groups.append([cx, cy, [r]])
    return groups


def detect_centers(target_count: int = 5, timeout: float = 10.0) -> list:
    """返回 N 个圆心坐标 (mm) [(x, y), ...].

    持续识别直到找到 N 个不同圆心或超时.
    失败/超时抛 RuntimeError.
    """
    cap = _open_usb()
    px = CONFIG["px_per_mm"]
    try:
        t0 = time.time()
        while time.time() - t0 < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            circles = _detect_circles(frame)
            groups = _group_concentric(circles)
            if len(groups) >= target_count:
                return [(g[0] / px, g[1] / px) for g in groups[:target_count]]
            time.sleep(0.1)
        raise RuntimeError(f"Ring: {timeout}s 内未找到 {target_count} 个圆环")
    finally:
        cap.release()


if __name__ == "__main__":
    print("圆心 (mm):", detect_centers())