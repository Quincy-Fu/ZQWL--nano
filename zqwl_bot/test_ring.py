#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ring.py - 同心圆环视觉定位测试入口

用途：车先靠轮子定位到圆环附近，再用 USB 摄像头识别 50/90/130/170/210mm
同心圆，自动估算 mm/px，通过下位机 FINE_MOVE 做 dx/dy 闭环修正；
对准后的固定前进/后退也使用 FINE_MOVE。
"""

import sys
import time
import importlib.util
from pathlib import Path

import comm


def _load_local_ring_module():
    """强制加载同目录 ring.py，避免误导入系统/旧版本 ring 模块。"""
    ring_path = Path(__file__).resolve().with_name("ring.py")
    spec = importlib.util.spec_from_file_location("zqwl_local_ring", ring_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载本目录 ring.py: {ring_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise RuntimeError(
            f"加载到的 ring.py 没有 CONFIG: {ring_path}\n"
            "请确认已经把新版 ring.py 和 test_ring.py 一起复制到车端同一目录。"
        )
    return module


ring = _load_local_ring_module()


def _parse_args(argv):
    """解析测试入口参数；模式参数和覆盖参数可以任意顺序传入。"""
    mode = "preview"
    usb_device = None
    usb_devices = None
    port = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        low = arg.lower()
        if low in ("--device", "--usb-device"):
            i += 1
            if i >= len(argv):
                raise SystemExit("缺少 USB 设备号，例如 --device 1")
            usb_device = argv[i]
        elif low.startswith("--device=") or low.startswith("--usb-device="):
            usb_device = arg.split("=", 1)[1]
        elif low in ("--devices", "--usb-devices"):
            i += 1
            if i >= len(argv):
                raise SystemExit("缺少 USB 设备列表，例如 --devices 1,0")
            usb_devices = argv[i]
        elif low.startswith("--devices=") or low.startswith("--usb-devices="):
            usb_devices = arg.split("=", 1)[1]
        elif low in ("--port", "--serial"):
            i += 1
            if i >= len(argv):
                raise SystemExit("缺少串口，例如 --port /dev/ttyUSB0")
            port = argv[i]
        elif low.startswith("--port=") or low.startswith("--serial="):
            port = arg.split("=", 1)[1]
        elif not arg.startswith("-"):
            mode = low
        else:
            raise SystemExit(f"未知参数: {arg}")
        i += 1
    return mode, usb_device, usb_devices, port


RING_TEST_CONFIG = {
    "comm_port": "/dev/ttyCH341USB*",
    "comm_baud": 115200,
    "usb_device": 1,
    "usb_devices": [1, 0],

    # 当前场地同心圆直径，单位 mm。注意这里是直径，不是半径。
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
    "diam_merge_abs_px": 28,
    "diam_merge_rel": 0.075,

    # 允许只看到部分圆弧：默认用椭圆/轮廓 + RANSAC；径向扫描只作手动调试兜底。
    "enable_ransac": False,
    "enable_radial_scan": False,
    "radial_trigger_offset_mm": 30.0,
    "radial_trigger_match_count": 4,
    "radial_prefer_confidence_below": 0.70,
    "radial_prefer_match_count_below": 4,
    "radial_prefer_min_score": 0.14,
    "radial_prefer_min_hits": 3,
    "radial_outer_radius_max_px": 360,
    "min_arc_coverage": 0.05,
    "min_scale_matches": 2,
    "match_rel_tol": 0.20,

    # 移动前用多帧确认；单次修正限幅，避免方向未标定时一次跑偏太多。
    "detect_frames": 5,
    "detect_min_hits": 2,
    "detect_timeout_s": 1.2,
    "fine_move_max_step_mm": 30.0,
    "align_tolerance_mm": 3.0,
    "max_align_iterations": 10,
    "checked_align_max_moves": 6,
    "align_detect_retry_count": 4,
    "align_detect_retry_sleep_s": 0.15,
    "push_move_timeout_s": 35.0,
    "back_move_timeout_s": 45.0,
    # 推送/后退距离只在 ring.py 的 CONFIG 里配置，避免测试入口覆盖后看起来“改了没生效”。
    "auto_forward_after_align": False,

    # 摄像头安装方向必须现场确认；预览界面可用 X/Y/S 热键临时翻转。
    "camera_dx_sign": -1.0,
    "camera_dy_sign": 1.0,
    "camera_swap_xy": False,
    "show_candidate_circles": False,
    "draw_selected_ellipse": False,
    "draw_manual_reference_rings": False,
    "draw_observed_fit_rings": True,
}


def configure_ring_for_test(usb_device=None, usb_devices=None, port=None):
    ring.configure(RING_TEST_CONFIG)
    ring.apply_usb_env()
    if usb_device is not None or usb_devices is not None:
        ring.configure_usb(device=usb_device, devices=usb_devices)
    if port is not None:
        ring.CONFIG["comm_port"] = port
    print(
        f"[test_ring] 推送距离: forward={ring.CONFIG['post_align_offset_y_mm']}mm, "
        f"back={ring.CONFIG['post_align_back_y_mm']}mm"
    )


def _run_mapping_selftest():
    """不打开摄像头和串口，只检查配置与像素偏差到车体 dx/dy 的映射。"""
    def fmt_mm(value):
        value = 0.0 if abs(float(value)) < 0.05 else float(value)
        return f"{value:+.1f}mm"

    ring.validate_config()
    samples = [
        ("图像圆心在右侧 10px", 10, 0),
        ("图像圆心在左侧 10px", -10, 0),
        ("图像圆心在下方 10px", 0, 10),
        ("图像圆心在上方 10px", 0, -10),
    ]
    print("[test_ring] 配置自检 OK")
    print(
        "[test_ring] 当前映射: "
        f"dx_sign={ring.CONFIG['camera_dx_sign']:+.0f}, "
        f"dy_sign={ring.CONFIG['camera_dy_sign']:+.0f}, "
        f"swap_xy={ring.CONFIG['camera_swap_xy']}, "
        f"manual_mm_per_px={ring.CONFIG['manual_mm_per_px']:.3f}"
    )
    for label, dx_px, dy_px in samples:
        mapped = ring.pixel_offset_to_body_mm(dx_px, dy_px, ring.CONFIG["manual_mm_per_px"])
        print(f"  {label:<18} -> 下发 dx={fmt_mm(mapped['dx_mm'])}, dy={fmt_mm(mapped['dy_mm'])}")
    return 0


def _init_comm(required=False):
    try:
        port = ring.resolve_comm_port(ring.CONFIG.get("comm_port"))
        ring.CONFIG["comm_port"] = port
        comm.init(port, ring.CONFIG["comm_baud"])
        print(f"[test_ring] 串口已连接: {port}")
        time.sleep(0.5)
        return True
    except Exception as exc:
        if required:
            raise
        print(f"[test_ring] 串口暂未连接，只开启视觉预览: {exc}")
        return False


def main():
    mode, usb_device, usb_devices, port = _parse_args(sys.argv[1:])
    configure_ring_for_test(usb_device=usb_device, usb_devices=usb_devices, port=port)

    print("=" * 64)
    print("同心圆环定位测试：只拟合圆心；mm/px 使用手动标定值")
    print("preview: 实时画面；once/detect: 只识别；align: 微调到位后前进105mm并后退200mm；mapping: 坐标映射自检")
    print("覆盖参数：--device 1 | --devices 1,0 | --port /dev/ttyUSB0")
    print("预览热键：A=微调到位+前进105mm+后退200mm，M=多次定位不前进，P=打印，X/Y/S=方向标定，C=候选圆显示")
    print("=" * 64)

    try:
        if mode in ("mapping", "selftest"):
            return _run_mapping_selftest()
        if mode in ("once", "detect"):
            offset = ring.detect_offset(verbose=True)
            if offset is None:
                print("[test_ring] 未检测到可靠同心圆")
                return 1
            return 0
        if mode in ("align", "a"):
            _init_comm(required=True)
            return 0 if ring.align_checked_then_forward(verbose=True) else 1
        _init_comm(required=False)
        ring.preview()
        return 0
    finally:
        ring.close()
        try:
            comm.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
