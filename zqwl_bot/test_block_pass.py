#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_block_pass.py - 物块快速划过颜色返回测试。

用途：车子按实际速度从 USB 摄像头前经过物块时，验证 block.py 的 viewer
是否能在画面显示颜色后立刻返回一次颜色结果。
"""

import argparse
import time

import block


DEFAULT_COLORS = ("黑", "白", "红", "绿", "蓝")


def _parse_args():
    parser = argparse.ArgumentParser(description="测试物块划过摄像头时是否能返回颜色")
    parser.add_argument("--device", type=int, default=None,
                        help="指定优先 USB 摄像头编号；默认 1 优先、0 回退")
    parser.add_argument("--devices", default=None,
                        help="指定 USB 摄像头候选顺序，例如 1,0 或 0,1")
    parser.add_argument("--window", type=float, default=0.80,
                        help="viewer 最近颜色缓存窗口，单位秒，默认 0.80")
    parser.add_argument("--timeout", type=float, default=0.05,
                        help="每轮等待颜色出现的时间，单位秒，默认 0.05")
    parser.add_argument("--cooldown", type=float, default=0.35,
                        help="同一次划过只打印一次的冷却时间，单位秒，默认 0.35")
    parser.add_argument("--exclude", default="",
                        help="排除颜色，逗号分隔，例如: 白,黑")
    parser.add_argument("--no-window", action="store_true",
                        help="不显示 OpenCV 画面，只打印结果")
    return parser.parse_args()


def main():
    args = _parse_args()
    exclude_colors = {c.strip() for c in args.exclude.split(",") if c.strip()}
    exclude_colors &= set(DEFAULT_COLORS)

    if args.device is not None:
        block.configure_usb(device=args.device, devices=args.devices)
    elif args.devices is not None:
        block.configure_usb(devices=args.devices)
    else:
        block.apply_usb_env()

    block.CONFIG["show_window"] = not args.no_window
    block.CONFIG["instant_recent_window_s"] = args.window
    block.CONFIG["motion_recent_window_s"] = max(block.CONFIG.get("motion_recent_window_s", 0.25), args.window)

    print("=== 物块划过颜色返回测试 ===", flush=True)
    print(f"USB 尝试顺序: {block.CONFIG.get('usb_devices')}", flush=True)
    print(f"缓存窗口: {args.window:.2f}s, 轮询等待: {args.timeout:.2f}s, 冷却: {args.cooldown:.2f}s", flush=True)
    print(f"排除颜色: {sorted(exclude_colors) if exclude_colors else '无'}", flush=True)
    print("把物块按实际速度从摄像头 ROI 圆圈前划过；看到 HIT 就说明已经返回。按 Ctrl+C 或窗口 q 退出。", flush=True)

    block.clear_recognition_cache()
    block.set_status("划过测试: 等待颜色进入 ROI")
    block.start_viewer()

    start = time.monotonic()
    last_hit_color = None
    last_hit_time = 0.0
    hit_count = 0

    try:
        while True:
            if getattr(block, "_viewer_stop", None) is not None and block._viewer_stop.is_set():
                print("\n[退出] viewer 窗口请求退出", flush=True)
                break

            now = time.monotonic()
            color = block.wait_for_display_color(
                timeout_s=args.timeout,
                window_s=args.window,
                exclude_colors=exclude_colors,
            )
            debug = block.recent_motion_debug(window_s=args.window, exclude_colors=exclude_colors)

            if color is None:
                # 没有颜色一段时间后重新允许同色下一次划过打印。
                if now - last_hit_time > args.cooldown:
                    last_hit_color = None
                time.sleep(0.01)
                continue

            if color == last_hit_color and (now - last_hit_time) < args.cooldown:
                time.sleep(0.01)
                continue

            hit_count += 1
            last_hit_color = color
            last_hit_time = now
            block.set_status(f"HIT {hit_count}: {color}")
            print(
                f"[HIT {hit_count:03d}] t={now - start:7.3f}s color={color} "
                f"samples={debug.get('samples')} votes={debug.get('votes')} "
                f"best_pct={debug.get('best_pct')}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\n[退出] 用户中断", flush=True)
    finally:
        block.close()


if __name__ == "__main__":
    main()
