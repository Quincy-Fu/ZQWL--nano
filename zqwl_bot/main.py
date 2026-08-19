#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - C/D 物块放置 + A/B 奖杯放置整合入口

运行方式：
  python main.py
  python main.py /dev/ttyCH341USB1
  python main.py --no-run          # 调试用：不等待实体 RUN 开关
  python main.py --no-preflight    # 跳过等待 RUN 前的摄像头预检
  python main.py /dev/ttyCH341USB1 --no-run

说明：
- 串口、ring 配置只初始化一次。
- 默认等待下位机 PD15 实体 RUN 开关为高电平后再启动全流程。
- 先执行 test_C&D，再执行 test_A&B。
- C/D 结束后不插入任何物理移动；A/B 开始仍由 A/B 自己同步起点位姿。
"""

import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Optional

# main 是最终入口，必须在任何视觉模块导入 cv2 前设置，避免异常 USB 设备阻塞很久。
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")
os.environ.setdefault("OPENCV_VIDEOIO_V4L_READ_ATTEMPTS", "1")

import block
import comm
import ring


BAUDRATE = 115200
RUN_ARG_SET = {"run", "--run", "wait-run", "--wait-run"}
NO_RUN_ARGS = {"no-run", "--no-run", "skip-run", "--skip-run"}
NO_PREFLIGHT_ARGS = {"no-preflight", "--no-preflight", "skip-preflight", "--skip-preflight"}
RUN_QUERY_TIMEOUT = 0.7
RUN_QUERY_INTERVAL = 0.2
RUN_WAIT_NOTICE_INTERVAL = 5.0
PREFLIGHT_BLOCK_TIMEOUT_S = 2.5
RUN_BLOCK_KEEPALIVE_INTERVAL = 1.5
BASE_DIR = Path(__file__).resolve().parent


def _load_task_module(module_name: str, file_name: str) -> ModuleType:
    """按文件路径加载带特殊字符的测试脚本，例如 test_C&D.py / test_A&B.py。"""
    path = BASE_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载任务脚本: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _need_run_gate(argv: list[str]) -> bool:
    """默认等待实体 RUN 开关；调试时可用 --no-run 跳过。"""
    return not any(arg.lower() in NO_RUN_ARGS for arg in argv)


def _need_preflight(argv: list[str]) -> bool:
    """默认等待 RUN 时先做摄像头预检；必要时可用 --no-preflight 跳过。"""
    return not any(arg.lower() in NO_PREFLIGHT_ARGS for arg in argv)


def _port_arg(argv: list[str]) -> Optional[str]:
    """命令行里除 RUN 控制参数之外的第一个参数视为串口；没有则自动匹配。"""
    for arg in argv:
        if arg.lower() in RUN_ARG_SET or arg.lower() in NO_RUN_ARGS or arg.lower() in NO_PREFLIGHT_ARGS:
            continue
        return arg
    return None


def _preflight_item(name: str, func) -> bool:
    """执行单项预检；失败只报警，不阻塞后续 RUN。"""
    try:
        ok = bool(func())
    except Exception as exc:
        print(f"[MAIN] 预检 {name}: FAIL ({exc})")
        return False
    print(f"[MAIN] 预检 {name}: {'OK' if ok else 'FAIL'}")
    return ok


def _preflight_block_usb() -> bool:
    """打开 block USB 摄像头拿有效帧，并保持热启动直到 RUN 后正式识别。"""
    block.CONFIG["show_window"] = True
    block.set_status("RUN 前预热: block USB 保持打开")
    return bool(block.wait_until_ready(timeout_s=PREFLIGHT_BLOCK_TIMEOUT_S))


def _preflight_ring_usb() -> bool:
    """短暂打开 ring USB 摄像头拿有效帧，然后释放，避免抢占 C/D 的 block。"""
    try:
        warmup = getattr(ring, "_warmup_usb_camera")
        return bool(warmup(verbose=True))
    finally:
        ring.close()


def _preflight_csi_camera() -> bool:
    """验证 QR1/QR2 共用的 CSI 摄像头能启动并拿到有效帧。"""
    import qr1
    return bool(qr1.preflight_camera("QR1/QR2-CSI预检"))


def _preflight_before_run(argv: list[str]) -> None:
    if not _need_preflight(argv):
        print("[MAIN] 已跳过 RUN 前预检。")
        return

    print("[MAIN] RUN 等待前预检：先检查 ring/CSI，再预热并保持 block USB，减少正式识别重开等待。")
    results = [
        _preflight_item("ring USB", _preflight_ring_usb),
        _preflight_item("QR1/QR2 CSI", _preflight_csi_camera),
        _preflight_item("block USB keep-warm", _preflight_block_usb),
    ]
    if all(results):
        print("[MAIN] RUN 前预检全部通过。")
    else:
        print("[MAIN] RUN 前预检存在失败项；继续等待 RUN，正式流程仍会按原逻辑重试。")


def _wait_run_start_if_requested(argv: list[str]) -> None:
    if not _need_run_gate(argv):
        print("[MAIN] 调试模式：跳过实体 RUN 开关等待。")
        return
    _preflight_before_run(argv)
    print("[MAIN] 等待实体 RUN 开关：PD15=高电平后启动全流程。")
    last_notice = time.monotonic()
    last_block_keepalive = time.monotonic()
    while True:
        if comm.run(RUN_QUERY_TIMEOUT):
            if not block.ensure_ready_for_use("RUN启动前block确认", timeout_s=0.5, restart_timeout_s=0.8):
                raise RuntimeError("RUN 已按下，但 block USB 摄像头不可用")
            print("[MAIN] RUN 已按下，开始全流程。")
            return
        now = time.monotonic()
        if _need_preflight(argv) and now - last_block_keepalive >= RUN_BLOCK_KEEPALIVE_INTERVAL:
            block.keep_warm(timeout_s=0.25)
            last_block_keepalive = now
        if now - last_notice >= RUN_WAIT_NOTICE_INTERVAL:
            print("[MAIN] 仍在等待 RUN 高电平；调试可用 --no-run 跳过。")
            last_notice = now
        time.sleep(RUN_QUERY_INTERVAL)


def _cleanup(comm_started: bool) -> None:
    if comm_started:
        try:
            comm.light(4, False, timeout=2.0)
        except Exception as exc:
            print(f"[清理] light 4 off failed: {exc}")
    try:
        block.close()
    except Exception as exc:
        print(f"[清理] block camera close failed: {exc}")
    try:
        ring.close()
    except Exception as exc:
        print(f"[清理] ring camera close failed: {exc}")
    if comm_started:
        try:
            comm.shutdown()
        except Exception as exc:
            print(f"[清理] comm shutdown failed: {exc}")


def _reset_vision_between_tasks() -> None:
    """阶段切换只释放 block，ring 保持预热，避免 A/B 首次放置时重开 USB。"""
    print("[MAIN] 阶段切换：释放 block，保持/预热 ring USB 摄像头。")
    try:
        block.close()
    except Exception as exc:
        print(f"[MAIN] block camera reset failed: {exc}")
    try:
        ring.prepare_camera_async(verbose=True)
    except Exception as exc:
        print(f"[MAIN] ring camera prewarm failed: {exc}")
    time.sleep(0.2)


def run_all() -> None:
    test_cd = _load_task_module("task_cd", "test_C&D.py")

    print("\n=== MAIN: C/D 物块放置 -> A/B 奖杯放置 ===")
    test_cd.run_task_cd()
    _reset_vision_between_tasks()

    test_ab = _load_task_module("task_ab", "test_A&B.py")
    print("\n=== MAIN: C/D 完成，直接进入 A/B ===")
    print("[MAIN] 不插入过渡 GOTO；A/B 起点只做位姿同步，不做物理移动。")
    test_ab.run_task_ab()

    print("\n=== MAIN: 全流程完成 ===")


def main() -> int:
    argv = sys.argv[1:]
    port = _port_arg(argv)
    comm_started = False
    exit_code = 0

    try:
        print(f"  SERIAL {port or 'AUTO'} @ {BAUDRATE}")
        comm.init(port, BAUDRATE)
        comm_started = True
        ring.configure({"comm_port": port, "comm_baud": BAUDRATE})
        time.sleep(1.0)

        _wait_run_start_if_requested(argv)
        run_all()
    except RuntimeError as exc:
        exit_code = 1
        print(f"\n[兜底] 主流程中断: {exc}")
    except KeyboardInterrupt:
        exit_code = 130
        print("\n[用户中断]")
    finally:
        _cleanup(comm_started)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
