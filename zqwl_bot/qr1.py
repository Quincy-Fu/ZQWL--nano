"""
QR1 - 左侧二维码识别
扫数字 1-16, 返回 '1' 到 '16'
颜色顺序查表: TASK1_PLANS[key] -> 5 元素颜色列表

用法:
    python3 qr1.py             # 普通识别
    python3 qr1.py preview     # 带实时预览 (调试用)
"""
import os
import time
import logging
import threading

# 必须在 cv2 首次导入前设置；本文件通常早于 block/ring 被导入。
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")
os.environ.setdefault("OPENCV_VIDEOIO_V4L_READ_ATTEMPTS", "1")

import cv2
import numpy as np

import comm

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

Gst.init(None)
log = logging.getLogger("qr1")

# ============== 配置 ==============
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 tnr-mode=2 tnr-strength=0.3 ee-mode=2 ee-strength=0.5 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx, width=1920, height=1080 ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! "
    "appsink name=sink sync=false max-buffers=2 drop=true"
)

WB_GAIN = np.array([[1.047, 0, 0], [0, 1.0, 0], [0, 0, 0.952]], dtype=np.float32)

ROI_W, ROI_H = 960, 540
ROI_OFFSET_X = (1920 - ROI_W) // 2
ROI_OFFSET_Y = (1080 - ROI_H) // 2

CSI_START_VERIFY_TIMEOUT_S = 2.0
CSI_NO_FRAME_RESTART_S = 0.8
CSI_NO_DETECT_RESTART_S = 3.0
CSI_MAX_RESTARTS = 3
CSI_RESTART_SLEEP_S = 0.25

_VALID = set("0123456789")


def _frame_is_valid(frame) -> bool:
    """判断 CSI 帧是否有效；防止管线已启动但实际无画面/黑屏。"""
    if frame is None:
        return False
    if getattr(frame, "size", 0) <= 0:
        return False
    try:
        mean_v = float(np.mean(frame))
        std_v = float(np.std(frame))
    except Exception:
        return False
    if mean_v < 5.0:
        return False
    return not (mean_v < 25.0 and std_v < 1.0)

# ============== 16 种颜色顺序方案 ==============
TASK1_PLANS = {
    "1":  ["黑", "白", "红", "绿", "蓝"],
    "2":  ["白", "黑", "红", "绿", "蓝"],
    "3":  ["白", "黑", "绿", "红", "蓝"],
    "4":  ["蓝", "白", "黑", "红", "绿"],
    "5":  ["白", "红", "蓝", "黑", "绿"],
    "6":  ["黑", "红", "蓝", "白", "绿"],
    "7":  ["蓝", "绿", "黑", "白", "红"],
    "8":  ["绿", "白", "蓝", "黑", "红"],
    "9":  ["白", "绿", "黑", "蓝", "红"],
    "10": ["黑", "红", "蓝", "绿", "白"],
    "11": ["红", "蓝", "绿", "黑", "白"],
    "12": ["绿", "红", "黑", "蓝", "白"],
    "13": ["白", "红", "蓝", "绿", "黑"],
    "14": ["红", "绿", "白", "蓝", "黑"],
    "15": ["蓝", "白", "绿", "红", "黑"],
    "16": ["绿", "蓝", "红", "白", "黑"],
}


class QRDetector:
    def __init__(self, model_dir):
        self.det_type, self.det = self._init(model_dir)
        log.info(f"使用识别器: {self.det_type}")

    def _init(self, model_dir):
        proto = os.path.join(model_dir, "detect.prototxt")
        model = os.path.join(model_dir, "detect.caffemodel")
        sr_proto = os.path.join(model_dir, "sr.prototxt")
        sr_model = os.path.join(model_dir, "sr.caffemodel")

        all_valid = all(
            os.path.exists(p) and os.path.getsize(p) > 5000
            for p in [proto, model, sr_proto, sr_model]
        )

        if all_valid:
            try:
                with open(proto, "r", errors="ignore") as f:
                    head = f.read(100)
                if "layer" in head or "input" in head:
                    return ("wechat", cv2.wechat_qrcode_WeChatQRCode(
                        proto, model, sr_proto, sr_model
                    ))
            except Exception as e:
                log.warning(f"WeChatQRCode 初始化失败: {e}")

        return ("opencv", cv2.QRCodeDetector())

    def detect(self, frame):
        h, w = frame.shape[:2]
        if w >= ROI_W + 2 * ROI_OFFSET_X and h >= ROI_H + 2 * ROI_OFFSET_Y:
            roi = frame[ROI_OFFSET_Y:ROI_OFFSET_Y + ROI_H,
                        ROI_OFFSET_X:ROI_OFFSET_X + ROI_W]
            result = self._try_detect(roi)
            if result:
                return result
        return self._try_detect(frame)

    def _try_detect(self, img):
        if self.det_type == "wechat":
            results, _ = self.det.detectAndDecode(img)
            if results and results[0]:
                return results[0]
        else:
            data, _, _ = self.det.detectAndDecode(img)
            if data:
                return data
        return None


class CSICamera:
    def __init__(self):
        self.pipe = None
        self.sink = None

    def start(self):
        self.pipe = Gst.parse_launch(CSI_PIPELINE)
        ret = self.pipe.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("CSI 启动失败")
        self.sink = self.pipe.get_by_name("sink")
        log.info("CSI 启动成功")

        t0 = time.monotonic()
        while time.monotonic() - t0 < CSI_START_VERIFY_TIMEOUT_S:
            frame = self.read()
            if _frame_is_valid(frame):
                return
            time.sleep(0.03)
        self.stop(async_stop=False)
        raise RuntimeError("CSI 启动后未读到有效画面，疑似黑屏/Argus卡住")

    def read(self):
        if self.sink is None:
            return None
        sample = self.sink.emit("try-pull-sample", 0.01)
        if sample is None:
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            frame = np.frombuffer(info.data, dtype=np.uint8).reshape((h, w, 3)).copy()
            return cv2.transform(frame, WB_GAIN)
        finally:
            buf.unmap(info)

    def stop(self, async_stop: bool = False):
        pipe = self.pipe
        self.pipe = None
        self.sink = None
        if not pipe:
            return

        def _stop_pipe():
            t0 = time.monotonic()
            pipe.set_state(Gst.State.NULL)
            dt = time.monotonic() - t0
            log.info("[QR1 timing] CSI stop %.3fs", dt)
            print(f"[QR1 timing] CSI stop {dt:.3f}s", flush=True)

        if async_stop:
            threading.Thread(target=_stop_pipe, name="qr1-csi-stop", daemon=True).start()
        else:
            _stop_pipe()


def _start_csi_with_retries(label: str = "QR1") -> CSICamera:
    """启动 CSI 并验证首帧；失败时释放后重试，避免黑屏管线继续跑。"""
    last_error = None
    for idx in range(CSI_MAX_RESTARTS + 1):
        cam = CSICamera()
        try:
            cam.start()
            if idx > 0:
                print(f"[{label}] CSI 重启成功: {idx}/{CSI_MAX_RESTARTS}", flush=True)
            return cam
        except Exception as exc:
            last_error = exc
            print(f"[{label}] CSI 启动/首帧验证失败: {exc}，重试 {idx + 1}/{CSI_MAX_RESTARTS + 1}", flush=True)
            cam.stop(async_stop=False)
            time.sleep(CSI_RESTART_SLEEP_S)
    raise RuntimeError(f"{label}: CSI 多次启动失败: {last_error}")


def _is_valid(data: str) -> bool:
    if not data or len(data) < 1 or len(data) > 2:
        return False
    if not all(c in _VALID for c in data):
        return False
    try:
        n = int(data)
        return 1 <= n <= 16
    except ValueError:
        return False


def _set_light(light_id: int, on: bool) -> None:
    """设置补光灯, comm 未初始化时不报错。"""
    try:
        comm.send_light(light_id, on)
    except RuntimeError:
        # 单独运行 qr1.py 时 comm 未初始化, 忽略
        pass


def recognize(timeout: float = 10.0) -> str:
    """识别数字 1-16, 扫到合法内容立刻返回. 超时抛 RuntimeError.

    返回: "1" 到 "16" 字符串
    """
    t_start = time.monotonic()
    log.info("[QR1] 开补光灯 3")
    _set_light(3, True)
    cam = None
    try:
        cam = _start_csi_with_retries("QR1")
        detector = QRDetector(MODEL_DIR)
        t0 = time.time()
        attempts = 0
        restart_count = 0
        last_frame_time = time.time()
        last_detect_activity_time = time.time()
        while time.time() - t0 < timeout:
            frame = cam.read()
            now = time.time()
            if not _frame_is_valid(frame):
                if now - last_frame_time >= CSI_NO_FRAME_RESTART_S and restart_count < CSI_MAX_RESTARTS:
                    restart_count += 1
                    print(f"[QR1] CSI 无有效帧 {now - last_frame_time:.1f}s，重启管线 {restart_count}/{CSI_MAX_RESTARTS}", flush=True)
                    cam.stop(async_stop=False)
                    cam = _start_csi_with_retries("QR1")
                    last_frame_time = time.time()
                time.sleep(0.02)
                continue
            last_frame_time = now
            attempts += 1
            data = detector.detect(frame)

            if data:
                last_detect_activity_time = now
                log.info(f"  [QR1] 扫到: '{data}' (尝试 {attempts} 次)")
                if _is_valid(data):
                    log.info(f"  [QR1] 确认: {data} -> {TASK1_PLANS[data]}")
                    dt = time.monotonic() - t_start
                    log.info("[QR1 timing] detected in %.3fs", dt)
                    print(f"[QR1 timing] detected in {dt:.3f}s", flush=True)
                    return data
                else:
                    log.info(f"  [QR1] '{data}' 不在 1-16 范围, 跳过")
            elif now - last_detect_activity_time >= CSI_NO_DETECT_RESTART_S and restart_count < CSI_MAX_RESTARTS:
                restart_count += 1
                print(f"[QR1] 有有效画面但 {CSI_NO_DETECT_RESTART_S:.1f}s 未扫到二维码，重启管线 {restart_count}/{CSI_MAX_RESTARTS}", flush=True)
                cam.stop(async_stop=False)
                cam = _start_csi_with_retries("QR1")
                last_frame_time = time.time()
                last_detect_activity_time = time.time()
                continue

            if attempts % 40 == 0:
                log.info(f"  [QR1] 已尝试 {attempts} 次, 未扫到合法二维码")
                print(f"[QR1] 有有效画面，已尝试 {attempts} 帧，未扫到合法二维码", flush=True)
            time.sleep(0.05)
        raise RuntimeError(f"QR1: {timeout}s 内未识别到合法二维码 (1-16)")
    finally:
        t_cleanup = time.monotonic()
        _set_light(3, False)
        dt_light = time.monotonic() - t_cleanup
        log.info("[QR1 timing] light off %.3fs", dt_light)
        print(f"[QR1 timing] light off {dt_light:.3f}s", flush=True)
        if cam is not None:
            cam.stop(async_stop=True)
        dt_cleanup = time.monotonic() - t_cleanup
        log.info("[QR1 timing] return after cleanup %.3fs", dt_cleanup)
        print(f"[QR1 timing] return after cleanup {dt_cleanup:.3f}s", flush=True)


def recognize_color_order(timeout: float = 3.0) -> list:
    """直接返回 5 元素颜色顺序列表, 例如 ['黑', '白', '红', '绿', '蓝']."""
    key = recognize(timeout)
    return TASK1_PLANS[key]


def preflight_camera(label: str = "QR1/QR2-CSI预检") -> bool:
    """只验证 CSI 摄像头能启动并拿到有效帧；不要求画面中有二维码。"""
    cam = None
    try:
        cam = _start_csi_with_retries(label)
        print(f"[{label}] CSI 摄像头预检 OK", flush=True)
        return True
    except Exception as exc:
        print(f"[{label}] CSI 摄像头预检失败: {exc}", flush=True)
        return False
    finally:
        if cam is not None:
            cam.stop(async_stop=False)


def recognize_with_preview(timeout: float = 30.0) -> str:
    """带实时预览 - 调试用, 按 q 提前退出."""
    log.info("[QR1] 开补光灯 3")
    _set_light(3, True)
    cam = None
    try:
        cam = _start_csi_with_retries("QR1-preview")
        detector = QRDetector(MODEL_DIR)
        cv2.namedWindow("QR1 Preview", cv2.WINDOW_NORMAL)
        t0 = time.time()
        attempts = 0
        restart_count = 0
        last_frame_time = time.time()
        last_detect_activity_time = time.time()
        while time.time() - t0 < timeout:
            frame = cam.read()
            now = time.time()
            if not _frame_is_valid(frame):
                if now - last_frame_time >= CSI_NO_FRAME_RESTART_S and restart_count < CSI_MAX_RESTARTS:
                    restart_count += 1
                    print(f"[QR1-preview] CSI 无有效帧 {now - last_frame_time:.1f}s，重启管线 {restart_count}/{CSI_MAX_RESTARTS}", flush=True)
                    cam.stop(async_stop=False)
                    cam = _start_csi_with_retries("QR1-preview")
                    last_frame_time = time.time()
                    last_detect_activity_time = time.time()
                time.sleep(0.02)
                continue
            last_frame_time = now
            attempts += 1
            data = detector.detect(frame)
            display = frame.copy()
            cv2.rectangle(display,
                         (ROI_OFFSET_X, ROI_OFFSET_Y),
                         (ROI_OFFSET_X + ROI_W, ROI_OFFSET_Y + ROI_H),
                         (255, 0, 0), 2)

            if data:
                last_detect_activity_time = now
                cv2.putText(display, f"识别: '{data}'", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                if _is_valid(data):
                    plan = TASK1_PLANS.get(data, [])
                    cv2.putText(display, f"合法! {data} -> {plan}", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                else:
                    cv2.putText(display, "(不在 1-16, 忽略)", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(display, f"请对准 (尝试 {attempts})", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                if now - last_detect_activity_time >= CSI_NO_DETECT_RESTART_S and restart_count < CSI_MAX_RESTARTS:
                    restart_count += 1
                    print(f"[QR1-preview] 有有效画面但 {CSI_NO_DETECT_RESTART_S:.1f}s 未扫到二维码，重启管线 {restart_count}/{CSI_MAX_RESTARTS}", flush=True)
                    cam.stop(async_stop=False)
                    cam = _start_csi_with_retries("QR1-preview")
                    last_frame_time = time.time()
                    last_detect_activity_time = time.time()
                    continue

            cv2.imshow("QR1 Preview", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise RuntimeError("用户中断")
            time.sleep(0.05)
        raise RuntimeError(f"QR1: {timeout}s 内未识别")
    finally:
        _set_light(3, False)
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if "preview" in sys.argv:
        print("QR1 (预览模式):")
        print(recognize_with_preview())
    elif "color" in sys.argv:
        print("QR1 (颜色顺序):")
        print(recognize_color_order())
    else:
        print("QR1:")
        print(recognize())
