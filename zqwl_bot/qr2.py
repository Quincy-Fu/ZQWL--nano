"""
qr2.py - 右侧二维码识别
扫数字 1-6, 查表返回 3 字母顺序 (ABC/CAB/.../CBA)
返回: "CAB" 这种字符串

用法:
    python3 qr2.py             # 普通识别
    python3 qr2.py preview     # 带实时预览 (调试用)
"""
import os
import time
import logging
import threading
import cv2
import numpy as np

import comm

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

Gst.init(None)
log = logging.getLogger("qr2")

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

# 数字 1-6 -> 3 字母顺序 (按规则文档核对!)
QR2_NUM_TO_LETTERS = {
    "1": "ABC",
    "2": "ACB",
    "3": "BAC",
    "4": "BCA",
    "5": "CAB",
    "6": "CBA",
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
            log.info("[QR2 timing] CSI stop %.3fs", dt)
            print(f"[QR2 timing] CSI stop {dt:.3f}s", flush=True)

        if async_stop:
            threading.Thread(target=_stop_pipe, name="qr2-csi-stop", daemon=True).start()
        else:
            _stop_pipe()


def _is_valid_num(data: str) -> bool:
    """检查是否是 1-6 的数字字符串"""
    if not data or len(data) != 1:
        return False
    return data in QR2_NUM_TO_LETTERS


def _set_light(light_id: int, on: bool) -> None:
    """设置补光灯, comm 未初始化时不报错。"""
    try:
        comm.send_light(light_id, on)
    except RuntimeError:
        # 单独运行 qr2.py 时 comm 未初始化, 忽略
        pass


def recognize(timeout: float = 3.0) -> str:
    """识别数字 1-6, 查表返回 3 字母顺序 ('CAB' 这种). 超时抛 RuntimeError."""
    cam = CSICamera()
    t_start = time.monotonic()
    cam.start()
    detector = QRDetector(MODEL_DIR)
    log.info("[QR2] 开补光灯 3")
    _set_light(3, True)
    try:
        t0 = time.time()
        attempts = 0
        while time.time() - t0 < timeout:
            frame = cam.read()
            if frame is None:
                continue
            attempts += 1
            data = detector.detect(frame)

            if data:
                log.info(f"  [QR2] 扫到: '{data}' (尝试 {attempts} 次)")
                if _is_valid_num(data):
                    letters = QR2_NUM_TO_LETTERS[data]
                    log.info(f"  [QR2] 确认: {data} -> {letters}")
                    dt = time.monotonic() - t_start
                    log.info("[QR2 timing] detected in %.3fs", dt)
                    print(f"[QR2 timing] detected in {dt:.3f}s", flush=True)
                    return letters
                else:
                    log.info(f"  [QR2] '{data}' 不在 1-6, 跳过")

            if attempts % 20 == 0:
                log.info(f"  [QR2] 已尝试 {attempts} 次, 未扫到合法二维码")
            time.sleep(0.05)
        raise RuntimeError(f"QR2: {timeout}s 内未识别到合法二维码 (1-6)")
    finally:
        t_cleanup = time.monotonic()
        _set_light(3, False)
        dt_light = time.monotonic() - t_cleanup
        log.info("[QR2 timing] light off %.3fs", dt_light)
        print(f"[QR2 timing] light off {dt_light:.3f}s", flush=True)
        cam.stop(async_stop=True)
        dt_cleanup = time.monotonic() - t_cleanup
        log.info("[QR2 timing] return after cleanup %.3fs", dt_cleanup)
        print(f"[QR2 timing] return after cleanup {dt_cleanup:.3f}s", flush=True)


def recognize_with_preview(timeout: float = 30.0) -> str:
    """带实时预览 - 调试用, 按 q 提前退出."""
    cam = CSICamera()
    cam.start()
    detector = QRDetector(MODEL_DIR)
    cv2.namedWindow("QR2 Preview", cv2.WINDOW_NORMAL)
    log.info("[QR2] 开补光灯 3")
    _set_light(3, True)
    try:
        t0 = time.time()
        attempts = 0
        while time.time() - t0 < timeout:
            frame = cam.read()
            if frame is None:
                continue
            attempts += 1
            data = detector.detect(frame)
            display = frame.copy()
            cv2.rectangle(display,
                         (ROI_OFFSET_X, ROI_OFFSET_Y),
                         (ROI_OFFSET_X + ROI_W, ROI_OFFSET_Y + ROI_H),
                         (255, 0, 0), 2)

            if data:
                cv2.putText(display, f"识别: '{data}'", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                if _is_valid_num(data):
                    letters = QR2_NUM_TO_LETTERS[data]
                    cv2.putText(display, f"映射: {data} -> {letters}", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                else:
                    cv2.putText(display, "(不在 1-6, 忽略)", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(display, f"请对准 (尝试 {attempts})", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

            cv2.imshow("QR2 Preview", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise RuntimeError("用户中断")
            time.sleep(0.05)
        raise RuntimeError(f"QR2: {timeout}s 内未识别")
    finally:
        _set_light(3, False)
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if "preview" in sys.argv:
        print("QR2 (预览模式):")
        print(recognize_with_preview())
    else:
        print("QR2:")
        print(recognize())
