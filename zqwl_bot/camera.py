"""CSI 摄像头硬件层 — Jetson Nano + IMX708.

模块级单例: init_csi() / shutdown_csi() / get_csi_frame()
管道 + WB_GAIN 沿袭 test/CSI_test.py, csi_pull_frame 沿袭 test/dual_test.py.
"""

import contextlib
import logging
import os
import sys

import cv2
import gi
import numpy as np

os.environ.setdefault("GST_DEBUG", "0")
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp  # noqa: E402

Gst.init(None)

log = logging.getLogger("zqwl.camera")


@contextlib.contextmanager
def _suppress_stderr():
    # ponytail: 临时吞 stderr 抑制 Argus/GStreamer C 层日志, finally 恢复. 主程序运行时 stderr 正常.
    _devnull = open(os.devnull, "w")
    _old = sys.stderr
    sys.stderr = _devnull
    try:
        yield
    finally:
        sys.stderr = _old
        _devnull.close()


CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 tnr-mode=2 tnr-strength=0.3 ee-mode=2 ee-strength=0.5 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx, width=1920, height=1080 ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! "
    "appsink name=sink sync=false max-buffers=2 drop=true"
)

# ponytail: 沿袭 CSI_test.py 标定值. 偏蓝降 [0,0], 偏黄升 [0,0], 偏绿降 [1,1], 偏红降 [2,2].
WB_GAIN = np.array([[1.047, 0, 0], [0, 1.0, 0], [0, 0, 0.952]], dtype=np.float32)


def _csi_pull_frame(sink):
    sample = sink.try_pull_sample(0.01)
    if sample is None:
        return None
    buf = sample.get_buffer()
    caps = sample.get_caps().get_structure(0)
    w, h = caps.get_value("width"), caps.get_value("height")
    ok, info = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        return np.frombuffer(info.data, dtype=np.uint8).reshape((h, w, 3)).copy()
    finally:
        buf.unmap(info)


class CSICamera:
    def __init__(self):
        self._pipe = None
        self._sink = None

    def start(self) -> None:
        with _suppress_stderr():
            self._pipe = Gst.parse_launch(CSI_PIPELINE)
            ret = self._pipe.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("CSI 管道启动失败, 检查排线 / sensor-id / argus 守护进程")
        self._sink = self._pipe.get_by_name("sink")

    def stop(self) -> None:
        if self._pipe is not None:
            with _suppress_stderr():
                self._pipe.set_state(Gst.State.NULL)
            self._pipe = None
            self._sink = None

    def get_frame(self):
        if self._sink is None:
            return None
        frame = _csi_pull_frame(self._sink)
        if frame is None:
            return None
        return cv2.transform(frame, WB_GAIN)


_csi: CSICamera | None = None


def init_csi() -> None:
    global _csi
    if _csi is not None:
        _csi.stop()
    _csi = CSICamera()
    _csi.start()


def shutdown_csi() -> None:
    global _csi
    if _csi is not None:
        _csi.stop()
        _csi = None


def get_csi_frame():
    if _csi is None:
        return None
    return _csi.get_frame()


def _self_check() -> None:
    """不接硬件, 验证常量 + 实例化."""
    assert "nvarguscamerasrc" in CSI_PIPELINE
    assert WB_GAIN.shape == (3, 3)
    c = CSICamera()
    assert c._pipe is None and c._sink is None
    print("self-check OK")


if __name__ == "__main__":
    _self_check()
