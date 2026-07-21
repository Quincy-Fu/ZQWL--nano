import os
import sys
import time
import cv2
import numpy as np
import gi

# ponytail: 抑制 nvarguscamerasrc/Argus 的 C 层 stderr 日志（GST_DEBUG=0 关 GStreamer 核心，
# stderr 重定向 devnull 吞 Argus 的 printf）。print 走 stdout 不受影响，错误靠 set_state 返回值判断。
os.environ.setdefault('GST_DEBUG', '0')
_devnull = open(os.devnull, 'w')

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp

sys.stderr = _devnull

from xbhdcc_tools import detect_cameras, WebStreamer

Gst.init(None)

# USB: 沿袭 usb_test.py — cv2.VideoCapture(0, CAP_V4L2) + MJPG + 1920x1080@30
# CSI: 沿袭 CSI_test.py — nvarguscamerasrc appsink + 1920x1080@55 + tnr/ee + WB_GAIN
CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 tnr-mode=2 tnr-strength=0.3 ee-mode=2 ee-strength=0.5 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx, width=1920, height=1080 ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! "
    "appsink name=sink sync=false max-buffers=2 drop=true"
)

# ponytail: WB_GAIN 沿袭 CSI_test.py。偏蓝降 [0,0]，偏黄升 [0,0]，偏绿降 [1,1]，偏红降 [2,2]。
WB_GAIN = np.array([[1.047, 0, 0], [0, 1.0, 0], [0, 0, 0.952]], dtype=np.float32)

def csi_pull_frame(sink):
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


if __name__ == "__main__":
    detect_cameras()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    csi_pipe = Gst.parse_launch(CSI_PIPELINE)
    if csi_pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("CSI 管道启动失败，检查排线 / sensor-id / argus 守护进程")
        raise SystemExit(1)
    csi_sink = csi_pipe.get_by_name("sink")

    streamer = WebStreamer(port=8082)
    usb_fps = 0
    csi_fps = 0
    usb_last = time.time()
    csi_last = time.time()

    try:
        while True:
            ret, usb_frame = cap.read()
            if ret:
                usb_drawn = usb_frame.copy()
                cv2.putText(usb_drawn, "USB fps: {}".format(round(usb_fps, 2)), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
                streamer.update_frame(0, usb_drawn)
                curr = time.time()
                usb_fps = 1 / (curr - usb_last) * 0.3 + usb_fps * 0.7
                usb_last = curr

            csi_frame = csi_pull_frame(csi_sink)
            if csi_frame is not None:
                csi_frame = cv2.transform(csi_frame, WB_GAIN)
                csi_drawn = csi_frame.copy()
                cv2.putText(csi_drawn, "CSI fps: {}".format(round(csi_fps, 2)), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
                streamer.update_frame(1, csi_drawn)
                curr = time.time()
                csi_fps = 1 / (curr - csi_last) * 0.3 + csi_fps * 0.7
                csi_last = curr

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        csi_pipe.set_state(Gst.State.NULL)
