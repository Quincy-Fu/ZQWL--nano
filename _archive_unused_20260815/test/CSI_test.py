import time
import cv2
import numpy as np
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp

from xbhdcc_tools import detect_cameras, WebStreamer

Gst.init(None)


# ponytail: OpenCV 自编版 GStreamer 后端为 NO，只能绕开 cv2.VideoCapture，
# 直接用 gi.repository.Gst 从 appsink 拉帧。统一 1920x1080@55（nvvidconv 硬件缩放）。
CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 tnr-mode=2 tnr-strength=0.3 ee-mode=2 ee-strength=0.5 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx, width=1920, height=1080 ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! "
    "appsink name=sink sync=false max-buffers=2 drop=true"
)

# ponytail: 装 camera_overrides.isp 后拍白纸标定 (跑 calib_wb.py) 得初始值，手动微调。
# 偏蓝降 [0,0] (b_gain)，偏黄升 [0,0]；偏绿降 [1,1] 或升 [0,0]/[2,2]；偏红降 [2,2] (r_gain)。
WB_GAIN = np.array([[1.047, 0, 0], [0, 1.0, 0], [0, 0, 0.952]], dtype=np.float32)
if __name__ == "__main__":
    detect_cameras()
    pipe = Gst.parse_launch(CSI_PIPELINE)
    state_ret = pipe.set_state(Gst.State.PLAYING)
    if state_ret == Gst.StateChangeReturn.FAILURE:
        print(f"管道启动失败，检查排线 / sensor-id / argus 守护进程")
        raise SystemExit(1)
    sink = pipe.get_by_name("sink")

    streamer = WebStreamer(port=8081)
    fps = 0
    last_time = time.time()

    try:
        while True:
            sample = sink.try_pull_sample(0.1)
            if sample is None:
                continue
            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            w, h = caps.get_value("width"), caps.get_value("height")
            ok, info = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                frame = np.frombuffer(info.data, dtype=np.uint8).reshape((h, w, 3))
                frame = cv2.transform(frame, WB_GAIN)
                streamer.update_frame(0, frame)
                frame_drawn = frame.copy()
                cv2.putText(frame_drawn, "fps: {}".format(round(fps, 2)), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
                streamer.update_frame(1, frame_drawn)
            finally:
                buf.unmap(info)

            curr = time.time()
            fps = 1 / (curr - last_time) * 0.3 + fps * 0.7
            last_time = curr
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.set_state(Gst.State.NULL)
