import os
import sys
import cv2
import numpy as np
import gi

os.environ.setdefault('GST_DEBUG', '0')
_devnull = open(os.devnull, 'w')

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp

sys.stderr = _devnull

from xbhdcc_tools import detect_cameras, WebStreamer

Gst.init(None)

# CSI: 沿袭 CSI_test.py — nvarguscamerasrc appsink + 1920x1080 + tnr/ee
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

    pipe = Gst.parse_launch(CSI_PIPELINE)
    if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("CSI 管道启动失败，检查排线 / sensor-id / argus 守护进程")
        raise SystemExit(1)
    sink = pipe.get_by_name("sink")

    streamer = WebStreamer(port=8086)
    qr = cv2.QRCodeDetector()
    last_text = ""

    try:
        while True:
            frame = csi_pull_frame(sink)
            if frame is None:
                continue
            frame = cv2.transform(frame, WB_GAIN)

            data, bbox, _ = qr.detectAndDecode(frame)

            if data and bbox is not None:
                pts = np.int32(bbox).reshape(-1, 2)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
                x, y = int(pts[0][0]), int(pts[0][1])
                cv2.putText(frame, "CSI: {}".format(data), (x, max(y - 10, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if data != last_text:
                    print("[CSI] {}".format(data))
                    last_text = data

            streamer.update_frame(0, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipe.set_state(Gst.State.NULL)
