import os
import time
import logging

import cv2
import numpy as np
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

Gst.init(None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("qr_only")


# ============ 配置 ============
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

TASK2_PLANS = {
    "1": ["A", "B", "C"],
    "2": ["A", "C", "B"],
    "3": ["B", "A", "C"],
    "4": ["B", "C", "A"],
    "5": ["C", "A", "B"],
    "6": ["C", "B", "A"],
}


# ============ 二维码识别 ============
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
                else:
                    log.warning(f"prototxt 文件内容不对: {head[:50]}")
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


# ============ 摄像头 ============
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
    
    def stop(self):
        if self.pipe:
            self.pipe.set_state(Gst.State.NULL)


# ============ 主程序 ============
def main():
    print("=" * 50)
    print("Jetson Nano - 任务2 二维码识别")
    print("=" * 50)
    
    cam = CSICamera()
    cam.start()
    detector = QRDetector(MODEL_DIR)
    
    print("\n按 'q' 退出")
    
    last_data = None
    stable_count = 0
    
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            
            data = detector.detect(frame)
            
            display = frame.copy()
            
            if data:
                cv2.putText(display, f"二维码: {data}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                
                plan = TASK2_PLANS.get(data)
                if plan:
                    cv2.putText(display, f"方案: {plan}", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
                
                if data == last_data:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_data = data
                
                if stable_count == 3:
                    print(f"\n>>> 确认: {data}")
                    if plan:
                        print(f"    方案: {plan}")
            else:
                cv2.putText(display, "请将二维码对准摄像头中央", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                last_data = None
                stable_count = 0
            
            cv2.rectangle(display,
                         (ROI_OFFSET_X, ROI_OFFSET_Y),
                         (ROI_OFFSET_X + ROI_W, ROI_OFFSET_Y + ROI_H),
                         (255, 0, 0), 2)
            
            cv2.imshow("QR Scanner", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()