import cv2
import numpy as np
from xbhdcc_tools import detect_cameras, WebStreamer

# ponytail: ROI 中央 960x540（1080p 一半），用户保证二维码正对中央。
# detect ROI 而非全图，耗时降到 1/4。offset 用于 bbox 映射回原帧画框。
ROI_W, ROI_H = 960, 540
ROI_OFFSET_X = (1920 - ROI_W) // 2  # 480
ROI_OFFSET_Y = (1080 - ROI_H) // 2  # 270


if __name__ == "__main__":
    detect_cameras()

    cap = None
    for device in (0, 1):
        candidate = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if candidate.isOpened():
            cap = candidate
            print(f"USB camera opened at device {device}")
            break
        candidate.release()
    if cap is None:
        raise RuntimeError("USB camera open failed (tried devices 0, 1)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 降抓帧延迟，避免读到旧帧

    streamer = WebStreamer(port=8084)
    qr = cv2.QRCodeDetector()
    last_text = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            roi = frame[ROI_OFFSET_Y:ROI_OFFSET_Y + ROI_H, ROI_OFFSET_X:ROI_OFFSET_X + ROI_W]
            data, bbox, _ = qr.detectAndDecode(roi)

            if data and bbox is not None:
                pts = np.int32(bbox).reshape(-1, 2)
                pts[:, 0] += ROI_OFFSET_X
                pts[:, 1] += ROI_OFFSET_Y
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
                x, y = int(pts[0][0]), int(pts[0][1])
                cv2.putText(frame, "USB: {}".format(data), (x, max(y - 10, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if data != last_text:
                    print("[USB] {}".format(data))
                    last_text = data

            streamer.update_frame(0, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
