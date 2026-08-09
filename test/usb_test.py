import cv2

from xbhdcc_tools import detect_cameras,WebStreamer
import time

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

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    streamer = WebStreamer(port=8081)
    fps=0

    last_time = time.time()

    while True:
        ret,frame=cap.read()
        if not ret:
            continue
        streamer.update_frame(0,frame)
        frame_drawn = frame.copy()
        cv2.putText(frame_drawn, "fps: {}".format(round(fps, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
        streamer.update_frame(1, frame_drawn)
        curr_time = time.time()
        fps = (1 / (curr_time - last_time))*0.3+fps*0.7
        last_time = curr_time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
