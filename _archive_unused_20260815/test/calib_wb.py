import time
import numpy as np
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp

Gst.init(None)

# ponytail: 拍白纸标定白平衡。复用 CSI_test.py 管道，appsink 拉帧，
# 取画面中心 256x256 区域算 R/G/B 均值，G 归一化为 1.0，R/B 按比例补偿。
# 用法：白纸放画面中央，跑这个脚本，3 秒后自动采 10 帧平均出 WB_GAIN。
CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 tnr-mode=2 tnr-strength=1.0 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx, width=2304, height=1296 ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! "
    "appsink name=sink sync=false max-buffers=2 drop=true"
)

SAMPLE_SIZE = 256  # 中心采样区域边长
N_AVG = 10  # 采 N 帧平均降噪


def pull_frame(sink):
    sample = sink.try_pull_sample(0.1)
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


def main():
    pipe = Gst.parse_launch(CSI_PIPELINE)
    if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("管道启动失败，检查排线 / sensor-id / argus 守护进程")
        return 1
    sink = pipe.get_by_name("sink")

    print("预热 3 秒，请把白纸放画面中央...")
    time.sleep(3)

    frames = []
    for _ in range(N_AVG * 4):
        f = pull_frame(sink)
        if f is not None:
            frames.append(f)
        if len(frames) >= N_AVG:
            break
        time.sleep(0.05)
    pipe.set_state(Gst.State.NULL)

    if len(frames) < N_AVG // 2:
        print(f"只拉到 {len(frames)} 帧，管道有问题")
        return 1

    h, w = frames[0].shape[:2]
    cx, cy = w // 2, h // 2
    half = SAMPLE_SIZE // 2
    patch = np.stack([f[cy-half:cy+half, cx-half:cx+half] for f in frames]).astype(np.float32).mean(axis=0)
    mean_b, mean_g, mean_r = patch[:, :, 0].mean(), patch[:, :, 1].mean(), patch[:, :, 2].mean()
    print(f"\n中心 RGB 均值 (BGR): B={mean_b:.1f} G={mean_g:.1f} R={mean_r:.1f}")
    print(f"R/G={mean_r/mean_g:.3f}  B/G={mean_b/mean_g:.3f}")
    if max(mean_b, mean_g, mean_r) / max(min(mean_b, mean_g, mean_r), 1.0) > 1.5:
        print("警告：RGB 差异大，可能没对准白纸，建议重跑")

    # ponytail: G 归一化为 1.0，R/B 按比例拉高。偏绿时 r_gain/b_gain > 1。
    # 如果 r_gain/b_gain > 1.5 说明偏色太重，拉高会过曝，那时再换压 G 方案。
    r_gain = mean_g / mean_r
    b_gain = mean_g / mean_b
    print("\n把下面这行复制到 CSI_test.py 替换原 WB_GAIN：")
    print(f"WB_GAIN = np.array([[{b_gain:.3f}, 0, 0], [0, 1.0, 0], [0, 0, {r_gain:.3f}]], dtype=np.float32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
