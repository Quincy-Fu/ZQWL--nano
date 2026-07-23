"""Jetson Nano <-> STM32F407 串口通信.

协议帧: [0xAA][0x55][type][len][payload...][crc16_lo][crc16_hi]
- type/len/payload 参与 CRC16-CCITT (poly 0x1021, init 0xFFFF)
- payload 为 float32 数组, 小端 (STM32 ARM little-endian)
- 上位机发 CMD_VEL (vx,vy,w) / ROTATE (触发旋转), 收 POSE (x,y,theta)
- 全局单例: init() / send_velocity() / send_rotate() / get_pose()
"""

import logging
import struct
import threading

import serial

log = logging.getLogger("zqwl.serial")

HEADER = (0xAA, 0x55)
TYPE_CMD_VEL = 0x01
TYPE_POSE = 0x02
TYPE_ROTATE = 0x03  # 转盘位置切换, payload 1B (0-4)
TYPE_ARM = 0x04  # 机械臂状态切换, payload 1B (0-7)
TYPE_LIGHT = 0x05  # 补光灯控制, payload 2B [id, on_off]
# ponytail: 单线程状态机 + if-elif 分发, 消息类型 >3 类再考虑 dispatch table

FRAME_OVERHEAD = 2 + 1 + 1 + 2  # header + type + len + crc16


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def pack_frame(msg_type: int, payload: bytes) -> bytes:
    body = bytes([msg_type, len(payload)]) + payload
    crc = crc16_ccitt(body)
    return bytes(HEADER) + body + struct.pack("<H", crc)


class SerialComm:
    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 115200, timeout: float = 0.1):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._ser: serial.Serial | None = None
        self._rx_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pose: tuple[float, float, float] | None = None
        self._reset_parser()

    def start(self) -> None:
        self._ser = serial.Serial(self._port, self._baudrate, timeout=self._timeout)
        self._stop.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="serial-rx", daemon=True)
        self._rx_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
        if self._ser:
            self._ser.close()
            self._ser = None

    def send_velocity(self, vx: float, vy: float, w: float) -> None:
        if not self._ser:
            raise RuntimeError("serial not started")
        frame = pack_frame(TYPE_CMD_VEL, struct.pack("<fff", vx, vy, w))
        self._ser.write(frame)

    def send_rotate(self, pos: int) -> None:
        if not self._ser:
            raise RuntimeError("serial not started")
        self._ser.write(pack_frame(TYPE_ROTATE, struct.pack("<B", pos)))

    def send_arm(self, state: int) -> None:
        if not self._ser:
            raise RuntimeError("serial not started")
        self._ser.write(pack_frame(TYPE_ARM, struct.pack("<B", state)))

    def send_light(self, light_id: int, on: bool) -> None:
        if not self._ser:
            raise RuntimeError("serial not started")
        self._ser.write(pack_frame(TYPE_LIGHT, struct.pack("<BB", light_id, on)))

    def get_pose(self) -> tuple[float, float, float] | None:
        with self._lock:
            return self._pose

    # --- RX loop & parser ---

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                n = self._ser.in_waiting if self._ser.in_waiting else 1
                chunk = self._ser.read(n)
            except Exception as exc:  # ponytail: 串口异常只记日志, 不杀线程
                log.warning("rx read error: %s", exc)
                continue
            if not chunk:
                continue
            for b in chunk:
                self._feed_byte(b)

    def _reset_parser(self) -> None:
        self._state = "H1"
        self._type = 0
        self._len = 0
        self._payload = bytearray()

    def _feed_byte(self, b: int) -> None:
        s = self._state
        if s == "H1":
            if b == HEADER[0]:
                self._state = "H2"
            return
        if s == "H2":
            # 0x55→TYPE; 0xAA 留 H2 视作新帧起点; 其他回 H1
            if b == HEADER[1]:
                self._state = "TYPE"
            elif b != HEADER[0]:
                self._state = "H1"
            return
        if s == "TYPE":
            self._type = b
            self._state = "LEN"
            return
        if s == "LEN":
            self._len = b
            self._payload = bytearray()
            self._state = "PAY" if b > 0 else "CRC_LO"
            return
        if s == "PAY":
            self._payload.append(b)
            if len(self._payload) == self._len:
                self._state = "CRC_LO"
            return
        if s == "CRC_LO":
            self._crc_lo = b
            self._state = "CRC_HI"
            return
        if s == "CRC_HI":
            crc_recv = self._crc_lo | (b << 8)
            body = bytes([self._type, self._len]) + bytes(self._payload)
            if crc16_ccitt(body) == crc_recv:
                self._dispatch(self._type, bytes(self._payload))
            else:
                log.warning("crc mismatch on type=0x%02x", self._type)
            self._reset_parser()
            return

    def _dispatch(self, msg_type: int, payload: bytes) -> None:
        if msg_type == TYPE_POSE and len(payload) == 12:
            x, y, theta = struct.unpack("<fff", payload)
            with self._lock:
                self._pose = (x, y, theta)
        elif msg_type == TYPE_CMD_VEL:
            pass  # 上位机发的指令, RX 端一般收不到 (环回除外)
        else:
            log.debug("unknown type 0x%02x len=%d", msg_type, len(payload))


# --- 模块级单例 (全局变量入口) ---

_comm: SerialComm | None = None


def init(port: str = "/dev/ttyACM0", baudrate: int = 115200) -> None:
    global _comm
    if _comm is not None:
        _comm.stop()
    _comm = SerialComm(port, baudrate)
    _comm.start()


def shutdown() -> None:
    global _comm
    if _comm is not None:
        _comm.stop()
        _comm = None


def send_velocity(vx: float, vy: float, w: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    _comm.send_velocity(vx, vy, w)


def send_rotate(pos: int) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    _comm.send_rotate(pos)


def send_arm(state: int) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    _comm.send_arm(state)


def send_light(light_id: int, on: bool) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    _comm.send_light(light_id, on)


def get_pose() -> tuple[float, float, float] | None:
    if _comm is None:
        return None
    return _comm.get_pose()


def _self_check() -> None:
    """不接硬件, 验证 pack/parse/CRC 一致."""

    class _Shim(SerialComm):
        # 复用 _feed_byte/_dispatch, 不开串口
        def start(self): self._stop = threading.Event(); self._reset_parser()

    c = _Shim()
    c.start()

    # 1. CMD_VEL 不应更新 pose
    for b in pack_frame(TYPE_CMD_VEL, struct.pack("<fff", 0.1, 0.2, 0.3)):
        c._feed_byte(b)
    assert c.get_pose() is None, "CMD_VEL must not update pose"

    # 2. POSE 帧应同步到 pose
    for b in pack_frame(TYPE_POSE, struct.pack("<fff", 1.0, 2.0, 3.0)):
        c._feed_byte(b)
    assert c.get_pose() == (1.0, 2.0, 3.0), c.get_pose()

    # 3. 翻转一个 payload 字节, CRC 失败, pose 不更新
    good = bytearray(pack_frame(TYPE_POSE, struct.pack("<fff", 9.0, 9.0, 9.0)))
    good[6] ^= 0xFF  # payload 第一字节
    before = c.get_pose()
    for b in good:
        c._feed_byte(b)
    assert c.get_pose() == before, "corrupted frame must be dropped"

    print("self-check OK")


if __name__ == "__main__":
    _self_check()
