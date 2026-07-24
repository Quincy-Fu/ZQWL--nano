"""Jetson Nano <-> STM32F407 串口通信.

协议帧: [0xAA][0x55][type][len][payload...][crc16_lo][crc16_hi]
- type/len/payload 参与 CRC16-CCITT (poly 0x1021, init 0xFFFF)
- payload 为 float32 数组, 小端 (STM32 ARM little-endian)
- 导航命令模式: 上位机发一条命令, MCU 执行完回一个 result, 上位机再发下一条
- 全局单例: init() / send_goto() / send_tox() / wait_nav_response() 等
"""

import logging
import struct
import threading

import serial

log = logging.getLogger("zqwl.serial")

HEADER = (0xAA, 0x55)

# 辅助命令
TYPE_CMD_VEL  = 0x01
TYPE_ROTATE   = 0x03   # 转盘位置切换, payload 1B (0-4)
TYPE_ARM      = 0x04   # 机械臂状态切换, payload 1B (0-7)
TYPE_LIGHT    = 0x05   # 补光灯控制, payload 2B [id, on_off]

# 导航命令 (偶数=命令, 奇数=响应)
TYPE_CMD_GOTO         = 0x10
TYPE_CMD_GOTO_RESP    = 0x11
TYPE_CMD_TOX          = 0x12
TYPE_CMD_TOX_RESP     = 0x13
TYPE_CMD_TOY          = 0x14
TYPE_CMD_TOY_RESP     = 0x15
TYPE_CMD_TURNTO       = 0x16
TYPE_CMD_TURNTO_RESP  = 0x17
TYPE_CMD_FINE_MOVE    = 0x18
TYPE_CMD_FINE_RESP    = 0x19
TYPE_CMD_SYNC_POSE    = 0x1A
TYPE_CMD_SYNC_RESP    = 0x1B
TYPE_CMD_ARC          = 0x1C
TYPE_CMD_ARC_RESP     = 0x1D

# 所有导航响应类型集合
_NAV_RESP_TYPES = frozenset({
    TYPE_CMD_GOTO_RESP, TYPE_CMD_TOX_RESP, TYPE_CMD_TOY_RESP,
    TYPE_CMD_TURNTO_RESP, TYPE_CMD_FINE_RESP, TYPE_CMD_SYNC_RESP,
    TYPE_CMD_ARC_RESP,
})

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
        # 导航响应
        self._nav_resp_event = threading.Event()
        self._nav_resp_type: int = 0
        self._nav_resp_status: int = 0
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

    # --- 辅助命令 (保留) ---

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

    # --- 导航命令发送 ---

    def _send_nav(self, msg_type: int, payload: bytes) -> None:
        """发送导航命令并清除上一次响应."""
        if not self._ser:
            raise RuntimeError("serial not started")
        self._nav_resp_event.clear()
        self._ser.write(pack_frame(msg_type, payload))

    def send_goto(self, x: float, y: float) -> None:
        self._send_nav(TYPE_CMD_GOTO, struct.pack("<ff", x, y))

    def send_tox(self, x: float) -> None:
        self._send_nav(TYPE_CMD_TOX, struct.pack("<f", x))

    def send_toy(self, y: float) -> None:
        self._send_nav(TYPE_CMD_TOY, struct.pack("<f", y))

    def send_turnto(self, yaw_deg: float) -> None:
        self._send_nav(TYPE_CMD_TURNTO, struct.pack("<f", yaw_deg))

    def send_arc(self, cx: float, cy: float, r: float,
                 start_deg: float, end_deg: float) -> None:
        self._send_nav(TYPE_CMD_ARC, struct.pack("<fffff", cx, cy, r, start_deg, end_deg))

    def send_fine_move(self, dx_mm: float, dy_mm: float) -> None:
        self._send_nav(TYPE_CMD_FINE_MOVE, struct.pack("<ff", dx_mm, dy_mm))

    def send_sync_pose(self, x: float, y: float) -> None:
        self._send_nav(TYPE_CMD_SYNC_POSE, struct.pack("<ff", x, y))

    # --- 导航响应接收 ---

    def wait_nav_response(self, timeout: float = 30.0) -> tuple[int, int] | None:
        """阻塞等待 MCU 导航响应.

        Returns:
            (resp_type, status) on success, None on timeout.
            status: 1 = 到位成功, 0 = 未到位/超时.
        """
        if self._nav_resp_event.wait(timeout):
            return (self._nav_resp_type, self._nav_resp_status)
        return None

    # --- RX loop & parser ---

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                n = self._ser.in_waiting if self._ser.in_waiting else 1
                chunk = self._ser.read(n)
            except Exception as exc:
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
        if msg_type in _NAV_RESP_TYPES and len(payload) == 1:
            self._nav_resp_type = msg_type
            self._nav_resp_status = payload[0]
            self._nav_resp_event.set()
        else:
            log.debug("unhandled type 0x%02x len=%d", msg_type, len(payload))


# --- 模块级单例 ---

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


# 导航命令单例 API
def send_goto(x: float, y: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_goto(x, y)

def send_tox(x: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_tox(x)

def send_toy(y: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_toy(y)

def send_turnto(yaw_deg: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_turnto(yaw_deg)

def send_arc(cx: float, cy: float, r: float,
             start_deg: float, end_deg: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_arc(cx, cy, r, start_deg, end_deg)

def send_fine_move(dx_mm: float, dy_mm: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_fine_move(dx_mm, dy_mm)

def send_sync_pose(x: float, y: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_sync_pose(x, y)

def wait_nav_response(timeout: float = 30.0) -> tuple[int, int] | None:
    if _comm is None:
        return None
    return _comm.wait_nav_response(timeout)


# --- 自检 ---

def _self_check() -> None:
    """不接硬件, 验证 pack/parse/CRC 一致."""

    class _Shim(SerialComm):
        def start(self): self._stop = threading.Event(); self._reset_parser()

    c = _Shim()
    c.start()

    # 1. CMD_VEL 不应触发导航响应
    for b in pack_frame(TYPE_CMD_VEL, struct.pack("<fff", 0.1, 0.2, 0.3)):
        c._feed_byte(b)
    assert not c._nav_resp_event.is_set(), "CMD_VEL must not trigger nav response"

    # 2. GOTO_RESP 应触发响应
    for b in pack_frame(TYPE_CMD_GOTO_RESP, struct.pack("<B", 1)):
        c._feed_byte(b)
    assert c._nav_resp_event.is_set()
    assert c._nav_resp_type == TYPE_CMD_GOTO_RESP
    assert c._nav_resp_status == 1

    # 3. CRC 错误, 响应不更新
    c._nav_resp_event.clear()
    bad = bytearray(pack_frame(TYPE_CMD_TOX_RESP, struct.pack("<B", 1)))
    bad[5] ^= 0xFF
    for b in bad:
        c._feed_byte(b)
    assert not c._nav_resp_event.is_set(), "corrupted frame must be dropped"

    # 4. ARC_RESP
    for b in pack_frame(TYPE_CMD_ARC_RESP, struct.pack("<B", 0)):
        c._feed_byte(b)
    assert c._nav_resp_event.is_set()
    assert c._nav_resp_type == TYPE_CMD_ARC_RESP
    assert c._nav_resp_status == 0

    print("self-check OK")


if __name__ == "__main__":
    _self_check()
