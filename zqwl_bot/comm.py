"""Jetson Nano <-> STM32F407 串口通信.

协议帧: [0xAA][0x55][type][len][payload...][crc16_lo][crc16_hi]
- type/len/payload 参与 CRC16-CCITT (poly 0x1021, init 0xFFFF)
- payload 为 float32 数组, 小端 (STM32 ARM little-endian)
- 导航命令模式: 上位机发一条命令, MCU 执行完回一个 result, 上位机再发下一条
- 全局单例: init() / send_goto() / send_tox() / wait_nav_response() 等
"""

import logging
import math
import struct
import threading
import time

import serial

log = logging.getLogger("zqwl.serial")

HEADER = (0xAA, 0x55)

# 辅助命令
TYPE_CMD_VEL  = 0x01
TYPE_ROTATE   = 0x03   # 转盘位置切换, payload 1B (0-4)
TYPE_ARM      = 0x04   # 机械臂状态切换, payload 1B (0-7, 0=默认位)
TYPE_LIGHT    = 0x05   # 补光灯控制, payload 2B [id, on_off] (id: 0=全部, 1-4=单灯, 4=PA3/TIM5)
TYPE_RUN      = 0x06   # 启动命令(模拟按键PD15), payload 空
TYPE_RUN_RESP = 0x07   # 启动响应, payload 1B status
TYPE_ROTATE_RESP = 0x08   # 转盘响应(估算移动时间后), payload 1B status
TYPE_ARM_RESP    = 0x09   # 机械臂响应, payload 1B status
TYPE_LIGHT_RESP  = 0x0A   # 补光灯响应, payload 1B status

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
    TYPE_CMD_ARC_RESP, TYPE_RUN_RESP,
    TYPE_ROTATE_RESP, TYPE_ARM_RESP, TYPE_LIGHT_RESP,
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
    def __init__(self, port: str = "/dev/ttyCH341USB0", baudrate: int = 115200, timeout: float = 0.1):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._ser: serial.Serial | None = None
        self._rx_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # 导航响应 (Condition + 序列号: 每次收到新响应 seq+1,
        # 等待方记录"已消费到哪个 seq", 可识别并跳过陈旧/错类型响应)
        self._resp_cond = threading.Condition()
        self._nav_resp_seq: int = 0
        self._resp_sent_seq: int = 0   # 最近一次发命令时的 seq 基线 (发送瞬间记录)
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
        if not (0 <= pos <= 4):
            raise ValueError("rotate pos must be 0-4")
        self._mark_send()
        self._ser.write(pack_frame(TYPE_ROTATE, struct.pack("<B", pos)))

    def send_arm(self, state: int) -> None:
        if not self._ser:
            raise RuntimeError("serial not started")
        if not (0 <= state <= 7):
            raise ValueError("arm state must be 0-7")
        self._mark_send()
        self._ser.write(pack_frame(TYPE_ARM, struct.pack("<B", state)))

    def send_run(self) -> None:
        if not self._ser:
            raise RuntimeError("serial not started")
        self._send_nav(TYPE_RUN, b"")

    def send_light(self, light_id: int, on: bool) -> None:
        if not self._ser:
            raise RuntimeError("serial not started")
        if not (0 <= light_id <= 4):
            raise ValueError("light id must be 0-4 (0=all, 1-4=single)")
        self._mark_send()
        self._ser.write(pack_frame(TYPE_LIGHT, struct.pack("<BB", light_id, 1 if on else 0)))

    # --- 导航命令发送 ---

    def _mark_send(self) -> None:
        """记录发送时刻的 seq 基线 (必须在 write 之前调用)."""
        with self._resp_cond:
            self._resp_sent_seq = self._nav_resp_seq

    def _send_nav(self, msg_type: int, payload: bytes) -> None:
        """发送导航命令. (响应由 seq 机制区分, 无需在此清除.)"""
        if not self._ser:
            raise RuntimeError("serial not started")
        self._mark_send()
        self._ser.write(pack_frame(msg_type, payload))

    @staticmethod
    def _check_finite(*vals: float) -> None:
        for v in vals:
            if not math.isfinite(v):
                raise ValueError(f"non-finite value: {v}")

    def send_goto(self, x: float, y: float) -> None:
        self._check_finite(x, y)
        self._send_nav(TYPE_CMD_GOTO, struct.pack("<ff", x, y))

    def send_tox(self, x: float) -> None:
        self._check_finite(x)
        self._send_nav(TYPE_CMD_TOX, struct.pack("<f", x))

    def send_toy(self, y: float) -> None:
        self._check_finite(y)
        self._send_nav(TYPE_CMD_TOY, struct.pack("<f", y))

    def send_turnto(self, yaw_deg: float) -> None:
        self._check_finite(yaw_deg)
        self._send_nav(TYPE_CMD_TURNTO, struct.pack("<f", yaw_deg))

    def send_arc(self, cx: float, cy: float, r: float,
                 start_deg: float, end_deg: float) -> None:
        self._check_finite(cx, cy, r, start_deg, end_deg)
        if r <= 0:
            raise ValueError(f"arc radius must be > 0, got {r}")
        self._send_nav(TYPE_CMD_ARC, struct.pack("<fffff", cx, cy, r, start_deg, end_deg))

    def send_fine_move(self, dx_mm: float, dy_mm: float) -> None:
        self._check_finite(dx_mm, dy_mm)
        self._send_nav(TYPE_CMD_FINE_MOVE, struct.pack("<ff", dx_mm, dy_mm))

    def send_sync_pose(self, x: float, y: float) -> None:
        self._check_finite(x, y)
        self._send_nav(TYPE_CMD_SYNC_POSE, struct.pack("<ff", x, y))

    # --- 导航响应接收 ---

    def wait_nav_response(self, timeout: float = 30.0) -> tuple[int, int] | None:
        """阻塞等待 MCU 的下一个导航响应 (任意类型).

        Returns:
            (resp_type, status) on success, None on timeout.
            status: 1 = 到位成功, 0 = 未到位/超时.
        """
        deadline = time.monotonic() + timeout
        with self._resp_cond:
            seen = self._nav_resp_seq
            while self._nav_resp_seq == seen:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return None
                self._resp_cond.wait(remain)
            return (self._nav_resp_type, self._nav_resp_status)

    def wait_for(self, expect: int, timeout: float) -> bool:
        """等待指定类型的响应, 跳过类型不匹配的(陈旧)响应.

        基线 = 发送该命令瞬间记录的 _resp_sent_seq, 因此:
        - 命令发出前就已存在的旧响应不会被误认为本次响应;
        - 响应即使在 wait 调用前就已到达(极快应答), 也不会漏掉.

        对抗场景: 上一条命令超时后上位机放弃, 但 MCU 稍后才回响应 ——
        该迟到响应若被下一条命令的等待吃掉, 会造成假成功.
        靠 type 匹配 + seq 递增来跳过.

        Returns:
            True = 收到 expect 类型且 status==1; 否则 False (超时/失败).
        """
        deadline = time.monotonic() + timeout
        with self._resp_cond:
            seen = self._resp_sent_seq
            while True:
                while self._nav_resp_seq > seen:
                    seen = self._nav_resp_seq
                    if self._nav_resp_type == expect:
                        return self._nav_resp_status == 1
                    log.warning("skip stale resp type=0x%02x (expect 0x%02x)",
                                self._nav_resp_type, expect)
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return False
                self._resp_cond.wait(remain)

    # --- 高层运动命令 ("选择形式", 阻塞式, 返回成败) ---
    # 与下位机一一对应: 锁轴=TOX/TOY(1数据), 走点=GOTO(2数据)或GOTO+TURNTO(3数据),
    # 圆弧=ARC(5数据). 坐标单位米, yaw 单位度(CW+). 每条命令 MCU 阻塞执行后回 status.

    def lock_axis(self, axis: str, target: float, timeout: float = 40.0) -> bool:
        """锁轴移动 (1 个数据).

        axis='x': 锁住 Y 轴, 把 X 移到 target (发 TOX).
        axis='y': 锁住 X 轴, 把 Y 移到 target (发 TOY).
        target 单位米. 阻塞等待下位机响应.

        Returns:
            True = 命令完成. 注: 下位机 TOX/TOY 恒回 status=1,
            此处等待主要用于串行时序对齐, 避免下一条命令抢跑.
        """
        if axis == 'x':
            self.send_tox(target)
            expect = TYPE_CMD_TOX_RESP
        elif axis == 'y':
            self.send_toy(target)
            expect = TYPE_CMD_TOY_RESP
        else:
            raise ValueError("axis must be 'x' or 'y'")
        return self.wait_for(expect, timeout)

    def goto(self, x: float, y: float, yaw: float | None = None,
             timeout: float = 40.0) -> bool:
        """走点 (2 或 3 个数据).

        yaw=None: 2 数据 GOTO, 只走到 (x, y), 不管朝向.
        yaw 给定: 3 数据, 先 GOTO 走到点 → 等到位 → 再 TURNTO 转到目标朝向.
                  (下位机无单条"走点+转向"命令, 上位机自动串联两条.)
        x/y 单位米, yaw 单位度(CW+). 阻塞等待每一步响应.

        Returns:
            True = 全部到位/转到位; 任一步超时或失败返回 False.
        """
        self.send_goto(x, y)
        if not self.wait_for(TYPE_CMD_GOTO_RESP, timeout):
            return False
        if yaw is None:
            return True
        return self.turnto(yaw, timeout)

    def turnto(self, yaw_deg: float, timeout: float = 20.0) -> bool:
        """原地转到目标朝向 (角度环).

        yaw_deg 单位度(CW+). 下位机 RotateTo 真实闭环, 到位回 status=1.

        Returns:
            True = 转到位; 超时/未到位返回 False.
        """
        self.send_turnto(yaw_deg)
        return self.wait_for(TYPE_CMD_TURNTO_RESP, timeout)

    def arc(self, cx: float, cy: float, r: float,
            start_deg: float, end_deg: float, timeout: float = 70.0) -> bool:
        """圆弧 (单独函数, 5 个数据).

        圆心 (cx, cy) 与半径 r 单位米; start_deg/end_deg 单位度(CW+).
        下位机速度固定 0.10 m/s, sweep 自动归一化定方向. 阻塞等待响应.

        Returns:
            True = 圆弧走完; 超时返回 False.
        """
        self.send_arc(cx, cy, r, start_deg, end_deg)
        return self.wait_for(TYPE_CMD_ARC_RESP, timeout)

    def run(self, timeout: float = 5.0) -> bool:
        """启动命令 (模拟按下启动按键).

        下位机收到后 PD15 拉高 500ms 再释放 (默认低电平=未开始),
        执行完回 TYPE_RUN_RESP status=1. 阻塞等待响应.

        Returns:
            True = 启动脉冲已执行; 超时/异常响应返回 False.
        """
        self.send_run()
        return self.wait_for(TYPE_RUN_RESP, timeout)

    def rotate(self, pos: int, timeout: float = 10.0) -> bool:
        """转盘切槽位 (0-4), 阻塞等待 TYPE_ROTATE_RESP.

        下位机收到后移动转盘, 按估算移动时间等待后回响应 (status=1).
        注: 下位机只发不收 CAN, 响应代表"命令已执行+估算时间已到",
        非电机真实到位信号. timeout 默认 10s 覆盖上电回零期(约5s)的命令.
        """
        self.send_rotate(pos)          # 越界抛 ValueError
        return self.wait_for(TYPE_ROTATE_RESP, timeout)

    def arm(self, state: int, timeout: float = 5.0) -> bool:
        """机械臂姿态切换 (0-7, 0=默认位), 阻塞等待 TYPE_ARM_RESP.

        下位机五次多项式缓动到目标角度, 缓动完成后回响应 (最多滞后约2s).
        """
        self.send_arm(state)           # 越界抛 ValueError
        return self.wait_for(TYPE_ARM_RESP, timeout)

    def light(self, light_id: int, on: bool, timeout: float = 5.0) -> bool:
        """补光灯控制 (id: 0=全部, 1-4=单灯), 阻塞等待 TYPE_LIGHT_RESP."""
        self.send_light(light_id, on)  # 越界抛 ValueError
        return self.wait_for(TYPE_LIGHT_RESP, timeout)

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
            with self._resp_cond:
                self._nav_resp_type = msg_type
                self._nav_resp_status = payload[0]
                self._nav_resp_seq += 1
                self._resp_cond.notify_all()
        else:
            log.debug("unhandled type 0x%02x len=%d", msg_type, len(payload))


# --- 模块级单例 ---

_comm: SerialComm | None = None


def init(port: str = "/dev/ttyCH341USB0", baudrate: int = 115200) -> None:
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

def wait_for(expect: int, timeout: float = 30.0) -> bool:
    """等待指定类型的响应 (跳过陈旧/错类型), True=status 1."""
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.wait_for(expect, timeout)


# 高层运动命令单例 API ("选择形式": 锁轴 / 走点 / 圆弧, 阻塞式返回成败)

def lock_axis(axis: str, target: float, timeout: float = 40.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.lock_axis(axis, target, timeout)

def goto(x: float, y: float, yaw: float | None = None,
         timeout: float = 40.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.goto(x, y, yaw, timeout)

def arc(cx: float, cy: float, r: float,
        start_deg: float, end_deg: float, timeout: float = 70.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.arc(cx, cy, r, start_deg, end_deg, timeout)

def run(timeout: float = 5.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.run(timeout)

def turnto(yaw_deg: float, timeout: float = 20.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.turnto(yaw_deg, timeout)

def rotate(pos: int, timeout: float = 10.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.rotate(pos, timeout)

def arm(state: int, timeout: float = 5.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.arm(state, timeout)

def light(light_id: int, on: bool, timeout: float = 5.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.light(light_id, on, timeout)


# --- 自检 ---

def _self_check() -> None:
    """不接硬件, 验证 pack/parse/CRC 一致."""

    class _Shim(SerialComm):
        def start(self): self._stop = threading.Event(); self._reset_parser()

    c = _Shim()
    c.start()

    # 1. CMD_VEL 不应触发导航响应 (seq 不增)
    seq0 = c._nav_resp_seq
    for b in pack_frame(TYPE_CMD_VEL, struct.pack("<fff", 0.1, 0.2, 0.3)):
        c._feed_byte(b)
    assert c._nav_resp_seq == seq0, "CMD_VEL must not trigger nav response"

    # 2. GOTO_RESP 应触发响应 (seq+1, type/status 正确)
    for b in pack_frame(TYPE_CMD_GOTO_RESP, struct.pack("<B", 1)):
        c._feed_byte(b)
    assert c._nav_resp_seq == seq0 + 1
    assert c._nav_resp_type == TYPE_CMD_GOTO_RESP
    assert c._nav_resp_status == 1

    # 3. CRC 错误, 响应不更新 (seq 不增)
    seq1 = c._nav_resp_seq
    bad = bytearray(pack_frame(TYPE_CMD_TOX_RESP, struct.pack("<B", 1)))
    bad[5] ^= 0xFF
    for b in bad:
        c._feed_byte(b)
    assert c._nav_resp_seq == seq1, "corrupted frame must be dropped"

    # 4. ARC_RESP status=0
    for b in pack_frame(TYPE_CMD_ARC_RESP, struct.pack("<B", 0)):
        c._feed_byte(b)
    assert c._nav_resp_seq == seq1 + 1
    assert c._nav_resp_type == TYPE_CMD_ARC_RESP
    assert c._nav_resp_status == 0

    # 5. wait_nav_response 语义: 只计调用之后到达的新响应
    def _later_feed():
        time.sleep(0.02)
        for b in pack_frame(TYPE_CMD_GOTO_RESP, struct.pack("<B", 1)):
            c._feed_byte(b)
    tf = threading.Thread(target=_later_feed, daemon=True)
    tf.start()
    assert c.wait_nav_response(2.0) == (TYPE_CMD_GOTO_RESP, 1)
    tf.join()
    assert c.wait_nav_response(0.05) is None, "已到达的旧响应不算新响应"

    # 6. 高层命令: 假串口收到命令帧后立即回对应响应帧 (模拟 MCU 即时应答)
    class _EchoSerial:
        RESP = {
            TYPE_CMD_GOTO:    TYPE_CMD_GOTO_RESP,
            TYPE_CMD_TOX:     TYPE_CMD_TOX_RESP,
            TYPE_CMD_TOY:     TYPE_CMD_TOY_RESP,
            TYPE_CMD_TURNTO:  TYPE_CMD_TURNTO_RESP,
            TYPE_CMD_ARC:     TYPE_CMD_ARC_RESP,
            TYPE_RUN:         TYPE_RUN_RESP,
            TYPE_ROTATE:      TYPE_ROTATE_RESP,
            TYPE_ARM:         TYPE_ARM_RESP,
            TYPE_LIGHT:       TYPE_LIGHT_RESP,
        }

        def __init__(self, comm):
            self._comm = comm
            self.status = 1
            self.written: list[bytes] = []

        def write(self, data):
            self.written.append(bytes(data))
            resp_type = self.RESP.get(bytes(data)[2])   # 帧格式 AA 55 [type] ...
            if resp_type is not None:                   # 无响应命令(如VEL)不回帧
                for b in pack_frame(resp_type, struct.pack("<B", self.status)):
                    self._comm._feed_byte(b)
            return len(data)

    c._ser = _EchoSerial(c)

    # 6.1 lock_axis('x') → 一帧 TOX
    assert c.lock_axis('x', 0.5)
    assert c._ser.written[-1] == pack_frame(TYPE_CMD_TOX, struct.pack("<f", 0.5))

    # 6.2 lock_axis('y') → 一帧 TOY
    assert c.lock_axis('y', -0.25)
    assert c._ser.written[-1] == pack_frame(TYPE_CMD_TOY, struct.pack("<f", -0.25))

    # 6.3 非法 axis 抛 ValueError
    try:
        c.lock_axis('z', 0.0)
        assert False, "bad axis must raise"
    except ValueError:
        pass

    # 6.4 两数据 goto → 仅一帧 GOTO
    n = len(c._ser.written)
    assert c.goto(0.5, 0.6)
    assert len(c._ser.written) == n + 1
    assert c._ser.written[-1] == pack_frame(TYPE_CMD_GOTO, struct.pack("<ff", 0.5, 0.6))

    # 6.5 三数据 goto → GOTO + TURNTO 两帧, 顺序正确
    n = len(c._ser.written)
    assert c.goto(0.5, 0.6, yaw=90.0)
    assert len(c._ser.written) == n + 2
    assert c._ser.written[-2] == pack_frame(TYPE_CMD_GOTO, struct.pack("<ff", 0.5, 0.6))
    assert c._ser.written[-1] == pack_frame(TYPE_CMD_TURNTO, struct.pack("<f", 90.0))

    # 6.6 GOTO 失败 (status=0) → 不发 TURNTO, 返回 False
    c._ser.status = 0
    n = len(c._ser.written)
    assert not c.goto(0.1, 0.1, yaw=45.0)
    assert len(c._ser.written) == n + 1, "TURNTO must not follow failed GOTO"
    c._ser.status = 1

    # 6.7 arc → 一帧 ARC, 5 个 float
    assert c.arc(0.0, 0.0, 0.3, 0.0, 90.0)
    assert c._ser.written[-1] == pack_frame(
        TYPE_CMD_ARC, struct.pack("<fffff", 0.0, 0.0, 0.3, 0.0, 90.0))

    # 6.8 run → 空 payload 帧 AA 55 06 00 + 收到 RUN_RESP
    assert c.run()
    assert c._ser.written[-1] == pack_frame(TYPE_RUN, b"")

    # 6.9 send_rotate 范围校验: 0-4 合法, 越界抛 ValueError
    c.send_rotate(4)
    assert c._ser.written[-1] == pack_frame(TYPE_ROTATE, struct.pack("<B", 4))
    for badv in (-1, 5, 255):
        try:
            c.send_rotate(badv)
            assert False, "rotate out-of-range must raise"
        except ValueError:
            pass

    # 6.10 send_arm 范围校验: 0-7 合法 (0=默认位), 越界抛 ValueError
    c.send_arm(0)
    assert c._ser.written[-1] == pack_frame(TYPE_ARM, struct.pack("<B", 0))
    c.send_arm(7)
    assert c._ser.written[-1] == pack_frame(TYPE_ARM, struct.pack("<B", 7))
    for badv in (-1, 8):
        try:
            c.send_arm(badv)
            assert False, "arm out-of-range must raise"
        except ValueError:
            pass

    # 6.11 高层 rotate: 帧 + ROTATE_RESP 确认; 越界抛 ValueError
    assert c.rotate(3)
    assert c._ser.written[-1] == pack_frame(TYPE_ROTATE, struct.pack("<B", 3))
    try:
        c.rotate(5)
        assert False, "rotate out-of-range must raise"
    except ValueError:
        pass

    # 6.12 高层 arm: 帧 + ARM_RESP 确认
    assert c.arm(2)
    assert c._ser.written[-1] == pack_frame(TYPE_ARM, struct.pack("<B", 2))

    # 6.13 高层 light: 2B 帧 + LIGHT_RESP 确认; id 0-4 合法, 越界抛 ValueError
    assert c.light(1, True)
    assert c._ser.written[-1] == pack_frame(TYPE_LIGHT, struct.pack("<BB", 1, 1))
    assert c.light(0, False)          # 0 = 全部
    assert c.light(4, True)           # 4 = PA3/TIM5
    for badv in (-1, 5):
        try:
            c.light(badv, True)
            assert False, "light id out-of-range must raise"
        except ValueError:
            pass

    # 6.14 高层 turnto → 一帧 TURNTO + TURNTO_RESP 确认
    assert c.turnto(90.0)
    assert c._ser.written[-1] == pack_frame(TYPE_CMD_TURNTO, struct.pack("<f", 90.0))

    # 6.15 非有限值防护: NaN/inf 不上线; 圆弧半径必须 > 0
    for badv in (float('nan'), float('inf')):
        try:
            c.send_goto(badv, 0.0)
            assert False, "NaN/inf must raise"
        except ValueError:
            pass
    try:
        c.send_arc(0.0, 0.0, 0.0, 0.0, 90.0)
        assert False, "arc r<=0 must raise"
    except ValueError:
        pass

    # 6.16 陈旧响应防护 A: 发送前已到达的旧响应不算本次响应
    for b in pack_frame(TYPE_CMD_TOX_RESP, struct.pack("<B", 1)):
        c._feed_byte(b)                        # 旧响应 seq+1
    assert c.run(), "旧响应(发送前到达)不得顶替 RUN_RESP"
    assert c._ser.written[-1] == pack_frame(TYPE_RUN, b"")

    # 6.17 陈旧响应防护 B: 基线后到达的错类型响应被跳过, 正确响应仍命中
    class _StaleEcho(_EchoSerial):
        def write(self, data):
            if bytes(data)[2] == TYPE_RUN:
                # 模拟上一条超时命令的迟到 TOX_RESP 抢先到达
                for b in pack_frame(TYPE_CMD_TOX_RESP, struct.pack("<B", 1)):
                    self._comm._feed_byte(b)
            return super().write(data)
    c._ser = _StaleEcho(c)
    assert c.run(), "迟到的错类型响应必须被跳过, RUN_RESP 仍要命中"

    print("self-check OK")


if __name__ == "__main__":
    _self_check()
