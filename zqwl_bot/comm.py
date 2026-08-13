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
from collections import deque

import serial

log = logging.getLogger("zqwl.serial")

HEADER = (0xAA, 0x55)

# 辅助命令
TYPE_CMD_VEL  = 0x01
TYPE_POSE     = 0x02   # 下位机50Hz上报位姿, payload 12B [x(f32), y(f32), theta(f32,CW弧度,编码器)]
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

# 连续路径跟踪: BEGIN装载参数, POINT装载路径点, EXEC触发执行
TYPE_CMD_PATH_BEGIN    = 0x22   # payload: speed(f32)+count(u8)
TYPE_CMD_PATH_POINT    = 0x23   # payload: x(f32)+y(f32)+target_theta(f32)+mode(u8)+pad(3B)
TYPE_CMD_PATH_EXEC     = 0x24   # payload 空
TYPE_CMD_PATH_RESP     = 0x25   # payload 1B status

PATH_MODE_NORMAL = 0  # 普通通过点: 不停, 进入通过半径后切下一段
PATH_MODE_KEY    = 1  # 关键点: 分段提前转到该点yaw, 位置和姿态到位后切段

# 视觉微调 (到位后视觉闭环方向微调, 体坐标系)
TYPE_CMD_VISION_NUDGE       = 0x27   # PC->MCU: payload 1B direction
TYPE_CMD_VISION_NUDGE_RESP  = 0x28   # MCU->PC: payload 1B status (1=executed)

# 转盘零点设置 (一次性标定: 将当前位置存为零点并写入 flash)
TYPE_CMD_SET_ZERO        = 0x29   # PC->MCU: payload 空
TYPE_CMD_SET_ZERO_RESP   = 0x2A   # MCU->PC: payload 1B status (1=saved)

# 视觉校正 (弧后视觉闭环: field-frame fine_move + sync_pose 原子组合)
TYPE_CMD_VISION_CORRECT       = 0x2B   # PC->MCU: 16B = field dx_mm + field dy_mm + target_x_m + target_y_m
TYPE_CMD_VISION_CORRECT_RESP  = 0x2C   # MCU->PC: payload 1B status (1=到位且坐标已重置)

# IMU 零偏校准 (车必须静止; 下位机转发 IMU 0x70 校准命令)
TYPE_CMD_IMU_CALIB       = 0x2D   # PC->MCU: payload 空
TYPE_CMD_IMU_CALIB_RESP  = 0x2E   # MCU->PC: payload 1B status (1=IMU帧恢复)
TYPE_CMD_ARC_ROTATE      = 0x2F   # PC->MCU: 圆弧 + 3个转盘触发点
TYPE_CMD_ARC_ROTATE_RESP = 0x30   # MCU->PC: payload 1B status

# 视觉微调方向码
NUDGE_STOP     = 0   # 立即停止+电磁锁死
NUDGE_FORWARD  = 1   # 体坐标系前进 (+Y)
NUDGE_BACKWARD = 2   # 体坐标系后退 (-Y)
NUDGE_LEFT     = 3   # 体坐标系左移 (-X)
NUDGE_RIGHT    = 4   # 体坐标系右移 (+X)

# 所有导航响应类型集合
_NAV_RESP_TYPES = frozenset({
    TYPE_CMD_GOTO_RESP, TYPE_CMD_TOX_RESP, TYPE_CMD_TOY_RESP,
    TYPE_CMD_TURNTO_RESP, TYPE_CMD_FINE_RESP, TYPE_CMD_SYNC_RESP,
    TYPE_CMD_ARC_RESP, TYPE_CMD_PATH_RESP, TYPE_RUN_RESP,
    TYPE_ROTATE_RESP, TYPE_ARM_RESP, TYPE_LIGHT_RESP,
    TYPE_CMD_VISION_NUDGE_RESP,
    TYPE_CMD_SET_ZERO_RESP,
    TYPE_CMD_VISION_CORRECT_RESP,
    TYPE_CMD_IMU_CALIB_RESP,
    TYPE_CMD_ARC_ROTATE_RESP,
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
        self._resp_history = deque(maxlen=32)  # (seq, type, status), 供并发命令事后确认
        # 位姿 (下位机50Hz常驻上报, 存最新一帧供随时读取)
        self._pose: tuple[float, float, float] | None = None   # (x, y, yaw_deg)
        self._pose_ts: float = 0.0                               # monotonic 时间戳
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

    def send_rotate(self, pos: int) -> int:
        if not self._ser:
            raise RuntimeError("serial not started")
        if not (0 <= pos <= 4):
            raise ValueError("rotate pos must be 0-4")
        seen = self._mark_send()
        self._ser.write(pack_frame(TYPE_ROTATE, struct.pack("<B", pos)))
        return seen

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

    def send_set_zero(self) -> None:
        """发送转盘零点设置命令 (payload 空, MCU 调 Emm_V5_Origin_Set_O 写 flash)."""
        if not self._ser:
            raise RuntimeError("serial not started")
        self._send_nav(TYPE_CMD_SET_ZERO, b"")

    def send_imu_calib(self) -> None:
        """Send IMU zero-bias calibration command. Keep the robot fully stationary."""
        if not self._ser:
            raise RuntimeError("serial not started")
        self._send_nav(TYPE_CMD_IMU_CALIB, b"")

    def send_light(self, light_id: int, on: bool) -> None:
        if not self._ser:
            raise RuntimeError("serial not started")
        if not (0 <= light_id <= 4):
            raise ValueError("light id must be 0-4 (0=all, 1-4=single)")
        self._mark_send()
        self._ser.write(pack_frame(TYPE_LIGHT, struct.pack("<BB", light_id, 1 if on else 0)))

    # --- 导航命令发送 ---

    def _mark_send(self) -> int:
        """记录发送时刻的 seq 基线 (必须在 write 之前调用)."""
        with self._resp_cond:
            self._resp_sent_seq = self._nav_resp_seq
            return self._resp_sent_seq

    def _send_nav(self, msg_type: int, payload: bytes) -> int:
        """发送导航命令. (响应由 seq 机制区分, 无需在此清除.)"""
        if not self._ser:
            raise RuntimeError("serial not started")
        seen = self._mark_send()
        self._ser.write(pack_frame(msg_type, payload))
        return seen

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

    def send_arc(self, radius: float, dir: int, sweep_deg: float,
                 speed: float | None = None) -> None:
        """发送圆弧命令 (新语义: 半径+方向+扫过角度, 圆心由 MCU 按当前位姿自动算).

        radius: 圆弧半径 m, 必须 > 0
        dir:    +1=右转(顺时针), -1=左转(逆时针), 0 非法
        sweep_deg: 圆弧扫过角度(度), 符号忽略, 转向由 dir 决定
        speed:  可选线速度 m/s, 缺省时 MCU 用默认速度 0.10
        payload: <fff>(12B) 或 <ffff>(16B, 含速度)
        """
        self._check_finite(radius, sweep_deg)
        if radius <= 0:
            raise ValueError(f"arc radius must be > 0, got {radius}")
        if dir == 0:
            raise ValueError("arc dir must be +1 (right turn) or -1 (left turn)")
        d = 1.0 if dir > 0 else -1.0
        sweep = abs(float(sweep_deg))
        if speed is None:
            self._send_nav(TYPE_CMD_ARC, struct.pack("<fff", radius, d, sweep))
        else:
            self._check_finite(speed)
            self._send_nav(TYPE_CMD_ARC,
                           struct.pack("<ffff", radius, d, sweep, speed))

    def send_arc_rotate(self, radius: float, dir: int, sweep_deg: float,
                        speed: float,
                        triggers: list[tuple[float, int]]) -> None:
        """发送圆弧+转盘触发命令。

        triggers 固定 3 组: (触发角度deg, 转盘槽位)。触发角度按下位机实际圆弧进度判断。
        payload: <fffffBfBfB> = radius,dir,sweep,speed,trig1,slot1,trig2,slot2,trig3,slot3
        """
        self._check_finite(radius, sweep_deg, speed)
        if radius <= 0:
            raise ValueError(f"arc radius must be > 0, got {radius}")
        if dir == 0:
            raise ValueError("arc dir must be +1 (right turn) or -1 (left turn)")
        if speed <= 0:
            raise ValueError(f"arc speed must be > 0, got {speed}")
        if len(triggers) != 3:
            raise ValueError("arc_rotate requires exactly 3 triggers")
        d = 1.0 if dir > 0 else -1.0
        sweep = abs(float(sweep_deg))
        vals = []
        for deg, slot in triggers:
            self._check_finite(float(deg))
            if not (0 <= int(slot) <= 4):
                raise ValueError(f"rotate slot must be 0-4, got {slot}")
            vals.append((float(deg), int(slot)))
        self._send_nav(
            TYPE_CMD_ARC_ROTATE,
            struct.pack("<fffffBfBfB", radius, d, sweep, speed,
                        vals[0][0], vals[0][1], vals[1][0], vals[1][1], vals[2][0], vals[2][1]),
        )

    def send_fine_move(self, dx_mm: float, dy_mm: float) -> None:
        self._check_finite(dx_mm, dy_mm)
        self._send_nav(TYPE_CMD_FINE_MOVE, struct.pack("<ff", dx_mm, dy_mm))

    def send_sync_pose(self, x: float, y: float, yaw_deg: float | None = None) -> None:
        """同步下位机里程计坐标。

        旧协议 8B payload 只同步 x/y，并保留下位机当前 yaw。
        新协议 12B payload 同步 x/y/yaw，用于重置当前朝向。
        """
        if yaw_deg is None:
            self._check_finite(x, y)
            self._send_nav(TYPE_CMD_SYNC_POSE, struct.pack("<ff", x, y))
        else:
            self._check_finite(x, y, yaw_deg)
            self._send_nav(TYPE_CMD_SYNC_POSE, struct.pack("<fff", x, y, yaw_deg))

    def send_path_begin(self, speed: float, count: int) -> None:
        """开始装载连续路径。count 是下位机将收到的总点数。"""
        self._check_finite(speed)
        if not (2 <= int(count) <= 255):
            raise ValueError(f"path count must be 2..255, got {count}")
        if speed <= 0.0:
            raise ValueError(f"path speed must be > 0, got {speed}")
        self._send_nav(TYPE_CMD_PATH_BEGIN, struct.pack("<fB", float(speed), int(count)))

    def send_path_point(self, x: float, y: float,
                        target_theta: float = 0.0,
                        mode: int = PATH_MODE_NORMAL) -> None:
        """追加一个路径点。mode=0普通通过点, mode=1关键点。"""
        self._check_finite(x, y, target_theta)
        if mode not in (PATH_MODE_NORMAL, PATH_MODE_KEY):
            raise ValueError(f"path mode must be 0 or 1, got {mode}")
        self._send_nav(TYPE_CMD_PATH_POINT,
                       struct.pack("<fffBxxx", float(x), float(y),
                                   float(target_theta), int(mode)))

    def send_path_exec(self) -> int:
        """触发执行已装载的连续路径。"""
        return self._send_nav(TYPE_CMD_PATH_EXEC, b"")

    def send_vision_nudge(self, direction: int) -> None:
        """发送视觉微调命令 (体坐标系方向, 1B payload).

        direction: NUDGE_STOP/FORWARD/BACKWARD/LEFT/RIGHT
                   0=停止+锁死, 1-4=慢速运动 (MCU 固定速度, 无需指定)
        """
        if direction not in (NUDGE_STOP, NUDGE_FORWARD, NUDGE_BACKWARD,
                             NUDGE_LEFT, NUDGE_RIGHT):
            raise ValueError(f"nudge direction must be 0-4, got {direction}")
        self._send_nav(TYPE_CMD_VISION_NUDGE, struct.pack("<B", direction))

    def send_vision_correct(self, dx_mm: float, dy_mm: float,
                            target_x: float, target_y: float) -> None:
        """发送视觉校正命令 (fine_move + sync_pose 原子组合, 16B payload).

        dx_mm/dy_mm: 场坐标修正量(mm), +X=右, +Y=前; MCU 据此物理修正位置.
        target_x/target_y: 修正成功后 MCU 把 odom 重置到此绝对坐标(米).
        """
        self._check_finite(dx_mm, dy_mm, target_x, target_y)
        self._send_nav(TYPE_CMD_VISION_CORRECT,
                       struct.pack("<ffff", dx_mm, dy_mm, target_x, target_y))

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
                    log.debug("skip stale resp type=0x%02x (expect 0x%02x)",
                              self._nav_resp_type, expect)
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return False
                self._resp_cond.wait(remain)

    # --- 高层运动命令 ("选择形式", 阻塞式, 返回成败) ---
    # 与下位机一一对应: 锁轴=TOX/TOY(1数据), 走点=GOTO(2数据)或GOTO+TURNTO(3数据),
    # 圆弧=ARC(3数据: 半径/方向/扫角, 可选第4个速度). 坐标单位米, yaw 单位度(CW+).
    # 每条命令 MCU 阻塞执行后回 status.

    def response_seq(self) -> int:
        """返回当前响应 seq, 用于并发命令的事后确认。"""
        with self._resp_cond:
            return self._nav_resp_seq

    def wait_for_after(self, expect: int, seen_seq: int, timeout: float) -> bool:
        """等待 seen_seq 之后出现过的指定响应。

        这个接口查询响应历史队列，适合“先发一条不阻塞命令，
        同时执行另一条阻塞命令，最后再确认第一条响应”的场景。
        """
        deadline = time.monotonic() + timeout
        seen = int(seen_seq)
        with self._resp_cond:
            while True:
                max_seen = seen
                for seq, msg_type, status in self._resp_history:
                    if seq <= seen:
                        continue
                    max_seen = max(max_seen, seq)
                    if msg_type == expect:
                        return status == 1
                seen = max_seen

                remain = deadline - time.monotonic()
                if remain <= 0:
                    return False
                self._resp_cond.wait(remain)

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

    def arc(self, radius: float, dir: int, sweep_deg: float,
            speed: float | None = None, timeout: float | None = None) -> bool:
        """圆弧 (新语义, 从当前位姿出发, 圆心由 MCU 自动算).

        radius: 半径 m; dir: +1=右转 / -1=左转; sweep_deg: 扫过角度(度).
        speed: 可选线速度 m/s; None 时使用下位机 MOVE_ARC_SPEED. 车头始终沿切线朝前, 不会原地打转.
        例: 机器人在 (0,0) 朝向0, arc(0.5, +1, 180) → 右转半圆到 (1,0), 朝向180.
        timeout=None 时按弧长自动估算 (2.5倍裕量 + 15s 基底).

        Returns:
            True = 圆弧走完; 超时/失败返回 False.
        """
        if timeout is None:
            v = speed if speed is not None else 0.30
            arclen = abs(sweep_deg) * math.pi / 180.0 * radius
            timeout = arclen / v * 2.5 + 15.0
        self.send_arc(radius, dir, sweep_deg, speed)
        return self.wait_for(TYPE_CMD_ARC_RESP, timeout)

    def arc_rotate(self, radius: float, dir: int, sweep_deg: float,
                   speed: float, triggers: list[tuple[float, int]],
                   timeout: float | None = None) -> bool:
        """圆弧运动，并由下位机按实际弧进度触发转盘。"""
        if timeout is None:
            arclen = abs(sweep_deg) * math.pi / 180.0 * radius
            timeout = arclen / max(speed, 0.01) * 2.5 + 15.0
        self.send_arc_rotate(radius, dir, sweep_deg, speed, triggers)
        return self.wait_for(TYPE_CMD_ARC_ROTATE_RESP, timeout)

    @staticmethod
    def _normalize_path_point(point) -> tuple[float, float, float, int]:
        """把 (x,y)、(x,y,yaw) 或 (x,y,yaw,mode) 统一成路径点。"""
        if len(point) == 2:
            x, y = point
            return float(x), float(y), 0.0, PATH_MODE_NORMAL
        if len(point) == 3:
            x, y, yaw = point
            return float(x), float(y), float(yaw), PATH_MODE_NORMAL
        if len(point) == 4:
            x, y, yaw, mode = point
            return float(x), float(y), float(yaw), int(mode)
        raise ValueError("path point must be (x,y), (x,y,yaw), or (x,y,yaw,mode)")

    def path(self, points, speed: float = 0.30,
             timeout: float | None = None,
             prepend_current: bool = True,
             final_yaw: float | None = None) -> bool:
        """连续路径跟踪。普通点不停, 关键点按下位机策略提前转向并减速通过。

        points: 目标点序列, 每点可为 (x,y)、(x,y,yaw) 或 (x,y,yaw,mode)。
        prepend_current=True 时自动把最新位姿作为 p[0], 避免下位机把第一个目标点当起点。
        final_yaw 不为 None 时, 最后一个点强制改为 PATH_MODE_KEY。
        """
        self._check_finite(speed)
        if speed <= 0.0:
            raise ValueError(f"path speed must be > 0, got {speed}")

        pts = [self._normalize_path_point(p) for p in points]
        if not pts:
            raise ValueError("path requires at least one target point")

        if final_yaw is not None:
            self._check_finite(final_yaw)
            x, y, _, _ = pts[-1]
            pts[-1] = (x, y, float(final_yaw), PATH_MODE_KEY)

        if prepend_current:
            pose = self.get_pose(max_age=1.0)
            if pose is None:
                raise RuntimeError("no fresh pose; cannot prepend current path start")
            px, py, pyaw = pose
            pts.insert(0, (px, py, pyaw, PATH_MODE_NORMAL))

        if len(pts) < 2:
            raise ValueError("path needs at least two points after start insertion")
        if len(pts) > 255:
            raise ValueError(f"path supports at most 255 points, got {len(pts)}")

        total_len = 0.0
        for (x0, y0, _, _), (x1, y1, _, _) in zip(pts, pts[1:]):
            total_len += math.hypot(x1 - x0, y1 - y0)
        if timeout is None:
            timeout = max(15.0, total_len / speed * 3.0 + 5.0)

        self.send_path_begin(speed, len(pts))
        for x, y, yaw, mode in pts:
            self.send_path_point(x, y, yaw, mode)
        self.send_path_exec()
        return self.wait_for(TYPE_CMD_PATH_RESP, timeout)

    def key_path(self, points, speed: float = 0.30,
                 timeout: float | None = None,
                 prepend_current: bool = False) -> bool:
        """关键点连续路径。

        每个点必须带 yaw: (x, y, yaw) 或 (x, y, yaw, mode)。本函数会把所有点
        强制作为 PATH_MODE_KEY 发送; 重复坐标点由下位机解释为原地转角点。
        默认不插入当前位姿, 适合上位机已经给出完整起点的固定轨迹。
        """
        key_pts = []
        for point in points:
            if len(point) not in (3, 4):
                raise ValueError("key_path point must be (x,y,yaw) or (x,y,yaw,mode)")
            x, y, yaw = point[:3]
            key_pts.append((float(x), float(y), float(yaw), PATH_MODE_KEY))
        if timeout is None:
            total_len = 0.0
            zero_turns = 0
            for (x0, y0, _, _), (x1, y1, _, _) in zip(key_pts, key_pts[1:]):
                seg_len = math.hypot(x1 - x0, y1 - y0)
                if seg_len < 1e-4:
                    zero_turns += 1
                total_len += seg_len
            timeout = max(30.0, total_len / speed * 6.0 + zero_turns * 8.0 + 15.0)
        return self.path(key_pts, speed=speed, timeout=timeout,
                         prepend_current=prepend_current, final_yaw=None)

    def vision_nudge(self, direction: int, timeout: float = 3.0) -> bool:
        """视觉微调 (体坐标系方向, 非阻塞运动).

        direction: NUDGE_STOP/FORWARD/BACKWARD/LEFT/RIGHT
                   0=停止+锁死, 1-4=慢速运动 (MCU 固定速度 0.05 m/s)

        MCU 收到 dir=1-4 后立即设慢速并回响应 (不阻塞), Emm_V5 维持速度
        直到收到 dir=0 停止. 上位机视觉闭环: 发方向→看画面→发停止.
        MotorTask 2s 看门狗: 无新命令自动停止+锁死.

        Returns:
            True = MCU 已执行 (收到 RESP); 超时返回 False.
        """
        self.send_vision_nudge(direction)
        return self.wait_for(TYPE_CMD_VISION_NUDGE_RESP, timeout)

    def vision_correct(self, dx_mm: float, dy_mm: float,
                       target_x: float, target_y: float,
                       timeout: float = 5.0) -> bool:
        """视觉校正 (弧后视觉闭环, fine_move + sync_pose 原子组合, 阻塞式).

        dx_mm/dy_mm: 场坐标修正量(mm), +X=右, +Y=前; MCU 物理修正此偏移.
        target_x/target_y: 修正成功后 odom 重置到此坐标(米), 消除累积漂移.
        MCU 先 MoveToAccurateTimed 走完偏移, 成功后 Move_InitPose 重置坐标.

        Returns:
            True = 修正成功且坐标已重置; 超时/fine_move 失败返回 False.
        """
        self.send_vision_correct(dx_mm, dy_mm, target_x, target_y)
        return self.wait_for(TYPE_CMD_VISION_CORRECT_RESP, timeout)

    def run(self, timeout: float = 5.0) -> bool:
        """启动命令 (模拟按下启动按键).

        下位机收到后 PD15 拉高 500ms 再释放 (默认低电平=未开始),
        执行完回 TYPE_RUN_RESP status=1. 阻塞等待响应.

        Returns:
            True = 启动脉冲已执行; 超时/异常响应返回 False.
        """
        self.send_run()
        return self.wait_for(TYPE_RUN_RESP, timeout)

    def set_zero(self, timeout: float = 5.0) -> bool:
        """转盘零点设置: 将当前位置存为零点并写入 flash, 阻塞等待 TYPE_CMD_SET_ZERO_RESP.

        使用方法: 先手动把转盘转到 slot 0 位置, 然后调用本函数.
        MCU 收到后调 Emm_V5_Origin_Set_O(0x05, true) 将当前位置写入 flash.
        以后每次上电, Emm_V5_Origin_Trigger_Return 会自动回到此零点.

        Returns:
            True = 零点已保存; 超时/异常返回 False.
        """
        self.send_set_zero()
        return self.wait_for(TYPE_CMD_SET_ZERO_RESP, timeout)

    def imu_calib(self, timeout: float = 12.0) -> bool:
        """IMU zero-bias calibration. Keep the robot stationary until this returns."""
        self.send_imu_calib()
        return self.wait_for(TYPE_CMD_IMU_CALIB_RESP, timeout)

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
                status = payload[0]
                self._nav_resp_type = msg_type
                self._nav_resp_status = status
                self._nav_resp_seq += 1
                self._resp_history.append((self._nav_resp_seq, msg_type, status))
                self._resp_cond.notify_all()
        elif msg_type == TYPE_POSE and len(payload) >= 12:
            x, y, theta_rad = struct.unpack("<fff", payload[:12])
            yaw = math.degrees(theta_rad)
            # 下位机 move_yaw(编码器) 内部是累加值(不解卷), 防御性归一到 (-180, 180],
            # 避免位姿显示出现 540/720 累加 (新固件已在MCU侧归一)
            yaw = (yaw + 180.0) % 360.0 - 180.0
            with self._lock:
                self._pose = (x, y, yaw)
                self._pose_ts = time.monotonic()
        else:
            log.debug("unhandled type 0x%02x len=%d", msg_type, len(payload))

    def get_pose(self, max_age: float = 1.0) -> tuple[float, float, float] | None:
        """读取最新位姿 (下位机50Hz常驻上报, 无需发请求).

        Args:
            max_age: 帧龄上限 s, 超过视为通信中断.
        Returns:
            (x, y, yaw_deg): x/y 米, yaw 度 CW+ (编码器里程计, 与命令约定一致);
            从未收到帧或帧过旧返回 None.
        """
        with self._lock:
            if self._pose is None:
                return None
            if time.monotonic() - self._pose_ts > max_age:
                return None
            return self._pose

    def pose_age(self) -> float:
        """最新位姿帧的年龄 s (从未收到返回 inf)."""
        with self._lock:
            if self._pose is None:
                return float('inf')
            return time.monotonic() - self._pose_ts


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


def get_pose(max_age: float = 1.0) -> tuple[float, float, float] | None:
    """读取最新位姿 (x米, y米, yaw度CW+), 帧过旧/未收到返回 None."""
    if _comm is None:
        return None
    return _comm.get_pose(max_age)


def pose_age() -> float:
    """最新位姿帧年龄 s (未收到=inf)."""
    if _comm is None:
        return float('inf')
    return _comm.pose_age()


def send_velocity(vx: float, vy: float, w: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    _comm.send_velocity(vx, vy, w)


def send_rotate(pos: int) -> int:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.send_rotate(pos)


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

def send_arc(radius: float, dir: int, sweep_deg: float,
             speed: float | None = None) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_arc(radius, dir, sweep_deg, speed)

def send_arc_rotate(radius: float, dir: int, sweep_deg: float,
                    speed: float,
                    triggers: list[tuple[float, int]]) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_arc_rotate(radius, dir, sweep_deg, speed, triggers)

def send_fine_move(dx_mm: float, dy_mm: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_fine_move(dx_mm, dy_mm)

def send_sync_pose(x: float, y: float, yaw_deg: float | None = None) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_sync_pose(x, y, yaw_deg)

def sync_pose(x: float, y: float, yaw_deg: float | None = None,
              timeout: float = 5.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    _comm.send_sync_pose(x, y, yaw_deg)
    return _comm.wait_for(TYPE_CMD_SYNC_RESP, timeout)

def send_path_begin(speed: float, count: int) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_path_begin(speed, count)

def send_path_point(x: float, y: float,
                    target_theta: float = 0.0,
                    mode: int = PATH_MODE_NORMAL) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_path_point(x, y, target_theta, mode)

def send_path_exec() -> int:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    return _comm.send_path_exec()

def send_vision_nudge(direction: int) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_vision_nudge(direction)

def send_vision_correct(dx_mm: float, dy_mm: float,
                        target_x: float, target_y: float) -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized")
    _comm.send_vision_correct(dx_mm, dy_mm, target_x, target_y)

def send_imu_calib() -> None:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    _comm.send_imu_calib()

def wait_nav_response(timeout: float = 30.0) -> tuple[int, int] | None:
    if _comm is None:
        return None
    return _comm.wait_nav_response(timeout)

def wait_for(expect: int, timeout: float = 30.0) -> bool:
    """等待指定类型的响应 (跳过陈旧/错类型), True=status 1."""
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.wait_for(expect, timeout)

def response_seq() -> int:
    """返回当前响应 seq, 用于并发命令的事后确认。"""
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.response_seq()

def wait_for_after(expect: int, seen_seq: int, timeout: float = 30.0) -> bool:
    """等待 seen_seq 之后出现过的指定响应。"""
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.wait_for_after(expect, seen_seq, timeout)


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

def arc(radius: float, dir: int, sweep_deg: float,
        speed: float | None = None, timeout: float | None = None) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.arc(radius, dir, sweep_deg, speed, timeout)

def arc_rotate(radius: float, dir: int, sweep_deg: float,
               speed: float, triggers: list[tuple[float, int]],
               timeout: float | None = None) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.arc_rotate(radius, dir, sweep_deg, speed, triggers, timeout)

def path(points, speed: float = 0.30,
         timeout: float | None = None,
         prepend_current: bool = True,
         final_yaw: float | None = None) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.path(points, speed, timeout, prepend_current, final_yaw)

def key_path(points, speed: float = 0.30,
             timeout: float | None = None,
             prepend_current: bool = False) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.key_path(points, speed, timeout, prepend_current)

def run(timeout: float = 5.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.run(timeout)

def set_zero(timeout: float = 5.0) -> bool:
    """转盘零点设置 (阻塞式): 将当前位置存为零点并写入 flash.

    先手动把转盘转到 slot 0, 再调用本函数.
    """
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.set_zero(timeout)

def imu_calib(timeout: float = 12.0) -> bool:
    """IMU 零偏校准: 车必须静止，等待下位机完成并确认 IMU 帧恢复。"""
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.imu_calib(timeout)

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

def vision_nudge(direction: int, timeout: float = 3.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.vision_nudge(direction, timeout)

def vision_correct(dx_mm: float, dy_mm: float,
                   target_x: float, target_y: float,
                   timeout: float = 5.0) -> bool:
    if _comm is None:
        raise RuntimeError("serial not initialized, call init() first")
    return _comm.vision_correct(dx_mm, dy_mm, target_x, target_y, timeout)


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

    # 4.5 POSE 帧: 不进响应计数, 位姿被正确缓存 (theta 弧度→度)
    assert c.get_pose() is None, "收帧前 get_pose 应为 None"
    seq2 = c._nav_resp_seq
    for b in pack_frame(TYPE_POSE, struct.pack("<fff", 0.5, -0.25, 1.5707963)):
        c._feed_byte(b)
    assert c._nav_resp_seq == seq2, "POSE 不得计入导航响应"
    pose = c.get_pose()
    assert pose is not None
    assert abs(pose[0] - 0.5) < 1e-6 and abs(pose[1] + 0.25) < 1e-6
    assert abs(pose[2] - 90.0) < 0.01, f"theta 应转为 90°, got {pose[2]}"
    assert c.pose_age() < 0.5

    # 4.5b POSE 角度归一: 下位机上报累加角 (540°/720°) 必须卷绕到 (-180,180]
    for b in pack_frame(TYPE_POSE, struct.pack("<fff", 0.0, 0.0, math.radians(540.0))):
        c._feed_byte(b)
    pose = c.get_pose()
    assert pose is not None
    assert abs(abs(pose[2]) - 180.0) < 0.01, f"540° 应卷绕为 ±180°, got {pose[2]}"
    for b in pack_frame(TYPE_POSE, struct.pack("<fff", 0.0, 0.0, math.radians(720.0))):
        c._feed_byte(b)
    pose = c.get_pose()
    assert abs(pose[2] - 0.0) < 0.01, f"720° 应卷绕为 0°, got {pose[2]}"
    for b in pack_frame(TYPE_POSE, struct.pack("<fff", 0.0, 0.0, math.radians(-450.0))):
        c._feed_byte(b)
    pose = c.get_pose()
    assert abs(pose[2] - (-90.0)) < 0.01, f"-450° 应卷绕为 -90°, got {pose[2]}"

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
            TYPE_CMD_PATH_EXEC: TYPE_CMD_PATH_RESP,
            TYPE_RUN:         TYPE_RUN_RESP,
            TYPE_ROTATE:      TYPE_ROTATE_RESP,
            TYPE_ARM:         TYPE_ARM_RESP,
            TYPE_LIGHT:       TYPE_LIGHT_RESP,
            TYPE_CMD_VISION_NUDGE: TYPE_CMD_VISION_NUDGE_RESP,
            TYPE_CMD_VISION_CORRECT: TYPE_CMD_VISION_CORRECT_RESP,
            TYPE_CMD_IMU_CALIB: TYPE_CMD_IMU_CALIB_RESP,
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

    # 6.7 arc → 一帧 ARC, 3 个 float (半径/方向/扫角); 带速度时 4 个 float
    assert c.arc(0.3, 1, 90.0)
    assert c._ser.written[-1] == pack_frame(
        TYPE_CMD_ARC, struct.pack("<fff", 0.3, 1.0, 90.0))
    assert c.arc(0.3, -1, 180.0, speed=0.08)
    assert c._ser.written[-1] == pack_frame(
        TYPE_CMD_ARC, struct.pack("<ffff", 0.3, -1.0, 180.0, 0.08))

    # 6.7.1 连续路径底层帧: BEGIN 5B, POINT 16B, EXEC 空payload
    c.send_path_begin(0.3, 3)
    assert c._ser.written[-1] == pack_frame(TYPE_CMD_PATH_BEGIN, struct.pack("<fB", 0.3, 3))
    c.send_path_point(0.1, 0.2, 90.0, PATH_MODE_KEY)
    assert c._ser.written[-1] == pack_frame(
        TYPE_CMD_PATH_POINT, struct.pack("<fffBxxx", 0.1, 0.2, 90.0, PATH_MODE_KEY))
    c.send_path_exec()
    assert c._ser.written[-1] == pack_frame(TYPE_CMD_PATH_EXEC, b"")

    # 6.7.2 高层 path: 自动插入当前位姿作为起点, final_yaw 强制终点关键点
    n = len(c._ser.written)
    assert c.path([(0.2, 0.0), (0.4, 0.0)], speed=0.3, timeout=0.1, final_yaw=90.0)
    assert len(c._ser.written) == n + 5
    assert c._ser.written[n] == pack_frame(TYPE_CMD_PATH_BEGIN, struct.pack("<fB", 0.3, 3))
    assert c._ser.written[n + 1][2] == TYPE_CMD_PATH_POINT
    assert c._ser.written[n + 3] == pack_frame(
        TYPE_CMD_PATH_POINT, struct.pack("<fffBxxx", 0.4, 0.0, 90.0, PATH_MODE_KEY))
    assert c._ser.written[n + 4] == pack_frame(TYPE_CMD_PATH_EXEC, b"")

    # 6.7.3 key_path: 不自动插入当前位姿, 每个点都作为关键点发送
    n = len(c._ser.written)
    assert c.key_path([(0.0, 0.0, -90.0), (0.2, 0.1, -55.0)], speed=0.3, timeout=0.1)
    assert len(c._ser.written) == n + 4
    assert c._ser.written[n] == pack_frame(TYPE_CMD_PATH_BEGIN, struct.pack("<fB", 0.3, 2))
    assert c._ser.written[n + 1] == pack_frame(
        TYPE_CMD_PATH_POINT, struct.pack("<fffBxxx", 0.0, 0.0, -90.0, PATH_MODE_KEY))
    assert c._ser.written[n + 2] == pack_frame(
        TYPE_CMD_PATH_POINT, struct.pack("<fffBxxx", 0.2, 0.1, -55.0, PATH_MODE_KEY))
    assert c._ser.written[n + 3] == pack_frame(TYPE_CMD_PATH_EXEC, b"")

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
        c.send_arc(0.0, 1, 90.0)
        assert False, "arc r<=0 must raise"
    except ValueError:
        pass
    try:
        c.send_arc(0.3, 0, 90.0)
        assert False, "arc dir=0 must raise"
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

    # 6.18 send_vision_nudge: 1B payload 方向码, 0-4 合法, 越界抛 ValueError
    c._ser = _EchoSerial(c)   # 重置为干净 Echo
    for d in (NUDGE_STOP, NUDGE_FORWARD, NUDGE_BACKWARD, NUDGE_LEFT, NUDGE_RIGHT):
        c.send_vision_nudge(d)
        assert c._ser.written[-1] == pack_frame(TYPE_CMD_VISION_NUDGE, struct.pack("<B", d))
    for badv in (-1, 5, 255):
        try:
            c.send_vision_nudge(badv)
            assert False, "nudge direction out-of-range must raise"
        except ValueError:
            pass

    # 6.19 高层 vision_nudge: 帧 + VISION_NUDGE_RESP 确认 (非阻塞, 即时应答)
    assert c.vision_nudge(NUDGE_FORWARD)
    assert c._ser.written[-1] == pack_frame(TYPE_CMD_VISION_NUDGE, struct.pack("<B", NUDGE_FORWARD))
    assert c.vision_nudge(NUDGE_STOP)
    assert c._ser.written[-1] == pack_frame(TYPE_CMD_VISION_NUDGE, struct.pack("<B", NUDGE_STOP))

    # 6.20 高层 vision_correct: 16B payload (4×f32: dx_mm/dy_mm/target_x/target_y) + RESP 确认
    assert c.vision_correct(30.0, -15.0, 0.5, 0.3)
    assert c._ser.written[-1] == pack_frame(
        TYPE_CMD_VISION_CORRECT, struct.pack("<ffff", 30.0, -15.0, 0.5, 0.3))
    # status=0 → fine_move 失败, 不重置坐标
    c._ser.status = 0
    assert not c.vision_correct(10.0, 10.0, 0.0, 0.0)
    c._ser.status = 1
    # NaN/inf 防护
    for badv in (float('nan'), float('inf')):
        try:
            c.send_vision_correct(badv, 0.0, 0.0, 0.0)
            assert False, "NaN/inf must raise"
        except ValueError:
            pass

    # 6.21 IMU 零偏校准: 空 payload + IMU_CALIB_RESP 确认
    assert c.imu_calib(timeout=0.1)
    assert c._ser.written[-1] == pack_frame(TYPE_CMD_IMU_CALIB, b"")

    print("self-check OK")


if __name__ == "__main__":
    _self_check()
