"""
STM32 导航测试脚本 (Windows PC)

用法:
    python nav_test.py COM5
    python nav_test.py COM5

命令:
    g x y     GOTO 坐标 (m)       例: g 0.3 0.5
    x <val>   TOX 移到X坐标 (m)   例: x 0.2
    y <val>   TOY 移到Y坐标 (m)   例: y 0.3
    t <deg>   TURNTO 转到角度     例: t 90
    s x y     SYNC_POSE 重置坐标  例: s 0 0
    p         打印当前坐标
    c         连续模式 (预设路径)
    q         退出

坐标系: +X=右, +Y=前, 角度CW正
MCU 上电后 odom 自动归零 (0, 0, 0)
"""

import sys
import struct
import threading
import time
import math

try:
    import serial
except ImportError:
    print("请先安装 pyserial: pip install pyserial")
    sys.exit(1)


# ── 协议常量 ──

HEADER = b'\xAA\x55'
TYPE_CMD_VEL       = 0x01
TYPE_POSE          = 0x02
TYPE_CMD_GOTO      = 0x10
TYPE_CMD_GOTO_RESP = 0x11
TYPE_CMD_TOX       = 0x12
TYPE_CMD_TOX_RESP  = 0x13
TYPE_CMD_TOY       = 0x14
TYPE_CMD_TOY_RESP  = 0x15
TYPE_CMD_TURNTO    = 0x16
TYPE_CMD_TURNTO_RESP = 0x17
TYPE_CMD_FINE_MOVE = 0x18
TYPE_CMD_FINE_RESP = 0x19
TYPE_CMD_SYNC_POSE = 0x1A
TYPE_CMD_SYNC_RESP = 0x1B
TYPE_CMD_ARC       = 0x1C
TYPE_CMD_ARC_RESP  = 0x1D

RESP_NAMES = {
    TYPE_CMD_GOTO_RESP:  "GOTO",
    TYPE_CMD_TOX_RESP:   "TOX",
    TYPE_CMD_TOY_RESP:   "TOY",
    TYPE_CMD_TURNTO_RESP: "TURNTO",
    TYPE_CMD_FINE_RESP:  "FINE",
    TYPE_CMD_SYNC_RESP:  "SYNC",
    TYPE_CMD_ARC_RESP:   "ARC",
}


# ── CRC16-CCITT ──

def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


# ── 组帧 ──

def make_frame(msg_type: int, payload: bytes) -> bytes:
    body = bytes([msg_type, len(payload)]) + payload
    crc = crc16_ccitt(body)
    return HEADER + body + struct.pack("<H", crc)


def pack_goto(x: float, y: float) -> bytes:
    return make_frame(TYPE_CMD_GOTO, struct.pack("<ff", x, y))

def pack_tox(x: float) -> bytes:
    return make_frame(TYPE_CMD_TOX, struct.pack("<f", x))

def pack_toy(y: float) -> bytes:
    return make_frame(TYPE_CMD_TOY, struct.pack("<f", y))

def pack_turnto(deg: float) -> bytes:
    return make_frame(TYPE_CMD_TURNTO, struct.pack("<f", deg))

def pack_sync_pose(x: float, y: float) -> bytes:
    return make_frame(TYPE_CMD_SYNC_POSE, struct.pack("<ff", x, y))


# ── 全局状态 ──

pose_x = 0.0
pose_y = 0.0
pose_yaw = 0.0       # radians (MCU sends radians)
pose_yaw_deg = 0.0
pose_ts = 0.0        # timestamp of last pose
pose_count = 0        # total pose frames received

nav_result_type = 0   # last nav response type
nav_result_status = -1 # -1=no result, 0=fail, 1=success
nav_result_event = threading.Event()


# ── 串口 ──

port = sys.argv[1] if len(sys.argv) > 1 else "COM5"
baudrate = 115200

try:
    ser = serial.Serial(port, baudrate, timeout=0.05)
    print(f"串口已打开: {port} @ {baudrate}")
except serial.SerialException as e:
    print(f"串口打开失败: {e}")
    sys.exit(1)


# ── 接收线程 ──

rx_buf = bytearray()

def rx_thread_func():
    global pose_x, pose_y, pose_yaw, pose_yaw_deg, pose_ts, pose_count
    global nav_result_type, nav_result_status

    while ser.is_open:
        try:
            data = ser.read(128)
            if not data:
                continue
            rx_buf.extend(data)

            while len(rx_buf) >= 6:
                # 找 AA 55 帧头
                idx = -1
                for i in range(len(rx_buf) - 1):
                    if rx_buf[i] == 0xAA and rx_buf[i+1] == 0x55:
                        idx = i
                        break
                if idx < 0:
                    rx_buf.clear()
                    break
                if idx > 0:
                    del rx_buf[:idx]

                if len(rx_buf) < 4:
                    break

                msg_type = rx_buf[2]
                payload_len = rx_buf[3]
                total = 4 + payload_len + 2  # header(2) + type(1) + len(1) + payload + crc(2)

                if len(rx_buf) < total:
                    break

                frame = bytes(rx_buf[:total])
                del rx_buf[:total]

                # CRC 校验
                body = frame[2:-2]  # type + len + payload
                recv_crc = struct.unpack("<H", frame[-2:])[0]
                calc_crc = crc16_ccitt(body)
                if recv_crc != calc_crc:
                    continue  # CRC 错误, 丢弃

                payload = frame[4:-2]

                # POSE (0x02): x, y, theta
                if msg_type == TYPE_POSE and len(payload) >= 12:
                    pose_x, pose_y, pose_yaw = struct.unpack("<fff", payload[:12])
                    pose_yaw_deg = math.degrees(pose_yaw)
                    pose_ts = time.monotonic()
                    pose_count += 1

                # 导航结果 (0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D)
                elif msg_type in RESP_NAMES and len(payload) >= 1:
                    nav_result_type = msg_type
                    nav_result_status = payload[0]
                    nav_result_event.set()

        except Exception as e:
            break

t_rx = threading.Thread(target=rx_thread_func, daemon=True)
t_rx.start()


# ── 工具函数 ──

def send_nav_cmd(frame: bytes, cmd_name: str, desc: str, timeout: float = 30.0) -> bool:
    """发送导航命令并等待完成。优先等 MCU 结果帧，同时监测坐标变化判断运动结束。"""
    nav_result_event.clear()
    ser.write(frame)
    print(f"  >> {cmd_name}: {desc}")

    # 记录起始坐标，用于判断运动是否结束
    start_x, start_y = pose_x, pose_y
    last_move_x, last_move_y = pose_x, pose_y
    last_move_ts = time.monotonic()
    last_print_ts = 0.0
    still_count = 0   # 坐标连续不变的次数
    STILL_THRESHOLD_M = 0.001   # 1mm 视为静止
    STILL_REQUIRED = 5          # 连续 5 次 (2.5s) 不变 → 判定运动结束
    SETTLE_DELAY = 0.8          # 运动结束后再等 0.8s 让坐标稳定

    deadline = time.monotonic() + timeout
    settled = False

    while time.monotonic() < deadline:
        # 优先检查 MCU 是否返回了结果帧
        if nav_result_event.is_set():
            ok = nav_result_status == 1
            status_str = "OK" if ok else "FAIL"
            resp_name = RESP_NAMES.get(nav_result_type, "??")
            time.sleep(0.3)  # 等几帧 POSE 让坐标稳定
            print(f"  << {resp_name} {status_str}  |  x={pose_x:.4f}  y={pose_y:.4f}  yaw={pose_yaw_deg:.1f}°")
            return ok

        # 实时显示坐标 (每 0.5s 刷一次)
        now = time.monotonic()
        if now - last_print_ts >= 0.5:
            elapsed = now - (deadline - timeout)
            dist = math.sqrt((pose_x - start_x)**2 + (pose_y - start_y)**2)
            print(f"  .. {elapsed:5.1f}s  x={pose_x:.4f}  y={pose_y:.4f}  yaw={pose_yaw_deg:.1f}°  d={dist:.4f}m", end="\r")
            last_print_ts = now

        # 检测运动是否停止 (坐标不再变化)
        move_dist = math.sqrt((pose_x - last_move_x)**2 + (pose_y - last_move_y)**2)
        if move_dist > STILL_THRESHOLD_M:
            last_move_x, last_move_y = pose_x, pose_y
            last_move_ts = time.monotonic()
            still_count = 0
        else:
            still_count += 1

        # 如果已检测到静止，等一小段时间让坐标稳定后判定完成
        if still_count >= STILL_REQUIRED and not settled:
            settled = True
            settle_ts = time.monotonic()

        if settled and time.monotonic() - settle_ts >= SETTLE_DELAY:
            total_dist = math.sqrt((pose_x - start_x)**2 + (pose_y - start_y)**2)
            elapsed = time.monotonic() - (deadline - timeout)
            print(f"  << 运动结束 (无结果帧, 坐标静止)  |  x={pose_x:.4f}  y={pose_y:.4f}  yaw={pose_yaw_deg:.1f}°  d={total_dist:.4f}m  t={elapsed:.1f}s")
            return True

        time.sleep(0.1)

    # 超时
    total_dist = math.sqrt((pose_x - start_x)**2 + (pose_y - start_y)**2)
    print(f"  << 超时 ({timeout}s)  |  x={pose_x:.4f}  y={pose_y:.4f}  yaw={pose_yaw_deg:.1f}°  d={total_dist:.4f}m")
    print(f"  !! 提示: MCU 结果帧未收到 (可能是 CommTask 50Hz POSE 抢占 UART6)")
    return False


def print_pose():
    dt = time.monotonic() - pose_ts if pose_ts > 0 else -1
    print(f"  坐标: x={pose_x:.4f}m  y={pose_y:.4f}m  yaw={pose_yaw_deg:.1f}°  "
          f"(pose帧数={pose_count}, 距上次={dt:.2f}s)")


# ── 交互主循环 ──

print()
print("═══  导航测试  ═══")
print("  g x y    GOTO 坐标")
print("  x <val>  移到X坐标 (轴锁定)")
print("  y <val>  移到Y坐标 (轴锁定)")
print("  t <deg>  转到角度 (CW正)")
print("  s x y    重置坐标")
print("  p        打印坐标")
print("  c        连续模式 (预设路径)")
print("  q        退出")
print("═" * 26)
print()

# 等待几帧 POSE 确认通信
print("等待 MCU 通信...", end=" ", flush=True)
wait_start = time.monotonic()
while pose_count < 5 and time.monotonic() - wait_start < 3.0:
    time.sleep(0.1)
if pose_count >= 5:
    print(f"OK (收到 {pose_count} 帧)")
    print_pose()
else:
    print(f"未收到 POSE 帧 (收到 {pose_count} 帧)")
    print("请确认: 1) 串口号正确  2) MCU 已上电  3) CommTask 正在发送 POSE")
print()

try:
    while True:
        try:
            line = input("> ").strip().lower()
        except EOFError:
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        try:
            if cmd == "q":
                break

            elif cmd == "p":
                print_pose()

            elif cmd == "g":
                if len(parts) < 3:
                    print("用法: g <x> <y>  例: g 0.3 0.5")
                    continue
                tx, ty = float(parts[1]), float(parts[2])
                send_nav_cmd(pack_goto(tx, ty), "GOTO", f"({tx}, {ty})")

            elif cmd == "x":
                if len(parts) < 2:
                    print("用法: x <val>  例: x 0.2")
                    continue
                tx = float(parts[1])
                send_nav_cmd(pack_tox(tx), "TOX", f"x={tx}")

            elif cmd == "y":
                if len(parts) < 2:
                    print("用法: y <val>  例: y 0.3")
                    continue
                ty = float(parts[1])
                send_nav_cmd(pack_toy(ty), "TOY", f"y={ty}")

            elif cmd == "t":
                if len(parts) < 2:
                    print("用法: t <deg>  例: t 90")
                    continue
                deg = float(parts[1])
                send_nav_cmd(pack_turnto(deg), "TURNTO", f"{deg}°")

            elif cmd == "s":
                if len(parts) < 3:
                    print("用法: s <x> <y>  例: s 0 0")
                    continue
                sx, sy = float(parts[1]), float(parts[2])
                send_nav_cmd(pack_sync_pose(sx, sy), "SYNC", f"({sx}, {sy})", timeout=5.0)

            elif cmd == "c":
                # 连续模式: 预设路径
                print("连续模式: 前进 0.3m → 右移 0.2m → 转 90° → 回原点")
                print("按 Ctrl+C 中断")
                path = [
                    ("GOTO 0, 0.3", pack_goto(0.0, 0.3)),
                    ("GOTO 0.2, 0.3", pack_goto(0.2, 0.3)),
                    ("TURNTO 90°", pack_turnto(90.0)),
                    ("TURNTO 0°", pack_turnto(0.0)),
                    ("GOTO 0, 0", pack_goto(0.0, 0.0)),
                ]
                for desc, frame in path:
                    if not send_nav_cmd(frame, "NAV", desc, timeout=30.0):
                        print("  !! 命令失败, 停止后续")
                        break
                    time.sleep(0.5)
                print_pose()

            else:
                print(f"未知命令: {cmd}")

        except ValueError:
            print("参数格式错误, 请输入数字")

except KeyboardInterrupt:
    pass

# ── 清理 ──
ser.close()
print("\n串口已关闭")
