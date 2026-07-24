"""比赛路径定义 — 生成 MCU 导航命令序列.

坐标系:
  - path 内部: mm, field 坐标 (0°=上/+Y, 顺时针)
  - MCU 输出:  m, Blu3场坐标 (+X=右, +Y=前, CW正)
  - 映射: MCU_x = field_x/1000, MCU_y = field_y/1000 (方向一致)

直线段自动选择 TOX 或 TOY (轴锁定模式, 防麦轮漂移).
圆弧段计算 MCU MoveArc 参数 (圆心m, 半径m, 起止角度deg).
"""

import math


# ============ 坐标转换 ============

def field_to_mcu(field_x_mm, field_y_mm):
    """field mm → MCU m.  Returns (mcu_x, mcu_y)."""
    return (field_x_mm / 1000.0, field_y_mm / 1000.0)


def field_to_mcu_single(field_x_mm):
    """field X mm (横) → MCU X m (右+)."""
    return field_x_mm / 1000.0


def field_to_mcu_forward(field_y_mm):
    """field Y mm (前) → MCU Y m (前+)."""
    return field_y_mm / 1000.0


# ============ 角度转换 ============

def u(user_deg):
    """用户角度 → atan2 弧度."""
    return math.radians((90 - user_deg) % 360)


def atan2_to_user(atan2_deg):
    return (90 - atan2_deg) % 360


def point_user_angle(dx, dy):
    """点(dx, dy)相对原点的用户角度."""
    return math.degrees(math.atan2(dx, dy)) % 360


# ============ 路径段定义 (mm, field坐标) ============

START = (0, 0, 0)

# 段1-3
SEG1_END = (0, 295, 0)
SEG2_END = (-661.58, 295.68, -90)
SEG3_END = (-661.58, 129.96, -90)

# 段4: 圆弧1 (顺时针 CW)
ARC1_CENTER = (-575, 1055)
ARC1_RADIUS = 949
ARC1_START_PT = (-661.58, 129.96)
ARC1_END_PT = (-632.03, 2022.31)

# 段5-11
SEG5_END = (-632.03, 1455.14, 180)
SEG6_END = (220.38, 1455.64, 180)
SEG7_END = (-427.37, 715.56, 0)
SEG8_END = (334.66, 714.64, 0)
SEG9_END = (836.21, 715.18, 0)
SEG10_END = (636.79, 295.68, 90)
SEG11_END = (636.79, 129.96, 90)

# 段12: 圆弧2 (逆时针 CCW)
ARC2_CENTER = (575, 1055)
ARC2_RADIUS = 949
ARC2_START_PT = (636.79, 129.96)
ARC2_END_PT = (575, 2024)

# 段13-16
SEG13_END = (270.17, 1779.38, 0)
SEG14_END = (-270.17, 1779.38, 0)
SEG15_END = (0, 1779.38, 0)
SEG16_END = (0, 0, 0)


# ============ 命令生成辅助 ============

def _straight_cmd(from_pt, to_pt, label=""):
    """根据方向自动选择 TOX/TOY.

    from_pt/to_pt: (field_x, field_y, user_angle)  in mm.
    Returns: nav command dict.
    """
    dx = abs(to_pt[0] - from_pt[0])
    dy = abs(to_pt[1] - from_pt[1])

    if dx >= dy:
        # X变化大 (横/右) → TOX (MCU X=右)
        target = field_to_mcu_single(to_pt[0])
        return {"cmd": "tox", "x": target, "label": label}
    else:
        # Y变化大 (前) → TOY (MCU Y=前)
        target = field_to_mcu_forward(to_pt[1])
        return {"cmd": "toy", "y": target, "label": label}


def _arc_mcu_params(center, start_pt, end_pt):
    """计算 MCU MoveArc 参数.

    Returns: (cx_m, cy_m, r_m, start_deg, end_deg)
    start/end_deg: CW正 (MoveArc边界取反→内部CCW+)
    """
    cx, cy = center
    # MCU 圆心: X=右=field_x, Y=前=field_y (直接映射)
    mcu_cx = cx / 1000.0
    mcu_cy = cy / 1000.0
    # MCU 半径
    mcu_r = math.hypot(start_pt[0] - cx, start_pt[1] - cy) / 1000.0
    # MCU 起止角 (CW正, MoveArc内部取反→CCW+)
    # 内部CCW+从右 = atan2(前差, 右差) = atan2(y_diff, x_diff)
    sx_diff = start_pt[0] - cx
    sy_diff = start_pt[1] - cy
    ex_diff = end_pt[0] - cx
    ey_diff = end_pt[1] - cy

    start_deg = math.degrees(math.atan2(sy_diff, sx_diff))
    end_deg = math.degrees(math.atan2(ey_diff, ex_diff))

    return mcu_cx, mcu_cy, mcu_r, start_deg, end_deg


def calc_arc(cx, cy, p_start, p_end):
    """计算圆弧参数 (保留供外部调用).

    Returns: (atan2_start_rad, atan2_end_rad, direction, end_theta_atan2_rad, span_deg)
    """
    a_user_start = point_user_angle(p_start[0] - cx, p_start[1] - cy)
    a_user_end = point_user_angle(p_end[0] - cx, p_end[1] - cy)

    diff_cw = (a_user_end - a_user_start) % 360

    if diff_cw > 180:
        direction = "CCW"
        span = 360 - diff_cw
    else:
        direction = "CW"
        span = diff_cw

    a_atan2_start_rad = u(a_user_start)
    a_atan2_end_rad = u(a_user_end)

    if direction == "CW":
        end_theta = a_atan2_end_rad - math.pi / 2
    else:
        end_theta = a_atan2_end_rad + math.pi / 2

    return a_atan2_start_rad, a_atan2_end_rad, direction, end_theta, span


# ============ 主函数: 生成命令序列 ============

def build_commands():
    """生成完整比赛路径的导航命令序列.

    Returns: list[dict], 每个 dict 是一条 nav command.
    """
    cmds = []

    # 段1: (0,0) → (0,295)  Y变化大(前) → TOY
    cmds.append(_straight_cmd(START, SEG1_END, "1: HOME→(0,295)"))

    # 段2: → (-661.58, 295.68)  X变化大(横) → TOX
    cmds.append(_straight_cmd(SEG1_END, SEG2_END, "2: →(-662,296)"))

    # 段3: → (-661.58, 129.96)  Y变化大(前) → TOY
    cmds.append(_straight_cmd(SEG2_END, SEG3_END, "3: →(-662,130)"))

    # 段4: 圆弧1 (CW)
    cx, cy, r, a0, a1 = _arc_mcu_params(ARC1_CENTER, ARC1_START_PT, ARC1_END_PT)
    cmds.append({
        "cmd": "arc", "cx": cx, "cy": cy, "r": r, "a0": a0, "a1": a1,
        "label": "4: ARC1(CW)→(-632,2022)"
    })

    # 段5: → (-632.03, 1455.14)  Y变化大(前) → TOY
    seg4_end = (ARC1_END_PT[0], ARC1_END_PT[1], 0)
    cmds.append(_straight_cmd(seg4_end, SEG5_END, "5: →(-632,1455)"))

    # 段6: → (220.38, 1455.64)  X变化大(横) → TOX
    cmds.append(_straight_cmd(SEG5_END, SEG6_END, "6: →(220,1456)"))

    # 段7: → (-427.37, 715.56)  X和Y都变 → 选大的
    cmds.append(_straight_cmd(SEG6_END, SEG7_END, "7: →(-427,716)"))

    # 段8: → (334.66, 714.64)  X变化大(横) → TOX
    cmds.append(_straight_cmd(SEG7_END, SEG8_END, "8: →(335,715)"))

    # 段9: → (836.21, 715.18)  X变化大(横) → TOX
    cmds.append(_straight_cmd(SEG8_END, SEG9_END, "9: →(836,715)"))

    # 段10: → (636.79, 295.68)  Y变化大(前) → TOY
    cmds.append(_straight_cmd(SEG9_END, SEG10_END, "10: →(637,296)"))

    # 段11: → (636.79, 129.96)  Y变化大(前) → TOY
    cmds.append(_straight_cmd(SEG10_END, SEG11_END, "11: →(637,130)"))

    # 段12: 圆弧2 (CCW)
    cx, cy, r, a0, a1 = _arc_mcu_params(ARC2_CENTER, ARC2_START_PT, ARC2_END_PT)
    cmds.append({
        "cmd": "arc", "cx": cx, "cy": cy, "r": r, "a0": a0, "a1": a1,
        "label": "12: ARC2(CCW)→(575,2024)"
    })

    # 段13: → (270.17, 1779.38)  X变化大(横) → TOX
    seg12_end = (ARC2_END_PT[0], ARC2_END_PT[1], 0)
    cmds.append(_straight_cmd(seg12_end, SEG13_END, "13: →(270,1779)"))

    # 段14: → (-270.17, 1779.38)  X变化大(横) → TOX
    cmds.append(_straight_cmd(SEG13_END, SEG14_END, "14: →(-270,1779)"))

    # 段15: → (0, 1779.38)  X变化大(横) → TOX
    cmds.append(_straight_cmd(SEG14_END, SEG15_END, "15: →(0,1779)"))

    # 段16: → (0, 0)  Y变化大(前) → TOY
    cmds.append(_straight_cmd(SEG15_END, SEG16_END, "16: →HOME(0,0)"))

    return cmds


# ============ 显示/调试 ============

def main():
    print("=" * 60)
    print("比赛路径 — MCU 导航命令序列")
    print("=" * 60)
    print()

    cmds = build_commands()
    print(f"总段数: {len(cmds)}")
    print()

    # 圆弧参数详情
    for arc_name, center, start, end in [
        ("圆弧1 (CW)", ARC1_CENTER, ARC1_START_PT, ARC1_END_PT),
        ("圆弧2 (CCW)", ARC2_CENTER, ARC2_START_PT, ARC2_END_PT),
    ]:
        cx, cy, r, a0, a1 = _arc_mcu_params(center, start, end)
        print(f"--- {arc_name} ---")
        print(f"  圆心 (field mm): {center}")
        print(f"  MCU圆心 (m): ({cx:.4f}, {cy:.4f})")
        print(f"  MCU半径 (m): {r:.4f}")
        print(f"  MCU起始角 (deg): {a0:.2f}")
        print(f"  MCU终止角 (deg): {a1:.2f}")
        print(f"  跨越角 (deg): {a1 - a0:.2f}")
        print()

    # 命令列表
    print("--- 命令序列 ---")
    for i, cmd in enumerate(cmds):
        if cmd["cmd"] == "tox":
            print(f"  [{i+1:2d}] TOX  x={cmd['x']:.4f} m (右)  ({cmd['label']})")
        elif cmd["cmd"] == "toy":
            print(f"  [{i+1:2d}] TOY  y={cmd['y']:.4f} m (前)  ({cmd['label']})")
        elif cmd["cmd"] == "arc":
            print(f"  [{i+1:2d}] ARC  cx={cmd['cx']:.3f} cy={cmd['cy']:.3f}"
                  f" r={cmd['r']:.3f} a0={cmd['a0']:.1f} a1={cmd['a1']:.1f}"
                  f"  ({cmd['label']})")
        else:
            print(f"  [{i+1:2d}] {cmd['cmd']}  {cmd}  ({cmd['label']})")


if __name__ == "__main__":
    main()
