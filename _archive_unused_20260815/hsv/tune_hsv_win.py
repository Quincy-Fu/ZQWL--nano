#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows 本地 HSV 调参神器（双通道版）

用法：
    1. 把 field_sample.jpg 放在同目录下
    2. python tune_hsv_win.py
    3. 调整滑块直到 Mask 窗口中目标物体变纯白、干扰物变纯黑
    4. 如果是对红色调参，勾选"启用第二段"覆盖色环边界
    5. 按 Q 退出，打印可直接复制到 config.py 的代码

H=色相(0-179), S=饱和度(0-255), V=明度(0-255)
"""
import cv2
import numpy as np


def nothing(x):
    pass


# ===== 窗口 =====
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Controls', 600, 500)

# ---- 第一段 HSV ----
cv2.createTrackbar('[1] H_L', 'Controls', 0,   179, nothing)
cv2.createTrackbar('[1] H_H', 'Controls', 10,  179, nothing)
cv2.createTrackbar('[1] S_L', 'Controls', 50,  255, nothing)
cv2.createTrackbar('[1] S_H', 'Controls', 255, 255, nothing)
cv2.createTrackbar('[1] V_L', 'Controls', 50,  255, nothing)
cv2.createTrackbar('[1] V_H', 'Controls', 255, 255, nothing)

# ---- 第二段 HSV（红色专属：跨 0°/180° 边界时启用） ----
cv2.createTrackbar('[2] 启用 (0=关/1=开)', 'Controls', 0, 1, nothing)
cv2.createTrackbar('[2] H_L', 'Controls', 170, 179, nothing)
cv2.createTrackbar('[2] H_H', 'Controls', 179, 179, nothing)
cv2.createTrackbar('[2] S_L', 'Controls', 100, 255, nothing)
cv2.createTrackbar('[2] S_H', 'Controls', 255, 255, nothing)
cv2.createTrackbar('[2] V_L', 'Controls', 100, 255, nothing)
cv2.createTrackbar('[2] V_H', 'Controls', 255, 255, nothing)

# ---- 形态学开关 ----
cv2.createTrackbar('开运算核大小 (0=关)', 'Controls', 5, 15, nothing)

cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

print("=" * 50)
print("  Windows 本地 HSV 双通道调参神器")
print("=" * 50)
print("  将 field_sample.jpg 放在同目录下")
print("  调参目标: Mask 窗口中目标变纯白, 其余变纯黑")
print("  红色需要开启 [2] 段 (跨 HSV 色环边界)")
print("  蓝色/绿色只需 [1] 段即可")
print("  调好按 Q 退出并打印 config.py 代码")
print("=" * 50)

while True:
    frame_bgr = cv2.imread('field_sample.jpg')
    if frame_bgr is None:
        print("错误: 未在当前目录下找到 field_sample.jpg")
        break

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # ---- 读取第一段 ----
    h_l  = cv2.getTrackbarPos('[1] H_L', 'Controls')
    h_h  = cv2.getTrackbarPos('[1] H_H', 'Controls')
    s_l  = cv2.getTrackbarPos('[1] S_L', 'Controls')
    s_h  = cv2.getTrackbarPos('[1] S_H', 'Controls')
    v_l  = cv2.getTrackbarPos('[1] V_L', 'Controls')
    v_h  = cv2.getTrackbarPos('[1] V_H', 'Controls')

    lower1 = np.array([h_l, s_l, v_l])
    upper1 = np.array([h_h, s_h, v_h])
    mask1  = cv2.inRange(hsv, lower1, upper1)

    # ---- 读取第二段 ----
    use2  = cv2.getTrackbarPos('[2] 启用 (0=关/1=开)', 'Controls')
    h2_l  = cv2.getTrackbarPos('[2] H_L', 'Controls')
    h2_h  = cv2.getTrackbarPos('[2] H_H', 'Controls')
    s2_l  = cv2.getTrackbarPos('[2] S_L', 'Controls')
    s2_h  = cv2.getTrackbarPos('[2] S_H', 'Controls')
    v2_l  = cv2.getTrackbarPos('[2] V_L', 'Controls')
    v2_h  = cv2.getTrackbarPos('[2] V_H', 'Controls')

    if use2:
        lower2 = np.array([h2_l, s2_l, v2_l])
        upper2 = np.array([h2_h, s2_h, v2_h])
        mask2  = cv2.inRange(hsv, lower2, upper2)
        mask   = cv2.bitwise_or(mask1, mask2)
    else:
        mask = mask1

    # ---- 形态学开运算 (去除噪点) ----
    ksize = cv2.getTrackbarPos('开运算核大小 (0=关)', 'Controls')
    if ksize > 0:
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Original', frame_bgr)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print()
        print("=" * 55)
        print("  调优完成！将以下代码复制到 config.py：")
        print("=" * 55)
        print(f"LOWER  = np.array([{h_l}, {s_l}, {v_l}])")
        print(f"UPPER  = np.array([{h_h}, {s_h}, {v_h}])")
        if use2:
            print(f"LOWER2 = np.array([{h2_l}, {s2_l}, {v2_l}])")
            print(f"UPPER2 = np.array([{h2_h}, {s2_h}, {v2_h}])")
        print("=" * 55)
        break

cv2.destroyAllWindows()
