"""USB camera based concentric-ring localization.

The innermost circle is a 50 mm reference.  Its fitted ellipse provides a
per-frame local pixel-to-millimetre scale, so camera height changes do not
require a fixed ``px_per_mm`` value.  This module only measures offsets; it
does not send motion commands.

Coordinate layers:
    pixel_offset_px: image coordinates, +u right and +v down
    image_offset_mm: perspective-scaled image coordinates
    body_offset_mm:  image_offset_mm after the configurable camera-to-body map

Keep ``direction_calibrated`` false until the body-axis signs/swaps have been
verified on the real vehicle.  ``RingOffset.body_correction_mm()`` deliberately
refuses to return a command while that flag is false.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass

try:
    import cv2
    import numpy as np
except ImportError:  # Pure geometry helpers remain testable without OpenCV.
    cv2 = None
    np = None


CONFIG = {
    "usb_device": 0,
    "width": 640,
    "height": 480,
    "fps": 30,

    # The detected contour must represent this exact physical circle.
    "inner_circle_diameter_mm": 50.0,
    "black_v_max": 100,
    "search_radius_fraction": 1.0,
    "min_contour_area": 60.0,
    "min_axis_px": 20.0,
    "max_axis_fraction": 0.90,
    "max_axis_ratio": 3.0,
    "max_ellipse_error": 0.25,
    "center_tolerance_px": 20.0,
    "min_concentric_contours": 2,
    "max_stroke_pair_ratio": 1.20,
    "morph_close_size": 3,
    "min_arc_coverage": 0.22,
    "single_candidate_min_coverage": 0.45,

    # A result is returned only after a stable sliding window is available.
    "stable_frames": 7,
    "max_center_mad_px": 2.5,
    "max_scale_rel_mad": 0.08,
    "frame_interval_s": 0.03,

    # Applied after perspective scaling. Test and update this 2x2 matrix.
    # Default: image-right -> body +X, image-down -> body +Y.
    "camera_to_body": ((1.0, 0.0), (0.0, 1.0)),
    "direction_calibrated": False,
    "deadband_mm": 2.0,
    "max_correction_mm": 150.0,
    "debug_window_name": "Ring Debug",
}


@dataclass(frozen=True)
class EllipseObservation:
    center_px: tuple[float, float]
    major_axis_px: float
    minor_axis_px: float
    major_angle_deg: float
    fit_error: float
    support_ratio: float = 1.0


@dataclass(frozen=True)
class RingOffset:
    center_px: tuple[float, float]
    reference_px: tuple[float, float]
    pixel_offset_px: tuple[float, float]
    image_offset_mm: tuple[float, float]
    body_offset_mm: tuple[float, float]
    inner_axes_px: tuple[float, float]
    major_angle_deg: float
    scale_mm_per_px: tuple[float, float]
    concentric_contours: int
    confidence: float
    samples: int
    direction_calibrated: bool
    deadband_mm: float
    max_correction_mm: float

    @property
    def aligned(self) -> bool:
        return math.hypot(*self.body_offset_mm) <= self.deadband_mm

    def body_correction_mm(self) -> tuple[float, float]:
        """Return a safe-to-send body offset after direction calibration."""
        if not self.direction_calibrated:
            raise RuntimeError(
                "Ring: direction is not calibrated; set CONFIG['camera_to_body'] "
                "and CONFIG['direction_calibrated'] first"
            )
        if math.hypot(*self.body_offset_mm) > self.max_correction_mm:
            raise RuntimeError(
                f"Ring: correction {self.body_offset_mm!r} mm exceeds "
                f"{self.max_correction_mm:.1f} mm safety limit"
            )
        return self.body_offset_mm

    def world_correction_mm(self, yaw_deg: float) -> tuple[float, float]:
        """Convert the calibrated body offset to the STM32 field frame.

        The project convention is +X right, +Y forward, yaw clockwise from
        forward.  STM32 ``fine_move``/``vision_correct`` consume field-frame
        dx/dy, not body-frame dx/dy.
        """
        bx, by = self.body_correction_mm()
        yaw = float(yaw_deg)
        if not math.isfinite(yaw):
            raise ValueError("yaw_deg must be finite")
        angle = math.radians(yaw)
        c, s = math.cos(angle), math.sin(angle)
        return c * bx + s * by, -s * bx + c * by


def _require_vision_dependencies() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("Ring: OpenCV and NumPy are required for camera detection")


def _open_usb():
    _require_vision_dependencies()
    cap = cv2.VideoCapture(CONFIG["usb_device"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["height"])
    cap.set(cv2.CAP_PROP_FPS, CONFIG["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            f"Ring: USB camera (device {CONFIG['usb_device']}) open failed"
        )
    return cap


def _normalise_ellipse(raw_ellipse) -> EllipseObservation:
    (cx, cy), (axis_a, axis_b), angle = raw_ellipse
    if axis_a >= axis_b:
        major, minor = float(axis_a), float(axis_b)
        major_angle = float(angle) % 180.0
    else:
        major, minor = float(axis_b), float(axis_a)
        major_angle = (float(angle) + 90.0) % 180.0
    return EllipseObservation(
        center_px=(float(cx), float(cy)),
        major_axis_px=major,
        minor_axis_px=minor,
        major_angle_deg=major_angle,
        fit_error=0.0,
    )


def _ellipse_fit_error(contour, ellipse: EllipseObservation) -> float:
    """Median normalized radial error between a contour and its fitted ellipse."""
    points = contour.reshape(-1, 2).astype(np.float64)
    dx = points[:, 0] - ellipse.center_px[0]
    dy = points[:, 1] - ellipse.center_px[1]
    angle = math.radians(ellipse.major_angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    along = c * dx + s * dy
    across = -s * dx + c * dy
    radius = np.sqrt(
        (along / (ellipse.major_axis_px * 0.5)) ** 2
        + (across / (ellipse.minor_axis_px * 0.5)) ** 2
    )
    return float(np.median(np.abs(radius - 1.0)))


def _ellipse_arc_coverage(contour, ellipse: EllipseObservation) -> float:
    """Fraction of ellipse angle bins supported by contour points."""
    points = contour.reshape(-1, 2).astype(np.float64)
    dx = points[:, 0] - ellipse.center_px[0]
    dy = points[:, 1] - ellipse.center_px[1]
    angle = math.radians(ellipse.major_angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    along = c * dx + s * dy
    across = -s * dx + c * dy
    theta = np.arctan2(
        across / max(ellipse.minor_axis_px * 0.5, 1e-6),
        along / max(ellipse.major_axis_px * 0.5, 1e-6),
    )
    bins = np.floor(((theta + math.pi) / (2.0 * math.pi)) * 36).astype(np.int32)
    bins = np.clip(bins, 0, 35)
    return float(len(set(int(v) for v in bins)) / 36.0)


def _black_mask(frame):
    height, width = frame.shape[:2]
    reference = (width * 0.5, height * 0.5)
    search_radius = int(min(width, height) * CONFIG["search_radius_fraction"])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([180, 255, CONFIG["black_v_max"]], dtype=np.uint8),
    )
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(roi, (round(reference[0]), round(reference[1])), search_radius, 255, -1)
    mask = cv2.bitwise_and(mask, roi)
    close_size = max(1, int(CONFIG["morph_close_size"]))
    if close_size % 2 == 0:
        close_size += 1
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8)
    )
    return mask


def _ellipse_candidates(frame, mask=None) -> list[EllipseObservation]:
    height, width = frame.shape[:2]
    if mask is None:
        mask = _black_mask(frame)

    found = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = found[-2]
    candidates = []
    max_axis = min(width, height) * CONFIG["max_axis_fraction"]

    for contour in contours:
        if len(contour) < 5 or cv2.contourArea(contour) < CONFIG["min_contour_area"]:
            continue
        ellipse = _normalise_ellipse(cv2.fitEllipse(contour))
        if ellipse.minor_axis_px < CONFIG["min_axis_px"]:
            continue
        if ellipse.major_axis_px > max_axis:
            continue
        if ellipse.major_axis_px / ellipse.minor_axis_px > CONFIG["max_axis_ratio"]:
            continue
        error = _ellipse_fit_error(contour, ellipse)
        if error > CONFIG["max_ellipse_error"]:
            continue
        support_ratio = _ellipse_arc_coverage(contour, ellipse)
        if support_ratio < CONFIG["min_arc_coverage"]:
            continue
        candidates.append(
            EllipseObservation(
                center_px=ellipse.center_px,
                major_axis_px=ellipse.major_axis_px,
                minor_axis_px=ellipse.minor_axis_px,
                major_angle_deg=ellipse.major_angle_deg,
                fit_error=error,
                support_ratio=support_ratio,
            )
        )
    return candidates


def _single_candidate_fallback(
    candidates: list[EllipseObservation], reference_px: tuple[float, float]
) -> tuple[EllipseObservation, tuple[float, float], int, float] | None:
    min_support = max(
        float(CONFIG["min_arc_coverage"]),
        float(CONFIG["single_candidate_min_coverage"]),
    )
    eligible = [item for item in candidates if item.support_ratio >= min_support]
    if not eligible:
        return None
    candidate = max(
        eligible,
        key=lambda item: (
            item.major_axis_px + item.minor_axis_px,
            item.support_ratio,
            -math.dist(item.center_px, reference_px),
            -item.fit_error,
        ),
    )
    fit_score = max(0.0, 1.0 - candidate.fit_error / CONFIG["max_ellipse_error"])
    confidence = min(1.0, candidate.support_ratio) * fit_score * 0.35
    return candidate, candidate.center_px, 1, confidence


def _select_concentric_group(
    candidates: list[EllipseObservation], reference_px: tuple[float, float]
) -> tuple[EllipseObservation, tuple[float, float], int, float] | None:
    if not candidates:
        return None

    tolerance = float(CONFIG["center_tolerance_px"])
    groups = []
    for seed in candidates:
        group = [
            item
            for item in candidates
            if math.dist(seed.center_px, item.center_px) <= tolerance
        ]
        if len(group) < CONFIG["min_concentric_contours"]:
            continue
        center = (
            statistics.median(item.center_px[0] for item in group),
            statistics.median(item.center_px[1] for item in group),
        )
        mean_error = statistics.fmean(item.fit_error for item in group)
        groups.append((group, center, mean_error))

    if not groups:
        return _single_candidate_fallback(candidates, reference_px)

    group, center, mean_error = min(
        groups,
        key=lambda item: (
            -len(item[0]),
            math.dist(item[1], reference_px),
            item[2],
        ),
    )
    ordered = sorted(group, key=lambda item: item.major_axis_px + item.minor_axis_px)
    innermost = ordered[0]
    if len(ordered) > 1:
        next_edge = ordered[1]
        pair_limit = float(CONFIG["max_stroke_pair_ratio"])
        is_stroke_pair = (
            next_edge.major_axis_px / innermost.major_axis_px <= pair_limit
            and next_edge.minor_axis_px / innermost.minor_axis_px <= pair_limit
        )
        if is_stroke_pair:
            # An outlined circle produces an inner and outer contour. Their
            # mean is the printed centre-line diameter specified as 50 mm.
            innermost = EllipseObservation(
                center_px=(
                    (innermost.center_px[0] + next_edge.center_px[0]) * 0.5,
                    (innermost.center_px[1] + next_edge.center_px[1]) * 0.5,
                ),
                major_axis_px=(innermost.major_axis_px + next_edge.major_axis_px) * 0.5,
                minor_axis_px=(innermost.minor_axis_px + next_edge.minor_axis_px) * 0.5,
                major_angle_deg=_mean_axis_angle_deg(
                    [innermost.major_angle_deg, next_edge.major_angle_deg]
                ),
                fit_error=(innermost.fit_error + next_edge.fit_error) * 0.5,
            )
    count_score = min(1.0, len(group) / (CONFIG["min_concentric_contours"] + 2.0))
    fit_score = max(0.0, 1.0 - mean_error / CONFIG["max_ellipse_error"])
    support_score = statistics.fmean(item.support_ratio for item in group)
    return innermost, center, len(group), count_score * fit_score * support_score


def _ellipse_offset_to_mm(
    pixel_offset: tuple[float, float],
    major_axis_px: float,
    minor_axis_px: float,
    major_angle_deg: float,
    diameter_mm: float,
) -> tuple[float, float]:
    """Undo the ellipse's local perspective stretch in image coordinates."""
    if major_axis_px <= 0 or minor_axis_px <= 0 or diameter_mm <= 0:
        raise ValueError("ellipse axes and physical diameter must be positive")
    du, dv = pixel_offset
    angle = math.radians(major_angle_deg)
    c, s = math.cos(angle), math.sin(angle)

    along = c * du + s * dv
    across = -s * du + c * dv
    along_mm = along * diameter_mm / major_axis_px
    across_mm = across * diameter_mm / minor_axis_px
    return c * along_mm - s * across_mm, s * along_mm + c * across_mm


def _apply_axis_transform(offset_mm: tuple[float, float]) -> tuple[float, float]:
    matrix = CONFIG["camera_to_body"]
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("CONFIG['camera_to_body'] must be a 2x2 matrix")
    x, y = offset_mm
    bx = float(matrix[0][0]) * x + float(matrix[0][1]) * y
    by = float(matrix[1][0]) * x + float(matrix[1][1]) * y
    if not math.isfinite(bx) or not math.isfinite(by):
        raise ValueError("CONFIG['camera_to_body'] produced a non-finite offset")
    return bx, by


def _measurement_from_candidates(
    frame, candidates: list[EllipseObservation]
) -> RingOffset | None:
    height, width = frame.shape[:2]
    reference = (width * 0.5, height * 0.5)
    selected = _select_concentric_group(candidates, reference)
    if selected is None:
        return None

    ellipse, center, contour_count, confidence = selected
    pixel_offset = (center[0] - reference[0], center[1] - reference[1])
    diameter = float(CONFIG["inner_circle_diameter_mm"])
    image_offset = _ellipse_offset_to_mm(
        pixel_offset,
        ellipse.major_axis_px,
        ellipse.minor_axis_px,
        ellipse.major_angle_deg,
        diameter,
    )
    body_offset = _apply_axis_transform(image_offset)
    return RingOffset(
        center_px=center,
        reference_px=reference,
        pixel_offset_px=pixel_offset,
        image_offset_mm=image_offset,
        body_offset_mm=body_offset,
        inner_axes_px=(ellipse.major_axis_px, ellipse.minor_axis_px),
        major_angle_deg=ellipse.major_angle_deg,
        scale_mm_per_px=(
            diameter / ellipse.major_axis_px,
            diameter / ellipse.minor_axis_px,
        ),
        concentric_contours=contour_count,
        confidence=confidence,
        samples=1,
        direction_calibrated=bool(CONFIG["direction_calibrated"]),
        deadband_mm=float(CONFIG["deadband_mm"]),
        max_correction_mm=float(CONFIG["max_correction_mm"]),
    )


def _measure_frame_with_mask(frame, mask) -> RingOffset | None:
    return _measurement_from_candidates(frame, _ellipse_candidates(frame, mask))


def measure_frame(frame) -> RingOffset | None:
    """Measure one frame; return None when no trustworthy ring is present."""
    _require_vision_dependencies()
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("frame must be a valid image")
    return _measure_frame_with_mask(frame, _black_mask(frame))


def _relative_mad(values: list[float]) -> float:
    middle = statistics.median(values)
    if middle <= 0:
        return math.inf
    return statistics.median(abs(value - middle) for value in values) / middle


def _mean_axis_angle_deg(angles: list[float]) -> float:
    """Circular mean for ellipse-axis angles, whose period is 180 degrees."""
    sin_sum = statistics.fmean(math.sin(math.radians(2.0 * a)) for a in angles)
    cos_sum = statistics.fmean(math.cos(math.radians(2.0 * a)) for a in angles)
    return (math.degrees(math.atan2(sin_sum, cos_sum)) * 0.5) % 180.0


def _aggregate_measurements(items: list[RingOffset]) -> RingOffset | None:
    if not items:
        return None

    center_x = [item.center_px[0] for item in items]
    center_y = [item.center_px[1] for item in items]
    mad_x = statistics.median(abs(v - statistics.median(center_x)) for v in center_x)
    mad_y = statistics.median(abs(v - statistics.median(center_y)) for v in center_y)
    major_scales = [item.scale_mm_per_px[0] for item in items]
    minor_scales = [item.scale_mm_per_px[1] for item in items]
    if max(mad_x, mad_y) > CONFIG["max_center_mad_px"]:
        return None
    if max(_relative_mad(major_scales), _relative_mad(minor_scales)) > CONFIG["max_scale_rel_mad"]:
        return None

    def pair_median(attr: str) -> tuple[float, float]:
        pairs = [getattr(item, attr) for item in items]
        return statistics.median(v[0] for v in pairs), statistics.median(v[1] for v in pairs)

    return RingOffset(
        center_px=(statistics.median(center_x), statistics.median(center_y)),
        reference_px=pair_median("reference_px"),
        pixel_offset_px=pair_median("pixel_offset_px"),
        image_offset_mm=pair_median("image_offset_mm"),
        body_offset_mm=pair_median("body_offset_mm"),
        inner_axes_px=pair_median("inner_axes_px"),
        major_angle_deg=_mean_axis_angle_deg([item.major_angle_deg for item in items]),
        scale_mm_per_px=(statistics.median(major_scales), statistics.median(minor_scales)),
        concentric_contours=round(statistics.median(item.concentric_contours for item in items)),
        confidence=statistics.median(item.confidence for item in items),
        samples=len(items),
        direction_calibrated=all(item.direction_calibrated for item in items),
        deadband_mm=float(CONFIG["deadband_mm"]),
        max_correction_mm=float(CONFIG["max_correction_mm"]),
    )


def detect_offset(timeout: float = 10.0, stable_frames: int | None = None) -> RingOffset:
    """Open the USB camera and return a stable ring offset measurement."""
    sample_count = int(CONFIG["stable_frames"] if stable_frames is None else stable_frames)
    if not math.isfinite(timeout) or timeout <= 0 or sample_count <= 0:
        raise ValueError("timeout and stable_frames must be positive")

    cap = _open_usb()
    recent: deque[RingOffset] = deque(maxlen=sample_count)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok:
                time.sleep(float(CONFIG["frame_interval_s"]))
                continue
            measurement = measure_frame(frame)
            if measurement is None:
                recent.clear()
            else:
                recent.append(measurement)
                if len(recent) == sample_count:
                    stable = _aggregate_measurements(list(recent))
                    if stable is not None:
                        return stable
            time.sleep(float(CONFIG["frame_interval_s"]))
    finally:
        cap.release()
    raise RuntimeError(f"Ring: no stable concentric ring found within {timeout:.1f}s")


def detect_centers(*_args, **_kwargs):
    """Reject the old ambiguous API instead of returning unsafe coordinates."""
    raise RuntimeError(
        "Ring: detect_centers() was removed because it mixed image and map "
        "coordinates; use detect_offset()"
    )


def _put_debug_text(image, text: str, origin: tuple[int, int], color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, 0.55, color, 1, cv2.LINE_AA)


def _render_debug_frame(
    frame, mask, measurement: RingOffset | None, stable: RingOffset | None,
    buffered_samples: int, required_samples: int, fps: float,
    candidates: list[EllipseObservation] | None = None,
):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    reference = (round(width * 0.5), round(height * 0.5))

    cv2.drawMarker(
        annotated, reference, (255, 255, 0), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA
    )

    for candidate in candidates or []:
        candidate_center = (
            round(candidate.center_px[0]), round(candidate.center_px[1])
        )
        candidate_axes = (
            max(1, round(candidate.major_axis_px * 0.5)),
            max(1, round(candidate.minor_axis_px * 0.5)),
        )
        cv2.ellipse(
            annotated, candidate_center, candidate_axes,
            candidate.major_angle_deg, 0, 360, (255, 120, 0), 1, cv2.LINE_AA,
        )

    shown = stable or measurement
    if stable is not None:
        status = f"STABLE {stable.samples}/{required_samples}"
        status_color = (0, 220, 0)
    elif measurement is not None:
        status = f"TRACKING {buffered_samples}/{required_samples}"
        status_color = (0, 220, 255)
    else:
        status = "NO RING"
        status_color = (0, 0, 255)

    if shown is not None:
        center = (round(shown.center_px[0]), round(shown.center_px[1]))
        axes = (
            max(1, round(shown.inner_axes_px[0] * 0.5)),
            max(1, round(shown.inner_axes_px[1] * 0.5)),
        )
        cv2.ellipse(
            annotated, center, axes, shown.major_angle_deg,
            0, 360, status_color, 2, cv2.LINE_AA,
        )
        cv2.drawMarker(
            annotated, center, (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA
        )
        cv2.line(annotated, reference, center, status_color, 2, cv2.LINE_AA)

    best_arc = max((item.support_ratio for item in candidates or []), default=0.0)
    lines = [
        status,
        f"FPS {fps:.1f} candidates={len(candidates or [])} best_arc={best_arc:.2f}",
    ]
    if shown is not None:
        lines.extend([
            f"pixel du={shown.pixel_offset_px[0]:+.1f} dv={shown.pixel_offset_px[1]:+.1f}",
            f"inner axes={shown.inner_axes_px[0]:.1f} x {shown.inner_axes_px[1]:.1f} px",
            f"scale={shown.scale_mm_per_px[0]:.4f}, {shown.scale_mm_per_px[1]:.4f} mm/px",
            f"image mm={shown.image_offset_mm[0]:+.1f}, {shown.image_offset_mm[1]:+.1f}",
            f"body mm={shown.body_offset_mm[0]:+.1f}, {shown.body_offset_mm[1]:+.1f}",
            f"confidence={shown.confidence:.2f} aligned={shown.aligned}",
            "direction=READY" if shown.direction_calibrated else "direction=LOCKED",
        ])
    for index, line in enumerate(lines):
        _put_debug_text(annotated, line, (12, 24 + index * 22), status_color)

    mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    _put_debug_text(mask_view, "BLACK THRESHOLD MASK", (12, 24), (255, 255, 255))
    return np.hstack((annotated, mask_view))


def _create_debug_trackbars(window: str) -> None:
    noop = lambda _value: None
    cv2.createTrackbar("V Max", window, int(CONFIG["black_v_max"]), 255, noop)
    cv2.createTrackbar("Close", window, int(CONFIG["morph_close_size"]), 15, noop)
    cv2.createTrackbar("Min Axis", window, int(CONFIG["min_axis_px"]), 200, noop)
    cv2.createTrackbar(
        "Center Tol", window, int(CONFIG["center_tolerance_px"]), 100, noop
    )
    cv2.createTrackbar(
        "Fit Err x100", window, round(CONFIG["max_ellipse_error"] * 100), 100, noop
    )
    cv2.createTrackbar(
        "Min Contours", window, int(CONFIG["min_concentric_contours"]), 12, noop
    )
    cv2.createTrackbar(
        "ROI Percent", window, round(CONFIG["search_radius_fraction"] * 100), 100, noop
    )
    cv2.createTrackbar(
        "Min Arc %", window, round(CONFIG["min_arc_coverage"] * 100), 100, noop
    )
    cv2.createTrackbar(
        "Single Arc %", window,
        round(CONFIG["single_candidate_min_coverage"] * 100), 100, noop,
    )


def _read_debug_trackbars(window: str) -> None:
    CONFIG["black_v_max"] = max(1, cv2.getTrackbarPos("V Max", window))
    close_size = max(1, cv2.getTrackbarPos("Close", window))
    if close_size % 2 == 0:
        close_size += 1
    CONFIG["morph_close_size"] = close_size
    CONFIG["min_axis_px"] = max(1, cv2.getTrackbarPos("Min Axis", window))
    CONFIG["center_tolerance_px"] = max(
        1, cv2.getTrackbarPos("Center Tol", window)
    )
    CONFIG["max_ellipse_error"] = max(
        0.01, cv2.getTrackbarPos("Fit Err x100", window) / 100.0
    )
    CONFIG["min_concentric_contours"] = max(
        1, cv2.getTrackbarPos("Min Contours", window)
    )
    CONFIG["search_radius_fraction"] = max(
        0.10, cv2.getTrackbarPos("ROI Percent", window) / 100.0
    )
    CONFIG["min_arc_coverage"] = max(
        0.05, cv2.getTrackbarPos("Min Arc %", window) / 100.0
    )
    CONFIG["single_candidate_min_coverage"] = max(
        0.05, cv2.getTrackbarPos("Single Arc %", window) / 100.0
    )


def _debug_tuning_values() -> dict:
    keys = (
        "black_v_max", "morph_close_size", "min_axis_px",
        "center_tolerance_px", "max_ellipse_error",
        "min_concentric_contours", "search_radius_fraction",
        "min_arc_coverage", "single_candidate_min_coverage",
    )
    return {key: CONFIG[key] for key in keys}


def debug_view() -> None:
    """Show live detection and threshold views until q/Esc or window close."""
    _require_vision_dependencies()
    cap = _open_usb()
    required = int(CONFIG["stable_frames"])
    recent: deque[RingOffset] = deque(maxlen=required)
    window = str(CONFIG["debug_window_name"])
    fps = 0.0
    previous_tick = time.monotonic()

    print("Ring debug controls: q/Esc=quit, p=print values, r=reset stability")
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        _create_debug_trackbars(window)
        while True:
            _read_debug_trackbars(window)
            ok, frame = cap.read()
            if not ok:
                time.sleep(float(CONFIG["frame_interval_s"]))
                continue

            mask = _black_mask(frame)
            candidates = _ellipse_candidates(frame, mask)
            measurement = _measurement_from_candidates(frame, candidates)
            if measurement is None:
                recent.clear()
            else:
                recent.append(measurement)
            stable = (
                _aggregate_measurements(list(recent))
                if len(recent) == required else None
            )

            now = time.monotonic()
            instant_fps = 1.0 / max(now - previous_tick, 1e-6)
            fps = instant_fps if fps == 0.0 else fps * 0.9 + instant_fps * 0.1
            previous_tick = now

            view = _render_debug_frame(
                frame, mask, measurement, stable, len(recent), required, fps,
                candidates,
            )
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                print("Ring tuning:", _debug_tuning_values())
                print(stable or measurement or "Ring: no measurement")
            elif key == ord("r"):
                recent.clear()
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except cv2.error as exc:
        raise RuntimeError(
            "Ring: OpenCV GUI is unavailable; run from the Jetson desktop "
            "session with DISPLAY configured"
        ) from exc
    finally:
        cap.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


if __name__ == "__main__":
    debug_view()
