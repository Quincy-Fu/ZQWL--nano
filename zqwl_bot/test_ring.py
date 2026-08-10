"""Pure geometry tests for ring localization; no camera/OpenCV required."""

import math
import unittest
from unittest import mock

import ring


def _measurement(**overrides):
    values = {
        "center_px": (330.0, 230.0),
        "reference_px": (320.0, 240.0),
        "pixel_offset_px": (10.0, -10.0),
        "image_offset_mm": (5.0, -5.0),
        "body_offset_mm": (5.0, -5.0),
        "inner_axes_px": (100.0, 100.0),
        "major_angle_deg": 0.0,
        "scale_mm_per_px": (0.5, 0.5),
        "concentric_contours": 6,
        "confidence": 0.9,
        "samples": 1,
        "direction_calibrated": True,
        "deadband_mm": 2.0,
        "max_correction_mm": 150.0,
    }
    values.update(overrides)
    return ring.RingOffset(**values)


class RingGeometryTests(unittest.TestCase):
    def setUp(self):
        self._config = dict(ring.CONFIG)

    def tearDown(self):
        ring.CONFIG.clear()
        ring.CONFIG.update(self._config)

    def assertPairAlmostEqual(self, actual, expected):
        self.assertAlmostEqual(actual[0], expected[0], places=6)
        self.assertAlmostEqual(actual[1], expected[1], places=6)

    def test_configured_ring_diameters_match_printed_target(self):
        self.assertEqual(
            ring._configured_ring_diameters_mm(),
            (50.0, 90.0, 130.0, 170.0, 210.0),
        )

    def test_usb_camera_falls_back_from_device_zero_to_one(self):
        failed = mock.Mock()
        failed.isOpened.return_value = False
        opened = mock.Mock()
        opened.isOpened.return_value = True
        ring.CONFIG["usb_devices"] = (0, 1)
        fake_cv2 = mock.Mock()
        fake_cv2.CAP_V4L2 = 200
        fake_cv2.CAP_PROP_FOURCC = 6
        fake_cv2.CAP_PROP_FRAME_WIDTH = 3
        fake_cv2.CAP_PROP_FRAME_HEIGHT = 4
        fake_cv2.CAP_PROP_FPS = 5
        fake_cv2.CAP_PROP_BUFFERSIZE = 38
        fake_cv2.VideoCapture.side_effect = [failed, opened]

        with mock.patch.object(ring, "cv2", fake_cv2):
            with mock.patch.object(ring, "np", mock.Mock()):
                result = ring._open_usb()

        self.assertIs(result, opened)
        self.assertEqual(
            [item.args[0] for item in fake_cv2.VideoCapture.call_args_list],
            [0, 1],
        )
        failed.release.assert_called_once_with()

    def test_circle_uses_dynamic_50_mm_scale(self):
        offset = ring._ellipse_offset_to_mm((20.0, -10.0), 100.0, 100.0, 0.0, 50.0)
        self.assertPairAlmostEqual(offset, (10.0, -5.0))

    def test_ellipse_scales_major_and_minor_axes_independently(self):
        offset = ring._ellipse_offset_to_mm((20.0, -10.0), 200.0, 100.0, 0.0, 50.0)
        self.assertPairAlmostEqual(offset, (5.0, -5.0))

    def test_rotated_ellipse_rotates_scale_axes_with_it(self):
        offset = ring._ellipse_offset_to_mm((20.0, -10.0), 200.0, 100.0, 90.0, 50.0)
        self.assertPairAlmostEqual(offset, (10.0, -2.5))

    def test_camera_to_body_matrix_supports_axis_swap_and_signs(self):
        ring.CONFIG["camera_to_body"] = ((0.0, -1.0), (1.0, 0.0))
        self.assertPairAlmostEqual(ring._apply_axis_transform((12.0, -7.0)), (7.0, 12.0))

    def test_direction_must_be_calibrated_before_command_output(self):
        measurement = _measurement(direction_calibrated=False)
        with self.assertRaisesRegex(RuntimeError, "direction is not calibrated"):
            measurement.body_correction_mm()

    def test_oversized_correction_is_rejected(self):
        measurement = _measurement(body_offset_mm=(151.0, 0.0))
        with self.assertRaisesRegex(RuntimeError, "safety limit"):
            measurement.body_correction_mm()

    def test_body_offset_rotates_to_clockwise_field_frame(self):
        measurement = _measurement(body_offset_mm=(0.0, 10.0))
        self.assertPairAlmostEqual(measurement.world_correction_mm(90.0), (10.0, 0.0))

    def test_non_finite_yaw_is_rejected(self):
        measurement = _measurement()
        with self.assertRaisesRegex(ValueError, "yaw_deg must be finite"):
            measurement.world_correction_mm(float("nan"))

    def test_stable_measurements_are_aggregated_by_median(self):
        items = [
            _measurement(center_px=(330.0 + jitter, 230.0 - jitter),
                         scale_mm_per_px=(0.50 + jitter * 0.001, 0.60))
            for jitter in (-1.0, 0.0, 1.0)
        ]
        result = ring._aggregate_measurements(items)
        self.assertIsNotNone(result)
        self.assertEqual(result.center_px, (330.0, 230.0))
        self.assertEqual(result.samples, 3)

    def test_unstable_center_is_not_accepted(self):
        ring.CONFIG["max_center_mad_px"] = 1.0
        items = [
            _measurement(center_px=(320.0, 240.0)),
            _measurement(center_px=(330.0, 240.0)),
            _measurement(center_px=(340.0, 240.0)),
        ]
        self.assertIsNone(ring._aggregate_measurements(items))

    def test_axis_angle_mean_handles_180_degree_wrap(self):
        angle = ring._mean_axis_angle_deg([179.0, 0.0, 1.0])
        self.assertTrue(math.isclose(angle, 0.0, abs_tol=1e-6)
                        or math.isclose(angle, 180.0, abs_tol=1e-6))

    def test_concentric_group_selects_smallest_ellipse(self):
        candidates = [
            ring.EllipseObservation((321.0, 239.0), size, size * 0.8, 5.0, 0.01)
            for size in (160.0, 120.0, 80.0, 40.0)
        ]
        selected = ring._select_concentric_group(candidates, (320.0, 240.0))
        self.assertIsNotNone(selected)
        self.assertEqual(selected.ellipse.major_axis_px, 40.0)
        self.assertEqual(selected.center_px, (321.0, 239.0))
        self.assertEqual(selected.contour_count, 4)

    def test_outlined_inner_circle_uses_mean_of_both_stroke_edges(self):
        ring.CONFIG["ring_diameters_mm"] = (50.0,)
        candidates = [
            ring.EllipseObservation((320.0, 240.0), 96.0, 76.0, 20.0, 0.01),
            ring.EllipseObservation((320.0, 240.0), 104.0, 84.0, 20.0, 0.01),
            ring.EllipseObservation((320.0, 240.0), 150.0, 120.0, 20.0, 0.01),
            ring.EllipseObservation((320.0, 240.0), 190.0, 152.0, 20.0, 0.01),
        ]
        selected = ring._select_concentric_group(candidates, (320.0, 240.0))
        self.assertPairAlmostEqual(
            (selected.ellipse.major_axis_px, selected.ellipse.minor_axis_px),
            (100.0, 80.0),
        )

    def test_printed_diameter_model_selects_outer_ring(self):
        candidates = [
            ring.EllipseObservation(
                (320.0, 240.0), diameter * 2.0, diameter * 1.6, 20.0, 0.01
            )
            for diameter in (50.0, 90.0, 130.0, 170.0, 210.0)
        ]
        selected = ring._select_concentric_group(candidates, (320.0, 240.0))
        self.assertIsNotNone(selected)
        self.assertEqual(selected.diameter_mm, 210.0)
        self.assertEqual(selected.diameter_index, 4)
        self.assertEqual(selected.ellipse.major_axis_px, 420.0)

    def test_diameter_model_rejects_inner_text_candidate(self):
        ring.CONFIG["ring_diameters_mm"] = (50.0, 100.0)
        candidates = [
            ring.EllipseObservation((320.0, 240.0), 40.0, 32.0, 0.0, 0.01, 1.0),
            ring.EllipseObservation((320.0, 240.0), 100.0, 80.0, 0.0, 0.01, 1.0),
            ring.EllipseObservation((320.0, 240.0), 200.0, 160.0, 0.0, 0.01, 1.0),
        ]
        selected = ring._select_concentric_group(candidates, (320.0, 240.0))
        self.assertIsNotNone(selected)
        self.assertEqual(selected.ellipse.major_axis_px, 200.0)
        self.assertEqual(selected.diameter_mm, 100.0)
        self.assertEqual(selected.diameter_index, 1)

    def test_single_arc_fallback_uses_outermost_diameter(self):
        ring.CONFIG["ring_diameters_mm"] = (50.0, 100.0, 150.0)
        candidates = [
            ring.EllipseObservation((320.0, 240.0), 60.0, 40.0, 0.0, 0.01, 1.0),
            ring.EllipseObservation((350.0, 220.0), 140.0, 80.0, 25.0, 0.02, 0.55),
        ]
        selected = ring._select_concentric_group(candidates, (320.0, 240.0))
        self.assertIsNotNone(selected)
        self.assertEqual(selected.ellipse.major_axis_px, 140.0)
        self.assertEqual(selected.center_px, (350.0, 220.0))
        self.assertEqual(selected.contour_count, 1)
        self.assertEqual(selected.diameter_mm, 150.0)
        self.assertLess(selected.confidence, 0.35)

    def test_single_complete_candidate_is_rejected_as_text_like(self):
        ring.CONFIG["ring_diameters_mm"] = (50.0, 100.0, 150.0)
        self.assertIsNone(ring._ring_model_match([
            ring.EllipseObservation((320.0, 240.0), 200.0, 160.0, 0.0, 0.01, 1.0),
        ]))

    @unittest.skipIf(ring.cv2 is None or ring.np is None, "OpenCV is unavailable")
    def test_complete_inner_text_does_not_override_partial_ring(self):
        image = ring.np.full((480, 640, 3), 255, ring.np.uint8)
        ring.cv2.ellipse(image, (320, 240), (45, 65), 0, 0, 360, (0, 0, 0), 6)
        ring.cv2.ellipse(image, (320, 240), (140, 140), 0, 205, 295, (0, 0, 0), 3)

        result = ring.measure_frame(image)
        self.assertIsNotNone(result)
        self.assertGreater(result.inner_axes_px[0], 240.0)
        self.assertLess(result.confidence, 0.35)

    @unittest.skipIf(ring.cv2 is None or ring.np is None, "OpenCV is unavailable")
    def test_synthetic_outlined_ellipse_is_detected_at_centerline_scale(self):
        image = ring.np.full((480, 640, 3), 255, ring.np.uint8)
        center = (320, 240)
        # The major/minor axes follow the configured 50/90/130/170/210 mm
        # diameter ratios.  The outer ring remains fully inside the frame.
        for diameter, minor_diameter in zip(
            (50, 90, 130, 170, 210), (30, 54, 78, 102, 126)
        ):
            axes = (round(diameter * 0.5), round(minor_diameter * 0.5))
            ring.cv2.ellipse(image, center, axes, 25, 0, 360, (0, 0, 0), 3)

        result = ring.measure_frame(image)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.center_px[0], 320.0, delta=0.2)
        self.assertAlmostEqual(result.center_px[1], 240.0, delta=0.2)
        self.assertAlmostEqual(result.inner_axes_px[0], 210.0, delta=2.0)
        self.assertAlmostEqual(result.inner_axes_px[1], 126.0, delta=2.0)
        self.assertEqual(result.selected_diameter_mm, 210.0)

        mask = ring._black_mask(image)
        rendered = ring._render_debug_frame(
            image, mask, result, result, 7, 7, 30.0,
            ring._ellipse_candidates(image, mask),
        )
        self.assertEqual(rendered.shape, (480, 1280, 3))

    @unittest.skipIf(ring.cv2 is None or ring.np is None, "OpenCV is unavailable")
    def test_partial_arc_is_completed_when_enough_ring_is_visible(self):
        image = ring.np.full((480, 640, 3), 255, ring.np.uint8)
        ring.cv2.ellipse(image, (350, 220), (70, 40), 25, 200, 360, (0, 0, 0), 3)

        result = ring.measure_frame(image)
        self.assertIsNotNone(result)
        self.assertEqual(result.concentric_contours, 1)
        self.assertAlmostEqual(result.center_px[0], 350.0, delta=2.0)
        self.assertAlmostEqual(result.center_px[1], 220.0, delta=2.0)
        self.assertAlmostEqual(result.inner_axes_px[0], 140.0, delta=3.0)
        self.assertAlmostEqual(result.inner_axes_px[1], 80.0, delta=4.0)

    @unittest.skipIf(ring.cv2 is None or ring.np is None, "OpenCV is unavailable")
    def test_short_arc_is_rejected_as_underconstrained(self):
        image = ring.np.full((480, 640, 3), 255, ring.np.uint8)
        ring.cv2.ellipse(image, (350, 220), (70, 40), 25, 200, 290, (0, 0, 0), 3)

        self.assertIsNone(ring.measure_frame(image))


if __name__ == "__main__":
    unittest.main()
