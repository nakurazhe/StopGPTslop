import base64
import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import fine_cleanup as fine
import webui


class FineCleanupTests(unittest.TestCase):
    def test_flat_and_ramp_are_preserved(self):
        for field in (np.full((65, 79), 127, np.uint8),
                      np.tile(np.arange(79, dtype=np.uint8) * 3, (65, 1))):
            rgb = np.repeat(field[..., None], 3, axis=2)
            masks = fine.analyze(rgb)
            out = fine.clean(rgb, masks, 1, 1)
            self.assertLessEqual(np.abs(out.astype(int) - rgb).max(), 1)

    def test_thin_lines_and_step_edge_are_preserved(self):
        field = np.full((96, 96), 100, np.uint8)
        field[:, 15] = 230
        field[35, :] = 230
        field[:, 65:] = 180
        rgb = np.repeat(field[..., None], 3, axis=2)
        out = fine.clean(rgb, fine.analyze(rgb), 1, 1)
        self.assertLessEqual(np.abs(out.astype(int) - rgb).max(), 2)

    def test_isolated_bright_and_dark_dots_are_reduced(self):
        rgb = np.full((65, 65, 3), 128, np.uint8)
        rgb[20, 20] = 180
        rgb[40, 40] = 76
        masks = fine.analyze(rgb)
        out = fine.clean(rgb, masks, 0, 0.5)
        self.assertLess(int(out[20, 20, 0]), 175)
        self.assertGreater(int(out[40, 40, 0]), 81)
        np.testing.assert_array_equal(out[5:10, 5:10], rgb[5:10, 5:10])

    def test_repeated_pattern_is_reduced(self):
        y, x = np.mgrid[:96, :96]
        field = np.round(128 + 18 * np.cos(x * np.pi / 2) * np.cos(y * np.pi / 2)).astype(np.uint8)
        rgb = np.repeat(field[..., None], 3, axis=2)
        masks = fine.analyze(rgb)
        out = fine.clean(rgb, masks, 0.8, 0)
        self.assertLess(out[10:-10, 10:-10].std(), rgb[10:-10, 10:-10].std() * 0.95)

    def test_compose_bypass_and_selective_preservation(self):
        rgb = np.full((32, 32, 3), 128, np.uint8)
        rgb[16, 16] = 180
        baseline = np.full(rgb.shape, 128 / 255, np.float32)
        masks = fine.analyze(rgb)
        np.testing.assert_array_equal(fine.compose(rgb, baseline, baseline, masks, 0, 0), rgb)
        np.testing.assert_array_equal(fine.compose(rgb, baseline, baseline, masks, 1, 1), rgb)
        out = fine.compose(rgb, baseline, baseline, masks, 0, 1)
        self.assertLess(out[16, 16, 0], rgb[16, 16, 0])

    def test_small_images(self):
        for shape in ((1, 1, 3), (2, 5, 3)):
            rgb = np.full(shape, 127, np.uint8)
            np.testing.assert_array_equal(fine.clean(rgb, fine.analyze(rgb), 1, 1), rgb)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        webui.CACHE.clear()
        webui.CACHE_ORDER.clear()
        self.old_state = webui.STATE.copy()
        webui.STATE.update(device="cpu", cache_n=2)
        self.rgb = np.full((64, 64, 3), 128, np.uint8)
        self.rgb[20, 20] = 180
        buf = io.BytesIO()
        Image.fromarray(self.rgb).save(buf, format="PNG")
        self.payload = base64.b64encode(buf.getvalue()).decode()

    def tearDown(self):
        webui.CACHE.clear()
        webui.CACHE_ORDER.clear()
        webui.STATE.clear()
        webui.STATE.update(self.old_state)

    def test_cpu_controls_reuse_decode_and_mask_does_not_change_export(self):
        with patch.object(webui, "_encode_z", return_value=(None, None, 64, 64)), \
             patch.object(webui, "_decode", return_value=self.rgb.astype(np.float32) / 255) as decode:
            first = webui.process(self.payload, 1, None, fine_enabled=True, micro=0.55)
            marked = webui.process(self.payload, 1, "stale-key", fine_enabled=True, micro=0.55, show_mask=True)
            self.assertEqual(first["result"], marked["result"])
            self.assertIsNotNone(marked["preview"])
            self.assertEqual(first["key"], marked["key"])
            webui.process(self.payload, 1, None, fine_enabled=True, micro=0.8, spots=0.5, preserve=0.5)
            self.assertEqual(decode.call_count, 2)
            webui.process(self.payload, 0.5, None, fine_enabled=True)
            self.assertEqual(decode.call_count, 3)

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            webui.process(self.payload, float("nan"), None)


if __name__ == "__main__":
    unittest.main()
