import base64
import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import color_cleanup as color
import webui


class ColourTests(unittest.TestCase):
    def test_wide_colour_patches_of_multiple_sizes(self):
        y, x = np.mgrid[:128, :160]
        for radius in (5, 12, 22):
            with self.subTest(radius=radius):
                patch = (0.085 * np.exp(-((x - 80) ** 2 + (y - 64) ** 2) /
                                       (2 * radius ** 2))).astype(np.float32)
                luma = (0.4 + 0.015 * np.sin(x * 1.5)).astype(np.float32)
                image = np.repeat(luma[..., None], 3, axis=2)
                image[..., 0] += patch
                image[..., 1] -= patch * color.LUMA[0] / color.LUMA[1]
                result = color.clean_chroma(image, 1)
                np.testing.assert_allclose(result @ color.LUMA, image @ color.LUMA, atol=1e-6)
                self.assertLess(result[64, 80, 0] - luma[64, 80], patch[64, 80] * 0.7)
                # A constant coloured material is not automatically desaturated.
                uniform = np.broadcast_to(image[64, 80], image.shape).copy()
                np.testing.assert_allclose(color.clean_chroma(uniform, 1), uniform, atol=1e-6)

    def test_bands_under_fine_texture(self):
        y, x = np.mgrid[:96, :256]
        ramp = (0.25 + 0.3 * x / 255).astype(np.float32)
        detail = (0.009 * np.cos(x * np.pi) * np.cos(y * np.pi)).astype(np.float32)
        banded = np.round(ramp * 32) / 32
        image = np.repeat((banded + detail)[..., None], 3, axis=2)
        result, mask = color.clean_gradients(image, 1)
        target = ramp + detail
        region = np.s_[16:-16, 40:-40]
        before = np.mean((image[..., 0][region] - target[region]) ** 2)
        after = np.mean((result[..., 0][region] - target[region]) ** 2)
        self.assertLess(after, before * 0.8)
        # The checker component should not be globally blurred away.
        original_detail = np.mean((image[1:-1, 40:-40, 0] - image[2:, 40:-40, 0]) ** 2)
        retained_detail = np.mean((result[1:-1, 40:-40, 0] - result[2:, 40:-40, 0]) ** 2)
        self.assertGreater(retained_detail, original_detail * 0.85)

    def test_box_max_matches_full_window(self):
        field = np.random.default_rng(7).random((5, 9), dtype=np.float32)
        for radius in (1, 2, 4, 8):
            padded = np.pad(field, radius, mode="edge")
            expected = np.array([[padded[y:y + 2*radius + 1, x:x + 2*radius + 1].max()
                                  for x in range(9)] for y in range(5)])
            np.testing.assert_array_equal(color._box_max(field, radius), expected)

    def test_chroma_noise_reduced_without_luma_change(self):
        rng = np.random.default_rng(42)
        y, x = np.mgrid[:64, :80]
        luma = (0.3 + 0.3 * x / 80 + 0.02 * (y % 2)).astype(np.float32)
        image = np.repeat(luma[..., None], 3, axis=2)
        noise = rng.normal(0, 0.022, luma.shape).astype(np.float32)
        image[..., 0] += noise
        image[..., 1] -= noise * color.LUMA[0] / color.LUMA[1]
        result = color.clean_chroma(image, 1)
        np.testing.assert_allclose(result @ color.LUMA, image @ color.LUMA, atol=1e-6)
        self.assertLess((result[..., 0] - luma).std(), noise.std() * 0.65)

    def test_isoluminant_colour_edge_and_gamut(self):
        image = np.full((48, 64, 3), 0.4, np.float32)
        image[:, 32:, 0] += 0.3
        image[:, 32:, 1] -= 0.3 * color.LUMA[0] / color.LUMA[1]
        result = color.clean_chroma(image, 1)
        np.testing.assert_allclose(result, image, atol=1e-5)
        rng = np.random.default_rng(1)
        image = rng.uniform(0, 1, (32, 32, 3)).astype(np.float32)
        image[0] = [1, 0, 0]
        result = color.clean_chroma(image, 1)
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)
        np.testing.assert_allclose(result @ color.LUMA, image @ color.LUMA, atol=1e-6)

    def test_bands_reduced_and_edges_protected(self):
        ramp = np.broadcast_to(np.linspace(0.3, 0.6, 256, dtype=np.float32), (64, 256))
        image = np.repeat((np.round(ramp * 64) / 64)[..., None], 3, axis=2)
        result, mask = color.clean_gradients(image, 1)
        region = np.s_[:, 20:-20, 0]
        before = np.mean((image[region] - ramp[:, 20:-20]) ** 2)
        after = np.mean((result[region] - ramp[:, 20:-20]) ** 2)
        self.assertLess(after, before * 0.65)
        self.assertGreater(mask.mean(), 0.1)
        image = np.full((64, 64, 3), 0.2, np.float32)
        image[:, 32:] = 0.8
        image[15, :] = 1
        result, _ = color.clean_gradients(image, 1)
        np.testing.assert_allclose(result, image, atol=1e-6)

    def test_texture_protection_and_neutrality(self):
        y, x = np.mgrid[:64, :64]
        image = np.repeat((0.5 + 0.08 * ((x + y) % 2))[..., None], 3, axis=2).astype(np.float32)
        result, _ = color.clean_gradients(image, 1)
        np.testing.assert_allclose(result, image, atol=1e-6)
        np.testing.assert_array_equal(result[..., 0], result[..., 2])

    def test_zero_small_flat_and_deterministic_dither(self):
        for shape in ((1, 1, 3), (2, 5, 3), (32, 32, 3)):
            image = np.full(shape, 127 / 255, np.float32)
            self.assertIs(color.clean_chroma(image, 0), image)
            self.assertIs(color.clean_gradients(image, 0)[0], image)
            result, mask = color.clean_gradients(color.clean_chroma(image, 1), 1)
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(color.finish(result, 0, 123, mask),
                                          np.full(shape, 127, np.uint8))
        image = np.full((64, 64, 3), 127.5 / 255, np.float32)
        mask = np.ones((64, 64), np.float32)
        first = color.finish(image, 0, 123, mask)
        np.testing.assert_array_equal(first, color.finish(image, 0, 123, mask))
        np.testing.assert_array_equal(first[..., 0], first[..., 1])
        self.assertEqual(set(np.unique(first)), {127, 128})


class ColourPipelineTests(unittest.TestCase):
    def test_bypass_caching_sr_and_export(self):
        old_state = webui.STATE.copy()
        webui.STATE.update(device="cpu", cache_n=2)
        webui.CACHE.clear()
        webui.CACHE_ORDER.clear()
        rng = np.random.default_rng(3)
        rgb = rng.integers(110, 135, (64, 64, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        payload = base64.b64encode(buf.getvalue()).decode()
        def pixels(response):
            return np.asarray(Image.open(io.BytesIO(base64.b64decode(response["result"].split(",")[1]))))
        try:
            with patch.object(webui, "_encode_z", return_value=(None, None, 64, 64)) as encode, \
                 patch.object(webui, "_decode", return_value=rgb.astype(np.float32) / 255) as decode, \
                 patch.object(webui, "_realesrgan", side_effect=lambda arr, scale, blend: np.repeat(np.repeat(arr, scale, 0), scale, 1)) as sr:
                before = webui.process(payload, 0, None)
                np.testing.assert_array_equal(pixels(before), rgb)
                changed = webui.process(payload, 0, None, chroma=1, deband=0.5)
                self.assertTrue(np.any(pixels(changed) != rgb))
                marked = webui.process(payload, 0, None, chroma=1, deband=0.5, fine_enabled=True, show_mask=True)
                plain = webui.process(payload, 0, None, chroma=1, deband=0.5, fine_enabled=True)
                self.assertEqual(marked["result"], plain["result"])
                webui.process(payload, 0, None, chroma=1, deband=0.5, dither=False)
                np.testing.assert_array_equal(pixels(webui.process(payload, 0, None)), rgb)
                for strength in (0.3, 0.8):
                    result = webui.process(payload, 1, None, chroma=strength, deband=strength, sr_mode="2x")
                    self.assertEqual(pixels(result).shape, (128, 128, 3))
                self.assertEqual(sr.call_count, 1)
                self.assertEqual(encode.call_count, 1)
                self.assertEqual(decode.call_count, 2)
                for parameter in ("chroma", "deband"):
                    with self.assertRaises(ValueError):
                        webui.process(payload, 0, None, **{parameter: float("nan")})
        finally:
            webui.STATE.clear()
            webui.STATE.update(old_state)
            webui.CACHE.clear()
            webui.CACHE_ORDER.clear()


if __name__ == "__main__":
    unittest.main()
