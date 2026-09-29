"""Exact regression checks against the pre-optimization CPU primitives."""
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

import fine_cleanup as fine
import color_cleanup as color


def legacy_box_mean(field, radius):
    result = field.astype(np.float32, copy=False)
    width = 2 * radius + 1
    for axis in (0, 1):
        padding = [(0, 0)] * result.ndim
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="edge")
        sums = np.cumsum(padded, axis=axis, dtype=np.float64)
        zero_shape = list(sums.shape)
        zero_shape[axis] = 1
        sums = np.concatenate((np.zeros(zero_shape), sums), axis=axis)
        hi, lo = [slice(None)] * result.ndim, [slice(None)] * result.ndim
        hi[axis], lo[axis] = slice(width, None), slice(None, -width)
        result = ((sums[tuple(hi)] - sums[tuple(lo)]) / width).astype(np.float32)
    return result


def legacy_local_range(gray):
    image = Image.fromarray(gray)
    return (np.asarray(image.filter(ImageFilter.MaxFilter(5)), dtype=np.float32)
            - np.asarray(image.filter(ImageFilter.MinFilter(5)), dtype=np.float32)) / 255


class CleanupEquivalenceTests(unittest.TestCase):
    def test_box_mean_exact_signed_rgb_and_noncontiguous(self):
        rng = np.random.default_rng(302)
        for shape in ((1, 1), (2, 5), (65, 79), (31, 47, 3)):
            for dtype in (np.float32, np.float64, np.uint8):
                field = (rng.normal(size=shape) * 64).astype(dtype)
                for view in (field, field[::-1, ::2], field.swapaxes(0, 1)):
                    saved = view.copy()
                    for radius in (0, 1, 2, 3, 4, 8, 16, 32, 34):
                        actual = fine.box_mean(view, radius)
                        np.testing.assert_array_equal(actual, legacy_box_mean(view, radius))
                        self.assertEqual(actual.dtype, np.float32)
                    np.testing.assert_array_equal(view, saved)

    def test_local_range_exact_including_borders_and_tiny_images(self):
        rng = np.random.default_rng(303)
        for shape in ((1, 1), (1, 9), (9, 1), (2, 3), (65, 79)):
            for field in (rng.integers(0, 256, shape, dtype=np.uint8),
                          np.full(shape, 127, np.uint8),
                          (np.indices(shape).sum(axis=0) % 2 * 255).astype(np.uint8)):
                for view in (field, field[::-1, ::2]):
                    saved = view.copy()
                    np.testing.assert_array_equal(fine._local_range_5(view), legacy_local_range(view))
                    np.testing.assert_array_equal(view, saved)

    def test_cpu_stages_and_mask_exact(self):
        rng = np.random.default_rng(304)
        y, x = np.mgrid[:65, :79]
        images = [rng.integers(0, 256, (65, 79, 3), dtype=np.uint8),
                  np.repeat((x * 3).astype(np.uint8)[..., None], 3, axis=2),
                  np.repeat(((x + y) % 2 * 120 + 60).astype(np.uint8)[..., None], 3, axis=2)]

        def stages(image, alpha, preserve, micro, spots):
            baseline = image.astype(np.float32) / 255
            decoded = np.clip(baseline + .01, 0, 1)
            masks = fine.analyze(image)
            composed = fine.compose(image, baseline, decoded, masks, preserve, alpha)
            cleanup_masks = fine.analyze(composed)
            cleaned = fine.clean(composed, cleanup_masks, micro, spots)
            chroma = color.clean_chroma(cleaned.astype(np.float32) / 255, .5)
            gradient, dither = color.clean_gradients(chroma, .5)
            return [*masks.values(), composed, *cleanup_masks.values(), cleaned, chroma,
                    gradient, dither, color.finish(gradient, .01, 42, dither),
                    fine.overlay(image, masks, micro, spots, preserve)]

        for image in images:
            for config in ((0, 1, 0, 0), (1, .65, .55, .35), (2, 0, 1, 1)):
                actual = stages(image, *config)
                with patch.object(fine, "box_mean", legacy_box_mean), \
                     patch.object(color, "box_mean", legacy_box_mean), \
                     patch.object(fine, "_local_range_5", legacy_local_range):
                    expected = stages(image, *config)
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
