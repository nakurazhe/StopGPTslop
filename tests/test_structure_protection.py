import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

import fine_cleanup as fine
import webui
import test_fine_cleanup


class StructureTests(unittest.TestCase):
    def line_image(self):
        y, x = np.mgrid[:128, :160]
        field = 0.45 + 0.08 * np.sin(x * .65 + .008 * y * y)
        field += 0.06 * np.cos(y * .8 - x * .3)
        return np.repeat(np.round(field[..., None] * 255).astype(np.uint8), 3, axis=2)

    def test_zero_and_unchanged_are_exact(self):
        source = self.line_image()
        image = source.astype(np.float32) / 255
        guide = fine.structure_guide(source)
        self.assertIs(fine.protect_structure(image, guide, 0), image)
        np.testing.assert_allclose(fine.protect_structure(image, guide, 1), image, atol=2e-6)

    def test_attenuated_crossing_lines_recover_without_overshoot(self):
        source = self.line_image()
        image = np.asarray(Image.fromarray(source).filter(ImageFilter.GaussianBlur(.9)), dtype=np.float32) / 255
        out = fine.protect_structure(image, fine.structure_guide(source), 1)
        ref = source.astype(np.float32) / 255
        self.assertLess(np.mean((out - ref) ** 2), np.mean((image - ref) ** 2) * .90)
        self.assertGreaterEqual(out.min(), 0)
        self.assertLessEqual(out.max(), 1)

    def test_flat_color_correction_does_not_return_source(self):
        source = np.full((64, 64, 3), [90, 140, 60], dtype=np.uint8)
        image = np.full(source.shape, [.4, .5, .3], dtype=np.float32)
        np.testing.assert_array_equal(fine.protect_structure(image, fine.structure_guide(source), 1), image)

    def test_monotonic_steps_do_not_undo_debanding(self):
        x = np.arange(160, dtype=np.float32)
        ramp = .2 + .003 * x
        banded = np.round(ramp * 32) / 32
        source = np.tile((banded * 255).round().astype(np.uint8)[None, :, None], (96, 1, 3))
        image = np.tile(ramp[None, :, None], (96, 1, 3))
        out = fine.protect_structure(image, fine.structure_guide(source), 1)
        np.testing.assert_array_equal(out[:, 10:-10], image[:, 10:-10])

    def test_dots_and_checkerboard_are_not_restored(self):
        y, x = np.mgrid[:96, :96]
        for field in (128 + 20 * ((x + y) % 2), 128 + 30 * ((x % 12 == 0) & (y % 12 == 0))):
            source = np.repeat(field[..., None], 3, axis=2).astype(np.uint8)
            image = np.asarray(Image.fromarray(source).filter(ImageFilter.GaussianBlur(.8)), dtype=np.float32) / 255
            out = fine.protect_structure(image, fine.structure_guide(source), 1)
            self.assertLess(np.mean(np.abs(out - image)), .1 / 255)

    def test_irregular_micro_cleanup_is_not_cancelled(self):
        y, x = np.mgrid[:96, :96]
        field = np.round(128 + 15 * np.sin(1.7*x + .011*y*y) * np.sin(1.3*y + .013*x*x))
        source = np.repeat(field[..., None], 3, axis=2).astype(np.uint8)
        image = fine.clean(source, fine.analyze(source), 1, 0).astype(np.float32) / 255
        out = fine.protect_structure(image, fine.structure_guide(source), 1)
        original_std = source.std() / 255
        self.assertGreater(original_std - out.std(), .85 * (original_std - image.std()))

    def test_chroma_and_gamut_are_preserved(self):
        source = self.line_image()
        image = np.asarray(Image.fromarray(source).filter(ImageFilter.GaussianBlur(1)), dtype=np.float32) / 255
        image[..., 0] += .4
        image[..., 2] -= .3
        image = np.clip(image, 0, 1)
        out = fine.protect_structure(image, fine.structure_guide(source), 1)
        np.testing.assert_allclose(out[..., 0] - out[..., 1], image[..., 0] - image[..., 1], atol=1e-7)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreaterEqual(out.min(), 0)
        self.assertLessEqual(out.max(), 1)

    def test_small_and_upscaled_images(self):
        for shape in ((1, 1, 3), (2, 5, 3), (64, 79, 3)):
            source = np.full(shape, 128, np.uint8)
            image = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1).astype(np.float32) / 255
            np.testing.assert_array_equal(fine.protect_structure(image, fine.structure_guide(source), 1), image)

    def test_nonflat_2x_and_spatial_locality(self):
        source = self.line_image()
        source[:, :40] = 128
        blurred = Image.fromarray(source).filter(ImageFilter.GaussianBlur(.9))
        image = np.asarray(blurred.resize((320, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        out = fine.protect_structure(image, fine.structure_guide(source), .8)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out[:, :40], image[:, :40])
        self.assertGreater(np.max(np.abs(out - image)), .001)
        self.assertTrue(np.isfinite(out).all())

    def test_amplified_lines_are_not_sharpened_again(self):
        source = self.line_image()
        image = source.astype(np.float32) / 255
        luma = image @ fine.LUMA
        image += .5 * (luma - fine.box_mean(luma, 3))[..., None]
        out = fine.protect_structure(image, fine.structure_guide(source), 1)
        self.assertLess(np.max(np.abs(out - image)), 1 / 255)


class StructurePipelineTests(unittest.TestCase):
    setUp = test_fine_cleanup.PipelineTests.setUp
    tearDown = test_fine_cleanup.PipelineTests.tearDown
    def test_protection_reuses_gpu_and_disabled_result(self):
        with patch.object(webui, '_encode_z', return_value=(None, None, 64, 64)), \
             patch.object(webui, '_decode', return_value=self.rgb.astype(np.float32) / 255) as decode, \
             patch.object(webui, '_realesrgan', side_effect=lambda arr, scale, blend: arr) as sr:
            for color in (0, .5):
                args = dict(fine_enabled=True, micro=.8, sr_mode='1x', chroma=color)
                before = webui.process(self.payload, 1, None, **args)
                calls = (decode.call_count, sr.call_count)
                webui.process(self.payload, 1, None, structure=.8, **args)
                after = webui.process(self.payload, 1, None, structure=0, **args)
                self.assertEqual(before['result'], after['result'])
                self.assertEqual(calls, (decode.call_count, sr.call_count))
            with self.assertRaises(ValueError):
                webui.process(self.payload, 1, None, structure=float('nan'))


if __name__ == '__main__':
    unittest.main()
