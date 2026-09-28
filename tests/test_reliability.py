import base64
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import modeling
import webui


class ReliabilityTests(unittest.TestCase):
    def test_exif_orientation_and_animation(self):
        buf = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (40, 20), "red").save(buf, format="JPEG", exif=exif)
        self.assertEqual(modeling.read_image(io.BytesIO(buf.getvalue())).shape, (40, 20, 3))
        gif = io.BytesIO()
        Image.new("RGB", (20, 20), "red").save(gif, format="GIF", save_all=True,
            append_images=[Image.new("RGB", (20, 20), "blue")])
        with self.assertRaisesRegex(ValueError, "Animated"):
            modeling.read_image(io.BytesIO(gif.getvalue()))

    def test_small_padding_and_restore_cpu(self):
        for h, w in [(1, 1), (3, 5), (32, 32), (65, 79)]:
            arr = np.full((h, w, 3), 127, np.uint8)
            for alpha in (0, .5):
                out = modeling.restore(arr, lambda x: x, lambda x: x,
                    lambda x: torch.zeros_like(x), alpha, "cpu", torch.float32)
                np.testing.assert_array_equal(out, arr)

    def test_failed_atomic_save_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"output.png"
            Image.new("RGB", (32, 32), "red").save(path)
            previous = path.read_bytes()
            with patch.object(Image.Image, "save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    modeling.save_image(path, np.zeros((32, 32, 3), np.uint8))
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["output.png"])

    def test_cache_memory_budget(self):
        old_cache, old_order = webui.CACHE.copy(), webui.CACHE_ORDER[:]
        try:
            webui.CACHE.clear()
            webui.CACHE.update(a={"rgb": np.zeros((10, 10), np.float32)}, b={"rgb": np.zeros((10, 10), np.float32)})
            webui.CACHE_ORDER[:] = ["a", "b"]
            with patch.object(webui, "CACHE_BYTES", 500):
                webui._trim_cache()
            self.assertEqual(list(webui.CACHE), ["b"])
        finally:
            webui.CACHE.clear()
            webui.CACHE.update(old_cache)
            webui.CACHE_ORDER[:] = old_order

    def test_bad_base64_and_image_size(self):
        with self.assertRaises(ValueError):
            webui.process("not valid base64!", 1, None)
        buf = io.BytesIO()
        Image.new("RGB", (20, 20)).save(buf, format="PNG")
        with patch.object(webui, "MAX_IMAGE_PIXELS", 100):
            with self.assertRaisesRegex(ValueError, "too large"):
                webui.process(base64.b64encode(buf.getvalue()).decode(), 1, None)

    def run_cli(self, source, output, *extra, save=None):
        with patch("sys.argv", ["modeling.py", "-i", str(source), "-o", str(output), "--device", "cpu", *extra]), \
             patch.object(modeling, "load_models", return_value=(None, None, {})), \
             patch.object(modeling, "build_fns", return_value=(None, None)), \
             patch.object(modeling, "restore", side_effect=lambda arr, *args: arr), \
             patch.object(modeling, "save_image", side_effect=save or modeling.save_image), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            modeling.main()

    def test_recursive_paths_and_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                (root/"input"/name).mkdir(parents=True)
                Image.new("RGB", (32, 32)).save(root/"input"/name/"same.png")
            self.run_cli(root/"input", root/"output", "--recursive", "--workers", "0")
            self.assertTrue((root/"output/a/same.png").is_file())
            self.assertTrue((root/"output/b/same.png").is_file())
            self.run_cli(root/"input", root/"output", "--recursive", save=lambda *a: self.fail("must skip existing"))

    def test_conversion_collision_and_source_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 32)).save(root/"same.png")
            Image.new("RGB", (32, 32)).save(root/"same.jpg")
            for source, output, args in [(root, root/"out", ["--format", "png"]),
                                          (root/"same.png", root, ["--overwrite"])]:
                with self.assertRaises(SystemExit) as raised:
                    self.run_cli(source, output, *args)
                self.assertEqual(raised.exception.code, 2)

    def test_recursive_run_excludes_its_output_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 32)).save(root/"one.png")
            self.run_cli(root, root/"out", "--recursive", "--workers", "0")
            self.run_cli(root, root/"out", "--recursive", "--workers", "0")
            self.assertTrue((root/"out/one.png").exists())
            self.assertFalse((root/"out/out").exists())

    def test_failed_writes_and_corrupt_inputs_report_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 32)).save(root/"one.png")
            for workers in ("0", "2"):
                with self.assertRaises(SystemExit) as raised:
                    self.run_cli(root/"one.png", root/"output", "--workers", workers,
                                 save=lambda *a: (_ for _ in ()).throw(OSError("disk full")))
                self.assertEqual(raised.exception.code, 1)
            (root/"bad.png").write_bytes(b"not an image")
            with self.assertRaises(SystemExit) as raised:
                self.run_cli(root, root/"output", "--workers", "2")
            self.assertEqual(raised.exception.code, 1)
            self.assertTrue((root/"output/one.png").exists())


if __name__ == "__main__":
    unittest.main()
