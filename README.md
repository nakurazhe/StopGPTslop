# StopGPTslop

**English** | [简体中文](README.zh-CN.md)

Removes the artifacts that GPT Image 2 leaves in its output — excessive sharpening,
random bright specks, unnatural edges, and scale-like texture patterns — while leaving
the image itself intact.

This is trained specifically for GPT Image 2. The artifacts it targets are particular to
that model, not generic AI-image artifacts, so it is unlikely to transfer usefully to
output from other generators.

```
input → FLUX.2-VAE encode → z + α · R(z) → FLUX.2-VAE decode → output
```

`R` is a small residual network (0.48M parameters) that predicts a correction to the
image's latent representation. `α` scales that correction, and it is the only knob you
need. Nothing is retrained when you change it.

## Quick start

With [uv](https://docs.astral.sh/uv/) — no manual environment setup:

```bash
git clone https://github.com/nakurazhe/StopGPTslop.git
cd StopGPTslop
uv run webui.py                 # web interface, opens in your browser
uv run modeling.py -i ./images -o ./cleaned      # batch processing
```

Or with pip:

```bash
pip install -e .
python webui.py
```

The VAE (~320 MB) is downloaded from HuggingFace on first run and cached. For offline
machines, fetch it separately and pass `--vae /path/to/FLUX.2-VAE`.

A CUDA GPU is recommended; it also runs on CPU (`--device cpu`) at roughly 10× the time.

### GPU driver and CUDA version

The pinned PyTorch build targets **CUDA 12.6**, which needs NVIDIA driver **525 or
newer**. Verify the GPU is actually being used:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False` while you do have an NVIDIA card, your driver is likely older
than the CUDA build. Edit the index URL in `pyproject.toml` to match your driver
(`cu118`, `cu124`, `cu128`, …) and re-run `uv sync`. Note that PyTorch falls back to
CPU silently in this situation, so it is worth checking once after installing.

## Web interface

```bash
uv run webui.py                 # then open http://127.0.0.1:7860
```

Drop, browse or paste one or more PNG/JPEG/WebP/BMP images into the queue. Adjust
the controls, then press **Generate**. Moving sliders, changing presets or loading
images never starts processing. Each run takes a snapshot of all settings and
processes the current queue sequentially, without concurrent GPU inference.
Completed images with identical settings are skipped; errors can be retried.
**Stop after current** lets the in-flight image finish. Clearing/removing an image
discards any late response for it; this does not interrupt an already running GPU call.

Select a queue item to inspect it with the before/after divider. **Download PNG**
and **Clear** are in the top-right corner. Downloads use the settings of the actual
result, even if controls have since changed. The Web UI comparison-export button
has been removed; CLI `--side-by-side` remains available. In 1:1 mode, upscaled
results are displayed at their output resolution.

Save, select, replace or delete named **My presets** in this browser's local storage.
Presets persist across reloads; image files, queue and results do not. Download
results before closing the page. Clear removes the queue, not your settings/presets.
The browser queue accepts up to 100 files, 40 MB per file and 512 MB total source
data; retained PNG results have a separate 512 MB limit. The API accepts up to
24 MP input / 40 MP output. These limits are safeguards, not a guarantee against OOM.
The server cache is LRU-limited to 512 MB of counted arrays/tensors and `--cache-n`
entries; active processing temporarily needs additional memory. No new dependencies.

The interface follows your system language (English / Русский / 简体中文) and colour theme, and
both can be switched from the header.

### Selective fine cleanup (experimental Web UI mode)

Enabled by default, this CPU-only finishing stage targets repeated fine patterns
and isolated tiny spots. It uses local masks to protect broad edges and directional
detail. It is heuristic: fabric, pores and other real textures may still be affected.

- **Micro-texture suppression** (default 0.55): attenuates small irregular clusters
  and repeating patterns using an edge-aware neighborhood. Detection runs on the
  remaining texture after latent cleanup; a perfectly regular grid is not required.
- **Tiny spots** (0.35): selectively reduces isolated bright/dark pixel outliers.
- **Preserve micro-texture** (0.65): higher keeps more of the original fine texture;
  lower returns less texture bypassed by the VAE, only in masked areas.
- **Show mask**: orange marks candidate processing areas. Downloaded PNGs remain
  clean, without the preview mask.

Disable **Selective fine cleanup** for the previous processing mode with the same
main strength and micro-texture setting. The new stage runs before optional SR.
Decoded images are cached so changing CPU cleanup controls with SR off does not
repeat GPU inference. No additional model weights or packages are required.
Frequency detail and CAS are attenuated in the cleanup mask to avoid amplifying
the same residual texture again. Grain remains a separate, intentional effect;
keep it at zero when evaluating fine-artifact removal.
The batch CLI is unchanged. Validate the tradeoff on 1:1 crops of your own images.

### Colour and gradients (experimental Web UI finishing)

Two independent CPU controls are **off by default (0)**, preserving the previous
output. Start at **0.50** on an affected image and inspect at 1:1 with grain at zero.

- **Colour blotches**: full-resolution colour/brightness-aware smoothing plus
  a quarter-resolution guide for patches at approximately 8–32 px radii. Opposite
  neighbours must agree before correcting a local colour/saturation outlier;
  luma is retained. Strength controls both correction and the accepted colour
  difference. Uniform coloured materials and strong colour boundaries are
  protected, but genuine isolated colour accents can still resemble blotches.
  Broad wrong lighting, strong fringes and colour moire may remain.
- **Gradient banding**: multiscale correction of a low-frequency guide up to a
  32 px radius, retaining the original high-frequency residual. Strength adapts
  the contrast/texture thresholds, allowing stronger steps and modest texture.
  RGB variance and full-window edge maxima protect contours and thin lines.
  It targets tonal steps, not jagged diagonal object edges. Strong bands can
  remain; weak genuine contours may soften at high strength.
- **Gradient dithering**: enabled by default, but active only with debanding and
  only where it changes pixels. Adds stable monochrome noise of at most half an
  8-bit code step before rounding, separately from artistic grain.

These controls work independently of selective fine cleanup. The orange preview
mask still describes **micro-texture cleanup only**, not colour or gradient masks.
The finishing stage runs after optional SR and sharpening, using float32 through
colour cleanup, debanding and grain until one final 8-bit PNG quantization. Earlier
VAE-composition/fine-cleanup/SR stages retain their existing 8-bit boundaries;
this is not end-to-end high-bit-depth processing or 16-bit export.

No additional weights, dependencies or GPU inference are required. Changing these
controls reuses the cached upstream result, including SR. CPU time and RAM grow
with resolution; no universal speed claim is made. The CLI remains unchanged.
The implementation is independent; algorithmic references are
[FFmpeg chromanr](https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_chromanr.c)
and [neo_f3kdb](https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb), not bundled
plugins or copied source. Validate on your own generated images before relying on
the heuristics for final output.

## Command line

```bash
uv run modeling.py -i INPUT -o OUTPUT             # file or directory
uv run modeling.py -i ./in -o ./out --alpha 0.25  # gentler
uv run modeling.py -i ./in -o ./out --recursive   # include subdirectories
uv run modeling.py -i ./in -o ./out --side-by-side
```

Images that already have an output are skipped, so an interrupted run can simply be
restarted. Use `--overwrite` to redo them. See `--help` for the full list of options.
Recursive batches preserve input subdirectories and exclude the output directory
if it is inside the input tree. Conflicting output names and
paths that would overwrite source files are rejected before inference. Read,
processing and save failures are reported with exit code 1; successful files remain
available. Saves are atomic so incomplete outputs are not mistaken for completed
files. EXIF orientation is applied in both CLI and Web UI; animated/multi-page
inputs are rejected rather than silently processing only their first frame.

## Strength

| α | Preset | Result |
|---|---|---|
| 0.25 | conservative | Detail essentially untouched, mild cleanup |
| **0.5** | **balanced (default)** | **Good middle ground** |
| 0.75 | strong | Noticeably cleaner, some texture loss |
| 1.0 | aggressive | Maximum cleanup, visible softening and grain |
| 0 | — | No cleanup; useful as a reference |

**How to choose:** start at 0.5. Raise it if artifacts remain. Lower it if the image
goes soft or the texture starts to smear. Among the settings that look right to you,
prefer the lowest.

One thing worth knowing: **don't judge the result by whether artifacts have vanished
completely.** Even a partial cleanup usually looks dramatically better, and pushing the
strength up to chase the last traces costs real detail. Judge by how the image looks,
not by how much was removed.

## Performance

Time depends on the device, resolution, enabled filters and cache state. The upstream
basic-pipeline timing is not a timing for the extended Web UI: a new image normally
needs one VAE encode, two decodes (baseline and correction), and CPU finishing.
Colour/gradient cleanup and optional SR add work; cached reruns can skip earlier stages.

CPU cleanup uses allocation-reduced float64 box sums and separable exact 5×5 min/max
filters. These preserve the original arithmetic, rounding and edge rules, without
new dependencies or additional GPU work. Regression tests compare masks, finishing
and pixels exactly against the previous primitives. Timing gains vary by image and
settings; they do not come from reducing resolution or weakening cleanup.

Local verification on Windows, Ryzen 5 5600G / RTX 4060 8 GB, PyTorch 2.13.0+cu126,
bf16, VAE tiling and no compilation: after warm-up, two uncached runs per version
averaged **8.80 → 7.46 s** at 1672×941 and **3.16 → 2.63 s** at 712×900.
Settings: alpha 1, micro 0.55, spots 0.35, preserve 0.65; SR, colour/gradient filters,
sharpening, grain and mask off. Times include processing and PNG encoding, not
model startup or browser transport. PNG bytes matched; peak PyTorch CUDA allocated
memory was unchanged (2692 / 1946 MiB respectively). These are two local examples,
not a general throughput guarantee or total RAM/VRAM measurement.

CLI compilation is automatic only above `--compile-min`; `--compile on` forces it.
The Web UI compiles by default and accepts `--no-compile`. Compilation requires a
compatible backend and adds warm-up time; it is not enabled in the local Windows
benchmark environment. VAE tiling remains available to limit VRAM use.

## Limitations

**Some detail cannot be recovered.** The artifacts live at the same scale as real
texture, and they overlap it spatially, so removing them always costs some genuine
detail. On a minority of images these two are coupled tightly enough that no strength
setting is satisfying: enough cleanup means visible softening, and keeping the detail
means keeping the artifacts. That is a property of the approach, not something a
setting can fix.

**The best strength varies by image.** The same change from 0.75 to 0.5 barely affects
some images and substantially changes others. There is no reliable way to pick
automatically, so for a batch of similar images it's worth testing a few values with
`--side-by-side` first.

**Every image passes through the VAE.** The effect is very small, but the output is not
a pixel-exact copy of the input in untouched areas.

## Weights

`weights/latent_residual.pt` (1.9 MB). Inspect its metadata with:

```python
import torch
print(torch.load("weights/latent_residual.pt", weights_only=False)["meta"])
```

## License

Released under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
— free to use, modify and share for noncommercial purposes. For commercial use, contact
the copyright holder. See [LICENSE](LICENSE); that file and the linked license text
govern, not this summary.

The FLUX.2-VAE model this tool builds on is published separately by Black Forest Labs
under the Apache License 2.0. It is downloaded at runtime and is not redistributed here.
