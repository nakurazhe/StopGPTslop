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

Drop, browse or paste an image, then drag the divider to compare before and after.
Move the strength slider and the result updates immediately. You can download either
the result alone or a side-by-side comparison.

The interface follows your system language (English / 简体中文) and colour theme, and
both can be switched from the header.

### Selective fine cleanup (experimental Web UI mode)

Enabled by default, this CPU-only finishing stage targets repeated fine patterns
and isolated tiny spots. It uses local masks to protect broad edges and directional
detail. It is heuristic: fabric, pores and other real textures may still be affected.

- **Micro-texture suppression** (default 0.55): attenuates short repeating patterns.
- **Tiny spots** (0.35): selectively reduces isolated bright/dark pixel outliers.
- **Preserve micro-texture** (0.65): higher keeps more of the original fine texture;
  lower returns less texture bypassed by the VAE, only in masked areas.
- **Show mask**: orange marks candidate processing areas. Downloaded PNGs remain
  clean; comparison export is disabled while the mask is displayed.

Disable **Selective fine cleanup** for the previous processing mode with the same
main strength and micro-texture setting. The new stage runs before optional SR.
Decoded images are cached so changing CPU cleanup controls with SR off does not
repeat GPU inference. No additional model weights or packages are required.
The batch CLI is unchanged. Validate the tradeoff on 1:1 crops of your own images.

## Command line

```bash
uv run modeling.py -i INPUT -o OUTPUT             # file or directory
uv run modeling.py -i ./in -o ./out --alpha 0.25  # gentler
uv run modeling.py -i ./in -o ./out --recursive   # include subdirectories
uv run modeling.py -i ./in -o ./out --side-by-side
```

Images that already have an output are skipped, so an interrupted run can simply be
restarted. Use `--overwrite` to redo them. See `--help` for the full list of options.

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

About **0.7 s per 1.5 MP image** on a recent GPU, using roughly 3 GB of VRAM. The VAE
accounts for essentially all of that time; the correction network itself is free.

The inference path is already tuned — the VAE is compiled and image encoding uses a
fast preset — and both are enabled by default. For batches the CLI enables compilation
automatically; for a long-running server pass `--compile on`.

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
