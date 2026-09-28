#!/usr/bin/env python3
"""StopGPTslop — inference for the VAE+R pipeline.

    input --> FLUX2-VAE.encode --> z + alpha * R(z) --> FLUX2-VAE.decode --> output

Removes the artifacts GPT Image 2 leaves in its output: excessive sharpening, random
bright specks, unnatural edges and scale-like texture patterns. It is trained for that
model specifically and is unlikely to transfer to other image generators.

R is a small residual UNet (0.48M parameters) working in the VAE's latent space.
`alpha` scales its correction at inference time: raise it to remove more artifacts,
lower it to keep more of the original texture. No retraining is involved.

This file is both the model definition (imported by webui.py) and a batch CLI.

Usage:
    python modeling.py -i INPUT -o OUTPUT                # directory or single file
    python modeling.py -i INPUT -o OUTPUT --alpha 0.25   # gentler
    python modeling.py -i in.png -o out/ --side-by-side  # export before/after pairs

Around 0.7s per 1.5MP image on a modern GPU. The inference path is already tuned
(compiled VAE, fast image encoding) and both optimisations are on by default.
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EXTS = (".webp", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# Input dimensions must be multiples of 32; images are padded up and cropped back.
ALIGN = 32


# ------------------------------------------------------------------------------ model
def _blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())


class LatentR(nn.Module):
    """Residual UNet over the latent code. R(z) has the same shape as z and is added
    back onto it before decoding."""

    def __init__(self, ch=32):
        super().__init__()
        self.d1 = _blk(ch, 64)
        self.d2 = _blk(64, 128)
        self.pool = nn.AvgPool2d(2)
        self.mid = _blk(128, 128)
        self.u2 = _blk(128 + 128, 64)
        self.u1 = _blk(64 + 64, 64)
        self.outc = nn.Conv2d(64, ch, 3, 1, 1)

    def forward(self, z):
        a = self.d1(z)
        b = self.d2(self.pool(a))
        m = self.mid(self.pool(b))
        u = self.u2(torch.cat([F.interpolate(m, scale_factor=2, mode="nearest"), b], 1))
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2, mode="nearest"), a], 1))
        return self.outc(u)


class _RefinerBlock(nn.Module):
    """Convolution, normalisation and activation used by the latent refiner."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, tensor):
        return self.act(self.norm(self.conv(tensor)))


class RefinerLatentR(nn.Module):
    """Residual latent refiner for the 32-channel FLUX.2 profile."""

    def __init__(self, latent_channels=32):
        super().__init__()
        self.enc_high = _RefinerBlock(latent_channels, 64)
        self.enc_mid = _RefinerBlock(64, 128)
        self.bottleneck = _RefinerBlock(128, 128)
        self.dec_mid = _RefinerBlock(256, 64)
        self.dec_high = _RefinerBlock(128, 64)
        self.to_residual = nn.Conv2d(64, latent_channels, 3, padding=1)

    def forward(self, latent):
        high = self.enc_high(latent)
        mid = self.enc_mid(F.avg_pool2d(high, 2))
        low = self.bottleneck(F.avg_pool2d(mid, 2))
        up_mid = F.interpolate(low, size=mid.shape[-2:], mode="nearest")
        up_mid = self.dec_mid(torch.cat((up_mid, mid), dim=1))
        up_high = F.interpolate(up_mid, size=high.shape[-2:], mode="nearest")
        up_high = self.dec_high(torch.cat((up_high, high), dim=1))
        return self.to_residual(up_high)


class SRVGGNetCompact(nn.Module):
    """Compact Real-ESRGAN v3 network (BSD-3-Clause, xinntao/Real-ESRGAN)."""

    def __init__(self, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64)])
        for _ in range(num_conv):
            self.body.extend([nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64)])
        self.body.append(nn.Conv2d(64, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, image):
        out = image
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(
            image, scale_factor=self.upscale, mode="nearest"
        )


def load_vae(vae_id, device):
    """Load the FLUX.2 VAE without loading either residual network."""
    from diffusers import AutoencoderKLFlux2

    # Prefer an existing Hugging Face cache. Besides making normal launches faster,
    # this avoids a metadata request preventing startup when the machine is offline.
    try:
        vae = AutoencoderKLFlux2.from_pretrained(vae_id, local_files_only=True)
    except OSError:
        vae = AutoencoderKLFlux2.from_pretrained(vae_id)
    vae = vae.to(device).eval()
    vae.requires_grad_(False)
    return vae


def load_models(weight, vae_id, device):
    """Weights stay in fp32; precision is handled per-operator by autocast."""

    # Restrict deserialization to tensors and basic containers.  PyTorch checkpoints
    # loaded with weights_only=False may execute arbitrary pickle payloads.
    ck = torch.load(weight, map_location="cpu", weights_only=True)
    R = LatentR(32)
    R.load_state_dict(ck["R_ema"])
    R = R.to(device).eval()
    vae = load_vae(vae_id, device)
    for m in (R,):
        for p in m.parameters():
            p.requires_grad_(False)
    return vae, R, ck.get("meta", {})


def load_refiner_model(weight, device):
    """Load and validate the selected FLUX.2 residual checkpoint."""
    ck = torch.load(weight, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or not isinstance(ck.get("R_ema"), dict):
        raise ValueError("Refiner checkpoint must contain an R_ema state dictionary")
    meta = ck.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Refiner checkpoint metadata is missing")
    if meta.get("architecture") != "ResidualLatentNet-v1":
        raise ValueError(f"Unsupported refiner architecture: {meta.get('architecture')!r}")
    if int(meta.get("latent_channels", -1)) != 32 or meta.get("vae_type") != "flux2":
        raise ValueError("Refiner checkpoint is not the 32-channel FLUX.2 profile")
    model = RefinerLatentR(32)
    model.load_state_dict(ck["R_ema"], strict=True)
    model.requires_grad_(False)
    return model.to(device).eval(), meta


def load_realesrgan_model(weight, device):
    """Safely load the official compact weak-denoise Real-ESRGAN v3 model."""
    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("params"), dict):
        raise ValueError("Real-ESRGAN checkpoint must contain a params state dictionary")
    model = SRVGGNetCompact(num_conv=32, upscale=4)
    model.load_state_dict(checkpoint["params"], strict=True)
    model.requires_grad_(False)
    dtype = torch.float16 if device.split(":")[0] == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval(), dtype


# -------------------------------------------------------------------------- inference
def build_fns(vae, compile_on):
    """Return (encode_fn, decode_fn), optionally compiled.

    dynamic=True is required: inputs arrive in many resolutions, and a static graph
    would recompile for each new one.
    """
    if not compile_on:
        return (lambda a: vae.encode(a).latent_dist.mode()), (lambda a: vae.decode(a).sample)
    return (torch.compile(lambda a: vae.encode(a).latent_dist.mode(), dynamic=True),
            torch.compile(lambda a: vae.decode(a).sample, dynamic=True))


@torch.no_grad()
def restore(arr_u8, enc_fn, dec_fn, R, alpha, device, dtype):
    """[H,W,3] uint8 -> [H,W,3] uint8, preserving the input dimensions."""
    H, W = arr_u8.shape[:2]
    x = torch.from_numpy(arr_u8.astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    ph, pw = (-H) % ALIGN, (-W) % ALIGN
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
        z = enc_fn(x * 2 - 1)
        if alpha != 0.0:
            z = z + alpha * R(z)
        y = dec_fn(z)
    y = ((y.float().clamp(-1, 1) + 1) / 2)[0, :, :H, :W]
    return (y.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def save_image(path, arr_u8, quality=95, webp_method=0):
    """`webp_method` only controls how hard the compressor searches (speed vs file
    size); `quality` is the image-quality knob. Method 0 is ~3x faster to encode."""
    im = Image.fromarray(arr_u8)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".webp":
        im.save(path, quality=quality, method=webp_method)
    elif ext in (".jpg", ".jpeg"):
        im.save(path, quality=quality)
    else:
        im.save(path)


# --------------------------------------------------------------------------- CLI
def collect_inputs(path, recursive):
    if os.path.isfile(path):
        return [path]
    pat = "**/*" if recursive else "*"
    fs = glob.glob(os.path.join(path, pat), recursive=recursive)
    return sorted(f for f in fs if os.path.splitext(f)[1].lower() in EXTS)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog="Strength presets (same as the web UI):\n"
               "  0.25 conservative | 0.5 balanced (default) | 0.75 strong | 1.0 aggressive\n"
               "Raise it if artifacts remain; lower it if the image goes soft or texture\n"
               "smears. Among the settings that look good, prefer the lowest.",
    )
    p.add_argument("-i", "--input", required=True, help="input image or directory")
    p.add_argument("-o", "--output", required=True, help="output directory")
    p.add_argument("-a", "--alpha", type=float, default=0.5,
                   help="cleanup strength (default 0.5). Higher removes more artifacts but "
                        "softens real texture; lower keeps more detail but leaves more behind. "
                        "0 disables cleanup entirely (useful as a reference)")
    p.add_argument("-r", "--recursive", action="store_true", help="recurse into subdirectories")
    p.add_argument("--weight", default=os.path.join(HERE, "weights", "latent_residual.pt"),
                   help="path to the R weights")
    p.add_argument("--vae", default="black-forest-labs/FLUX.2-VAE",
                   help="HuggingFace model id or a local directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp32", action="store_true",
                   help="run in fp32 instead of bf16: slower and needs more memory, with no "
                        "visible difference")
    p.add_argument("--format", default="keep", choices=("keep", "webp", "png", "jpg"),
                   help="output format; keep reuses the input extension (default)")
    p.add_argument("--quality", type=int, default=95, help="webp/jpg quality")
    p.add_argument("--webp-method", type=int, default=0, choices=range(7), metavar="0-6",
                   help="webp compression effort; does not affect image quality. Higher is "
                        "slower for marginally smaller files (default 0)")
    p.add_argument("--compile", choices=("auto", "on", "off"), default="auto",
                   help="compile the VAE for ~1.5x throughput. A one-off compilation cost is "
                        "paid on the first image, so auto enables it only for larger batches")
    p.add_argument("--compile-min", type=int, default=48,
                   help="batch size at which auto turns compilation on (default 48). For a "
                        "long-running service, pass --compile on instead")
    p.add_argument("--side-by-side", action="store_true",
                   help="write [input | output] pairs instead of the output alone")
    p.add_argument("--overwrite", action="store_true",
                   help="reprocess images that already have an output (default: skip)")
    p.add_argument("--tile", action="store_true",
                   help="encode/decode the VAE in tiles; use this if you run out of GPU "
                        "memory on very large images. Slightly slower")
    p.add_argument("--workers", type=int, default=3,
                   help="background threads for reading and encoding, 0 to run synchronously")
    args = p.parse_args()

    files = collect_inputs(args.input, args.recursive)
    if not files:
        sys.exit(f"No images found in {args.input}")
    os.makedirs(args.output, exist_ok=True)

    def dst_of(f):
        stem = os.path.splitext(os.path.basename(f))[0]
        ext = os.path.splitext(f)[1].lower() if args.format == "keep" else "." + args.format
        if ext not in EXTS:
            ext = ".webp"
        return os.path.join(args.output, stem + ext)

    todo = files if args.overwrite else [f for f in files if not os.path.exists(dst_of(f))]
    if len(todo) < len(files):
        print(f"Skipping {len(files)-len(todo)} image(s) that already have output "
              f"(use --overwrite to redo them)", flush=True)
    if not todo:
        print(f"Nothing to do. Output directory: {args.output}", flush=True)
        return

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    do_compile = (args.compile == "on"
                  or (args.compile == "auto" and len(todo) >= args.compile_min))
    t0 = time.time()
    vae, R, meta = load_models(args.weight, args.vae, args.device)
    if args.tile:
        try:
            vae.enable_tiling()
        except Exception as e:
            print(f"! This VAE does not support tiling: {e}", flush=True)
    enc_fn, dec_fn = build_fns(vae, do_compile)
    print(f"Ready in {time.time()-t0:.1f}s | alpha={args.alpha} | {args.device} "
          f"{'fp32' if args.fp32 else 'bf16'} | compiled={do_compile} | "
          f"{len(todo)} image(s) to process", flush=True)
    if do_compile:
        print("  The first image takes noticeably longer while the model is compiled",
              flush=True)
    elif args.compile == "auto":
        print(f"  Not compiling: {len(todo)} image(s) is below the --compile-min "
              f"threshold of {args.compile_min}. Pass --compile on to force it", flush=True)

    pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers else None

    def read(f):
        return np.asarray(Image.open(f).convert("RGB"))

    def write(dst, src_u8, out_u8):
        if args.side_by_side:
            gap = np.full((out_u8.shape[0], 6, 3), 255, np.uint8)
            out_u8 = np.concatenate([src_u8, gap, out_u8], 1)
        save_image(dst, out_u8, args.quality, args.webp_method)

    t_start = time.time()
    done_px = 0
    pend = []
    nxt = pool.submit(read, todo[0]) if pool else None
    failed = []
    for i, f in enumerate(todo):
        try:
            arr = nxt.result() if pool else read(f)
        except Exception as e:
            print(f"! Could not read {os.path.basename(f)}: {e}", flush=True)
            failed.append(f)
            nxt = pool.submit(read, todo[i + 1]) if (pool and i + 1 < len(todo)) else None
            continue
        if pool:
            nxt = pool.submit(read, todo[i + 1]) if i + 1 < len(todo) else None
        H, W = arr.shape[:2]
        t1 = time.time()
        try:
            out = restore(arr, enc_fn, dec_fn, R, args.alpha, args.device, dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"! Out of GPU memory on {os.path.basename(f)} ({W}x{H}); "
                  f"try --tile", flush=True)
            failed.append(f)
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t1
        dst = dst_of(f)
        if pool:
            pend = [q for q in pend if not q.done()]
            pend.append(pool.submit(write, dst, arr, out))
        else:
            write(dst, arr, out)
        done_px += H * W
        el = time.time() - t_start
        eta = el / (i + 1) * (len(todo) - i - 1)
        print(f"[{i+1}/{len(todo)}] {os.path.basename(f)[:44]:46s} {W}x{H}  "
              f"{dt:5.2f}s ({H*W/1e6/dt:.2f} MP/s)  ETA {eta/60:4.1f}min", flush=True)

    if pool:
        for q in pend:
            q.result()
        pool.shutdown(wait=True)
    el = time.time() - t_start
    print(f"Done: {len(todo)-len(failed)}/{len(todo)} image(s), {done_px/1e6:.1f}MP in "
          f"{el/60:.1f}min ({el/max(len(todo),1):.2f}s per image, "
          f"{done_px/1e6/max(el,1e-9):.2f} MP/s)", flush=True)
    if failed:
        print(f"Failed on {len(failed)}: {[os.path.basename(f) for f in failed[:5]]}", flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
