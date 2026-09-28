"""CPU-only, conservative fine-artifact cleanup at the source resolution.

The masks are heuristics, not semantic artifact detectors. Directional detail and
large edges are protected; woven fabric can still resemble a repeated artifact.
"""
import numpy as np
from PIL import Image, ImageFilter

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def smoothstep(low, high, value):
    x = np.clip((value - low) / (high - low), 0, 1)
    return x * x * (3 - 2 * x)


def box_mean(field, radius):
    """Float-preserving local mean, including signed fields and RGB images."""
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


def _shift(field, dy, dx):
    pad = max(abs(dy), abs(dx))
    p = np.pad(field, pad, mode="edge")
    h, w = field.shape
    return p[pad + dy:pad + dy + h, pad + dx:pad + dx + w]


def analyze(image_u8):
    """Return soft masks for repeated fine texture and isolated tiny outliers."""
    image = image_u8.astype(np.float32) / 255
    luma = image @ LUMA
    base = box_mean(luma, 1)
    fine = luma - base
    gx = (_shift(luma, 0, 1) - _shift(luma, 0, -1)) * 0.5
    gy = (_shift(luma, 1, 0) - _shift(luma, -1, 0)) * 0.5
    xx, yy, xy = box_mean(gx * gx, 3), box_mean(gy * gy, 3), box_mean(gx * gy, 3)
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2) / (xx + yy + 1e-8)
    broad = box_mean(luma, 3)
    bx = (_shift(broad, 0, 1) - _shift(broad, 0, -1)) * 0.5
    by = (_shift(broad, 1, 0) - _shift(broad, -1, 0)) * 0.5
    edge = smoothstep(0.012, 0.045, np.hypot(bx, by))
    protection = (1 - edge) * (1 - smoothstep(0.45, 0.9, coherence))
    # Crossings and letter corners are not directional. Protect their strong
    # local contrast as well, where a structure tensor alone is ambiguous.
    gray = Image.fromarray((luma * 255).round().astype(np.uint8))
    local_range = (np.asarray(gray.filter(ImageFilter.MaxFilter(5)), dtype=np.float32)
                   - np.asarray(gray.filter(ImageFilter.MinFilter(5)), dtype=np.float32)) / 255
    protection *= 1 - smoothstep(0.22, 0.40, local_range)

    # Positive correlation at short lags in BOTH axes favors repeated speckle
    # over an isolated line. It cannot distinguish artificial grids from fabric.
    energy = box_mean(fine * fine, 3)
    repeats = []
    for axis in (0, 1):
        best = np.zeros_like(luma)
        for lag in (2, 3, 4, 6):
            shifted = _shift(fine, lag if axis == 0 else 0, lag if axis == 1 else 0)
            correlation = box_mean(fine * shifted, 3)
            denominator = np.sqrt(energy * box_mean(shifted * shifted, 3)) + 1e-8
            best = np.maximum(best, correlation / denominator)
        repeats.append(best)
    periodic = smoothstep(0.15, 0.65, np.minimum(*repeats))
    texture = smoothstep(0.006, 0.035, np.sqrt(energy))
    pattern = np.clip(box_mean(texture * periodic * protection, 1) * protection, 0, 1)

    # Require disagreement with every neighbor, with the same sign. Averaging
    # opposing pairs would incorrectly classify pixels next to letter corners.
    differences = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        differences.extend((luma - _shift(luma, dy, dx), luma - _shift(luma, -dy, -dx)))
    bright = np.minimum.reduce(differences)
    dark = np.minimum.reduce([-d for d in differences])
    dots = smoothstep(0.025, 0.10, np.maximum(bright, dark)) * protection
    return {"pattern": pattern, "dots": dots}


def compose(source_u8, baseline, decoded, masks, preserve, alpha):
    """Keep the learned delta; selectively reduce high-frequency VAE bypass.

    baseline/decoded are float RGB in [0, 1]. preserve=1 keeps the original
    preservation path. Alpha zero is an exact bypass of this stage.
    """
    source = source_u8.astype(np.float32) / 255
    bypass = source - baseline
    high = bypass - box_mean(bypass, 2)
    mask = np.maximum(masks["pattern"], masks["dots"])
    amount = (1 - preserve) * min(abs(alpha), 1.0)
    result = source + (decoded - baseline) - amount * mask[..., None] * high
    return np.clip(result * 255, 0, 255).round().astype(np.uint8)


def clean(image_u8, masks, micro, spots):
    """Bounded, masked corrections; never blur the whole output image."""
    image = image_u8.astype(np.float32) / 255
    output = image.copy()
    if micro > 0:
        base = np.asarray(Image.fromarray(image_u8).filter(ImageFilter.GaussianBlur(0.65)),
                          dtype=np.float32) / 255
        correction = np.clip(image - base, -0.10, 0.10)
        output -= micro * 0.75 * masks["pattern"][..., None] * correction
    if spots > 0:
        median = np.asarray(Image.fromarray(image_u8).filter(ImageFilter.MedianFilter(3)),
                            dtype=np.float32) / 255
        correction = np.clip(image - median, -0.16, 0.16)
        output -= spots * masks["dots"][..., None] * correction
    return np.clip(output * 255, 0, 255).round().astype(np.uint8)


def overlay(image_u8, masks, micro, spots, preserve):
    """Orange indicates where the enabled fine-cleanup stages can intervene."""
    mask = np.maximum(micro * masks["pattern"], spots * masks["dots"])
    mask = np.maximum(mask, (1 - preserve) * np.maximum(masks["pattern"], masks["dots"]))
    tint = np.array([255, 96, 24], dtype=np.float32)
    alpha = (0.75 * mask)[..., None]
    return (image_u8 * (1 - alpha) + tint * alpha).round().astype(np.uint8)
