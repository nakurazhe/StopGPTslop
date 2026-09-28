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

    # Repetition increases confidence, but is not mandatory: actual generated
    # micro-patterns are often irregular or directional, unlike a perfect grid.
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
    periodic = smoothstep(0.15, 0.65, np.maximum(*repeats))
    texture = smoothstep(0.004, 0.025, np.sqrt(energy))
    # Apply protection once; multiplying it twice made the mask nearly empty
    # on real, irregular texture even with the strength slider at maximum.
    pattern = np.clip(box_mean(texture * (0.45 + 0.55 * periodic), 1) * protection, 0, 1)

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


def _surface_base(image):
    """Small edge-aware neighborhood for clustered 1–4 px texture.

    Luminance differences reduce cross-edge mixing. Unlike a Gaussian base,
    opposite sides of a contour do not contribute equally to the average.
    """
    h, w = image.shape[:2]
    padded = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode="edge")
    luma = image @ LUMA
    total = np.zeros_like(image)
    weights = np.zeros((h, w), np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            neighbor = padded[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]
            distance = (neighbor @ LUMA - luma) / 0.065
            weight = np.exp(-0.5 * distance * distance - (dx * dx + dy * dy) / 4.5)
            total += neighbor * weight[..., None]
            weights += weight
    return total / weights[..., None]


def clean(image_u8, masks, micro, spots):
    """Bounded, masked corrections; never blur the whole output image."""
    image = image_u8.astype(np.float32) / 255
    output = image.copy()
    if micro > 0:
        base = np.asarray(Image.fromarray(image_u8).filter(ImageFilter.GaussianBlur(0.65)),
                          dtype=np.float32) / 255
        # A subpixel Gaussian alone barely reaches clustered remnants. Most of
        # the correction now comes from a wider, edge-aware neighborhood.
        surface = _surface_base(image)
        correction = np.clip(0.25 * (image - base) + 0.75 * (image - surface), -0.12, 0.12)
        output -= micro * masks["pattern"][..., None] * correction
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


def structure_guide(source_u8):
    """Source-scale luma bands, not an object/colour/density classifier.

    Coherence in small neighborhoods admits crossing contours without treating
    all high-frequency energy as detail. A coarser guide rejects pixel noise.
    This is still a heuristic: structured artifacts can resemble real lines.
    """
    luma = source_u8.astype(np.float32) @ LUMA / 255
    small, broad = box_mean(luma, 1), box_mean(luma, 3)
    gx = (_shift(small, 0, 1) - _shift(small, 0, -1)) * 0.5
    gy = (_shift(small, 1, 0) - _shift(small, -1, 0)) * 0.5
    xx, yy, xy = (box_mean(v, 1) for v in (gx * gx, gy * gy, gx * gy))
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-8)
    confidence = smoothstep(0.35, 0.80, coherence)
    confidence *= smoothstep(0.002, 0.012, np.sqrt(xx + yy))
    # A monotonic ramp/step is not a thin ridge. Do not undo debanding just
    # because a band boundary also has a coherent gradient.
    ridge = np.zeros_like(luma)
    for dy, dx in ((0, 2), (2, 0), (2, 2), (2, -2)):
        a, b = small - _shift(small, dy, dx), small - _shift(small, -dy, -dx)
        ridge = np.maximum(ridge, np.maximum(np.minimum(a, b), np.minimum(-a, -b)))
    confidence *= smoothstep(0.001, 0.008, box_mean(ridge, 2))
    # Require supporting detail at the next scale, not density. Fine-only
    # irregular clusters are precisely what the existing cleaner targets.
    fine, medium = luma - small, small - broad
    scale_support = np.sqrt(box_mean(medium * medium, 3) /
                            (box_mean(fine * fine, 3) + 1e-8))
    confidence *= smoothstep(0.55, 0.90, scale_support)
    # Do not rescue isolated impulses or one-pixel checkerboards based on their
    # energy alone. Their fine detail has no supporting longer-scale structure.
    confidence = box_mean(confidence, 1)
    return (fine, medium, confidence)


def protect_structure(image, guide, amount):
    """Restore only correlated, attenuated source luma bands, after cleanup/SR.

    No chroma residual or low-frequency source image is blended back. Evaluate
    at source resolution even for 2x output; never invent higher-resolution detail.
    """
    if amount <= 0:
        return image
    fine, medium, confidence = guide
    h, w = fine.shape
    luma = image @ LUMA
    if luma.shape != fine.shape:
        luma = np.asarray(Image.fromarray(luma).resize((w, h), Image.Resampling.BOX))
    small = box_mean(luma, 1)
    bands = (luma - small, small - box_mean(luma, 3))
    correction = np.zeros_like(fine)
    for source, current in zip((fine, medium), bands):
        energy = box_mean(source * source, 3)
        current_energy = box_mean(current * current, 3)
        covariance = box_mean(source * current, 3)
        correlation = covariance / np.sqrt(energy * current_energy + 1e-12)
        gain = covariance / (energy + 1e-8)
        # Do nothing to unchanged/amplified detail, or to replaced/displaced
        # structure. Local correlation is evidence, not semantic certainty.
        eligible = confidence * smoothstep(0.25, 0.75, correlation)
        eligible *= smoothstep(0.003, 0.012, np.sqrt(energy))
        lost = np.clip(1 - gain, 0, 0.85)
        candidate = source * lost
        # Never cross the original band or add a correction against its residual.
        residual = source - current
        candidate = np.sign(candidate) * np.minimum(np.abs(candidate), np.abs(residual))
        candidate *= (candidate * residual > 0)
        correction += candidate * eligible
    correction *= float(np.clip(amount, 0, 1))
    if image.shape[:2] != fine.shape:
        correction = np.asarray(Image.fromarray(correction).resize(
            (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR))
    # Equal RGB increments keep channel differences (chroma) unchanged, including
    # near gamut limits: limit the increment instead of clipping each channel.
    correction = np.clip(correction, -np.min(image, axis=2), 1 - np.max(image, axis=2))
    return image + correction[..., None]
