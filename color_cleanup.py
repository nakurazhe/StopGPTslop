"""Lightweight, CPU-only colour and gradient finishing (no model weights).

Independent implementation of local colour-distance smoothing and thresholded
multiscale gradient smoothing; not copied from FFmpeg or f3kdb. These are
heuristics, not semantic detectors of wrong colours or missing image content.
All processing stays float32 until the final, optionally dithered quantization.
"""
import numpy as np
from PIL import Image

from fine_cleanup import LUMA, box_mean, smoothstep


def _neighbors(image, radius):
    h, w = image.shape[:2]
    pads = [(radius, radius), (radius, radius)] + [(0, 0)] * (image.ndim - 2)
    padded = np.pad(image, pads, mode="edge")
    return lambda dy, dx: padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]


def _max_rgb(image):
    return np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])


def _box_max(field, radius):
    """Separable exact maximum with logarithmic passes, no large window tensor."""
    result = field
    width = 2 * radius + 1
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        result = np.pad(result, padding, mode="edge")
        span = 1
        while span < width:
            step = min(span, width - span)
            a, b = [slice(None)] * 2, [slice(None)] * 2
            a[axis], b[axis] = slice(step, None), slice(None, -step)
            result = np.maximum(result[tuple(a)], result[tuple(b)])
            span += step
    return result


def _wide_chroma_correction(image, amount):
    """Estimate 8–32 px chroma patches on a quarter-size guide.

    Opposite donors must agree with each other, not just with the centre. A
    persistent colour boundary has disagreeing donors and is not a blotch.
    A genuine isolated colour accent can still look like a blotch.
    """
    h, w = image.shape[:2]
    if min(h, w) < 16:
        return np.zeros((h, w, 2), np.float32)
    # Area reduction has a well-defined pixel-centre mapping for interpolation.
    small = np.stack([np.asarray(Image.fromarray(image[..., c]).resize(
        ((w + 3) // 4, (h + 3) // 4), Image.Resampling.BOX)) for c in range(3)], axis=2)
    y = small @ LUMA
    chroma = small[..., (0, 2)] - y[..., None]
    sample_c, sample_y = _neighbors(chroma, 8), _neighbors(y, 8)
    correction = np.zeros_like(chroma)
    agreement = 0.018 + 0.022 * amount
    reach = 0.045 + 0.085 * amount
    for radius in (2, 4, 8):
        total = np.zeros_like(chroma)
        weights = np.zeros_like(y)
        for dy, dx in ((0, radius), (radius, 0), (radius, radius), (radius, -radius)):
            a, b = sample_c(dy, dx), sample_c(-dy, -dx)
            delta = (a + b) * 0.5 - chroma
            pair_diff = np.maximum(np.abs(a[..., 0] - b[..., 0]),
                                   np.abs(a[..., 1] - b[..., 1]))
            distance = np.maximum(np.abs(delta[..., 0]), np.abs(delta[..., 1]))
            light_diff = np.maximum(np.abs(sample_y(dy, dx) - y),
                                    np.abs(sample_y(-dy, -dx) - y))
            weight = 1 - smoothstep(agreement, 2 * agreement, pair_diff)
            weight *= 1 - smoothstep(reach, 1.5 * reach, distance)
            weight *= 1 - smoothstep(0.035 + 0.025 * amount, 0.10, light_diff)
            total += weight[..., None] * delta
            weights += weight
        candidate = total / np.maximum(weights[..., None], 1e-6)
        confidence = smoothstep(1.5, 3.5, weights)
        correction += confidence[..., None] * (candidate - correction)
    up = np.stack([np.asarray(Image.fromarray(correction[..., c]).resize(
        (w, h), Image.Resampling.BILINEAR)) for c in range(2)], axis=2)
    # Keep fine colour markings and strong contours out of the coarse correction.
    full_chroma = image[..., (0, 2)] - (image @ LUMA)[..., None]
    fine = full_chroma - box_mean(full_chroma, 2)
    energy = np.sqrt(box_mean(np.maximum(fine[..., 0] ** 2, fine[..., 1] ** 2), 1))
    protection = 1 - smoothstep(0.018, 0.060, energy)
    return 0.8 * amount * protection[..., None] * np.clip(up, -reach, reach)


def clean_chroma(image, amount):
    """Smooth colour differences only; retain luma and reject colour boundaries."""
    if amount == 0:
        return image
    luma = image @ LUMA
    # Two opponent channels, at full resolution (no chroma subsampling).
    chroma = image[..., (0, 2)] - luma[..., None]
    total = chroma.copy()
    weights = np.ones(luma.shape, np.float32)
    sample_c, sample_y = _neighbors(chroma, 4), _neighbors(luma, 4)
    # Fixed 24 samples keep cost bounded, even for the wider neighbourhood.
    for radius in (1, 2, 4):
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            neighbor = sample_c(dy * radius, dx * radius)
            delta_y = sample_y(dy * radius, dx * radius) - luma
            delta_c = neighbor - chroma
            distance = ((delta_y / 0.035) ** 2 + (delta_c[..., 0] / 0.055) ** 2
                        + (delta_c[..., 1] / 0.055) ** 2)
            weight = np.exp(-0.5 * distance) / radius
            total += neighbor * weight[..., None]
            weights += weight
    correction = amount * np.clip(total / weights[..., None] - chroma, -0.06, 0.06)
    correction += _wide_chroma_correction(image, amount)
    delta = np.empty_like(image)
    delta[..., 0] = correction[..., 0]
    delta[..., 2] = correction[..., 1]
    delta[..., 1] = -(LUMA[0] * delta[..., 0] + LUMA[2] * delta[..., 2]) / LUMA[1]
    # Scale the whole correction at gamut limits, rather than clipping channels
    # independently and changing the intended luma/hue relationship.
    limit = np.where(delta > 0, (1 - image) / np.maximum(delta, 1e-8),
                     image / np.maximum(-delta, 1e-8))
    limit = np.where(np.abs(delta) < 1e-8, 1, limit)
    scale = np.minimum(1, -_max_rgb(-limit))
    return np.clip(image + delta * scale[..., None], 0, 1)


def clean_gradients(image, amount):
    """Smooth low-contrast plateaus, protecting detail in all RGB channels.

    This is debanding, not antialiasing of diagonal object edges. Weak real
    contours can still resemble bands; strength zero is the exact bypass.
    """
    mask = np.zeros(image.shape[:2], np.float32)
    if amount == 0:
        return image, mask
    threshold = (2 + 14 * amount) / 255
    # Reject local texture, including isoluminant coloured texture.
    original_sample = _neighbors(image, 1)
    cross = [original_sample(dy, dx) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))]
    local_min, local_max = np.minimum.reduce(cross), np.maximum.reduce(cross)
    # Separate alternating pixel-scale detail without putting a weak step edge
    # into the bypass. A plain blurred guide would restore part of every band
    # through its high-pass residual, leaving visible contour lines behind.
    residual = 0.5 * (np.maximum(image - local_max, 0) + np.minimum(image - local_min, 0))
    signal = image - residual
    high = _max_rgb(np.abs(image - box_mean(image, 1)))
    texture = np.sqrt(box_mean(high * high, 2))
    quiet = 1 - smoothstep((2 + 2 * amount) / 255, (5 + 3 * amount) / 255, texture)
    correction = np.zeros_like(image)
    sample = _neighbors(signal, 32)
    squared = image * image
    edges = np.maximum(_max_rgb(np.abs(original_sample(0, 1) - image)),
                       _max_rgb(np.abs(original_sample(1, 0) - image)))
    for radius in (2, 4, 8, 16, 32):
        largest = np.zeros_like(mask)
        for dy, dx in ((0, radius), (0, -radius), (radius, 0), (-radius, 0),
                       (radius, radius), (-radius, -radius),
                       (-radius, radius), (radius, -radius)):
            difference = _max_rgb(np.abs(sample(dy, dx) - signal))
            largest = np.maximum(largest, difference)
        base = box_mean(signal, radius)
        # Keep a two-pixel guard margin around the guide's averaging window.
        mean = box_mean(image, radius + 2)
        deviation = np.sqrt(np.maximum(_max_rgb(box_mean(squared, radius + 2) - mean * mean), 0))
        # Sparse reference samples can jump over a thin line inside the window.
        # Full-window variance prevents its colour leaking into nearby plateaus.
        eligible = quiet * (1 - smoothstep(threshold, threshold * 1.5, largest))
        eligible *= 1 - smoothstep(threshold * 0.6, threshold, deviation)
        # Variance dilutes narrow lines in large windows; an exact maximum of
        # local edge strengths protects them even when all sparse donors miss.
        eligible *= 1 - smoothstep(threshold * 1.5, threshold * 2,
                                   _box_max(edges, radius + 2))
        # Correct the low-frequency guide only; add its delta to the untouched
        # full-resolution image, preserving the original high-frequency residual.
        candidate = np.clip(base - signal, -threshold, threshold)
        # A wider valid neighbourhood replaces the narrow estimate, rather
        # than stacking smoothing passes and gradually eroding contours.
        correction += eligible[..., None] * (candidate - correction)
        mask = np.maximum(mask, eligible)
    output = np.clip(image + amount * correction, 0, 1)
    # Dither only where debanding actually changed a value; flat fields stay flat.
    changed = smoothstep(0, 0.5 / 255, _max_rgb(np.abs(output - image)))
    return output, changed * mask


def finish(image, grain, seed, dither_mask=None):
    """Single 8-bit quantization; stable monochrome dither <= half a code step."""
    rng = np.random.default_rng(seed)
    if grain > 0:
        luma = image @ LUMA
        noise = rng.normal(0, 1, luma.shape).astype(np.float32)
        image = image + noise[..., None] * (0.55 + 0.45 * (1 - luma))[..., None] * grain
    if dither_mask is not None:
        # Use a separate stream so changing artistic grain cannot move the dither.
        noise = np.random.default_rng(seed ^ 0xDEBA).uniform(-0.5, 0.5, image.shape[:2])
        image = image + (noise.astype(np.float32) * dither_mask / 255)[..., None]
    return np.clip(image * 255, 0, 255).round().astype(np.uint8)
