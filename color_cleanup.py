"""Lightweight, CPU-only colour and gradient finishing (no model weights).

Independent implementation of local colour-distance smoothing and thresholded
multiscale gradient smoothing; not copied from FFmpeg or f3kdb. These are
heuristics, not semantic detectors of wrong colours or missing image content.
All processing stays float32 until the final, optionally dithered quantization.
"""
import numpy as np

from fine_cleanup import LUMA, box_mean, smoothstep


def _neighbors(image, radius):
    h, w = image.shape[:2]
    pads = [(radius, radius), (radius, radius)] + [(0, 0)] * (image.ndim - 2)
    padded = np.pad(image, pads, mode="edge")
    return lambda dy, dx: padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]


def _max_rgb(image):
    return np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])


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
    delta = np.empty_like(image)
    delta[..., 0] = correction[..., 0]
    delta[..., 2] = correction[..., 1]
    delta[..., 1] = -(LUMA[0] * delta[..., 0] + LUMA[2] * delta[..., 2]) / LUMA[1]
    # Scale the whole correction at gamut limits, rather than clipping channels
    # independently and changing the intended luma/hue relationship.
    limit = np.where(delta > 0, (1 - image) / np.maximum(delta, 1e-8),
                     image / np.maximum(-delta, 1e-8))
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
    threshold = (2 + 6 * amount) / 255
    # Reject local texture, including isoluminant coloured texture.
    high = _max_rgb(np.abs(image - box_mean(image, 1)))
    texture = np.sqrt(box_mean(high * high, 2))
    quiet = 1 - smoothstep(1.5 / 255, 5 / 255, texture)
    correction = np.zeros_like(image)
    sample = _neighbors(image, 16)
    squared = image * image
    for radius in (2, 4, 8, 16):
        largest = np.zeros_like(mask)
        for dy, dx in ((0, radius), (0, -radius), (radius, 0), (-radius, 0),
                       (radius, radius), (-radius, -radius),
                       (-radius, radius), (radius, -radius)):
            difference = _max_rgb(np.abs(sample(dy, dx) - image))
            largest = np.maximum(largest, difference)
        base = box_mean(image, radius)
        deviation = np.sqrt(np.maximum(_max_rgb(box_mean(squared, radius) - base * base), 0))
        # Sparse reference samples can jump over a thin line inside the window.
        # Full-window variance prevents its colour leaking into nearby plateaus.
        eligible = quiet * (1 - smoothstep(threshold, threshold * 1.5, largest))
        eligible *= 1 - smoothstep(threshold * 0.6, threshold, deviation)
        candidate = np.clip(base - image, -threshold, threshold)
        # A wider valid neighbourhood replaces the narrow estimate, rather
        # than stacking four smoothing passes and gradually eroding contours.
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
