"""Cinema DI color science, Planckian colorimetry, 33x33x33 .cube 3D LUT synthesis, and RGB Parade scopes."""

from __future__ import annotations

import logging
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..models.schema import ClipColorGrade, ColorGradeReport
from .titles import _load_font_tier

logger = logging.getLogger(__name__)

DEFAULT_LUT_SIZE = 33

# Curated 8-stock DI film print & modern cinema emulation specs (+ neutral_reference bypass)
CINEMA_FILM_STOCK_SPECS: dict[str, dict[str, Any]] = {
    "archival_kodak_2383": {
        "display_name": "Kodak 2383 Print",
        "subtitle": "Rich subtractive shadows & warm highlights",
        "lift": (-0.015, -0.004, 0.019),
        "gamma": (0.008, 0.002, -0.010),
        "gain": (0.024, 0.007, -0.022),
        "contrast": 1.07,
        "saturation": 1.05,
        "vibrance": 0.06,
        "shadow_toe": 0.018,
        "shoulder_rolloff": 0.09,
        "subtractive_density": 0.09,
        "saturation_curves": (1.02, 1.06, 0.94),
        "skin_protection": 0.72,
        "lut_intensity": 0.82,
    },
    "fuji_eterna_soft": {
        "display_name": "Fuji Eterna 250D",
        "subtitle": "Velvety shadow roll-off & pastel skin tones",
        "lift": (-0.008, 0.006, 0.012),
        "gamma": (0.000, 0.005, -0.004),
        "gain": (0.009, 0.006, -0.011),
        "contrast": 1.01,
        "saturation": 0.98,
        "vibrance": 0.08,
        "shadow_toe": 0.028,
        "shoulder_rolloff": 0.11,
        "subtractive_density": 0.06,
        "saturation_curves": (0.98, 1.00, 0.92),
        "skin_protection": 0.78,
        "lut_intensity": 0.78,
    },
    "arri_natural_cinema": {
        "display_name": "ARRI Natural Reveal",
        "subtitle": "True-to-life 10-bit editorial dynamic range",
        "lift": (-0.004, 0.000, 0.005),
        "gamma": (0.002, 0.000, -0.002),
        "gain": (0.007, 0.002, -0.006),
        "contrast": 1.05,
        "saturation": 1.03,
        "vibrance": 0.04,
        "shadow_toe": 0.012,
        "shoulder_rolloff": 0.09,
        "subtractive_density": 0.05,
        "saturation_curves": (1.00, 1.04, 0.96),
        "skin_protection": 0.75,
        "lut_intensity": 0.76,
    },
    "pacific_coastal_chrome": {
        "display_name": "Ektachrome 100D",
        "subtitle": "Crisp coastal blues & luminous clean sand",
        "lift": (-0.012, -0.002, 0.016),
        "gamma": (-0.004, 0.004, 0.008),
        "gain": (0.014, 0.008, -0.008),
        "contrast": 1.06,
        "saturation": 1.07,
        "vibrance": 0.10,
        "shadow_toe": 0.010,
        "shoulder_rolloff": 0.07,
        "subtractive_density": 0.07,
        "saturation_curves": (1.04, 1.08, 0.98),
        "skin_protection": 0.70,
        "lut_intensity": 0.80,
    },
    "golden_hour_warmth": {
        "display_name": "Vision3 Golden Hour",
        "subtitle": "Amber highlight glow & nostalgic warmth",
        "lift": (-0.006, -0.002, 0.010),
        "gamma": (0.019, 0.006, -0.021),
        "gain": (0.032, 0.012, -0.030),
        "contrast": 1.05,
        "saturation": 1.06,
        "vibrance": 0.08,
        "shadow_toe": 0.015,
        "shoulder_rolloff": 0.09,
        "subtractive_density": 0.08,
        "saturation_curves": (1.02, 1.08, 0.92),
        "skin_protection": 0.74,
        "lut_intensity": 0.82,
    },
    "museum_chiaroscuro": {
        "display_name": "Gallery Chiaroscuro",
        "subtitle": "Sculpted indoor highlights & velvety darks",
        "lift": (-0.012, -0.006, 0.008),
        "gamma": (0.005, 0.000, -0.007),
        "gain": (0.018, 0.006, -0.016),
        "contrast": 1.08,
        "saturation": 0.96,
        "vibrance": 0.04,
        "shadow_toe": 0.010,
        "shoulder_rolloff": 0.10,
        "subtractive_density": 0.07,
        "saturation_curves": (0.94, 0.98, 0.90),
        "skin_protection": 0.76,
        "lut_intensity": 0.78,
    },
    "twilight_teal_amber": {
        "display_name": "Dusk Teal & Amber",
        "subtitle": "Cinema dusk split-tone with protected skin",
        "lift": (-0.024, 0.006, 0.026),
        "gamma": (0.010, 0.002, -0.012),
        "gain": (0.028, 0.008, -0.026),
        "contrast": 1.07,
        "saturation": 1.05,
        "vibrance": 0.06,
        "shadow_toe": 0.014,
        "shoulder_rolloff": 0.09,
        "subtractive_density": 0.09,
        "saturation_curves": (1.04, 1.06, 0.92),
        "skin_protection": 0.74,
        "lut_intensity": 0.82,
    },
    "a24_indie_matte": {
        "display_name": "Indie A24 Matte",
        "subtitle": "Lifted archival toe & restrained editorial palette",
        "lift": (0.008, 0.004, 0.014),
        "gamma": (0.006, 0.004, -0.008),
        "gain": (0.012, 0.002, -0.016),
        "contrast": 1.02,
        "saturation": 0.95,
        "vibrance": 0.05,
        "shadow_toe": 0.036,
        "shoulder_rolloff": 0.12,
        "subtractive_density": 0.05,
        "saturation_curves": (0.94, 0.96, 0.90),
        "skin_protection": 0.78,
        "lut_intensity": 0.78,
    },
    "neutral_reference": {
        "display_name": "Neutral Reference",
        "subtitle": "Unity technical pass (no creative look)",
        "lift": (0.0, 0.0, 0.0),
        "gamma": (0.0, 0.0, 0.0),
        "gain": (0.0, 0.0, 0.0),
        "contrast": 1.00,
        "saturation": 1.00,
        "vibrance": 0.0,
        "shadow_toe": 0.0,
        "shoulder_rolloff": 0.0,
        "subtractive_density": 0.0,
        "saturation_curves": (1.0, 1.0, 1.0),
        "skin_protection": 0.0,
        "lut_intensity": 0.0,
    },
}

CREATIVE_FILM_STOCK_ORDER: tuple[str, ...] = (
    "archival_kodak_2383",
    "fuji_eterna_soft",
    "arri_natural_cinema",
    "pacific_coastal_chrome",
    "golden_hour_warmth",
    "museum_chiaroscuro",
    "twilight_teal_amber",
    "a24_indie_matte",
)

_LUT_ARRAY_CACHE: dict[tuple[str, int], np.ndarray] = {}


def is_synthetic_graphic_clip(
    source: str | Path,
    item_id: str = "",
    label: str = "",
) -> bool:
    """Return True for rendered title cards, chapter plates, or travel maps that must stay color-bypassed."""
    src_str = str(source or "").lower().replace("\\", "/")
    fname = Path(src_str).name if src_str else ""
    cid = str(item_id or "").lower()
    lbl = str(label or "").lower()

    synthetic_prefixes = ("sec_title_", "flyover_", "route_map_", "chapter_plate_", "title_card_", "kinetic_")
    if fname.startswith(synthetic_prefixes):
        return True
    if "/_titles/" in src_str or "/titles/" in src_str or "/travel_maps/" in src_str:
        return True
    synthetic_tokens = (
        "sec_title",
        "section_title",
        "title_card",
        "intro_card",
        "closing_card",
        "flyover",
        "travel_map",
        "route_map",
        "chapter_plate",
    )
    combined = f"{cid} {lbl} {fname}"
    return any(tok in combined for tok in synthetic_tokens)


def estimate_kelvin_and_tint_from_rgb(mean_r: float, mean_g: float, mean_b: float) -> tuple[float, float]:
    """Estimate correlated color temperature (Kelvin) and green-magenta tint via CIE XYZ + McCamy."""
    r = max(1e-4, float(mean_r))
    g = max(1e-4, float(mean_g))
    b = max(1e-4, float(mean_b))

    # Linearize sRGB channel ratios before XYZ transform
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl, gl, bl = _lin(r), _lin(g), _lin(b)
    x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    denom = x + y + z
    if denom <= 1e-6:
        return 5500.0, 0.0

    cx = x / denom
    cy = y / denom
    n_denom = 0.1858 - cy
    if abs(n_denom) < 1e-4:
        cct = 5500.0
    else:
        n = (cx - 0.3320) / n_denom
        cct = 449.0 * (n ** 3) + 3525.0 * (n ** 2) + 6823.3 * n + 5520.33
    cct = float(max(2500.0, min(9800.0, cct)))

    # Green vs magenta deviation relative to red/blue average (+ = green tint, - = magenta tint)
    tint = (g - 0.5 * (r + b)) * 0.85
    tint = float(max(-0.30, min(0.30, tint)))
    return round(cct, 1), round(tint, 4)


def compute_skin_mask_float(rgb_float: np.ndarray) -> np.ndarray:
    """Return smooth [0..1] skin-locus mask from normalized RGB float32 array in [0, 1]."""
    r = rgb_float[..., 0] * 255.0
    g = rgb_float[..., 1] * 255.0
    b = rgb_float[..., 2] * 255.0
    y_val = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    # Elliptical Chai-Ngan skin center (~Cb=102, Cr=152) with soft falloff
    dist_sq = ((cb - 103.0) / 21.0) ** 2 + ((cr - 153.0) / 18.0) ** 2
    chroma_w = np.clip(1.0 - np.maximum(0.0, dist_sq - 0.45) / 0.85, 0.0, 1.0)
    gate = (
        np.clip((y_val - 38.0) / 24.0, 0.0, 1.0)
        * np.clip((r - g - 4.0) / 14.0, 0.0, 1.0)
        * np.clip((r - b - 6.0) / 16.0, 0.0, 1.0)
    )
    return (chroma_w * gate).astype(np.float32)


def analyze_rgb_sample_array(rgb_frames: np.ndarray) -> dict[str, Any]:
    """Compute perceptual luma quantiles, Planckian Kelvin/tint, and skin ratio from RGB uint8 array(s)."""
    arr = np.asarray(rgb_frames, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.size == 0:
        return {
            "est_kelvin": 5500.0,
            "est_tint": 0.0,
            "p02_luma": 0.04,
            "p05_luma": 0.08,
            "p50_luma": 0.45,
            "p95_luma": 0.90,
            "p98_luma": 0.94,
            "skin_pixel_ratio": 0.0,
            "clipped_highlights_ratio": 0.0,
            "crushed_shadows_ratio": 0.0,
            "mean_rgb": (0.45, 0.45, 0.45),
        }

    norm = arr / 255.0 if float(np.max(arr)) > 1.5 else np.clip(arr, 0.0, 1.0)
    r = norm[..., 0]
    g = norm[..., 1]
    b = norm[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    p02, p05, p50, p95, p98 = [float(x) for x in np.percentile(luma, [2, 5, 50, 95, 98])]
    clipped_hi = float(np.mean(luma > 0.94))
    crushed_lo = float(np.mean(luma < 0.04))

    # Estimate illuminant from non-clipped mid-luminance pixels first; prefer neutral-ish subset when present
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    chroma_span = max_c - min_c
    mid_mask = (luma >= 0.08) & (luma <= 0.92)
    neutral_mask = mid_mask & (chroma_span <= 0.28)

    if int(np.sum(mid_mask)) >= 32:
        global_r = float(np.mean(r[mid_mask]))
        global_g = float(np.mean(g[mid_mask]))
        global_b = float(np.mean(b[mid_mask]))
    else:
        global_r = float(np.mean(r))
        global_g = float(np.mean(g))
        global_b = float(np.mean(b))

    if int(np.sum(neutral_mask)) >= max(32, int(0.06 * luma.size)):
        neu_r = float(np.mean(r[neutral_mask]))
        neu_g = float(np.mean(g[neutral_mask]))
        neu_b = float(np.mean(b[neutral_mask]))
        est_r = 0.65 * neu_r + 0.35 * global_r
        est_g = 0.65 * neu_g + 0.35 * global_g
        est_b = 0.65 * neu_b + 0.35 * global_b
    else:
        est_r, est_g, est_b = global_r, global_g, global_b

    est_kelvin, est_tint = estimate_kelvin_and_tint_from_rgb(est_r, est_g, est_b)
    skin_w = compute_skin_mask_float(norm)
    skin_ratio = float(np.mean(skin_w > 0.35))

    return {
        "est_kelvin": round(est_kelvin, 1),
        "est_tint": round(est_tint, 4),
        "p02_luma": round(p02, 4),
        "p05_luma": round(p05, 4),
        "p50_luma": round(p50, 4),
        "p95_luma": round(p95, 4),
        "p98_luma": round(p98, 4),
        "skin_pixel_ratio": round(skin_ratio, 4),
        "clipped_highlights_ratio": round(clipped_hi, 4),
        "crushed_shadows_ratio": round(crushed_lo, 4),
        "mean_rgb": (round(est_r, 4), round(est_g, 4), round(est_b, 4)),
    }


def _probe_duration_sec(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not video_path.exists():
        return 0.0
    try:
        res = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return max(0.0, float((res.stdout or "0").strip() or 0.0))
    except Exception:
        return 0.0


def _decode_video_sample_frames(
    video_path: Path,
    timestamps: list[float],
    sample_width: int = 160,
    sample_height: int = 90,
) -> list[np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not video_path.exists():
        return []
    frames: list[np.ndarray] = []
    expected_bytes = sample_width * sample_height * 3
    for ts in timestamps:
        try:
            cmd = [
                ffmpeg,
                "-v",
                "error",
                "-ss",
                f"{max(0.0, float(ts)):.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={sample_width}:{sample_height}:flags=bilinear,format=rgb24",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=6)
            if res.returncode == 0 and len(res.stdout) >= expected_bytes:
                buf = np.frombuffer(res.stdout[:expected_bytes], dtype=np.uint8)
                frames.append(buf.reshape((sample_height, sample_width, 3)))
        except Exception:
            continue
    return frames


def measure_media_colorimetry(
    media_path: str | Path,
    media_type: str = "video",
    duration: float = 0.0,
    start_offset: float = 0.0,
) -> dict[str, Any]:
    """Sample 3 trimmed video frames (or EXIF-oriented photo) and extract colorimetry telemetry."""
    path = Path(media_path)
    if not path.exists():
        return analyze_rgb_sample_array(np.empty((0, 0, 3), dtype=np.uint8))

    ext = path.suffix.lower()
    is_photo = media_type == "photo" or ext in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif"}
    if is_photo:
        try:
            with Image.open(path) as img:
                im = ImageOps.exif_transpose(img).convert("RGB")
                im.thumbnail((192, 192), Image.Resampling.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)
            return analyze_rgb_sample_array(arr)
        except Exception as exc:
            logger.debug("Photo colorimetry fallback for %s: %s", path.name, exc)
            return analyze_rgb_sample_array(np.empty((0, 0, 3), dtype=np.uint8))

    probed_dur = float(duration or 0.0)
    if probed_dur <= 0.1:
        probed_dur = _probe_duration_sec(path)
    base = max(0.0, float(start_offset or 0.0))
    if probed_dur > 0.35:
        timestamps = [base + probed_dur * frac for frac in (0.15, 0.50, 0.82)]
    else:
        timestamps = [base + 0.05]

    frames = _decode_video_sample_frames(path, timestamps=timestamps)
    if not frames and base > 0.0:
        frames = _decode_video_sample_frames(path, timestamps=[0.1])
    if not frames:
        return analyze_rgb_sample_array(np.empty((0, 0, 3), dtype=np.uint8))
    return analyze_rgb_sample_array(np.stack(frames, axis=0))


def build_film_stock_lut_table(look_name: str, lut_size: int = DEFAULT_LUT_SIZE) -> np.ndarray:
    """Synthesize a 33x33x33 float32 RGB 3D LUT grid indexed [b, g, r, 3] for the named stock."""
    size = max(9, int(lut_size))
    key = (str(look_name), size)
    cached = _LUT_ARRAY_CACHE.get(key)
    if cached is not None:
        return cached

    spec = CINEMA_FILM_STOCK_SPECS.get(look_name) or CINEMA_FILM_STOCK_SPECS["archival_kodak_2383"]
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b_g, g_g, r_g = np.meshgrid(axis, axis, axis, indexing="ij")
    rgb = np.stack([r_g, g_g, b_g], axis=-1)  # shape (S, S, S, 3), R fastest in C-order flatten

    if look_name == "neutral_reference":
        _LUT_ARRAY_CACHE[key] = rgb
        return rgb

    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    # 1. Soft film toe + contrast + highlight shoulder roll-off
    contrast = float(spec["contrast"])
    shadow_toe = float(spec["shadow_toe"])
    shoulder = float(spec["shoulder_rolloff"])

    shaped = np.clip(0.45 + (rgb - 0.45) * contrast, 0.0, 1.25)
    toe_add = shadow_toe * np.power(np.clip(1.0 - luma, 0.0, 1.0), 3.2)[..., None]
    shaped = shaped + toe_add * (1.0 - np.clip(rgb, 0.0, 1.0))

    # Rational highlight knee compressing signal above 0.72 without crushing diffuse whites
    hi_excess = np.maximum(0.0, shaped - 0.72)
    shaped = shaped - shoulder * (hi_excess ** 1.6) / (0.28 + hi_excess + 1e-4)

    # 2. Shadow / Midtone / Highlight split-toning
    w_lift = np.power(np.clip(1.0 - luma, 0.0, 1.0), 2.1)[..., None]
    w_gamma = (4.0 * np.clip(luma, 0.0, 1.0) * np.clip(1.0 - luma, 0.0, 1.0))[..., None]
    w_gain = np.power(np.clip(luma, 0.0, 1.0), 1.65)[..., None]

    lift_vec = np.asarray(spec["lift"], dtype=np.float32) * 0.65
    gamma_vec = np.asarray(spec["gamma"], dtype=np.float32) * 0.65
    gain_vec = np.asarray(spec["gain"], dtype=np.float32) * 0.65

    # Taper tint at exact black/white boundaries for clean neutrality on extremes
    extreme_taper = np.clip(4.2 * luma * (1.0 - luma) + 0.35, 0.0, 1.0)[..., None]
    shaped = shaped + (w_lift * lift_vec + w_gamma * gamma_vec + w_gain * gain_vec) * extreme_taper

    # 3. Subtractive film density & shadow/mid/highlight saturation curve
    y_shaped = (0.2126 * shaped[..., 0] + 0.7152 * shaped[..., 1] + 0.0722 * shaped[..., 2])[..., None]
    chroma_vec = shaped - y_shaped
    chroma_mag = np.sqrt(np.clip(np.sum(chroma_vec ** 2, axis=-1, keepdims=True), 0.0, 1.0))

    sat_s, sat_m, sat_h = [float(v) for v in spec["saturation_curves"]]
    sat_curve = (
        w_lift * sat_s + w_gamma * sat_m + w_gain * sat_h
    ) / np.maximum(1e-4, w_lift + w_gamma + w_gain)
    sat_total = sat_curve * float(spec["saturation"])

    sub_factor = float(spec["subtractive_density"]) * chroma_mag * (0.4 + 0.6 * w_gamma)
    dense_y = y_shaped * (1.0 - sub_factor)
    lut_rgb = dense_y + chroma_vec * sat_total

    # 4. Protect skin-tone vector inside the LUT so creative split tones keep healthy skin warmth
    skin_prot = float(spec["skin_protection"])
    if skin_prot > 0.0:
        skin_w = compute_skin_mask_float(np.clip(rgb, 0.0, 1.0))[..., None] * skin_prot * 0.65
        lut_rgb = lut_rgb * (1.0 - skin_w) + (0.55 * rgb + 0.45 * lut_rgb) * skin_w

    out = np.clip(lut_rgb, 0.0, 1.0).astype(np.float32)
    _LUT_ARRAY_CACHE[key] = out
    return out


def write_film_stock_cube_lut(
    look_name: str,
    out_path: str | Path,
    lut_size: int = DEFAULT_LUT_SIZE,
    intensity: float = 1.0,
) -> Path:
    """Write an industry-standard Adobe/Resolve .cube 3D LUT file (R-fastest loop order)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size > 10000:
        return p

    table = build_film_stock_lut_table(look_name=look_name, lut_size=lut_size)
    eff_int = max(0.0, min(1.0, float(intensity)))
    if eff_int < 0.999 and look_name != "neutral_reference":
        id_table = build_film_stock_lut_table(look_name="neutral_reference", lut_size=lut_size)
        table = np.clip(id_table * (1.0 - eff_int) + table * eff_int, 0.0, 1.0).astype(np.float32)
    size = table.shape[0]
    flat = table.reshape(-1, 3)

    spec = CINEMA_FILM_STOCK_SPECS.get(look_name) or CINEMA_FILM_STOCK_SPECS["archival_kodak_2383"]
    lines = [
        f'TITLE "{spec["display_name"]} ({look_name}, int={eff_int:.2f})"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]
    lines.extend(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}" for row in flat)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def ensure_project_luts(project_dir: str | Path, look_names: Iterable[str] | None = None) -> dict[str, str]:
    """Materialize .cube 3D LUTs into <project_dir>/_luts/ and return look -> relative/path mapping."""
    lut_dir = Path(project_dir) / "_luts"
    lut_dir.mkdir(parents=True, exist_ok=True)
    targets = list(look_names or CREATIVE_FILM_STOCK_ORDER)
    out: dict[str, str] = {}
    for look in targets:
        if look not in CINEMA_FILM_STOCK_SPECS or look == "neutral_reference":
            continue
        lut_path = write_film_stock_cube_lut(look, lut_dir / f"{look}.cube", lut_size=DEFAULT_LUT_SIZE)
        out[look] = str(lut_path)
    return out


def _trilinear_lut_lookup(rgb_float: np.ndarray, lut_table: np.ndarray) -> np.ndarray:
    """Vectorized trilinear interpolation over a (S, S, S, 3) [B, G, R, 3] 3D LUT."""
    size = lut_table.shape[0]
    scale = float(size - 1)
    r_idx = np.clip(rgb_float[..., 0] * scale, 0.0, scale)
    g_idx = np.clip(rgb_float[..., 1] * scale, 0.0, scale)
    b_idx = np.clip(rgb_float[..., 2] * scale, 0.0, scale)

    r0 = np.floor(r_idx).astype(np.int32)
    g0 = np.floor(g_idx).astype(np.int32)
    b0 = np.floor(b_idx).astype(np.int32)
    r1 = np.minimum(r0 + 1, size - 1)
    g1 = np.minimum(g0 + 1, size - 1)
    b1 = np.minimum(b0 + 1, size - 1)

    fr = (r_idx - r0)[..., None]
    fg = (g_idx - g0)[..., None]
    fb = (b_idx - b0)[..., None]

    c000 = lut_table[b0, g0, r0]
    c001 = lut_table[b0, g0, r1]
    c010 = lut_table[b0, g1, r0]
    c011 = lut_table[b0, g1, r1]
    c100 = lut_table[b1, g0, r0]
    c101 = lut_table[b1, g0, r1]
    c110 = lut_table[b1, g1, r0]
    c111 = lut_table[b1, g1, r1]

    c00 = c000 * (1.0 - fr) + c001 * fr
    c01 = c010 * (1.0 - fr) + c011 * fr
    c10 = c100 * (1.0 - fr) + c101 * fr
    c11 = c110 * (1.0 - fr) + c111 * fr

    c0 = c00 * (1.0 - fg) + c01 * fg
    c1 = c10 * (1.0 - fg) + c11 * fg
    return c0 * (1.0 - fb) + c1 * fb


def _extract_grade_params(grade: ClipColorGrade | dict[str, Any] | None) -> dict[str, Any]:
    if grade is None:
        return {}
    if isinstance(grade, ClipColorGrade):
        return grade.model_dump()
    if hasattr(grade, "parameters") and isinstance(getattr(grade, "parameters"), dict):
        return dict(getattr(grade, "parameters"))
    return dict(grade)


def apply_color_grade_to_rgb(
    rgb: np.ndarray,
    grade: ClipColorGrade | dict[str, Any] | None,
    lut_table: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply primary Lift/Gamma/Gain, Exposure/Contrast/Vibrance, and 3D LUT with skin protection."""
    params = _extract_grade_params(grade)
    if not params or params.get("bypass"):
        return rgb

    is_uint8 = rgb.dtype == np.uint8
    img = rgb.astype(np.float32) / 255.0 if is_uint8 else np.clip(rgb.astype(np.float32), 0.0, 1.0)

    look_name = str(params.get("chapter_look") or params.get("look") or "archival_kodak_2383")
    if look_name == "neutral_reference" and abs(float(params.get("exposure_ev", 0.0))) < 1e-4:
        return rgb

    skin_prot = float(params.get("skin_protection", 0.65))
    skin_mask = compute_skin_mask_float(img)[..., None] * max(0.0, min(1.0, skin_prot))

    # 1. Exposure EV with smooth highlight shoulder
    exp_ev = float(params.get("exposure_ev", 0.0))
    if abs(exp_ev) > 1e-4:
        gain_mult = 2.0 ** max(-1.5, min(1.5, exp_ev))
        img = img * gain_mult
        if gain_mult > 1.0:
            hi = np.maximum(0.0, img - 0.85)
            img = np.minimum(img, 0.85 + hi / (1.0 + hi * 1.4))
        img = np.clip(img, 0.0, 1.0)

    # 2. Primary White Balance & 3-Way Lift / Gamma / Gain
    wb_shift = np.asarray(
        [
            float(params.get("wb_red_shift", params.get("rs", 0.0))),
            float(params.get("wb_green_shift", params.get("gs", 0.0))),
            float(params.get("wb_blue_shift", params.get("bs", 0.0))),
        ],
        dtype=np.float32,
    )
    lift_v = np.asarray(params.get("lift_rgb", (0.0, 0.0, 0.0)), dtype=np.float32)
    gamma_v = np.asarray(params.get("gamma_rgb", (0.0, 0.0, 0.0)), dtype=np.float32)
    gain_v = np.asarray(params.get("gain_rgb", (0.0, 0.0, 0.0)), dtype=np.float32)
    black_lift = float(params.get("black_lift", 0.0))

    luma = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])[..., None]
    w_s = (1.0 - luma) ** 2.0
    w_m = 4.0 * luma * (1.0 - luma)
    w_h = luma ** 1.5

    # WB shift applies uniformly across exposure range to neutralize true camera Kelvin offset
    img = img + wb_shift * (0.35 + 0.65 * w_m + 0.45 * w_h) * np.clip(luma * 3.0 + 0.12, 0.0, 1.0)
    # Creative Lift/Gamma/Gain gently attenuates extreme channel pushes on skin
    lgg_delta = (w_s * lift_v + w_m * gamma_v + w_h * gain_v) * (1.0 - 0.35 * skin_mask)
    img = img + lgg_delta + black_lift * w_s
    img = np.clip(img, 0.0, 1.0)

    # 3. Contrast, scalar Gamma, Saturation & Vibrance
    contrast = float(params.get("contrast", 1.0))
    gamma_val = max(0.65, min(1.50, float(params.get("gamma", 1.0))))
    sat = float(params.get("saturation", 1.0))
    vibrance = float(params.get("vibrance", 0.0))

    if abs(contrast - 1.0) > 1e-4:
        img = np.clip(0.45 + (img - 0.45) * contrast, 0.0, 1.0)
    if abs(gamma_val - 1.0) > 1e-4:
        img = np.power(np.clip(img, 1e-5, 1.0), 1.0 / gamma_val)

    if abs(sat - 1.0) > 1e-4 or abs(vibrance) > 1e-4:
        y2 = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])[..., None]
        chroma = img - y2
        pix_sat = (np.max(img, axis=-1, keepdims=True) - np.min(img, axis=-1, keepdims=True))
        vib_scale = 1.0 + vibrance * (1.0 - pix_sat) * (1.0 - 0.65 * skin_mask)
        img = np.clip(y2 + chroma * sat * vib_scale, 0.0, 1.0)

    # 4. 3D Film-Print LUT Transform + Elliptical Skin-Vector Protection
    lut_int = float(params.get("lut_intensity", 0.0))
    if lut_int > 1e-4 and look_name != "neutral_reference":
        table = lut_table if lut_table is not None else build_film_stock_lut_table(look_name)
        lut_out = _trilinear_lut_lookup(img, table)
        eff_int = np.clip(lut_int * (1.0 - 0.50 * skin_mask), 0.0, 1.0)
        img = img * (1.0 - eff_int) + lut_out * eff_int

    img = np.clip(img, 0.0, 1.0)
    if is_uint8:
        return np.round(img * 255.0).astype(np.uint8)
    return img.astype(np.float32)


def apply_color_grade_to_pil_image(
    im: Image.Image,
    grade: ClipColorGrade | dict[str, Any] | None,
    lut_table: Optional[np.ndarray] = None,
) -> Image.Image:
    """Color-grade a PIL Image (RGB or RGBA, preserving alpha matte cleanly)."""
    params = _extract_grade_params(grade)
    if not params or params.get("bypass"):
        return im

    has_alpha = im.mode in ("RGBA", "LA")
    if has_alpha:
        rgba = im.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.uint8)
        graded_rgb = apply_color_grade_to_rgb(arr[..., :3], params, lut_table=lut_table)
        merged = np.dstack([graded_rgb, arr[..., 3:4]])
        return Image.fromarray(merged, mode="RGBA")

    rgb_im = im.convert("RGB")
    arr = np.asarray(rgb_im, dtype=np.uint8)
    graded_rgb = apply_color_grade_to_rgb(arr, params, lut_table=lut_table)
    return Image.fromarray(graded_rgb, mode="RGB")


def _fit_cover_rgb(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    return ImageOps.fit(im.convert("RGB"), (target_w, target_h), method=Image.Resampling.LANCZOS)


def render_color_lookbook_sheet(
    sample_paths: list[str | Path],
    output_path: str | Path,
    project_title: str = "Cinema DI Color Lookbook",
    width: int = 2560,
    height: int = 1440,
) -> Path:
    """Render a 2560x1440 4x2 visual lookbook sheet comparing all 8 curated cinema film stocks."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    valid_samples = [Path(p) for p in sample_paths if p and Path(p).exists()]
    if not valid_samples:
        raise FileNotFoundError("No valid sample image/video files provided for render_color_lookbook_sheet.")

    loaded_samples: list[Image.Image] = []
    for sp in valid_samples[:4]:
        if sp.suffix.lower() in {".mp4", ".mov", ".mkv", ".m4v"}:
            dur = _probe_duration_sec(sp)
            frames = _decode_video_sample_frames(sp, [max(0.1, dur * 0.45)], sample_width=960, sample_height=540)
            if frames:
                loaded_samples.append(Image.fromarray(frames[0], mode="RGB"))
        else:
            try:
                with Image.open(sp) as img:
                    loaded_samples.append(ImageOps.exif_transpose(img).convert("RGB"))
            except Exception:
                continue

    if not loaded_samples:
        raise RuntimeError("Could not decode any sample frame/image for color lookbook.")

    canvas = Image.new("RGB", (width, height), (14, 16, 20))
    draw = ImageDraw.Draw(canvas)

    font_title = _load_font_tier(40, tier="serif", bold=True)
    font_sub = _load_font_tier(20, tier="sans", bold=False)
    font_card_title = _load_font_tier(22, tier="serif", bold=True)
    font_mono = _load_font_tier(16, tier="mono", bold=False)

    draw.text((64, 34), f"DI COLOR LOOKBOOK  ·  {project_title.upper()}", fill=(244, 240, 232), font=font_title)
    draw.text(
        (64, 86),
        "8 Curated 33x33x33 3D LUT Emulations  |  Left Split: Camera Original   Right: Film Print   + RGB Histogram",
        fill=(165, 174, 188),
        font=font_sub,
    )

    cols, rows = 4, 2
    margin_x, margin_top, gap_x, gap_y = 64, 134, 26, 28
    card_w = (width - 2 * margin_x - (cols - 1) * gap_x) // cols
    card_h = (height - margin_top - 48 - (rows - 1) * gap_y) // rows
    img_h = int(card_w * 9 / 16)

    for idx, stock_name in enumerate(CREATIVE_FILM_STOCK_ORDER):
        r_idx = idx // cols
        c_idx = idx % cols
        x0 = margin_x + c_idx * (card_w + gap_x)
        y0 = margin_top + r_idx * (card_h + gap_y)

        spec = CINEMA_FILM_STOCK_SPECS[stock_name]
        sample_im = loaded_samples[idx % len(loaded_samples)]
        tile_raw = _fit_cover_rgb(sample_im, card_w, img_h)
        raw_arr = np.asarray(tile_raw, dtype=np.uint8)

        grade = ClipColorGrade(
            clip_id=f"lookbook_{idx+1}",
            chapter_look=stock_name,  # type: ignore[arg-type]
            lift_rgb=spec["lift"],
            gamma_rgb=spec["gamma"],
            gain_rgb=spec["gain"],
            contrast=spec["contrast"],
            saturation=spec["saturation"],
            vibrance=spec["vibrance"],
            skin_protection=spec["skin_protection"],
            lut_intensity=spec["lut_intensity"],
        )
        graded_arr = apply_color_grade_to_rgb(raw_arr, grade)

        # Split-wipe (left 28% original, right 72% graded) for immediate tonal comparison
        split_x = int(card_w * 0.28)
        comp_arr = graded_arr.copy()
        comp_arr[:, :split_x, :] = raw_arr[:, :split_x, :]
        tile_img = Image.fromarray(comp_arr, mode="RGB")

        # Card bezel
        draw.rounded_rectangle([x0 - 2, y0 - 2, x0 + card_w + 2, y0 + card_h + 2], radius=10, fill=(24, 27, 34), outline=(56, 62, 74), width=1)
        canvas.paste(tile_img, (x0, y0))
        draw.line([(x0 + split_x, y0), (x0 + split_x, y0 + img_h)], fill=(238, 232, 220), width=2)

        # Mini RGB histogram overlay in bottom-right of thumbnail
        hist_w, hist_h = 112, 42
        hx0, hy0 = x0 + card_w - hist_w - 10, y0 + img_h - hist_h - 10
        draw.rectangle([hx0, hy0, hx0 + hist_w, hy0 + hist_h], fill=(10, 12, 16), outline=(70, 76, 88), width=1)
        bins = 28
        for ch_idx, ch_col in enumerate([(235, 90, 90), (95, 220, 125), (90, 155, 245)]):
            hist, _ = np.histogram(graded_arr[..., ch_idx], bins=bins, range=(0, 255))
            hmax = max(1, int(hist.max()))
            pts = [
                (
                    hx0 + 4 + int((bi / float(bins - 1)) * (hist_w - 8)),
                    hy0 + hist_h - 4 - int((float(hv) / hmax) * (hist_h - 8)),
                )
                for bi, hv in enumerate(hist)
            ]
            draw.line(pts, fill=ch_col, width=2)

        # Metadata footer
        ty = y0 + img_h + 14
        draw.text((x0 + 16, ty), spec["display_name"], fill=(244, 240, 232), font=font_card_title)
        draw.text((x0 + 16, ty + 30), stock_name, fill=(145, 178, 215), font=font_mono)
        draw.text((x0 + 16, ty + 54), spec["subtitle"], fill=(176, 182, 192), font=font_sub)

        # Footer numeric telemetry pills
        info_txt = (
            f"CON {spec['contrast']:.2f}  SAT {spec['saturation']:.2f}  "
            f"LUT {int(round(spec['lut_intensity']*100))}%  SKIN {int(round(spec['skin_protection']*100))}%"
        )
        draw.text((x0 + 16, ty + 90), info_txt, fill=(214, 192, 148), font=font_mono)

    canvas.save(out, quality=92)
    return out


def _draw_rgb_parade(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    rgb_arr: np.ndarray,
    box: tuple[int, int, int, int],
) -> None:
    """Render an authentic 3-column RGB Parade waveform scope inside [x0, y0, x1, y1]."""
    x0, y0, x1, y1 = box
    w = max(90, x1 - x0)
    h = max(60, y1 - y0)
    draw.rectangle([x0, y0, x1, y1], fill=(10, 12, 15), outline=(52, 58, 70), width=1)

    # IRE grid lines (25%, 50%, 75%)
    for frac in (0.25, 0.50, 0.75):
        gy = y0 + int(frac * h)
        draw.line([(x0 + 2, gy), (x1 - 2, gy)], fill=(30, 34, 42), width=1)

    col_w = (w - 8) // 3
    small_h = min(72, rgb_arr.shape[0])
    small_w = col_w
    scope_img = np.zeros((h, w, 3), dtype=np.float32)

    ds = Image.fromarray(rgb_arr, mode="RGB").resize((small_w, small_h), Image.Resampling.BILINEAR)
    ds_arr = np.asarray(ds, dtype=np.int32)

    channel_tints = [
        (0, (1.0, 0.28, 0.28)),
        (1, (0.28, 0.95, 0.42)),
        (2, (0.34, 0.62, 1.0)),
    ]
    for ch_idx, tint in channel_tints:
        sub_x0 = 4 + ch_idx * col_w
        vals = ds_arr[..., ch_idx]  # (small_h, small_w)
        y_bins = np.clip(((255 - vals) * (h - 6) // 255) + 3, 2, h - 3)
        for col in range(small_w):
            hist_counts = np.bincount(y_bins[:, col], minlength=h).astype(np.float32)
            density = np.clip(hist_counts / max(1.0, small_h * 0.14), 0.0, 1.0)
            px = min(w - 1, sub_x0 + col)
            for c_i in range(3):
                scope_img[:, px, c_i] = np.maximum(scope_img[:, px, c_i], density * tint[c_i])

    # Convert accumulated float density -> composited scope region
    base_crop = np.asarray(canvas.crop((x0, y0, x1, y1)), dtype=np.float32)
    parade_u8 = np.clip(base_crop + scope_img * 235.0, 0, 255).astype(np.uint8)
    canvas.paste(Image.fromarray(parade_u8, mode="RGB"), (x0, y0))
    draw.line([(x0 + col_w + 4, y0), (x0 + col_w + 4, y1)], fill=(38, 42, 50), width=1)
    draw.line([(x0 + 2 * col_w + 4, y0), (x0 + 2 * col_w + 4, y1)], fill=(38, 42, 50), width=1)


def render_color_grading_scopes_sheet(
    sample_entries: list[dict[str, Any]],
    output_path: str | Path,
    report: Optional[ColorGradeReport] = None,
    project_title: str = "",
    width: int = 2560,
    height: int = 1440,
) -> Path:
    """Render a 2560x1440 DI Before/Graded Split-Wipe & RGB Parade Waveform Scopes Sheet."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not sample_entries:
        raise ValueError("sample_entries must contain at least 1 graded clip for scopes sheet.")

    canvas = Image.new("RGB", (width, height), (12, 14, 18))
    draw = ImageDraw.Draw(canvas)

    font_title = _load_font_tier(38, tier="serif", bold=True)
    font_sub = _load_font_tier(19, tier="sans", bold=False)
    font_card = _load_font_tier(20, tier="serif", bold=True)
    font_mono = _load_font_tier(15, tier="mono", bold=False)

    title_str = project_title or (report.project_title if report else "Cinema Timeline")
    primary_look = report.primary_film_stock_look if report else "archival_kodak_2383"
    strategy = report.grading_strategy if report else "chapter_adaptive"
    spread_str = (
        f"Kelvin Spread: {report.kelvin_spread_before:.0f}K -> {report.kelvin_spread_after:.0f}K"
        if report and report.kelvin_spread_before > 0
        else "3-Way Lift / Gamma / Gain + 33x33x33 .cube LUT"
    )

    draw.text((60, 32), f"DI COLOR GRADING SCOPES  ·  {title_str.upper()}", fill=(244, 240, 232), font=font_title)
    draw.text(
        (60, 82),
        f"Primary Spine: {primary_look}   |   Strategy: {strategy}   |   {spread_str}",
        fill=(170, 180, 195),
        font=font_sub,
    )

    entries = sample_entries[:6]
    cols = 3 if len(entries) > 4 else 2
    rows = 2 if len(entries) > 2 else 1
    margin_x, margin_top, gap_x, gap_y = 60, 128, 26, 26
    card_w = (width - 2 * margin_x - (cols - 1) * gap_x) // cols
    card_h = (height - margin_top - 48 - (rows - 1) * gap_y) // rows
    img_h = int(card_h * 0.56)

    for idx, entry in enumerate(entries):
        r_idx = idx // cols
        c_idx = idx % cols
        x0 = margin_x + c_idx * (card_w + gap_x)
        y0 = margin_top + r_idx * (card_h + gap_y)

        grade_obj = entry.get("grade")
        gparams = _extract_grade_params(grade_obj)
        src_path = Path(str(entry.get("source") or ""))
        mtype = str(entry.get("media_type") or gparams.get("media_type") or "video")
        start_sec = float(entry.get("start") or 0.0)
        clip_dur = float(entry.get("duration") or 2.0)

        raw_img: Optional[Image.Image] = None
        if src_path.exists():
            if mtype == "photo" or src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff"}:
                with Image.open(src_path) as im:
                    raw_img = ImageOps.exif_transpose(im).convert("RGB")
            else:
                frames = _decode_video_sample_frames(
                    src_path, [max(0.05, start_sec + clip_dur * 0.45)], sample_width=card_w, sample_height=img_h
                )
                if frames:
                    raw_img = Image.fromarray(frames[0], mode="RGB")

        if raw_img is None:
            raw_img = Image.new("RGB", (card_w, img_h), (42, 46, 54))
        tile_raw = _fit_cover_rgb(raw_img, card_w, img_h)
        raw_arr = np.asarray(tile_raw, dtype=np.uint8)
        graded_arr = apply_color_grade_to_rgb(raw_arr, gparams)

        draw.rounded_rectangle(
            [x0 - 2, y0 - 2, x0 + card_w + 2, y0 + card_h + 2],
            radius=10,
            fill=(22, 25, 31),
            outline=(56, 62, 74),
            width=1,
        )

        # Split wipe 45% raw / 55% graded
        split_x = int(card_w * 0.45)
        split_arr = graded_arr.copy()
        split_arr[:, :split_x, :] = raw_arr[:, :split_x, :]
        canvas.paste(Image.fromarray(split_arr, mode="RGB"), (x0, y0))
        draw.line([(x0 + split_x, y0), (x0 + split_x, y0 + img_h)], fill=(240, 236, 224), width=2)

        draw.rectangle([x0 + 10, y0 + 10, x0 + 96, y0 + 34], fill=(14, 16, 22))
        draw.text((x0 + 18, y0 + 14), "ORIGINAL", fill=(215, 220, 228), font=font_mono)
        draw.rectangle([x0 + card_w - 114, y0 + 10, x0 + card_w - 10, y0 + 34], fill=(20, 24, 32))
        draw.text((x0 + card_w - 104, y0 + 14), "DI GRADED", fill=(235, 206, 152), font=font_mono)

        # RGB Parade Scope (left 54%) + Telemetry block (right 46%)
        scope_top = y0 + img_h + 12
        scope_bot = y0 + card_h - 14
        parade_w = int(card_w * 0.54)
        _draw_rgb_parade(draw, canvas, graded_arr, (x0 + 12, scope_top, x0 + 12 + parade_w, scope_bot))

        tx = x0 + 24 + parade_w
        cid = str(entry.get("clip_id") or gparams.get("clip_id") or f"clip_{idx+1}")
        look_nm = str(gparams.get("chapter_look") or primary_look)
        kelvin = float(gparams.get("measured_kelvin", 5500.0))
        ev = float(gparams.get("exposure_ev", 0.0))
        skin_r = float(gparams.get("skin_pixel_ratio", 0.0)) * 100.0
        wb_r = float(gparams.get("wb_red_shift", 0.0))
        wb_b = float(gparams.get("wb_blue_shift", 0.0))

        draw.text((tx, scope_top + 2), f"{cid[:22]} ({mtype})", fill=(242, 238, 230), font=font_card)
        draw.text((tx, scope_top + 32), f"Stock: {look_nm}", fill=(145, 182, 220), font=font_mono)
        draw.text((tx, scope_top + 56), f"Kelvin: {kelvin:.0f}K  EV: {ev:+.2f}", fill=(205, 210, 218), font=font_mono)
        draw.text((tx, scope_top + 80), f"WB R/B: {wb_r:+.3f} / {wb_b:+.3f}", fill=(195, 200, 210), font=font_mono)
        draw.text((tx, scope_top + 104), f"Skin Locus: {skin_r:.1f}%", fill=(224, 194, 146), font=font_mono)

    canvas.save(out, quality=92)
    return out
