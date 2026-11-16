"""Neural monocular depth estimation (Depth-Anything-V2-Small) & disocclusion-inpainted MPI layers."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import threading
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ..hardware import select_torch_device

_DEPTH_LOCK = threading.Lock()
_DEPTH_MODEL_BUNDLE: Optional[tuple[Any, Any, str]] = None
_DEPTH_MODEL_FAILED = False


def _load_depth_anything_model() -> Optional[tuple[Any, Any, str]]:
    """Lazy-load Depth-Anything-V2-Small-hf once across threads when network/cache allows."""
    global _DEPTH_MODEL_BUNDLE, _DEPTH_MODEL_FAILED
    if os.environ.get("AI_EDITOR_OFFLINE_DEPTH", "").strip() == "1":
        return None
    if _DEPTH_MODEL_BUNDLE is not None:
        return _DEPTH_MODEL_BUNDLE
    if _DEPTH_MODEL_FAILED:
        return None

    with _DEPTH_LOCK:
        if _DEPTH_MODEL_BUNDLE is not None:
            return _DEPTH_MODEL_BUNDLE
        if _DEPTH_MODEL_FAILED:
            return None
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_id = os.environ.get("AI_EDITOR_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
            dev = select_torch_device()
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id)
            model.eval()
            if dev == "cuda":
                model = model.to("cuda", dtype=torch.float16)
            elif dev == "mps":
                model = model.to("mps")
            _DEPTH_MODEL_BUNDLE = (processor, model, dev)
            return _DEPTH_MODEL_BUNDLE
        except Exception:
            _DEPTH_MODEL_FAILED = True
            return None


def _heuristic_depth_map(rgb_small: np.ndarray, focal_point: tuple[float, float]) -> np.ndarray:
    """Deterministic edge-aware saliency + horizon-perspective fallback depth in [0.0, 1.0]."""
    h, w = rgb_small.shape[:2]
    gray = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx * gx + gy * gy)
    edge_blur = cv2.GaussianBlur(edge_mag, (0, 0), sigmaX=max(3.0, min(w, h) * 0.04))
    p95 = float(np.percentile(edge_blur, 95)) if edge_blur.size else 1.0
    saliency = np.clip(edge_blur / max(0.05, p95), 0.0, 1.0)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = xx / max(1.0, float(w - 1))
    ny = yy / max(1.0, float(h - 1))
    fx_c = max(0.18, min(0.82, float(focal_point[0])))
    fy_c = max(0.18, min(0.82, float(focal_point[1])))

    # Ground-plane depth prior (lower portion closer) + focal ellipse prior + detail saliency
    horizon_prior = np.clip(ny ** 0.85, 0.0, 1.0)
    dist_sq = ((nx - fx_c) ** 2) / 0.14 + ((ny - fy_c) ** 2) / 0.18
    focal_prior = np.exp(-0.55 * dist_sq)

    raw = 0.42 * focal_prior + 0.33 * saliency + 0.25 * horizon_prior
    raw_u8 = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
    # Bilateral smooth so depth respects real object contours cleanly
    smoothed = cv2.bilateralFilter(raw_u8, d=9, sigmaColor=28, sigmaSpace=12).astype(np.float32) / 255.0
    lo = float(np.percentile(smoothed, 2))
    hi = float(np.percentile(smoothed, 98))
    if hi - lo < 1e-4:
        return np.full((h, w), 0.5, dtype=np.float32)
    return np.clip((smoothed - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def estimate_depth_map(
    img: Image.Image,
    focal_point: tuple[float, float] = (0.5, 0.5),
    cache_dir: Optional[Path] = None,
    photo_key: Optional[str] = None,
    target_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Compute a normalized continuous disparity map in [0.0, 1.0] (1.0 = closest foreground).

    Uses Depth-Anything-V2-Small-hf when available, with edge-guided bilateral smoothing
    and an automatic offline heuristic fallback. Cached to disk as 16-bit PNG when `cache_dir`
    is provided.
    """
    out_w, out_h = target_size if target_size is not None else img.size
    out_w = max(32, int(out_w))
    out_h = max(32, int(out_h))

    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key_src = photo_key or hashlib.sha1(img.resize((64, 64), Image.Resampling.BILINEAR).tobytes()).hexdigest()[:16]
        cache_file = cache_dir / f"depth_{key_src}_{out_w}x{out_h}.png"
        if cache_file.exists() and cache_file.stat().st_size > 128:
            try:
                loaded = cv2.imread(str(cache_file), cv2.IMREAD_UNCHANGED)
                if loaded is not None and loaded.shape[:2] == (out_h, out_w):
                    return np.clip(loaded.astype(np.float32) / 65535.0, 0.0, 1.0)
            except Exception:
                pass

    # Infer depth at clean working resolution (~518px max edge) before upscaling cleanly
    max_edge = 518
    aspect = img.width / max(1, img.height)
    if aspect >= 1.0:
        inf_w = min(max_edge, max(128, img.width))
        inf_h = max(64, int(round(inf_w / aspect)))
    else:
        inf_h = min(max_edge, max(128, img.height))
        inf_w = max(64, int(round(inf_h * aspect)))
    small_pil = img.resize((inf_w, inf_h), Image.Resampling.BILINEAR)
    small_rgb = np.asarray(small_pil, dtype=np.uint8)

    depth_inf: Optional[np.ndarray] = None
    bundle = _load_depth_anything_model()
    if bundle is not None:
        processor, model, dev = bundle
        try:
            import torch

            inputs = processor(images=small_pil, return_tensors="pt")
            with _DEPTH_LOCK, torch.inference_mode():
                if dev == "cuda":
                    inputs = {k: v.to("cuda", dtype=torch.float16) if torch.is_floating_point(v) else v.to("cuda") for k, v in inputs.items()}
                elif dev == "mps":
                    inputs = {k: v.to("mps") for k, v in inputs.items()}
                out = model(**inputs)
                pred = out.predicted_depth
                pred_resized = torch.nn.functional.interpolate(
                    pred.unsqueeze(1).float(),
                    size=(inf_h, inf_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                raw_arr = pred_resized.detach().cpu().numpy().astype(np.float32)

            lo = float(np.percentile(raw_arr, 2))
            hi = float(np.percentile(raw_arr, 98))
            if hi - lo > 1e-4:
                norm = np.clip((raw_arr - lo) / (hi - lo), 0.0, 1.0)
                # Edge-aware bilateral refine to suppress subtle transformer patch blockiness
                norm_u8 = (norm * 255.0).astype(np.uint8)
                refined = cv2.bilateralFilter(norm_u8, d=7, sigmaColor=18, sigmaSpace=8).astype(np.float32) / 255.0
                depth_inf = np.clip(refined, 0.0, 1.0).astype(np.float32)
        except Exception:
            depth_inf = None

    if depth_inf is None:
        depth_inf = _heuristic_depth_map(small_rgb, focal_point=focal_point)

    if (inf_w, inf_h) != (out_w, out_h):
        depth_out = cv2.resize(depth_inf, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        depth_out = np.clip(depth_out, 0.0, 1.0).astype(np.float32)
    else:
        depth_out = depth_inf

    if cache_file is not None:
        try:
            u16 = np.clip(np.round(depth_out * 65535.0), 0, 65535).astype(np.uint16)
            cv2.imwrite(str(cache_file), u16)
        except Exception:
            pass

    return depth_out


def compute_depth_suitability(depth: np.ndarray) -> float:
    """
    Score [0.35, 1.0] measuring foreground-to-background separation clarity.

    Flat/cluttered scenes receive lower scores so parallax amplitude automatically dials
    down and avoids distracting wobbling on non-dimensional frames.
    """
    if depth.size == 0:
        return 0.5
    p92 = float(np.percentile(depth, 92))
    p12 = float(np.percentile(depth, 12))
    span = max(0.0, min(1.0, p92 - p12))
    # High-frequency spatial gradient density across depth map indicates busy fragmented clutter
    gx = np.abs(np.diff(depth, axis=1)).mean() if depth.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(depth, axis=0)).mean() if depth.shape[0] > 1 else 0.0
    clutter_penalty = min(0.45, max(0.0, (float(gx + gy) - 0.028) * 6.0))
    raw_score = span * (1.0 - clutter_penalty)
    return float(max(0.35, min(1.0, 0.30 + raw_score * 0.90)))


def _smoothstep(x: np.ndarray) -> np.ndarray:
    c = np.clip(x, 0.0, 1.0)
    return c * c * (3.0 - 2.0 * c)


def _pick_valley_threshold(depth: np.ndarray, lo: float, hi: float, fallback: float) -> float:
    """Pick a depth threshold sitting in a density valley so cohesive subjects are never sliced mid-face."""
    if lo >= hi:
        return float(fallback)
    hist, edges = np.histogram(depth.ravel(), bins=64, range=(0.0, 1.0))
    hist_f = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1), (1, 5), sigmaX=1.2).ravel()
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= lo) & (centers <= hi)
    if not np.any(mask):
        return float(fallback)
    # Slightly favor center of interval to prevent extreme empty-bin snaps
    penalty = 0.08 * float(hist_f.max() + 1.0) * np.abs(centers - fallback) / max(0.05, hi - lo)
    score = np.where(mask, hist_f + penalty, 1e12)
    return float(centers[int(np.argmin(score))])


def _inpaint_occluded_fringe(
    rgb_quarter: np.ndarray,
    occluder_mask_quarter: np.ndarray,
    dilate_px: int,
) -> np.ndarray:
    """Inpaint background RGB strictly inside the occluder core behind closer strata."""
    ksize = max(3, int(dilate_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    hole_u8 = (occluder_mask_quarter > 0.5).astype(np.uint8) * 255
    if int(hole_u8.max()) == 0:
        return rgb_quarter
    # Dilate slightly outward only to avoid TELEA sampling foreground boundary colors into the fill,
    # then mask the blended output strictly inside the occluded interior so edges never halo.
    dilated = cv2.dilate(hole_u8, kernel, iterations=1)
    bgr = cv2.cvtColor(rgb_quarter, cv2.COLOR_RGB2BGR)
    filled_bgr = cv2.inpaint(bgr, dilated, inpaintRadius=max(4, ksize // 3), flags=cv2.INPAINT_TELEA)
    filled_rgb = cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2RGB)
    core_mask = cv2.GaussianBlur(hole_u8.astype(np.float32) / 255.0, (0, 0), sigmaX=max(1.2, ksize * 0.16))
    core_mask = np.clip(core_mask, 0.0, 1.0)[:, :, None]
    out = filled_rgb.astype(np.float32) * core_mask + rgb_quarter.astype(np.float32) * (1.0 - core_mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_mpi_depth_layers(
    bg_sharp: Image.Image,
    depth: np.ndarray,
    scale_factor: float,
) -> dict[str, Any]:
    """
    Build 4 ordered RGBA Multi-Plane Image (MPI) depth strata + disocclusion-inpainted fills.

    Strata L0 (far, opaque) -> L3 (hero foreground, crisp feathered alpha). Depth cutoffs snap to
    histogram valleys so portraits and foreground subjects stay intact in a single cohesive plane,
    and inpainting stays strictly behind opaque closer pixels so edges remain clean.
    """
    bg_w, bg_h = bg_sharp.size
    if depth.shape[:2] != (bg_h, bg_w):
        depth_full = cv2.resize(depth.astype(np.float32), (bg_w, bg_h), interpolation=cv2.INTER_CUBIC)
        depth_full = np.clip(depth_full, 0.0, 1.0)
    else:
        depth_full = np.clip(depth.astype(np.float32), 0.0, 1.0)

    suitability = compute_depth_suitability(depth_full)

    # Valley-snapped depth bands keep coherent portraits/subjects intact inside L3
    p25 = float(np.clip(np.percentile(depth_full, 25), 0.16, 0.38))
    p52 = float(np.clip(np.percentile(depth_full, 52), max(p25 + 0.14, 0.40), 0.62))
    p72 = float(np.clip(np.percentile(depth_full, 72), max(p52 + 0.14, 0.60), 0.82))
    t1 = _pick_valley_threshold(depth_full, 0.15, min(0.40, p52 - 0.12), p25)
    t2 = _pick_valley_threshold(depth_full, max(t1 + 0.12, 0.38), min(0.66, p72 - 0.10), p52)
    t3 = _pick_valley_threshold(depth_full, max(t2 + 0.12, 0.56), 0.84, p72)
    feather = 0.055

    qw, qh = max(48, bg_w // 4), max(27, bg_h // 4)
    rgb_full = np.asarray(bg_sharp, dtype=np.uint8)
    rgb_q = cv2.resize(rgb_full, (qw, qh), interpolation=cv2.INTER_AREA)
    depth_q = cv2.resize(depth_full, (qw, qh), interpolation=cv2.INTER_AREA)

    dilate_q = max(7, int(round(9.5 * scale_factor)))
    blur_sigma_far = max(1.4, 2.0 * scale_factor)

    # Stratum 0 (Far): inpaint strictly behind opaque L1+ core (depth >= t1 + 0.025) + subtle natural far softness
    far_inpainted_q = _inpaint_occluded_fringe(rgb_q, (depth_q >= t1 + 0.025).astype(np.float32), dilate_px=dilate_q)
    far_soft_q = cv2.GaussianBlur(far_inpainted_q, (0, 0), sigmaX=blur_sigma_far)
    far_blur_mix = np.clip((1.0 - (depth_q / max(0.08, t1 + 0.08))) * 0.28, 0.0, 0.28)[:, :, None]
    far_comp_q = np.clip(far_inpainted_q.astype(np.float32) * (1.0 - far_blur_mix) + far_soft_q.astype(np.float32) * far_blur_mix, 0, 255).astype(np.uint8)
    far_comp_full = cv2.resize(far_comp_q, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    # Apply inpaint replacement ONLY strictly behind opaque L1 (depth >= t1 + feather * 0.55) so exposed fringe never blurs
    occ0_full = _smoothstep((depth_full - (t1 + feather * 0.50)) / 0.05)[:, :, None]
    far_defocus_w = np.clip((1.0 - (depth_full / max(0.08, t1 + 0.08))) * 0.28, 0.0, 0.28)[:, :, None]
    l0_rgb = np.clip(
        rgb_full.astype(np.float32) * (1.0 - np.maximum(occ0_full, far_defocus_w))
        + far_comp_full.astype(np.float32) * np.maximum(occ0_full, far_defocus_w),
        0,
        255,
    ).astype(np.uint8)
    l0_img = Image.fromarray(l0_rgb, mode="RGB").convert("RGBA")

    # Helper for upper strata L1, L2, L3
    thresholds = [(t1, t2), (t2, t3), (t3, None)]
    layers: list[Image.Image] = [l0_img]
    for lo_t, next_t in thresholds:
        alpha_full = _smoothstep((depth_full - lo_t) / feather)
        alpha_u8 = np.clip(alpha_full * 255.0, 0, 255).astype(np.uint8)
        alpha_img = Image.fromarray(alpha_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=max(0.8, 1.0 * scale_factor)))

        if next_t is not None:
            inpainted_q = _inpaint_occluded_fringe(rgb_q, (depth_q >= next_t + 0.025).astype(np.float32), dilate_px=dilate_q)
            inpainted_full = cv2.resize(inpainted_q, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
            occ_next = _smoothstep((depth_full - (next_t + feather * 0.50)) / 0.05)[:, :, None]
            layer_rgb = np.clip(
                rgb_full.astype(np.float32) * (1.0 - occ_next) + inpainted_full.astype(np.float32) * occ_next,
                0,
                255,
            ).astype(np.uint8)
        else:
            layer_rgb = rgb_full

        l_rgba = Image.fromarray(layer_rgb, mode="RGB").convert("RGBA")
        l_rgba.putalpha(alpha_img)
        layers.append(l_rgba)

    return {
        "layers": layers,
        "suitability": suitability,
        "thresholds": (t1, t2, t3),
    }


def build_rack_focus_assets(
    raw_primary: Image.Image,
    width: int,
    height: int,
    focal_point: tuple[float, float],
    scale_factor: float,
    cache_dir: Optional[Path] = None,
    photo_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build continuous depth-driven optical rack-focus assets (`rf_sharp`, `rf_bokeh`, `rf_depth_q`, `focus_span`).
    Always pulls smoothly from scenic far plane onto the foreground hero subject so shots conclude tack-sharp.
    """
    oversample = 1.05
    work_w = max(64, int(round(width * oversample)))
    work_h = max(64, int(round(height * oversample)))
    fitted = ImageOps.fit(raw_primary, (work_w, work_h), method=Image.Resampling.BILINEAR)

    depth_full = estimate_depth_map(
        fitted,
        focal_point=focal_point,
        cache_dir=cache_dir,
        photo_key=photo_key,
        target_size=(work_w, work_h),
    )

    qw, qh = max(48, work_w // 4), max(27, work_h // 4)
    small_rgb = fitted.resize((qw, qh), Image.Resampling.BILINEAR)
    bokeh_small = small_rgb.filter(ImageFilter.GaussianBlur(radius=max(2.6, 3.6 * scale_factor)))
    bokeh_full = bokeh_small.resize((work_w, work_h), Image.Resampling.BILINEAR)

    depth_q = cv2.resize(depth_full, (qw, qh), interpolation=cv2.INTER_AREA)
    # Locate dominant foreground subject inside compositional region [0.18..0.82] x [0.12..0.88]
    roi = depth_q[int(qh * 0.12) : max(int(qh * 0.12) + 2, int(qh * 0.88)), int(qw * 0.18) : max(int(qw * 0.18) + 2, int(qw * 0.82))]
    fg_hero_d = float(np.percentile(roi, 88)) if roi.size > 0 else float(np.percentile(depth_q, 86))
    fx_i = max(0, min(qw - 1, int(round(focal_point[0] * (qw - 1)))))
    fy_i = max(0, min(qh - 1, int(round(focal_point[1] * (qh - 1)))))
    pt_d = float(np.median(depth_q[max(0, fy_i - 3) : fy_i + 4, max(0, fx_i - 3) : fx_i + 4]))
    # If explicit focal point hits foreground subject, honor it; otherwise lock onto compositional foreground hero
    hero_d = pt_d if pt_d >= fg_hero_d - 0.18 else fg_hero_d
    far_d = float(np.percentile(depth_q, 14))
    if abs(hero_d - far_d) < 0.22:
        start_focus = max(0.08, hero_d - 0.48)
    else:
        start_focus = far_d
    end_focus = hero_d

    return {
        "sharp": fitted,
        "bokeh": bokeh_full,
        "depth_q": depth_q.astype(np.float32),
        "work_size": (work_w, work_h),
        "focus_start": float(np.clip(start_focus, 0.05, 0.95)),
        "focus_end": float(np.clip(end_focus, 0.05, 0.95)),
    }
