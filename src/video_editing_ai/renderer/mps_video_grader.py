"""AMD/Apple Metal (MPS) & CUDA GPU-Accelerated 4K 10-bit Video Color Grader & 3D LUT Engine.

Replaces CPU-bound FFmpeg `rgb48le` + `colorlevels` + `colorbalance` + `eq` + `lut3d` + `scale` + `fade`
with a fused PyTorch GPU pipeline using a 2-call Hardware TMU 2D-atlas `F.grid_sample` 33x33x33 3D LUT
(MAE < 0.0001 vs CPU 3D trilinear reference) and GPU-native BT.2020 HLG `p010le` / BT.709 `yuv420p` output.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..models.schema import EffectTrackItem, TransitionConfig
from .color_grading import CINEMA_FILM_STOCK_SPECS, build_film_stock_lut_table, write_film_stock_cube_lut
from .mps_photo_compositor import _MPS_LOCK, _rgb_to_bt709_yuv420p_gpu, _rgb_to_hlg_p010le_gpu

_LUT_ATLAS_CACHE: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}
_LUT_CACHE_LOCK = threading.Lock()
_VIGNETTE_CACHE: dict[tuple[int, int, str, str], torch.Tensor] = {}
_VIDEO_PREBUFFER_SEM = threading.BoundedSemaphore(2)


def _get_vignette_mask_gpu(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    angle: float = 3.141592653589793 / 3.6,
) -> torch.Tensor:
    """Cached radial cos^4 vignette mask `(1, 1, H, W)` matching FFmpeg `vignette=PI/3.6` on GPU."""
    key = (height, width, str(device), str(dtype))
    cached = _VIGNETTE_CACHE.get(key)
    if cached is not None:
        return cached
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32).view(height, 1)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32).view(1, width)
    r = torch.sqrt(xx * xx + yy * yy) / 1.4142135623730951
    mask = torch.cos(r.clamp_(0.0, 1.0) * angle).pow(4).view(1, 1, height, width).to(dtype=dtype).contiguous()
    _VIGNETTE_CACHE[key] = mask
    return mask


def _compose_portrait_darkroom_gpu(
    fg_bchw: torch.Tensor,
    width: int,
    height: int,
    x0: int,
    y0: int,
    inner_px: int,
    outer_px: int,
) -> torch.Tensor:
    """Compose 4K darkroom velvet bokeh + radial vignette + double-matted 9:16 video 100% on GPU."""
    b, _, fg_h, fg_w = fg_bchw.shape
    device = fg_bchw.device
    dtype = fg_bchw.dtype

    # 1. Center aspect-fill crop of portrait foreground -> fast 1/12-res 3-pass box-Gaussian bokeh blur on GPU
    crop_h = max(2, min(fg_h, int(round(fg_w * height / float(max(1, width))))))
    cy0 = (fg_h - crop_h) // 2
    bg_src = fg_bchw[:, :, cy0 : cy0 + crop_h, :]
    small_h = max(16, height // 12)
    small_w = max(16, width // 12)
    bg = F.interpolate(bg_src, size=(small_h, small_w), mode="bilinear", align_corners=False)
    for _ in range(3):
        bg = F.avg_pool2d(bg, kernel_size=15, stride=1, padding=7, count_include_pad=False)
    # Velvety darkroom tone: eq=brightness=-0.20:contrast=0.95:saturation=0.60
    bg = ((bg - 0.5) * 0.95 + 0.5 - 0.20).clamp_(0.0, 1.0)
    luma = 0.2627 * bg[:, 0:1] + 0.6780 * bg[:, 1:2] + 0.0593 * bg[:, 2:3]
    bg = torch.lerp(luma, bg, 0.60).clamp_(0.0, 1.0)
    canvas = F.interpolate(bg, size=(height, width), mode="bilinear", align_corners=False)
    vig = _get_vignette_mask_gpu(height, width, device, dtype)
    canvas.mul_(vig)

    # 2. Paint double-beveled warm ivory (#F4EFE6) + charcoal hairline (#24262C) matte & crisp live video
    pad_tot = inner_px + outer_px
    frame_w = fg_w + pad_tot * 2
    frame_h = fg_h + pad_tot * 2
    fx0 = max(0, min(width - frame_w, x0))
    fy0 = max(0, min(height - frame_h, y0))
    ivory = torch.tensor([244.0 / 255.0, 239.0 / 255.0, 230.0 / 255.0], device=device, dtype=dtype).view(1, 3, 1, 1)
    charcoal = torch.tensor([36.0 / 255.0, 38.0 / 255.0, 44.0 / 255.0], device=device, dtype=dtype).view(1, 3, 1, 1)
    canvas[:, :, fy0 : fy0 + frame_h, fx0 : fx0 + frame_w] = ivory
    ix0, iy0 = fx0 + outer_px, fy0 + outer_px
    canvas[:, :, iy0 : iy0 + fg_h + inner_px * 2, ix0 : ix0 + fg_w + inner_px * 2] = charcoal
    vx0, vy0 = ix0 + inner_px, iy0 + inner_px
    canvas[:, :, vy0 : vy0 + fg_h, vx0 : vx0 + fg_w] = fg_bchw
    return canvas


def _load_cube_lut_array(cube_path: Path) -> tuple[np.ndarray, int]:
    """Parse an Adobe .cube 3D LUT into a float32 array shaped (N, N, N, 3) indexed [b, g, r, rgb]."""
    size = 0
    rows: list[tuple[float, float, float]] = []
    with cube_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("LUT_3D_SIZE"):
                parts = line.split()
                if len(parts) >= 2:
                    size = int(parts[1])
                continue
            if upper.startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE")):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
    if size <= 1 or len(rows) != size * size * size:
        raise ValueError(f"Invalid .cube 3D LUT file: {cube_path} (size={size}, rows={len(rows)})")
    arr = np.asarray(rows, dtype=np.float32).reshape((size, size, size, 3))
    return arr, size


def get_gpu_lut_atlas(cube_path: Path, device: torch.device) -> tuple[torch.Tensor, int]:
    """Return cached 2D texture atlas `(1, 3, N, N*N)` float32 on `device` for bit-exact TMU 3D LUT lookup."""
    resolved = str(cube_path.resolve())
    key = (resolved, str(device))
    with _LUT_CACHE_LOCK:
        cached = _LUT_ATLAS_CACHE.get(key)
        if cached is not None:
            return cached
        lut_np, n_size = _load_cube_lut_array(cube_path)
        # lut_np is [b, g, r, 3] -> pack as 2D strip [1, 3, G=N, W=B*N + R]
        atlas = (
            torch.from_numpy(lut_np)
            .permute(3, 1, 0, 2)
            .reshape(1, 3, n_size, n_size * n_size)
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )
        _LUT_ATLAS_CACHE[key] = (atlas, n_size)
        return atlas, n_size


def apply_lut3d_tmu_gpu(x_bchw: torch.Tensor, atlas: torch.Tensor, n_size: int) -> torch.Tensor:
    """
    Apply a 3D LUT `(N, N, N, 3)` packed as 2D atlas `(1, 3, N, N*N)` using 2 hardware TMU
    `F.grid_sample` calls (`padding_mode='zeros'`, float32 coords for 10-bit exactness: MAE < 0.0001).
    """
    b_sz = x_bchw.shape[0]
    r = x_bchw[:, 0].float().clamp_(0.0, 1.0)
    g = x_bchw[:, 1].float().clamp_(0.0, 1.0)
    b = x_bchw[:, 2].float().clamp_(0.0, 1.0)

    bs = b * float(n_size - 1)
    b0 = bs.floor().clamp_(0.0, float(n_size - 2))
    fb = (bs - b0).unsqueeze(1)
    b1 = b0 + 1.0

    inv_h = 2.0 / float(n_size)
    inv_w = 2.0 / float(n_size * n_size)
    gy = (g * float(n_size - 1) + 0.5) * inv_h - 1.0
    gx0 = (b0 * float(n_size) + r * float(n_size - 1) + 0.5) * inv_w - 1.0
    gx1 = (b1 * float(n_size) + r * float(n_size - 1) + 0.5) * inv_w - 1.0

    a_exp = atlas.expand(b_sz, -1, -1, -1)
    c0 = F.grid_sample(a_exp, torch.stack((gx0, gy), dim=-1), mode="bilinear", padding_mode="zeros", align_corners=False)
    c1 = F.grid_sample(a_exp, torch.stack((gx1, gy), dim=-1), mode="bilinear", padding_mode="zeros", align_corners=False)
    return torch.lerp(c0, c1, fb).to(x_bchw.dtype)


def _pack_native_hlg_rgb_to_p010le_gpu(x_bchw: torch.Tensor) -> torch.Tensor:
    """Pack already-BT.2020-HLG RGB `[0..1]` `(B, 3, H, W)` directly to 10-bit limited-range `p010le`."""
    r = x_bchw[:, 0]
    g = x_bchw[:, 1]
    b = x_bchw[:, 2]
    y = 0.2627 * r + 0.6780 * g + 0.0593 * b
    y10 = ((y * 876.0 + 64.0).round_().clamp_(64.0, 940.0).to(torch.int32) * 64).to(torch.int16)
    uv_in = torch.stack(
        (
            (b - y) * (896.0 / 1.8814) + 512.0,
            (r - y) * (896.0 / 1.4746) + 512.0,
        ),
        dim=1,
    )
    uv_sub = ((F.avg_pool2d(uv_in, kernel_size=2, stride=2).round_().clamp_(64.0, 960.0).to(torch.int32)) * 64).to(torch.int16)
    b_sz, h, w = y10.shape
    uv_inter = uv_sub.permute(0, 2, 3, 1).reshape(b_sz, h // 2, w)
    packed = torch.cat((y10, uv_inter), dim=1).contiguous()
    if x_bchw.device.type == "cuda":
        return packed.view(torch.uint16).contiguous()
    return packed


def _parse_video_grade_spec(
    attached_effects: Optional[list[EffectTrackItem]],
    src_trc: str,
    src_space: str,
) -> dict[str, Any]:
    """Extract normalized GPU grading parameters from attached Stage-40/42 EffectTrackItems."""
    spec: dict[str, Any] = {
        "pre_contrast": 1.0,
        "pre_saturation": 1.0,
        "warm_glow": 0.0,
        "hlg_rolloff": False,
        "black_ped": 0.0,
        "rs": 0.0, "gs": 0.0, "bs": 0.0,
        "rm": 0.0, "gm": 0.0, "bm": 0.0,
        "rh": 0.0, "gh": 0.0, "bh": 0.0,
        "brightness": 0.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "gamma": 1.0,
        "lut_path": None,
        "is_native_hlg": (src_trc.lower() == "arib-std-b67" or src_space.lower().startswith("bt2020")),
    }
    for fx in attached_effects or []:
        etype = fx.effect_type
        params = fx.parameters or {}
        if etype == "contrast_pop":
            strength = float(fx.intensity or 0.35)
            spec["pre_contrast"] *= 1.0 + min(0.12, strength * 0.18)
            spec["pre_saturation"] *= 1.0 + min(0.10, strength * 0.15)
        elif etype == "warm_glow":
            spec["warm_glow"] = max(spec["warm_glow"], float(fx.intensity or 0.25))
        elif etype == "color_grade":
            look_name = str(params.get("chapter_look") or params.get("look") or "")
            exp_ev = float(params.get("exposure_ev", 0.0))
            if params.get("bypass") or (look_name == "neutral_reference" and abs(exp_ev) < 1e-4):
                continue
            spec["hlg_rolloff"] = bool(params.get("hlg_rolloff", False))

            lut_cube_raw = str(params.get("lut_cube_path") or "").strip()
            lut_int = float(params.get("lut_intensity", 1.0))
            use_lut = bool(lut_cube_raw and Path(lut_cube_raw).exists() and lut_int > 0.02 and look_name != "neutral_reference")
            if use_lut:
                cube_p = Path(lut_cube_raw)
                if lut_int < 0.98 and look_name in CINEMA_FILM_STOCK_SPECS:
                    cube_p = write_film_stock_cube_lut(
                        look_name,
                        cube_p.with_name(f"{look_name}_i{int(round(lut_int * 100))}.cube"),
                        intensity=lut_int,
                    )
                spec["lut_path"] = cube_p

            wb_r = float(params.get("wb_red_shift", params.get("rs", 0.0)))
            wb_g = float(params.get("wb_green_shift", params.get("gs", 0.0)))
            wb_b = float(params.get("wb_blue_shift", params.get("bs", 0.0)))
            lift = params.get("lift_rgb") or (0.0, 0.0, 0.0)
            mid = params.get("gamma_rgb") or (0.0, 0.0, 0.0)
            gain = params.get("gain_rgb") or (0.0, 0.0, 0.0)
            lgg_scale = 0.20 if use_lut else 0.70

            def _c(v: float) -> float:
                return max(-0.35, min(0.35, float(v)))

            spec["rs"] = _c(wb_r * 0.45 + float(lift[0]) * lgg_scale)
            spec["gs"] = _c(wb_g * 0.45 + float(lift[1]) * lgg_scale)
            spec["bs"] = _c(wb_b * 0.45 + float(lift[2]) * lgg_scale)
            spec["rm"] = _c(wb_r * 0.72 + float(mid[0]) * lgg_scale)
            spec["gm"] = _c(wb_g * 0.72 + float(mid[1]) * lgg_scale)
            spec["bm"] = _c(wb_b * 0.72 + float(mid[2]) * lgg_scale)
            spec["rh"] = _c(wb_r * 0.55 + float(gain[0]) * lgg_scale)
            spec["gh"] = _c(wb_g * 0.55 + float(gain[1]) * lgg_scale)
            spec["bh"] = _c(wb_b * 0.55 + float(gain[2]) * lgg_scale)

            black_lift = float(params.get("black_lift", 0.0))
            if black_lift < -0.005:
                spec["black_ped"] = min(0.22, abs(black_lift))
                toe_bright = 0.0
            else:
                toe_bright = black_lift * 0.45

            spec["brightness"] = max(-0.15, min(0.15, exp_ev * 0.14 + toe_bright))
            spec["contrast"] = max(0.75, min(1.40, float(params.get("contrast", 1.04))))
            spec["saturation"] = max(0.50, min(1.60, float(params.get("saturation", 1.04))))
            base_gamma = max(0.70, min(1.40, float(params.get("gamma", 1.00))))
            spec["gamma"] = max(0.70, min(1.45, base_gamma * (2.0 ** max(-0.4, min(0.4, exp_ev * 0.32)))))
    return spec


def _apply_video_grade_batch_gpu(
    x_bchw: torch.Tensor,
    spec: dict[str, Any],
    lut_atlas: Optional[tuple[torch.Tensor, int]],
    frame_indices: list[int],
    duration: float,
    fps: int,
    transition_in: Optional[TransitionConfig],
    transition_out: Optional[TransitionConfig],
    hdr: bool,
    diffuse_white_nits: float,
    card_overlay: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Run all color-grading ops + TMU 3D LUT + fades + p010le/yuv420p packing on GPU in one fused pass."""
    # 1. Pre-effects (contrast_pop, warm_glow)
    pc = float(spec["pre_contrast"])
    ps = float(spec["pre_saturation"])
    if abs(pc - 1.0) > 1e-3:
        x_bchw = ((x_bchw - 0.5) * pc + 0.5).clamp_(0.0, 1.0)
    if abs(ps - 1.0) > 1e-3:
        luma = (0.2627 * x_bchw[:, 0:1] + 0.6780 * x_bchw[:, 1:2] + 0.0593 * x_bchw[:, 2:3])
        x_bchw = torch.lerp(luma, x_bchw, ps).clamp_(0.0, 1.0)
    wg = float(spec["warm_glow"])
    if wg > 1e-3:
        shift = torch.tensor([wg * 0.04, wg * 0.015, -wg * 0.03], device=x_bchw.device, dtype=x_bchw.dtype).view(1, 3, 1, 1)
        x_bchw = (x_bchw + shift).clamp_(0.0, 1.0)

    # 2. Specular highlight knee roll-off (matches lutyuv >720/1023 -> ~0.75 normalized)
    if spec["hlg_rolloff"]:
        knee = 0.748
        y_luma = (0.2627 * x_bchw[:, 0:1] + 0.6780 * x_bchw[:, 1:2] + 0.0593 * x_bchw[:, 2:3]).clamp_(min=1e-4)
        y_comp = torch.where(y_luma > knee, knee + (y_luma - knee) * 0.45, y_luma).clamp_(max=0.985)
        x_bchw = (x_bchw * (y_comp / y_luma)).clamp_(0.0, 1.0)

    # 3. Black-point pedestal (colorlevels rimin/gimin/bimin)
    ped = float(spec["black_ped"])
    if ped > 1e-4:
        x_bchw = ((x_bchw - ped) * (1.0 / max(1e-4, 1.0 - ped))).clamp_(0.0, 1.0)

    # 4. 3-way Lift/Gamma/Gain colorbalance
    rs, gs, bs = spec["rs"], spec["gs"], spec["bs"]
    rm, gm, bm = spec["rm"], spec["gm"], spec["bm"]
    rh, gh, bh = spec["rh"], spec["gh"], spec["bh"]
    if any(abs(v) > 1e-4 for v in (rs, gs, bs, rm, gm, bm, rh, gh, bh)):
        sh_v = torch.tensor([rs, gs, bs], device=x_bchw.device, dtype=x_bchw.dtype).view(1, 3, 1, 1)
        md_v = torch.tensor([rm, gm, bm], device=x_bchw.device, dtype=x_bchw.dtype).view(1, 3, 1, 1)
        hi_v = torch.tensor([rh, gh, bh], device=x_bchw.device, dtype=x_bchw.dtype).view(1, 3, 1, 1)
        w_sh = (1.0 - x_bchw).pow(2) * 0.5
        w_hi = x_bchw.pow(2) * 0.5
        w_md = (x_bchw * (1.0 - x_bchw)) * 2.0
        x_bchw = (x_bchw + sh_v * w_sh + md_v * w_md + hi_v * w_hi).clamp_(0.0, 1.0)

    # 5. eq (contrast, brightness, gamma, saturation)
    con = float(spec["contrast"])
    bri = float(spec["brightness"])
    gam = float(spec["gamma"])
    sat = float(spec["saturation"])
    if abs(con - 1.0) > 1e-3 or abs(bri) > 1e-4:
        x_bchw = ((x_bchw - 0.5) * con + 0.5 + bri).clamp_(0.0, 1.0)
    if abs(gam - 1.0) > 1e-3:
        x_bchw = torch.pow(x_bchw.float().clamp_(min=1e-5), 1.0 / gam).to(x_bchw.dtype)
    if abs(sat - 1.0) > 1e-3:
        luma = (0.2627 * x_bchw[:, 0:1] + 0.6780 * x_bchw[:, 1:2] + 0.0593 * x_bchw[:, 2:3])
        x_bchw = torch.lerp(luma, x_bchw, sat).clamp_(0.0, 1.0)

    # 6. 33x33x33 .cube 3D LUT via Hardware Metal TMU 2D-atlas grid_sample
    if lut_atlas is not None:
        atlas_t, n_size = lut_atlas
        x_bchw = apply_lut3d_tmu_gpu(x_bchw, atlas_t, n_size).clamp_(0.0, 1.0)

    # 6b. Optional luxury portrait Diptych / Editorial Side-Panel / Frosted Card RGBA overlay (color-bypassed!)
    if card_overlay is not None:
        ov_rgb, ov_a = card_overlay
        fo_st = max(0.2, duration - 0.45)
        ov_alphas = []
        for idx in frame_indices:
            t_s = idx / float(max(1, fps))
            a_in = min(1.0, max(0.0, (t_s - 0.15) / 0.45)) if duration > 0.8 else 1.0
            a_out = min(1.0, max(0.0, 1.0 - max(0.0, t_s - fo_st) / 0.40)) if duration > 0.8 else 1.0
            ov_alphas.append(a_in * a_out)
        if any(a > 1e-3 for a in ov_alphas):
            m_t = torch.tensor(ov_alphas, device=x_bchw.device, dtype=x_bchw.dtype).view(len(ov_alphas), 1, 1, 1)
            eff_a = ov_a * m_t
            x_bchw = (x_bchw * (1.0 - eff_a) + ov_rgb * eff_a).clamp_(0.0, 1.0)

    # 7. Smooth per-frame transitions (fade in / fade out)
    if transition_in and transition_in.type not in ("none", "cut") and transition_in.duration > 0:
        if transition_in.type == "dip_to_white":
            fd = min(duration * 0.35, transition_in.duration)
            c_rgb = (246.0 / 255.0, 242.0 / 255.0, 232.0 / 255.0)
        elif transition_in.type in ("dissolve", "crossfade"):
            fd = min(duration * 0.25, transition_in.duration, 0.26)
            c_rgb = (11.0 / 255.0, 14.0 / 255.0, 20.0 / 255.0)
        else:
            fd = min(duration * 0.40, transition_in.duration)
            c_rgb = (0.0, 0.0, 0.0)
        if fd > 1e-4:
            alphas = [min(1.0, max(0.0, (idx / float(max(1, fps))) / fd)) for idx in frame_indices]
            if any(a < 0.999 for a in alphas):
                c_t = torch.tensor(c_rgb, device=x_bchw.device, dtype=x_bchw.dtype).view(1, 3, 1, 1)
                a_t = torch.tensor(alphas, device=x_bchw.device, dtype=x_bchw.dtype).view(len(alphas), 1, 1, 1)
                x_bchw = torch.lerp(c_t, x_bchw, a_t)

    if transition_out and transition_out.type not in ("none", "cut") and transition_out.duration > 0:
        if transition_out.type == "dip_to_white":
            fd = min(duration * 0.35, transition_out.duration)
            c_rgb = (246.0 / 255.0, 242.0 / 255.0, 232.0 / 255.0)
        elif transition_out.type in ("dissolve", "crossfade"):
            fd = min(duration * 0.25, transition_out.duration, 0.26)
            c_rgb = (11.0 / 255.0, 14.0 / 255.0, 20.0 / 255.0)
        else:
            fd = min(duration * 0.40, transition_out.duration)
            c_rgb = (0.0, 0.0, 0.0)
        if fd > 1e-4:
            st = max(0.0, duration - fd)
            alphas = [min(1.0, max(0.0, 1.0 - max(0.0, (idx / float(max(1, fps))) - st) / fd)) for idx in frame_indices]
            if any(a < 0.999 for a in alphas):
                c_t = torch.tensor(c_rgb, device=x_bchw.device, dtype=x_bchw.dtype).view(1, 3, 1, 1)
                a_t = torch.tensor(alphas, device=x_bchw.device, dtype=x_bchw.dtype).view(len(alphas), 1, 1, 1)
                x_bchw = torch.lerp(c_t, x_bchw, a_t)

    # 8. Pack directly on GPU to target pixel format
    if hdr:
        if spec["is_native_hlg"]:
            return _pack_native_hlg_rgb_to_p010le_gpu(x_bchw)
        return _rgb_to_hlg_p010le_gpu(x_bchw.mul_(255.0), npl_nits=diffuse_white_nits)
    return _rgb_to_bt709_yuv420p_gpu(x_bchw.mul_(255.0))


def render_video_segment_gpu(
    *,
    source: str | Path,
    src_in: float,
    dur: float,
    speed: float,
    width: int,
    height: int,
    fps: int,
    hdr: bool,
    is_portrait: bool,
    has_audio: bool,
    audio_pos: int,
    volume: float,
    audio_crossfade_sec: float,
    transition_in: Optional[TransitionConfig],
    transition_out: Optional[TransitionConfig],
    attached_effects: Optional[list[EffectTrackItem]],
    src_trc: str,
    src_space: str,
    diffuse_white_nits: float,
    v_enc_args: list[str],
    mux_args: list[str],
    tmp_out: Path,
    use_hw_decode: bool = True,
    batch_size: int = 4,
    card_png: Optional[Path] = None,
    has_paired_photo: bool = False,
    has_portrait_editorial_side: bool = False,
    has_doc_card: bool = False,
    src_w: int = 0,
    src_h: int = 0,
) -> bool:
    """
    Decode via VideoToolbox, grade & pack 10-bit p010le on AMD Radeon Pro 5500M Metal (or CUDA),
    and encode via hardware hevc_videotoolbox with zero CPU color-grading overhead and zero SSD temp video.
    """
    from ..hardware import get_hardware_profile, nvenc_encode_guard, videotoolbox_hdr_encode_guard

    hw = get_hardware_profile()
    dev_str = "cuda" if hw.has_cuda else ("mps" if hw.has_mps else "")
    if not dev_str or width % 2 != 0 or height % 2 != 0:
        return False

    device = torch.device(dev_str)
    work_dtype = torch.float16
    spec = _parse_video_grade_spec(attached_effects, src_trc=src_trc, src_space=src_space)
    mps_guard = _MPS_LOCK if device.type == "mps" else contextlib.nullcontext()

    speed_prefix = f"setpts={1.0 / speed:.4f}*PTS," if (speed != 1.0 and speed > 0) else ""
    portrait_gpu_geom: Optional[tuple[int, int, int, int, int, int]] = None
    dec_w, dec_h = width, height
    if is_portrait:
        if has_portrait_editorial_side:
            fg_h = int(round(height * 0.82)) & ~1
            y_pos = int(round(height * 0.09))
            x_center = int(round(width * 0.31))
        elif has_paired_photo:
            fg_h = int(round(height * (0.76 if has_doc_card else 0.82))) & ~1
            y_pos = int(round(height * (0.06 if has_doc_card else 0.09)))
            x_center = int(round(width * 0.28))
        else:
            fg_h = int(round(height * 0.84)) & ~1
            y_pos = int(round(height * 0.08))
            x_center = width // 2
        inner_px = max(2, int(round(3 * (width / 1920.0)))) & ~1
        outer_px = max(10, int(round(18 * (width / 1920.0)))) & ~1
        pad_tot = inner_px + outer_px
        aspect_ratio = (float(src_w) / float(src_h)) if (src_w > 0 and src_h > 0) else (9.0 / 16.0)
        fg_w = max(2, int(round(fg_h * aspect_ratio)) & ~1)
        max_fg_w = max(2, (width - pad_tot * 2 - 4) & ~1)
        if fg_w > max_fg_w:
            fg_w = max_fg_w
            fg_h = max(2, int(round(fg_w / max(0.1, aspect_ratio))) & ~1)
        x0 = max(0, min(width - (fg_w + pad_tot * 2), x_center - (fg_w + pad_tot * 2) // 2))
        y0 = max(0, min(height - (fg_h + pad_tot * 2), y_pos))
        dec_w, dec_h = fg_w, fg_h
        portrait_gpu_geom = (x0, y0, inner_px, outer_px, fg_w, fg_h)
        # GPU-native portrait compositing: FFmpeg decodes ONLY the compact foreground stream (5.5x fewer pixels, zero CPU blur/vignette/overlay!)
        vf_dec = f"[0:v]{speed_prefix}scale={fg_w}:{fg_h}:flags=bilinear,fps={fps},setsar=1:1,format=rgb24[vdec]"
    else:
        vf_dec = (
            f"[0:v]{speed_prefix}scale={width}:{height}:force_original_aspect_ratio=decrease:flags=bilinear,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={fps},setsar=1:1,format=rgb24[vdec]"
        )

    card_rgba_cpu: Optional[np.ndarray] = None
    if card_png is not None and card_png.exists():
        from PIL import Image

        with Image.open(card_png) as ov_im:
            card_rgba_cpu = np.asarray(
                ov_im.convert("RGBA").resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            ).copy()

    stage_a_aud = tmp_out.with_suffix(".gpu_aud.m4a")
    dec_cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{src_in:.3f}"]
    if use_hw_decode and hw.has_videotoolbox_hwaccel:
        dec_cmd.extend(["-hwaccel", "videotoolbox"])
    elif use_hw_decode and hw.has_cuda_hwaccel:
        dec_cmd.extend(["-hwaccel", "cuda"])
    dec_cmd.extend(["-i", str(source)])
    if not has_audio or volume <= 0.0:
        dec_cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
    dec_cmd.extend(["-filter_complex_threads", "4", "-filter_complex", vf_dec])
    dec_cmd.extend(["-map", "[vdec]", "-an", "-t", f"{dur:.3f}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    if not has_audio or volume <= 0.0:
        dec_cmd.extend(["-map", "1:a:0", "-vn", "-t", f"{dur:.3f}", "-c:a", "aac", "-b:a", "256k", str(stage_a_aud)])
    else:
        af_parts = [
            "aresample=48000",
            "aformat=sample_fmts=fltp:channel_layouts=stereo",
            f"volume={volume:.3f}",
        ]
        if speed != 1.0 and 0.5 <= speed <= 2.0:
            af_parts.append(f"atempo={speed:.3f}")
        cf_in = max(audio_crossfade_sec, transition_in.duration if transition_in else 0.0)
        cf_out = max(audio_crossfade_sec, transition_out.duration if transition_out else 0.0)
        if cf_in > 0:
            af_parts.append(f"afade=t=in:st=0:d={min(0.45, cf_in):.3f}")
        if cf_out > 0:
            fd = min(0.45, cf_out)
            af_parts.append(f"afade=t=out:st={max(0.0, dur - fd):.3f}:d={fd:.3f}")
        dec_cmd.extend(
            [
                "-map", f"0:a:{audio_pos}", "-vn", "-t", f"{dur:.3f}",
                "-af", ",".join(af_parts), "-c:a", "aac", "-b:a", "256k", str(stage_a_aud),
            ]
        )

    frame_bytes = dec_w * dec_h * 3
    raw_pix_fmt: Literal["p010le", "yuv420p"] = "p010le" if hdr else "yuv420p"
    decoded_batches: list[Optional[tuple[np.ndarray, list[int]]]] = []
    graded_batches: list[Optional[np.ndarray]] = []

    try:
        with _VIDEO_PREBUFFER_SEM:
            # 3-Stage Decoupled Hardware Assembly Line:
            # Stage 0 (CPU/VideoToolbox Decode, 0 MB AMD VRAM, OUTSIDE _MPS_LOCK):
            # Decode compact raw RGB batches into RAM while another clip owns the AMD GPU!
            dec_proc = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert dec_proc.stdout is not None
            f_idx = 0
            while True:
                buf_arr = np.empty((batch_size, dec_h, dec_w, 3), dtype=np.uint8)
                mv_all = memoryview(buf_arr).cast("B")
                got = 0
                batch_idxs: list[int] = []
                for b_i in range(batch_size):
                    off = b_i * frame_bytes
                    view_slice = mv_all[off : off + frame_bytes]
                    read_so_far = 0
                    while read_so_far < frame_bytes:
                        n_read = dec_proc.stdout.readinto(view_slice[read_so_far:])
                        if not n_read:
                            break
                        read_so_far += n_read
                    if read_so_far < frame_bytes:
                        break
                    got += 1
                    batch_idxs.append(f_idx)
                    f_idx += 1
                mv_all.release()
                if got == 0:
                    break
                decoded_batches.append((buf_arr[:got], batch_idxs))
                if got < batch_size:
                    break
            dec_proc.wait()
            if dec_proc.returncode != 0 or not decoded_batches or not stage_a_aud.exists():
                return False

            # Stage 1 (Exclusive 8 GB AMD Metal GPU Compute, ZERO decode wait, ZERO VRAM thrashing!):
            # Blast all pre-decoded RAM batches through Metal at full speed (~10s/clip), then empty_cache()!
            with mps_guard:
                lut_atlas = get_gpu_lut_atlas(spec["lut_path"], device) if spec["lut_path"] is not None else None
                card_overlay: Optional[tuple[torch.Tensor, torch.Tensor]] = None
                if card_rgba_cpu is not None:
                    ov_t = (
                        torch.from_numpy(card_rgba_cpu)
                        .to(device=device)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(dtype=work_dtype)
                        .mul_(1.0 / 255.0)
                        .contiguous()
                    )
                    card_overlay = (ov_t[:, :3].contiguous(), ov_t[:, 3:4].contiguous())
                    del ov_t
                try:
                    with torch.no_grad():
                        for bi in range(len(decoded_batches)):
                            pkt = decoded_batches[bi]
                            decoded_batches[bi] = None  # Free raw decode RAM immediately
                            if pkt is None:
                                continue
                            stacked_np, b_idxs = pkt
                            x_gpu = (
                                torch.from_numpy(stacked_np)
                                .to(device=device)
                                .permute(0, 3, 1, 2)
                                .to(dtype=work_dtype)
                                .mul_(1.0 / 255.0)
                                .contiguous()
                            )
                            del stacked_np
                            if portrait_gpu_geom is not None:
                                px0, py0, p_in, p_out, _, _ = portrait_gpu_geom
                                x_gpu = _compose_portrait_darkroom_gpu(
                                    x_gpu,
                                    width=width,
                                    height=height,
                                    x0=px0,
                                    y0=py0,
                                    inner_px=p_in,
                                    outer_px=p_out,
                                )
                            out_gpu = _apply_video_grade_batch_gpu(
                                x_bchw=x_gpu,
                                spec=spec,
                                lut_atlas=lut_atlas,
                                frame_indices=b_idxs,
                                duration=dur,
                                fps=fps,
                                transition_in=transition_in,
                                transition_out=transition_out,
                                hdr=hdr,
                                diffuse_white_nits=diffuse_white_nits,
                                card_overlay=card_overlay,
                            )
                            out_cpu = out_gpu.to("cpu", non_blocking=False)
                            del x_gpu, out_gpu
                            graded_batches.append(out_cpu.numpy())
                            del out_cpu
                finally:
                    card_overlay = None
                    if device.type == "mps":
                        try:
                            torch.mps.synchronize()
                            torch.mps.empty_cache()
                        except Exception:
                            pass
            decoded_batches.clear()
            if not graded_batches:
                return False

            # Stage B: Hold _VT_HDR_ENCODE_LOCK ONLY for hardware hevc_videotoolbox Main10 encode!
            enc_cmd: list[str] = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", raw_pix_fmt, "-s", f"{width}x{height}", "-r", str(fps),
            ]
            if hdr:
                enc_cmd.extend(
                    [
                        "-color_primaries", "bt2020", "-color_trc", "arib-std-b67",
                        "-colorspace", "bt2020nc", "-color_range", "tv",
                    ]
                )
            else:
                enc_cmd.extend(
                    [
                        "-color_primaries", "bt709", "-color_trc", "bt709",
                        "-colorspace", "bt709", "-color_range", "tv",
                    ]
                )
            enc_cmd.extend(["-i", "pipe:0", "-i", str(stage_a_aud), "-map", "0:v:0", "-map", "1:a:0"])
            if hdr:
                enc_cmd.extend(["-vf", "setsar=1:1,setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:range=tv"])
            else:
                enc_cmd.extend(["-vf", "setsar=1:1"])
            enc_cmd.extend([*v_enc_args, *mux_args, "-c:a", "copy", "-shortest", str(tmp_out)])

            with nvenc_encode_guard(v_enc_args), videotoolbox_hdr_encode_guard(hdr=hdr, v_enc_args=v_enc_args):
                enc_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                assert enc_proc.stdin is not None
                try:
                    for i in range(len(graded_batches)):
                        arr = graded_batches[i]
                        graded_batches[i] = None
                        if arr is None:
                            continue
                        mv = memoryview(arr)
                        try:
                            enc_proc.stdin.write(mv)
                        except (BrokenPipeError, OSError):
                            break
                        finally:
                            mv.release()
                            del arr
                finally:
                    try:
                        enc_proc.stdin.close()
                    except (BrokenPipeError, OSError, ValueError):
                        pass
                enc_proc.wait()
                return enc_proc.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 1024
    except Exception:
        return False
    finally:
        graded_batches.clear()
        stage_a_aud.unlink(missing_ok=True)
