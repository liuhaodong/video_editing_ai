"""Batched GPU (NVIDIA CUDA / Apple Metal MPS) 4K photo compositor & async pipe streamer."""

from __future__ import annotations

import contextlib
import math
import queue
import threading
from typing import IO, Any, Iterator, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ..hardware import select_torch_device


# Serialize Metal command buffer submissions across threads to avoid PyTorch 2.2 MPS driver races.
_MPS_LOCK = threading.Lock()
_CUDA_MEM_FRACTION_SET: set[int] = set()


def _ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t * t * t
    p = 2.0 * t - 2.0
    return 0.5 * p * p * p + 1.0


def _ease_out_cubic(t: float) -> float:
    t_c = max(0.0, min(1.0, t))
    p = 1.0 - t_c
    return 1.0 - p * p * p


def _solve_perspective_coeffs(
    dst_pts: list[tuple[float, float]],
    src_pts: list[tuple[float, float]],
) -> tuple[float, ...]:
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(dst_pts, src_pts):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)
    coeffs = np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    return tuple(float(c) for c in coeffs)


def _pil_to_tensor(
    img: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Upload a PIL Image ('RGB', 'RGBA', or 'L') to (1, C, H, W) tensor on target device in [0, 255]."""
    src = img if (mode is None or img.mode == mode) else img.convert(mode)
    arr = np.array(src, dtype=np.uint8, copy=True)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype)


class MPSPhotoCompositor:
    """
    GPU-accelerated batched photo frame compositor targeting NVIDIA CUDA ('cuda') and
    Apple/AMD Metal ('mps') with identical CPU PyTorch fallback for parity tests and CI.
    """

    def __init__(self, ctx: dict[str, Any], device_override: Optional[str] = None) -> None:
        self.ctx = ctx
        dev_name = device_override or select_torch_device()
        self.device = torch.device(dev_name)
        self.work_dtype = torch.float16 if self.device.type in ("mps", "cuda") else torch.float32
        self.width = int(ctx["width"])
        self.height = int(ctx["height"])
        self.mode = str(ctx["mode"])
        self.scale_factor = float(ctx["scale_factor"])

        self._textures: dict[str, Any] = {}
        self._grid_1d_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._gauss_kernel_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._closed = False
        if self.device.type == "cuda":
            dev_idx = self.device.index if self.device.index is not None else 0
            with _MPS_LOCK:
                if dev_idx not in _CUDA_MEM_FRACTION_SET:
                    try:
                        # Allow generous GPU headroom on >=12GB cards while still bounding multi-worker allocations
                        props = torch.cuda.get_device_properties(dev_idx)
                        vram_gb = float(props.total_memory) / float(1024 ** 3)
                        frac = 0.82 if vram_gb >= 12.0 else 0.70
                        torch.cuda.set_per_process_memory_fraction(frac, dev_idx)
                    except Exception:
                        pass
                    _CUDA_MEM_FRACTION_SET.add(dev_idx)
        self._cuda_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None

        with self._accel_guard():
            self._upload_ctx_textures()
            if self._cuda_stream is not None:
                self._cuda_stream.synchronize()

    @contextlib.contextmanager
    def _accel_guard(self) -> Iterator[None]:
        """Serialize Metal command buffers on MPS; isolate per-compositor CUDA streams on NVIDIA."""
        if self.device.type == "mps":
            with _MPS_LOCK:
                yield
        elif self.device.type == "cuda" and self._cuda_stream is not None:
            with torch.cuda.stream(self._cuda_stream):
                yield
        else:
            yield

    def close(self) -> None:
        """Release GPU textures explicitly."""
        if self._closed:
            return
        self._closed = True
        with self._accel_guard():
            self._textures.clear()
            self._grid_1d_cache.clear()
            self._gauss_kernel_cache.clear()
            if self.device.type == "mps":
                try:
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                except Exception:
                    pass
            elif self.device.type == "cuda":
                try:
                    if self._cuda_stream is not None:
                        self._cuda_stream.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _up(self, img: Image.Image, mode: Optional[str] = None) -> torch.Tensor:
        return _pil_to_tensor(img, device=self.device, dtype=self.work_dtype, mode=mode)

    def _upload_ctx_textures(self) -> None:
        ctx = self.ctx
        t = self._textures
        mode = self.mode

        if "bg_plane" in ctx and ctx["bg_plane"] is not None:
            if mode in (
                "diptych_frame",
                "triptych_cinematic",
                "polaroid_scrapbook",
                "parallax_gallery",
                "glassmorphic_pedestal",
                "medium_format_slide",
                "floating_tilt_plate",
            ):
                t["bg_plane"] = self._up(ctx["bg_plane"], "RGB")

        if mode == "diptych_frame":
            t["fg_card_1"] = self._up(ctx["fg_card_1"], "RGBA")
            t["fg_card_2"] = self._up(ctx["fg_card_2"], "RGBA")
        elif mode == "triptych_cinematic":
            t["triptych_panels"] = [self._up(p, "RGBA") for p in ctx["triptych_panels"]]
        elif mode == "polaroid_scrapbook":
            t["polaroid_1"] = self._up(ctx["polaroid_1"], "RGBA")
            t["polaroid_2"] = self._up(ctx["polaroid_2"], "RGBA") if ctx.get("polaroid_2") is not None else None
        elif mode in ("cinematic_depth_3d", "dolly_zoom_vertigo"):
            t["mpi_layers"] = [self._up(layer, "RGBA") for layer in ctx["mpi_layers"]]
        elif mode == "aperture_rack_focus":
            t["rf_sharp"] = self._up(ctx["rf_sharp"], "RGB")
            t["rf_bokeh"] = self._up(ctx["rf_bokeh"], "RGB")
            t["rf_depth_q"] = (
                torch.from_numpy(np.ascontiguousarray(ctx["rf_depth_q"], dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device=self.device, dtype=torch.float32)
            )
        elif mode == "archival_color_bloom":
            t["bloom_color"] = self._up(ctx["bloom_color"], "RGB")
            t["bloom_bw"] = self._up(ctx["bloom_bw"], "RGB")
            t["bloom_depth_q"] = (
                torch.from_numpy(np.ascontiguousarray(ctx["bloom_depth_q"], dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device=self.device, dtype=torch.float32)
            )
        elif mode == "curator_loupe":
            t["loupe_dimmed"] = self._up(ctx["loupe_dimmed"], "RGB")
            t["loupe_sharp"] = self._up(ctx["loupe_sharp"], "RGB")
            t["loupe_mask"] = self._up(ctx["loupe_mask"], "L") / 255.0
            t["loupe_bezel"] = self._up(ctx["loupe_bezel"], "RGBA")
        elif mode == "editorial_magazine":
            t["mag_left_over"] = self._up(ctx["mag_left_over"], "RGB")
            t["mag_right_col"] = self._up(ctx["mag_right_col"], "RGB")
        elif mode == "cinema_anamorphic":
            t["anamorphic_base"] = self._up(ctx["anamorphic_base"], "RGB")
        elif mode == "parallax_gallery":
            t["fg_card_1"] = self._up(ctx["fg_card_1"], "RGBA")
        elif mode == "contact_sheet_zoom":
            t["contact_sheet"] = self._up(ctx["contact_sheet"], "RGB")
        elif mode == "split_diopter_focus":
            t["sd_left_over"] = self._up(ctx["sd_left_over"], "RGB")
            t["sd_right_over"] = self._up(ctx["sd_right_over"], "RGB")
            t["sd_seam"] = self._up(ctx["sd_seam"], "RGBA")
        elif mode == "glassmorphic_pedestal":
            ga = ctx["glass_assets"]
            t["glass_dark_tint"] = self._up(ga["dark_tint"], "RGB")
            t["glass_fg_rgba"] = self._up(ga["fg_rgba"], "RGBA")
            t["glass_sheen"] = self._up(ga["sheen_img"], "RGBA")
            t["glass_mask"] = self._up(ga["mask_l"], "L") / 255.0
        elif mode == "medium_format_slide":
            t["mf_slide_card"] = self._up(ctx["mf_slide_card"], "RGBA")
        elif mode == "floating_tilt_plate":
            t["tilt_framed_rgba"] = self._up(ctx["tilt_framed_rgba"], "RGBA")
            sh_alpha = ctx["tilt_shadow_rgba"].getchannel("A")
            sh_orig_w, sh_orig_h = sh_alpha.size
            sh_q_w = max(16, sh_orig_w // 4)
            sh_q_h = max(16, sh_orig_h // 4)
            t["tilt_shadow_orig_size"] = (sh_orig_w, sh_orig_h)
            t["tilt_shadow_alpha_q"] = (
                self._up(sh_alpha.resize((sh_q_w, sh_q_h), Image.Resampling.BILINEAR), "L") / 255.0
            )
        elif mode == "criterion_atelier_split":
            at = ctx["atelier"]
            t["at_left"] = self._up(at["left_over"], "RGB")
            t["at_rtop"] = self._up(at["right_top_over"], "RGB")
            t["at_rbot"] = self._up(at["right_bot_over"], "RGB")
        elif mode == "passepartout_shadowbox":
            sb = ctx["shadowbox"]
            t["sb_board_rgb"] = self._up(sb["board_rgb"], "RGB")
            t["sb_photo_over"] = self._up(sb["photo_over"], "RGB")
            t["sb_recess_mask"] = self._up(sb["recess_mask"], "L") / 255.0
            t["sb_recess_dark"] = torch.tensor(
                [6.0, 8.0, 12.0], device=self.device, dtype=self.work_dtype
            ).view(1, 3, 1, 1)
        elif mode == "kinfolk_linen_folio":
            kf = ctx["kinfolk_folio"]
            t["folio_page_rgb"] = self._up(kf["page_rgb"], "RGB")
            t["folio_photo_over"] = self._up(kf["photo_over"], "RGB")
        elif mode == "panoramic_frieze_scroll":
            t["frieze_canvas_rgb"] = self._up(ctx["frieze"]["canvas_rgb"], "RGB")
        elif mode == "atelier_layered_duo":
            ld = ctx["layered_duo"]
            t["duo_bg_rgb"] = self._up(ld["bg_rgb"], "RGB")
            t["duo_anchor_rgba"] = self._up(ld["anchor_rgba"], "RGBA")
            t["duo_fg_rgba"] = self._up(ld["fg_rgba"], "RGBA")
        elif mode == "leica_frame_finder":
            lf = ctx["leica_finder"]
            t["leica_sharp_over"] = self._up(lf["sharp_over"], "RGB")
            t["leica_surround_mask"] = self._up(lf["surround_mask"], "L") / 255.0
            t["leica_dim_rgb"] = torch.tensor(
                [8.0, 10.0, 14.0], device=self.device, dtype=self.work_dtype
            ).view(1, 3, 1, 1)
            t["leica_reticle_rgba"] = self._up(lf["reticle_rgba"], "RGBA")
        else:
            t["base_static_img"] = self._up(ctx["base_static_img"], "RGB")

        # Atmospheric finisher textures
        fx = ctx.get("fx_assets") or {"type": "none"}
        fx_type = fx.get("type", "none")
        t["fx_type"] = fx_type
        if fx_type == "golden_light_leak":
            t["fx_leak"] = self._up(fx["leak_img"], "RGB")
        elif fx_type == "museum_spotlight":
            t["fx_spot_dark"] = self._up(fx["dark_rgb"], "RGB")
            t["fx_spot_mask"] = self._up(fx["mask_l"], "L") / 255.0
        elif fx_type == "nostalgic_film_grain":
            t["fx_grains"] = [self._up(g, "RGB") for g in fx["grain_frames"]]
        elif fx_type == "anamorphic_streak":
            t["fx_streak"] = self._up(fx["streak_img"], "RGB")
        elif fx_type == "prism_chromatic_edge":
            t["fx_edge_mask"] = self._up(fx["edge_mask_l"], "L") / 255.0
        elif fx_type == "velvet_vignette_rembrandt":
            t["fx_key_rgb"] = self._up(fx["key_rgb"], "RGB")
            t["fx_key_mask"] = self._up(fx["key_mask"], "L") / 255.0
            t["fx_vig_rgb"] = self._up(fx["vig_rgb"], "RGB")
            t["fx_vig_mask"] = self._up(fx["vig_mask"], "L") / 255.0
        elif fx_type == "fine_art_rag_texture":
            t["fx_rag"] = self._up(fx["rag_img"], "RGB")
        elif fx_type == "caustic_water_shimmer":
            t["fx_shimmer"] = self._up(fx["shimmer_img"], "RGB")
        elif fx_type == "architectural_window_dapple":
            t["fx_dapple_sun_over"] = self._up(fx["sun_mask_over"], "L") / 255.0
            t["fx_dapple_shade_over"] = self._up(fx["shade_mask_over"], "L") / 255.0
            t["fx_dapple_warm"] = torch.tensor(
                [244.0, 232.0, 206.0], device=self.device, dtype=self.work_dtype
            ).view(1, 3, 1, 1)
            t["fx_dapple_shade"] = torch.tensor(
                [14.0, 16.0, 22.0], device=self.device, dtype=self.work_dtype
            ).view(1, 3, 1, 1)

        # Plane 3 Frosted documentary lower-third card
        if ctx.get("has_card") and ctx.get("card_fg") is not None:
            fg_rgba = self._up(ctx["card_fg"], "RGBA")
            fg_a = fg_rgba[:, 3:4] / 255.0
            t["card_fg_premul"] = fg_rgba[:, :3] * fg_a
            t["card_fg_inv_a"] = 1.0 - fg_a
            t["card_mask"] = self._up(ctx["card_mask"], "L") / 255.0
            t["card_tint"] = self._up(ctx["card_tint"], "RGB")

    def _get_1d_grid(self, out_h: int, out_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (out_h, out_w)
        cached = self._grid_1d_cache.get(key)
        if cached is None:
            u = (torch.arange(out_w, device=self.device, dtype=torch.float32) + 0.5) / float(out_w)
            v = (torch.arange(out_h, device=self.device, dtype=torch.float32) + 0.5) / float(out_h)
            cached = (u.view(1, 1, out_w), v.view(1, out_h, 1))
            self._grid_1d_cache[key] = cached
        return cached

    def _warp_extents_batch(
        self,
        src: torch.Tensor,
        out_size: tuple[int, int],
        boxes: list[tuple[float, float, float, float]],
    ) -> torch.Tensor:
        """
        Resample a single source tensor `(1, C, src_h, src_w)` for `B` normalized-pixel boxes
        `(x0, y0, x1, y1)` into `(B, C, out_h, out_w)` using clamped [-1, 1] grid_sample
        (`padding_mode='zeros'` because MPS does not support 'border').
        """
        out_w, out_h = out_size
        batch_n = len(boxes)
        if batch_n == 0:
            return torch.empty((0, int(src.shape[1]), out_h, out_w), device=self.device, dtype=self.work_dtype)

        # Micro-chunk large 4K multi-channel warps (e.g. 4-layer RGBA MPI) to bound peak float32 grid_sample VRAM
        ch = int(src.shape[1])
        pix_budget = 18_000_000 if ch <= 3 else 10_000_000
        if self.device.type == "cuda" and (batch_n * out_h * out_w > pix_budget) and batch_n > 1:
            sub_bs = 1 if (out_w >= 3840 and ch >= 4) else min(2, batch_n - 1)
            if sub_bs < batch_n:
                chunks: list[torch.Tensor] = []
                for start_i in range(0, batch_n, sub_bs):
                    chunks.append(self._warp_extents_batch(src, out_size, boxes[start_i : start_i + sub_bs]))
                return torch.cat(chunks, dim=0)

        _, _, src_h, src_w = src.shape
        u_1d, v_1d = self._get_1d_grid(out_h, out_w)

        x0 = torch.tensor([b[0] for b in boxes], device=self.device, dtype=torch.float32).view(batch_n, 1, 1)
        y0 = torch.tensor([b[1] for b in boxes], device=self.device, dtype=torch.float32).view(batch_n, 1, 1)
        x1 = torch.tensor([b[2] for b in boxes], device=self.device, dtype=torch.float32).view(batch_n, 1, 1)
        y1 = torch.tensor([b[3] for b in boxes], device=self.device, dtype=torch.float32).view(batch_n, 1, 1)

        ix = x0 + u_1d * (x1 - x0)
        iy = y0 + v_1d * (y1 - y0)
        gx = ((2.0 / max(1, src_w)) * ix - 1.0).clamp(-1.0, 1.0).expand(batch_n, out_h, out_w)
        gy = ((2.0 / max(1, src_h)) * iy - 1.0).clamp(-1.0, 1.0).expand(batch_n, out_h, out_w)
        grid = torch.stack((gx, gy), dim=-1)

        # PyTorch MPS/CPU F.grid_sample expects matching float32 inputs & grid for exact sub-pixel stability
        src_f32 = src.float().expand(batch_n, -1, -1, -1)
        warped = F.grid_sample(src_f32, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        out = warped.to(dtype=self.work_dtype)
        del grid, src_f32, warped
        return out

    def _crop_plane_batch(
        self,
        plane: torch.Tensor,
        out_size: tuple[int, int],
        zooms: list[float],
        cxs: list[float],
        cys: list[float],
    ) -> torch.Tensor:
        _, _, ph, pw = plane.shape
        boxes: list[tuple[float, float, float, float]] = []
        for z, cx, cy in zip(zooms, cxs, cys):
            cw = pw / max(1.0, z)
            ch = ph / max(1.0, z)
            bx0 = min(max(0.0, cx * pw - cw * 0.5), max(0.0, pw - cw))
            by0 = min(max(0.0, cy * ph - ch * 0.5), max(0.0, ph - ch))
            boxes.append((bx0, by0, bx0 + cw, by0 + ch))
        return self._warp_extents_batch(plane, out_size=out_size, boxes=boxes)

    def _paste_rgba_single(
        self,
        canvas_slice: torch.Tensor,
        src_rgba: torch.Tensor,
        px: int,
        py: int,
        dst_w: Optional[int] = None,
        dst_h: Optional[int] = None,
    ) -> None:
        """Alpha-composite an RGBA tensor `(1, 4, sh, sw)` onto a `(1, 3, H, W)` canvas in-place."""
        _, _, ch, cw = canvas_slice.shape
        _, _, sh, sw = src_rgba.shape
        tw = sw if dst_w is None else max(1, dst_w)
        th = sh if dst_h is None else max(1, dst_h)

        ox0 = max(0, px)
        oy0 = max(0, py)
        ox1 = min(cw, px + tw)
        oy1 = min(ch, py + th)
        if ox1 <= ox0 or oy1 <= oy0:
            return

        sx0 = (ox0 - px) * (sw / float(tw))
        sy0 = (oy0 - py) * (sh / float(th))
        sx1 = (ox1 - px) * (sw / float(tw))
        sy1 = (oy1 - py) * (sh / float(th))

        if dst_w is None and dst_h is None:
            sub = src_rgba[:, :, int(round(sy0)) : int(round(sy1)), int(round(sx0)) : int(round(sx1))]
            if sub.shape[2] != (oy1 - oy0) or sub.shape[3] != (ox1 - ox0):
                sub = self._warp_extents_batch(src_rgba, (ox1 - ox0, oy1 - oy0), [(sx0, sy0, sx1, sy1)])
        else:
            sub = self._warp_extents_batch(src_rgba, (ox1 - ox0, oy1 - oy0), [(sx0, sy0, sx1, sy1)])

        a = sub[:, 3:4] / 255.0
        roi = canvas_slice[:, :, oy0:oy1, ox0:ox1]
        canvas_slice[:, :, oy0:oy1, ox0:ox1] = roi * (1.0 - a) + sub[:, :3] * a

    def _gaussian_blur_2d(self, tensor_bchw: torch.Tensor, sigma: float) -> torch.Tensor:
        """Separable 2D Gaussian blur on GPU with replicate edge clamping."""
        channels = int(tensor_bchw.shape[1])
        radius = max(2, int(round(sigma * 2.5)))
        ksize = radius * 2 + 1
        key = (channels, ksize * 100 + int(round(sigma * 10)))
        cached = self._gauss_kernel_cache.get(key)
        if cached is None:
            coords = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.float32)
            g1d = torch.exp(-(coords * coords) / max(1e-6, 2.0 * sigma * sigma))
            g1d = (g1d / g1d.sum()).to(dtype=self.work_dtype)
            kx = g1d.view(1, 1, 1, ksize).expand(channels, 1, 1, ksize).contiguous()
            ky = g1d.view(1, 1, ksize, 1).expand(channels, 1, ksize, 1).contiguous()
            cached = (kx, ky)
            self._gauss_kernel_cache[key] = cached
        kx, ky = cached
        x = F.pad(tensor_bchw, (radius, radius, 0, 0), mode="replicate")
        x = F.conv2d(x, kx, groups=channels)
        x = F.pad(x, (0, 0, radius, radius), mode="replicate")
        x = F.conv2d(x, ky, groups=channels)
        return x

    def _warp_perspective_patch(
        self,
        src_rgba: torch.Tensor,
        out_w: int,
        out_h: int,
        coeffs: tuple[float, ...],
    ) -> torch.Tensor:
        """Projective homography resample `(1, 4, H, W)` -> `(1, 4, out_h, out_w)` with out-of-bounds zero mask."""
        _, _, src_h, src_w = src_rgba.shape
        ox = (torch.arange(out_w, device=self.device, dtype=torch.float32) + 0.5).view(1, out_w)
        oy = (torch.arange(out_h, device=self.device, dtype=torch.float32) + 0.5).view(out_h, 1)

        a, b, c, d, e, f, g, h = coeffs
        denom = g * ox + h * oy + 1.0
        denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
        u = (a * ox + b * oy + c) / denom
        v = (d * ox + e * oy + f) / denom

        in_bounds = ((u >= 0.0) & (u <= float(src_w)) & (v >= 0.0) & (v <= float(src_h))).to(torch.float32)
        gx = ((2.0 / max(1, src_w)) * u - 1.0).clamp(-1.0, 1.0)
        gy = ((2.0 / max(1, src_h)) * v - 1.0).clamp(-1.0, 1.0)
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)

        warped = F.grid_sample(
            src_rgba.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        warped = warped * in_bounds.unsqueeze(0).unsqueeze(0)
        return warped.to(dtype=self.work_dtype)

    def _warp_perspective_batch(
        self,
        src_t: torch.Tensor,
        out_w: int,
        out_h: int,
        coeffs_list: list[tuple[float, ...]],
    ) -> torch.Tensor:
        """Vectorized batched projective homography `(1, C, H, W)` -> `(B, C, out_h, out_w)`."""
        batch_n = len(coeffs_list)
        if batch_n == 0:
            return torch.empty((0, int(src_t.shape[1]), out_h, out_w), device=self.device, dtype=self.work_dtype)

        _, _, src_h, src_w = src_t.shape
        ox = (torch.arange(out_w, device=self.device, dtype=torch.float32) + 0.5).view(1, 1, out_w)
        oy = (torch.arange(out_h, device=self.device, dtype=torch.float32) + 0.5).view(1, out_h, 1)

        coeff_t = torch.tensor(coeffs_list, device=self.device, dtype=torch.float32)
        a = coeff_t[:, 0].view(batch_n, 1, 1)
        b = coeff_t[:, 1].view(batch_n, 1, 1)
        c = coeff_t[:, 2].view(batch_n, 1, 1)
        d = coeff_t[:, 3].view(batch_n, 1, 1)
        e = coeff_t[:, 4].view(batch_n, 1, 1)
        f = coeff_t[:, 5].view(batch_n, 1, 1)
        g = coeff_t[:, 6].view(batch_n, 1, 1)
        h = coeff_t[:, 7].view(batch_n, 1, 1)

        denom = g * ox + h * oy + 1.0
        denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
        u = (a * ox + b * oy + c) / denom
        v = (d * ox + e * oy + f) / denom

        in_bounds = ((u >= 0.0) & (u <= float(src_w)) & (v >= 0.0) & (v <= float(src_h))).to(dtype=self.work_dtype).unsqueeze(1)
        gx = ((2.0 / max(1, src_w)) * u - 1.0).clamp(-1.0, 1.0)
        gy = ((2.0 / max(1, src_h)) * v - 1.0).clamp(-1.0, 1.0)
        grid = torch.stack((gx, gy), dim=-1)

        warped = F.grid_sample(
            src_t.float().expand(batch_n, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped.to(dtype=self.work_dtype) * in_bounds

    def _synthesize_base_modes(
        self,
        t_norms: list[float],
    ) -> torch.Tensor:
        """Render the primary visual layer for a batch of frame times `t_norms` -> `(B, 3, H, W)`."""
        ctx = self.ctx
        tex = self._textures
        mode = self.mode
        w, h = self.width, self.height
        sf = self.scale_factor
        batch_n = len(t_norms)
        eased_list = [_ease_in_out_cubic(t) for t in t_norms]

        s_zoom, e_zoom = float(ctx["s_zoom"]), float(ctx["e_zoom"])
        sx, sy, ex, ey = float(ctx["sx"]), float(ctx["sy"]), float(ctx["ex"]), float(ctx["ey"])
        card_cy_ratio = float(ctx["card_cy_ratio"])

        if mode == "diptych_frame":
            zooms = [1.10 - 0.04 * e for e in eased_list]
            cxs = [0.52 - 0.04 * e for e in eased_list]
            cys = [0.50] * batch_n
            canvas = self._crop_plane_batch(tex["bg_plane"], (w, h), zooms, cxs, cys)
            c1, c2 = tex["fg_card_1"], tex["fg_card_2"]
            left_cx, right_cx = int(w * 0.28), int(w * 0.72)
            mid_cy = int(h * card_cy_ratio)
            for b, eased in enumerate(eased_list):
                drift_px = int(round((eased - 0.5) * 24.0 * sf))
                self._paste_rgba_single(canvas[b : b + 1], c1, left_cx - int(c1.shape[3]) // 2, mid_cy - int(c1.shape[2]) // 2 - drift_px)
                self._paste_rgba_single(canvas[b : b + 1], c2, right_cx - int(c2.shape[3]) // 2, mid_cy - int(c2.shape[2]) // 2 + drift_px)
            return canvas

        if mode == "triptych_cinematic":
            zooms = [1.10 - 0.04 * e for e in eased_list]
            cxs = [0.50 - 0.03 * (e - 0.5) for e in eased_list]
            canvas = self._crop_plane_batch(tex["bg_plane"], (w, h), zooms, cxs, [0.50] * batch_n)
            panels: list[torch.Tensor] = tex["triptych_panels"]
            centers_x = [int(w * 0.20), int(w * 0.50), int(w * 0.80)]
            mid_cy = int(h * card_cy_ratio)
            drift_dirs = [-1.0, 0.85, -1.0]
            for b, (t_norm, eased) in enumerate(zip(t_norms, eased_list)):
                for p_i, p_card in enumerate(panels):
                    stagger_start = p_i * 0.06
                    enter_p = _ease_out_cubic((t_norm - stagger_start) / 0.18) if t_norm > stagger_start else 0.0
                    if enter_p <= 0.02:
                        continue
                    slide_in_y = int(round((1.0 - enter_p) * 38.0 * sf))
                    drift_y = int(round((eased - 0.5) * 20.0 * sf * drift_dirs[p_i]))
                    px = centers_x[p_i] - int(p_card.shape[3]) // 2
                    py = max(8, mid_cy - int(p_card.shape[2]) // 2 + drift_y + slide_in_y)
                    self._paste_rgba_single(canvas[b : b + 1], p_card, px, py)
            return canvas

        if mode == "polaroid_scrapbook":
            zooms = [1.10 - 0.04 * e for e in eased_list]
            cxs = [0.50 + 0.03 * (e - 0.5) for e in eased_list]
            canvas = self._crop_plane_batch(tex["bg_plane"], (w, h), zooms, cxs, [0.50] * batch_n)
            p1, p2 = tex["polaroid_1"], tex["polaroid_2"]
            mid_cy = int(h * card_cy_ratio)
            for b, (t_norm, eased) in enumerate(zip(t_norms, eased_list)):
                if p2 is not None:
                    drift = int(round((eased - 0.5) * 18.0 * sf))
                    x1 = int(w * 0.33) - int(p1.shape[3]) // 2
                    y1 = max(8, mid_cy - int(p1.shape[2]) // 2 - drift)
                    self._paste_rgba_single(canvas[b : b + 1], p1, x1, y1)
                    enter_2 = _ease_out_cubic((t_norm - 0.08) / 0.24) if t_norm > 0.08 else 0.0
                    if enter_2 > 0.02:
                        slide_x = int(round((1.0 - enter_2) * 52.0 * sf))
                        x2 = int(w * 0.67) - int(p2.shape[3]) // 2 + slide_x
                        y2 = max(8, mid_cy - int(p2.shape[2]) // 2 + drift)
                        self._paste_rgba_single(canvas[b : b + 1], p2, x2, y2)
                else:
                    scale_p = 0.96 + 0.05 * eased
                    cw = max(1, int(round(int(p1.shape[3]) * scale_p)))
                    ch = max(1, int(round(int(p1.shape[2]) * scale_p)))
                    px = (w - cw) // 2 + int(round((eased - 0.5) * 22.0 * sf))
                    py = max(8, mid_cy - ch // 2)
                    self._paste_rgba_single(canvas[b : b + 1], p1, px, py, dst_w=cw, dst_h=ch)
            return canvas

        if mode == "cinematic_depth_3d":
            amp = float(ctx.get("parallax_amp", 1.0))
            mpi_layers: list[torch.Tensor] = tex["mpi_layers"]
            base_zooms = [1.032 + 0.028 * e for e in eased_list]
            arc_xs = [
                math.sin((e - 0.5) * math.pi) * 0.014 * amp + (ex - sx) * 0.12 * amp * (e - 0.5)
                for e in eased_list
            ]
            arc_ys = [
                -0.004 * amp * math.cos((e - 0.5) * math.pi) + (ey - sy) * 0.10 * amp * (e - 0.5)
                for e in eased_list
            ]
            layer_specs = ((-0.55, 0.994), (-0.18, 1.000), (0.18, 1.008), (0.55, 1.016))
            canvas = torch.empty((batch_n, 3, h, w), device=self.device, dtype=self.work_dtype)
            for li, (d_mult, z_mul) in enumerate(layer_specs):
                zooms_i = [bz * z_mul for bz in base_zooms]
                cxs_i = [0.5 + d_mult * ax for ax in arc_xs]
                cys_i = [0.5 + d_mult * ay for ay in arc_ys]
                cropped = self._crop_plane_batch(mpi_layers[li], (w, h), zooms_i, cxs_i, cys_i)
                if li == 0:
                    canvas = cropped[:, :3]
                else:
                    a = cropped[:, 3:4] / 255.0
                    canvas = canvas * (1.0 - a) + cropped[:, :3] * a
            return canvas

        if mode == "dolly_zoom_vertigo":
            amp = float(ctx.get("parallax_amp", 1.0))
            mpi_layers = tex["mpi_layers"]
            v_mixs = [e * (0.55 + 0.45 * amp) for e in eased_list]
            dx_vs = [(ex - sx) * 0.25 * e for e in eased_list]
            dy_vs = [(ey - sy) * 0.20 * e for e in eased_list]
            vertigo_specs = ((1.045, 0.030, -0.20), (1.035, 0.020, -0.06), (1.026, 0.011, 0.08), (1.018, -0.004, 0.20))
            canvas = torch.empty((batch_n, 3, h, w), device=self.device, dtype=self.work_dtype)
            for li, (z0, z_delta, d_shift) in enumerate(vertigo_specs):
                zooms_i = [z0 - z_delta * vm for vm in v_mixs]
                cxs_i = [0.5 + d_shift * dx for dx in dx_vs]
                cys_i = [0.5 + d_shift * dy for dy in dy_vs]
                cropped = self._crop_plane_batch(mpi_layers[li], (w, h), zooms_i, cxs_i, cys_i)
                if li == 0:
                    canvas = cropped[:, :3]
                else:
                    a = cropped[:, 3:4] / 255.0
                    canvas = canvas * (1.0 - a) + cropped[:, :3] * a
            return canvas

        if mode == "aperture_rack_focus":
            rf_sharp = tex["rf_sharp"]
            rf_bokeh = tex["rf_bokeh"]
            rf_depth_q = tex["rf_depth_q"]
            f_start = float(ctx["rf_focus_start"])
            f_end = float(ctx["rf_focus_end"])
            _, _, rh, rw = rf_sharp.shape
            _, _, qh, qw = rf_depth_q.shape

            zooms = [1.01 + 0.04 * e for e in eased_list]
            cxs = [0.5 + (ex - sx) * 0.25 * e for e in eased_list]
            cys = [0.5 + (ey - sy) * 0.25 * e for e in eased_list]
            sharp_c = self._crop_plane_batch(rf_sharp, (w, h), zooms, cxs, cys)
            bokeh_c = self._crop_plane_batch(rf_bokeh, (w, h), zooms, cxs, cys)

            masks_b: list[torch.Tensor] = []
            for t_norm, z, cx, cy in zip(t_norms, zooms, cxs, cys):
                focus_p = _ease_in_out_cubic(min(1.0, t_norm / 0.85))
                focus_d = f_start + (f_end - f_start) * focus_p
                dist = (rf_depth_q - focus_d).abs()
                lin = ((dist - 0.06) / 0.24).clamp(0.0, 1.0)
                blur_w_q = lin * lin * (3.0 - 2.0 * lin) * 0.85
                cw_r = rw / max(1.0, z)
                ch_r = rh / max(1.0, z)
                x0 = min(max(0.0, cx * rw - cw_r * 0.5), max(0.0, rw - cw_r))
                y0 = min(max(0.0, cy * rh - ch_r * 0.5), max(0.0, rh - ch_r))
                qx0 = min(max(0.0, (x0 / rw) * qw), qw - 1.0)
                qy0 = min(max(0.0, (y0 / rh) * qh), qh - 1.0)
                qx1 = min(float(qw), qx0 + (cw_r / rw) * qw)
                qy1 = min(float(qh), qy0 + (ch_r / rh) * qh)
                masks_b.append(self._warp_extents_batch(blur_w_q, (w, h), [(qx0, qy0, qx1, qy1)]))
            blur_mask = torch.cat(masks_b, dim=0)
            return sharp_c * (1.0 - blur_mask) + bokeh_c * blur_mask

        if mode == "archival_color_bloom":
            bloom_col = tex["bloom_color"]
            bloom_bw = tex["bloom_bw"]
            bloom_depth_q = tex["bloom_depth_q"]
            _, _, bh, bw_w = bloom_col.shape
            _, _, qh, qw = bloom_depth_q.shape

            zooms = [1.01 + 0.045 * e for e in eased_list]
            cxs = [0.5 + (ex - sx) * 0.30 * e for e in eased_list]
            cys = [0.5 + (ey - sy) * 0.30 * e for e in eased_list]
            bw_c = self._crop_plane_batch(bloom_bw, (w, h), zooms, cxs, cys)
            col_c = self._crop_plane_batch(bloom_col, (w, h), zooms, cxs, cys)

            masks_b = []
            for t_norm, z, cx, cy in zip(t_norms, zooms, cxs, cys):
                bloom_p = _ease_in_out_cubic(min(1.0, t_norm / 0.88))
                thresh = 1.06 - 1.26 * bloom_p
                soft_k = ((bloom_depth_q - thresh) / 0.32).clamp(0.0, 1.0)
                color_w_q = soft_k * soft_k * (3.0 - 2.0 * soft_k)
                cw_b = bw_w / max(1.0, z)
                ch_b = bh / max(1.0, z)
                x0 = min(max(0.0, cx * bw_w - cw_b * 0.5), max(0.0, bw_w - cw_b))
                y0 = min(max(0.0, cy * bh - ch_b * 0.5), max(0.0, bh - ch_b))
                qx0 = min(max(0.0, (x0 / bw_w) * qw), qw - 1.0)
                qy0 = min(max(0.0, (y0 / bh) * qh), qh - 1.0)
                qx1 = min(float(qw), qx0 + (cw_b / bw_w) * qw)
                qy1 = min(float(qh), qy0 + (ch_b / bh) * qh)
                masks_b.append(self._warp_extents_batch(color_w_q, (w, h), [(qx0, qy0, qx1, qy1)]))
            col_mask = torch.cat(masks_b, dim=0)
            return bw_c * (1.0 - col_mask) + col_c * col_mask

        if mode == "curator_loupe":
            canvas = tex["loupe_dimmed"].expand(batch_n, -1, -1, -1).clone()
            rad = int(ctx["loupe_radius"])
            diam = rad * 2
            mag = 1.65
            src_half = rad / mag
            loupe_mask = tex["loupe_mask"]
            bezel = tex["loupe_bezel"]
            bz_pad = (int(bezel.shape[3]) - diam) // 2

            boxes: list[tuple[float, float, float, float]] = []
            centers: list[tuple[int, int]] = []
            s_lx, s_ly = ctx["loupe_start_xy"]
            e_lx, e_ly = ctx["loupe_end_xy"]
            for eased in eased_list:
                lcx = max(0, min(w - 1, int(round(s_lx + (e_lx - s_lx) * eased))))
                lcy = max(
                    0,
                    min(h - 1, int(round(s_ly + (e_ly - s_ly) * eased - 4.0 * sf * math.sin(math.pi * eased)))),
                )
                sx0 = max(0.0, min(float(w) - 2.0 * src_half, lcx - src_half))
                sy0 = max(0.0, min(float(h) - 2.0 * src_half, lcy - src_half))
                boxes.append((sx0, sy0, sx0 + 2.0 * src_half, sy0 + 2.0 * src_half))
                centers.append((lcx, lcy))

            patches = self._warp_extents_batch(tex["loupe_sharp"], (diam, diam), boxes)
            for b, (lcx, lcy) in enumerate(centers):
                dst_x = lcx - rad
                dst_y = lcy - rad
                ox0, oy0 = max(0, dst_x), max(0, dst_y)
                ox1, oy1 = min(w, dst_x + diam), min(h, dst_y + diam)
                if ox1 > ox0 and oy1 > oy0:
                    sx_i0, sy_i0 = ox0 - dst_x, oy0 - dst_y
                    sx_i1, sy_i1 = sx_i0 + (ox1 - ox0), sy_i0 + (oy1 - oy0)
                    m = loupe_mask[:, :, sy_i0:sy_i1, sx_i0:sx_i1]
                    p = patches[b : b + 1, :, sy_i0:sy_i1, sx_i0:sx_i1]
                    roi = canvas[b : b + 1, :, oy0:oy1, ox0:ox1]
                    canvas[b : b + 1, :, oy0:oy1, ox0:ox1] = roi * (1.0 - m) + p * m
                self._paste_rgba_single(canvas[b : b + 1], bezel, dst_x - bz_pad, dst_y - bz_pad)
            return canvas

        if mode == "editorial_magazine":
            left_w = int(ctx["mag_left_w"])
            left_over = tex["mag_left_over"]
            zooms = [1.0 + 0.08 * e for e in eased_list]
            cxs = [sx + (ex - sx) * e for e in eased_list]
            cys = [sy + (ey - sy) * e for e in eased_list]
            left_crop = self._crop_plane_batch(left_over, (left_w, h), zooms, cxs, cys)
            right_rep = tex["mag_right_col"].expand(batch_n, -1, -1, -1)
            return torch.cat((left_crop, right_rep), dim=3)

        if mode == "cinema_anamorphic":
            zooms = [s_zoom + (e_zoom - s_zoom) * e for e in eased_list]
            cxs = [sx + (ex - sx) * e for e in eased_list]
            cys = [sy + (ey - sy) * e for e in eased_list]
            return self._crop_plane_batch(tex["anamorphic_base"], (w, h), zooms, cxs, cys)

        if mode == "parallax_gallery":
            zooms = [1.07 - 0.025 * e for e in eased_list]
            cxs = [0.5 - (ex - sx) * e * 0.22 for e in eased_list]
            cys = [0.5 - (ey - sy) * e * 0.22 for e in eased_list]
            canvas = self._crop_plane_batch(tex["bg_plane"], (w, h), zooms, cxs, cys)
            fg_card = tex["fg_card_1"]
            fh_orig, fw_orig = int(fg_card.shape[2]), int(fg_card.shape[3])
            for b, eased in enumerate(eased_list):
                fg_dx = (ex - sx) * eased
                fg_dy = (ey - sy) * eased
                fg_scale = 0.982 + 0.030 * eased
                cw = max(1, int(round(fw_orig * fg_scale)))
                ch = max(1, int(round(fh_orig * fg_scale)))
                px = (w - cw) // 2 + int(round(fg_dx * w * 0.06))
                py = max(10, int(h * card_cy_ratio) - ch // 2 + int(round(fg_dy * h * 0.045)))
                self._paste_rgba_single(canvas[b : b + 1], fg_card, px, py, dst_w=cw, dst_h=ch)
            return canvas

        if mode == "contact_sheet_zoom":
            hcx, hcy = ctx["contact_hero_xy"]
            zooms, cxs, cys = [], [], []
            for t_norm in t_norms:
                if t_norm <= 0.14:
                    zooms.append(1.00 + 0.04 * (t_norm / 0.14))
                    cxs.append(0.50)
                    cys.append(0.50)
                else:
                    push_p = _ease_in_out_cubic(min(1.0, (t_norm - 0.14) / 0.86))
                    zooms.append(1.04 + 1.48 * push_p)
                    cxs.append(0.50 + (hcx - 0.50) * push_p)
                    cys.append(0.50 + (hcy - 0.50) * push_p)
            return self._crop_plane_batch(tex["contact_sheet"], (w, h), zooms, cxs, cys)

        if mode == "split_diopter_focus":
            left_w = int(ctx["sd_left_w"])
            right_w = w - left_w
            l_zooms = [1.09 - 0.08 * e for e in eased_list]
            l_cxs = [0.50 - 0.03 * (e - 0.5) for e in eased_list]
            left_crop = self._crop_plane_batch(tex["sd_left_over"], (left_w, h), l_zooms, l_cxs, [0.50] * batch_n)

            r_zooms = [1.00 + 0.11 * e for e in eased_list]
            r_cxs = [0.50 + (ex - sx) * e * 0.35 for e in eased_list]
            r_cys = [0.50 + (ey - sy) * e * 0.35 for e in eased_list]
            right_crop = self._crop_plane_batch(tex["sd_right_over"], (right_w, h), r_zooms, r_cxs, r_cys)

            canvas = torch.cat((left_crop, right_crop), dim=3)
            seam = tex["sd_seam"]
            seam_x = max(0, left_w - int(seam.shape[3]) // 2)
            for b in range(batch_n):
                self._paste_rgba_single(canvas[b : b + 1], seam, seam_x, 0)
            return canvas

        if mode == "glassmorphic_pedestal":
            ga = ctx["glass_assets"]
            vw, vh = int(ga["vitrine_w"]), int(ga["vitrine_h"])
            zooms = [1.10 - 0.04 * e for e in eased_list]
            cxs = [0.50 - (ex - sx) * e * 0.45 for e in eased_list]
            canvas = self._crop_plane_batch(tex["bg_plane"], (w, h), zooms, cxs, [0.50] * batch_n)
            sw_p, sh_p = max(16, vw // 4), max(16, vh // 4)
            sheen = tex["glass_sheen"]
            sheen_w = int(sheen.shape[3])
            sheen_y = (vh - int(sheen.shape[2])) // 2
            dark_tint = tex["glass_dark_tint"]
            fg_rgba = tex["glass_fg_rgba"]
            fg_a = fg_rgba[:, 3:4] / 255.0
            fg_rgb = fg_rgba[:, :3]
            mask = tex["glass_mask"]

            for b, eased in enumerate(eased_list):
                fg_dx = (ex - sx) * eased
                px = max(4, min(w - vw - 4, (w - vw) // 2 + int(round(fg_dx * w * 0.16))))
                py = max(4, min(h - vh - 4, int(h * card_cy_ratio) - vh // 2))
                patch = canvas[b : b + 1, :, py : py + vh, px : px + vw]
                small = F.interpolate(patch.float(), size=(sh_p, sw_p), mode="bilinear", align_corners=False)
                blurred_small = self._gaussian_blur_2d(small.to(dtype=self.work_dtype), sigma=6.0)
                blurred = F.interpolate(
                    blurred_small.float(), size=(vh, vw), mode="bilinear", align_corners=False
                ).to(dtype=self.work_dtype)
                vitrine = blurred * 0.58 + dark_tint * 0.42
                vitrine = vitrine * (1.0 - fg_a) + fg_rgb * fg_a
                sheen_x = int(round(-sheen_w * 0.4 + (vw + sheen_w * 0.8) * eased))
                self._paste_rgba_single(vitrine, sheen, sheen_x, sheen_y)
                canvas[b : b + 1, :, py : py + vh, px : px + vw] = patch * (1.0 - mask) + vitrine * mask
            return canvas

        if mode == "medium_format_slide":
            zooms = [1.10 - 0.04 * e for e in eased_list]
            cxs = [0.50 + (ex - sx) * e * 0.35 for e in eased_list]
            canvas = self._crop_plane_batch(tex["bg_plane"], (w, h), zooms, cxs, [0.50] * batch_n)
            slide_card = tex["mf_slide_card"]
            sh_orig, sw_orig = int(slide_card.shape[2]), int(slide_card.shape[3])
            for b, eased in enumerate(eased_list):
                scale_s = 0.96 + 0.05 * eased
                cw = max(1, int(round(sw_orig * scale_s)))
                ch = max(1, int(round(sh_orig * scale_s)))
                px = (w - cw) // 2 + int(round((eased - 0.5) * 18.0 * sf))
                py = max(8, int(h * card_cy_ratio) - ch // 2)
                self._paste_rgba_single(canvas[b : b + 1], slide_card, px, py, dst_w=cw, dst_h=ch)
            return canvas

        if mode == "floating_tilt_plate":
            zooms = [1.10 - 0.04 * e for e in eased_list]
            cxs = [0.52 - 0.04 * e for e in eased_list]
            canvas = self._crop_plane_batch(tex["bg_plane"], (w, h), zooms, cxs, [0.50] * batch_n)
            framed_t = tex["tilt_framed_rgba"]
            shadow_q = tex["tilt_shadow_alpha_q"]
            fh, fw = int(framed_t.shape[2]), int(framed_t.shape[3])
            sw_sh, sh_sh = tex["tilt_shadow_orig_size"]
            sh_q_h, sh_q_w = int(shadow_q.shape[2]), int(shadow_q.shape[3])
            sh_ratio_w = sw_sh / max(1.0, float(fw))
            sh_ratio_h = sh_sh / max(1.0, float(fh))
            fg_src = [(0.0, 0.0), (float(fw), 0.0), (float(fw), float(fh)), (0.0, float(fh))]
            sh_src_q = [(0.0, 0.0), (float(sh_q_w), 0.0), (float(sh_q_w), float(sh_q_h)), (0.0, float(sh_q_h))]

            sh_geom: list[tuple[float, float, float, float, float, int, int, int, int]] = []
            fg_geom: list[tuple[float, float, float, float, float, int, int, int, int]] = []
            for eased in eased_list:
                yaw_norm = 0.095 * (1.0 - 0.85 * eased)
                s_out = 0.95 + 0.045 * eased
                half_w = fw * s_out * (1.0 - 0.14 * abs(yaw_norm)) * 0.5
                half_h_l = (fh * s_out * 0.5) * (1.0 + yaw_norm)
                half_h_r = (fh * s_out * 0.5) * (1.0 - yaw_norm)
                cx_out = w * (0.49 + 0.02 * eased)
                cy_out = h * card_cy_ratio - (1.0 - eased) * 6.0 * sf

                sh_off_x = 12.0 * sf + 14.0 * sf * (1.0 - eased)
                sh_off_y = 18.0 * sf
                sh_cx, sh_cy = cx_out + sh_off_x, cy_out + sh_off_y
                sh_hw = half_w * sh_ratio_w
                sh_hhl, sh_hhr = half_h_l * sh_ratio_h, half_h_r * sh_ratio_h
                sb0 = max(0, int(math.floor(sh_cx - sh_hw)) - 2)
                sb1 = min(w, int(math.ceil(sh_cx + sh_hw)) + 2)
                st0 = max(0, int(math.floor(sh_cy - max(sh_hhl, sh_hhr))) - 2)
                st1 = min(h, int(math.ceil(sh_cy + max(sh_hhl, sh_hhr))) + 2)
                sh_geom.append((sh_cx, sh_cy, sh_hw, sh_hhl, sh_hhr, sb0, sb1, st0, st1))

                bx0 = max(0, int(math.floor(cx_out - half_w)) - 2)
                bx1 = min(w, int(math.ceil(cx_out + half_w)) + 2)
                by0 = max(0, int(math.floor(cy_out - max(half_h_l, half_h_r))) - 2)
                by1 = min(h, int(math.ceil(cy_out + max(half_h_l, half_h_r))) + 2)
                fg_geom.append((cx_out, cy_out, half_w, half_h_l, half_h_r, bx0, bx1, by0, by1))

            valid_sh = [b_i for b_i, g in enumerate(sh_geom) if g[6] > g[5] + 4 and g[8] > g[7] + 4]
            if valid_sh:
                s_ux0 = min(sh_geom[b_i][5] for b_i in valid_sh)
                s_ux1 = max(sh_geom[b_i][6] for b_i in valid_sh)
                s_uy0 = min(sh_geom[b_i][7] for b_i in valid_sh)
                s_uy1 = max(sh_geom[b_i][8] for b_i in valid_sh)
                s_uw, s_uh = s_ux1 - s_ux0, s_uy1 - s_uy0
                if s_uw > 4 and s_uh > 4:
                    qs_w, qs_h = max(8, s_uw // 4), max(8, s_uh // 4)
                    sx_q, sy_q = qs_w / float(s_uw), qs_h / float(s_uh)
                    sh_coeffs_list: list[tuple[float, ...]] = []
                    for b_i in valid_sh:
                        sh_cx, sh_cy, sh_hw, sh_hhl, sh_hhr, _, _, _, _ = sh_geom[b_i]
                        sh_dst_q = [
                            ((sh_cx - sh_hw - s_ux0) * sx_q, (sh_cy - sh_hhl - s_uy0) * sy_q),
                            ((sh_cx + sh_hw - s_ux0) * sx_q, (sh_cy - sh_hhr - s_uy0) * sy_q),
                            ((sh_cx + sh_hw - s_ux0) * sx_q, (sh_cy + sh_hhr - s_uy0) * sy_q),
                            ((sh_cx - sh_hw - s_ux0) * sx_q, (sh_cy + sh_hhl - s_uy0) * sy_q),
                        ]
                        sh_coeffs_list.append(_solve_perspective_coeffs(sh_dst_q, sh_src_q))
                    sh_warped_q = self._warp_perspective_batch(shadow_q, qs_w, qs_h, sh_coeffs_list)
                    sh_alpha = F.interpolate(
                        sh_warped_q.float(), size=(s_uh, s_uw), mode="bilinear", align_corners=False
                    ).to(dtype=self.work_dtype)
                    if len(valid_sh) == batch_n:
                        canvas[:, :, s_uy0:s_uy1, s_ux0:s_ux1] = (
                            canvas[:, :, s_uy0:s_uy1, s_ux0:s_ux1] * (1.0 - sh_alpha)
                        )
                    else:
                        idx_t = torch.tensor(valid_sh, device=self.device, dtype=torch.long)
                        canvas[idx_t, :, s_uy0:s_uy1, s_ux0:s_ux1] = (
                            canvas[idx_t, :, s_uy0:s_uy1, s_ux0:s_ux1] * (1.0 - sh_alpha)
                        )

            valid_fg = [b_i for b_i, g in enumerate(fg_geom) if g[6] > g[5] + 4 and g[8] > g[7] + 4]
            if valid_fg:
                f_ux0 = min(fg_geom[b_i][5] for b_i in valid_fg)
                f_ux1 = max(fg_geom[b_i][6] for b_i in valid_fg)
                f_uy0 = min(fg_geom[b_i][7] for b_i in valid_fg)
                f_uy1 = max(fg_geom[b_i][8] for b_i in valid_fg)
                f_uw, f_uh = f_ux1 - f_ux0, f_uy1 - f_uy0
                if f_uw > 4 and f_uh > 4:
                    fg_coeffs_list: list[tuple[float, ...]] = []
                    for b_i in valid_fg:
                        cx_out, cy_out, half_w, half_h_l, half_h_r, _, _, _, _ = fg_geom[b_i]
                        fg_dst = [
                            (cx_out - half_w - f_ux0, cy_out - half_h_l - f_uy0),
                            (cx_out + half_w - f_ux0, cy_out - half_h_r - f_uy0),
                            (cx_out + half_w - f_ux0, cy_out + half_h_r - f_uy0),
                            (cx_out - half_w - f_ux0, cy_out + half_h_l - f_uy0),
                        ]
                        fg_coeffs_list.append(_solve_perspective_coeffs(fg_dst, fg_src))
                    fg_warped = self._warp_perspective_batch(framed_t, f_uw, f_uh, fg_coeffs_list)
                    a = fg_warped[:, 3:4] / 255.0
                    if len(valid_fg) == batch_n:
                        roi = canvas[:, :, f_uy0:f_uy1, f_ux0:f_ux1]
                        canvas[:, :, f_uy0:f_uy1, f_ux0:f_ux1] = roi * (1.0 - a) + fg_warped[:, :3] * a
                    else:
                        idx_t = torch.tensor(valid_fg, device=self.device, dtype=torch.long)
                        roi = canvas[idx_t, :, f_uy0:f_uy1, f_ux0:f_ux1]
                        canvas[idx_t, :, f_uy0:f_uy1, f_ux0:f_ux1] = roi * (1.0 - a) + fg_warped[:, :3] * a
            return canvas

        if mode == "criterion_atelier_split":
            at = ctx["atelier"]
            lw, rw = int(at["left_w"]), int(at["right_w"])
            th, bh = int(at["top_h"]), int(at["bot_h"])
            seam_px = int(at["seam_px"])
            canvas = torch.empty((batch_n, 3, h, w), device=self.device, dtype=self.work_dtype)
            canvas[:, 0] = 14.0
            canvas[:, 1] = 16.0
            canvas[:, 2] = 20.0
            l_crop = self._crop_plane_batch(
                tex["at_left"], (lw, h), [1.0 + 0.075 * e for e in eased_list], [0.49 + 0.02 * e for e in eased_list], [0.50] * batch_n
            )
            rt_crop = self._crop_plane_batch(
                tex["at_rtop"], (rw, th), [1.075 - 0.075 * e for e in eased_list], [0.52 - 0.03 * e for e in eased_list], [0.50] * batch_n
            )
            rb_crop = self._crop_plane_batch(
                tex["at_rbot"], (rw, bh), [1.0 + 0.09 * e for e in eased_list], [0.50] * batch_n, [0.50] * batch_n
            )
            canvas[:, :, :, :lw] = l_crop
            rx0 = lw + seam_px
            canvas[:, :, :th, rx0 : rx0 + rw] = rt_crop
            ry0 = th + seam_px
            canvas[:, :, ry0 : ry0 + bh, rx0 : rx0 + rw] = rb_crop
            hair = torch.tensor([122.0, 118.0, 110.0], device=self.device, dtype=self.work_dtype).view(1, 3, 1)
            vx = lw + seam_px // 2
            hy = th + seam_px // 2
            if 0 <= vx < w:
                canvas[:, :, :, vx] = hair
            if 0 <= hy < h:
                canvas[:, :, hy, lw + seam_px : w] = hair
            return canvas

        if mode == "passepartout_shadowbox":
            sb = ctx["shadowbox"]
            wx, wy, ww, wh = int(sb["wx"]), int(sb["wy"]), int(sb["ww"]), int(sb["wh"])
            z_ins = [1.00 + 0.055 * e for e in eased_list]
            cx_ins = [sx + (ex - sx) * e * 0.65 for e in eased_list]
            cy_ins = [sy + (ey - sy) * e * 0.65 for e in eased_list]
            win_crops = self._crop_plane_batch(tex["sb_photo_over"], (ww, wh), z_ins, cx_ins, cy_ins)
            rm = tex["sb_recess_mask"]
            win_crops = win_crops * (1.0 - rm) + tex["sb_recess_dark"] * rm

            z_outs = [1.00 + 0.020 * e for e in eased_list]
            cx_outs = [0.50 + (ex - sx) * 0.12 * (e - 0.5) for e in eased_list]
            cy_outs = [0.50 + (ey - sy) * 0.10 * (e - 0.5) for e in eased_list]
            board_work = tex["sb_board_rgb"].clone()
            out_frames: list[torch.Tensor] = []
            for b in range(batch_n):
                board_work[0, :, wy : wy + wh, wx : wx + ww] = win_crops[b]
                out_frames.append(
                    self._crop_plane_batch(board_work, (w, h), [z_outs[b]], [cx_outs[b]], [cy_outs[b]])
                )
            return torch.cat(out_frames, dim=0)

        if mode == "kinfolk_linen_folio":
            kf = ctx["kinfolk_folio"]
            px, py, pw, ph = int(kf["px"]), int(kf["py"]), int(kf["pw"]), int(kf["ph"])
            z_ins = [1.00 + 0.060 * e for e in eased_list]
            cx_ins = [sx + (ex - sx) * e * 0.70 for e in eased_list]
            cy_ins = [sy + (ey - sy) * e * 0.70 for e in eased_list]
            win_crops = self._crop_plane_batch(tex["folio_photo_over"], (pw, ph), z_ins, cx_ins, cy_ins)
            canvas = tex["folio_page_rgb"].expand(batch_n, -1, -1, -1).clone()
            canvas[:, :, py : py + ph, px : px + pw] = win_crops
            return canvas

        if mode == "panoramic_frieze_scroll":
            fr = ctx["frieze"]
            kb_cfg = ctx.get("kb")
            preset_val = getattr(kb_cfg, "preset", "pan_left_to_right") if kb_cfg is not None else "pan_left_to_right"
            fw_f, fh_f = float(fr["fw"]), float(fr["fh"])
            outer_pad_f = float(fr.get("outer_pad", 0.0))
            boxes = []
            for eased in eased_list:
                pan_p = (1.0 - eased) if preset_val == "pan_right_to_left" else eased
                pan_z = 1.005 + 0.018 * math.sin(math.pi * eased)
                view_h = fh_f / pan_z
                view_w = view_h * (float(w) / max(1.0, float(h)))
                if view_w > fw_f:
                    view_w = fw_f
                    view_h = view_w * (float(h) / max(1.0, float(w)))
                max_x0 = max(0.0, fw_f - view_w)
                x_margin = min(max_x0 * 0.5, outer_pad_f * 0.25)
                bx0 = x_margin + (max_x0 - 2.0 * x_margin) * pan_p
                by0 = max(0.0, (fh_f - view_h) * 0.5)
                boxes.append((bx0, by0, bx0 + view_w, by0 + view_h))
            return self._warp_extents_batch(tex["frieze_canvas_rgb"], (w, h), boxes)

        if mode == "atelier_layered_duo":
            zooms = [1.03 - 0.02 * e for e in eased_list]
            canvas = self._crop_plane_batch(tex["duo_bg_rgb"], (w, h), zooms, [0.50] * batch_n, [0.50] * batch_n)
            anc = tex["duo_anchor_rgba"]
            fg_c = tex["duo_fg_rgba"]
            anc_h, anc_w = int(anc.shape[2]), int(anc.shape[3])
            fg_h, fg_w = int(fg_c.shape[2]), int(fg_c.shape[3])
            has_card_f = bool(ctx.get("has_card"))
            for b, eased in enumerate(eased_list):
                s1 = 0.990 + 0.020 * eased
                w1 = max(1, int(round(anc_w * s1)))
                h1 = max(1, int(round(anc_h * s1)))
                x1 = int(w * 0.39) - w1 // 2 + int(round((0.5 - eased) * 18.0 * sf))
                y1 = max(8, int(h * (0.42 if has_card_f else 0.46)) - h1 // 2)
                self._paste_rgba_single(canvas[b : b + 1], anc, x1, y1, dst_w=w1, dst_h=h1)

                s2 = 0.975 + 0.038 * eased
                w2 = max(1, int(round(fg_w * s2)))
                h2 = max(1, int(round(fg_h * s2)))
                x2 = int(w * 0.68) - w2 // 2 + int(round((eased - 0.5) * 32.0 * sf))
                y2 = max(
                    12,
                    int(h * (0.53 if has_card_f else 0.58)) - h2 // 2 + int(round((0.5 - eased) * 12.0 * sf)),
                )
                self._paste_rgba_single(canvas[b : b + 1], fg_c, x2, y2, dst_w=w2, dst_h=h2)
            return canvas

        if mode == "leica_frame_finder":
            zooms = [1.00 + 0.050 * e for e in eased_list]
            cxs = [sx + (ex - sx) * e * 0.55 for e in eased_list]
            cys = [sy + (ey - sy) * e * 0.55 for e in eased_list]
            canvas = self._crop_plane_batch(tex["leica_sharp_over"], (w, h), zooms, cxs, cys)
            sm = tex["leica_surround_mask"]
            canvas = canvas * (1.0 - sm) + tex["leica_dim_rgb"] * sm
            reticle = tex["leica_reticle_rgba"]
            for b, t_norm in enumerate(t_norms):
                lock_p = _ease_out_cubic(min(1.0, t_norm / 0.32))
                if lock_p >= 0.999:
                    self._paste_rgba_single(canvas[b : b + 1], reticle, 0, 0)
                else:
                    ret_scale = 1.0 + 0.028 * (1.0 - lock_p)
                    rw = max(1, int(round(w * ret_scale)))
                    rh = max(1, int(round(h * ret_scale)))
                    px = (w - rw) // 2
                    py = (h - rh) // 2
                    self._paste_rgba_single(canvas[b : b + 1], reticle, px, py, dst_w=rw, dst_h=rh)
            return canvas

        # gallery_frame & full_bleed (and any default)
        zooms = [s_zoom + (e_zoom - s_zoom) * e for e in eased_list]
        cxs = [sx + (ex - sx) * e for e in eased_list]
        cys = [sy + (ey - sy) * e for e in eased_list]
        return self._crop_plane_batch(tex["base_static_img"], (w, h), zooms, cxs, cys)

    def _apply_overlay_fx_batch(
        self,
        canvas: torch.Tensor,
        t_norms: list[float],
        frame_indices: list[int],
    ) -> torch.Tensor:
        """Apply composable atmospheric finisher matching `_apply_overlay_fx`."""
        tex = self._textures
        fx_type = tex.get("fx_type", "none")
        if fx_type == "none":
            return canvas

        batch_n = len(t_norms)
        w, h = self.width, self.height

        if fx_type == "golden_light_leak":
            weights = torch.tensor(
                [0.12 + 0.10 * math.sin(math.pi * t) for t in t_norms], device=self.device, dtype=self.work_dtype
            ).view(batch_n, 1, 1, 1)
            return canvas * (1.0 - weights) + tex["fx_leak"] * weights

        if fx_type == "museum_spotlight":
            m = tex["fx_spot_mask"]
            return canvas * (1.0 - m) + tex["fx_spot_dark"] * m

        if fx_type == "champagne_bokeh":
            sw, sh = max(64, w // 4), max(36, h // 4)
            orbs = (self.ctx.get("fx_assets") or {}).get("orbs") or []
            if not orbs:
                return canvas
            yy, xx = torch.meshgrid(
                torch.arange(sh, device=self.device, dtype=torch.float32),
                torch.arange(sw, device=self.device, dtype=torch.float32),
                indexing="ij",
            )
            fill_rgb = torch.tensor([255.0, 222.0, 138.0], device=self.device, dtype=self.work_dtype).view(1, 3, 1, 1)
            rim_rgb = torch.tensor([255.0, 240.0, 190.0], device=self.device, dtype=self.work_dtype).view(1, 3, 1, 1)
            overlay_small = torch.zeros((batch_n, 4, sh, sw), device=self.device, dtype=self.work_dtype)
            for b, t_norm in enumerate(t_norms):
                acc_rgb = torch.zeros((1, 3, sh, sw), device=self.device, dtype=self.work_dtype)
                acc_a = torch.zeros((1, 1, sh, sw), device=self.device, dtype=self.work_dtype)
                for ox, oy, rad, alpha, phase in orbs:
                    cx = ((ox + 0.03 * math.sin(phase + t_norm * 4.0)) % 1.0) * sw
                    cy = ((oy - 0.10 * t_norm) % 1.0) * sh
                    r_small = max(3.0, float(rad // 4))
                    pulse_fill = min(1.0, (alpha * (0.75 + 0.25 * math.sin(phase + t_norm * 6.0))) / 255.0)
                    pulse_rim = min(1.0, (alpha * (0.75 + 0.25 * math.sin(phase + t_norm * 6.0)) + 25.0) / 255.0)
                    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + 1e-4)
                    in_fill = (dist <= r_small).to(self.work_dtype)
                    in_rim = ((dist > r_small - 1.2) & (dist <= r_small + 0.8)).to(self.work_dtype)
                    orb_a = (in_fill * pulse_fill).maximum(in_rim * pulse_rim).unsqueeze(0).unsqueeze(0)
                    orb_col = torch.where(in_rim.unsqueeze(0).unsqueeze(0) > 0, rim_rgb, fill_rgb)
                    acc_rgb = orb_col * orb_a + acc_rgb * (1.0 - orb_a)
                    acc_a = orb_a + acc_a * (1.0 - orb_a)
                overlay_small[b : b + 1, :3] = acc_rgb
                overlay_small[b : b + 1, 3:4] = acc_a
            overlay_blur = self._gaussian_blur_2d(overlay_small, sigma=3.0)
            overlay_full = F.interpolate(
                overlay_blur.float(), size=(h, w), mode="bilinear", align_corners=False
            ).to(dtype=self.work_dtype)
            a = overlay_full[:, 3:4].clamp(0.0, 1.0)
            return canvas * (1.0 - a) + overlay_full[:, :3] * a

        if fx_type == "nostalgic_film_grain":
            grains: list[torch.Tensor] = tex["fx_grains"]
            g_batch = torch.cat([grains[idx % len(grains)] for idx in frame_indices], dim=0)
            return canvas * 0.935 + g_batch * 0.065

        if fx_type == "anamorphic_streak":
            sw, sh = max(48, w // 8), max(27, h // 8)
            small = F.interpolate(canvas.float(), size=(sh, sw), mode="bilinear", align_corners=False)
            luma = 0.299 * small[:, 0:1] + 0.587 * small[:, 1:2] + 0.114 * small[:, 2:3]
            hi = ((luma - 205.0) / 50.0).clamp(0.0, 1.0)
            streak_x = hi.clone()
            for shift in (2, 5, 9, 14, 20):
                if shift < sw:
                    left_s = F.pad(hi[:, :, :, shift:], (0, shift, 0, 0), mode="constant", value=0.0)
                    right_s = F.pad(hi[:, :, :, :-shift], (shift, 0, 0, 0), mode="constant", value=0.0)
                    streak_x = streak_x + 0.45 * (left_s + right_s)
            streak_x = (streak_x * 0.28).clamp(0.0, 1.0)
            tint = torch.tensor([135.0, 195.0, 255.0], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
            dyn_small = streak_x * tint
            dyn_full = F.interpolate(dyn_small, size=(h, w), mode="bilinear", align_corners=False).to(dtype=self.work_dtype)
            # PIL ImageChops.screen in [0, 255]: a + b - a*b/255
            s_ref = tex["fx_streak"]
            combined_flare = dyn_full + s_ref - (dyn_full * s_ref) / 255.0
            screened = canvas + combined_flare - (canvas * combined_flare) / 255.0
            weights = torch.tensor(
                [0.42 + 0.20 * math.sin(math.pi * t) for t in t_norms], device=self.device, dtype=self.work_dtype
            ).view(batch_n, 1, 1, 1)
            return canvas * (1.0 - weights) + screened * weights

        if fx_type == "prism_chromatic_edge":
            dxs = [max(2.0, min(w * 0.012, (4.0 + 1.5 * math.sin(math.pi * t)) * self.scale_factor)) for t in t_norms]
            dys = [dx * (h / max(1.0, float(w))) for dx in dxs]
            r_boxes = [(dx, dy, float(w) - dx, float(h) - dy) for dx, dy in zip(dxs, dys)]
            b_boxes = [(0.0, 0.0, float(w) - 0.5 * dx, float(h) - 0.5 * dy) for dx, dy in zip(dxs, dys)]
            r_list = [
                self._warp_extents_batch(canvas[b : b + 1, 0:1], (w, h), [r_boxes[b]])
                for b in range(batch_n)
            ]
            b_list = [
                self._warp_extents_batch(canvas[b : b + 1, 2:3], (w, h), [b_boxes[b]])
                for b in range(batch_n)
            ]
            ca_img = torch.cat((torch.cat(r_list, dim=0), canvas[:, 1:2], torch.cat(b_list, dim=0)), dim=1)
            m = tex["fx_edge_mask"]
            return ca_img * m + canvas * (1.0 - m)

        if fx_type == "velvet_vignette_rembrandt":
            km, krgb = tex["fx_key_mask"], tex["fx_key_rgb"]
            vm, vrgb = tex["fx_vig_mask"], tex["fx_vig_rgb"]
            canvas = canvas * (1.0 - km) + krgb * km
            canvas = canvas * (1.0 - vm) + vrgb * vm
            return canvas

        if fx_type == "fine_art_rag_texture":
            return canvas * 0.945 + tex["fx_rag"] * 0.055

        if fx_type == "halation_bloom_35mm":
            sw, sh = max(48, w // 6), max(27, h // 6)
            small = F.interpolate(canvas.float(), size=(sh, sw), mode="bilinear", align_corners=False)
            luma = 0.299 * small[:, 0:1] + 0.587 * small[:, 1:2] + 0.114 * small[:, 2:3]
            hi = ((luma - 206.0) / 49.0).clamp(0.0, 1.0).to(dtype=self.work_dtype)
            halo_s = self._gaussian_blur_2d(hi, sigma=4.0).clamp(0.0, 1.0)
            tint = torch.tensor([255.0, 124.0, 68.0], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
            halo_full = F.interpolate(halo_s.float() * tint, size=(h, w), mode="bilinear", align_corners=False).to(
                dtype=self.work_dtype
            )
            screened = canvas + halo_full - (canvas * halo_full) / 255.0
            weights = torch.tensor(
                [0.24 + 0.08 * math.sin(math.pi * t) for t in t_norms], device=self.device, dtype=self.work_dtype
            ).view(batch_n, 1, 1, 1)
            return canvas * (1.0 - weights) + screened * weights

        if fx_type == "caustic_water_shimmer":
            shimmer = tex["fx_shimmer"]
            screened = canvas + shimmer - (canvas * shimmer) / 255.0
            weights = torch.tensor(
                [0.09 + 0.045 * math.sin(math.pi * t) for t in t_norms], device=self.device, dtype=self.work_dtype
            ).view(batch_n, 1, 1, 1)
            return canvas * (1.0 - weights) + screened * weights

        if fx_type == "architectural_window_dapple":
            fx = self.ctx.get("fx_assets") or {}
            dw = float(fx.get("dw", w))
            boxes = []
            for t_norm in t_norms:
                x0 = (dw - w) * (1.0 - _ease_in_out_cubic(t_norm))
                boxes.append((x0, 0.0, x0 + float(w), float(h)))
            sun_m = self._warp_extents_batch(tex["fx_dapple_sun_over"], (w, h), boxes)
            shade_m = self._warp_extents_batch(tex["fx_dapple_shade_over"], (w, h), boxes)
            canvas = canvas * (1.0 - sun_m) + tex["fx_dapple_warm"] * sun_m
            canvas = canvas * (1.0 - shade_m) + tex["fx_dapple_shade"] * shade_m
            return canvas

        return canvas

    def _composite_frosted_card_batch(
        self,
        canvas: torch.Tensor,
        frame_indices: list[int],
        duration: float,
        fps: int,
    ) -> torch.Tensor:
        ctx = self.ctx
        if not ctx.get("has_card") or "card_fg_premul" not in self._textures:
            return canvas

        tex = self._textures
        fg_premul = tex["card_fg_premul"]
        fg_inv_a = tex["card_fg_inv_a"]
        card_mask = tex["card_mask"]
        dark_tint = tex["card_tint"]
        _, _, ph, pw = fg_premul.shape
        card_x = int(ctx["card_x"])
        card_y = int(ctx["card_y"])
        small_w = max(16, pw // 4)
        small_h = max(16, ph // 4)

        active: list[tuple[int, int, int, float]] = []
        for b, f_idx in enumerate(frame_indices):
            t_sec = f_idx / max(1.0, float(fps))
            rem_sec = max(0.0, duration - t_sec)
            entry_prog = _ease_out_cubic((t_sec - 0.15) / 0.50) if t_sec > 0.15 else 0.0
            exit_prog = min(1.0, rem_sec / 0.40)
            alpha_mult = min(entry_prog, exit_prog)
            if alpha_mult <= 0.04:
                continue

            slide_y = int(round((1.0 - entry_prog) * 16.0 * self.scale_factor))
            dst_x = max(0, min(self.width - pw, card_x))
            dst_y = max(0, min(self.height - ph, card_y + slide_y))
            active.append((b, dst_x, dst_y, alpha_mult))

        idx = 0
        n_active = len(active)
        while idx < n_active:
            b_start, dst_x, dst_y, _ = active[idx]
            end_idx = idx + 1
            while (
                end_idx < n_active
                and active[end_idx][0] == active[end_idx - 1][0] + 1
                and active[end_idx][1] == dst_x
                and active[end_idx][2] == dst_y
            ):
                end_idx += 1
            b_end = active[end_idx - 1][0] + 1
            alphas = [active[j][3] for j in range(idx, end_idx)]
            k = b_end - b_start

            patches = canvas[b_start:b_end, :, dst_y : dst_y + ph, dst_x : dst_x + pw]
            small = F.interpolate(patches.float(), size=(small_h, small_w), mode="bilinear", align_corners=False)
            blurred_small = self._gaussian_blur_2d(small.to(dtype=self.work_dtype), sigma=5.0)
            blurred = F.interpolate(
                blurred_small.float(), size=(ph, pw), mode="bilinear", align_corners=False
            ).to(dtype=self.work_dtype)

            w_tint = max(0.0, min(1.0, float(ctx.get("card_tint_weight", 0.58))))
            frosted = blurred * (1.0 - w_tint) + dark_tint * w_tint
            frosted = frosted * fg_inv_a + fg_premul

            if all(am >= 0.98 for am in alphas):
                eff_mask: torch.Tensor = card_mask
            else:
                am_t = torch.tensor(
                    [1.0 if am >= 0.98 else am for am in alphas],
                    device=self.device,
                    dtype=self.work_dtype,
                ).view(k, 1, 1, 1)
                eff_mask = card_mask * am_t
            canvas[b_start:b_end, :, dst_y : dst_y + ph, dst_x : dst_x + pw] = (
                patches * (1.0 - eff_mask) + frosted * eff_mask
            )
            idx = end_idx

        return canvas

    def _render_batch_impl(
        self,
        frame_indices: list[int],
        total_frames: int,
        duration: float,
        fps: int,
    ) -> torch.Tensor:
        denom = max(1, total_frames - 1)
        t_norms = [f_idx / denom for f_idx in frame_indices]

        with self._accel_guard():
            with torch.no_grad():
                canvas = self._synthesize_base_modes(t_norms)
                canvas = self._apply_overlay_fx_batch(canvas, t_norms=t_norms, frame_indices=frame_indices)

                if self.mode == "cinema_anamorphic":
                    bar_h = int(self.ctx.get("anamorphic_bar_h") or 0)
                    if bar_h > 0:
                        bar_col = torch.tensor([4.0, 5.0, 8.0], device=self.device, dtype=self.work_dtype).view(1, 3, 1, 1)
                        canvas[:, :, :bar_h, :] = bar_col
                        canvas[:, :, self.height - bar_h :, :] = bar_col

                canvas = self._composite_frosted_card_batch(
                    canvas, frame_indices=frame_indices, duration=duration, fps=fps
                )
                gpu_u8 = canvas.clamp(0.0, 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
                del canvas
                out_u8 = gpu_u8.to("cpu", non_blocking=False)
                if self._cuda_stream is not None:
                    self._cuda_stream.synchronize()
                return out_u8

    def render_batch_rgb(
        self,
        frame_indices: list[int],
        total_frames: int,
        duration: float,
        fps: int,
    ) -> torch.Tensor:
        """Synthesize a batch of frames and return contiguous CPU uint8 tensor `(B, H, W, 3)`."""
        if not frame_indices:
            return torch.empty((0, self.height, self.width, 3), dtype=torch.uint8)

        try:
            return self._render_batch_impl(
                frame_indices=frame_indices,
                total_frames=total_frames,
                duration=duration,
                fps=fps,
            )
        except torch.OutOfMemoryError:
            if self.device.type != "cuda" or len(frame_indices) <= 1:
                raise
            torch.cuda.empty_cache()
            singles = [
                self._render_batch_impl(
                    frame_indices=[f_idx],
                    total_frames=total_frames,
                    duration=duration,
                    fps=fps,
                )
                for f_idx in frame_indices
            ]
            return torch.cat(singles, dim=0)


def synthesize_batch_rgb_mps(
    compositor: MPSPhotoCompositor,
    frame_indices: list[int],
    total_frames: int,
    duration: float,
    fps: int,
) -> torch.Tensor:
    """Render specific frame indices using `MPSPhotoCompositor` -> CPU `uint8` `(B, H, W, 3)`."""
    return compositor.render_batch_rgb(
        frame_indices=frame_indices,
        total_frames=total_frames,
        duration=duration,
        fps=fps,
    )


def stream_photo_frames_to_pipe(
    ctx: dict[str, Any],
    total_frames: int,
    duration: float,
    fps: int,
    pipe_stdin: IO[bytes],
    batch_size: int = 8,
    device_override: Optional[str] = None,
) -> None:
    """
    Synthesize photo clip frames in batches on GPU (CUDA/MPS) and stream contiguous RGB24 buffers into
    `pipe_stdin` through a bounded async queue so GPU compute overlaps with FFmpeg encoding.
    """
    comp = MPSPhotoCompositor(ctx=ctx, device_override=device_override)
    q: queue.Queue[Optional[torch.Tensor]] = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    writer_error: list[BaseException] = []

    def _writer_loop() -> None:
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                try:
                    arr_np = item.numpy()
                    if not arr_np.flags["C_CONTIGUOUS"]:
                        arr_np = np.ascontiguousarray(arr_np)
                    mv = memoryview(arr_np)
                    pipe_stdin.write(mv)
                except (BrokenPipeError, OSError):
                    stop_event.set()
                    return
                finally:
                    del item
        except BaseException as exc:
            writer_error.append(exc)
            stop_event.set()

    writer = threading.Thread(target=_writer_loop, daemon=True)
    writer.start()

    try:
        eff_batch = (
            min(max(1, batch_size), 4)
            if (comp.device.type != "cuda" and int(ctx.get("width", 0)) >= 3840)
            else max(1, batch_size)
        )
        for start_idx in range(0, total_frames, eff_batch):
            if stop_event.is_set():
                break
            batch_indices = list(range(start_idx, min(total_frames, start_idx + eff_batch)))
            batch_u8 = comp.render_batch_rgb(
                frame_indices=batch_indices,
                total_frames=total_frames,
                duration=duration,
                fps=fps,
            )
            while not stop_event.is_set():
                try:
                    q.put(batch_u8, timeout=0.2)
                    break
                except queue.Full:
                    continue
            del batch_u8
    finally:
        for _ in range(10):
            if not writer.is_alive():
                break
            if stop_event.is_set():
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        writer.join(timeout=5.0)
        comp.close()

    if writer_error:
        raise writer_error[0]
