"""Multi-track FFmpeg video/audio compositor with Track V2 B-Roll cutaways, pluggable VFX, parallel NVENC rendering, and smart audio ducking."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable, Iterator, Optional

from ..analyzer.live_photo_filter import is_live_photo_video
from ..analyzer.media_probe import probe_video_file
from ..hardware import (
    caffeinate_guard,
    get_hardware_profile,
    get_io_worker_count,
    get_render_worker_count,
    has_ffmpeg_filter,
    nvenc_encode_guard,
)
from ..models.schema import EditTimeline, EffectTrackItem, TitleOverlayItem, VisualTrackItem
from .color_grading import CINEMA_FILM_STOCK_SPECS, apply_color_grade_to_pil_image, write_film_stock_cube_lut
from .ken_burns import (
    build_hdr_color_filters,
    build_segment_overlay_filtergraph,
    build_uniform_mux_args,
    get_video_encoder_args,
    has_hevc_nvenc,
    render_frosted_video_overlay_png,
    render_photo_clip,
)
from .kinetic_titles import compute_kinetic_roi, render_glass_mask_png, render_kinetic_title_mov
from .subtitles import export_ass_subtitles, export_srt_subtitles
from .titles import compute_glass_pill_rect, render_title_png


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape a file path for FFmpeg libavfilter single-quoted filename='...' arguments."""
    posix = path.resolve().as_posix()
    return posix.replace(":", r"\:").replace("'", r"\'")


@contextmanager
def _unlink_on_exit(paths: list[Path]) -> Iterator[None]:
    """Ensure temporary per-render kinetic title / glass mask files are cleaned up on exit."""
    try:
        yield
    finally:
        for tmp_art in paths:
            try:
                if tmp_art.exists():
                    tmp_art.unlink()
            except Exception:
                pass


def _build_color_grade_filter(fx: EffectTrackItem, hdr: Optional[bool] = None) -> str:
    """Build Stage 42 DI color_grade filter chain (3-way LGG balance, eq, 3D LUT + safe YUV matrix, HLG knee)."""
    params = fx.parameters or {}
    look_name = str(params.get("chapter_look") or params.get("look") or "")
    exp_ev = float(params.get("exposure_ev", 0.0))
    if params.get("bypass") or (look_name == "neutral_reference" and abs(exp_ev) < 1e-4):
        return "null"

    parts: list[str] = []
    if params.get("hlg_rolloff"):
        # Smooth 10-bit limited-range YUV shoulder knee (compresses blown Pixel HLG highlights to ~203 nits)
        parts.append(
            "format=p010le,"
            "lutyuv=y='clip(64+(val-64)*0.78-if(gt(val,520),0.26*(val-520),0),64,760)':"
            "u='512+(val-512)*0.94':v='512+(val-512)*0.94'"
        )

    lut_cube_raw = str(params.get("lut_cube_path") or "").strip()
    lut_int = float(params.get("lut_intensity", 1.0))
    use_lut = bool(
        lut_cube_raw
        and Path(lut_cube_raw).exists()
        and lut_int > 0.02
        and look_name != "neutral_reference"
        and has_ffmpeg_filter("lut3d")
    )

    wb_r = float(params.get("wb_red_shift", params.get("rs", 0.0)))
    wb_g = float(params.get("wb_green_shift", params.get("gs", 0.0)))
    wb_b = float(params.get("wb_blue_shift", params.get("bs", 0.0)))
    lift = params.get("lift_rgb") or (0.0, 0.0, 0.0)
    mid = params.get("gamma_rgb") or (0.0, 0.0, 0.0)
    gain = params.get("gain_rgb") or (0.0, 0.0, 0.0)
    # Attenuate creative LGG when the 3D .cube LUT already embodies the stock split-tone curve
    lgg_scale = 0.20 if use_lut else 0.70

    def _c(v: float) -> float:
        return max(-0.35, min(0.35, float(v)))

    rs = _c(wb_r * 0.45 + float(lift[0]) * lgg_scale)
    gs = _c(wb_g * 0.45 + float(lift[1]) * lgg_scale)
    bs = _c(wb_b * 0.45 + float(lift[2]) * lgg_scale)
    rm = _c(wb_r * 0.72 + float(mid[0]) * lgg_scale)
    gm = _c(wb_g * 0.72 + float(mid[1]) * lgg_scale)
    bm = _c(wb_b * 0.72 + float(mid[2]) * lgg_scale)
    rh = _c(wb_r * 0.55 + float(gain[0]) * lgg_scale)
    gh = _c(wb_g * 0.55 + float(gain[1]) * lgg_scale)
    bh = _c(wb_b * 0.55 + float(gain[2]) * lgg_scale)

    parts.append(
        f"colorbalance=rs={rs:.3f}:gs={gs:.3f}:bs={bs:.3f}:"
        f"rm={rm:.3f}:gm={gm:.3f}:bm={bm:.3f}:"
        f"rh={rh:.3f}:gh={gh:.3f}:bh={bh:.3f}"
    )

    black_lift = float(params.get("black_lift", 0.0))
    b_adj = max(-0.15, min(0.15, exp_ev * 0.14 + black_lift * 0.45))
    contrast = max(0.75, min(1.40, float(params.get("contrast", 1.04))))
    sat = max(0.50, min(1.60, float(params.get("saturation", 1.04))))
    base_gamma = max(0.70, min(1.40, float(params.get("gamma", 1.00))))
    eff_gamma = max(0.70, min(1.45, base_gamma * (2.0 ** max(-0.4, min(0.4, exp_ev * 0.32)))))
    parts.append(f"eq=brightness={b_adj:.3f}:contrast={contrast:.2f}:saturation={sat:.2f}:gamma={eff_gamma:.2f}")

    if use_lut:
        cube_p = Path(lut_cube_raw)
        if lut_int < 0.98 and look_name in CINEMA_FILM_STOCK_SPECS:
            cube_p = write_film_stock_cube_lut(
                look_name,
                cube_p.with_name(f"{look_name}_i{int(round(lut_int * 100))}.cube"),
                intensity=lut_int,
            )
        parts.append(f"lut3d=file='{_escape_ffmpeg_filter_path(cube_p)}':interp=tetrahedral")
        # lut3d emits planar RGB (gbrp); restore explicit YUV matrix before downstream zscale/encoder
        src_trc = str(params.get("src_trc") or "").lower()
        src_space = str(params.get("src_space") or "").lower()
        hdr_mode = bool(params.get("hdr", True)) if hdr is None else bool(hdr)
        if src_trc == "arib-std-b67" or src_space.startswith("bt2020"):
            parts.append("scale=out_color_matrix=bt2020:out_range=tv,format=p010le")
        else:
            parts.append(f"scale=out_color_matrix=bt709:out_range=tv,format={'yuv444p10le' if hdr_mode else 'yuv420p'}")

    return ",".join(parts)


def _build_video_stabilize_filter(fx: EffectTrackItem) -> str:
    """Stabilize handheld clips and crop out mirrored deshake border margins."""
    rx = max(4, min(32, int(fx.parameters.get("rx", 16)))) & ~1
    ry = max(4, min(32, int(fx.parameters.get("ry", 16)))) & ~1
    return (
        f"deshake=rx={rx}:ry={ry}:edge=mirror,"
        f"crop=iw-{rx * 2}:ih-{ry * 2}:{rx}:{ry},"
        f"scale=iw+{rx * 2}:ih+{ry * 2}:flags=lanczos"
    )


# Extensible registry mapping Stage-40/32/42 EffectTrackItem.effect_type -> FFmpeg filter builder
EFFECT_FILTER_BUILDERS: dict[str, Callable[[EffectTrackItem], str]] = {
    "warm_glow": lambda fx: (
        f"eq=saturation={float(fx.parameters.get('saturation', 1.08)):.2f}:"
        f"brightness={float(fx.parameters.get('brightness', 0.02)):.3f}"
    ),
    "vignette_focus": lambda fx: f"vignette=angle={float(fx.parameters.get('angle_rad', 0.42)):.2f}",
    "contrast_pop": lambda fx: (
        f"eq=contrast={float(fx.parameters.get('contrast', 1.06)):.2f}:"
        f"saturation={float(fx.parameters.get('saturation', 1.05)):.2f}"
    ),
    "color_grade": _build_color_grade_filter,
    "video_stabilize": _build_video_stabilize_filter,
}


def register_effect_filter(effect_type: str, builder: Callable[[EffectTrackItem], str]) -> None:
    """Register a custom FFmpeg filter builder for a Stage-40 EffectTrackItem."""
    EFFECT_FILTER_BUILDERS[effect_type] = builder


def _video_encoder_args(hdr: bool, force_software: bool = False) -> list[str]:
    """Return cross-platform hardware-accelerated encoder arguments (VideoToolbox / NVENC / AMF / QSV / libx265)."""
    return get_video_encoder_args(hdr=hdr, force_software=force_software)


def _probe_video_extradata_sig(mp4_path: Path) -> tuple[str, str, int, int, bool]:
    """Return (codec_name, extradata, width, height, has_rot_sidedata) of stream v:0."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,extradata:stream_side_data",
                "-show_data",
                "-of",
                "json",
                str(mp4_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return ("", "", 0, 0, False)
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return ("", "", 0, 0, False)
        s = streams[0]
        sd_list = s.get("side_data_list") or []
        has_rot = any(
            "display" in str(sd.get("side_data_type") or "").lower() or "rotation" in sd
            for sd in sd_list
            if isinstance(sd, dict)
        )
        return (
            str(s.get("codec_name") or ""),
            str(s.get("extradata") or ""),
            int(s.get("width") or 0),
            int(s.get("height") or 0),
            has_rot,
        )
    except Exception:
        return ("", "", 0, 0, False)


def _concat_segments_safe(
    segment_files: list[Path],
    output_mp4: Path,
    w_dir: Path,
    fps: int,
    hdr: bool,
    list_filename: str = "segments_concat.txt",
) -> Path:
    """Concatenate rendered Track V1 segments safely, falling back to chunked filter concat when hvcC extradata differs."""
    if not segment_files:
        raise ValueError("segment_files must not be empty")

    concat_list_path = w_dir / list_filename
    concat_lines = [f"file '{p.as_posix()}'" for p in segment_files]
    concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    if len(segment_files) == 1:
        shutil.copy2(segment_files[0], output_mp4)
        return output_mp4

    if len(segment_files) > 2:
        with ThreadPoolExecutor(max_workers=min(8, get_io_worker_count(8), len(segment_files))) as ex:
            sigs = list(ex.map(_probe_video_extradata_sig, segment_files))
    else:
        sigs = [_probe_video_extradata_sig(p) for p in segment_files]
    uniform_extradata = (
        len(set(sigs)) == 1
        and bool(sigs[0][0])
        and bool(sigs[0][1])
        and sigs[0][2] > 0
        and sigs[0][3] > 0
        and not sigs[0][4]
    )

    if uniform_extradata:
        concat_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-display_rotation",
            "0",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            "-metadata:s:v:0",
            "rotate=0",
            *(["-tag:v", "hvc1"] if hdr else []),
            str(output_mp4),
        ]
        proc = subprocess.run(concat_cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return output_mp4

    def _run_filter_concat(inputs: list[Path], target_mp4: Path) -> None:
        if len(inputs) == 1:
            shutil.copy2(inputs[0], target_mp4)
            return

        target_w, target_h = 3840, 2160
        first_sig = _probe_video_extradata_sig(inputs[0])
        if first_sig[2] > 0 and first_sig[3] > 0:
            target_w, target_h = first_sig[2], first_sig[3]
        sd_part = "sidedata=mode=delete," if has_ffmpeg_filter("sidedata") else ""
        v_norm = (
            f"{sd_part}setsar=1:1,"
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
        )
        v_post = (
            f"fps={fps},format=p010le,setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:range=tv"
            if hdr
            else f"fps={fps},format=yuv420p"
        )
        norm_filters = [f"[{i}:v:0]{v_norm}[vn{i}]" for i in range(len(inputs))]
        in_pads = "".join(f"[vn{i}][{i}:a:0]" for i in range(len(inputs)))
        fc = (
            f"{';'.join(norm_filters)};"
            f"{in_pads}concat=n={len(inputs)}:v=1:a=1[vcat][acat];"
            f"[vcat]{v_post}[vout];"
            f"[acat]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[aout]"
        )
        in_args: list[str] = []
        for p in inputs:
            in_args.extend(["-display_rotation:v:0", "0", "-i", str(p)])

        def _cmd(force_sw: bool) -> list[str]:
            return [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                *in_args,
                "-filter_complex",
                fc,
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                *_video_encoder_args(hdr, force_software=force_sw),
                "-video_track_timescale",
                "30000",
                "-metadata:s:v:0",
                "rotate=0",
                "-c:a",
                "aac",
                "-b:a",
                "320k" if hdr else "192k",
                str(target_mp4),
            ]

        with nvenc_encode_guard(_video_encoder_args(hdr, force_software=False)):
            proc = subprocess.run(_cmd(force_sw=False), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            proc = subprocess.run(_cmd(force_sw=True), capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg filter concat failed: {proc.stderr}")

    chunk_size = 16
    if len(segment_files) <= chunk_size:
        _run_filter_concat(segment_files, output_mp4)
        return output_mp4

    chunk_specs = [
        (segment_files[c_idx : c_idx + chunk_size], w_dir / f"concat_chunk_{c_idx // chunk_size:02d}.mp4")
        for c_idx in range(0, len(segment_files), chunk_size)
    ]
    chunk_files = [cm for _, cm in chunk_specs]
    try:
        chunk_workers = max(1, min(2, len(chunk_specs)))
        if chunk_workers > 1:
            with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
                futs = [pool.submit(_run_filter_concat, ci, cm) for ci, cm in chunk_specs]
                for fut in futs:
                    fut.result()
        else:
            for ci, cm in chunk_specs:
                _run_filter_concat(ci, cm)
        _concat_segments_safe(
            segment_files=chunk_files,
            output_mp4=output_mp4,
            w_dir=w_dir,
            fps=fps,
            hdr=hdr,
            list_filename="chunks_concat.txt",
        )
    finally:
        for cf in chunk_files:
            if cf.exists():
                cf.unlink()

    return output_mp4


def _strip_pix_fmt_args(enc_args: list[str]) -> list[str]:
    """Strip explicit -pix_fmt pairs for zero-copy hardware surface encoding."""
    cleaned: list[str] = []
    skip_next = False
    for tok in enc_args:
        if skip_next:
            skip_next = False
            continue
        if tok == "-pix_fmt":
            skip_next = True
            continue
        cleaned.append(tok)
    return cleaned


def _render_video_segment(
    item: VisualTrackItem,
    output_mp4: Path,
    width: int,
    height: int,
    fps: int,
    hdr: bool = True,
    attached_effects: Optional[list[EffectTrackItem]] = None,
    segment_broll: Optional[list[tuple[Path, float, float]]] = None,
    segment_titles: Optional[list[tuple[Path, float, float, float, float] | dict[str, Any]]] = None,
) -> Path:
    """Trim, scale/composite (with ambient blurred backdrop for portrait 9:16 videos), overlay documentary card, and encode."""
    info = probe_video_file(Path(item.source))
    has_audio = bool(info.get("has_audio", False))
    audio_pos = int(info.get("audio_stream_pos") or 0)
    src_trc = str(info.get("color_transfer") or info.get("color_trc") or "bt709")
    src_prim = str(info.get("color_primaries") or "bt709")
    src_space = str(info.get("color_space") or "bt709")
    src_pfmt = str(info.get("pix_fmt") or "yuv420p").lower()
    is_log_clip = "log" in src_trc.lower()
    diffuse_nits = 203.0
    for fx in attached_effects or []:
        if fx.effect_type == "color_grade" and fx.parameters:
            if fx.parameters.get("src_trc"):
                src_trc = str(fx.parameters["src_trc"])
            if fx.parameters.get("src_primaries"):
                src_prim = str(fx.parameters["src_primaries"])
            if fx.parameters.get("src_space"):
                src_space = str(fx.parameters["src_space"])
            if "is_log" in fx.parameters:
                is_log_clip = bool(fx.parameters["is_log"])
            if fx.parameters.get("diffuse_white_nits"):
                diffuse_nits = float(fx.parameters["diffuse_white_nits"])
    src_w = int(info.get("width", 1920) or 1920)
    src_h = int(info.get("height", 1080) or 1080)
    is_portrait = src_w < src_h
    dur = item.duration
    src_in = item.source_in

    kb = item.ken_burns
    has_doc_card = bool(kb and kb.caption_title and kb.caption_title.strip())
    has_paired_photo = bool(
        is_portrait
        and item.paired_visual_source
        and Path(item.paired_visual_source).exists()
    )
    has_bottom_seg_title = any(
        isinstance(t_ent, dict)
        and isinstance(t_ent.get("item"), TitleOverlayItem)
        and getattr(t_ent["item"], "style", "")
        in ("lower_third", "magazine_chapter_plate", "curved_gps_ribbon", "act_banner", "opening_card")
        for t_ent in (segment_titles or [])
    )
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_mp4.with_name(f".{output_mp4.stem}.tmp_{os.getpid()}_{threading.get_ident()}.mp4")
    card_png: Optional[Path] = None
    if has_doc_card or has_paired_photo:
        card_png = output_mp4.with_name(f".{output_mp4.stem}.card_{os.getpid()}_{threading.get_ident()}.png")
        if has_doc_card:
            render_frosted_video_overlay_png(
                output_png=card_png,
                width=width,
                height=height,
                caption_title=kb.caption_title or "",
                caption_subtitle=kb.caption_subtitle or "",
                camera_badge=kb.camera_badge or "",
            )
        else:
            from PIL import Image

            Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(card_png, "PNG")

        if has_paired_photo and item.paired_visual_source:
            try:
                from PIL import Image, ImageOps
                from .ken_burns import _build_double_matte_card

                base_ov = Image.open(card_png).convert("RGBA")
                p_img = ImageOps.exif_transpose(Image.open(item.paired_visual_source)).convert("RGB")
                color_fx = next((fx for fx in (attached_effects or []) if fx.effect_type == "color_grade"), None)
                if color_fx is not None:
                    p_img = apply_color_grade_to_pil_image(p_img, color_fx)
                p_max_h = int(height * (0.68 if has_bottom_seg_title else (0.74 if has_doc_card else 0.80)))
                p_card = _build_double_matte_card(
                    p_img,
                    max_w=int(width * 0.40),
                    max_h=p_max_h,
                    scale_factor=width / 1920.0,
                )
                rx = int(width * 0.72) - p_card.width // 2
                ry_anchor = 0.40 if has_bottom_seg_title else (0.43 if has_doc_card else 0.48)
                ry = int(height * ry_anchor) - p_card.height // 2
                base_ov.paste(p_card, (rx, max(8, ry)), p_card)
                base_ov.save(card_png, "PNG")
            except Exception:
                has_paired_photo = False

    hw = get_hardware_profile()
    same_aspect = abs((src_w / max(1, src_h)) - (width / max(1, height))) < 0.02
    src_matches_range = (
        (hdr and src_trc.lower() == "arib-std-b67" and "10" in src_pfmt and not is_log_clip)
        or (not hdr and "10" not in src_pfmt and not is_log_clip)
    )
    can_zero_copy_vt = (
        hw.has_videotoolbox_hwaccel
        and hw.has_scale_vt
        and not is_portrait
        and same_aspect
        and card_png is None
        and not segment_broll
        and not segment_titles
        and not attached_effects
        and item.speed == 1.0
        and item.transition_in.type in ("none", "cut")
        and item.transition_out.type in ("none", "cut")
        and src_matches_range
    )

    post_filters: list[str] = [f"fps={fps}", "setsar=1:1"]
    if has_ffmpeg_filter("sidedata"):
        post_filters.append("sidedata=mode=delete")
    for fx in attached_effects or []:
        if fx.effect_type == "color_grade":
            fx_merged = fx.model_copy(
                update={
                    "parameters": {
                        **dict(fx.parameters or {}),
                        "src_trc": (fx.parameters or {}).get("src_trc") or src_trc,
                        "src_space": (fx.parameters or {}).get("src_space") or src_space,
                        "hdr": hdr,
                    }
                }
            )
            post_filters.append(_build_color_grade_filter(fx_merged, hdr=hdr))
        else:
            builder = EFFECT_FILTER_BUILDERS.get(fx.effect_type)
            if builder is not None:
                post_filters.append(builder(fx))

    if item.transition_in.type not in ("none", "cut") and item.transition_in.duration > 0:
        if item.transition_in.type == "dip_to_white":
            fd = min(dur * 0.35, item.transition_in.duration)
            post_filters.append(f"fade=t=in:st=0:d={fd:.3f}:c=#F6F2E8")
        elif item.transition_in.type in ("dissolve", "crossfade"):
            fd = min(dur * 0.25, item.transition_in.duration, 0.26)
            post_filters.append(f"fade=t=in:st=0:d={fd:.3f}:c=#0B0E14")
        else:
            fd = min(dur * 0.40, item.transition_in.duration)
            post_filters.append(f"fade=t=in:st=0:d={fd:.3f}")
    if item.transition_out.type not in ("none", "cut") and item.transition_out.duration > 0:
        if item.transition_out.type == "dip_to_white":
            fd = min(dur * 0.35, item.transition_out.duration)
            st = max(0.0, dur - fd)
            post_filters.append(f"fade=t=out:st={st:.3f}:d={fd:.3f}:c=#F6F2E8")
        elif item.transition_out.type in ("dissolve", "crossfade"):
            fd = min(dur * 0.25, item.transition_out.duration, 0.26)
            st = max(0.0, dur - fd)
            post_filters.append(f"fade=t=out:st={st:.3f}:d={fd:.3f}:c=#0B0E14")
        else:
            fd = min(dur * 0.40, item.transition_out.duration)
            st = max(0.0, dur - fd)
            post_filters.append(f"fade=t=out:st={st:.3f}:d={fd:.3f}")

    if hdr:
        post_filters.extend(
            build_hdr_color_filters(
                src_trc=src_trc if src_trc != "bt709" else "709",
                src_primaries=src_prim,
                src_space=src_space,
                is_log=is_log_clip,
                diffuse_white_nits=diffuse_nits,
            )
        )
    else:
        post_filters.append("format=yuv420p")

    speed_prefix = f"setpts={1.0 / item.speed:.4f}*PTS," if (item.speed != 1.0 and item.speed > 0) else ""

    # Build filtergraph: handles portrait 9:16 ambient blurred backdrop (+ optional Live Video+Photo Diptych) + frosted card
    filter_complex_parts: list[str] = []
    if is_portrait:
        fg_h = int(round(height * (0.68 if has_bottom_seg_title else (0.74 if (has_doc_card or has_paired_photo) else 0.84))))
        fg_h = fg_h - (fg_h % 2)
        pad_px = max(6, int(round(12 * (width / 1920.0))))
        pad_px = pad_px - (pad_px % 2)
        y_pos = int(round(height * (0.05 if (has_doc_card or has_paired_photo or has_bottom_seg_title) else 0.08)))
        x_expr = "(W*28/100)-(w/2)" if has_paired_photo else "(W-w)/2"
        filter_complex_parts.append(
            f"[0:v]{speed_prefix}split=2[vbg][vfg];"
            f"[vbg]scale={width // 4}:{height // 4}:force_original_aspect_ratio=increase,"
            f"crop={width // 4}:{height // 4},boxblur=16:8,eq=brightness=-0.18:saturation=0.85,"
            f"scale={width}:{height}[bg_blur];"
            f"[vfg]scale=-2:{fg_h},pad=iw+{pad_px * 2}:ih+{pad_px * 2}:{pad_px}:{pad_px}:color=#F8F6F0[fg_framed];"
            f"[bg_blur][fg_framed]overlay={x_expr}:{y_pos}[vbase]"
        )
    else:
        filter_complex_parts.append(
            f"[0:v]{speed_prefix}scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black[vbase]"
        )

    cur_v = "[vbase]"
    next_in_idx = 1 if (has_audio and item.volume > 0.0) else 2
    if card_png is not None and card_png.exists():
        card_input_idx = next_in_idx
        next_in_idx += 1
        fo_st = max(0.2, dur - 0.45)
        filter_complex_parts.append(
            f"[{card_input_idx}:v]format=rgba,"
            f"fade=t=in:st=0.15:d=0.45:alpha=1,"
            f"fade=t=out:st={fo_st:.3f}:d=0.40:alpha=1[vcard];"
            f"{cur_v}[vcard]overlay=0:0:eof_action=pass[vwithcard]"
        )
        cur_v = "[vwithcard]"

    has_seg_overlays = bool(segment_broll or segment_titles)
    extra_ov_inputs: list[str] = []
    if has_seg_overlays:
        filter_complex_parts.append(f"{cur_v}{','.join(post_filters)}[vsegbase]")
        extra_ov_inputs, ov_parts, out_v_label = build_segment_overlay_filtergraph(
            base_label="[vsegbase]",
            hdr=hdr,
            next_input_idx=next_in_idx,
            segment_broll=segment_broll,
            segment_titles=segment_titles,
        )
        filter_complex_parts.extend(ov_parts)
    else:
        out_v_label = "[vout]"
        filter_complex_parts.append(f"{cur_v}{','.join(post_filters)}{out_v_label}")

    def _build_cmd(force_sw: bool, use_zero_copy_vt: bool = False, use_hw_decode: bool = True) -> list[str]:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{src_in:.3f}",
        ]
        if not force_sw and use_hw_decode:
            if hw.has_videotoolbox_hwaccel:
                cmd.extend(["-hwaccel", "videotoolbox"])
                if use_zero_copy_vt:
                    cmd.extend(["-hwaccel_output_format", "videotoolbox_vld"])
            elif hw.has_cuda_hwaccel and "422" not in src_pfmt:
                cmd.extend(["-hwaccel", "cuda"])
        cmd.extend(["-i", str(item.source)])

        if not has_audio or item.volume <= 0.0:
            cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
        if card_png is not None and card_png.exists():
            cmd.extend(["-loop", "1", "-t", f"{dur:.3f}", "-i", str(card_png)])
        if extra_ov_inputs and not use_zero_copy_vt:
            cmd.extend(extra_ov_inputs)

        fc_str = f"[0:v]scale_vt=w={width}:h={height}[vout]" if use_zero_copy_vt else ";".join(filter_complex_parts)
        v_map = "[vout]" if use_zero_copy_vt else out_v_label

        if not has_audio or item.volume <= 0.0:
            cmd.extend(
                [
                    "-t",
                    f"{dur:.3f}",
                    "-filter_complex_threads",
                    "4",
                    "-filter_complex",
                    fc_str,
                    "-map",
                    v_map,
                    "-map",
                    "1:a:0",
                ]
            )
        else:
            af_parts = [
                "aresample=48000",
                "aformat=sample_fmts=fltp:channel_layouts=stereo",
                f"volume={item.volume:.3f}",
            ]
            if item.speed != 1.0 and 0.5 <= item.speed <= 2.0:
                af_parts.append(f"atempo={item.speed:.3f}")
            cf_in = max(item.audio_crossfade_sec, item.transition_in.duration)
            cf_out = max(item.audio_crossfade_sec, item.transition_out.duration)
            if cf_in > 0:
                af_parts.append(f"afade=t=in:st=0:d={min(0.45, cf_in):.3f}")
            if cf_out > 0:
                fd = min(0.45, cf_out)
                af_parts.append(f"afade=t=out:st={max(0.0, dur - fd):.3f}:d={fd:.3f}")
            cmd.extend(
                [
                    "-t",
                    f"{dur:.3f}",
                    "-filter_complex_threads",
                    "4",
                    "-filter_complex",
                    fc_str,
                    "-af",
                    ",".join(af_parts),
                    "-map",
                    v_map,
                    "-map",
                    f"0:a:{audio_pos}",
                ]
            )

        v_enc = _video_encoder_args(hdr, force_software=force_sw)
        if use_zero_copy_vt:
            extra_vt = ["-fps_mode", "cfr", "-aspect", f"{width}:{height}"]
            if "-color_range" not in v_enc:
                extra_vt.extend(["-color_range", "tv"])
            v_enc = _strip_pix_fmt_args(v_enc) + extra_vt

        cmd.extend(
            [
                *v_enc,
                *build_uniform_mux_args(fps=fps, hdr=hdr, v_enc_args=v_enc),
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-shortest",
                str(tmp_out),
            ]
        )
        return cmd

    from ..hardware import videotoolbox_hdr_encode_guard

    try:
        hw_enc_args = _video_encoder_args(hdr, force_software=False)
        # Keep VideoToolbox hwdecode on macOS; use deterministic CPU decode before CPU 10-bit filters on CUDA systems
        # so multi-worker runs never trigger NVDEC OpenGOP -ss surface tears or WDDM VRAM contention.
        prefer_hw_dec = bool(hw.has_videotoolbox_hwaccel)
        with nvenc_encode_guard(hw_enc_args), videotoolbox_hdr_encode_guard(hdr=hdr, v_enc_args=hw_enc_args):
            proc = subprocess.run(
                _build_cmd(force_sw=False, use_zero_copy_vt=can_zero_copy_vt, use_hw_decode=prefer_hw_dec),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0 and can_zero_copy_vt:
                tmp_out.unlink(missing_ok=True)
                proc = subprocess.run(
                    _build_cmd(force_sw=False, use_zero_copy_vt=False, use_hw_decode=prefer_hw_dec),
                    capture_output=True,
                    text=True,
                    check=False,
                )
            if proc.returncode != 0 and prefer_hw_dec:
                # Retry hardware encode with CPU decode if source pixel format rejects hwaccel
                tmp_out.unlink(missing_ok=True)
                proc = subprocess.run(
                    _build_cmd(force_sw=False, use_zero_copy_vt=False, use_hw_decode=False),
                    capture_output=True,
                    text=True,
                    check=False,
                )
        if proc.returncode != 0:
            # Automatic retry with software libx265 if hardware encoder session limit was reached
            tmp_out.unlink(missing_ok=True)
            proc = subprocess.run(
                _build_cmd(force_sw=True, use_zero_copy_vt=False, use_hw_decode=False),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg video segment render failed for {item.source}: {proc.stderr}")
        if is_portrait and not has_ffmpeg_filter("sidedata") and tmp_out.exists():
            # Fallback container rotation strip only when FFmpeg lacks libavfilter `sidedata`
            clean_tmp = tmp_out.with_suffix(".norot.mp4")
            remux = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-display_rotation:v:0",
                    "0",
                    "-i",
                    str(tmp_out),
                    "-c",
                    "copy",
                    "-metadata:s:v:0",
                    "rotate=0",
                    *(["-tag:v", "hvc1"] if hdr else []),
                    str(clean_tmp),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if remux.returncode == 0 and clean_tmp.exists() and clean_tmp.stat().st_size > 1024:
                os.replace(clean_tmp, tmp_out)
            else:
                clean_tmp.unlink(missing_ok=True)
        os.replace(tmp_out, output_mp4)
        return output_mp4
    finally:
        try:
            tmp_out.unlink(missing_ok=True)
        except OSError:
            pass
        if card_png is not None:
            try:
                card_png.unlink(missing_ok=True)
            except OSError:
                pass


def _build_ducking_volume_expr(
    timeline: EditTimeline,
    base_vol: float,
    duck_vol: float,
    diegetic_windows: Optional[list[tuple[float, float]]] = None,
) -> str:
    """
    Build an FFmpeg volume expression that:
    1. Deep-mutes/ducks BGM by -30 dB+ during clips flagged with has_diegetic_music (live stage shows / singing).
    2. Ducks BGM by -14 dB during spoken dialogue or voiceover narration.
    3. Never ducks BGM for silent documentary/educational text captions (`speaker` in {'', 'narrator', 'caption', 'plaque'}).
    """
    windows: list[tuple[float, float]] = []
    for sub in timeline.subtitles.items:
        spk = (sub.speaker or "").strip().lower()
        if spk and spk not in ("narrator", "caption", "plaque", "info", "subtitle", "doc"):
            windows.append((max(0.0, sub.start - 0.35), sub.end + 0.45))

    for clip in timeline.visual_track:
        if clip.type == "video" and clip.volume >= 0.45:
            windows.append((clip.timeline_start, clip.timeline_start + clip.duration))

    for a_trk in timeline.audio_tracks:
        if a_trk.role == "voiceover":
            vo_dur = a_trk.duration or 3.2
            windows.append((max(0.0, a_trk.timeline_start - 0.25), a_trk.timeline_start + vo_dur + 0.35))

    def _merge_wins(raw: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not raw:
            return []
        srt = sorted(raw, key=lambda w: w[0])
        out: list[tuple[float, float]] = [srt[0]]
        for s, e in srt[1:]:
            ps, pe = out[-1]
            if s <= pe + 0.4:
                out[-1] = (ps, max(pe, e))
            else:
                out.append((s, e))
        return out

    merged_speech = _merge_wins(windows)
    merged_diegetic = _merge_wins(diegetic_windows or [])

    if not merged_speech and not merged_diegetic:
        return f"{base_vol:.4f}"

    base_expr = f"{base_vol:.4f}"
    if merged_speech:
        cond_speech = "+".join(f"between(t,{s:.2f},{e:.2f})" for s, e in merged_speech[:25])
        base_expr = f"if(gt({cond_speech},0),{duck_vol:.4f},{base_expr})"

    if merged_diegetic:
        # -30 dB attenuation (~0.0316x of base_vol, exceeding -28 dB minimum diegetic ducking requirement)
        diegetic_vol = round(base_vol * (10.0 ** (-30.0 / 20.0)), 5)
        cond_dieg = "+".join(f"between(t,{s:.2f},{e:.2f})" for s, e in merged_diegetic[:20])
        base_expr = f"if(gt({cond_dieg},0),{diegetic_vol:.5f},{base_expr})"

    return base_expr


def _render_single_title_artifact(task: dict[str, Any]) -> dict[str, Any]:
    """Worker-safe renderer for one static or kinetic Track V3 title overlay + optional glass mask."""
    t_item = TitleOverlayItem.model_validate(task["item_dict"])
    width = int(task["width"])
    height = int(task["height"])
    fps = int(task["fps"])
    hdr = bool(task["hdr"])
    kind = str(task["kind"])
    ov_path = Path(str(task["overlay_path"]))
    glass_rect_raw = task.get("glass_rect")
    glass_rect = tuple(int(v) for v in glass_rect_raw) if glass_rect_raw is not None else None
    mask_path_str = task.get("glass_mask_path")

    ov_path.parent.mkdir(parents=True, exist_ok=True)
    if glass_rect is not None and mask_path_str:
        mask_p = Path(str(mask_path_str))
        if not (mask_p.exists() and mask_p.stat().st_size > 64):
            tmp_mask = mask_p.with_name(f".{mask_p.stem}.tmp_{os.getpid()}{mask_p.suffix}")
            try:
                render_glass_mask_png(glass_rect, tmp_mask)
                os.replace(tmp_mask, mask_p)
            except Exception:
                tmp_mask.unlink(missing_ok=True)
                raise

    if kind == "static_png":
        tmp_ov = ov_path.with_name(f".{ov_path.stem}.tmp_{os.getpid()}{ov_path.suffix}")
        try:
            render_title_png(t_item, tmp_ov, width=width, height=height, hdr_safe=hdr)
            os.replace(tmp_ov, ov_path)
        except Exception:
            tmp_ov.unlink(missing_ok=True)
            raise
        roi = (0, 0, width, height)
    else:
        tmp_ov = ov_path.with_name(f".{ov_path.stem}.tmp_{os.getpid()}.mov")
        try:
            _, roi = render_kinetic_title_mov(
                t_item,
                tmp_ov,
                width=width,
                height=height,
                fps=fps,
                hdr_safe=hdr,
                transparent_glass_body=(glass_rect is not None),
            )
            os.replace(tmp_ov, ov_path)
        except Exception:
            tmp_ov.unlink(missing_ok=True)
            raise

    return {
        "idx": int(task["idx"]),
        "kind": kind,
        "overlay_path": str(ov_path),
        "roi": tuple(int(v) for v in roi),
        "glass_rect": glass_rect,
        "glass_mask_path": str(mask_path_str) if mask_path_str else None,
    }


def render_timeline(
    timeline: EditTimeline,
    output_mp4: str | Path,
    work_dir: Optional[str | Path] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> dict[str, Path]:
    """Render an EditTimeline (Track V1 + Track V2 B-Roll Cutaways + Track V3 Titles + 3-Layer Audio) into MP4."""
    if not timeline.visual_track:
        raise ValueError("Cannot render an empty visual_track.")

    for track_item in (*timeline.visual_track, *timeline.broll_track):
        if track_item.type == "video" and is_live_photo_video(track_item.source):
            raise ValueError(
                f"Live Photo / Motion Photo video '{track_item.source}' (clip '{track_item.id}') "
                "is prohibited due to low resolution and pre-shutter motion blur; use the full-resolution "
                "still photo or a standalone camera video instead."
            )

    out_path = Path(output_mp4).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_dir = Path(work_dir).resolve() if work_dir else (out_path.parent / "render_cache")
    seg_dir = w_dir / "segments"
    titles_dir = w_dir / "titles"
    seg_dir.mkdir(parents=True, exist_ok=True)
    titles_dir.mkdir(parents=True, exist_ok=True)

    timeline.recalculate_sequential_starts()

    hw = get_hardware_profile()
    render_workers_cap = get_render_worker_count(hw)
    width = timeline.settings.width
    height = timeline.settings.height
    fps = timeline.settings.fps
    hdr = timeline.settings.hdr
    total_dur = timeline.total_duration
    temp_title_artifacts: list[Path] = []

    with caffeinate_guard(), _unlink_on_exit(temp_title_artifacts):
        # 1. Export .srt and .ass subtitle files alongside the output video
        srt_path = out_path.with_suffix(".srt")
        ass_path = out_path.with_suffix(".ass")
        export_srt_subtitles(timeline, srt_path)
        export_ass_subtitles(timeline, ass_path, hdr=hdr)

        effects_by_clip: dict[str, list[EffectTrackItem]] = {}
        for fx in timeline.effects_track:
            if fx.target_clip_id:
                effects_by_clip.setdefault(fx.target_clip_id, []).append(fx)

        # 2a. Prepare Track V2 B-Roll cutaways and Track V3 title plates BEFORE Track V1 so segment-scoped
        #     overlays bake directly into V1 segments and Step 4 final mux uses -c:v copy.
        active_broll = [b for b in timeline.broll_track if b.timeline_start < total_dur]
        broll_files_all: list[Path] = [Path()] * len(active_broll)
        broll_tasks: list[tuple[int, VisualTrackItem, Path, list[EffectTrackItem]]] = []
        for b_idx, b_item in enumerate(active_broll):
            b_fxs = effects_by_clip.get(b_item.id, [])
            sig_payload = json.dumps(
                {
                    "v": 6,
                    "mux_norm": "tv_norot_v2",
                    "b_item": b_item.model_dump(mode="json"),
                    "fx": [f.model_dump(mode="json") for f in b_fxs],
                    "w": width,
                    "h": height,
                    "fps": fps,
                    "hdr": hdr,
                },
                sort_keys=True,
            )
            sig_hash = hashlib.sha1(sig_payload.encode("utf-8")).hexdigest()[:8]
            b_mp4 = seg_dir / f"broll_{b_idx + 1:03d}_{b_item.id}_{sig_hash}.mp4"
            broll_files_all[b_idx] = b_mp4
            if not b_mp4.exists() or b_mp4.stat().st_size < 1024:
                broll_tasks.append((b_idx, b_item, b_mp4, b_fxs))

        if broll_tasks:
            def _render_broll_one(bt: tuple[int, VisualTrackItem, Path, list[EffectTrackItem]]) -> Path:
                _, b_item, b_mp4, b_fxs = bt
                if b_item.type == "photo":
                    return render_photo_clip(
                        photo_path=b_item.source,
                        output_mp4=b_mp4,
                        duration=b_item.duration,
                        width=width,
                        height=height,
                        fps=fps,
                        hdr=hdr,
                        ken_burns=b_item.ken_burns,
                        transition_in=b_item.transition_in,
                        transition_out=b_item.transition_out,
                        attached_effects=b_fxs,
                    )
                return _render_video_segment(
                    item=b_item,
                    output_mp4=b_mp4,
                    width=width,
                    height=height,
                    fps=fps,
                    hdr=hdr,
                    attached_effects=b_fxs,
                )

            b_workers = max(1, min(render_workers_cap, len(broll_tasks)))
            with ThreadPoolExecutor(max_workers=b_workers) as pool:
                for fut in as_completed([pool.submit(_render_broll_one, bt) for bt in broll_tasks]):
                    fut.result()

        active_titles = [
            t
            for t in timeline.title_track
            if t.timeline_start < total_dur
        ]
        title_entries_all: list[dict[str, Any]] = [{} for _ in active_titles]
        title_miss_tasks: list[dict[str, Any]] = []
        for t_idx, t_item in enumerate(active_titles):
            glass_rect = compute_glass_pill_rect(t_item, width=width, height=height) if t_item.backdrop_blur else None
            kind = "static_png" if getattr(t_item, "motion_preset", "masked_rise") == "static" else "kinetic_mov"
            roi = (0, 0, width, height) if kind == "static_png" else compute_kinetic_roi(t_item, width=width, height=height)
            item_dict = t_item.model_dump(mode="json")
            t_sig_payload = json.dumps(
                {
                    "v": "kin_v3",
                    "item": item_dict,
                    "w": width,
                    "h": height,
                    "fps": fps,
                    "hdr": hdr,
                    "glass": glass_rect is not None,
                },
                sort_keys=True,
            )
            t_hash = hashlib.sha1(t_sig_payload.encode("utf-8")).hexdigest()[:8]
            safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(t_item.id or "title"))[:48] or "title"
            glass_mask_path: Optional[Path] = (
                titles_dir / f"glass_mask_{t_idx + 1:03d}_{safe_id}_{t_hash}.png"
                if glass_rect is not None
                else None
            )
            ov_path = titles_dir / (
                f"title_{t_idx + 1:03d}_{safe_id}_{t_hash}.png"
                if kind == "static_png"
                else f"kin_title_{t_idx + 1:03d}_{safe_id}_{t_hash}.mov"
            )
            ov_ok = ov_path.exists() and ov_path.stat().st_size > 100
            mask_ok = (glass_mask_path is None) or (glass_mask_path.exists() and glass_mask_path.stat().st_size > 64)
            if ov_ok and mask_ok:
                title_entries_all[t_idx] = {
                    "item": t_item,
                    "kind": kind,
                    "overlay_path": ov_path,
                    "roi": roi,
                    "glass_rect": glass_rect,
                    "glass_mask_path": glass_mask_path,
                }
            else:
                title_miss_tasks.append(
                    {
                        "idx": t_idx,
                        "item_dict": item_dict,
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "hdr": hdr,
                        "kind": kind,
                        "overlay_path": str(ov_path),
                        "glass_rect": tuple(int(v) for v in glass_rect) if glass_rect is not None else None,
                        "glass_mask_path": str(glass_mask_path) if glass_mask_path is not None else None,
                    }
                )

        def _store_rendered_title(res: dict[str, Any]) -> None:
            nonlocal done_titles
            r_idx = int(res["idx"])
            title_entries_all[r_idx] = {
                "item": active_titles[r_idx],
                "kind": res["kind"],
                "overlay_path": Path(res["overlay_path"]),
                "roi": tuple(res["roi"]),
                "glass_rect": res["glass_rect"],
                "glass_mask_path": Path(res["glass_mask_path"]) if res["glass_mask_path"] else None,
            }
            done_titles += 1
            if progress_callback:
                progress_callback(
                    f"Rendered title overlay {done_titles}/{len(active_titles)} ({active_titles[r_idx].id})",
                    0.05 * (done_titles / max(1, len(active_titles))),
                )

        if len(title_miss_tasks) == 1:
            done_titles = len(active_titles) - 1
            _store_rendered_title(_render_single_title_artifact(title_miss_tasks[0]))
        elif title_miss_tasks:
            title_workers = min(16, max(2, (os.cpu_count() or 8) // 2), len(title_miss_tasks))
            done_titles = len(active_titles) - len(title_miss_tasks)
            try:
                with ProcessPoolExecutor(max_workers=title_workers) as pool:
                    futs = {pool.submit(_render_single_title_artifact, tk): tk for tk in title_miss_tasks}
                    for fut in as_completed(futs):
                        _store_rendered_title(fut.result())
            except (PermissionError, OSError, NotImplementedError):
                # Fallback to threaded execution only when spawning worker processes is blocked
                with ThreadPoolExecutor(max_workers=min(8, title_workers)) as pool:
                    futs = {pool.submit(_render_single_title_artifact, tk): tk for tk in title_miss_tasks}
                    for fut in as_completed(futs):
                        _store_rendered_title(fut.result())

        num_clips = len(timeline.visual_track)
        v1_broll_entries: list[list[tuple[VisualTrackItem, Path, float, float]]] = [[] for _ in range(num_clips)]
        v1_title_entries: list[list[dict[str, Any]]] = [[] for _ in range(num_clips)]
        step4_broll_pairs: list[tuple[VisualTrackItem, Path]] = []
        step4_title_entries: list[dict[str, Any]] = []

        def _find_containing_segment(ov_start: float, ov_end: float) -> Optional[int]:
            for idx, v_item in enumerate(timeline.visual_track):
                c_st = v_item.timeline_start
                c_en = c_st + v_item.duration
                if ov_start >= c_st - 0.02 and ov_end <= c_en + 0.02:
                    return idx
            return None

        for b_item, b_mp4 in zip(active_broll, broll_files_all):
            seg_idx = _find_containing_segment(b_item.timeline_start, b_item.timeline_start + b_item.duration)
            if seg_idx is not None:
                v_item = timeline.visual_track[seg_idx]
                local_st = max(0.0, round(b_item.timeline_start - v_item.timeline_start, 3))
                local_dur = round(max(0.1, min(v_item.duration - local_st, b_item.duration)), 3)
                v1_broll_entries[seg_idx].append((b_item, b_mp4, local_st, local_dur))
            else:
                step4_broll_pairs.append((b_item, b_mp4))

        for ent in title_entries_all:
            t_item = ent["item"]
            seg_idx = _find_containing_segment(t_item.timeline_start, t_item.timeline_start + t_item.duration)
            if seg_idx is not None:
                v_item = timeline.visual_track[seg_idx]
                local_st = max(0.0, round(t_item.timeline_start - v_item.timeline_start, 3))
                local_dur = round(max(0.1, min(v_item.duration - local_st, t_item.duration)), 3)
                v1_title_entries[seg_idx].append(
                    {
                        **ent,
                        "local_st": local_st,
                        "local_dur": local_dur,
                        "fade_in": t_item.fade_in,
                        "fade_out": t_item.fade_out,
                        "blur_sigma": getattr(t_item, "backdrop_blur_sigma", 18.0),
                    }
                )
            else:
                step4_title_entries.append(ent)

        # 2b. Render Track V1 visual clips in parallel (MPS photo synthesis + Apple T2 VideoToolbox)
        segment_files: list[Path] = [Path()] * num_clips
        tasks_to_run: list[
            tuple[
                int,
                VisualTrackItem,
                Path,
                list[EffectTrackItem],
                list[tuple[Path, float, float]],
                list[dict[str, Any]],
            ]
        ] = []

        for idx, item in enumerate(timeline.visual_track):
            clip_fxs = effects_by_clip.get(item.id, [])
            b_assigned = v1_broll_entries[idx]
            t_assigned = v1_title_entries[idx]
            has_stab = any(f.effect_type == "video_stabilize" for f in clip_fxs)
            sig_payload = json.dumps(
                {
                    "v": 6,
                    "mux_norm": "tv_norot_v2",
                    "stab_v": "crop_v2" if has_stab else "",
                    **({"dec_v": "sw_clean_v1"} if item.type == "video" else {}),
                    "item": item.model_dump(mode="json"),
                    "fx": [f.model_dump(mode="json") for f in clip_fxs],
                    "seg_broll": [
                        {
                            "item": bi.model_dump(mode="json"),
                            "st": lst,
                            "dur": ldur,
                            "mtime_ns": int(bmp.stat().st_mtime_ns) if bmp.exists() else 0,
                        }
                        for bi, bmp, lst, ldur in b_assigned
                    ],
                    "seg_titles": [
                        {
                            "item": te["item"].model_dump(mode="json"),
                            "st": te["local_st"],
                            "dur": te["local_dur"],
                        }
                        for te in t_assigned
                    ],
                    "w": width,
                    "h": height,
                    "fps": fps,
                    "hdr": hdr,
                },
                sort_keys=True,
            )
            sig_hash = hashlib.sha1(sig_payload.encode("utf-8")).hexdigest()[:8]
            seg_mp4 = seg_dir / f"seg_{idx + 1:03d}_{item.id}_{sig_hash}.mp4"
            segment_files[idx] = seg_mp4
            if not seg_mp4.exists() or seg_mp4.stat().st_size < 1024:
                seg_b_args = [(mp4, lst, ldur) for _, mp4, lst, ldur in b_assigned]
                tasks_to_run.append((idx, item, seg_mp4, clip_fxs, seg_b_args, t_assigned))

        def _render_one_v1(
            task_tuple: tuple[
                int,
                VisualTrackItem,
                Path,
                list[EffectTrackItem],
                list[tuple[Path, float, float]],
                list[dict[str, Any]],
            ],
        ) -> tuple[int, Path]:
            t_idx, t_item, t_mp4, t_fxs, t_broll, t_titles = task_tuple
            if t_item.type == "photo":
                render_photo_clip(
                    photo_path=t_item.source,
                    output_mp4=t_mp4,
                    duration=t_item.duration,
                    width=width,
                    height=height,
                    fps=fps,
                    hdr=hdr,
                    ken_burns=t_item.ken_burns,
                    transition_in=t_item.transition_in,
                    transition_out=t_item.transition_out,
                    segment_broll=t_broll or None,
                    segment_titles=t_titles or None,
                    attached_effects=t_fxs,
                )
            else:
                _render_video_segment(
                    item=t_item,
                    output_mp4=t_mp4,
                    width=width,
                    height=height,
                    fps=fps,
                    hdr=hdr,
                    attached_effects=t_fxs,
                    segment_broll=t_broll or None,
                    segment_titles=t_titles or None,
                )
            return t_idx, t_mp4

        if tasks_to_run:
            max_workers = max(1, min(render_workers_cap, len(tasks_to_run)))
            completed_count = num_clips - len(tasks_to_run)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_render_one_v1, t): t for t in tasks_to_run}
                for fut in as_completed(futures):
                    done_idx, _ = fut.result()
                    completed_count += 1
                    if progress_callback:
                        frac_v1 = (completed_count / max(1, num_clips)) * 0.68
                        progress_callback(
                            f"Rendered V1 clip {completed_count}/{num_clips} ({Path(timeline.visual_track[done_idx].source).name})",
                            max(0.05 if active_titles else 0.0, frac_v1),
                        )

        # 3. Concatenate normalized V1 segments into base visual + camera audio stream
        if progress_callback:
            progress_callback("Concatenating Track V1 storyline segments...", 0.75)

        base_concat_mp4 = w_dir / "base_concat.mp4"
        _concat_segments_safe(
            segment_files=segment_files,
            output_mp4=base_concat_mp4,
            w_dir=w_dir,
            fps=fps,
            hdr=hdr,
        )

        # 4. Composite any cross-cut V2/V3 overlays + mix mastered BGM (using -c:v copy when V2/V3 already baked)
        if progress_callback:
            progress_callback("Compositing final timeline and mastered audio mix...", 0.85)

        step4_broll = [p[0] for p in step4_broll_pairs]
        broll_files = [p[1] for p in step4_broll_pairs]

        diegetic_windows: list[tuple[float, float]] = []
        mix_manifest_path = out_path.parent / "audio_mix_manifest.json"
        if mix_manifest_path.exists():
            try:
                mix_raw = json.loads(mix_manifest_path.read_text(encoding="utf-8"))
                for seg in mix_raw.get("ducking_envelope", []) or []:
                    if isinstance(seg, dict) and seg.get("mode") == "diegetic_music_mute":
                        diegetic_windows.append((float(seg.get("start_sec", 0.0)), float(seg.get("end_sec", 0.0))))
            except Exception:
                pass

        active_bgm = [a for a in timeline.audio_tracks if Path(a.source).exists()]
        burn_subs = bool(
            timeline.subtitles.enabled
            and timeline.subtitles.burn_in
            and timeline.subtitles.items
            and ass_path.exists()
        )
        has_step4_video_ops = bool(step4_broll or step4_title_entries or burn_subs)

        if not has_step4_video_ops and not active_bgm:
            shutil.copy2(base_concat_mp4, out_path)
            if progress_callback:
                progress_callback("Render complete!", 1.0)
            return {"video": out_path, "srt": srt_path, "ass": ass_path}

        def _build_final_cmd(force_sw: bool) -> list[str]:
            final_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(base_concat_mp4),
            ]

            for b_mp4 in broll_files:
                final_cmd.extend(["-i", str(b_mp4)])

            next_in_idx = 1 + len(step4_broll)
            for ent in step4_title_entries:
                t_item = ent["item"]
                if ent["glass_mask_path"] is not None:
                    final_cmd.extend(["-loop", "1", "-t", f"{t_item.duration:.3f}", "-i", str(ent["glass_mask_path"])])
                    ent["mask_in_idx"] = next_in_idx
                    next_in_idx += 1
                else:
                    ent["mask_in_idx"] = None
                if ent["kind"] == "static_png":
                    final_cmd.extend(["-loop", "1", "-t", f"{t_item.duration:.3f}", "-i", str(ent["overlay_path"])])
                else:
                    final_cmd.extend(["-i", str(ent["overlay_path"])])
                ent["ov_in_idx"] = next_in_idx
                next_in_idx += 1

            bgm_input_start_idx = next_in_idx
            for a_item in active_bgm:
                if a_item.role == "music":
                    # Enable -stream_loop -1 on BGM music tracks so they cover the full video runtime
                    final_cmd.extend(["-stream_loop", "-1", "-ss", f"{a_item.source_in:.3f}", "-i", str(a_item.source)])
                else:
                    final_cmd.extend(["-ss", f"{a_item.source_in:.3f}", "-i", str(a_item.source)])

            filter_parts: list[str] = []
            cur_v = "[0:v]"
            ov_fmt = "yuv444p10" if hdr else "yuv444"

            # Composite cross-segment Track V2 B-Roll Cutaways over Track V1
            for b_idx, b_item in enumerate(step4_broll):
                in_idx = 1 + b_idx
                st = b_item.timeline_start
                dur = b_item.duration
                end_t = st + dur
                bv_label = f"[bv{b_idx}]"
                next_v = f"[vb{b_idx + 1}]"
                filter_parts.append(f"[{in_idx}:v]setpts=PTS+{st:.3f}/TB{bv_label}")
                filter_parts.append(
                    f"{cur_v}{bv_label}overlay=0:0:format={ov_fmt}:eof_action=pass:enable='between(t,{st:.3f},{end_t:.3f})'{next_v}"
                )
                cur_v = next_v

            # Composite cross-segment Track V3 Live Frosted-Glass Refraction & Kinetic/RGBA Title Overlays
            for idx, ent in enumerate(step4_title_entries):
                t_item = ent["item"]
                in_idx = int(ent["ov_in_idx"])
                st = t_item.timeline_start
                dur = t_item.duration
                end_t = st + dur
                fi = min(dur * 0.4, t_item.fade_in)
                fo = min(dur * 0.4, t_item.fade_out)
                fo_st = max(0.0, dur - fo)

                if ent["glass_rect"] is not None and ent["mask_in_idx"] is not None:
                    gx, gy, gw, gh, _ = ent["glass_rect"]
                    mask_idx = int(ent["mask_in_idx"])
                    sigma = max(4.0, float(getattr(t_item, "backdrop_blur_sigma", 18.0)) * 0.65)
                    filter_parts.append(f"{cur_v}split=2[v_main{idx}][v_src{idx}]")
                    filter_parts.append(
                        f"[v_src{idx}]trim=start={st:.3f}:end={end_t:.3f},setpts=PTS-STARTPTS,"
                        f"crop={gw}:{gh}:{gx}:{gy},"
                        f"scale=w=max(16\\,trunc(iw/8)*2):h=max(16\\,trunc(ih/8)*2):flags=bilinear,"
                        f"gblur=sigma={sigma:.1f},scale={gw}:{gh}:flags=bicubic,"
                        f"lutyuv=y='max(16,(val-16)*0.74+16)',format=yuva444p[v_blur{idx}]"
                    )
                    filter_parts.append(
                        f"[{mask_idx}:v]format=gray,"
                        f"fade=t=in:st=0:d={fi:.3f},"
                        f"fade=t=out:st={fo_st:.3f}:d={fo:.3f}[v_gmask{idx}]"
                    )
                    filter_parts.append(f"[v_blur{idx}][v_gmask{idx}]alphamerge,setpts=PTS+{st:.3f}/TB[v_glass{idx}]")
                    filter_parts.append(
                        f"[v_main{idx}][v_glass{idx}]overlay=x={gx}:y={gy}:format={ov_fmt}:"
                        f"eof_action=pass:enable='between(t,{st:.3f},{end_t:.3f})'[v_pg{idx}]"
                    )
                    cur_v = f"[v_pg{idx}]"

                mov_rx, mov_ry, _, _ = ent["roi"]
                ov_label = f"[ov{idx}]"
                next_v = f"[vt{idx + 1}]"
                if ent["kind"] == "static_png":
                    filter_parts.append(
                        f"[{in_idx}:v]format=rgba,"
                        f"fade=t=in:st=0:d={fi:.3f}:alpha=1,"
                        f"fade=t=out:st={fo_st:.3f}:d={fo:.3f}:alpha=1,"
                        f"setpts=PTS+{st:.3f}/TB{ov_label}"
                    )
                else:
                    filter_parts.append(f"[{in_idx}:v]format=rgba,setpts=PTS+{st:.3f}/TB{ov_label}")
                filter_parts.append(
                    f"{cur_v}{ov_label}overlay=x={mov_rx}:y={mov_ry}:format={ov_fmt}:"
                    f"eof_action=pass:enable='between(t,{st:.3f},{end_t:.3f})'{next_v}"
                )
                cur_v = next_v

            if burn_subs:
                esc_ass = _escape_ffmpeg_filter_path(ass_path)
                filter_parts.append(f"{cur_v}ass=filename='{esc_ass}'[vsub]")
                cur_v = "[vsub]"

            if has_step4_video_ops:
                if hdr:
                    filter_parts.append(
                        f"{cur_v}scale=flags=lanczos+accurate_rnd,format=p010le,"
                        f"setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:range=tv[vhdr]"
                    )
                    cur_v = "[vhdr]"
                else:
                    filter_parts.append(f"{cur_v}scale=flags=lanczos+accurate_rnd,format=yuv420p[vsdr]")
                    cur_v = "[vsdr]"

            # Audio mixing & smart auto-ducking
            if active_bgm:
                mix_inputs = ["[0:a]"]
                for a_idx, a_item in enumerate(active_bgm):
                    in_idx = bgm_input_start_idx + a_idx
                    # Ensure primary BGM covers 100% of total_dur when starting at 0.0, while voiceover/sfx play their exact duration
                    if a_item.role == "music" and a_item.timeline_start <= 0.1:
                        a_dur = total_dur
                    else:
                        a_dur = min(max(0.2, total_dur - a_item.timeline_start), a_item.duration or total_dur)
                    vol_expr = (
                        _build_ducking_volume_expr(
                            timeline,
                            a_item.volume,
                            a_item.duck_volume,
                            diegetic_windows=diegetic_windows,
                        )
                        if a_item.duck_under_dialogue
                        else f"{a_item.volume:.3f}"
                    )
                    af_chain = [
                        f"atrim=0:{a_dur:.3f}",
                        "asetpts=PTS-STARTPTS",
                        "aresample=48000",
                        "aformat=sample_fmts=fltp:channel_layouts=stereo",
                        f"volume='{vol_expr}':eval=frame",
                    ]
                    if a_item.fade_in > 0:
                        af_chain.append(f"afade=t=in:st=0:d={min(a_dur * 0.4, a_item.fade_in):.3f}")
                    if a_item.fade_out > 0:
                        fo = min(a_dur * 0.4, a_item.fade_out)
                        af_chain.append(f"afade=t=out:st={max(0.0, a_dur - fo):.3f}:d={fo:.3f}")
                    if a_item.timeline_start > 0:
                        delay_ms = int(round(a_item.timeline_start * 1000))
                        af_chain.append(f"adelay={delay_ms}|{delay_ms}")

                    lbl = f"[bgm{a_idx}]"
                    filter_parts.append(f"[{in_idx}:a]{','.join(af_chain)}{lbl}")
                    mix_inputs.append(lbl)

                filter_parts.append(
                    f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=2:normalize=0[aout]"
                )
                final_audio_map = "[aout]"
            else:
                final_audio_map = "0:a:0"

            if has_step4_video_ops:
                final_cmd.extend(
                    [
                        "-filter_complex",
                        ";".join(filter_parts),
                        "-map",
                        cur_v,
                        "-map",
                        final_audio_map,
                        "-t",
                        f"{total_dur:.3f}",
                        *_video_encoder_args(hdr, force_software=force_sw),
                        "-c:a",
                        "aac",
                        "-b:a",
                        "320k" if hdr else "192k",
                        str(out_path),
                    ]
                )
            else:
                # Fast stream-copy video when V2/V3 overlays are already baked per segment!
                final_cmd.extend(
                    [
                        "-filter_complex",
                        ";".join(filter_parts),
                        "-map",
                        "0:v:0",
                        "-map",
                        final_audio_map,
                        "-t",
                        f"{total_dur:.3f}",
                        "-c:v",
                        "copy",
                        *(["-tag:v", "hvc1"] if hdr else []),
                        "-c:a",
                        "aac",
                        "-b:a",
                        "320k" if hdr else "192k",
                        str(out_path),
                    ]
                )
            return final_cmd

        with nvenc_encode_guard(_video_encoder_args(hdr, force_software=False) if has_step4_video_ops else ()):
            proc = subprocess.run(_build_final_cmd(force_sw=False), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            proc = subprocess.run(_build_final_cmd(force_sw=True), capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg final composite failed: {proc.stderr}")

        if progress_callback:
            progress_callback("Render complete!", 1.0)

        return {"video": out_path, "srt": srt_path, "ass": ass_path}


def build_segment_subtimeline(
    timeline: EditTimeline,
    start_sec: float,
    end_sec: float,
    fast_preview: bool = True,
) -> EditTimeline:
    """Mechanically slice an EditTimeline to [start_sec, end_sec] for fast surgical preview rendering."""
    st_bound = max(0.0, float(start_sec))
    en_bound = min(float(timeline.total_duration), float(end_sec))
    if en_bound <= st_bound:
        en_bound = min(float(timeline.total_duration), st_bound + 5.0)

    sub_tl = timeline.model_copy(deep=True)
    if fast_preview:
        sub_tl.settings.width = 1280
        sub_tl.settings.height = 720
        sub_tl.settings.fps = 24
        sub_tl.settings.hdr = False

    new_v1 = []
    for clip in sub_tl.visual_track:
        c_st = clip.timeline_start
        c_en = c_st + clip.duration
        if c_en <= st_bound + 0.02 or c_st >= en_bound - 0.02:
            continue
        ov_st = max(st_bound, c_st)
        ov_en = min(en_bound, c_en)
        lead_trim = round(ov_st - c_st, 3)
        new_dur = round(max(0.5, ov_en - ov_st), 2)
        if clip.type == "video":
            clip.source_in = round(clip.source_in + lead_trim, 2)
        clip.duration = new_dur
        clip.source_out = round(clip.source_in + new_dur, 2)
        new_v1.append(clip)

    sub_tl.visual_track = new_v1
    sub_tl.recalculate_sequential_starts()
    seg_dur = round(sub_tl.total_duration, 2)

    new_broll = []
    for b in sub_tl.broll_track:
        b_st = b.timeline_start
        b_en = b_st + b.duration
        if b_en <= st_bound + 0.05 or b_st >= en_bound - 0.05:
            continue
        ov_st = max(st_bound, b_st)
        ov_en = min(en_bound, b_en)
        b.timeline_start = round(max(0.0, ov_st - st_bound), 3)
        b.duration = round(max(0.5, min(seg_dur - b.timeline_start, ov_en - ov_st)), 2)
        b.source_out = round(b.source_in + b.duration, 2)
        new_broll.append(b)
    sub_tl.broll_track = new_broll

    new_titles = []
    for t in sub_tl.title_track:
        t_st = t.timeline_start
        t_en = t_st + t.duration
        if t_en <= st_bound + 0.05 or t_st >= en_bound - 0.05:
            continue
        ov_st = max(st_bound, t_st)
        ov_en = min(en_bound, t_en)
        t.timeline_start = round(max(0.0, ov_st - st_bound), 3)
        t.duration = round(max(0.5, min(seg_dur - t.timeline_start, ov_en - ov_st)), 2)
        new_titles.append(t)
    sub_tl.title_track = new_titles

    kept_ids = {c.id for c in sub_tl.visual_track}
    new_fx = []
    for fx in sub_tl.effects_track:
        if fx.target_clip_id in kept_ids:
            target_c = next((c for c in sub_tl.visual_track if c.id == fx.target_clip_id), None)
            if target_c is not None:
                fx.timeline_start = target_c.timeline_start
                fx.duration = target_c.duration
                new_fx.append(fx)
    sub_tl.effects_track = new_fx

    new_subs = []
    for s in sub_tl.subtitles.items:
        if s.end <= st_bound + 0.05 or s.start >= en_bound - 0.05:
            continue
        s.start = round(max(0.0, s.start - st_bound), 2)
        s.end = round(min(seg_dur, s.end - st_bound), 2)
        if s.end - s.start >= 0.2:
            new_subs.append(s)
    sub_tl.subtitles.items = new_subs

    for a in sub_tl.audio_tracks:
        a.source_in = round(a.source_in + st_bound, 2)
        a.timeline_start = 0.0
        a.duration = seg_dur
        a.fade_in = min(0.4, a.fade_in)
        a.fade_out = min(0.5, a.fade_out)

    return sub_tl


def render_timeline_segment(
    timeline: EditTimeline,
    output_mp4: Optional[str | Path] = None,
    start_sec: float = 0.0,
    end_sec: float = 8.0,
    fast_preview: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    output_mp4_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Render a [start_sec, end_sec] time slice of an EditTimeline for rapid surgical previewing."""
    target_mp4 = Path(output_mp4_path or output_mp4 or "segment_preview.mp4")
    sub_tl = build_segment_subtimeline(
        timeline=timeline,
        start_sec=start_sec,
        end_sec=end_sec,
        fast_preview=fast_preview,
    )
    out = render_timeline(sub_tl, target_mp4, progress_callback=progress_callback)
    return {
        "video": str(out["video"]),
        "srt": str(out["srt"]),
        "ass": str(out["ass"]),
        "start_sec": round(float(start_sec), 2),
        "end_sec": round(float(end_sec), 2),
        "duration_sec": round(sub_tl.total_duration, 2),
        "fast_preview": fast_preview,
    }
