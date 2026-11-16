"""Technical media probing, HEIC conversion, keyframe extraction, and contact sheets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Optional

import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageFont, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

_UNSUPPORTED_AUDIO_CODECS = {"apac", "apple_apac"}


def select_decodable_audio_stream_pos(streams: list[dict]) -> Optional[int]:
    """Return 0-based audio stream index for the first non-APAC decodable audio stream."""
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    for idx, s in enumerate(audio_streams):
        codec = str(s.get("codec_name") or "").strip().lower()
        if codec not in _UNSUPPORTED_AUDIO_CODECS:
            return idx
    return None


def find_decodable_audio_stream_pos(media_path: Path) -> Optional[int]:
    """Return 0-based position of the first decodable audio stream for FFmpeg `-map 0:a:<pos>`."""
    pos = probe_video_file(media_path).get("audio_stream_pos")
    return int(pos) if pos is not None else None


from ..hardware import get_hardware_profile, get_io_worker_count
from .live_photo_filter import is_live_photo_video
from ..models.schema import (
    AudioAsset,
    DITIngestEntry,
    DITIngestManifest,
    MediaCatalog,
    PhotoAsset,
    SceneSegment,
    VideoAsset,
)

VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi", ".m4v", ".webm"}
PHOTO_EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def _get_cjk_font(size: int = 24) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _parse_fps(fps_str: str) -> float:
    if not fps_str or fps_str == "0/0":
        return 30.0
    if "/" in fps_str:
        num, den = fps_str.split("/", 1)
        try:
            d = float(den)
            return round(float(num) / d, 3) if d != 0 else 30.0
        except ValueError:
            return 30.0
    try:
        return round(float(fps_str), 3)
    except ValueError:
        return 30.0


def probe_video_file(video_path: Path) -> dict:
    """Run ffprobe on a video file and return normalized technical metadata."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "duration": 5.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "codec": "unknown",
            "color_transfer": "bt709",
            "codec": "h264",
            "color_primaries": "bt709",
            "color_transfer": "bt709",
            "color_trc": "bt709",
            "color_space": "bt709",
            "pix_fmt": "yuv420p",
            "has_audio": False,
            "audio_stream_pos": None,
            "creation_time": None,
            "camera_make_model": None,
            "gps_lat": None,
            "gps_lon": None,
            "gps_alt": None,
        }

    data = json.loads(proc.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    decodable_pos = select_decodable_audio_stream_pos(streams)
    a_stream = audio_streams[decodable_pos] if decodable_pos is not None else None

    duration = float(fmt.get("duration") or v_stream.get("duration") or 0.0)
    width = int(v_stream.get("width") or 1920)
    height = int(v_stream.get("height") or 1080)

    # Check rotation tag or side_data_list
    rotation = 0
    tags = v_stream.get("tags", {}) or {}
    if "rotate" in tags:
        try:
            rotation = int(tags["rotate"])
        except ValueError:
            pass
    for sd in v_stream.get("side_data_list", []):
        if "rotation" in sd:
            try:
                rotation = int(float(sd["rotation"]))
            except ValueError:
                pass
    if abs(rotation) in (90, 270):
        width, height = height, width

    fps = _parse_fps(v_stream.get("r_frame_rate") or v_stream.get("avg_frame_rate") or "30/1")
    codec = v_stream.get("codec_name", "h264")
    color_primaries = v_stream.get("color_primaries") or "bt709"
    color_transfer = v_stream.get("color_transfer") or "bt709"
    color_space = v_stream.get("color_space") or "bt709"
    pix_fmt = v_stream.get("pix_fmt") or "yuv420p"
    fmt_tags = fmt.get("tags", {}) or {}
    creation_time = (
        fmt_tags.get("com.apple.quicktime.creationdate")
        or fmt_tags.get("creation_time")
        or tags.get("creation_time")
    )
    make_str = str(
        fmt_tags.get("com.apple.quicktime.make") or fmt_tags.get("com.android.manufacturer") or ""
    ).strip()
    model_str = str(
        fmt_tags.get("com.apple.quicktime.model") or fmt_tags.get("com.android.model") or ""
    ).strip()
    cam_make_model = f"{make_str} {model_str}".strip() or None

    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None
    iso_loc = str(fmt_tags.get("com.apple.quicktime.location.ISO6709") or fmt_tags.get("location") or "")
    if iso_loc:
        m_gps = re.match(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?", iso_loc)
        if m_gps:
            try:
                gps_lat = round(float(m_gps.group(1)), 6)
                gps_lon = round(float(m_gps.group(2)), 6)
                if m_gps.group(3):
                    gps_alt = round(float(m_gps.group(3)), 2)
            except ValueError:
                pass

    return {
        "duration": round(duration, 3),
        "width": width,
        "height": height,
        "fps": fps,
        "codec": codec,
        "color_primaries": color_primaries,
        "color_transfer": color_transfer,
        "color_trc": color_transfer,
        "color_space": color_space,
        "pix_fmt": pix_fmt,
        "has_audio": a_stream is not None,
        "audio_stream_pos": decodable_pos,
        "creation_time": creation_time,
        "camera_make_model": cam_make_model,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "gps_alt": gps_alt,
    }


def compute_file_fingerprint(path: Path) -> tuple[int, int, str]:
    """Return (file_size_bytes, mtime_ns, fast_sha1_hex) using head+tail 64KB chunks for O(1) ingest caching."""
    import hashlib

    st = path.stat()
    h = hashlib.sha1()
    h.update(f"{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
    try:
        with open(path, "rb") as f:
            head = f.read(65536)
            h.update(head)
            if st.st_size > 131072:
                f.seek(max(0, st.st_size - 65536))
                h.update(f.read(65536))
    except Exception:
        pass
    return st.st_size, st.st_mtime_ns, h.hexdigest()[:20]


def probe_audio_signal_metrics(media_path: Path, max_seconds: float = 16.0) -> dict[str, Any]:
    """Measure integrated LUFS approximation, SNR (dB), and diegetic music presence from mono 8kHz PCM."""
    import math

    default_res = {"integrated_lufs": -24.0, "snr_db": 22.0, "has_diegetic_music": False}
    if not media_path.exists():
        return default_res
    a_pos = find_decodable_audio_stream_pos(media_path)
    if a_pos is None:
        return default_res
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        f"{max_seconds:.1f}",
        "-i",
        str(media_path),
        "-map",
        f"0:a:{a_pos}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "8000",
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or len(proc.stdout) < 1600:
        return default_res

    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    mean_sq = float(np.mean(samples * samples)) + 1e-12
    lufs = round(max(-60.0, min(-5.0, -0.691 + 10.0 * math.log10(mean_sq))), 2)

    win = 400  # 50ms windows at 8kHz
    n_win = len(samples) // win
    if n_win < 4:
        return {"integrated_lufs": lufs, "snr_db": 20.0, "has_diegetic_music": False}

    framed = samples[: n_win * win].reshape(n_win, win)
    rms_win = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    db_win = 20.0 * np.log10(rms_win)
    p90 = float(np.percentile(db_win, 90))
    p15 = float(np.percentile(db_win, 15))
    snr_db = round(max(6.0, min(45.0, p90 - p15 + 12.0)), 2)

    # Diegetic music heuristic: sustained active energy + rhythmic envelope modulation + multi-band harmonicity
    active_ratio = float(np.mean(db_win > (p90 - 14.0)))
    env_std = float(np.std(db_win))
    spec = np.abs(np.fft.rfft(framed, axis=1)) + 1e-9
    geo_mean = np.exp(np.mean(np.log(spec), axis=1))
    arith_mean = np.mean(spec, axis=1)
    flatness = float(np.mean(geo_mean / arith_mean))
    # Pure unmodulated sine test tones have env_std ~ 0.0; live music/singing has rhythmic envelope variance (1.2..9.5 dB)
    has_music = bool(active_ratio >= 0.72 and 1.2 <= env_std <= 11.0 and flatness < 0.28 and lufs > -34.0)
    return {
        "integrated_lufs": lufs,
        "snr_db": snr_db,
        "has_diegetic_music": has_music,
    }


def probe_audio_file(audio_path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(audio_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            data = json.loads(proc.stdout)
            return round(float(data.get("format", {}).get("duration", 0.0)), 3)
        except Exception:
            pass
    return 0.0


def detect_scenes_and_extract_keyframes(
    video_path: Path,
    duration: float,
    keyframes_dir: Path,
    scene_threshold: float = 0.30,
) -> list[SceneSegment]:
    """Segment a video into scenes using FFmpeg scene detection + fallback intervals, and extract keyframes."""
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # Detect scene change timestamps via FFmpeg showinfo
    cut_times: list[float] = [0.0]
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    use_vt = get_hardware_profile().has_videotoolbox_hwaccel
    hw_prefix = ["-hwaccel", "videotoolbox"] if use_vt else []

    if duration > 3.5:
        sc_tail = [
            "-i",
            str(video_path),
            "-vf",
            f"scale=320:-2:flags=fast_bilinear,select='gt(scene,{scene_threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(["ffmpeg", "-hide_banner", *hw_prefix, *sc_tail], capture_output=True, text=True, check=False)
        if proc.returncode != 0 and hw_prefix:
            proc = subprocess.run(["ffmpeg", "-hide_banner", *sc_tail], capture_output=True, text=True, check=False)
        pts_pattern = re.compile(r"pts_time:([0-9.]+)")
        for line in proc.stderr.splitlines():
            if "pts_time:" in line:
                m = pts_pattern.search(line)
                if m:
                    t = float(m.group(1))
                    if t - cut_times[-1] >= 2.0 and (duration - t) >= 1.5:
                        cut_times.append(round(t, 2))

    # If no cuts were detected and clip is long (>10s), subdivide into ~8s segments for richer annotation
    if len(cut_times) == 1 and duration >= 10.0:
        num_parts = min(4, max(2, int(duration // 8.0)))
        step = duration / num_parts
        cut_times = [round(i * step, 2) for i in range(num_parts)]

    scenes: list[SceneSegment] = []
    for idx, start_t in enumerate(cut_times):
        end_t = cut_times[idx + 1] if idx + 1 < len(cut_times) else round(duration, 2)
        if end_t <= start_t:
            end_t = round(start_t + 1.0, 2)
        mid_t = round((start_t + end_t) / 2.0, 2)
        kf_path = keyframes_dir / f"{stem}_scene{idx + 1:02d}_{int(mid_t * 10):04d}ds.jpg"

        if not kf_path.exists():
            extract_tail = [
                "-ss",
                str(mid_t),
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale='min(1280,iw)':-2",
                "-q:v",
                "3",
                str(kf_path),
            ]
            ex_proc = subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *hw_prefix, *extract_tail],
                capture_output=True,
                check=False,
            )
            if (ex_proc.returncode != 0 or not kf_path.exists()) and hw_prefix:
                subprocess.run(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *extract_tail],
                    capture_output=True,
                    check=False,
                )

        scenes.append(
            SceneSegment(
                scene_id=f"{stem}_s{idx + 1:02d}",
                start_time=start_t,
                end_time=end_t,
                summary=f"Video segment {idx + 1} from {video_path.name} ({start_t:.1f}s - {end_t:.1f}s)",
                visual_tags=[],
                camera_motion="natural",
                emotion="joyful",
                quality_score=8.0,
                keyframe_path=str(kf_path.resolve()) if kf_path.exists() else None,
            )
        )

    return scenes


def _estimate_focal_point(img: Image.Image) -> tuple[float, float]:
    """Estimate normalized (x, y) center of visual saliency using gradient energy."""
    try:
        small = img.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)
        gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
        gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
        energy = gx + gy
        # Weight slightly toward center to avoid edge noise
        yy, xx = np.mgrid[0:64, 0:64]
        center_weight = np.exp(-((xx - 32) ** 2 + (yy - 28) ** 2) / (2 * 22.0**2))
        weighted = energy * center_weight
        total = float(weighted.sum())
        if total > 1e-5:
            cx = float((weighted * xx).sum() / total) / 63.0
            cy = float((weighted * yy).sum() / total) / 63.0
            return (round(min(max(cx, 0.25), 0.75), 3), round(min(max(cy, 0.25), 0.75), 3))
    except Exception:
        pass
    return (0.5, 0.5)


def _normalize_exif_tz_offset(raw_offset: object) -> Optional[str]:
    if raw_offset is None:
        return None
    txt = str(raw_offset).strip()
    if not txt:
        return None
    if txt.upper() == "Z":
        return "Z"
    m_off = re.match(r"^([+-])(\d{2}):?(\d{2})$", txt)
    if m_off:
        return f"{m_off.group(1)}{m_off.group(2)}:{m_off.group(3)}"
    return None


def _extract_exif_datetime(img: Image.Image, photo_path: Optional[Path] = None) -> Optional[str]:
    raw_dt: Optional[str] = None
    raw_offset: object = None
    try:
        exif = img.getexif()
        if exif:
            exif_ifd = None
            try:
                exif_ifd = exif.get_ifd(0x8769)
            except Exception:
                exif_ifd = None

            # 1. Check EXIF Sub-IFD (0x8769) for DateTimeOriginal (36867) or DateTimeDigitized (36868)
            if exif_ifd:
                if exif_ifd.get(36867):
                    raw_dt = str(exif_ifd.get(36867))
                    raw_offset = exif_ifd.get(36881) or exif_ifd.get(36880) or exif.get(36881) or exif.get(36880)
                elif exif_ifd.get(36868):
                    raw_dt = str(exif_ifd.get(36868))
                    raw_offset = exif_ifd.get(36882) or exif_ifd.get(36880) or exif.get(36882) or exif.get(36880)

            # 2. Fallback to top-level DateTimeOriginal (36867), DateTimeDigitized (36868), or DateTime (306)
            if not raw_dt:
                for tag_id, off_tags in ((36867, (36881, 36880)), (36868, (36882, 36880)), (306, (36880,))):
                    if exif.get(tag_id):
                        raw_dt = str(exif.get(tag_id))
                        for o_tag in off_tags:
                            raw_offset = (exif_ifd.get(o_tag) if exif_ifd else None) or exif.get(o_tag)
                            if raw_offset:
                                break
                        break
    except Exception:
        pass

    if raw_dt:
        cleaned = str(raw_dt).strip()
        tail_tz = _normalize_exif_tz_offset(cleaned[19:]) if len(cleaned) > 19 else None
        # Normalize EXIF "YYYY:MM:DD HH:MM:SS" -> ISO "YYYY-MM-DDTHH:MM:SS"
        if len(cleaned) >= 19 and cleaned[4] == ":" and cleaned[7] == ":":
            cleaned = f"{cleaned[:4]}-{cleaned[5:7]}-{cleaned[8:10]}T{cleaned[11:19]}"
        tz_suffix = tail_tz or _normalize_exif_tz_offset(raw_offset)
        if tz_suffix and not (cleaned.endswith("Z") or "+" in cleaned[10:] or "-" in cleaned[10:]):
            cleaned = f"{cleaned}{tz_suffix}"
        return cleaned

    if photo_path is not None:
        m = re.search(r"(20\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", photo_path.stem)
        if m:
            y, mo, d, hh, mm, ss = m.groups()
            iso = f"{y}-{mo}-{d}T{hh}:{mm}:{ss}"
            if photo_path.stem.upper().startswith("PXL_"):
                iso = f"{iso}Z"
            return iso
    return None


def _extract_exif_camera_info(img: Image.Image, photo_path: Optional[Path] = None) -> tuple[str, str]:
    """Mechanically extract (camera_make_model, exif_summary) from photo EXIF headers."""
    make_str = ""
    model_str = ""
    exposure_parts: list[str] = []
    try:
        exif = img.getexif()
        if exif:
            make_str = str(exif.get(271) or "").strip()
            model_str = str(exif.get(272) or "").strip()
            try:
                exif_ifd = exif.get_ifd(0x8769)
            except Exception:
                exif_ifd = None
            if exif_ifd:
                fl = exif_ifd.get(37386)
                fn = exif_ifd.get(33434)
                iso = exif_ifd.get(34855)
                if fl:
                    try:
                        exposure_parts.append(f"{float(fl):.0f}mm")
                    except Exception:
                        pass
                if fn:
                    try:
                        exposure_parts.append(f"f/{float(fn):.1f}")
                    except Exception:
                        pass
                if iso:
                    exposure_parts.append(f"ISO {iso}")
    except Exception:
        pass

    if make_str and model_str and model_str.lower().startswith(make_str.lower()):
        combined = model_str
    else:
        combined = f"{make_str} {model_str}".strip()
    cam_badge = combined.upper() if combined else ""

    exif_summary = " · ".join(exposure_parts) if exposure_parts else None
    return cam_badge, exif_summary or ""


def _parse_exif_rational(val: Any) -> Optional[float]:
    """Convert Pillow IFDRational, (num, den) tuple, or float into a float."""
    if val is None:
        return None
    try:
        if isinstance(val, (tuple, list)) and len(val) == 2:
            den = float(val[1])
            return float(val[0]) / den if abs(den) > 1e-9 else None
        return float(val)
    except Exception:
        return None


def _extract_exif_gps_and_alt(img: Image.Image) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Mechanically extract signed (latitude, longitude, altitude_m) from EXIF GPSInfo (0x8825)."""
    try:
        exif = img.getexif()
        if not exif:
            return None, None, None
        gps_ifd = exif.get_ifd(0x8825)
        if not gps_ifd:
            return None, None, None

        def _dms_to_deg(dms: Any, ref: str) -> Optional[float]:
            if not dms or len(dms) < 3:
                return None
            d0 = _parse_exif_rational(dms[0])
            d1 = _parse_exif_rational(dms[1])
            d2 = _parse_exif_rational(dms[2])
            if d0 is None or d1 is None or d2 is None:
                return None
            deg = d0 + d1 / 60.0 + d2 / 3600.0
            if str(ref).upper().strip() in ("S", "W"):
                deg = -deg
            return round(deg, 6)

        lat = _dms_to_deg(gps_ifd.get(2), str(gps_ifd.get(1, "N")))
        lon = _dms_to_deg(gps_ifd.get(4), str(gps_ifd.get(3, "E")))
        alt_val = _parse_exif_rational(gps_ifd.get(6))
        alt_ref = gps_ifd.get(5, 0)
        gps_alt: Optional[float] = None
        if alt_val is not None:
            is_below = alt_ref in (1, "1", b"\x01") or (isinstance(alt_ref, (bytes, bytearray)) and len(alt_ref) > 0 and alt_ref[0] == 1)
            gps_alt = round(-abs(alt_val) if is_below else alt_val, 2)
        if lat is not None and lon is not None:
            return lat, lon, gps_alt
    except Exception:
        pass
    return None, None, None


def _extract_exif_gps(img: Image.Image) -> tuple[Optional[float], Optional[float]]:
    """Mechanically extract signed decimal (latitude, longitude) from EXIF GPSInfo (0x8825)."""
    lat, lon, _ = _extract_exif_gps_and_alt(img)
    return lat, lon


def compute_image_color_stats(img: Image.Image) -> dict[str, float]:
    """Mechanically compute RGB channel means, luminance percentiles (p05, p50, p95), and estimated Kelvin."""
    try:
        small = img.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)
        mean_r = float(arr[:, :, 0].mean())
        mean_g = float(arr[:, :, 1].mean())
        mean_b = float(arr[:, :, 2].mean())
        luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        p05, p50, p95 = np.percentile(luma, [5, 50, 95])
        # Approximate correlated color temperature from R/B ratio around 5500K neutral daylight
        rb_ratio = (mean_r + 1.0) / (mean_b + 1.0)
        est_kelvin = max(2800.0, min(9000.0, 5500.0 / (rb_ratio ** 0.65)))
        return {
            "mean_r": round(mean_r, 2),
            "mean_g": round(mean_g, 2),
            "mean_b": round(mean_b, 2),
            "p05_luma": round(float(p05), 2),
            "p50_luma": round(float(p50), 2),
            "p95_luma": round(float(p95), 2),
            "est_kelvin": round(est_kelvin, 0),
        }
    except Exception:
        return {
            "mean_r": 128.0,
            "mean_g": 128.0,
            "mean_b": 128.0,
            "p05_luma": 24.0,
            "p50_luma": 128.0,
            "p95_luma": 232.0,
            "est_kelvin": 5500.0,
        }


def inspect_photo_file(photo_path: Path, previews_dir: Path) -> PhotoAsset:
    """Open a photo (.HEIC, .JPG, .PNG), extract EXIF/dimensions, save RGB preview JPEG, and compute focal point."""
    previews_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(photo_path)
    creation_time = _extract_exif_datetime(img, photo_path)
    camera_make_model, exif_summary = _extract_exif_camera_info(img, photo_path)
    gps_lat, gps_lon, gps_alt = _extract_exif_gps_and_alt(img)
    orig_w, orig_h = img.size
    try:
        exif_orient = img.getexif().get(274, 1)
        if exif_orient in (5, 6, 7, 8):
            orig_w, orig_h = orig_h, orig_w
    except Exception:
        pass

    # Use DCT draft mode on large JPEGs for 15x faster decoding while preserving full-res path for 4K rendering
    try:
        if max(img.size) > 1920:
            img.draft("RGB", (1280, 1280))
    except Exception:
        pass

    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = orig_w, orig_h
    color_stats = compute_image_color_stats(img)

    if w > h * 1.05:
        orientation = "landscape"
    elif h > w * 1.05:
        orientation = "portrait"
    else:
        orientation = "square"

    preview_path = previews_dir / f"{photo_path.stem}.jpg"
    if not preview_path.exists():
        prev = img.copy()
        prev.thumbnail((1280, 1280), Image.Resampling.BILINEAR)
        prev.save(preview_path, "JPEG", quality=88)

    focal = _estimate_focal_point(img)
    rec_effect = "gallery_ken_burns" if orientation == "portrait" else "zoom_in"

    return PhotoAsset(
        id=f"img_{photo_path.stem}",
        path=str(photo_path.resolve()),
        preview_path=str(preview_path.resolve()),
        filename=photo_path.name,
        width=w,
        height=h,
        orientation=orientation,
        creation_time=creation_time,
        camera_make_model=camera_make_model,
        exif_summary=exif_summary or None,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        gps_alt=gps_alt,
        color_stats=color_stats,
        caption=f"Photo {photo_path.name} ({orientation}, {w}x{h})",
        subjects=[],
        mood="warm",
        focal_point=focal,
        recommended_effect=rec_effect,
        quality_score=8.0,
    )


def generate_contact_sheets(
    catalog: MediaCatalog,
    output_dir: Path,
    cols: int = 4,
    rows: int = 4,
    cell_w: int = 480,
    cell_h: int = 270,
    max_photo_samples: int = 48,
) -> list[str]:
    """Generate labeled contact sheet grids combining video keyframes and photo previews for rapid AI/user inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, Path]] = []

    for v in catalog.videos:
        for sc in v.scenes:
            if sc.keyframe_path and Path(sc.keyframe_path).exists():
                label = f"{v.filename} [{sc.start_time:.1f}-{sc.end_time:.1f}s]"
                items.append((label, Path(sc.keyframe_path)))

    photos_to_show = catalog.photos
    if len(photos_to_show) > max_photo_samples and max_photo_samples > 0:
        step = len(photos_to_show) / max_photo_samples
        photos_to_show = [photos_to_show[int(i * step)] for i in range(max_photo_samples)]

    for p in photos_to_show:
        img_p = Path(p.preview_path) if p.preview_path and Path(p.preview_path).exists() else Path(p.path)
        if img_p.exists():
            label = f"{p.filename} ({p.orientation[0].upper()})"
            items.append((label, img_p))

    if not items:
        return []

    per_sheet = cols * rows
    font = _get_cjk_font(20)
    sheet_paths: list[str] = []

    for sheet_idx in range(0, len(items), per_sheet):
        batch = items[sheet_idx : sheet_idx + per_sheet]
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (18, 18, 24))
        draw = ImageDraw.Draw(sheet)

        for i, (label, img_path) in enumerate(batch):
            r = i // cols
            c = i % cols
            x0, y0 = c * cell_w, r * cell_h
            try:
                im = Image.open(img_path).convert("RGB")
                im.thumbnail((cell_w - 8, cell_h - 36), Image.Resampling.LANCZOS)
                px = x0 + (cell_w - im.width) // 2
                py = y0 + 4 + (cell_h - 36 - im.height) // 2
                sheet.paste(im, (px, py))
            except Exception:
                pass

            # Label bar at bottom of cell
            draw.rectangle([x0, y0 + cell_h - 30, x0 + cell_w, y0 + cell_h], fill=(10, 10, 15))
            draw.text((x0 + 8, y0 + cell_h - 27), label, fill=(255, 220, 80), font=font)
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(55, 60, 75), width=1)

        out_file = output_dir / f"contact_sheet_{sheet_idx // per_sheet:02d}.jpg"
        sheet.save(out_file, "JPEG", quality=88)
        sheet_paths.append(str(out_file.resolve()))

    return sheet_paths


def iter_media_files_recursive(media_root: Path) -> list[Path]:
    """Recursively scan media_root for media files, ignoring cache/hidden directories."""
    ignored_dirs = {"cache", "render_cache", ".git", "__pycache__", "node_modules"}
    seen: set[Path] = set()
    result: list[Path] = []
    if not media_root.exists():
        return result
    for p in media_root.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(media_root).parts[:-1]
        if any(part.lower() in ignored_dirs or part.startswith(".") for part in rel_parts):
            continue
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(p)
    result.sort(key=lambda p: p.name.lower())
    return result


def build_initial_catalog(
    media_dir: str | Path,
    cache_dir: str | Path,
    extra_audio_paths: Optional[list[str | Path]] = None,
    dit_manifest: Optional[DITIngestManifest] = None,
) -> MediaCatalog:
    """Scan media_dir (recursively) for videos, photos, and audio; extract keyframes, previews, and contact sheets."""
    media_root = Path(media_dir).resolve()
    cache_root = Path(cache_dir).resolve()
    keyframes_dir = cache_root / "keyframes"
    previews_dir = cache_root / "photos"
    sheets_dir = cache_root / "contact_sheets"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    if dit_manifest is None:
        manifest_candidate = cache_root.parent / "dit_manifest.json"
        if manifest_candidate.exists():
            try:
                loaded_manifest = DITIngestManifest.from_json_file(manifest_candidate)
                if Path(loaded_manifest.source_directory).resolve() == media_root:
                    dit_manifest = loaded_manifest
            except Exception:
                dit_manifest = None

    dit_by_path: dict[str, DITIngestEntry] = {}
    if dit_manifest is not None:
        for ent in dit_manifest.entries:
            try:
                dit_by_path[str(Path(ent.source_path).resolve())] = ent
            except Exception:
                pass

    files = iter_media_files_recursive(media_root)
    folder_filenames: dict[Path, set[str]] = {}
    for p in files:
        folder_filenames.setdefault(p.parent, set()).add(p.name)

    def _build_one(item: tuple[int, Path]) -> Optional[tuple[int, str, object]]:
        idx, f = item
        ext = f.suffix.lower()
        if ext not in VIDEO_EXTENSIONS and ext not in PHOTO_EXTENSIONS and ext not in AUDIO_EXTENSIONS:
            return None
        if ext in VIDEO_EXTENSIONS and is_live_photo_video(f, sibling_filenames=folder_filenames.get(f.parent)):
            return None

        resolved_str = str(f.resolve())
        dit_ent = dit_by_path.get(resolved_str)
        is_fresh = False
        if dit_ent is not None:
            try:
                st = f.stat()
                is_fresh = (
                    dit_ent.file_size_bytes == st.st_size
                    and dit_ent.mtime_ns == int(st.st_mtime_ns)
                )
            except OSError:
                is_fresh = False

        if ext in VIDEO_EXTENSIONS:
            if is_fresh and dit_ent is not None and dit_ent.media_type == "video":
                duration = float(dit_ent.duration)
                width = int(dit_ent.width)
                height = int(dit_ent.height)
                fps = float(dit_ent.fps)
                codec = dit_ent.codec
                color_primaries = dit_ent.color_primaries
                color_transfer = dit_ent.color_trc
                color_space = dit_ent.color_space
                pix_fmt = dit_ent.pix_fmt
                has_audio = bool(dit_ent.has_audio)
                f_hash = dit_ent.file_hash
                integrated_lufs = float(dit_ent.integrated_lufs)
                snr_db = float(dit_ent.snr_db)
                has_diegetic_music = bool(dit_ent.has_diegetic_music)
                creation_time = dit_ent.creation_time_raw
                gps_lat = dit_ent.gps_lat
                gps_lon = dit_ent.gps_lon
                gps_alt = dit_ent.gps_alt
            else:
                info = probe_video_file(f)
                _, _, f_hash = compute_file_fingerprint(f)
                aud_sig = probe_audio_signal_metrics(f) if info["has_audio"] else {
                    "integrated_lufs": -60.0,
                    "snr_db": 20.0,
                    "has_diegetic_music": False,
                }
                duration = float(info["duration"])
                width = int(info["width"])
                height = int(info["height"])
                fps = float(info["fps"])
                codec = str(info["codec"])
                color_primaries = str(info["color_primaries"])
                color_transfer = str(info["color_transfer"])
                color_space = str(info["color_space"])
                pix_fmt = str(info["pix_fmt"])
                has_audio = bool(info["has_audio"])
                integrated_lufs = float(aud_sig["integrated_lufs"])
                snr_db = float(aud_sig["snr_db"])
                has_diegetic_music = bool(aud_sig["has_diegetic_music"])
                creation_time = info["creation_time"]
                gps_lat = info["gps_lat"]
                gps_lon = info["gps_lon"]
                gps_alt = info.get("gps_alt")

            scenes = detect_scenes_and_extract_keyframes(f, duration, keyframes_dir)
            v_color_stats: dict[str, float] = {}
            for sc in scenes:
                if sc.keyframe_path and Path(sc.keyframe_path).exists():
                    try:
                        with Image.open(sc.keyframe_path) as kf_im:
                            v_color_stats = compute_image_color_stats(kf_im)
                        break
                    except Exception:
                        pass
            return (
                idx,
                "video",
                VideoAsset(
                    id=f"vid_{f.stem}",
                    path=resolved_str,
                    filename=f.name,
                    duration=duration,
                    width=width,
                    height=height,
                    fps=fps,
                    codec=codec,
                    color_primaries=color_primaries,
                    color_transfer=color_transfer,
                    color_space=color_space,
                    pix_fmt=pix_fmt,
                    has_audio=has_audio,
                    integrated_lufs=integrated_lufs,
                    snr_db=snr_db,
                    has_diegetic_music=has_diegetic_music,
                    file_hash=f_hash,
                    creation_time=creation_time,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                    gps_alt=gps_alt,
                    color_stats=v_color_stats,
                    overall_summary=f"Video clip {f.name} ({duration:.1f}s, {width}x{height})",
                    scenes=scenes,
                    dialogues=[],
                ),
            )
        elif ext in PHOTO_EXTENSIONS:
            if (
                is_fresh
                and dit_ent is not None
                and dit_ent.media_type == "photo"
                and dit_ent.preview_path
                and Path(dit_ent.preview_path).exists()
            ):
                w = int(dit_ent.width)
                h = int(dit_ent.height)
                orient = "square" if abs(w - h) <= max(w, h, 1) * 0.05 else ("landscape" if w >= h else "portrait")
                return (
                    idx,
                    "photo",
                    PhotoAsset(
                        id=dit_ent.asset_id,
                        path=resolved_str,
                        preview_path=dit_ent.preview_path,
                        filename=f.name,
                        width=w,
                        height=h,
                        orientation=orient,
                        creation_time=dit_ent.creation_time_raw,
                        camera_make_model=dit_ent.camera_make_model,
                        exif_summary=dit_ent.exif_summary,
                        gps_lat=dit_ent.gps_lat,
                        gps_lon=dit_ent.gps_lon,
                        gps_alt=dit_ent.gps_alt,
                        color_stats=dict(dit_ent.color_stats),
                        focal_point=dit_ent.focal_point,
                    ),
                )
            return idx, "photo", inspect_photo_file(f, previews_dir)
        else:
            if is_fresh and dit_ent is not None and dit_ent.media_type == "audio":
                dur = float(dit_ent.duration)
            else:
                dur = probe_audio_file(f)
            return (
                idx,
                "audio",
                AudioAsset(
                    id=f"aud_{f.stem}",
                    path=resolved_str,
                    filename=f.name,
                    duration=dur,
                    role="music",
                    description=f"Audio track {f.name} ({dur:.1f}s)",
                ),
            )

    videos: list[VideoAsset] = []
    photos: list[PhotoAsset] = []
    audio_files: list[AudioAsset] = []

    if files:
        indexed_files = list(enumerate(files))
        workers = max(1, min(get_io_worker_count(8), len(indexed_files)))
        if workers == 1:
            built_rows = [_build_one(it) for it in indexed_files]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                built_rows = list(pool.map(_build_one, indexed_files))
        valid_rows = sorted([r for r in built_rows if r is not None], key=lambda r: r[0])
        for _, kind, asset in valid_rows:
            if kind == "video":
                videos.append(asset)  # type: ignore[arg-type]
            elif kind == "photo":
                photos.append(asset)  # type: ignore[arg-type]
            elif kind == "audio":
                audio_files.append(asset)  # type: ignore[arg-type]

    # Also check parent directory or extra_audio_paths for BGM files if none in media_dir
    if extra_audio_paths:
        for ap in extra_audio_paths:
            ap_path = Path(ap).resolve()
            if ap_path.exists() and ap_path.is_file():
                dur = probe_audio_file(ap_path)
                audio_files.append(
                    AudioAsset(
                        id=f"aud_{ap_path.stem}",
                        path=str(ap_path),
                        filename=ap_path.name,
                        duration=dur,
                        role="music",
                        description=f"Background music {ap_path.name} ({dur:.1f}s)",
                    )
                )

    catalog = MediaCatalog(
        source_directory=str(media_root),
        created_at=datetime.now(timezone.utc).isoformat(),
        videos=videos,
        photos=photos,
        audio_files=audio_files,
    )
    catalog.contact_sheet_paths = generate_contact_sheets(catalog, sheets_dir)
    return catalog
