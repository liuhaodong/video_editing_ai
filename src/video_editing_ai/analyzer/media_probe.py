"""Technical media probing, HEIC conversion, keyframe extraction, and contact sheets."""

from __future__ import annotations

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

from ..models.schema import (
    AudioAsset,
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
            "color_space": "bt709",
            "has_audio": False,
            "creation_time": None,
        }

    data = json.loads(proc.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

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
    color_transfer = v_stream.get("color_transfer", "bt709")
    color_space = v_stream.get("color_space", "bt709")
    fmt_tags = fmt.get("tags", {}) or {}
    creation_time = fmt_tags.get("creation_time") or tags.get("creation_time")

    return {
        "duration": round(duration, 3),
        "width": width,
        "height": height,
        "fps": fps,
        "codec": codec,
        "color_transfer": color_transfer,
        "color_space": color_space,
        "has_audio": a_stream is not None,
        "creation_time": creation_time,
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
    if duration > 3.5:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(video_path),
            "-vf",
            f"select='gt(scene,{scene_threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
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
            extract_cmd = [
                "ffmpeg",
                "-y",
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
            subprocess.run(extract_cmd, capture_output=True, check=False)

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


def _extract_exif_datetime(img: Image.Image, photo_path: Optional[Path] = None) -> Optional[str]:
    raw_dt: Optional[str] = None
    try:
        exif = img.getexif()
        if exif:
            # 1. Check EXIF Sub-IFD (0x8769) for DateTimeOriginal (36867) or DateTimeDigitized (36868)
            try:
                exif_ifd = exif.get_ifd(0x8769)
                if exif_ifd:
                    raw_dt = exif_ifd.get(36867) or exif_ifd.get(36868)
            except Exception:
                pass
            # 2. Fallback to top-level DateTimeOriginal (36867) or DateTime (306)
            if not raw_dt:
                for tag_id in (36867, 36868, 306):
                    if exif.get(tag_id):
                        raw_dt = str(exif.get(tag_id))
                        break
    except Exception:
        pass

    if raw_dt:
        cleaned = str(raw_dt).strip()
        # Normalize EXIF "YYYY:MM:DD HH:MM:SS" -> ISO "YYYY-MM-DDTHH:MM:SS"
        if len(cleaned) >= 19 and cleaned[4] == ":" and cleaned[7] == ":":
            cleaned = f"{cleaned[:4]}-{cleaned[5:7]}-{cleaned[8:10]}T{cleaned[11:19]}"
        return cleaned

    if photo_path is not None:
        m = re.search(r"(20\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", photo_path.stem)
        if m:
            y, mo, d, hh, mm, ss = m.groups()
            return f"{y}-{mo}-{d}T{hh}:{mm}:{ss}"
    return None


def inspect_photo_file(photo_path: Path, previews_dir: Path) -> PhotoAsset:
    """Open a photo (.HEIC, .JPG, .PNG), extract EXIF/dimensions, save RGB preview JPEG, and compute focal point."""
    previews_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(photo_path)
    creation_time = _extract_exif_datetime(img, photo_path)
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
) -> MediaCatalog:
    """Scan media_dir (recursively) for videos, photos, and audio; extract keyframes, previews, and contact sheets."""
    media_root = Path(media_dir).resolve()
    cache_root = Path(cache_dir).resolve()
    keyframes_dir = cache_root / "keyframes"
    previews_dir = cache_root / "photos"
    sheets_dir = cache_root / "contact_sheets"

    files = iter_media_files_recursive(media_root)

    videos: list[VideoAsset] = []
    photos: list[PhotoAsset] = []
    audio_files: list[AudioAsset] = []

    for f in files:
        ext = f.suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            info = probe_video_file(f)
            scenes = detect_scenes_and_extract_keyframes(f, info["duration"], keyframes_dir)
            videos.append(
                VideoAsset(
                    id=f"vid_{f.stem}",
                    path=str(f.resolve()),
                    filename=f.name,
                    duration=info["duration"],
                    width=info["width"],
                    height=info["height"],
                    fps=info["fps"],
                    codec=info["codec"],
                    color_transfer=info["color_transfer"],
                    color_space=info["color_space"],
                    has_audio=info["has_audio"],
                    creation_time=info["creation_time"],
                    overall_summary=f"Video clip {f.name} ({info['duration']:.1f}s, {info['width']}x{info['height']})",
                    scenes=scenes,
                    dialogues=[],
                )
            )
        elif ext in PHOTO_EXTENSIONS:
            photos.append(inspect_photo_file(f, previews_dir))
        elif ext in AUDIO_EXTENSIONS:
            dur = probe_audio_file(f)
            audio_files.append(
                AudioAsset(
                    id=f"aud_{f.stem}",
                    path=str(f.resolve()),
                    filename=f.name,
                    duration=dur,
                    role="music",
                    description=f"Audio track {f.name} ({dur:.1f}s)",
                )
            )

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
