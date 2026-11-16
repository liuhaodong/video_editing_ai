"""Detection and rejection filters for low-resolution Live Photo and Motion Photo video streams."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Iterable, Optional

LIVE_PHOTO_STILL_EXTENSIONS = {".jpg", ".jpeg", ".heic", ".heif", ".dng"}
LIVE_PHOTO_VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v"}
_EXTRACTED_DIR_NAMES = {"motion_photos", "live_photos"}


def _probe_duration_fast(video_path: Path) -> Optional[float]:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            dur = json.loads(proc.stdout).get("format", {}).get("duration")
            if dur is not None:
                return float(dur)
    except Exception:
        pass
    return None


def _base_stem_variants(name_lower: str, stem_lower: str) -> set[str]:
    """Return candidate normalized photo stems corresponding to a photo or video filename."""
    variants = {stem_lower}
    if stem_lower.endswith(".mp"):
        variants.add(stem_lower[:-3])
    if stem_lower.startswith("mp_"):
        variants.add(stem_lower[3:])
    if stem_lower.startswith("mvimg_"):
        variants.add(stem_lower[6:])
    return {v for v in variants if v}


def is_live_photo_video(
    video_path: str | Path,
    sibling_filenames: Optional[Iterable[str]] = None,
    duration_sec: Optional[float] = None,
) -> bool:
    """
    Return True when a video file is an extracted Google/Samsung Motion Photo video stream
    or a short Apple Live Photo companion MOV/MP4 paired with a still photo.

    Panoramic video renders baked from static PANO stills (``pano_*.mp4``) are explicitly
    allowed because they derive from full-resolution still panoramas rather than pre-shutter
    ring-buffer video streams.
    """
    p = Path(video_path)
    name_lower = p.name.lower()
    stem_lower = p.stem.lower()
    ext_lower = p.suffix.lower()

    # Keep static panorama camera-move renders (e.g. pano_etoile_boulevards_4k.mp4)
    if name_lower.startswith("pano_") and ext_lower == ".mp4":
        return False

    # 1. Extracted Motion Photo / MicroVideo artifacts by naming convention or cache folder
    if ext_lower in LIVE_PHOTO_VIDEO_EXTENSIONS:
        if (
            name_lower.startswith("mp_")
            or name_lower.startswith("mvimg_")
            or stem_lower.endswith(".mp")
            or any(part.lower() in _EXTRACTED_DIR_NAMES for part in p.parts[:-1])
        ):
            return True

    if ext_lower not in LIVE_PHOTO_VIDEO_EXTENSIONS:
        return False

    # 2. Live Photo companion MOV/MP4 sharing the exact stem with a sibling HEIC/JPG photo
    if sibling_filenames is None:
        try:
            sibs = [child.name for child in p.parent.iterdir() if child.is_file()] if p.parent.exists() else []
        except OSError:
            sibs = []
    else:
        sibs = list(sibling_filenames)

    video_stems = _base_stem_variants(name_lower, stem_lower)
    has_paired_still = False
    for sib_name in sibs:
        if sib_name.lower() == name_lower:
            continue
        sp = Path(sib_name)
        if sp.suffix.lower() not in LIVE_PHOTO_STILL_EXTENSIONS:
            continue
        still_stems = _base_stem_variants(sp.name.lower(), sp.stem.lower())
        if video_stems & still_stems:
            has_paired_still = True
            break

    if not has_paired_still:
        return False

    if duration_sec is not None:
        return float(duration_sec) <= 4.0

    probed_dur = _probe_duration_fast(p)
    if probed_dur is not None:
        return probed_dur <= 4.0
    # Paired same-stem still exists and duration couldn't be queried: fail safe and block
    return True
