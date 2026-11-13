"""Stage 05: Deterministic Digital Imaging Technician (DIT) Ingest Agent (dit_ingest)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Optional

from ..analyzer.media_probe import (
    AUDIO_EXTENSIONS,
    PHOTO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    compute_file_fingerprint,
    inspect_photo_file,
    iter_media_files_recursive,
    probe_audio_file,
    probe_audio_signal_metrics,
    probe_video_file,
)
from ..models.schema import DITIngestEntry, DITIngestManifest
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


def _parse_iso_dt(raw: Optional[str]) -> tuple[Optional[datetime], bool]:
    """Parse an EXIF or QuickTime timestamp; returns (utc_dt, had_explicit_tz)."""
    if not raw:
        return None, False
    txt = str(raw).strip()
    if not txt:
        return None, False
    had_tz = txt.endswith("Z") or ("+" in txt[10:]) or ("-" in txt[10:])
    norm = txt.replace("Z", "+00:00")
    for fmt in (None, "%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            if fmt is None:
                dt = datetime.fromisoformat(norm)
            else:
                dt = datetime.strptime(txt[:19], fmt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc), False
            return dt.astimezone(timezone.utc), True
        except Exception:
            continue
    return None, False


def _compute_multicam_clock_offsets(entries: list[DITIngestEntry]) -> dict[str, float]:
    """
    Estimate per-camera clock drift / timezone offset (seconds) when mixing TZ-aware QuickTime
    video timestamps with naive local EXIF mirrorless camera timestamps.
    """
    tz_aware_epochs: list[float] = []
    cam_naive_epochs: dict[str, list[float]] = {}
    all_cams: set[str] = set()

    for ent in entries:
        all_cams.add(ent.camera_id)
        dt, had_tz = _parse_iso_dt(ent.creation_time_raw)
        if dt is None:
            continue
        ts = dt.timestamp()
        if had_tz:
            tz_aware_epochs.append(ts)
        else:
            cam_naive_epochs.setdefault(ent.camera_id, []).append(ts)

    offsets: dict[str, float] = {cam: 0.0 for cam in sorted(all_cams) or ["primary_cam"]}
    if not tz_aware_epochs or not cam_naive_epochs:
        return offsets

    tz_median = sorted(tz_aware_epochs)[len(tz_aware_epochs) // 2]
    for cam, ts_list in cam_naive_epochs.items():
        if not ts_list:
            continue
        cam_median = sorted(ts_list)[len(ts_list) // 2]
        diff = tz_median - cam_median
        # If naive EXIF differs from UTC QuickTime by ~N hours (1h..14h) within 25m, snap to whole half-hour TZ offset
        nearest_half_hour = round(diff / 1800.0) * 1800.0
        if 3600.0 <= abs(nearest_half_hour) <= 50400.0 and abs(diff - nearest_half_hour) <= 2400.0:
            offsets[cam] = float(nearest_half_hour)
    return offsets


@register_agent
class DITIngestAgent(BaseEditingAgent):
    """Stage 05 Agent: Recursive media discovery, hash+mtime probe caching, clock-drift sync, and color/LUFS probing."""

    role_id = "dit_ingest"
    display_name = "0b. DIT Media Ingest & Clock-Sync Specialist"
    stage_order = 5
    substage_priority = 0
    description = (
        "Deterministic Digital Imaging Technician (DIT) ingest specialist that performs recursive media "
        "discovery, caches technical FFprobe/EXIF metadata by file hash + mtime, normalizes multi-camera "
        "DateTimeOriginal clock drift, and probes HDR/SDR color spaces and audio LUFS/SNR."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json"]
    produces_artifacts = ["dit_manifest.json"]
    skill_reference_file = "01a_dit_ingest_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        cache_dir = blackboard.project_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "dit_ingest_cache.json"
        previews_dir = cache_dir / "photos"

        disk_cache: dict[str, dict[str, Any]] = {}
        if cache_file.exists():
            try:
                raw_c = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(raw_c, dict):
                    disk_cache = raw_c
            except Exception:
                disk_cache = {}

        files = iter_media_files_recursive(blackboard.media_dir)
        entries: list[DITIngestEntry] = []
        cache_hits = 0
        cache_misses = 0

        for idx, f in enumerate(files, start=1):
            ext = f.suffix.lower()
            if ext not in VIDEO_EXTENSIONS and ext not in PHOTO_EXTENSIONS and ext not in AUDIO_EXTENSIONS:
                continue

            resolved_str = str(f.resolve())
            size_b, mtime_ns, f_hash = compute_file_fingerprint(f)
            cache_key = f"{resolved_str}|{size_b}|{mtime_ns}|{f_hash}"

            cached_payload = disk_cache.get(cache_key)
            if isinstance(cached_payload, dict):
                try:
                    ent = DITIngestEntry.model_validate({**cached_payload, "cache_hit": True})
                    entries.append(ent)
                    cache_hits += 1
                    continue
                except Exception:
                    pass

            cache_misses += 1
            if ext in VIDEO_EXTENSIONS:
                v_info = probe_video_file(f)
                aud_sig = (
                    probe_audio_signal_metrics(f)
                    if v_info["has_audio"]
                    else {"integrated_lufs": -60.0, "snr_db": 20.0, "has_diegetic_music": False}
                )
                trc = str(v_info.get("color_transfer") or "bt709").lower()
                prim = str(v_info.get("color_primaries") or "bt709").lower()
                cspace = str(v_info.get("color_space") or "bt709").lower()
                pfmt = str(v_info.get("pix_fmt") or "yuv420p").lower()
                is_hdr = trc in ("arib-std-b67", "smpte2084") or "2020" in prim
                is_log = "log" in trc or any(k in f.stem.lower() for k in ("slog", "vlog", "flog", "clog"))
                cam_id = (v_info.get("camera_make_model") or "video_cam").replace(" ", "_").lower()
                ent = DITIngestEntry(
                    asset_id=f"vid_{f.stem}",
                    source_path=resolved_str,
                    filename=f.name,
                    media_type="video",
                    file_size_bytes=size_b,
                    mtime_ns=mtime_ns,
                    file_hash=f_hash,
                    cache_hit=False,
                    creation_time_raw=v_info.get("creation_time"),
                    normalized_timestamp=v_info.get("creation_time") or f"1970-01-01T00:{idx % 60:02d}:00Z",
                    camera_id=cam_id,
                    duration=v_info["duration"],
                    width=v_info["width"],
                    height=v_info["height"],
                    fps=v_info["fps"],
                    codec=v_info["codec"],
                    color_primaries=prim,
                    color_trc=trc,
                    color_space=cspace,
                    pix_fmt=pfmt,
                    is_hdr=is_hdr,
                    is_log=is_log,
                    has_audio=bool(v_info["has_audio"]),
                    integrated_lufs=float(aud_sig["integrated_lufs"]),
                    snr_db=float(aud_sig["snr_db"]),
                    has_diegetic_music=bool(aud_sig["has_diegetic_music"]),
                    gps_lat=v_info.get("gps_lat"),
                    gps_lon=v_info.get("gps_lon"),
                    camera_make_model=v_info.get("camera_make_model"),
                )
            elif ext in PHOTO_EXTENSIONS:
                ph = inspect_photo_file(f, previews_dir)
                cam_id = (ph.camera_make_model or "photo_cam").replace(" ", "_").lower()
                ent = DITIngestEntry(
                    asset_id=ph.id,
                    source_path=resolved_str,
                    filename=f.name,
                    media_type="photo",
                    file_size_bytes=size_b,
                    mtime_ns=mtime_ns,
                    file_hash=f_hash,
                    cache_hit=False,
                    creation_time_raw=ph.creation_time,
                    normalized_timestamp=ph.creation_time or f"1970-01-01T01:{idx % 60:02d}:00Z",
                    camera_id=cam_id,
                    duration=0.0,
                    width=ph.width,
                    height=ph.height,
                    fps=0.0,
                    codec=ext.lstrip("."),
                    color_primaries="bt709",
                    color_trc="iec61966-2-1",
                    color_space="srgb",
                    pix_fmt="rgb24",
                    is_hdr=False,
                    is_log=False,
                    has_audio=False,
                    integrated_lufs=-60.0,
                    snr_db=30.0,
                    has_diegetic_music=False,
                    gps_lat=ph.gps_lat,
                    gps_lon=ph.gps_lon,
                    camera_make_model=ph.camera_make_model,
                    exif_summary=ph.exif_summary,
                    color_stats=ph.color_stats,
                )
            else:
                dur = probe_audio_file(f)
                aud_sig = probe_audio_signal_metrics(f)
                ent = DITIngestEntry(
                    asset_id=f"aud_{f.stem}",
                    source_path=resolved_str,
                    filename=f.name,
                    media_type="audio",
                    file_size_bytes=size_b,
                    mtime_ns=mtime_ns,
                    file_hash=f_hash,
                    cache_hit=False,
                    creation_time_raw=None,
                    normalized_timestamp=f"1970-01-01T02:{idx % 60:02d}:00Z",
                    camera_id="audio_recorder",
                    duration=dur,
                    width=0,
                    height=0,
                    fps=0.0,
                    codec=ext.lstrip("."),
                    color_primaries="none",
                    color_trc="none",
                    color_space="none",
                    pix_fmt="none",
                    has_audio=True,
                    integrated_lufs=float(aud_sig["integrated_lufs"]),
                    snr_db=float(aud_sig["snr_db"]),
                    has_diegetic_music=True,
                )

            disk_cache[cache_key] = ent.model_dump(mode="json")
            entries.append(ent)

        # Normalize multi-camera DateTimeOriginal clock drift
        clock_offsets = _compute_multicam_clock_offsets(entries)
        for ent in entries:
            offset_s = clock_offsets.get(ent.camera_id, 0.0)
            ent.clock_offset_sec = offset_s
            dt, _ = _parse_iso_dt(ent.creation_time_raw)
            if dt is not None:
                norm_dt = dt + timedelta(seconds=offset_s)
                ent.normalized_timestamp = norm_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            cache_file.write_text(json.dumps(disk_cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        manifest = DITIngestManifest(
            source_directory=str(blackboard.media_dir),
            created_at=datetime.now(timezone.utc).isoformat(),
            cache_hits_count=cache_hits,
            cache_misses_count=cache_misses,
            camera_clock_offsets_sec=clock_offsets,
            entries=entries,
        )
        out_path = blackboard.project_dir / "dit_manifest.json"
        manifest.to_json_file(out_path)
        blackboard.dit_manifest = manifest
        blackboard.set_custom_artifact("dit_manifest.json", manifest.model_dump(mode="json"))

        blackboard.emit_activity(
            self.role_id,
            f"Probed {len(entries)} media files ({cache_hits} cache hits, {cache_misses} fresh probes)",
            detail=f"Camera clock offsets: {clock_offsets}",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="culler",
            message_type="handoff",
            subject="Deterministic DIT ingest & clock-drift normalization complete",
            payload={
                "entries_count": len(entries),
                "cache_hits_count": cache_hits,
                "cache_misses_count": cache_misses,
            },
        )
        return {
            "dit_manifest_path": str(out_path),
            "entries_count": len(entries),
            "cache_hits_count": cache_hits,
            "cache_misses_count": cache_misses,
        }
