"""Confirmation-gated YouTube direct publisher (Data API v3 OAuth PKCE Resumable Upload & YouTube Studio Browser)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Literal, Optional
from urllib.parse import parse_qs, urlparse
import webbrowser

import httpx
from PIL import Image, ImageDraw, ImageOps

from ..analyzer.media_probe import _get_cjk_font, probe_video_file
from ..models.schema import (
    VALID_YOUTUBE_PRIVACY,
    EditTimeline,
    YouTubeChapterEntry,
    YouTubePublishDraft,
    YouTubePublishReceipt,
)
from .google_photos import (
    GooglePhotosAuthManager,
    get_default_google_photos_config_dir,
    try_copy_to_clipboard,
)

YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_CAPTIONS_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
DEFAULT_YOUTUBE_API_BASE = "https://www.googleapis.com"
YOUTUBE_STUDIO_URL = "https://studio.youtube.com"
RESUMABLE_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB (multiple of 256 KB per YouTube Resumable Upload spec)
MAX_THUMBNAIL_BYTES = 1_950_000


def get_default_youtube_config_dir() -> Path:
    """Return per-user OS config directory outside the git repo for YouTube OAuth secrets/tokens."""
    env_dir = os.environ.get("YOUTUBE_CONFIG_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return (base / "video_editing_ai" / "youtube").resolve()


class YouTubeAuthManager(GooglePhotosAuthManager):
    """Installed-App OAuth 2.0 PKCE token manager for YouTube Data API v3."""

    def __init__(
        self,
        client_secrets_path: Optional[str | Path] = None,
        token_path: Optional[str | Path] = None,
        include_captions_scope: bool = False,
        scope: Optional[str] = None,
    ) -> None:
        cfg_dir = get_default_youtube_config_dir()
        env_secrets = os.environ.get("YOUTUBE_CLIENT_SECRETS", "").strip()
        env_token = os.environ.get("YOUTUBE_TOKEN_PATH", "").strip()

        resolved_secrets = Path(
            client_secrets_path or env_secrets or (cfg_dir / "client_secret.json")
        ).expanduser()
        if not client_secrets_path and not env_secrets and not resolved_secrets.exists():
            shared_secrets = get_default_google_photos_config_dir() / "client_secret.json"
            if shared_secrets.is_file():
                resolved_secrets = shared_secrets

        resolved_token = Path(token_path or env_token or (cfg_dir / "token.json")).expanduser()
        resolved_scope = scope or (
            f"{YOUTUBE_UPLOAD_SCOPE} {YOUTUBE_CAPTIONS_SCOPE}"
            if include_captions_scope
            else YOUTUBE_UPLOAD_SCOPE
        )
        self.client_secrets_path = resolved_secrets
        self.token_path = resolved_token
        self.scope = resolved_scope

    def format_setup_instructions(self) -> str:
        return (
            "YouTube Data API v3 OAuth 2.0 setup (optional for CLI/headless automation):\n"
            "  1. Open Google Cloud Console -> APIs & Services -> Library -> enable 'YouTube Data API v3'.\n"
            "  2. Configure OAuth Consent Screen (External -> Testing -> add your Google account under Test Users).\n"
            "  3. Create Credentials -> OAuth Client ID -> Application type: 'Desktop app' -> Download JSON.\n"
            f"  4. Save that JSON to: {self.client_secrets_path}\n"
            "     (or set YOUTUBE_CLIENT_SECRETS / pass --client-secrets <path>).\n"
            "  5. Run: ai-editor youtube-auth\n"
            "Alternative with ZERO Google Cloud setup: approve the prepared draft and upload via YouTube Studio Browser mode."
        )

    def get_valid_access_token(self) -> str:
        """Return a valid YouTube OAuth access token, refreshing automatically when near expiry."""
        try:
            return super().get_valid_access_token()
        except RuntimeError as exc:
            msg = str(exc).replace("Google Photos", "YouTube").replace("google-photos-auth", "youtube-auth")
            raise RuntimeError(msg) from exc


def resolve_youtube_auth_mode(
    requested_mode: str = "auto",
    auth_manager: Optional[YouTubeAuthManager] = None,
) -> Literal["browser", "oauth"]:
    """Resolve explicit or automatic YouTube publish authentication mode."""
    mode = (requested_mode or "auto").strip().lower()
    if mode not in ("auto", "browser", "oauth"):
        raise ValueError(f"Unsupported YouTube auth_mode '{requested_mode}'. Expected auto, browser, or oauth.")
    if mode == "browser":
        return "browser"
    if mode == "oauth":
        return "oauth"
    mgr = auth_manager or YouTubeAuthManager()
    status = mgr.get_auth_status()
    return "oauth" if status.get("ready") else "browser"


def sanitize_youtube_title(raw: str) -> str:
    """Clean forbidden angle brackets and clamp title to YouTube's 100-character limit."""
    cleaned = str(raw or "").replace("<", "").replace(">", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = "Untitled Video Story"
    if len(cleaned) <= 100:
        return cleaned
    return cleaned[:97].rstrip() + "..."


def sanitize_youtube_description(raw: str) -> str:
    """Replace angle brackets forbidden by YouTube Data API v3 and clamp to 5000 characters."""
    text = str(raw or "").replace("\r\n", "\n")
    text = text.replace("->", "\u2192").replace("<-", "\u2190")
    text = text.replace("<", "(").replace(">", ")").strip()
    if len(text) <= 5000:
        return text
    return text[:4995].rstrip() + "\n..."


def sanitize_youtube_tags(raw_tags: list[str] | str | None) -> list[str]:
    """Normalize tags, remove forbidden characters, deduplicate, and enforce cumulative char limits."""
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        parts = [p.strip() for p in re.split(r"[,;\n]+", raw_tags) if p.strip()]
    else:
        parts = [str(p).strip() for p in raw_tags if str(p).strip()]

    out: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    for tag in parts:
        cleaned = re.sub(r"\s+", " ", tag.replace("<", "").replace(">", "")).strip()
        if not cleaned or len(cleaned) > 60:
            cleaned = cleaned[:60].strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        cost = len(cleaned) + (2 if "," in cleaned else 0) + (1 if out else 0)
        if total_chars + cost > 450:
            break
        seen.add(key)
        out.append(cleaned)
        total_chars += cost
    return out


def format_youtube_timecode(sec: float) -> str:
    """Format seconds as `MM:SS` or `H:MM:SS` (always `00:00` at start)."""
    total = max(0, int(round(float(sec))))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def build_youtube_chapters_from_timeline(
    timeline: Optional[EditTimeline],
    paper_edit: Optional[dict[str, Any]] = None,
) -> list[YouTubeChapterEntry]:
    """
    Build YouTube-compliant chapter markers from timeline.yaml markers / section titles.

    Enforces YouTube's three chapter rules:
      1) Starts at `00:00`,
      2) At least 10 seconds between consecutive chapter timestamps,
      3) At least 3 chapters overall.
    """
    if timeline is None or timeline.total_duration < 30.0:
        return []

    raw_points: list[tuple[float, str]] = []
    for mk in timeline.markers:
        label = sanitize_youtube_title(mk.name)[:72]
        if label:
            raw_points.append((max(0.0, float(mk.time_sec)), label))

    if len(raw_points) < 3:
        for ov in timeline.title_track:
            if ov.style in ("magazine_chapter_plate", "act_banner", "opening_card"):
                label = sanitize_youtube_title(ov.text)[:72]
                if label:
                    raw_points.append((max(0.0, float(ov.timeline_start)), label))

    if len(raw_points) < 3:
        seen_acts: set[str] = set()
        for v_item in timeline.visual_track:
            act_name = str(v_item.act or "").strip()
            if act_name and act_name not in seen_acts:
                seen_acts.add(act_name)
                raw_points.append((max(0.0, float(v_item.timeline_start)), sanitize_youtube_title(act_name)[:72]))

    if len(raw_points) < 3 and isinstance(paper_edit, dict):
        chapters = paper_edit.get("chapters") or []
        if isinstance(chapters, list):
            for ch in chapters:
                if isinstance(ch, dict) and ch.get("start_sec") is not None and ch.get("title"):
                    raw_points.append((max(0.0, float(ch["start_sec"])), sanitize_youtube_title(str(ch["title"]))[:72]))

    if not raw_points:
        return []

    raw_points.sort(key=lambda pair: pair[0])
    # Snap early opening marker to 00:00 or prepend Opening Highlights if first marker >= 10s
    normalized: list[tuple[float, str]] = []
    if raw_points[0][0] >= 10.0:
        normalized.append((0.0, "Opening Highlights"))
    else:
        normalized.append((0.0, raw_points[0][1]))
        raw_points = raw_points[1:]

    for ts, label in raw_points:
        prev_ts = normalized[-1][0]
        if ts - prev_ts >= 10.0:
            if ts < timeline.total_duration - 2.0:
                normalized.append((ts, label))

    if len(normalized) < 3:
        return []

    return [
        YouTubeChapterEntry(
            timestamp_sec=round(ts, 2),
            timecode=format_youtube_timecode(ts),
            title=title,
        )
        for ts, title in normalized
    ]


def _read_json_dict(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def synthesize_youtube_metadata(
    project_dir: Path,
    timeline: Optional[EditTimeline],
    video_probe: dict[str, Any],
    title_override: Optional[str] = None,
    description_override: Optional[str] = None,
    tags_override: Optional[list[str] | str] = None,
    category_override: Optional[str] = None,
) -> dict[str, Any]:
    """Generate clean YouTube Title, Description (with 00:00 Chapters), Tags, and Category."""
    brief = _read_json_dict(project_dir / "creative_brief.json") or {}
    paper_edit = _read_json_dict(project_dir / "paper_edit.json") or {}
    travel_map = _read_json_dict(project_dir / "travel_map_manifest.json") or {}
    proj_meta = _read_json_dict(project_dir / "project_meta.json") or {}

    chapters = build_youtube_chapters_from_timeline(timeline, paper_edit=paper_edit)

    raw_title = (
        title_override
        or (timeline.project_name if timeline and timeline.project_name else "")
        or str(brief.get("story_title") or brief.get("project_title") or "")
        or str(proj_meta.get("name") or "")
        or project_dir.name.replace("_project", "").replace("_", " ").title()
    )
    title = sanitize_youtube_title(raw_title)

    waypoints = [
        wp.get("label")
        for wp in (travel_map.get("waypoints") or [])
        if isinstance(wp, dict) and wp.get("label")
    ]
    has_multi_stop_route = len(waypoints) >= 2
    category_id = str(category_override or ("19" if has_multi_stop_route else "22")).strip()

    if description_override is not None and description_override.strip():
        description = sanitize_youtube_description(description_override)
    else:
        desc_parts: list[str] = []
        summary = (
            (timeline.story_arc_summary if timeline else "")
            or str(paper_edit.get("logline") or "")
            or str(brief.get("story_theme") or brief.get("user_prompt") or "")
        ).strip()
        if summary:
            desc_parts.append(summary)
        else:
            desc_parts.append(f"{title} — crafted highlights film.")

        if waypoints:
            route_str = " -> ".join(str(w) for w in waypoints)
            total_km = float(travel_map.get("total_distance_km") or 0.0)
            dist_note = f" ({total_km:.0f} km total)" if total_km > 0 else ""
            desc_parts.append(f"Route{dist_note}: {route_str}")

        if chapters:
            chap_lines = ["Chapters:"] + [f"{ch.timecode} {ch.title}" for ch in chapters]
            desc_parts.append("\n".join(chap_lines))

        w = int(video_probe.get("width") or 1920)
        h = int(video_probe.get("height") or 1080)
        trc = str(video_probe.get("color_transfer") or "").lower()
        is_hdr = ("b67" in trc) or ("2084" in trc) or ("hlg" in trc)
        fmt_tag = "4K UHD" if w >= 3840 else ("2.5K QHD" if w >= 2560 else f"{w}x{h}")
        hdr_tag = " · 10-bit HLG HDR" if is_hdr else ""
        desc_parts.append(f"Master Format: {fmt_tag}{hdr_tag}")
        description = sanitize_youtube_description("\n\n".join(desc_parts))

    if tags_override is not None:
        tags = sanitize_youtube_tags(tags_override)
    else:
        auto_tags: list[str] = []
        for w in waypoints:
            auto_tags.append(str(w))
        if brief.get("narrative_style"):
            auto_tags.append(str(brief["narrative_style"]).replace("_", " "))
        w = int(video_probe.get("width") or 0)
        trc = str(video_probe.get("color_transfer") or "").lower()
        if w >= 3840:
            auto_tags.append("4K")
        if "b67" in trc or "2084" in trc:
            auto_tags.append("HDR")
        auto_tags.extend(["travel film", "family highlights"])
        tags = sanitize_youtube_tags(auto_tags)

    return {
        "title": title,
        "description": description,
        "tags": tags,
        "category_id": category_id,
        "chapters": chapters,
    }


def _pick_thumbnail_source(
    project_dir: Path,
    timeline: Optional[EditTimeline],
    video_path: Path,
    explicit_source: Optional[str | Path] = None,
) -> Optional[Path]:
    if explicit_source:
        p = Path(explicit_source).expanduser()
        if p.is_file():
            return p.resolve()

    curation = _read_json_dict(project_dir / "photo_curation.json")
    if curation and isinstance(curation.get("items"), list):
        candidates: list[tuple[float, Path]] = []
        for it in curation["items"]:
            if not isinstance(it, dict):
                continue
            src = Path(str(it.get("source_path") or ""))
            if not src.is_file():
                continue
            score = float(it.get("overall_taste_score") or 7.5)
            if it.get("is_selected"):
                score += 5.0
            if it.get("is_favorite"):
                score += 2.5
            if it.get("orientation") == "landscape":
                score += 1.5
            candidates.append((score, src))
        if candidates:
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            return candidates[0][1].resolve()

    if timeline is not None:
        for v_item in timeline.visual_track:
            if v_item.type == "photo":
                src = Path(v_item.source)
                if src.is_file():
                    return src.resolve()
    return None


def _extract_video_frame_rgb(video_path: Path, timestamp_sec: float, is_hdr: bool) -> Image.Image:
    """Extract a representative still frame from video_path as an RGB PIL Image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_png = Path(tf.name)
    try:
        vf_parts = ["scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720"]
        if is_hdr:
            # Ensure clean full-range sRGB transfer for SDR thumbnail rendering
            vf_parts.append("format=rgb24")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timestamp_sec):.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            ",".join(vf_parts),
            str(tmp_png),
        ]
        subprocess.run(cmd, check=True, timeout=30)
        with Image.open(tmp_png) as im:
            return im.convert("RGB")
    finally:
        if tmp_png.exists():
            try:
                tmp_png.unlink()
            except Exception:
                pass


def render_youtube_thumbnail(
    project_dir: Path,
    timeline: Optional[EditTimeline],
    video_path: Path,
    title: str,
    subtitle: str = "",
    is_hdr: bool = False,
    duration_sec: float = 0.0,
    explicit_source: Optional[str | Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Render a crisp 1280x720 YouTube custom thumbnail strictly under 2 MB."""
    out_jpg = (output_path or (project_dir / "youtube_thumbnail.jpg")).resolve()
    out_jpg.parent.mkdir(parents=True, exist_ok=True)

    src_photo = _pick_thumbnail_source(project_dir, timeline, video_path, explicit_source=explicit_source)
    if src_photo is not None:
        try:
            with Image.open(src_photo) as raw_im:
                base_rgb = ImageOps.exif_transpose(raw_im).convert("RGB")
            canvas = ImageOps.fit(base_rgb, (1280, 720), method=Image.Resampling.LANCZOS)
        except Exception:
            src_photo = None

    if src_photo is None:
        try:
            ts = max(0.2, min(duration_sec * 0.25 if duration_sec > 2.0 else 0.5, max(0.2, duration_sec - 0.3)))
            canvas = _extract_video_frame_rgb(video_path, ts, is_hdr=is_hdr)
        except Exception:
            canvas = Image.new("RGB", (1280, 720), (18, 26, 40))

    # Subtle bottom-left cinema scrim for crisp title legibility without obscuring hero center
    overlay = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay, "RGBA")
    for y in range(380, 720):
        frac = (y - 380) / 340.0
        alpha = int(round(195 * (frac**1.4)))
        odraw.line([(0, y), (1280, y)], fill=(8, 12, 20, alpha))

    # Warm gold accent bar + typography
    odraw.rectangle([58, 536, 146, 542], fill=(242, 186, 86, 245))
    title_font = _get_cjk_font(54)
    sub_font = _get_cjk_font(25)

    display_title = title.strip()[:52] + ("..." if len(title.strip()) > 52 else "")
    odraw.text((60, 558), display_title, font=title_font, fill=(0, 0, 0, 175))
    odraw.text((58, 556), display_title, font=title_font, fill=(248, 250, 252, 255))

    meta_bits: list[str] = []
    if subtitle.strip():
        meta_bits.append(subtitle.strip()[:64])
    if is_hdr:
        meta_bits.append("4K HDR")
    if meta_bits:
        sub_line = "  |  ".join(meta_bits)
        odraw.text((58, 626), sub_line, font=sub_font, fill=(220, 228, 240, 240))

    composed = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    for qual in (92, 86, 80, 74, 68):
        composed.save(out_jpg, format="JPEG", quality=qual, optimize=True)
        if out_jpg.stat().st_size <= MAX_THUMBNAIL_BYTES:
            break

    return out_jpg


def compute_youtube_confirmation_token(
    *,
    video_path: str | Path,
    video_size_bytes: int,
    video_mtime_ns: int,
    title: str,
    description: str,
    privacy_status: str,
    self_declared_made_for_kids: bool,
    category_id: str,
    tags: list[str],
) -> str:
    """Compute a deterministic SHA-256 confirmation token binding user consent to exact upload parameters."""
    canonical = {
        "category_id": str(category_id),
        "description": str(description),
        "privacy_status": str(privacy_status),
        "self_declared_made_for_kids": bool(self_declared_made_for_kids),
        "tags": [str(t) for t in tags],
        "title": str(title),
        "video_mtime_ns": int(video_mtime_ns),
        "video_path": str(Path(video_path).resolve()),
        "video_size_bytes": int(video_size_bytes),
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_draft_confirmation_token(draft: YouTubePublishDraft) -> str:
    """Recompute the expected SHA-256 confirmation token from a YouTubePublishDraft."""
    return compute_youtube_confirmation_token(
        video_path=draft.video_path,
        video_size_bytes=draft.video_size_bytes,
        video_mtime_ns=draft.video_mtime_ns,
        title=draft.title,
        description=draft.description,
        privacy_status=draft.privacy_status,
        self_declared_made_for_kids=draft.self_declared_made_for_kids,
        category_id=draft.category_id,
        tags=draft.tags,
    )


def verify_publish_confirmation(
    draft: YouTubePublishDraft,
    confirmed: bool,
    confirmation_token: Optional[str],
) -> None:
    """Enforce explicit user confirmation and cryptographic match to the reviewed draft + video file."""
    if not confirmed:
        raise PermissionError("YouTube upload blocked: explicit user confirmation is required.")
    tok = str(confirmation_token or "").strip()
    if not tok:
        raise PermissionError("YouTube upload blocked: missing confirmation_token.")

    vid = Path(draft.video_path)
    if not vid.is_file():
        raise FileNotFoundError(f"Prepared video file no longer exists: {vid}")
    st = vid.stat()
    if int(st.st_size) != int(draft.video_size_bytes) or int(st.st_mtime_ns) != int(draft.video_mtime_ns):
        raise PermissionError(
            "Video file changed after prepare_youtube_publish(); re-prepare and obtain fresh confirmation."
        )

    expected = compute_draft_confirmation_token(draft)
    if not hmac.compare_digest(tok, expected):
        raise PermissionError(
            "Confirmation token does not match the prepared YouTube draft. Re-prepare and confirm again."
        )


def _find_best_srt_path(project_dir: Path, explicit_srt: Optional[str | Path] = None) -> Optional[Path]:
    candidates = [explicit_srt] if explicit_srt else [project_dir / "final_story.srt", project_dir / "subtitles.srt"]
    for cand in candidates:
        if cand is None:
            continue
        p = Path(cand).expanduser()
        if p.is_file() and p.stat().st_size > 8 and p.read_text(encoding="utf-8", errors="ignore").strip():
            return p.resolve()
    return None


def _build_ask_question_payload(draft: YouTubePublishDraft) -> dict[str, Any]:
    size_mb = draft.video_size_bytes / (1024 * 1024)
    hdr_lbl = "10-bit HLG HDR" if draft.is_hdr else "SDR"
    chap_lbl = f"{len(draft.chapters)} chapters" if draft.chapters else "no chapters"
    kids_lbl = "Yes (Made for Kids)" if draft.self_declared_made_for_kids else "No (Not Made for Kids)"
    q_text = (
        f"Ready to publish \"{draft.title}\" ({draft.width}x{draft.height} {hdr_lbl}, {size_mb:.1f} MB, "
        f"visibility: {draft.privacy_status}, audience: {kids_lbl}, {chap_lbl}, mode: {draft.auth_mode_resolved}) to YouTube?"
    )
    return {
        "questions": [
            {
                "question": q_text,
                "is_multi_select": False,
                "options": [
                    f"(Recommended) Confirm & Publish as {draft.privacy_status.upper()}",
                    "Change Visibility or Edit Title / Description First",
                    "Cancel Publish (Keep Local Video Only)",
                ],
            }
        ]
    }


def prepare_youtube_publish(
    project_dir: str | Path,
    video_path: Optional[str | Path] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[list[str] | str] = None,
    privacy_status: str = "unlisted",
    self_declared_made_for_kids: bool = False,
    category_id: Optional[str] = None,
    thumbnail_source: Optional[str | Path] = None,
    captions_srt_path: Optional[str | Path] = None,
    upload_captions: bool = True,
    auth_mode: str = "auto",
    auth_manager: Optional[YouTubeAuthManager] = None,
) -> YouTubePublishDraft:
    """
    Phase 1 (Zero Network Upload): Inspect rendered video, generate metadata + 00:00 chapters + thumbnail,
    compute a bound SHA-256 confirmation token, and save `youtube_publish_draft.json`.
    """
    proj = Path(project_dir).expanduser().resolve()
    vid = Path(video_path).expanduser().resolve() if video_path else (proj / "final_story.mp4").resolve()
    if not vid.is_file():
        raise FileNotFoundError(f"Rendered video not found for YouTube publishing: {vid}")

    priv = (privacy_status or "unlisted").strip().lower()
    if priv not in VALID_YOUTUBE_PRIVACY:
        raise ValueError(f"Unsupported privacy_status '{privacy_status}'. Must be one of {VALID_YOUTUBE_PRIVACY}.")

    tl_path = proj / "timeline.yaml"
    timeline: Optional[EditTimeline] = None
    if tl_path.is_file():
        try:
            timeline = EditTimeline.from_yaml(tl_path)
        except Exception:
            timeline = None

    probe = probe_video_file(vid)
    st = vid.stat()
    trc = str(probe.get("color_transfer") or "").lower()
    is_hdr = ("b67" in trc) or ("2084" in trc) or ("hlg" in trc)

    meta = synthesize_youtube_metadata(
        project_dir=proj,
        timeline=timeline,
        video_probe=probe,
        title_override=title,
        description_override=description,
        tags_override=tags,
        category_override=category_id,
    )

    subtitle_hint = ""
    if meta["chapters"]:
        subtitle_hint = f"{len(meta['chapters'])} Chapters"
    thumb_path = render_youtube_thumbnail(
        project_dir=proj,
        timeline=timeline,
        video_path=vid,
        title=meta["title"],
        subtitle=subtitle_hint,
        is_hdr=is_hdr,
        duration_sec=float(probe.get("duration") or 0.0),
        explicit_source=thumbnail_source,
    )

    srt_found = _find_best_srt_path(proj, captions_srt_path)
    resolved_mode = resolve_youtube_auth_mode(auth_mode, auth_manager=auth_manager)

    warnings: list[str] = []
    if resolved_mode == "oauth" and priv != "private":
        warnings.append(
            "Unverified personal Google Cloud OAuth apps may lock API uploads to 'private' until audited; "
            "switch visibility inside YouTube Studio or use Studio Browser mode for immediate unlisted/public publishing."
        )
    if float(probe.get("duration") or 0.0) > 900.0 and resolved_mode == "browser":
        warnings.append(
            "Videos longer than 15 minutes require phone verification on the destination YouTube channel."
        )

    token = compute_youtube_confirmation_token(
        video_path=vid,
        video_size_bytes=int(st.st_size),
        video_mtime_ns=int(st.st_mtime_ns),
        title=meta["title"],
        description=meta["description"],
        privacy_status=priv,
        self_declared_made_for_kids=bool(self_declared_made_for_kids),
        category_id=meta["category_id"],
        tags=meta["tags"],
    )

    draft = YouTubePublishDraft(
        project_dir=str(proj),
        video_path=str(vid),
        video_size_bytes=int(st.st_size),
        video_mtime_ns=int(st.st_mtime_ns),
        duration_sec=round(float(probe.get("duration") or 0.0), 3),
        width=int(probe.get("width") or 1920),
        height=int(probe.get("height") or 1080),
        fps=round(float(probe.get("fps") or 30.0), 3),
        is_hdr=is_hdr,
        title=meta["title"],
        description=meta["description"],
        tags=meta["tags"],
        category_id=meta["category_id"],
        privacy_status=priv,  # type: ignore[arg-type]
        self_declared_made_for_kids=bool(self_declared_made_for_kids),
        default_language="en",
        chapters=meta["chapters"],
        thumbnail_path=str(thumb_path),
        captions_srt_path=str(srt_found) if srt_found else None,
        upload_captions=bool(upload_captions and srt_found),
        auth_mode_requested=auth_mode,  # type: ignore[arg-type]
        auth_mode_resolved=resolved_mode,
        status="awaiting_confirmation",
        confirmation_token=token,
        created_at=datetime.now(timezone.utc).isoformat(),
        warnings=warnings,
    )
    draft.ask_question_tool_payload = _build_ask_question_payload(draft)
    draft.to_json_file(proj / "youtube_publish_draft.json")
    return draft


class YouTubeResumableUploader:
    """Chunked YouTube Data API v3 Resumable Upload client with 308 & 5xx byte-range recovery."""

    def __init__(
        self,
        access_token: str,
        api_base: Optional[str] = None,
        chunk_size: int = RESUMABLE_CHUNK_SIZE,
        max_retries: int = 3,
        timeout_sec: float = 120.0,
    ) -> None:
        self.access_token = access_token
        base = api_base or os.environ.get("YOUTUBE_API_BASE", "").strip() or DEFAULT_YOUTUBE_API_BASE
        self.api_base = base.rstrip("/")
        self.chunk_size = max(256 * 1024, int(chunk_size))
        self.max_retries = max(1, int(max_retries))
        self.timeout_sec = float(timeout_sec)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def initiate_resumable_session(self, draft: YouTubePublishDraft) -> str:
        url = f"{self.api_base}/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
        body = {
            "snippet": {
                "title": draft.title,
                "description": draft.description,
                "tags": draft.tags,
                "categoryId": draft.category_id,
                "defaultLanguage": draft.default_language,
            },
            "status": {
                "privacyStatus": draft.privacy_status,
                "selfDeclaredMadeForKids": bool(draft.self_declared_made_for_kids),
            },
        }
        headers = {
            **self._auth_headers(),
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(draft.video_size_bytes),
            "X-Upload-Content-Type": "video/mp4",
        }
        resp = httpx.post(url, headers=headers, json=body, timeout=self.timeout_sec)
        if resp.status_code >= 400:
            raise RuntimeError(f"YouTube upload session init failed ({resp.status_code}): {resp.text[:400]}")
        loc = resp.headers.get("Location") or resp.headers.get("location")
        if not loc:
            raise RuntimeError("YouTube resumable init succeeded without a Location upload URI.")
        if loc.startswith("/"):
            return f"{self.api_base}{loc}"
        return loc

    def _query_committed_offset(self, upload_uri: str, total_size: int) -> Optional[int]:
        try:
            resp = httpx.put(
                upload_uri,
                headers={
                    **self._auth_headers(),
                    "Content-Length": "0",
                    "Content-Range": f"bytes */{total_size}",
                },
                timeout=30.0,
            )
            if resp.status_code == 308:
                rng = resp.headers.get("Range") or resp.headers.get("range") or ""
                m = re.search(r"bytes=0-(\d+)", rng)
                if m:
                    return int(m.group(1)) + 1
                return 0
            if resp.status_code in (200, 201):
                return total_size
        except Exception:
            return None
        return None

    def upload_video_file(
        self,
        upload_uri: str,
        video_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, Any]:
        total = int(video_path.stat().st_size)
        offset = 0
        if progress_callback:
            progress_callback(0, total)

        with open(video_path, "rb") as fh:
            while offset < total:
                fh.seek(offset)
                chunk = fh.read(self.chunk_size)
                if not chunk:
                    break
                end_byte = offset + len(chunk) - 1
                chunk_headers = {
                    **self._auth_headers(),
                    "Content-Length": str(len(chunk)),
                    "Content-Type": "video/mp4",
                    "Content-Range": f"bytes {offset}-{end_byte}/{total}",
                }

                last_err: Optional[Exception] = None
                for attempt in range(self.max_retries):
                    try:
                        resp = httpx.put(
                            upload_uri,
                            headers=chunk_headers,
                            content=chunk,
                            timeout=max(self.timeout_sec, 60.0),
                        )
                    except httpx.TransportError as exc:
                        last_err = exc
                        queried = self._query_committed_offset(upload_uri, total)
                        if queried is not None and queried > offset:
                            offset = queried
                            break
                        time.sleep(min(2.0 * (attempt + 1), 6.0))
                        continue

                    if resp.status_code in (200, 201):
                        if progress_callback:
                            progress_callback(total, total)
                        return dict(resp.json())

                    if resp.status_code == 308:
                        rng = resp.headers.get("Range") or resp.headers.get("range") or ""
                        m = re.search(r"bytes=0-(\d+)", rng)
                        offset = (int(m.group(1)) + 1) if m else (end_byte + 1)
                        if progress_callback:
                            progress_callback(min(offset, total), total)
                        break

                    if resp.status_code in (500, 502, 503, 504):
                        last_err = RuntimeError(f"Transient YouTube upload error {resp.status_code}: {resp.text[:300]}")
                        queried = self._query_committed_offset(upload_uri, total)
                        if queried is not None:
                            offset = queried
                            if progress_callback:
                                progress_callback(min(offset, total), total)
                        time.sleep(min(1.5 * (attempt + 1), 5.0))
                        continue

                    raise RuntimeError(f"YouTube upload chunk rejected ({resp.status_code}): {resp.text[:400]}")
                else:
                    raise RuntimeError(f"YouTube resumable upload failed after {self.max_retries} retries: {last_err}")

        raise RuntimeError("YouTube resumable upload ended before receiving final video metadata response.")

    def upload_thumbnail(self, video_id: str, thumbnail_path: Path) -> bool:
        if not thumbnail_path.is_file():
            return False
        url = f"{self.api_base}/upload/youtube/v3/thumbnails/set?videoId={video_id}"
        headers = {
            **self._auth_headers(),
            "Content-Type": "image/jpeg",
        }
        resp = httpx.post(url, headers=headers, content=thumbnail_path.read_bytes(), timeout=60.0)
        if resp.status_code >= 400:
            raise RuntimeError(f"Thumbnail upload failed ({resp.status_code}): {resp.text[:300]}")
        return True

    def upload_captions_srt(
        self,
        video_id: str,
        srt_path: Path,
        language: str = "en",
        name: str = "Subtitles",
    ) -> bool:
        if not srt_path.is_file():
            return False
        init_url = f"{self.api_base}/upload/youtube/v3/captions?uploadType=resumable&part=snippet"
        srt_bytes = srt_path.read_bytes()
        init_resp = httpx.post(
            init_url,
            headers={
                **self._auth_headers(),
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(len(srt_bytes)),
                "X-Upload-Content-Type": "application/octet-stream",
            },
            json={
                "snippet": {
                    "videoId": video_id,
                    "language": language,
                    "name": name,
                    "isDraft": False,
                }
            },
            timeout=45.0,
        )
        if init_resp.status_code >= 400:
            raise RuntimeError(f"Captions init failed ({init_resp.status_code}): {init_resp.text[:300]}")
        cap_uri = init_resp.headers.get("Location") or init_resp.headers.get("location")
        if not cap_uri:
            return False
        if cap_uri.startswith("/"):
            cap_uri = f"{self.api_base}{cap_uri}"
        put_resp = httpx.put(
            cap_uri,
            headers={
                **self._auth_headers(),
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(srt_bytes)),
            },
            content=srt_bytes,
            timeout=60.0,
        )
        if put_resp.status_code >= 400:
            raise RuntimeError(f"Captions upload failed ({put_resp.status_code}): {put_resp.text[:300]}")
        return True


def extract_youtube_video_id(url_or_id: str) -> str:
    """Parse a YouTube watch/short/Studio URL or 11-character video ID."""
    raw = str(url_or_id or "").strip()
    if not raw:
        raise ValueError("Missing YouTube URL or video ID.")
    if raw == "pending_studio_upload":
        return raw
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        return raw
    parsed = urlparse(raw)
    if parsed.hostname and "youtu.be" in parsed.hostname:
        seg = parsed.path.strip("/").split("/")[0]
        if seg:
            return seg
    qs = parse_qs(parsed.query)
    if qs.get("v"):
        return str(qs["v"][0])
    m = re.search(r"/(?:video|shorts|embed)/([A-Za-z0-9_-]{8,16})", parsed.path)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract a valid YouTube video ID from '{raw}'.")


def record_browser_publish_receipt(
    project_dir: str | Path,
    draft: YouTubePublishDraft,
    confirmed: bool,
    confirmation_token: str,
    video_url_or_id: str,
    auth_mode_used: Literal["antigravity_browser", "browser_assisted"] = "antigravity_browser",
    notes: Optional[list[str]] = None,
) -> YouTubePublishReceipt:
    """Validate confirmation token and persist `youtube_publish_receipt.json` after Studio browser publish."""
    verify_publish_confirmation(draft, confirmed=confirmed, confirmation_token=confirmation_token)
    proj = Path(project_dir).expanduser().resolve()
    vid_id = extract_youtube_video_id(video_url_or_id)
    watch_url = f"https://youtu.be/{vid_id}" if vid_id != "pending_studio_upload" else ""
    studio_url = f"{YOUTUBE_STUDIO_URL}/video/{vid_id}/edit" if vid_id != "pending_studio_upload" else YOUTUBE_STUDIO_URL

    receipt = YouTubePublishReceipt(
        video_id=vid_id,
        watch_url=watch_url,
        studio_url=studio_url,
        auth_mode_used=auth_mode_used,
        title=draft.title,
        privacy_status=draft.privacy_status,
        effective_privacy_status=draft.privacy_status,
        self_declared_made_for_kids=draft.self_declared_made_for_kids,
        thumbnail_uploaded=bool(draft.thumbnail_path and vid_id != "pending_studio_upload"),
        captions_uploaded=False,
        video_size_bytes=draft.video_size_bytes,
        confirmation_token=confirmation_token,
        published_at=datetime.now(timezone.utc).isoformat(),
        notes=list(notes or []),
    )
    receipt.to_json_file(proj / "youtube_publish_receipt.json")
    draft.status = "published"
    draft.to_json_file(proj / "youtube_publish_draft.json")
    return receipt


def _format_clipboard_publish_pack(draft: YouTubePublishDraft) -> str:
    tags_line = ", ".join(draft.tags)
    return (
        f"TITLE:\n{draft.title}\n\n"
        f"DESCRIPTION:\n{draft.description}\n\n"
        f"VISIBILITY: {draft.privacy_status}\n"
        f"MADE FOR KIDS: {draft.self_declared_made_for_kids}\n"
        f"TAGS:\n{tags_line}\n"
    )


def execute_youtube_publish(
    draft: YouTubePublishDraft,
    confirmed: bool,
    confirmation_token: str,
    auth_manager: Optional[YouTubeAuthManager] = None,
    api_base: Optional[str] = None,
    chunk_size: int = RESUMABLE_CHUNK_SIZE,
    open_browser: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> YouTubePublishReceipt:
    """
    Phase 2: Strictly verify explicit user confirmation + SHA-256 token, then publish via OAuth API
    or prepare an assisted YouTube Studio browser handoff.
    """
    verify_publish_confirmation(draft, confirmed=confirmed, confirmation_token=confirmation_token)
    proj = Path(draft.project_dir).resolve()
    notes: list[str] = []

    if draft.auth_mode_resolved == "browser":
        pack = _format_clipboard_publish_pack(draft)
        if try_copy_to_clipboard(pack):
            notes.append("Copied Title, Description (with 00:00 chapters), and Tags to system clipboard.")
        if open_browser:
            try:
                webbrowser.open(YOUTUBE_STUDIO_URL, new=2)
                notes.append(f"Opened {YOUTUBE_STUDIO_URL} in default browser.")
            except Exception:
                pass
        notes.append(f"Upload file: {draft.video_path}")
        if draft.thumbnail_path:
            notes.append(f"Custom thumbnail: {draft.thumbnail_path}")
        if progress_callback:
            progress_callback(draft.video_size_bytes, draft.video_size_bytes, "Prepared YouTube Studio handoff")
        return record_browser_publish_receipt(
            project_dir=proj,
            draft=draft,
            confirmed=True,
            confirmation_token=confirmation_token,
            video_url_or_id="pending_studio_upload",
            auth_mode_used="browser_assisted",
            notes=notes,
        )

    mgr = auth_manager or YouTubeAuthManager(include_captions_scope=bool(draft.upload_captions and draft.captions_srt_path))
    st = mgr.get_auth_status()
    if not st["ready"]:
        if not st["client_secrets_configured"]:
            raise FileNotFoundError(mgr.format_setup_instructions())
        if progress_callback:
            progress_callback(0, draft.video_size_bytes, "Completing YouTube OAuth sign-in...")
        mgr.authenticate_interactive(open_browser=open_browser)

    token = mgr.get_valid_access_token()
    uploader = YouTubeResumableUploader(access_token=token, api_base=api_base, chunk_size=chunk_size)
    if progress_callback:
        progress_callback(0, draft.video_size_bytes, "Initiating resumable upload session...")
    upload_uri = uploader.initiate_resumable_session(draft)

    def _on_chunk(sent: int, total: int) -> None:
        if progress_callback:
            progress_callback(sent, total, f"Uploading {sent / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB")

    video_res = uploader.upload_video_file(upload_uri, Path(draft.video_path), progress_callback=_on_chunk)
    vid_id = str(video_res.get("id") or "").strip()
    if not vid_id:
        raise RuntimeError(f"YouTube upload response missing video id: {video_res}")

    effective_privacy = str((video_res.get("status") or {}).get("privacyStatus") or draft.privacy_status)
    thumb_ok = False
    if draft.thumbnail_path:
        try:
            thumb_ok = uploader.upload_thumbnail(vid_id, Path(draft.thumbnail_path))
        except Exception as exc:
            notes.append(f"Custom thumbnail skipped ({exc}).")

    cap_ok = False
    if draft.upload_captions and draft.captions_srt_path:
        try:
            cap_ok = uploader.upload_captions_srt(vid_id, Path(draft.captions_srt_path), language=draft.default_language)
        except Exception as exc:
            notes.append(f"Captions upload skipped ({exc}).")

    receipt = YouTubePublishReceipt(
        video_id=vid_id,
        watch_url=f"https://youtu.be/{vid_id}",
        studio_url=f"{YOUTUBE_STUDIO_URL}/video/{vid_id}/edit",
        auth_mode_used="oauth",
        title=draft.title,
        privacy_status=draft.privacy_status,
        effective_privacy_status=effective_privacy,
        self_declared_made_for_kids=draft.self_declared_made_for_kids,
        thumbnail_uploaded=thumb_ok,
        captions_uploaded=cap_ok,
        video_size_bytes=draft.video_size_bytes,
        confirmation_token=confirmation_token,
        published_at=datetime.now(timezone.utc).isoformat(),
        notes=notes,
    )
    receipt.to_json_file(proj / "youtube_publish_receipt.json")
    draft.status = "published"
    draft.to_json_file(proj / "youtube_publish_draft.json")
    return receipt
