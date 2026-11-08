"""AI Story Director that builds a production-quality timeline.yaml and storyboard.md."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from typing import Optional

from ..models.schema import (
    AudioTrackItem,
    EditTimeline,
    KenBurnsConfig,
    MediaCatalog,
    PhotoAsset,
    SubtitleConfig,
    SubtitleItem,
    TimelineMarker,
    TimelineSettings,
    TitleOverlayItem,
    TransitionConfig,
    VideoAsset,
    VisualTrackItem,
)


def _select_video_trim(video: VideoAsset, max_no_speech_dur: float = 6.5) -> tuple[float, float, str]:
    """Determine optimal (source_in, duration, note) for a video clip, preserving complete spoken sentences."""
    dur = video.duration
    if video.dialogues:
        # Find main speech span so we never cut off spoken sentences or song lyrics
        first_start = max(0.0, video.dialogues[0].start_time - 0.4)
        last_end = min(dur, video.dialogues[-1].end_time + 0.5)
        span = last_end - first_start
        if span <= 15.0:
            clip_dur = round(max(3.5, min(dur - first_start, span)), 2)
            quotes = "; ".join(f'"{d.text}"' for d in video.dialogues[:2])
            return round(first_start, 2), clip_dur, f"Preserve complete spoken dialogue: {quotes}"
        else:
            # Pick the highest-density or opening dialogue window up to ~14s
            best_in = round(first_start, 2)
            clip_dur = round(min(14.0, dur - best_in), 2)
            quotes = "; ".join(f'"{d.text}"' for d in video.dialogues[:2])
            return best_in, clip_dur, f"Highlight dialogue segment: {quotes}"

    # No dialogue: pick the highest-scoring scene segment
    if video.scenes:
        best_scene = max(video.scenes, key=lambda s: s.quality_score)
        sc_dur = best_scene.end_time - best_scene.start_time
        if sc_dur >= 3.0:
            in_pt = round(best_scene.start_time, 2)
            clip_dur = round(min(max_no_speech_dur, sc_dur), 2)
            return in_pt, clip_dur, best_scene.summary

    # Fallback: trim 0.5s head shake if clip > 6s
    in_pt = 1.0 if dur > 7.0 else 0.0
    clip_dur = round(min(max_no_speech_dur, max(2.5, dur - in_pt)), 2)
    return in_pt, clip_dur, video.overall_summary or f"Action highlight from {video.filename}"


def _select_diverse_photos(photos: list[PhotoAsset], target_count: int = 16) -> list[PhotoAsset]:
    """Select high-quality, non-duplicate photos distributed evenly across the event chronology."""
    if not photos:
        return []
    if len(photos) <= target_count:
        return list(photos)

    # Divide photos into chronological buckets and pick the highest quality_score photo in each bucket
    selected: list[PhotoAsset] = []
    n = len(photos)
    for i in range(target_count):
        start_idx = int(round(i * n / target_count))
        end_idx = int(round((i + 1) * n / target_count))
        bucket = photos[start_idx: max(start_idx + 1, end_idx)]
        best = max(bucket, key=lambda p: p.quality_score)
        if best not in selected:
            selected.append(best)
    return selected


def _build_baseline_timeline(
    catalog: MediaCatalog,
    user_prompt: str,
    target_duration: Optional[float] = None,
    width: int = 3840,
    height: int = 2160,
    fps: int = 30,
    hdr: bool = True,
    photo_mode: str = "gallery_frame",
) -> EditTimeline:
    """Construct a structured 4-Act baseline EditTimeline from the analyzed MediaCatalog."""
    project_title = "Family Story Highlights"
    if user_prompt.strip():
        # Derive a clean title from the user prompt
        cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\s_-]", "", user_prompt.strip())
        words = cleaned.split()
        if words:
            project_title = " ".join(words[:6]).title()

    acts = [
        ("Act 1: Arrival & Opening", "Green", "Setting the scene, anticipation, and opening moments"),
        ("Act 2: Adventure & Action", "Yellow", "Dynamic motion, activities, and candid reactions"),
        ("Act 3: Iconic Highlights & Exploration", "Sand", "Signature landmarks, details, and group portraits"),
        ("Act 4: Twilight Magic & Finale", "Purple", "Atmospheric evening highlights, music, and warm closing"),
    ]

    # Calculate how many photos to weave between video clips
    num_videos = len(catalog.videos)
    if target_duration and target_duration > 15:
        est_video_dur = sum(min(10.0, v.duration) for v in catalog.videos)
        remaining = max(9.0, target_duration - est_video_dur)
        target_photos = max(4, min(20, int(remaining // 3.0)))
    else:
        target_photos = min(18, max(6, len(catalog.photos) // 4)) if len(catalog.photos) > 8 else len(catalog.photos)

    chosen_photos = _select_diverse_photos(catalog.photos, target_count=target_photos)

    # Interleave videos and photos in chronological order by filename stem number
    combined_assets: list[tuple[str, VideoAsset | PhotoAsset]] = []
    for v in catalog.videos:
        combined_assets.append(("video", v))
    for p in chosen_photos:
        combined_assets.append(("photo", p))
    combined_assets.sort(key=lambda item: item[1].filename.lower())

    visual_track: list[VisualTrackItem] = []
    title_track: list[TitleOverlayItem] = []
    subtitle_items: list[SubtitleItem] = []
    markers: list[TimelineMarker] = []

    kb_presets = ["floating_gallery", "zoom_in", "pan_left_to_right", "zoom_out", "pan_right_to_left"]
    cursor = 0.0
    total_items = max(1, len(combined_assets))
    current_act_idx = -1

    for idx, (kind, asset) in enumerate(combined_assets):
        act_idx = min(3, int((idx / total_items) * 4))
        act_name, act_color, act_desc = acts[act_idx]

        if act_idx != current_act_idx:
            current_act_idx = act_idx
            markers.append(
                TimelineMarker(
                    time_sec=round(cursor, 2),
                    name=act_name,
                    color=act_color,
                    note=act_desc,
                )
            )
            if act_idx == 0:
                title_track.append(
                    TitleOverlayItem(
                        id="title_opening",
                        style="opening_card",
                        text=project_title,
                        subtitle=user_prompt[:70] if user_prompt else "A Curated Story Film",
                        timeline_start=round(cursor + 0.3, 2),
                        duration=4.2,
                        accent_color="#E8E4DC",
                    )
                )

        clip_id = f"clip_{idx + 1:03d}"
        if kind == "video":
            v_asset: VideoAsset = asset  # type: ignore[assignment]
            src_in, clip_dur, note = _select_video_trim(v_asset)
            src_out = round(src_in + clip_dur, 2)

            visual_track.append(
                VisualTrackItem(
                    id=clip_id,
                    type="video",
                    source=v_asset.path,
                    act=act_name,
                    timeline_start=round(cursor, 2),
                    duration=clip_dur,
                    source_in=src_in,
                    source_out=src_out,
                    speed=1.0,
                    volume=1.0,
                    transition_in=TransitionConfig(type="fade_black", duration=0.35),
                    transition_out=TransitionConfig(type="fade_black", duration=0.35),
                    narrative_note=note,
                )
            )

            # Map overlapping video dialogues onto master timeline subtitles and callouts
            for d_idx, d in enumerate(v_asset.dialogues):
                if d.end_time > src_in and d.start_time < src_out:
                    sub_start = round(cursor + max(0.0, d.start_time - src_in), 2)
                    sub_end = round(cursor + min(clip_dur, d.end_time - src_in), 2)
                    if sub_end - sub_start >= 0.4:
                        subtitle_items.append(
                            SubtitleItem(
                                id=f"sub_{len(subtitle_items) + 1:03d}",
                                start=sub_start,
                                end=sub_end,
                                speaker=d.speaker,
                                text=d.text,
                                secondary_text=d.translation,
                            )
                        )

            cursor = round(cursor + clip_dur, 2)
        else:
            p_asset: PhotoAsset = asset  # type: ignore[assignment]
            photo_dur = 3.5 if idx == len(combined_assets) - 1 else 3.0
            preset = kb_presets[idx % len(kb_presets)]
            fx, fy = p_asset.focal_point

            # Configure smooth Ken Burns / Gallery Frame trajectory toward the photo's focal point
            if photo_mode == "gallery_frame" or p_asset.orientation == "portrait":
                kb = KenBurnsConfig(
                    mode="gallery_frame",
                    preset="floating_gallery" if preset == "floating_gallery" else preset,  # type: ignore[arg-type]
                    start_zoom=1.0,
                    end_zoom=1.08,
                    start_center=(0.5, 0.5),
                    end_center=(fx, fy),
                )
            else:
                kb = KenBurnsConfig(
                    mode="full_bleed",
                    preset=preset,  # type: ignore[arg-type]
                    start_zoom=1.0 if preset != "zoom_out" else 1.18,
                    end_zoom=1.18 if preset != "zoom_out" else 1.0,
                    start_center=(0.5, 0.5),
                    end_center=(fx, fy),
                )

            visual_track.append(
                VisualTrackItem(
                    id=clip_id,
                    type="photo",
                    source=p_asset.path,
                    act=act_name,
                    timeline_start=round(cursor, 2),
                    duration=photo_dur,
                    source_in=0.0,
                    source_out=photo_dur,
                    speed=1.0,
                    volume=0.0,
                    ken_burns=kb,
                    transition_in=TransitionConfig(type="fade_black", duration=0.35),
                    transition_out=TransitionConfig(
                        type="fade_black",
                        duration=0.8 if idx == len(combined_assets) - 1 else 0.35,
                    ),
                    narrative_note=p_asset.caption or f"Photo highlight {p_asset.filename}",
                )
            )
            cursor = round(cursor + photo_dur, 2)

    # Add closing card on the final clip
    if visual_track:
        last_clip = visual_track[-1]
        title_track.append(
            TitleOverlayItem(
                id="title_closing",
                style="closing_card",
                text=f"{project_title} • Timeless Memories",
                subtitle="A Curated Visual Archive",
                timeline_start=round(max(0.0, last_clip.timeline_start), 2),
                duration=last_clip.duration,
                accent_color="#E8E4DC",
            )
        )

    # Attach background music if available in catalog
    audio_tracks: list[AudioTrackItem] = []
    if catalog.audio_files:
        bgm = catalog.audio_files[0]
        audio_tracks.append(
            AudioTrackItem(
                id="bgm_main",
                role="music",
                source=bgm.path,
                timeline_start=0.0,
                duration=round(cursor, 2),
                source_in=0.0,
                volume=0.25,
                duck_under_dialogue=True,
                duck_volume=0.08,
                fade_in=1.2,
                fade_out=3.0,
            )
        )

    return EditTimeline(
        project_name=project_title.replace(" ", "_"),
        user_prompt=user_prompt,
        story_arc_summary=(
            "A 4-act narrative progression combining natural spoken dialogue clips with "
            "floating gallery-framed & Ken Burns animated photos, chapter banners, and auto-ducked BGM."
        ),
        settings=TimelineSettings(width=width, height=height, fps=fps, hdr=hdr),
        markers=markers,
        visual_track=visual_track,
        title_track=title_track,
        audio_tracks=audio_tracks,
        subtitles=SubtitleConfig(
            enabled=True,
            burn_in=True,
            items=subtitle_items,
        ),
    )


async def _refine_timeline_with_antigravity_async(
    catalog: MediaCatalog,
    baseline: EditTimeline,
    user_prompt: str,
) -> EditTimeline:
    """Use the Google Antigravity SDK Agent to refine the story arc, titles, and narrative notes."""
    from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig

    prompt = (
        "You are an award-winning video story director. Given the user's story prompt and our baseline edit plan, "
        "refine the project title, story_arc_summary, marker names/notes, and title_track text/subtitles to deliver "
        "a compelling emotional story.\n\n"
        f"User Prompt: {user_prompt}\n"
        f"Current Markers: {json.dumps([m.model_dump() for m in baseline.markers], ensure_ascii=False)}\n"
        f"Current Title Track: {json.dumps([t.model_dump() for t in baseline.title_track], ensure_ascii=False)}\n"
        f"Subtitles/Dialogues: {json.dumps([s.model_dump() for s in baseline.subtitles.items], ensure_ascii=False)}\n\n"
        "Return ONLY a valid JSON object with:\n"
        "{\n"
        '  "project_name": "...",\n'
        '  "story_arc_summary": "...",\n'
        '  "markers": [{"time_sec": 0.0, "name": "...", "color": "Green", "note": "..."}],\n'
        '  "titles": {"<title_id>": {"text": "...", "subtitle": "..."}}\n'
        "}"
    )

    config = LocalAgentConfig(
        system_instructions="You are a film director and story editor. Output strictly valid JSON.",
        capabilities=CapabilitiesConfig(),
    )

    tokens: list[str] = []
    async with Agent(config) as agent:
        response = await agent.chat(prompt)
        async for tok in response:
            tokens.append(str(tok))

    raw = "".join(tokens)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        data = json.loads(raw[start : end + 1])
        if data.get("project_name"):
            baseline.project_name = str(data["project_name"]).replace(" ", "_")
        if data.get("story_arc_summary"):
            baseline.story_arc_summary = str(data["story_arc_summary"])
        if isinstance(data.get("markers"), list) and len(data["markers"]) == len(baseline.markers):
            for m_obj, m_new in zip(baseline.markers, data["markers"]):
                if isinstance(m_new, dict):
                    m_obj.name = str(m_new.get("name", m_obj.name))
                    m_obj.note = str(m_new.get("note", m_obj.note))
        titles_map = data.get("titles", {})
        if isinstance(titles_map, dict):
            for t_obj in baseline.title_track:
                tu = titles_map.get(t_obj.id)
                if isinstance(tu, dict):
                    if tu.get("text"):
                        t_obj.text = str(tu["text"])
                    if "subtitle" in tu and tu["subtitle"] is not None:
                        t_obj.subtitle = str(tu["subtitle"])

    return baseline


def generate_storyboard_markdown(timeline: EditTimeline) -> str:
    """Render a production-quality human-readable Markdown storyboard from an EditTimeline."""
    lines: list[str] = [
        f"# Production Storyboard: {timeline.project_name.replace('_', ' ')}",
        "",
        f"**User Story Prompt**: {timeline.user_prompt or 'Highlight reel'}",
        f"**Narrative Arc**: {timeline.story_arc_summary}",
        f"**Format**: {timeline.settings.width}x{timeline.settings.height} @ {timeline.settings.fps} fps | "
        f"**Total Duration**: {timeline.total_duration:.1f}s ({len(timeline.visual_track)} visual cuts)",
        "",
        "---",
        "",
        "## 1. Story Acts & Markers",
        "",
        "| Time (s) | Act / Marker | Color | Narrative Purpose |",
        "|---:|:---|:---|:---|",
    ]
    for m in timeline.markers:
        lines.append(f"| `{m.time_sec:05.1f}s` | **{m.name}** | `{m.color}` | {m.note} |")

    lines.extend(
        [
            "",
            "## 2. Visual Timeline (Track V1 — Videos & Photos)",
            "",
            "| Clip ID | Timeline Window | Duration | Type | Source File | Source Trim / Effect | Editorial Rationale |",
            "|:---|:---|---:|:---|:---|:---|:---|",
        ]
    )
    for clip in timeline.visual_track:
        src_name = Path(clip.source).name
        t_win = f"`{clip.timeline_start:05.1f}s - {clip.timeline_start + clip.duration:05.1f}s`"
        if clip.type == "video":
            effect_desc = f"In `{clip.source_in:.1f}s` → Out `{clip.source_out or (clip.source_in + clip.duration):.1f}s`"
        else:
            kb = clip.ken_burns
            effect_desc = (
                f"{kb.mode} (`{kb.preset}`, zoom `{kb.start_zoom:.2f}→{kb.end_zoom:.2f}`)"
                if kb
                else "static"
            )
        lines.append(
            f"| `{clip.id}` | {t_win} | `{clip.duration:.1f}s` | **{clip.type.upper()}** | "
            f"`{src_name}` | {effect_desc} | {clip.narrative_note} |"
        )

    lines.extend(
        [
            "",
            "## 3. On-Screen Titles & Graphic Overlays (Tracks V2/V3)",
            "",
            "| ID | Style | Window | Primary Text | Subtitle |",
            "|:---|:---|:---|:---|:---|",
        ]
    )
    for t in timeline.title_track:
        win = f"`{t.timeline_start:05.1f}s - {t.timeline_start + t.duration:05.1f}s`"
        lines.append(f"| `{t.id}` | `{t.style}` | {win} | **{t.text}** | {t.subtitle or '—'} |")

    lines.extend(
        [
            "",
            "## 4. Audio Tracks & Dialogue Subtitles",
            "",
        ]
    )
    if timeline.audio_tracks:
        for a in timeline.audio_tracks:
            lines.append(
                f"- **Audio Track (`{a.id}`)**: `{Path(a.source).name}` | Role: `{a.role}` | "
                f"Base Volume: `{a.volume:.2f}` | Auto-Duck Under Dialogue: `{a.duck_under_dialogue}` "
                f"(ducked to `{a.duck_volume:.2f}`)"
            )
    else:
        lines.append("- **Audio Track**: Native camera dialogue audio only (no external BGM attached).")

    if timeline.subtitles.items:
        lines.extend(
            [
                "",
                "### Timed Dialogue Subtitles",
                "",
                "| Start | End | Speaker | Dialogue / Lyrics | Translation |",
                "|---:|---:|:---|:---|:---|",
            ]
        )
        for s in timeline.subtitles.items:
            lines.append(
                f"| `{s.start:05.1f}s` | `{s.end:05.1f}s` | {s.speaker or 'Speaker'} | "
                f"{s.text} | {s.secondary_text or '—'} |"
            )

    lines.append("")
    return "\n".join(lines)


def generate_edit_plan(
    catalog: MediaCatalog,
    user_prompt: str,
    project_dir: str | Path,
    target_duration: Optional[float] = None,
    width: int = 3840,
    height: int = 2160,
    fps: int = 30,
    hdr: bool = True,
    photo_mode: str = "gallery_frame",
    use_antigravity: bool = True,
) -> tuple[EditTimeline, Path, Path]:
    """Full Step 2 pipeline: build EditTimeline, refine with Antigravity SDK, and write timeline.yaml + storyboard.md."""
    proj_path = Path(project_dir).resolve()
    proj_path.mkdir(parents=True, exist_ok=True)

    timeline = _build_baseline_timeline(
        catalog=catalog,
        user_prompt=user_prompt,
        target_duration=target_duration,
        width=width,
        height=height,
        fps=fps,
        hdr=hdr,
        photo_mode=photo_mode,
    )

    if use_antigravity:
        try:
            timeline = asyncio.run(
                asyncio.wait_for(
                    _refine_timeline_with_antigravity_async(catalog, timeline, user_prompt),
                    timeout=60.0,
                )
            )
        except Exception:
            pass

    timeline_path = proj_path / "timeline.yaml"
    storyboard_path = proj_path / "storyboard.md"

    timeline.to_yaml(timeline_path)
    storyboard_md = generate_storyboard_markdown(timeline)
    storyboard_path.write_text(storyboard_md, encoding="utf-8")

    return timeline, timeline_path, storyboard_path
