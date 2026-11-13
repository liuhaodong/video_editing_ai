"""Verification suite for Intelligent Photo Curation, Multi-Moment Video Culling & Dramaturgical Storyline Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pytest

from video_editing_ai.agents import (
    BaseEditingAgent,
    ClientIntakeAgent,
    CullerAgent,
    PhotoCuratorAgent,
    PipelineBlackboard,
    StoryProducerAgent,
)
from video_editing_ai.analyzer.semantic_intelligence import (
    SemanticEmbeddingEngine,
    analyze_human_and_aesthetic_perception,
    cie_lab_harmony_score,
    compute_zoom_tolerant_visual_similarity,
)
from video_editing_ai.analyzer.video_moment_analyzer import (
    analyze_video_temporal_telemetry,
    extract_candidate_video_moments,
    is_operator_filler_utterance,
)
from video_editing_ai.models.schema import (
    CreativeBrief,
    CulledSubClip,
    DialogueSegment,
    MediaCatalog,
    MediaIndex,
    PhotoAsset,
    VideoAsset,
)


def _draw_portrait_photo(
    path: Path,
    *,
    eyes_open: bool = True,
    smiling: bool = True,
    bg_rgb: tuple[int, int, int] = (72, 54, 40),
    shirt_rgb: tuple[int, int, int] = (168, 92, 52),
    bokeh_bg: bool = False,
) -> Path:
    """Create a realistic synthetic portrait photo with open or closed eyes and warm/cool palette."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 900, 1200
    bg = Image.new("RGB", (w, h), bg_rgb)
    bg_draw = ImageDraw.Draw(bg)
    for gx in range(40, w, 90):
        for gy in range(40, h, 90):
            bg_draw.rectangle(
                (gx, gy, gx + 42, gy + 42),
                outline=(min(255, bg_rgb[0] + 38), min(255, bg_rgb[1] + 34), min(255, bg_rgb[2] + 30)),
                width=3,
            )
    if bokeh_bg:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=9.0))

    draw = ImageDraw.Draw(bg)
    draw.pieslice((180, 680, 720, 1340), 180, 360, fill=shirt_rgb)
    skin = (228, 182, 152)
    draw.ellipse((280, 210, 620, 640), fill=skin)

    # Crisp collar/hair texture inside subject ROI so portrait subject stays razor-sharp
    draw.arc((275, 200, 625, 540), 185, 355, fill=(48, 34, 26), width=18)
    for hx in range(320, 590, 24):
        draw.line((hx, 215, hx - 8, 265), fill=(58, 40, 30), width=3)

    if eyes_open:
        for ex in (380, 520):
            draw.ellipse((ex - 34, 350, ex + 34, 392), fill=(250, 250, 252), outline=(45, 32, 28), width=3)
            draw.ellipse((ex - 15, 354, ex + 15, 388), fill=(22, 18, 16))
            draw.ellipse((ex - 5, 360, ex + 3, 368), fill=(255, 255, 255))
    else:
        for ex in (380, 520):
            draw.line((ex - 32, 372, ex + 32, 372), fill=(204, 158, 130), width=3)

    if smiling:
        draw.pieslice((385, 488, 515, 565), 0, 180, fill=(248, 246, 244), outline=(155, 72, 68), width=4)
    else:
        draw.line((410, 530, 490, 530), fill=(168, 108, 96), width=4)

    bg.save(path, "JPEG", quality=94)
    return path


def _draw_textured_scene_photo(
    path: Path,
    *,
    size: tuple[int, int] = (1200, 900),
    base_rgb: tuple[int, int, int] = (64, 88, 118),
    accent_rgb: tuple[int, int, int] = (218, 174, 96),
    seed: int = 7,
) -> Image.Image:
    """Create a richly textured landscape/architectural photo for zoom-crop & MMR tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = size
    img = Image.new("RGB", (w, h), base_rgb)
    draw = ImageDraw.Draw(img)
    for y in range(0, h // 2, 24):
        shade = (
            min(255, base_rgb[0] + y // 12),
            min(255, base_rgb[1] + y // 10),
            min(255, base_rgb[2] + y // 9),
        )
        draw.rectangle((0, y, w, y + 24), fill=shade)
    draw.rectangle((140, h // 3, w - 140, h - 90), fill=accent_rgb, outline=(38, 34, 30), width=6)
    for col_x in range(210, w - 200, 130):
        draw.rectangle((col_x, h // 3 + 45, col_x + 65, h - 130), fill=(48, 56, 68), outline=(245, 232, 205), width=4)
    for _ in range(35):
        rx = int(rng.integers(200, w - 220))
        ry = int(rng.integers(h // 3 + 30, h - 140))
        rw = int(rng.integers(18, 48))
        draw.ellipse((rx, ry, rx + rw, ry + rw), outline=(92, 62, 36), width=3)
    img.save(path, "JPEG", quality=94)
    return img


def _make_shaky_tail_video(path: Path) -> Path:
    """
    Synthesize a 14.0s 320x180 20fps MP4 with:
    - 0.0-11.75s: crisp, smooth steady/gently-panning textured courtyard scene
    - 11.75-14.0s: violent oscillating camera-drop shake + motion blur at the tail
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = 20
    duration = 14.0
    total_frames = int(fps * duration)
    w, h = 320, 180

    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas[:, :] = (58, 86, 118)
    for gy in range(20, 340, 36):
        for gx in range(20, 620, 44):
            color = (210, 185, 125) if ((gx + gy) // 40) % 2 == 0 else (40, 52, 68)
            cv2.rectangle(canvas, (gx, gy), (gx + 26, gy + 22), color, -1)
            cv2.rectangle(canvas, (gx, gy), (gx + 26, gy + 22), (242, 240, 235), 1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    assert writer.isOpened(), "OpenCV VideoWriter failed to open synthetic MP4"
    try:
        for f_idx in range(total_frames):
            t = f_idx / float(fps)
            if t < 7.0:
                ox = 120 + int(round(2.0 * np.sin(t * 0.8)))
                oy = 90
                crop = canvas[oy : oy + h, ox : ox + w].copy()
            elif t < 11.75:
                pan_progress = (t - 7.0) / 4.75
                ox = 120 + int(round(pan_progress * 140.0))
                oy = 90
                crop = canvas[oy : oy + h, ox : ox + w].copy()
            else:
                # Every 4fps sample step (every 5 frames) jumps violently & suffers severe motion blur
                step_k = f_idx // 5
                sign = 1 if (step_k % 2 == 0) else -1
                ox = int(np.clip(180 + sign * 85, 0, 640 - w))
                oy = int(np.clip(90 - sign * 60, 0, 360 - h))
                crop = cv2.GaussianBlur(canvas[oy : oy + h, ox : ox + w], (19, 19), 7.0)
            writer.write(crop)
    finally:
        writer.release()
    return path


# ---------------------------------------------------------------------------
# Gate 1: Open-Eye vs. Closed-Eye Blink Rejection, Bokeh Preservation & Zoom Dedup
# ---------------------------------------------------------------------------
def test_photo_blink_rejection_and_non_adjacent_zoom_dedup(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir(parents=True, exist_ok=True)

    open_eye_path = _draw_portrait_photo(media_dir / "IMG_0101_portrait.jpg", eyes_open=True, smiling=True, bokeh_bg=True)
    blink_path = _draw_portrait_photo(media_dir / "IMG_0102_portrait.jpg", eyes_open=False, smiling=False, bokeh_bg=True)

    m_open = analyze_human_and_aesthetic_perception(open_eye_path, open_eye_path.name)
    m_blink = analyze_human_and_aesthetic_perception(blink_path, blink_path.name)
    assert len(m_open["face_boxes"]) >= 1
    assert m_open["eye_open_score"] > m_blink["eye_open_score"] + 0.30
    assert m_blink["eye_open_score"] < 0.45
    assert m_open["blur_type"] in {"intentional_bokeh", "crisp_all"}
    assert m_open["overall_taste_score"] > m_blink["overall_taste_score"] + 0.6

    wide_img = _draw_textured_scene_photo(media_dir / "IMG_0103_courtyard_wide.jpg", seed=11)
    _draw_textured_scene_photo(
        media_dir / "IMG_0104_garden_statue.jpg",
        base_rgb=(36, 98, 58),
        accent_rgb=(196, 84, 98),
        seed=99,
    )
    cw, ch = wide_img.size
    zoom_crop = wide_img.crop((int(cw * 0.17), int(ch * 0.17), int(cw * 0.83), int(ch * 0.83))).resize(
        (cw, ch), Image.Resampling.BICUBIC
    )
    zoom_path = media_dir / "IMG_0105_courtyard_zoom.jpg"
    zoom_crop.save(zoom_path, "JPEG", quality=94)

    m_wide = analyze_human_and_aesthetic_perception(media_dir / "IMG_0103_courtyard_wide.jpg", "IMG_0103_courtyard_wide.jpg")
    m_zoom = analyze_human_and_aesthetic_perception(zoom_path, zoom_path.name)
    zoom_sim = compute_zoom_tolerant_visual_similarity(
        m_wide["visual_signature"],
        m_wide["center_signature"],
        m_wide["inner_signature"],
        m_zoom["visual_signature"],
        m_zoom["center_signature"],
        m_zoom["inner_signature"],
    )
    assert zoom_sim >= 0.945

    photos = [
        ("p_open", open_eye_path, "2026-06-15T10:00:00", "portrait"),
        ("p_blink", blink_path, "2026-06-15T10:00:02", "portrait"),
        ("p_wide", media_dir / "IMG_0103_courtyard_wide.jpg", "2026-06-15T10:01:00", "landscape"),
        ("p_other", media_dir / "IMG_0104_garden_statue.jpg", "2026-06-15T10:01:20", "landscape"),
        ("p_zoom", zoom_path, "2026-06-15T10:01:45", "landscape"),
    ]
    subclips = [
        CulledSubClip(
            subclip_id=pid,
            source_asset_id=pid,
            source_path=str(pp),
            filename=pp.name,
            media_type="photo",
            roll_type="still",
            normalized_timestamp=ts,
            golden_in=0.0,
            golden_out=3.2,
            usable_in=0.0,
            usable_out=8.0,
            action_peak_tc=1.6,
            duration=3.2,
            orientation=orient,  # type: ignore[arg-type]
            shot_scale="closeup" if orient == "portrait" else "wide",
            summary=f"Photo {pp.name}",
            emotion_score=8.2,
            aesthetic_score=8.2,
        )
        for pid, pp, ts, orient in photos
    ]

    bb = PipelineBlackboard(
        media_dir=media_dir,
        project_dir=proj_dir,
        user_prompt="Warm family portraits and courtyard",
        use_antigravity=False,
    )
    bb.creative_brief = CreativeBrief(project_title="Test_Trip", target_duration_sec=45.0)
    bb.media_index = MediaIndex(source_directory=str(media_dir), created_at="2026-06-15T10:05:00", subclips=subclips)
    bb.catalog = MediaCatalog(
        source_directory=str(media_dir),
        created_at="2026-06-15T10:05:00",
        photos=[
            PhotoAsset(
                id=pid,
                path=str(pp),
                filename=pp.name,
                width=900 if orient == "portrait" else 1200,
                height=1200 if orient == "portrait" else 900,
                creation_time=ts,
                orientation=orient,  # type: ignore[arg-type]
                preview_path=str(pp),
            )
            for pid, pp, ts, orient in photos
        ],
    )

    PhotoCuratorAgent().run(bb)
    report = bb.photo_curation
    assert report is not None
    by_id = {it.subclip_id: it for it in report.items}

    # 1. Open-eye portrait wins burst cluster over closed-eye portrait
    assert by_id["p_open"].cluster_id == by_id["p_blink"].cluster_id
    assert by_id["p_open"].is_selected is True
    assert by_id["p_blink"].is_selected is False
    assert "burst" in (by_id["p_blink"].rejection_reason or "")

    # 2. Non-adjacent 1.5x zoom crop is grouped into the same burst cluster as p_wide!
    assert by_id["p_wide"].cluster_id == by_id["p_zoom"].cluster_id
    assert sum(1 for k in ("p_wide", "p_zoom") if by_id[k].is_selected) == 1
    assert by_id["p_other"].is_selected is True


# ---------------------------------------------------------------------------
# Gate 2: Submodular MMR Portfolio Diversity & CIE-Lab Chromatic Diptych Pairing
# ---------------------------------------------------------------------------
def test_submodular_mmr_diversity_and_chromatic_diptych_pairing(tmp_path: Path) -> None:
    media_dir = tmp_path / "media_mmr"
    proj_dir = tmp_path / "proj_mmr"
    proj_dir.mkdir(parents=True, exist_ok=True)

    subclips: list[CulledSubClip] = []
    photo_assets: list[PhotoAsset] = []

    for i in range(18):
        p = media_dir / f"IMG_20{i:02d}_arch.jpg"
        _draw_textured_scene_photo(
            p,
            base_rgb=(66 + (i % 4) * 6, 82 + (i % 3) * 6, 108),
            accent_rgb=(124 + (i * 5) % 24, 134, 146),
            seed=200 + i * 31,
        )
        with Image.open(p) as im_edit:
            d_edit = ImageDraw.Draw(im_edit)
            d_edit.polygon(
                [(60 + i * 35, 290), (210 + i * 35, 95 + (i % 4) * 35), (360 + i * 35, 290)],
                fill=(92, 104, 122),
                outline=(230, 234, 240),
            )
            im_edit.save(p, "JPEG", quality=94)
        ts = f"2026-07-01T{8 + i // 3:02d}:{(i % 3) * 20:02d}:00"
        subclips.append(
            CulledSubClip(
                subclip_id=f"arch_{i}",
                source_asset_id=f"arch_{i}",
                source_path=str(p),
                filename=p.name,
                media_type="photo",
                roll_type="still",
                normalized_timestamp=ts,
                golden_in=0.0,
                golden_out=3.0,
                usable_in=0.0,
                usable_out=8.0,
                action_peak_tc=1.5,
                duration=3.0,
                orientation="landscape",
                shot_scale="wide",
                summary="Stone facade wing",
                emotion_score=8.0,
                aesthetic_score=8.0,
            )
        )
        photo_assets.append(
            PhotoAsset(
                id=f"arch_{i}",
                path=str(p),
                filename=p.name,
                width=1200,
                height=900,
                creation_time=ts,
                orientation="landscape",
                preview_path=str(p),
            )
        )

    warm_a = _draw_portrait_photo(
        media_dir / "IMG_3001_warm_portrait_a.jpg",
        eyes_open=True,
        smiling=True,
        bg_rgb=(142, 96, 52),
        shirt_rgb=(192, 118, 56),
    )
    warm_b = _draw_portrait_photo(
        media_dir / "IMG_3002_warm_portrait_b.jpg",
        eyes_open=True,
        smiling=True,
        bg_rgb=(156, 108, 58),
        shirt_rgb=(148, 76, 48),
    )
    cold_cyan = _draw_portrait_photo(
        media_dir / "IMG_3003_cold_portrait_c.jpg",
        eyes_open=True,
        smiling=True,
        bg_rgb=(12, 168, 212),
        shirt_rgb=(18, 135, 198),
    )
    vivid_art = media_dir / "IMG_3004_sculpture_detail.jpg"
    _draw_textured_scene_photo(vivid_art, base_rgb=(168, 48, 36), accent_rgb=(244, 198, 78), seed=777)
    with Image.open(vivid_art) as im_art:
        d_art = ImageDraw.Draw(im_art)
        for r_rad in range(60, 260, 28):
            d_art.ellipse(
                (600 - r_rad, 450 - r_rad, 600 + r_rad, 450 + r_rad),
                outline=(252, 236, 182),
                width=5,
            )
        im_art.save(vivid_art, "JPEG", quality=95)

    specials = [
        ("warm_a", warm_a, "2026-07-01T14:05:00", "portrait", "closeup"),
        ("warm_b", warm_b, "2026-07-01T14:25:00", "portrait", "medium"),
        ("cold_c", cold_cyan, "2026-07-01T14:45:00", "portrait", "closeup"),
        ("vivid_art", vivid_art, "2026-07-01T15:05:00", "landscape", "closeup"),
    ]
    for sid, sp, sts, sorient, sscale in specials:
        subclips.append(
            CulledSubClip(
                subclip_id=sid,
                source_asset_id=sid,
                source_path=str(sp),
                filename=sp.name,
                media_type="photo",
                roll_type="still",
                normalized_timestamp=sts,
                golden_in=0.0,
                golden_out=3.2,
                usable_in=0.0,
                usable_out=8.0,
                action_peak_tc=1.6,
                duration=3.2,
                orientation=sorient,  # type: ignore[arg-type]
                shot_scale=sscale,  # type: ignore[arg-type]
                summary=f"Special study {sp.name}",
                emotion_score=8.4,
                aesthetic_score=8.4,
            )
        )
        photo_assets.append(
            PhotoAsset(
                id=sid,
                path=str(sp),
                filename=sp.name,
                width=900 if sorient == "portrait" else 1200,
                height=1200 if sorient == "portrait" else 900,
                creation_time=sts,
                orientation=sorient,  # type: ignore[arg-type]
                preview_path=str(sp),
            )
        )

    ma = analyze_human_and_aesthetic_perception(warm_a, warm_a.name)
    mb = analyze_human_and_aesthetic_perception(warm_b, warm_b.name)
    mc = analyze_human_and_aesthetic_perception(cold_cyan, cold_cyan.name)
    assert cie_lab_harmony_score(ma["lab_mean"], mb["lab_mean"]) > cie_lab_harmony_score(ma["lab_mean"], mc["lab_mean"]) + 0.20

    bb = PipelineBlackboard(
        media_dir=media_dir,
        project_dir=proj_dir,
        user_prompt="Portraits and gallery sculpture details",
        use_antigravity=False,
    )
    bb.creative_brief = CreativeBrief(
        project_title="Gallery_Day",
        target_duration_sec=30.0,
        photo_mode="parallax_gallery",
        enabled_photo_modes=["parallax_gallery", "diptych_frame", "gallery_frame"],
    )
    bb.media_index = MediaIndex(source_directory=str(media_dir), created_at="2026-07-01T16:00:00", subclips=subclips)
    bb.catalog = MediaCatalog(source_directory=str(media_dir), created_at="2026-07-01T16:00:00", photos=photo_assets)

    PhotoCuratorAgent().run(bb)
    report = bb.photo_curation
    assert report is not None
    selected = [it for it in report.items if it.is_selected]
    selected_ids = {it.subclip_id for it in selected}

    assert "vivid_art" in selected_ids
    assert all(it.mmr_diversity_score > 0.0 for it in selected)

    by_id = {it.subclip_id: it for it in report.items}
    assert by_id["warm_a"].paired_with_subclip_id == "warm_b" or by_id["warm_b"].paired_with_subclip_id == "warm_a"


# ---------------------------------------------------------------------------
# Gate 3: Multi-Moment Video Splitting, Utterance Hygiene & Optical-Flow Shake Trim
# ---------------------------------------------------------------------------
def test_video_optical_flow_shake_trim_and_multi_moment_splitting(tmp_path: Path) -> None:
    assert is_operator_filler_utterance("Wait, is it recording?") is True
    assert is_operator_filler_utterance("One two three") is True
    assert is_operator_filler_utterance("Hold on") is True
    assert is_operator_filler_utterance("Look at the sunlit stone arches across the river!") is False

    vid_path = _make_shaky_tail_video(tmp_path / "media_vid" / "CLIP_001.mp4")
    telemetry = analyze_video_temporal_telemetry(
        vid_path,
        duration=14.0,
        cache_dir=tmp_path / "cache",
    )
    assert telemetry["clean_usable_in"] <= 0.5
    assert telemetry["clean_usable_out"] <= 12.25, (
        f"Expected violent tail shake (11.75-14.0s) trimmed off, got clean_usable_out={telemetry['clean_usable_out']}"
    )

    vid_asset = VideoAsset(
        id="vid_001",
        path=str(vid_path),
        filename=vid_path.name,
        width=320,
        height=180,
        fps=20.0,
        duration=14.0,
        creation_time="2026-08-10T10:15:00",
        overall_summary="Courtyard promenade and sunlit arches",
        dialogues=[
            DialogueSegment(
                start_time=0.2,
                end_time=1.6,
                text="Wait is it recording?",
            ),
            DialogueSegment(
                start_time=2.4,
                end_time=5.8,
                text="Look at how the morning light hits these ancient arches!",
            ),
        ],
    )
    moments = extract_candidate_video_moments(vid_asset, telemetry, max_broll_sec=4.0)
    assert len(moments) >= 2, f"Expected 1 -> N multi-moment split (A-roll + scenic B-roll), got {len(moments)}"

    roll_types = [m["roll_type"] for m in moments]
    assert "a_roll" in roll_types
    assert "b_roll" in roll_types

    aroll_m = next(m for m in moments if m["roll_type"] == "a_roll")
    broll_m = next(m for m in moments if m["roll_type"] == "b_roll")
    aroll_text = " ".join(d.text for d in aroll_m["dialogues"]).lower()
    assert "recording" not in aroll_text
    assert "ancient arches" in aroll_text
    assert aroll_m["golden_in"] >= 1.9
    assert broll_m["golden_in"] >= 6.0
    assert broll_m["usable_out"] <= 12.25
    assert broll_m["golden_out"] <= broll_m["usable_out"]

    proj_dir = tmp_path / "proj_vid"
    proj_dir.mkdir(parents=True, exist_ok=True)
    bb = PipelineBlackboard(
        media_dir=vid_path.parent,
        project_dir=proj_dir,
        user_prompt="Ancient sunlit courtyard arches",
        use_antigravity=False,
    )
    bb.creative_brief = CreativeBrief(
        project_title="Courtyard_Walk",
        must_include_topics=["ancient sunlit arches"],
        target_duration_sec=45.0,
    )
    bb.catalog = MediaCatalog(
        source_directory=str(vid_path.parent),
        created_at="2026-08-10T10:20:00",
        videos=[vid_asset],
    )
    CullerAgent().run(bb)
    assert bb.media_index is not None
    emitted_vids = [s for s in bb.media_index.subclips if s.media_type == "video" and not s.is_discarded]
    assert len(emitted_vids) >= 2
    assert all((s.usable_out or 0.0) <= 12.25 for s in emitted_vids)


# ---------------------------------------------------------------------------
# Gate 4: Spatio-Temporal + GPS + Semantic Dynamic Programming Chapter Clustering
# ---------------------------------------------------------------------------
def test_spatio_temporal_gps_semantic_chapter_clustering(tmp_path: Path) -> None:
    proj_dir = tmp_path / "proj_story_dp"
    proj_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        # Cluster A (2 items, Morning Botanical Garden)
        ("cA_1", "video", "2026-05-10T09:00:00", 48.8430, 2.3560, "wide", "Morning botanical greenhouse ferns and orchids", False, ""),
        ("cA_2", "photo", "2026-05-10T09:04:00", 48.8432, 2.3562, "closeup", "Dewdrops on tropical botanical orchid petals", False, ""),
        # Cluster B (7 items, Midday Sculpture Plaza ~2.5km away)
        ("cB_1", "video", "2026-05-10T12:30:00", 48.8610, 2.3360, "wide", "Grand marble sculpture plaza courtyard panorama", False, ""),
        ("cB_2", "video", "2026-05-10T12:34:00", 48.8611, 2.3361, "medium", "Talking about classical marble statues in the plaza", True, "These marble statues in the courtyard are breathtaking!"),
        ("cB_3", "photo", "2026-05-10T12:38:00", 48.8612, 2.3362, "closeup", "Carved marble drapery detail on classical statue", False, ""),
        ("cB_4", "photo", "2026-05-10T12:42:00", 48.8612, 2.3363, "medium", "Family smiling beside limestone fountain", False, ""),
        ("cB_5", "video", "2026-05-10T12:46:00", 48.8613, 2.3363, "medium", "Walking along the colonnade gallery arches", False, ""),
        ("cB_6", "photo", "2026-05-10T12:50:00", 48.8614, 2.3364, "closeup", "Bronze frieze relief medallion closeup", False, ""),
        ("cB_7", "photo", "2026-05-10T12:55:00", 48.8615, 2.3365, "wide", "Sunlit courtyard atrium looking back at plaza", False, ""),
        # Cluster C (3 items, Night River Harbor ~3.1km away)
        ("cC_1", "video", "2026-05-10T21:10:00", 48.8580, 2.2940, "wide", "Night river harbor bridge sparkling lights reflection", False, ""),
        ("cC_2", "photo", "2026-05-10T21:15:00", 48.8582, 2.2942, "medium", "Evening illuminated boat cruising on dark water", False, ""),
        ("cC_3", "video", "2026-05-10T21:20:00", 48.8584, 2.2945, "wide", "Midnight skyline finale shimmer over harbor", False, ""),
    ]

    subclips = [
        CulledSubClip(
            subclip_id=sid,
            source_asset_id=sid,
            source_path=f"/fake/{sid}.mp4" if mtype == "video" else f"/fake/{sid}.jpg",
            filename=f"{sid}.mp4" if mtype == "video" else f"{sid}.jpg",
            media_type=mtype,  # type: ignore[arg-type]
            roll_type="a_roll" if speech else ("still" if mtype == "photo" else "b_roll"),
            normalized_timestamp=ts,
            golden_in=0.0,
            golden_out=4.0 if speech else 3.2,
            usable_in=0.0,
            usable_out=6.0,
            action_peak_tc=2.0,
            duration=4.0 if speech else 3.2,
            shot_scale=scale,  # type: ignore[arg-type]
            spoken_text=spoken,
            summary=summary,
            gps_lat=lat,
            gps_lon=lon,
            emotion_score=8.5,
            aesthetic_score=8.5,
        )
        for sid, mtype, ts, lat, lon, scale, summary, speech, spoken in specs
    ]

    bb = PipelineBlackboard(
        media_dir=tmp_path,
        project_dir=proj_dir,
        user_prompt="Create a 3-act travel film",
        use_antigravity=False,
    )
    bb.creative_brief = CreativeBrief(
        project_title="Three_Stop_Day",
        raw_user_prompt="Create a 3-act travel film",
        story_arc_mode="chronological_cinema",
        cold_open_hook=False,
        target_duration_sec=55.0,
    )
    bb.media_index = MediaIndex(source_directory=str(tmp_path), created_at="2026-05-10T22:00:00", subclips=subclips)

    StoryProducerAgent().run(bb)
    pe = bb.paper_edit
    assert pe is not None
    assert len(pe.chapters) == 3

    def _chapter_all_ids(ch: Any) -> set[str]:
        ids = set(ch.selected_subclip_ids)
        for v2_list in ch.broll_anchor_map.values():
            ids.update(v2_list)
        return ids

    ch0_ids = _chapter_all_ids(pe.chapters[0])
    ch1_ids = _chapter_all_ids(pe.chapters[1])
    ch2_ids = _chapter_all_ids(pe.chapters[2])
    assert ch0_ids == {"cA_1", "cA_2"}, f"Expected Chapter 1 = Botanical Garden (2 clips), got {ch0_ids}"
    assert ch2_ids == {"cC_1", "cC_2", "cC_3"}, f"Expected Chapter 3 = Night Harbor (3 clips), got {ch2_ids}"
    assert "cB_2" in ch1_ids and len(ch1_ids) == 7

    # Also test thematic_essay mode when culinary & cathedral clips are interleaved across time!
    interleaved_specs = [
        ("t1", "video", "2026-05-10T09:00:00", "wide", "Fresh bakery croissant espresso cafe table"),
        ("t2", "video", "2026-05-10T09:05:00", "wide", "Gothic cathedral stained glass spire"),
        ("t3", "photo", "2026-05-10T11:00:00", "closeup", "Warm pastry tart and cappuccino latte art"),
        ("t4", "photo", "2026-05-10T11:05:00", "closeup", "Cathedral vaulted stone arch and rose window"),
        ("t5", "photo", "2026-05-10T15:00:00", "medium", "Artisanal bakery bread loaves and coffee"),
        ("t6", "video", "2026-05-10T15:05:00", "medium", "Cathedral bell tower facade and gargoyle"),
    ]
    thematic_clips = [
        CulledSubClip(
            subclip_id=sid,
            source_asset_id=sid,
            source_path=f"/fake/{sid}.mp4",
            filename=f"{sid}.mp4",
            media_type=mtype,  # type: ignore[arg-type]
            roll_type="still" if mtype == "photo" else "b_roll",
            normalized_timestamp=ts,
            golden_in=0.0,
            golden_out=3.2,
            usable_in=0.0,
            usable_out=6.0,
            action_peak_tc=1.6,
            duration=3.2,
            shot_scale=scale,  # type: ignore[arg-type]
            summary=summ,
            emotion_score=8.4,
            aesthetic_score=8.4,
        )
        for sid, mtype, ts, scale, summ in interleaved_specs
    ]
    bb_th = PipelineBlackboard(
        media_dir=tmp_path,
        project_dir=proj_dir,
        user_prompt="2-act thematic essay",
        use_antigravity=False,
    )
    bb_th.creative_brief = CreativeBrief(
        project_title="Thematic_Test",
        raw_user_prompt="Create a 2-act thematic essay",
        story_arc_mode="thematic_essay",
        cold_open_hook=False,
        target_duration_sec=35.0,
    )
    bb_th.media_index = MediaIndex(
        source_directory=str(tmp_path),
        created_at="2026-05-10T22:00:00",
        subclips=thematic_clips,
    )
    StoryProducerAgent().run(bb_th)
    assert bb_th.paper_edit is not None
    assert bb_th.paper_edit.story_arc_mode == "thematic_essay"
    assert len(bb_th.paper_edit.chapters) == 2
    grp0 = _chapter_all_ids(bb_th.paper_edit.chapters[0])
    grp1 = _chapter_all_ids(bb_th.paper_edit.chapters[1])
    assert grp0 in ({"t1", "t3", "t5"}, {"t2", "t4", "t6"})
    assert grp1 in ({"t1", "t3", "t5"}, {"t2", "t4", "t6"})


# ---------------------------------------------------------------------------
# Gate 5: Five-Shot Film Grammar, Neural Semantic V2 Cutaway & Cold-Open Dedup
# ---------------------------------------------------------------------------
def test_five_shot_cinematic_grammar_and_neural_v2_cutaway_matching(tmp_path: Path) -> None:
    proj_dir = tmp_path / "proj_grammar"
    proj_dir.mkdir(parents=True, exist_ok=True)

    sem = SemanticEmbeddingEngine.instance()
    sim_bakery = sem.semantic_similarity(
        "We stopped for warm pastries and morning coffee",
        "Fresh Croissant Espresso Bakery Counter",
    )
    sim_unrelated = sem.semantic_similarity(
        "We stopped for warm pastries and morning coffee",
        "Subway escalator concrete tunnel ticket gate",
    )
    assert sim_bakery > sim_unrelated + 0.15
    assert sim_bakery >= 0.22

    subclips = [
        # Short visual peak hook candidate (v_hook, 3.0s) — with 4 videos total, should NOT be duplicated in Chapter 1!
        CulledSubClip(
            subclip_id="v_hook",
            source_asset_id="v_hook",
            source_path="/fake/v_hook.mp4",
            filename="v_hook.mp4",
            media_type="video",
            roll_type="b_roll",
            normalized_timestamp="2026-05-12T08:55:00",
            golden_in=0.0,
            golden_out=3.0,
            usable_in=0.0,
            usable_out=5.0,
            action_peak_tc=1.5,
            duration=3.0,
            shot_scale="wide",
            camera_movement="pan_right",
            summary="Golden sunrise panorama across river bridges",
            prosody_excitement_score=0.92,
            stability_score=0.95,
            emotion_score=9.4,
            aesthetic_score=9.3,
        ),
        # Mundane logistical question (should NOT beat v_hook for Cold-Open!)
        CulledSubClip(
            subclip_id="v_mundane",
            source_asset_id="v_mundane",
            source_path="/fake/v_mundane.mp4",
            filename="v_mundane.mp4",
            media_type="video",
            roll_type="a_roll",
            normalized_timestamp="2026-05-12T09:02:00",
            golden_in=0.0,
            golden_out=3.5,
            usable_in=0.0,
            usable_out=5.0,
            action_peak_tc=1.5,
            duration=3.5,
            shot_scale="medium",
            spoken_text="Where is the bathroom near the parking lot?",
            summary="Asking where the restroom is",
            emotion_score=7.5,
            aesthetic_score=7.8,
        ),
        # Closeup comes FIRST chronologically in Chapter 1, Wide comes later -> Five-Shot Grammar promotes Wide to slot 0!
        CulledSubClip(
            subclip_id="p_ticket_closeup",
            source_asset_id="p_ticket_closeup",
            source_path="/fake/p_ticket.jpg",
            filename="p_ticket.jpg",
            media_type="photo",
            roll_type="still",
            normalized_timestamp="2026-05-12T09:05:00",
            golden_in=0.0,
            golden_out=3.0,
            usable_in=0.0,
            usable_out=8.0,
            action_peak_tc=1.5,
            duration=3.0,
            shot_scale="closeup",
            summary="Subway escalator concrete tunnel ticket gate",
            emotion_score=7.6,
            aesthetic_score=7.9,
        ),
        # 5.2s A-Roll speaking about warm pastries & coffee (< 8.5s, yet eligible via neural semantic match!)
        CulledSubClip(
            subclip_id="v_aroll_cafe",
            source_asset_id="v_aroll_cafe",
            source_path="/fake/v_aroll_cafe.mp4",
            filename="v_aroll_cafe.mp4",
            media_type="video",
            roll_type="a_roll",
            normalized_timestamp="2026-05-12T09:08:00",
            golden_in=0.0,
            golden_out=5.2,
            usable_in=0.0,
            usable_out=7.0,
            action_peak_tc=2.6,
            duration=5.2,
            shot_scale="medium",
            spoken_text="We stopped for warm pastries and morning coffee!",
            summary="Talking at street table",
            emotion_score=8.9,
            aesthetic_score=8.7,
        ),
        CulledSubClip(
            subclip_id="p_establishing_boulevard",
            source_asset_id="p_establishing_boulevard",
            source_path="/fake/p_boulevard.jpg",
            filename="p_boulevard.jpg",
            media_type="photo",
            roll_type="still",
            normalized_timestamp="2026-05-12T09:10:00",
            golden_in=0.0,
            golden_out=3.2,
            usable_in=0.0,
            usable_out=8.0,
            action_peak_tc=1.6,
            duration=3.2,
            shot_scale="wide",
            summary="Wide morning boulevard terrace architecture",
            emotion_score=8.6,
            aesthetic_score=8.6,
        ),
        CulledSubClip(
            subclip_id="p_croissant_bakery",
            source_asset_id="p_croissant_bakery",
            source_path="/fake/p_croissant.jpg",
            filename="p_croissant.jpg",
            media_type="photo",
            roll_type="still",
            normalized_timestamp="2026-05-12T09:12:00",
            golden_in=0.0,
            golden_out=3.0,
            usable_in=0.0,
            usable_out=8.0,
            action_peak_tc=1.5,
            duration=3.0,
            shot_scale="closeup",
            summary="Fresh Croissant Espresso Bakery Counter",
            annotation_title="Fresh Croissant Bakery",
            emotion_score=8.8,
            aesthetic_score=8.9,
        ),
        # Second chapter selects (Afternoon Park)
        CulledSubClip(
            subclip_id="v_park_wide",
            source_asset_id="v_park_wide",
            source_path="/fake/v_park_wide.mp4",
            filename="v_park_wide.mp4",
            media_type="video",
            roll_type="b_roll",
            normalized_timestamp="2026-05-12T15:30:00",
            golden_in=0.0,
            golden_out=3.8,
            usable_in=0.0,
            usable_out=6.0,
            action_peak_tc=1.9,
            duration=3.8,
            shot_scale="wide",
            summary="Sunlit afternoon park lawn and fountain",
            emotion_score=8.5,
            aesthetic_score=8.5,
        ),
        CulledSubClip(
            subclip_id="p_park_portrait",
            source_asset_id="p_park_portrait",
            source_path="/fake/p_park_portrait.jpg",
            filename="p_park_portrait.jpg",
            media_type="photo",
            roll_type="still",
            normalized_timestamp="2026-05-12T15:35:00",
            golden_in=0.0,
            golden_out=3.2,
            usable_in=0.0,
            usable_out=8.0,
            action_peak_tc=1.6,
            duration=3.2,
            shot_scale="medium",
            summary="Smiling family portrait under park trees",
            emotion_score=8.7,
            aesthetic_score=8.8,
        ),
    ]

    bb = PipelineBlackboard(
        media_dir=tmp_path,
        project_dir=proj_dir,
        user_prompt="2-act morning cafe and park story",
        use_antigravity=False,
    )
    bb.creative_brief = CreativeBrief(
        project_title="Morning_Cafe_And_Park",
        raw_user_prompt="Create a 2-act story",
        target_duration_sec=50.0,
    )
    bb.media_index = MediaIndex(source_directory=str(tmp_path), created_at="2026-05-12T18:00:00", subclips=subclips)

    StoryProducerAgent().run(bb)
    pe = bb.paper_edit
    assert pe is not None
    assert pe.cold_open_hook is not None
    assert pe.cold_open_hook.subclip_id == "v_hook"
    all_chapter_selects = [sid for ch in pe.chapters for sid in ch.selected_subclip_ids]
    assert "v_hook" not in all_chapter_selects

    ch1 = pe.chapters[0]
    assert ch1.selected_subclip_ids[0] == "p_establishing_boulevard"
    assert ch1.shot_grammar_sequence[0] == "establishing_wide"

    assert ch1.broll_anchor_map.get("v_aroll_cafe") == ["p_croissant_bakery"]


# ---------------------------------------------------------------------------
# Gate 6: Multi-Provider VLM Gateway (OpenAI / Anthropic / Gemini) & CLI Wiring
# ---------------------------------------------------------------------------
class _DummyGatewayAgent(BaseEditingAgent):
    role_id = "dummy_vlm_gateway_probe"
    display_name = "Dummy VLM Probe"
    stage_order = 99
    enabled_by_default = False

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        return {}


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def test_multi_provider_vlm_gateway_and_story_arc_intake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj_dir = tmp_path / "proj_vlm"
    media_dir = tmp_path / "media_vlm"
    proj_dir.mkdir(parents=True, exist_ok=True)
    img_path = _draw_portrait_photo(media_dir / "IMG_9001.jpg")

    bb = PipelineBlackboard(media_dir=media_dir, project_dir=proj_dir, user_prompt="Hero's journey adventure")
    agent = _DummyGatewayAgent()

    # 1. OpenAI Vision JSON routing
    monkeypatch.setenv("AI_EDITOR_VLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")
    with patch("urllib.request.urlopen") as mock_url:
        mock_url.return_value = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"selected_hero": "IMG_9001.jpg", "confidence": 0.96}),
                        }
                    }
                ]
            }
        )
        res_oa = agent.ask_ai_creative_json(
            bb,
            "Pick hero frame",
            image_paths=[img_path],
            decision_key="probe_openai_decision",
        )
        assert res_oa == {"selected_hero": "IMG_9001.jpg", "confidence": 0.96}

    # 2. Anthropic Claude Vision JSON routing
    monkeypatch.setenv("AI_EDITOR_VLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    with patch("urllib.request.urlopen") as mock_url:
        mock_url.return_value = _FakeHTTPResponse(
            {
                "content": [
                    {
                        "type": "text",
                        "text": "```json\n{\"chapter_theme\": \"Golden Arrival\", \"score\": 9.4}\n```",
                    }
                ]
            }
        )
        res_cl = agent.ask_ai_creative_json(
            bb,
            "Name chapter",
            image_paths=[img_path],
            decision_key="probe_anthropic_decision",
        )
        assert res_cl == {"chapter_theme": "Golden Arrival", "score": 9.4}

    # 3. ClientIntakeAgent infers story_arc_mode cleanly from prompt / interview_answers
    bb_intake = PipelineBlackboard(
        media_dir=media_dir,
        project_dir=proj_dir,
        user_prompt="Edit our trip as a dramatic hero's journey film",
        use_antigravity=False,
    )
    ClientIntakeAgent().run(bb_intake)
    assert bb_intake.creative_brief is not None
    assert bb_intake.creative_brief.story_arc_mode == "hero_journey"
