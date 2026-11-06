"""Multimodal visual & semantic media annotation powered by the Google Antigravity SDK."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from typing import Optional

import numpy as np
from PIL import Image

from ..models.schema import MediaCatalog, PhotoAsset, VideoAsset
from .media_probe import build_initial_catalog
from .transcriber import transcribe_catalog_videos


def _compute_visual_metrics(img_path: str | Path) -> dict[str, float | str]:
    """Compute local visual metrics (sharpness, brightness, warmth, lighting condition) for an image/keyframe."""
    try:
        im = Image.open(img_path).convert("RGB")
        im.thumbnail((320, 320))
        arr = np.asarray(im, dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        brightness = float(luma.mean())
        contrast = float(luma.std())

        # Laplacian-like sharpness metric
        dx = np.diff(luma, axis=1)
        dy = np.diff(luma, axis=0)
        sharpness = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))

        warmth = float(np.mean(r - b))
        if brightness < 75:
            lighting = "twilight_or_night"
        elif brightness > 175:
            lighting = "bright_daylight"
        elif warmth > 22:
            lighting = "warm_golden_light"
        else:
            lighting = "natural_daylight"

        # Score between 6.5 and 9.6 based on sharpness, contrast, and balanced exposure
        exposure_penalty = abs(brightness - 128.0) / 160.0
        raw_score = 7.2 + min(1.8, sharpness / 18.0) + min(0.8, contrast / 65.0) - exposure_penalty * 0.8
        score = round(min(9.8, max(5.5, raw_score)), 1)

        return {
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "sharpness": round(sharpness, 2),
            "lighting": lighting,
            "quality_score": score,
        }
    except Exception:
        return {
            "brightness": 120.0,
            "contrast": 50.0,
            "sharpness": 12.0,
            "lighting": "natural_daylight",
            "quality_score": 8.0,
        }


def apply_local_visual_enrichment(catalog: MediaCatalog) -> MediaCatalog:
    """Enrich scenes and photos with local visual quality scores, lighting tags, and dialogue context."""
    total_vids = len(catalog.videos)
    for v_idx, video in enumerate(catalog.videos):
        for sc in video.scenes:
            if sc.keyframe_path and Path(sc.keyframe_path).exists():
                metrics = _compute_visual_metrics(sc.keyframe_path)
                sc.quality_score = float(metrics["quality_score"])
                lighting = str(metrics["lighting"])
                if lighting not in sc.visual_tags:
                    sc.visual_tags.append(lighting)
                if lighting == "twilight_or_night":
                    sc.emotion = "atmospheric_magical"
                elif lighting == "warm_golden_light":
                    sc.emotion = "warm_joyful"

            # Attach overlapping dialogues to the scene summary
            overlapping = [
                d.text
                for d in video.dialogues
                if d.end_time > sc.start_time and d.start_time < sc.end_time
            ]
            if overlapping:
                joined = " | ".join(overlapping)
                sc.summary = (
                    f"Action in {video.filename} ({sc.start_time:.1f}s-{sc.end_time:.1f}s) "
                    f"with dialogue: \"{joined}\""
                )
                if "dialogue" not in sc.visual_tags:
                    sc.visual_tags.append("dialogue")
                sc.quality_score = round(min(9.9, sc.quality_score + 0.6), 1)
            else:
                stage = "opening" if v_idx == 0 else ("finale" if v_idx == total_vids - 1 else "activity")
                sc.summary = (
                    f"Visual {stage} scene in {video.filename} ({sc.start_time:.1f}s-{sc.end_time:.1f}s, "
                    f"{', '.join(sc.visual_tags) or 'natural motion'})"
                )

    for p_idx, photo in enumerate(catalog.photos):
        img_target = photo.preview_path if photo.preview_path and Path(photo.preview_path).exists() else photo.path
        metrics = _compute_visual_metrics(img_target)
        photo.quality_score = float(metrics["quality_score"])
        lighting = str(metrics["lighting"])
        if lighting not in photo.subjects:
            photo.subjects.append(lighting)
        if lighting == "twilight_or_night":
            photo.mood = "magical_evening"
        elif lighting == "warm_golden_light":
            photo.mood = "warm_nostalgic"
        else:
            photo.mood = "cheerful_bright"

        progress_pct = int((p_idx / max(1, len(catalog.photos) - 1)) * 100)
        photo.caption = (
            f"{photo.orientation.capitalize()} photo {photo.filename} "
            f"({lighting.replace('_', ' ')}, story progression {progress_pct}%)"
        )

    return catalog


def _extract_json_payload(text: str) -> Optional[dict]:
    """Extract a JSON object from an LLM/Agent response string."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except Exception:
            return None
    return None


async def _run_antigravity_annotation_async(
    catalog: MediaCatalog,
    user_hint: str = "",
    timeout_sec: float = 120.0,
) -> MediaCatalog:
    """Use the Google Antigravity SDK Agent to inspect contact sheets & keyframes and enrich annotations."""
    from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig

    # Prepare a concise summary of assets and contact sheets for the Antigravity agent
    video_briefs = [
        {
            "id": v.id,
            "filename": v.filename,
            "duration": v.duration,
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "keyframe_path": s.keyframe_path,
                }
                for s in v.scenes
            ],
            "dialogues": [d.model_dump() for d in v.dialogues],
        }
        for v in catalog.videos
    ]
    photo_briefs = [
        {
            "id": p.id,
            "filename": p.filename,
            "orientation": p.orientation,
            "preview_path": p.preview_path,
            "quality_score": p.quality_score,
        }
        for p in catalog.photos[:40]
    ]

    prompt = (
        "You are a multimodal film editor assistant analyzing a media catalog.\n"
        f"User story context hint: {user_hint or 'Family/travel highlight video'}\n"
        f"Contact sheet images available to view via read_file: {catalog.contact_sheet_paths[:4]}\n"
        f"Videos summary: {json.dumps(video_briefs, ensure_ascii=False)}\n"
        f"Photos summary (first 40): {json.dumps(photo_briefs, ensure_ascii=False)}\n\n"
        "Inspect the contact sheets (using read_file if available) and dialogue transcripts, then return ONLY a "
        "valid JSON object with the following structure:\n"
        "{\n"
        '  "videos": {\n'
        '    "<video_id>": {\n'
        '      "overall_summary": "...",\n'
        '      "scenes": {\n'
        '        "<scene_id>": {"summary": "...", "emotion": "...", "quality_score": 9.0, "visual_tags": ["..."]}\n'
        "      },\n"
        '      "dialogue_translations": ["optional English translations matching dialogue index"]\n'
        "    }\n"
        "  },\n"
        '  "photos": {\n'
        '    "<photo_id>": {"caption": "...", "mood": "...", "quality_score": 8.8, "subjects": ["..."]}\n'
        "  }\n"
        "}"
    )

    config = LocalAgentConfig(
        system_instructions=(
            "You are an expert multimodal video & photo annotator. Return strictly valid JSON matching the requested schema."
        ),
        capabilities=CapabilitiesConfig(),
    )

    collected_tokens: list[str] = []
    async with Agent(config) as agent:
        response = await agent.chat(prompt)
        async for token in response:
            collected_tokens.append(str(token))

    full_text = "".join(collected_tokens)
    payload = _extract_json_payload(full_text)
    if not payload:
        return catalog

    v_updates = payload.get("videos", {})
    if isinstance(v_updates, dict):
        for v in catalog.videos:
            u = v_updates.get(v.id) or v_updates.get(v.filename)
            if isinstance(u, dict):
                if u.get("overall_summary"):
                    v.overall_summary = str(u["overall_summary"])
                sc_map = u.get("scenes", {})
                if isinstance(sc_map, dict):
                    for sc in v.scenes:
                        su = sc_map.get(sc.scene_id)
                        if isinstance(su, dict):
                            if su.get("summary"):
                                sc.summary = str(su["summary"])
                            if su.get("emotion"):
                                sc.emotion = str(su["emotion"])
                            if isinstance(su.get("quality_score"), (int, float)):
                                sc.quality_score = float(su["quality_score"])
                            if isinstance(su.get("visual_tags"), list):
                                sc.visual_tags = [str(t) for t in su["visual_tags"]]
                translations = u.get("dialogue_translations")
                if isinstance(translations, list):
                    for d, tr in zip(v.dialogues, translations):
                        if tr:
                            d.translation = str(tr)

    p_updates = payload.get("photos", {})
    if isinstance(p_updates, dict):
        for p in catalog.photos:
            pu = p_updates.get(p.id) or p_updates.get(p.filename)
            if isinstance(pu, dict):
                if pu.get("caption"):
                    p.caption = str(pu["caption"])
                if pu.get("mood"):
                    p.mood = str(pu["mood"])
                if isinstance(pu.get("quality_score"), (int, float)):
                    p.quality_score = float(pu["quality_score"])
                if isinstance(pu.get("subjects"), list):
                    p.subjects = [str(s) for s in pu["subjects"]]

    return catalog


def enrich_catalog_with_antigravity(
    catalog: MediaCatalog,
    user_hint: str = "",
    use_antigravity: bool = True,
) -> MediaCatalog:
    """Enrich the MediaCatalog with local computer vision metrics and optional Antigravity SDK multimodal inspection."""
    catalog = apply_local_visual_enrichment(catalog)
    if not use_antigravity:
        return catalog

    try:
        return asyncio.run(
            asyncio.wait_for(
                _run_antigravity_annotation_async(catalog, user_hint=user_hint),
                timeout=90.0,
            )
        )
    except Exception:
        # Gracefully fall back to the local + Whisper enriched catalog if Antigravity SDK is unavailable/offline
        return catalog


def analyze_media_directory(
    media_dir: str | Path,
    project_dir: str | Path,
    whisper_model: str = "small",
    language: Optional[str] = None,
    use_antigravity: bool = True,
    user_hint: str = "",
    extra_audio_paths: Optional[list[str | Path]] = None,
) -> MediaCatalog:
    """Full Step 1 pipeline: probe media, extract keyframes & previews, transcribe dialogues, and annotate."""
    proj_path = Path(project_dir).resolve()
    cache_dir = proj_path / "cache"
    catalog = build_initial_catalog(
        media_dir=media_dir,
        cache_dir=cache_dir,
        extra_audio_paths=extra_audio_paths,
    )
    catalog = transcribe_catalog_videos(
        catalog,
        model_size=whisper_model,
        language=language,
    )
    catalog = enrich_catalog_with_antigravity(
        catalog,
        user_hint=user_hint,
        use_antigravity=use_antigravity,
    )
    catalog.to_yaml(proj_path / "annotations.yaml")
    return catalog
