"""Stage 0: Client Intake & Creative Brief Producer Agent (client_intake)."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Optional

from ..analyzer.media_probe import (
    AUDIO_EXTENSIONS,
    PHOTO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    iter_media_files_recursive,
)
from ..models.schema import CreativeBrief, InterviewQuestion
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


def inspect_media_library(
    media_dir: str | Path,
    workspace_root: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Recursively inspect a media directory and return counts, camera badges, suggested project path, and default prompt."""
    m_dir = Path(media_dir)
    v_count = 0
    p_count = 0
    a_count = 0
    total_bytes = 0
    cameras: set[str] = set()
    sample_files: list[str] = []

    if m_dir.exists() and m_dir.is_dir():
        for f in iter_media_files_recursive(m_dir):
            ext = f.suffix.lower()
            fname = f.name
            try:
                total_bytes += f.stat().st_size
            except Exception:
                pass
            if len(sample_files) < 6:
                sample_files.append(fname)

            if fname.startswith(("_DSC", "A7C")):
                cameras.add("Sony A7C II (33MP / 4K XAVC-S)")
            elif fname.startswith("PXL_"):
                cameras.add("Google Pixel (4K HDR)")
            elif fname.startswith(("IMG_", "MVI_")):
                cameras.add("Apple iPhone / Canon")

            if ext in VIDEO_EXTENSIONS:
                v_count += 1
            elif ext in PHOTO_EXTENSIONS:
                p_count += 1
            elif ext in AUDIO_EXTENSIONS:
                a_count += 1

    path_low = str(m_dir).lower() + " " + " ".join(sample_files).lower()
    ws_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()

    if "louvre" in path_low or "paris" in path_low:
        theme = "louvre_paris"
        display_title = "Louvre Museum Paris Family Day Trip"
        suggested_proj = str(ws_root / "louvre_day_trip_project")
        default_prompt = (
            "Create a cinematic, heartwarming 4-act family day trip highlight video at the Louvre Museum in Paris. "
            "Combine our Sony A7C2 mirrorless camera and Pixel phone clips and photos, showcase the Louvre Pyramid, "
            "iconic masterpieces, sculptures, and family portraits with museum annotations and 2.5D parallax motion."
        )
    elif "halloween" in path_low or "gilroy" in path_low or "raw_media" in path_low:
        theme = "halloween_park"
        display_title = "Gilroy Gardens Family Halloween Night"
        suggested_proj = str(ws_root / "gilroy_halloween_project")
        default_prompt = (
            "Create an upbeat, heartwarming 4-act family Halloween highlight video at Gilroy Gardens. "
            "Keep the kids' excited Mandarin dialogue in the car and on the spinning ride, cover long takes with "
            "B-roll cutaways, feature the giant pumpkin pyramid and maze photos with 2.5D gallery frames, "
            "include the Great Big Boo live stage song, and end with Boo Alley night lights."
        )
    else:
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", m_dir.name or "family_story").strip("_").lower() or "family_story"
        theme = "family_travel"
        display_title = f"{m_dir.name or 'Family Travel'} Highlight Story"
        suggested_proj = str(ws_root / f"{slug}_project")
        default_prompt = (
            f"Create a cinematic, heartwarming 4-act family travel highlight video from '{m_dir.name}'. "
            "Balance candid spoken moments with beat-synced B-roll, curated high-taste photos with bilingual "
            "annotations and 2.5D parallax motion, and a mood-matched soundtrack in 4K 10-bit HLG HDR."
        )

    return {
        "media_dir": str(m_dir.resolve()) if m_dir.exists() else str(m_dir),
        "exists": m_dir.exists() and m_dir.is_dir(),
        "display_title": display_title,
        "theme": theme,
        "video_count": v_count,
        "photo_count": p_count,
        "audio_count": a_count,
        "total_size_mb": round(total_bytes / (1024 * 1024), 1),
        "detected_cameras": sorted(cameras) if cameras else ["Multi-Cam / Phone & Mirrorless"],
        "sample_files": sample_files,
        "suggested_project_dir": suggested_proj,
        "default_prompt": default_prompt,
    }


def generate_intake_questions(
    media_dir: str | Path,
    initial_prompt: str = "",
) -> list[InterviewQuestion]:
    """Pre-scan the raw media folder (recursively) and generate context-aware client discovery questions."""
    info = inspect_media_library(media_dir)
    v_count = info["video_count"]
    p_count = info["photo_count"]
    theme = info["theme"]
    prompt_to_use = initial_prompt.strip() or info["default_prompt"]

    media_note = f"({v_count} videos, {p_count} photos detected)" if (v_count or p_count) else ""
    if theme == "louvre_paris":
        must_options = [
            "Louvre Glass Pyramid, Mona Lisa & Keepsake Portrait, Winged Victory, Venus de Milo, Coronation of Napoleon, Liberty Leading the People, and Parisian Bistro Finale",
            "Focus primarily on iconic Louvre paintings and classical sculptures with museum plaques",
            "Prioritize candid family portraits and courtyard walks alongside top masterpieces",
        ]
    elif theme == "halloween_park":
        must_options = [
            "All spoken kid/family dialogues, spinning rides, giant pumpkin pyramid portraits, and Great Big Boo stage finale",
            "Fast-paced music montage of Halloween lights, rides, and pumpkin displays",
        ]
    else:
        must_options = [
            "All spoken kid/family dialogues, action highlights, iconic landmarks, and warm family portraits",
            "Scenic landmarks, architecture, and curated portrait highlights",
        ]

    return [
        InterviewQuestion(
            id="narrative_style",
            question=f"1. Story Arc & Structure {media_note}",
            options=[
                "family_keepsake_4act (Classic 4-Act Family Story: Arrival -> Exploration -> Masterpieces/Landmarks -> Finale)",
                "day_in_the_life_vlog (Chronological Day-in-the-Life YouTube Vlog)",
                "upbeat_montage (Fast-Paced Beat-Driven Highlight Montage)",
                "dialogue_story (Narrative A-Roll Driven with Track V2 B-Roll Cutaways)",
            ],
            default_answer="family_keepsake_4act",
        ),
        InterviewQuestion(
            id="photo_mode",
            question="2. Still Photo Visual Effect & Framing Style",
            options=[
                "parallax_gallery (2.5D Counter-Panning Parallax + Double Ivory/Gold Museum Matte + Bilingual Exhibition Plaque + Diptych Split-Screens)",
                "gallery_frame (Classic Floating Gallery Frame + Blurred Ambient Backdrop)",
                "full_bleed (Full-Screen 16:9 Crop + Classic Ken Burns Pan/Zoom)",
            ],
            default_answer="parallax_gallery",
        ),
        InterviewQuestion(
            id="bgm_mood",
            question="3. Background Music & Soundtrack Direction (Stage 05 Music Supervisor)",
            options=[
                "auto_match (Auto-Scout & Match Destination Mood — e.g., Erik Satie French Classical for Louvre / Festive for Theme Parks)",
                "classical_chamber (Elegant Classical Piano, Harp & Strings Chamber Score)",
                "warm_acoustic (Heartwarming Acoustic Guitar & Piano Family Keepsake)",
            ],
            default_answer="auto_match",
        ),
        InterviewQuestion(
            id="dialogue_preference",
            question="4. Spoken Camera Dialogue & Audio Mix",
            options=[
                "keep_full_sentences_with_broll_cutaways (Keep full sentences on Track A1 + Track V2 B-Roll Cutaways + -14dB BGM Auto-Ducking)",
                "highlights_only (Keep only the shortest candid quotes)",
                "music_mostly (Focus primarily on musical rhythm and ambient -20dB Nat-Sound)",
            ],
            default_answer="keep_full_sentences_with_broll_cutaways",
        ),
        InterviewQuestion(
            id="target_duration_sec",
            question="5. Target Video Duration & Pacing",
            options=[
                "90 (Balanced ~70-90s Cinematic Highlight Cut, 3.0-4.5s B-Roll)",
                "60 (Punchy ~60s Social Reel, fast 2.5-3.5s B-Roll)",
                "120 (Extended ~2 min Family Documentary Cut)",
            ],
            default_answer="90",
        ),
        InterviewQuestion(
            id="must_include_topics",
            question="6. Must-Include Highlights & Moments",
            options=must_options,
            default_answer=prompt_to_use,
        ),
    ]


@register_agent
class ClientIntakeAgent(BaseEditingAgent):
    """Agent 0: Interviews the client (or synthesizes prompt + answers) into creative_brief.json."""

    role_id = "client_intake"
    display_name = "0. Client Intake & Brief Producer"
    stage_order = 0
    description = "Interviews the client on story arc, tone, must-haves, and pacing to produce creative_brief.json."
    enabled_by_default = True
    requires_artifacts = []
    produces_artifacts = ["creative_brief.json"]
    skill_reference_file = "00_client_intake_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        questions = generate_intake_questions(blackboard.media_dir, blackboard.user_prompt)
        answers = blackboard.interview_answers

        def _clean_choice(key: str, default: str) -> str:
            raw = answers.get(key, default).strip()
            return raw.split(" ")[0].strip() if raw else default

        narrative_style = _clean_choice("narrative_style", "family_keepsake_4act")
        if narrative_style not in {
            "family_keepsake_4act",
            "day_in_the_life_vlog",
            "upbeat_montage",
            "dialogue_story",
        }:
            narrative_style = "family_keepsake_4act"

        dialogue_pref = _clean_choice("dialogue_preference", "keep_full_sentences_with_broll_cutaways")
        if dialogue_pref not in {
            "keep_full_sentences_with_broll_cutaways",
            "highlights_only",
            "music_mostly",
        }:
            dialogue_pref = "keep_full_sentences_with_broll_cutaways"

        photo_mode = _clean_choice("photo_mode", "parallax_gallery")
        if photo_mode not in {"parallax_gallery", "diptych_frame", "gallery_frame", "full_bleed"}:
            photo_mode = "parallax_gallery"

        try:
            target_dur = float(_clean_choice("target_duration_sec", "110"))
        except ValueError:
            target_dur = 110.0

        # Extract must-include topics from user_prompt and answers
        raw_must = answers.get("must_include_topics") or blackboard.user_prompt
        must_topics: list[str] = []
        if raw_must:
            parts = [p.strip() for p in re.split(r"[,;•\n]+|\band\b", raw_must) if len(p.strip()) > 2]
            must_topics = parts[:8]
        if not must_topics:
            must_topics = ["spoken dialogue", "rides", "pumpkin pyramid", "stage show"]

        # Derive project title from user_prompt
        project_title = "Family_Travel_Story"
        prompt_low = blackboard.user_prompt.lower()
        if "louvre" in prompt_low:
            project_title = "Louvre_Museum_Paris_Day_Trip"
        elif "halloween" in prompt_low or "gilroy" in prompt_low:
            project_title = "Halloween_Park_Family_Night"
        elif blackboard.user_prompt.strip():
            words = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\s_-]", "", blackboard.user_prompt).split()
            if words:
                project_title = "_".join(words[:5])

        qa_records = [
            {
                "id": q.id,
                "question": q.question,
                "answer": answers.get(q.id, q.default_answer),
            }
            for q in questions
        ]

        brief = CreativeBrief(
            project_title=project_title,
            raw_user_prompt=blackboard.user_prompt,
            narrative_style=narrative_style,  # type: ignore[arg-type]
            tone="warm_joyful",
            target_duration_sec=target_dur,
            must_include_topics=must_topics,
            avoid_topics=["blurry pocket shots", "duplicate photo bursts", "long static dead air"],
            dialogue_preference=dialogue_pref,  # type: ignore[arg-type]
            bilingual_overlay_captions=True,
            photo_mode=photo_mode,  # type: ignore[arg-type]
            max_broll_duration_sec=3.5 if target_dur <= 65 else 4.5,
            max_consecutive_same_subject=3,
            nat_sound_db=-20.0,
            music_duck_db=-14.0,
            enabled_optional_agents=(
                [a.strip() for a in answers.get("enabled_optional_agents", "").split(",") if a.strip()]
            ),
            interview_qa=qa_records,
        )

        # Optional Antigravity SDK refinement
        if blackboard.use_antigravity and blackboard.user_prompt:
            ag_resp = self.ask_antigravity_json(
                f"User prompt: {blackboard.user_prompt}\n"
                f"Interview Q&A: {qa_records}\n"
                f"Current CreativeBrief JSON: {brief.model_dump_json()}\n"
                "Refine project_title, tone, and must_include_topics if helpful, returning the full CreativeBrief JSON.",
                timeout_sec=30.0,
            )
            if isinstance(ag_resp, dict):
                try:
                    brief = CreativeBrief.model_validate({**brief.model_dump(), **ag_resp})
                except Exception:
                    pass

        brief_path = blackboard.project_dir / "creative_brief.json"
        brief.to_json_file(brief_path)
        blackboard.creative_brief = brief

        # Coordinate with all downstream agents on the Blackboard Message Bus
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="culler",
            message_type="directive",
            subject="Boost client must-have topics & preserve spoken dialogue windows",
            payload={
                "must_include_topics": brief.must_include_topics,
                "dialogue_preference": brief.dialogue_preference,
            },
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="story_producer",
            message_type="directive",
            subject="Structure chapters according to client narrative style and target duration",
            payload={
                "narrative_style": brief.narrative_style,
                "target_duration_sec": brief.target_duration_sec,
                "photo_mode": brief.photo_mode,
            },
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="showrunner",
            message_type="directive",
            subject="Enforce client pacing caps and verify must-have coverage",
            payload={
                "max_broll_duration_sec": brief.max_broll_duration_sec,
                "max_consecutive_same_subject": brief.max_consecutive_same_subject,
                "must_include_topics": brief.must_include_topics,
            },
        )

        return {
            "creative_brief_path": str(brief_path),
            "project_title": brief.project_title,
            "must_include_topics": brief.must_include_topics,
        }
