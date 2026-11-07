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


def generate_intake_questions(
    media_dir: str | Path,
    initial_prompt: str = "",
) -> list[InterviewQuestion]:
    """Pre-scan the raw media folder (recursively) and generate context-aware client discovery questions."""
    m_dir = Path(media_dir)
    v_count = 0
    p_count = 0
    a_count = 0
    if m_dir.exists() and m_dir.is_dir():
        for f in iter_media_files_recursive(m_dir):
            ext = f.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                v_count += 1
            elif ext in PHOTO_EXTENSIONS:
                p_count += 1
            elif ext in AUDIO_EXTENSIONS:
                a_count += 1

    media_note = f"({v_count} videos, {p_count} photos detected)" if (v_count or p_count) else ""
    return [
        InterviewQuestion(
            id="narrative_style",
            question=f"How should the story arc be structured across your media {media_note}?",
            options=[
                "family_keepsake_4act (Classic 4-Act Family Story: Arrival -> Action -> Landmarks -> Finale)",
                "day_in_the_life_vlog (Chronological Day-in-the-Life YouTube Vlog)",
                "upbeat_montage (Fast-Paced Beat-Driven Highlight Montage)",
                "dialogue_story (Narrative A-Roll Driven with B-Roll Cutaways)",
            ],
            default_answer="family_keepsake_4act",
        ),
        InterviewQuestion(
            id="dialogue_preference",
            question="How should spoken camera dialogue and ambient sound be handled?",
            options=[
                "keep_full_sentences_with_broll_cutaways (Keep full spoken sentences on A1, cover long takes with V2 B-Roll cutaways, keep -20dB Nat-Sound)",
                "highlights_only (Keep only the shortest punchline quotes)",
                "music_mostly (Focus primarily on music bed and visual rhythm)",
            ],
            default_answer="keep_full_sentences_with_broll_cutaways",
        ),
        InterviewQuestion(
            id="photo_mode",
            question="How should vertical/horizontal still photos be framed and animated?",
            options=[
                "gallery_frame (Uncropped MrAlexTech Gallery Frame + Blurred Ambient Backdrop + Floating Ken Burns)",
                "full_bleed (Full-Screen 16:9 Crop with Classic Ken Burns Pan/Zoom)",
            ],
            default_answer="gallery_frame",
        ),
        InterviewQuestion(
            id="target_duration_sec",
            question="What is your target total duration and B-roll pacing speed?",
            options=[
                "110 (Standard ~1.8 min Story Cut, max 4.5s B-roll)",
                "60 (Tight 60s Reel, fast 2.5s-3.5s B-roll)",
                "150 (Extended ~2.5 min Family Vlog)",
            ],
            default_answer="110",
        ),
        InterviewQuestion(
            id="must_include_topics",
            question="Which specific moments, quotes, or landmarks are MUST-HAVES in the final edit?",
            options=[
                "All spoken kid/family dialogues, rides, iconic landmarks, and finale show",
            ],
            default_answer=initial_prompt or "All spoken kid/family dialogues, rides, iconic landmarks, and finale show",
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

        photo_mode = _clean_choice("photo_mode", "gallery_frame")
        if photo_mode not in {"gallery_frame", "full_bleed"}:
            photo_mode = "gallery_frame"

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
