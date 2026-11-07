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

    base_info: dict[str, Any] = {
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
    dur_rec = compute_director_duration_recommendation(base_info, default_prompt)
    base_info.update(dur_rec)
    return base_info


def _format_mmss(seconds: int | float) -> str:
    sec = max(1, int(round(float(seconds))))
    mins, rem = divmod(sec, 60)
    if mins > 0 and rem > 0:
        return f"{mins}m {rem:02d}s"
    if mins > 0:
        return f"{mins}m 00s"
    return f"{rem}s"


def compute_director_duration_recommendation(
    lib_info: dict[str, Any],
    user_prompt: str = "",
) -> dict[str, Any]:
    """Executive Director's algorithmic runtime recommendation based on library volume and narrative density."""
    v_count = int(lib_info.get("video_count", 0))
    p_count = int(lib_info.get("photo_count", 0))
    theme = str(lib_info.get("theme", "family_travel"))

    # Estimate usable A-roll/B-roll video spine duration (~11s per raw video clip on average)
    est_video_sec = min(v_count * 11.5, 210.0)
    # Estimate distinct photo burst clusters (~1 distinct scene per 8-10 burst shots in large albums, or 1 per 3 in small)
    if p_count > 120:
        distinct_photo_clusters = max(18, min(55, int(round(p_count / 9.5))))
    elif p_count > 35:
        distinct_photo_clusters = max(12, int(round(p_count / 3.2)))
    else:
        distinct_photo_clusters = max(6, p_count)

    # Curated hero plates for the Director's Cut (~75% of distinct clusters, at ~4.2s per 2.5D parallax plate)
    recommended_plates = max(10, min(48, int(round(distinct_photo_clusters * 0.75))))
    raw_rec_sec = est_video_sec + (recommended_plates * 4.1)
    # Round to nearest 5s, minimum 75s, up to 420s (7 mins) naturally
    rec_sec = int(round(max(75.0, min(420.0, raw_rec_sec)) / 5.0) * 5)

    # Check if user explicitly asked for a specific duration in user_prompt
    prompt_low = (user_prompt or "").lower()
    explicit_sec: Optional[int] = None
    m_min = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes|分钟)", prompt_low)
    m_sec = re.search(r"\b(\d{2,3})\s*(?:sec|secs|seconds|s|秒)\b", prompt_low)
    if m_min:
        explicit_sec = int(round(float(m_min.group(1)) * 60))
    elif m_sec:
        explicit_sec = int(m_sec.group(1))
    elif any(k in prompt_low for k in ["longer", "extended", "long cut", "full documentary", "更长"]):
        explicit_sec = int(round((rec_sec * 1.45) / 5.0) * 5)

    if explicit_sec and 30 <= explicit_sec <= 600:
        rec_sec = explicit_sec
        recommended_plates = max(8, min(60, int(round(max(20.0, rec_sec - est_video_sec) / 3.5))))

    ext_sec = int(round(max(rec_sec + 45, rec_sec * 1.4) / 5.0) * 5)
    fast_sec = 75

    rec_mmss = _format_mmss(rec_sec)
    ext_mmss = _format_mmss(ext_sec)

    if theme == "louvre_paris":
        rationale = (
            f"Director's Assessment: With {v_count} 4K video clips and {p_count} high-res photos (~{distinct_photo_clusters} "
            f"distinct museum gallery clusters), a 2-minute cap would rush the exhibition plaques or drop iconic artworks. "
            f"I recommend {rec_sec}s ({rec_mmss}) to showcase ~{recommended_plates} curated 2.5D parallax plates & diptychs "
            f"at a breathable 4.0–4.5s museum cadence alongside your 4 video clips (or {ext_sec}s / {ext_mmss} for an Extended Exhibition Cut)."
        )
    else:
        rationale = (
            f"Director's Assessment: Based on {v_count} videos (~{int(est_video_sec)}s usable motion spine) and "
            f"{p_count} photos (~{distinct_photo_clusters} distinct scene clusters), I recommend {rec_sec}s ({rec_mmss}) "
            f"so all spoken dialogue and ~{recommended_plates} hero photo plates breathe naturally without artificial 2-minute truncation "
            f"(or choose {ext_sec}s / {ext_mmss} for an Extended Family Documentary)."
        )

    return {
        "director_recommended_duration_sec": rec_sec,
        "director_recommended_mmss": rec_mmss,
        "extended_duration_sec": ext_sec,
        "extended_mmss": ext_mmss,
        "fast_duration_sec": fast_sec,
        "recommended_photo_plates": recommended_plates,
        "distinct_photo_clusters": distinct_photo_clusters,
        "director_duration_rationale": rationale,
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

    rec_sec = info["director_recommended_duration_sec"]
    rec_mmss = info["director_recommended_mmss"]
    ext_sec = info["extended_duration_sec"]
    ext_mmss = info["extended_mmss"]
    plates = info["recommended_photo_plates"]

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
            question=f"5. Target Video Duration & Pacing (🎬 Director Recommends {rec_sec}s / {rec_mmss})",
            options=[
                f"{rec_sec} (🎬 Director's Recommendation: ~{rec_mmss} — {v_count} videos + ~{plates} curated 2.5D photo plates)",
                f"{ext_sec} (Extended Documentary Cut: ~{ext_mmss} — Deep gallery & story immersion, up to 45+ photos)",
                "120 (Standard 2-Minute Cut: ~2m 00s — Balanced 4-act family keepsake)",
                "75 (Fast Social Highlight Reel: ~1m 15s — Punchy beat-synced montage)",
            ],
            default_answer=str(rec_sec),
        ),
        InterviewQuestion(
            id="must_include_topics",
            question="6. Must-Include Highlights & Moments",
            options=must_options,
            default_answer=prompt_to_use,
        ),
    ]


def refine_intake_interactively(
    media_dir: str | Path,
    user_message: str = "",
    interview_answers: Optional[dict[str, str]] = None,
    chat_history: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """Run Client Intake Agent + Executive Director consultation live as the user types or chats in Step 2."""
    info = inspect_media_library(media_dir)
    answers = dict(interview_answers or {})
    raw_msg = (user_message or "").strip() or info["default_prompt"]
    msg_low = raw_msg.lower()

    dur_rec = compute_director_duration_recommendation(info, raw_msg)
    rec_sec = dur_rec["director_recommended_duration_sec"]
    ext_sec = dur_rec["extended_duration_sec"]

    # Detect interactive refinements from user's natural language input
    refinements_applied: list[str] = []

    # 1. Duration & Pacing refinement
    m_min = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes|分钟)", msg_low)
    m_sec = re.search(r"\b(\d{2,3})\s*(?:sec|secs|seconds|s|秒)\b", msg_low)
    if m_min:
        new_dur = max(30, min(600, int(round(float(m_min.group(1)) * 60))))
        answers["target_duration_sec"] = str(new_dur)
        refinements_applied.append(f"Target Duration -> {new_dur}s ({_format_mmss(new_dur)})")
    elif m_sec:
        new_dur = max(30, min(600, int(m_sec.group(1))))
        answers["target_duration_sec"] = str(new_dur)
        refinements_applied.append(f"Target Duration -> {new_dur}s ({_format_mmss(new_dur)})")
    elif any(k in msg_low for k in ["director's recommendation", "director recommendation", "director cut", "director's cut"]):
        answers["target_duration_sec"] = str(rec_sec)
        refinements_applied.append(f"Target Duration -> Director's Recommendation ({rec_sec}s / {_format_mmss(rec_sec)})")
    elif any(k in msg_low for k in ["extended", "longer", "long documentary", "more photos", "3 min", "4 min", "更长"]):
        answers["target_duration_sec"] = str(ext_sec)
        refinements_applied.append(f"Target Duration -> Extended Documentary ({ext_sec}s / {_format_mmss(ext_sec)})")
    elif any(k in msg_low for k in ["short", "fast reel", "punchy", "social reel", "quick"]):
        answers["target_duration_sec"] = "75"
        refinements_applied.append("Target Duration -> Fast Highlight Reel (75s / 1m 15s)")
    elif not answers.get("target_duration_sec"):
        answers["target_duration_sec"] = str(rec_sec)

    # 2. Narrative style refinement
    if "vlog" in msg_low or "chronological" in msg_low:
        answers["narrative_style"] = "day_in_the_life_vlog"
        refinements_applied.append("Story Arc -> Chronological Day-in-the-Life Vlog")
    elif "montage" in msg_low or "fast-paced" in msg_low:
        answers["narrative_style"] = "upbeat_montage"
        refinements_applied.append("Story Arc -> Beat-Driven Highlight Montage")
    elif "dialogue_story" in msg_low or "interview" in msg_low:
        answers["narrative_style"] = "dialogue_story"
        refinements_applied.append("Story Arc -> Dialogue-Driven A-Roll + B-Roll Cutaways")
    elif not answers.get("narrative_style"):
        answers["narrative_style"] = "family_keepsake_4act"

    # 3. Photo mode refinement
    if "full_bleed" in msg_low or "full screen" in msg_low or "full-bleed" in msg_low:
        answers["photo_mode"] = "full_bleed"
        refinements_applied.append("Photo Effects -> Full-Bleed 16:9 Ken Burns")
    elif "parallax" in msg_low or "diptych" in msg_low or "museum" in msg_low or "plaque" in msg_low:
        answers["photo_mode"] = "parallax_gallery"
        refinements_applied.append("Photo Effects -> 2.5D Parallax + Museum Matte + Diptych Split-Screen")
    elif not answers.get("photo_mode"):
        answers["photo_mode"] = "parallax_gallery"

    # 4. BGM mood refinement
    if any(k in msg_low for k in ["classical", "satie", "piano", "french", "chamber", "orchestral"]):
        answers["bgm_mood"] = "classical_chamber"
        refinements_applied.append("Soundtrack -> Classical Chamber Piano & Strings (Erik Satie / French Impressionist)")
    elif any(k in msg_low for k in ["acoustic", "guitar", "warm", "folk"]):
        answers["bgm_mood"] = "warm_acoustic"
        refinements_applied.append("Soundtrack -> Heartwarming Acoustic Guitar & Piano")
    elif not answers.get("bgm_mood"):
        answers["bgm_mood"] = "auto_match"

    # 5. Dialogue preference
    if "music only" in msg_low or "no dialogue" in msg_low or "mute speech" in msg_low:
        answers["dialogue_preference"] = "music_mostly"
        refinements_applied.append("Dialogue Mix -> Music-Focused (-20dB Ambient Nat-Sound)")
    elif not answers.get("dialogue_preference"):
        answers["dialogue_preference"] = "keep_full_sentences_with_broll_cutaways"

    # 6. Synthesize updated must_include_topics & draft CreativeBrief
    try:
        chosen_dur = float(str(answers.get("target_duration_sec", rec_sec)).split(" ")[0])
    except ValueError:
        chosen_dur = float(rec_sec)
    chosen_dur = max(30.0, min(600.0, chosen_dur))

    raw_must = answers.get("must_include_topics") or raw_msg
    must_topics = [
        p.strip()
        for p in re.split(r"[,;•\n]+|\band\b|\bplus\b|\bwith\b", raw_must)
        if len(p.strip()) > 3
    ][:10]
    if not must_topics:
        must_topics = ["Iconic Landmarks", "Family Portraits", "Candid Spoken Moments", "Cinematic B-Roll"]

    #Estimate how many curated photos will be included at chosen_dur
    est_video_sec = min(info["video_count"] * 11.5, 180.0)
    est_photo_plates = min(
        max(8, info["photo_count"]),
        max(10, min(60, int(round(max(18.0, chosen_dur - est_video_sec) / 3.3)))),
    )

    project_title = info["display_title"].replace(" ", "_")
    draft_brief = CreativeBrief(
        project_title=project_title,
        raw_user_prompt=raw_msg,
        narrative_style=str(answers["narrative_style"]).split(" ")[0],  # type: ignore[arg-type]
        tone="warm_joyful",
        target_duration_sec=chosen_dur,
        must_include_topics=must_topics,
        avoid_topics=["blurry pocket shots", "duplicate photo bursts", "long static dead air"],
        dialogue_preference=str(answers["dialogue_preference"]).split(" ")[0],  # type: ignore[arg-type]
        bilingual_overlay_captions=True,
        photo_mode=str(answers["photo_mode"]).split(" ")[0],  # type: ignore[arg-type]
        max_broll_duration_sec=3.5 if chosen_dur <= 75 else 4.5,
        max_consecutive_same_subject=3,
        nat_sound_db=-20.0,
        music_duck_db=-14.0,
    )

    chosen_mmss = _format_mmss(chosen_dur)
    changes_badge = (
        f"Applied refinements: {' • '.join(refinements_applied)}. "
        if refinements_applied
        else "Analyzed your story vision against the media library. "
    )

    agent_reply = (
        f"📋 **Client Intake Producer (`client_intake`)**: {changes_badge}"
        f"I've structured **`{draft_brief.project_title}`** as a `{draft_brief.narrative_style}` cut targeting "
        f"**{int(chosen_dur)}s ({chosen_mmss})**. I locked in **{len(must_topics)} must-have story beats** "
        f"({', '.join(must_topics[:4])}{'...' if len(must_topics) > 4 else ''}), configured **`{draft_brief.photo_mode}`** "
        f"with bilingual exhibition plaques, and budgeted **~{est_photo_plates} curated high-taste photo plates** "
        f"plus **{info['video_count']} video clips**."
    )

    director_note = (
        f"🎬 **Executive Director (`production_director`)**: {dur_rec['director_duration_rationale']} "
        f"At your current target of **{int(chosen_dur)}s ({chosen_mmss})**, I will instruct `photo_curator` to select "
        f"**~{est_photo_plates} top-scoring images** (filtering out burst duplicates from {info['photo_count']} raw photos), "
        f"pair vertical portraits into **Diptych Split-Screens**, and direct `music_supervisor` to lay down a "
        f"`{answers.get('bgm_mood', 'auto_match').split(' ')[0]}` score with -14dB dialogue ducking."
    )

    if info["theme"] == "louvre_paris":
        suggested_chips = [
            f"🎬 Use Director's Recommended Length ({rec_sec}s / {dur_rec['director_recommended_mmss']})",
            f"📽️ Make it an Extended Museum Documentary ({ext_sec}s / {dur_rec['extended_mmss']}, ~35+ artworks)",
            "🖼️ Prioritize Mona Lisa, Winged Victory, Venus de Milo & Family Portraits with Diptychs",
            "🎻 Use Erik Satie Gymnopédie No. 1 French Classical Piano Score",
            "⚡ Make a Fast 75s Social Highlight Reel",
        ]
    else:
        suggested_chips = [
            f"🎬 Use Director's Recommended Length ({rec_sec}s / {dur_rec['director_recommended_mmss']})",
            f"📽️ Make it an Extended Family Documentary ({ext_sec}s / {dur_rec['extended_mmss']})",
            "🗣️ Keep all candid kid & family spoken dialogue with B-roll cutaways",
            "🖼️ Enable 2.5D Parallax Gallery + Diptych Split-Screens & Bilingual Captions",
            "⚡ Make a Fast 75s Social Highlight Reel",
        ]

    return {
        "agent_reply": agent_reply,
        "director_note": director_note,
        "refinements_applied": refinements_applied,
        "updated_answers": answers,
        "draft_brief": draft_brief.model_dump(mode="json"),
        "estimated_photo_plates": est_photo_plates,
        "duration_recommendation": dur_rec,
        "suggested_chips": suggested_chips,
    }


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
        lib_info = inspect_media_library(blackboard.media_dir)
        rec_default_dur = str(lib_info["director_recommended_duration_sec"])
        blackboard.emit_activity(
            self.role_id,
            f"Scanning library ({lib_info['video_count']} videos, {lib_info['photo_count']} photos) & consulting Director duration target",
            detail=lib_info["director_duration_rationale"],
        )
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
            target_dur = float(_clean_choice("target_duration_sec", rec_default_dur))
        except ValueError:
            target_dur = float(rec_default_dur)
        target_dur = max(30.0, min(600.0, target_dur))

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
        blackboard.emit_activity(
            self.role_id,
            f"Locked CreativeBrief '{brief.project_title}' (target: {int(brief.target_duration_sec)}s / {_format_mmss(brief.target_duration_sec)}, photo_mode: {brief.photo_mode})",
            detail=f"Must-include topics: {', '.join(brief.must_include_topics[:6])}",
        )

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
