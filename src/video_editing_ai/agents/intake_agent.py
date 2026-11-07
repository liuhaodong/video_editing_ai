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
from ..models.schema import (
    VALID_PHOTO_MODES,
    VALID_PHOTO_OVERLAY_FX,
    CreativeBrief,
    InterviewQuestion,
)
from .base import BaseEditingAgent, PipelineBlackboard, register_agent

VALID_OVERLAY_FX = VALID_PHOTO_OVERLAY_FX


def _extract_option_code(raw_val: str) -> str:
    """Extract the canonical option token, stripping any leading '(Recommended)' prefix."""
    s = (raw_val or "").strip()
    s = re.sub(r"^\(Recommended\)\s*", "", s, flags=re.IGNORECASE).strip()
    if not s:
        return ""
    return s.split(" ")[0].split("—")[0].strip().rstrip(",;:")


def _extract_multi_codes(
    raw_val: str | list[str],
    valid_set: tuple[str, ...] | set[str],
    default_list: Optional[list[str]] = None,
) -> list[str]:
    """Parse comma/pipe/newline-separated or list multi-select answers into validated option codes."""
    if isinstance(raw_val, list):
        parts = [str(x) for x in raw_val]
    else:
        parts = re.split(r"[,;|\n]+", str(raw_val or ""))
    out: list[str] = []
    for p in parts:
        code = _extract_option_code(p)
        if code in valid_set and code not in out:
            out.append(code)
        else:
            # Also check if any valid token appears as a substring of the option label
            for v in valid_set:
                if re.search(rf"\b{re.escape(v)}\b", p) and v not in out:
                    out.append(v)
    if not out and default_list:
        return [c for c in default_list if c in valid_set]
    return out


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
    """
    Pre-scan the raw media folder (recursively) and generate interactive client discovery questions
    with explicit Director/Producer recommendations, rationales, and multi-choice (multi-select) support.
    """
    info = inspect_media_library(media_dir)
    v_count = info["video_count"]
    p_count = info["photo_count"]
    theme = info["theme"]
    cams = ", ".join(info.get("detected_cameras") or ["Multi-Cam"])
    prompt_to_use = initial_prompt.strip() or info["default_prompt"]

    media_note = f"({v_count} videos, {p_count} photos • {cams})" if (v_count or p_count) else ""
    if theme == "louvre_paris":
        must_options = [
            "Morning Parisian Croissants & Hotel Breakfast (Recommended — Warm Chapter I Opener)",
            "Seine River Sightseeing Cruise & Pont Alexandre III (Recommended — Chapter I Kinetic River Views)",
            "Climbing the Eiffel Tower & 1st Floor Geodesic Dome (Recommended — Chapter II Iconic Landmark)",
            "Reopened Cathédrale Notre-Dame de Paris Gothic Interior & Rose Window (Recommended — Chapter III Sacred Art)",
            "Montmartre Cobblestone Stairs & Basilique du Sacré-Cœur Summit (Recommended — Chapter IV Hilltop Panorama)",
            "4-Course Traditional French Bistro Dinner Finale (Recommended — Chapter IV Culinary Closing)",
            "Louvre Glass Pyramid, Mona Lisa, Winged Victory & Venus de Milo Masterpieces",
        ]
        rec_must = must_options[:6]
        must_rationale = (
            f"Detected {v_count} videos and {p_count} photos spanning a full Parisian day trip. "
            "Selecting all 6 chronological chapters creates a complete morning-to-night narrative arc."
        )
    elif theme == "halloween_park":
        must_options = [
            "Kids' Excited Spoken Mandarin Dialogue in Car & Rides (Recommended — Emotional A-Roll Spine)",
            "Spinning Rides & Action B-Roll Cutaways (Recommended — High-Energy Motion)",
            "Giant Pumpkin Pyramid & Maze Family Portraits (Recommended — Hero Keepsake Stills)",
            "Great Big Boo Live Stage Musical Show (Recommended — Musical Climax)",
            "Boo Alley Night Lights Finale (Recommended — Atmospheric Closing Chapter)",
        ]
        rec_must = must_options[:5]
        must_rationale = "Preserves both the candid family dialogue and the signature Halloween park visual spectacles."
    else:
        must_options = [
            "Iconic Landmarks & Architecture Establishing Shots (Recommended — Visual Anchors)",
            "Candid Family & Solo Portraits with Warm Expressions (Recommended — Emotional Heart)",
            "Spoken Camera Dialogue & Laughter with B-Roll Cutaways (Recommended — Narrative Spine)",
            "Local Food, Café & Cultural Moments (Recommended — Sensory Pacing Variety)",
            "Sunset / Evening Finale Walk (Recommended — Memorable Closing Act)",
        ]
        rec_must = must_options[:4]
        must_rationale = f"Balanced selection across your {v_count} videos and {p_count} photos."

    rec_sec = info["director_recommended_duration_sec"]
    rec_mmss = info["director_recommended_mmss"]
    ext_sec = info["extended_duration_sec"]
    ext_mmss = info["extended_mmss"]
    plates = info["recommended_photo_plates"]

    return [
        InterviewQuestion(
            id="target_duration_sec",
            question=f"1. Target Video Duration & Reading Pacing (🎬 Director Recommends {rec_sec}s / {rec_mmss})",
            options=[
                f"{rec_sec} (Recommended) — 🎬 Director's Cut (~{rec_mmss}): {v_count} videos + ~{plates} photo plates at comfortable 6s caption reading pace",
                f"{ext_sec} — Extended Documentary Cut (~{ext_mmss}): Deep gallery & story immersion with 45+ annotated scenes",
                "180 — Mid-Length 3-Minute Story (~3m 00s): Balanced 4-chapter highlight film",
                "120 — Standard 2-Minute Cut (~2m 00s): Brisk 4-act keepsake",
                "75 — Fast Social Highlight Reel (~1m 15s): Punchy beat-synced montage",
            ],
            default_answer=str(rec_sec),
            is_multi_select=False,
            recommended_options=[str(rec_sec)],
            recommendation_rationale=info["director_duration_rationale"],
        ),
        InterviewQuestion(
            id="narrative_style",
            question=f"2. Story Arc & Chapter Structure {media_note}",
            options=[
                "family_keepsake_4act (Recommended) — 4-Act Documentary Story with 3.5s Cold-Open Teaser Hook, Chapter Banners & Frosted Lower-Third Cards",
                "day_in_the_life_vlog — Chronological Morning-to-Night Travelogue (Breakfast -> Landmarks -> Dinner Finale)",
                "dialogue_story — Narrative A-Roll Driven with Track V2 B-Roll Cutaways & J/L Audio Cuts",
                "upbeat_montage — Fast-Paced Beat-Driven Musical Highlight Montage",
            ],
            default_answer="family_keepsake_4act",
            is_multi_select=False,
            recommended_options=["family_keepsake_4act"],
            recommendation_rationale=(
                "A 4-act structure with a Cold-Open Teaser Hook immediately grabs viewer attention in the first 3.5 seconds "
                "and separates distinct locations with clean chapter banners."
            ),
        ),
        InterviewQuestion(
            id="photo_effects_suite",
            question="3. Premium Photo Presentation Effects Suite (Multi-Select — Pick all styles to choreograph across chapters)",
            options=[
                "cinematic_depth_3d (Recommended) — 3-Plane Subject Depth-Mask Parallax & Bokeh Rack-Focus (Hero Landmarks & Sculptures)",
                "triptych_cinematic (Recommended) — 3-Panel Staggered Storyboard Triptych (3-Course French Meals & Architectural Sequences)",
                "polaroid_scrapbook (Recommended) — Tilted Vintage Instant-Film Polaroid Stack (Candid Selfies, Café & Keepsakes)",
                "curator_loupe (Recommended) — 1.65x Optical Magnifying Glass Spotlight with Gold Bezel (Cathedral Carvings, Menus & Artworks)",
                "editorial_magazine (Recommended) — Asymmetric 62/38 Luxury Architectural Monograph Spread (Chapter Openers)",
                "cinema_anamorphic (Recommended) — 2.39:1 CinemaScope Widescreen Letterbox with 35mm Highlight Halation (River & City Panoramas)",
                "parallax_gallery (Recommended) — 2.5D Counter-Panning Double Ivory/Gold Museum Frame",
                "diptych_frame (Recommended) — Side-by-Side Vertical Portrait Diptych Split-Screen",
                "gallery_frame — Classic Static Floating Gallery Frame",
                "full_bleed — Full-Screen 16:9 Crop with Classic Ken Burns Pan/Zoom",
            ],
            default_answer="cinematic_depth_3d,triptych_cinematic,polaroid_scrapbook,curator_loupe,editorial_magazine,cinema_anamorphic,parallax_gallery,diptych_frame",
            is_multi_select=True,
            recommended_options=[
                "cinematic_depth_3d",
                "triptych_cinematic",
                "polaroid_scrapbook",
                "curator_loupe",
                "editorial_magazine",
                "cinema_anamorphic",
                "parallax_gallery",
                "diptych_frame",
            ],
            recommendation_rationale=(
                "Combining these 8 cinema-grade modes lets the Photo Curator & VFX Designer match each photo's geometry "
                "(e.g., 3-panel Triptych for your 3-course French meal, Optical Loupe for Notre-Dame carvings, Polaroid Stack for selfies, "
                "and 3D Depth Parallax for the Eiffel Tower) so no two consecutive stills ever feel repetitive."
            ),
        ),
        InterviewQuestion(
            id="photo_overlay_finishers",
            question="4. Composable Atmospheric Light & Film Finishers (Multi-Select — Subtle optical polish over photo plates)",
            options=[
                "golden_light_leak (Recommended) — Warm Amber Optical Sun-Ray Corner Wash (Morning & Golden Hour Shots)",
                "museum_spotlight (Recommended) — Radial Gallery Track-Light Spotlight on Subject Focal Point (Interiors & Artworks)",
                "champagne_bokeh (Recommended) — Softly Drifting Golden Bokeh Orbs (River Cruises, Evening & Bistro Finale)",
                "nostalgic_film_grain (Recommended) — Subtle 35mm Kodak Portra Organic Luminance Grain (Polaroid & Street Keepsakes)",
                "none — Clean Unfiltered Digital Master Only",
            ],
            default_answer="golden_light_leak,museum_spotlight,champagne_bokeh,nostalgic_film_grain",
            is_multi_select=True,
            recommended_options=[
                "golden_light_leak",
                "museum_spotlight",
                "champagne_bokeh",
                "nostalgic_film_grain",
            ],
            recommendation_rationale=(
                "Blending subtle optical finishers (warm sun leaks on outdoor views, gallery spotlights on cathedral interiors, "
                "and champagne bokeh on evening dining) makes still photos feel alive like moving cinema footage."
            ),
        ),
        InterviewQuestion(
            id="photo_mode",
            question="5. Primary Anchor Photo Mode (Default fallback when a single mode is used)",
            options=[
                "cinematic_depth_3d (Recommended) — 3-Plane Subject Depth Parallax + Bokeh Defocus + Frosted Documentary Card",
                "parallax_gallery — 2.5D Counter-Panning Museum Matte Frame + Frosted Documentary Card",
                "editorial_magazine — Asymmetric 62/38 Luxury Magazine Monograph Spread",
                "polaroid_scrapbook — Vintage Instant-Film Polaroid Scrapbook",
                "cinema_anamorphic — 2.39:1 CinemaScope Widescreen Letterbox",
                "gallery_frame — Classic Floating Gallery Frame",
                "full_bleed — Full-Screen 16:9 Ken Burns",
            ],
            default_answer="cinematic_depth_3d",
            is_multi_select=False,
            recommended_options=["cinematic_depth_3d"],
            recommendation_rationale=(
                "Uses 3-plane depth parallax as the primary anchor while rotating through your multi-selected effects suite."
            ),
        ),
        InterviewQuestion(
            id="bgm_mood",
            question="6. Royalty-Free Soundtrack & Loudness Mastering (Stage 05 Dedicated BGM Agent)",
            options=[
                "auto_match (Recommended) — Multi-Movement Royalty-Free Suite (Accordion Waltz -> Satie Gymnopédie -> Debussy Chamber -> Bistro Jazz) Mastered to 100% Video Length at -14 LUFS",
                "classical_chamber — Elegant French Classical Piano, Harp & Strings Chamber Score (Erik Satie / Debussy)",
                "warm_acoustic — Heartwarming Acoustic Guitar & Piano Family Keepsake Score",
            ],
            default_answer="auto_match",
            is_multi_select=False,
            recommended_options=["auto_match"],
            recommendation_rationale=(
                "For videos over 2.5 minutes, a single track cuts off early or feels monotonous. "
                "The Dedicated BGM Agent chains and crossfades multiple royalty-free movements covering 100% of the runtime at broadcast -14.0 LUFS."
            ),
        ),
        InterviewQuestion(
            id="dialogue_preference",
            question="7. Spoken Camera Dialogue, Captions & Audio Mix",
            options=[
                "keep_full_sentences_with_broll_cutaways (Recommended) — Frosted-Glass Educational Captions + Keep Spoken Sentences + -20dB Nat-Sound",
                "highlights_only — Keep Only Short Candid Quotes + Vibrant Full-Volume BGM",
                "music_mostly — Solo/Silent Travel Mode: Full-Volume -14 LUFS Soundtrack + Educational Frosted Lower-Third Cards (Zero False Ducking)",
            ],
            default_answer="keep_full_sentences_with_broll_cutaways",
            is_multi_select=False,
            recommended_options=["keep_full_sentences_with_broll_cutaways"],
            recommendation_rationale=(
                "Automatically keeps real spoken dialogue with -14dB BGM ducking while ensuring silent educational captions never dip the music volume."
            ),
        ),
        InterviewQuestion(
            id="must_include_topics",
            question="8. Must-Include Chapters & Highlights (Multi-Select — Choose all story beats to feature)",
            options=must_options,
            default_answer=" | ".join(rec_must) if rec_must else prompt_to_use,
            is_multi_select=True,
            recommended_options=rec_must,
            recommendation_rationale=must_rationale,
        ),
    ]


def build_ask_question_tool_payload(questions: list[InterviewQuestion], max_questions: int = 4) -> list[dict[str, Any]]:
    """
    Format top discovery questions into the exact schema required by Antigravity's native `ask_question` tool:
    - `question`: string (with concise Director recommendation context)
    - `options`: list of unnumbered strings with recommended option(s) first, prefixed with '(Recommended) '
    - `is_multi_select`: bool
    """
    priority_ids = ["target_duration_sec", "photo_effects_suite", "bgm_mood", "must_include_topics"]
    by_id = {q.id: q for q in questions}
    ordered = [by_id[qid] for qid in priority_ids if qid in by_id] + [
        q for q in questions if q.id not in priority_ids
    ]

    payload: list[dict[str, Any]] = []
    for q in ordered[:max_questions]:
        clean_q = re.sub(r"^\d+\.\s*", "", q.question).strip()
        if q.recommendation_rationale:
            short_rat = q.recommendation_rationale.split(".")[0].strip() + "."
            clean_q = f"{clean_q} — {short_rat}"

        formatted_opts: list[str] = []
        rec_set = set(q.recommended_options or [])
        for opt in q.options:
            code = _extract_option_code(opt)
            is_rec = ("(Recommended" in opt) or (code in rec_set) or (opt in rec_set)
            # Strip inline '(Recommended)' so we can place '(Recommended) ' cleanly at the very front
            stripped = re.sub(r"\s*\(Recommended[^)]*\)", "", opt).strip()
            if is_rec:
                formatted_opts.append(f"(Recommended) {stripped}")
            else:
                formatted_opts.append(stripped)

        payload.append(
            {
                "id": q.id,
                "question": clean_q,
                "options": formatted_opts,
                "is_multi_select": bool(q.is_multi_select),
            }
        )
    return payload


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
    questions = generate_intake_questions(media_dir, raw_msg)
    q_by_id = {q.id: q for q in questions}

    # Detect interactive refinements from user's natural language input
    refinements_applied: list[str] = []

    if any(
        k in msg_low
        for k in [
            "apply all director recommendations",
            "accept all recommendations",
            "use all director recommendations",
        ]
    ):
        for q in questions:
            if q.is_multi_select and q.recommended_options:
                answers[q.id] = ", ".join(q.recommended_options)
            elif q.recommended_options:
                answers[q.id] = q.recommended_options[0]
            else:
                answers[q.id] = _extract_option_code(q.default_answer)
        refinements_applied.append("Applied All Director & Producer Recommendations Across All 8 Questions")

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

    # 3. Primary Photo mode & Multi-Select Photo Effects Suite refinement
    if "full_bleed" in msg_low or "full screen" in msg_low or "full-bleed" in msg_low:
        answers["photo_mode"] = "full_bleed"
        refinements_applied.append("Primary Photo Mode -> Full-Bleed 16:9 Ken Burns")
    elif "3d depth" in msg_low or "cinematic_depth_3d" in msg_low:
        answers["photo_mode"] = "cinematic_depth_3d"
        refinements_applied.append("Primary Photo Mode -> 3D Depth-Map Parallax + Bokeh")
    elif "editorial_magazine" in msg_low or "editorial magazine" in msg_low:
        answers["photo_mode"] = "editorial_magazine"
        refinements_applied.append("Primary Photo Mode -> Vogue/Kinfolk Editorial Spread")
    elif "polaroid" in msg_low or "scrapbook" in msg_low:
        answers["photo_mode"] = "polaroid_scrapbook"
        refinements_applied.append("Primary Photo Mode -> Vintage Polaroid Instant Print")
    elif "curator_loupe" in msg_low or "loupe" in msg_low:
        answers["photo_mode"] = "curator_loupe"
        refinements_applied.append("Primary Photo Mode -> Curator Magnifying Loupe")
    elif "parallax" in msg_low or "diptych" in msg_low or "museum" in msg_low or "plaque" in msg_low:
        answers["photo_mode"] = "parallax_gallery"
        refinements_applied.append("Primary Photo Mode -> 2.5D Parallax + Museum Matte + Diptych Split-Screen")
    elif not answers.get("photo_mode"):
        answers["photo_mode"] = _extract_option_code(q_by_id["photo_mode"].default_answer) if "photo_mode" in q_by_id else "parallax_gallery"

    default_fx_suite = q_by_id["photo_effects_suite"].recommended_options if "photo_effects_suite" in q_by_id else [
        "cinematic_depth_3d",
        "parallax_gallery",
        "editorial_magazine",
        "diptych_frame",
        "triptych_cinematic",
        "polaroid_scrapbook",
    ]
    enabled_modes = _extract_multi_codes(
        answers.get("photo_effects_suite", ""),
        VALID_PHOTO_MODES,
        default_fx_suite,
    )
    answers["photo_effects_suite"] = ", ".join(enabled_modes)

    default_finishers = q_by_id["photo_overlay_finishers"].recommended_options if "photo_overlay_finishers" in q_by_id else [
        "golden_light_leak",
        "champagne_bokeh",
        "museum_spotlight",
        "nostalgic_film_grain",
    ]
    enabled_overlays = _extract_multi_codes(
        answers.get("photo_overlay_finishers", ""),
        VALID_OVERLAY_FX,
        default_finishers,
    )
    answers["photo_overlay_finishers"] = ", ".join(enabled_overlays)

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
        chosen_dur = float(_extract_option_code(str(answers.get("target_duration_sec", rec_sec))))
    except ValueError:
        chosen_dur = float(rec_sec)
    chosen_dur = max(30.0, min(600.0, chosen_dur))

    raw_must = answers.get("must_include_topics") or raw_msg
    cleaned_must_raw = re.sub(r"\(Recommended[^)]*\)", "", raw_must)
    must_topics = [
        p.strip()
        for p in re.split(r"[,;|•\n]+|\band\b|\bplus\b|\bwith\b", cleaned_must_raw)
        if len(p.strip()) > 3
    ][:10]
    if not must_topics:
        must_topics = ["Iconic Landmarks", "Family Portraits", "Candid Spoken Moments", "Cinematic B-Roll"]

    # Estimate how many curated photos will be included at chosen_dur
    est_video_sec = min(info["video_count"] * 11.5, 180.0)
    est_photo_plates = min(
        max(8, info["photo_count"]),
        max(10, min(60, int(round(max(18.0, chosen_dur - est_video_sec) / 3.3)))),
    )

    primary_photo_mode = _extract_option_code(str(answers["photo_mode"]))
    if primary_photo_mode not in VALID_PHOTO_MODES:
        primary_photo_mode = "parallax_gallery"

    director_recommendations_summary = {
        "recommended_duration_sec": rec_sec,
        "recommended_duration_mmss": dur_rec["director_recommended_mmss"],
        "duration_rationale": dur_rec["director_duration_rationale"],
        "recommended_photo_modes": default_fx_suite,
        "recommended_overlay_fx": default_finishers,
        "question_recommendations": {
            q.id: {
                "is_multi_select": q.is_multi_select,
                "recommended_options": q.recommended_options,
                "rationale": q.recommendation_rationale,
            }
            for q in questions
        },
    }

    project_title = info["display_title"].replace(" ", "_")
    draft_brief = CreativeBrief(
        project_title=project_title,
        raw_user_prompt=raw_msg,
        narrative_style=_extract_option_code(str(answers["narrative_style"])),  # type: ignore[arg-type]
        tone="warm_joyful",
        target_duration_sec=chosen_dur,
        must_include_topics=must_topics,
        avoid_topics=["blurry pocket shots", "duplicate photo bursts", "long static dead air"],
        dialogue_preference=_extract_option_code(str(answers["dialogue_preference"])),  # type: ignore[arg-type]
        bilingual_overlay_captions=True,
        photo_mode=primary_photo_mode,  # type: ignore[arg-type]
        enabled_photo_modes=enabled_modes,  # type: ignore[arg-type]
        enabled_overlay_fx=enabled_overlays,  # type: ignore[arg-type]
        max_broll_duration_sec=3.5 if chosen_dur <= 75 else 4.5,
        max_consecutive_same_subject=3,
        nat_sound_db=-20.0,
        music_duck_db=-14.0,
        director_recommendations=director_recommendations_summary,
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
        f"({', '.join(must_topics[:4])}{'...' if len(must_topics) > 4 else ''}), enabled **{len(enabled_modes)} photo effects** "
        f"(`{', '.join(enabled_modes[:4])}{'...' if len(enabled_modes) > 4 else ''}`) with **{len(enabled_overlays)} atmospheric finishers**, "
        f"and budgeted **~{est_photo_plates} curated high-taste photo plates** plus **{info['video_count']} video clips**."
    )

    director_note = (
        f"🎬 **Executive Director (`production_director`)**: {dur_rec['director_duration_rationale']} "
        f"At your current target of **{int(chosen_dur)}s ({chosen_mmss})**, I will instruct `photo_curator` to select "
        f"**~{est_photo_plates} top-scoring images** (filtering out burst duplicates from {info['photo_count']} raw photos), "
        f"rotate across your **{len(enabled_modes)} active photo treatments** (`{', '.join(enabled_modes[:4])}`), and direct `music_supervisor` to lay down a "
        f"`{_extract_option_code(str(answers.get('bgm_mood', 'auto_match')))}` score with -14dB dialogue ducking."
    )

    if info["theme"] == "louvre_paris":
        suggested_chips = [
            "✨ Apply All Director Recommendations",
            f"🎬 Use Director's Recommended Length ({rec_sec}s / {dur_rec['director_recommended_mmss']})",
            f"📽️ Make it an Extended Museum Documentary ({ext_sec}s / {dur_rec['extended_mmss']}, ~35+ artworks)",
            "🖼️ Enable 3D Depth Parallax + Curator Loupe + Editorial Magazine + Diptychs",
            "🎻 Use Erik Satie Gymnopédie No. 1 French Classical Piano Score",
        ]
    else:
        suggested_chips = [
            "✨ Apply All Director Recommendations",
            f"🎬 Use Director's Recommended Length ({rec_sec}s / {dur_rec['director_recommended_mmss']})",
            f"📽️ Make it an Extended Family Documentary ({ext_sec}s / {dur_rec['extended_mmss']})",
            "🗣️ Keep all candid kid & family spoken dialogue with B-roll cutaways",
            "🖼️ Enable 3D Depth Parallax + Polaroid Scrapbook + Diptych Split-Screens",
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
        "ask_question_payload": build_ask_question_tool_payload(questions),
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
        q_by_id = {q.id: q for q in questions}
        answers = blackboard.interview_answers

        def _clean_choice(key: str, default: str) -> str:
            raw = answers.get(key, default).strip()
            return _extract_option_code(raw) if raw else default

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

        default_primary_mode = _extract_option_code(q_by_id["photo_mode"].default_answer) if "photo_mode" in q_by_id else "parallax_gallery"
        photo_mode = _clean_choice("photo_mode", default_primary_mode)
        if photo_mode not in VALID_PHOTO_MODES:
            photo_mode = "parallax_gallery"

        default_fx_suite = q_by_id["photo_effects_suite"].recommended_options if "photo_effects_suite" in q_by_id else [
            "cinematic_depth_3d",
            "parallax_gallery",
            "editorial_magazine",
            "diptych_frame",
            "triptych_cinematic",
            "polaroid_scrapbook",
        ]
        enabled_modes = _extract_multi_codes(
            answers.get("photo_effects_suite", ""),
            VALID_PHOTO_MODES,
            default_fx_suite,
        )

        default_finishers = q_by_id["photo_overlay_finishers"].recommended_options if "photo_overlay_finishers" in q_by_id else [
            "golden_light_leak",
            "champagne_bokeh",
            "museum_spotlight",
            "nostalgic_film_grain",
        ]
        enabled_overlays = _extract_multi_codes(
            answers.get("photo_overlay_finishers", ""),
            VALID_OVERLAY_FX,
            default_finishers,
        )

        try:
            target_dur = float(_clean_choice("target_duration_sec", rec_default_dur))
        except ValueError:
            target_dur = float(rec_default_dur)
        target_dur = max(30.0, min(600.0, target_dur))

        # Extract must-include topics from user_prompt and multi-select answers
        raw_must = answers.get("must_include_topics") or blackboard.user_prompt
        must_topics: list[str] = []
        if raw_must:
            cleaned_must = re.sub(r"\(Recommended[^)]*\)", "", raw_must)
            parts = [p.strip() for p in re.split(r"[,;|•\n]+|\band\b", cleaned_must) if len(p.strip()) > 2]
            must_topics = parts[:10]
        if not must_topics:
            must_topics = ["spoken dialogue", "rides", "pumpkin pyramid", "stage show"]

        # Derive project title from user_prompt
        project_title = "Family_Travel_Story"
        prompt_low = blackboard.user_prompt.lower()
        if "louvre" in prompt_low or "paris" in prompt_low:
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
                "is_multi_select": q.is_multi_select,
                "recommended_options": q.recommended_options,
                "recommendation_rationale": q.recommendation_rationale,
            }
            for q in questions
        ]

        director_recommendations_summary = {
            "recommended_duration_sec": lib_info["director_recommended_duration_sec"],
            "recommended_duration_mmss": lib_info["director_recommended_mmss"],
            "duration_rationale": lib_info["director_duration_rationale"],
            "recommended_photo_modes": default_fx_suite,
            "recommended_overlay_fx": default_finishers,
            "question_recommendations": {
                q.id: {
                    "is_multi_select": q.is_multi_select,
                    "recommended_options": q.recommended_options,
                    "rationale": q.recommendation_rationale,
                }
                for q in questions
            },
        }

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
            enabled_photo_modes=enabled_modes,  # type: ignore[arg-type]
            enabled_overlay_fx=enabled_overlays,  # type: ignore[arg-type]
            max_broll_duration_sec=3.5 if target_dur <= 65 else 4.5,
            max_consecutive_same_subject=3,
            nat_sound_db=-20.0,
            music_duck_db=-14.0,
            enabled_optional_agents=(
                [a.strip() for a in answers.get("enabled_optional_agents", "").split(",") if a.strip()]
            ),
            interview_qa=qa_records,
            director_recommendations=director_recommendations_summary,
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
            f"Locked CreativeBrief '{brief.project_title}' (target: {int(brief.target_duration_sec)}s / {_format_mmss(brief.target_duration_sec)}, {len(brief.enabled_photo_modes)} photo modes)",
            detail=f"Must-include topics: {', '.join(brief.must_include_topics[:6])} | Photo modes: {', '.join(brief.enabled_photo_modes[:5])}",
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
            recipient_role="photo_curator",
            message_type="directive",
            subject="Apply client-selected multi-choice photo effects & atmospheric finishers",
            payload={
                "photo_mode": brief.photo_mode,
                "enabled_photo_modes": brief.enabled_photo_modes,
                "enabled_overlay_fx": brief.enabled_overlay_fx,
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
