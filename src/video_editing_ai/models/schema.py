"""Pydantic schemas for MediaCatalog, EditTimeline, and the Modular Multi-Agent JSON Contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional
import yaml
from pydantic import BaseModel, Field

PhotoModeType = Literal[
    "gallery_frame",
    "parallax_gallery",
    "diptych_frame",
    "full_bleed",
    "cinematic_depth_3d",
    "triptych_cinematic",
    "polaroid_scrapbook",
    "curator_loupe",
    "editorial_magazine",
    "cinema_anamorphic",
]

PhotoOverlayFxType = Literal[
    "none",
    "golden_light_leak",
    "champagne_bokeh",
    "museum_spotlight",
    "nostalgic_film_grain",
]

VALID_PHOTO_MODES: tuple[str, ...] = (
    "gallery_frame",
    "parallax_gallery",
    "diptych_frame",
    "full_bleed",
    "cinematic_depth_3d",
    "triptych_cinematic",
    "polaroid_scrapbook",
    "curator_loupe",
    "editorial_magazine",
    "cinema_anamorphic",
)

VALID_PHOTO_OVERLAY_FX: tuple[str, ...] = (
    "none",
    "golden_light_leak",
    "champagne_bokeh",
    "museum_spotlight",
    "nostalgic_film_grain",
)



# ============================================================================
# Step 1: Media Analysis & Annotation Schemas (annotations.yaml)
# ============================================================================


class DialogueSegment(BaseModel):
    """Timestamped spoken dialogue or song lyrics extracted from a video clip."""

    start_time: float = Field(..., description="Start timestamp in seconds within the source video")
    end_time: float = Field(..., description="End timestamp in seconds within the source video")
    speaker: str = Field(default="Speaker", description="Speaker label or role")
    text: str = Field(..., description="Transcribed spoken text or lyrics")
    translation: Optional[str] = Field(default=None, description="Optional English/bilingual translation")
    confidence: float = Field(default=0.9, description="ASR confidence score (0.0 to 1.0)")


class SceneSegment(BaseModel):
    """Timestamped visual scene/shot within a video clip."""

    scene_id: str = Field(..., description="Unique scene identifier within the video")
    start_time: float = Field(..., description="Scene start time in seconds")
    end_time: float = Field(..., description="Scene end time in seconds")
    summary: str = Field(..., description="What is happening visually in this scene")
    visual_tags: list[str] = Field(default_factory=list, description="Visual objects, actions, landmarks")
    camera_motion: str = Field(default="static", description="static, pan, tilt, tracking, handheld")
    emotion: str = Field(default="neutral", description="joyful, excited, calm, nostalgic, energetic, etc.")
    quality_score: float = Field(default=8.0, description="Visual quality & storytelling value (1-10)")
    keyframe_path: Optional[str] = Field(default=None, description="Path to extracted representative JPEG frame")


class VideoAsset(BaseModel):
    """Technical metadata and semantic annotations for a single video file."""

    id: str = Field(..., description="Unique asset ID (e.g., vid_IMG_0219)")
    path: str = Field(..., description="Path to the source video file")
    filename: str = Field(..., description="Base filename")
    duration: float = Field(..., description="Total duration in seconds")
    width: int = Field(default=1920)
    height: int = Field(default=1080)
    fps: float = Field(default=30.0)
    codec: str = Field(default="h264")
    color_transfer: str = Field(default="bt709", description="e.g., arib-std-b67 (HLG), smpte2084 (PQ), bt709")
    color_space: str = Field(default="bt709")
    has_audio: bool = Field(default=True)
    creation_time: Optional[str] = Field(default=None)
    overall_summary: str = Field(default="", description="High-level summary of the clip and its story value")
    scenes: list[SceneSegment] = Field(default_factory=list)
    dialogues: list[DialogueSegment] = Field(default_factory=list)


class PhotoAsset(BaseModel):
    """Technical metadata and semantic annotations for a single photo file (.HEIC, .JPG, .PNG)."""

    id: str = Field(..., description="Unique asset ID (e.g., img_IMG_0222)")
    path: str = Field(..., description="Path to the original photo file")
    preview_path: Optional[str] = Field(default=None, description="Path to converted RGB JPEG preview")
    filename: str = Field(..., description="Base filename")
    width: int = Field(default=1920)
    height: int = Field(default=1080)
    orientation: Literal["landscape", "portrait", "square"] = Field(default="landscape")
    creation_time: Optional[str] = Field(default=None, description="EXIF timestamp if available")
    caption: str = Field(default="", description="Detailed description of what is in the photo")
    subjects: list[str] = Field(default_factory=list, description="Main people, objects, or landmarks")
    mood: str = Field(default="warm", description="Emotional tone of the photo")
    focal_point: tuple[float, float] = Field(
        default=(0.5, 0.5),
        description="Normalized (x, y) center of interest in [0.0, 1.0] for Ken Burns focus",
    )
    recommended_effect: str = Field(
        default="gallery_ken_burns",
        description="Recommended photo treatment (e.g., gallery_ken_burns, zoom_in, pan_right, zoom_out)",
    )
    quality_score: float = Field(default=8.0, description="Composition & story quality score (1-10)")


class AudioAsset(BaseModel):
    """Background music or standalone voiceover audio asset."""

    id: str
    path: str
    filename: str
    duration: float
    role: Literal["music", "voiceover", "sfx"] = "music"
    description: str = ""


class MediaCatalog(BaseModel):
    """Complete Step 1 output containing all analyzed videos, photos, and audio files."""

    source_directory: str
    created_at: str
    contact_sheet_paths: list[str] = Field(default_factory=list)
    videos: list[VideoAsset] = Field(default_factory=list)
    photos: list[PhotoAsset] = Field(default_factory=list)
    audio_files: list[AudioAsset] = Field(default_factory=list)

    def to_yaml(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                f,
                allow_unicode=True,
                sort_keys=False,
                width=110,
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MediaCatalog":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# ============================================================================
# Modular Multi-Agent JSON Contracts
# ============================================================================


class InterviewQuestion(BaseModel):
    """A discovery question generated by Agent 0 (ClientIntakeAgent) for the client."""

    id: str
    question: str
    options: list[str] = Field(default_factory=list)
    default_answer: str = ""
    is_multi_select: bool = Field(
        default=False,
        description="Whether the client can select multiple options simultaneously (checkboxes / multi-choice)",
    )
    recommended_options: list[str] = Field(
        default_factory=list,
        description="Option code(s) recommended by the Intake Agent & Executive Director",
    )
    recommendation_rationale: str = Field(
        default="",
        description="Why the Intake Agent & Executive Director recommend these option(s) for this media library",
    )


class CreativeBrief(BaseModel):
    """Agent 0 Output (creative_brief.json): North-Star client requirements guiding all downstream agents."""

    project_title: str = Field(default="Family_Travel_Vlog")
    raw_user_prompt: str = Field(default="")
    narrative_style: Literal[
        "family_keepsake_4act",
        "day_in_the_life_vlog",
        "upbeat_montage",
        "dialogue_story",
    ] = Field(default="family_keepsake_4act")
    tone: Literal["warm_joyful", "energetic_fun", "cinematic_nostalgic", "playful_comedy"] = Field(
        default="warm_joyful"
    )
    target_duration_sec: float = Field(default=110.0, gt=5.0)
    must_include_topics: list[str] = Field(default_factory=list)
    avoid_topics: list[str] = Field(default_factory=lambda: ["blurry pocket shots", "duplicate photo bursts"])
    dialogue_preference: Literal[
        "keep_full_sentences_with_broll_cutaways",
        "highlights_only",
        "music_mostly",
    ] = Field(default="keep_full_sentences_with_broll_cutaways")
    bilingual_overlay_captions: bool = Field(default=True)
    photo_mode: PhotoModeType = Field(
        default="parallax_gallery"
    )
    enabled_photo_modes: list[PhotoModeType] = Field(
        default_factory=lambda: [
            "cinematic_depth_3d",
            "triptych_cinematic",
            "polaroid_scrapbook",
            "curator_loupe",
            "editorial_magazine",
            "parallax_gallery",
            "diptych_frame",
        ],
        description="Multi-selected photo presentation modes choreographed across chapters",
    )
    enabled_overlay_fx: list[PhotoOverlayFxType] = Field(
        default_factory=lambda: [
            "golden_light_leak",
            "museum_spotlight",
            "champagne_bokeh",
            "nostalgic_film_grain",
        ],
        description="Multi-selected atmospheric photo finishers",
    )
    director_recommendations: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured Intake Agent & Executive Director recommendations and rationales",
    )
    max_broll_duration_sec: float = Field(default=4.5, ge=1.5, le=12.0)
    max_consecutive_same_subject: int = Field(default=3, ge=1, le=10)
    nat_sound_db: float = Field(default=-20.0, description="Ambient B-roll natural sound level in dB (-18 to -24)")
    music_duck_db: float = Field(default=-14.0, description="Music attenuation in dB during A-roll dialogue")
    enabled_optional_agents: list[str] = Field(
        default_factory=list,
        description="Optional plugin agent role_ids enabled by the user (e.g., ['vfx_designer'])",
    )
    cold_open_hook: bool = Field(
        default=True,
        description="Start film with a 3-4s Cold-Open Teaser Hook before the Main Opening Title Card",
    )
    interview_qa: list[dict[str, Any]] = Field(default_factory=list)

    def to_json_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "CreativeBrief":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class WordTimestamp(BaseModel):
    """Word-level or phrase-level timestamp within a video clip."""

    word: str
    start: float
    end: float
    confidence: float = 0.9


class CulledSubClip(BaseModel):
    """A culled sub-clip or photo select in media_index.json produced by Agent 1 (CullerAgent)."""

    subclip_id: str
    source_asset_id: str
    source_path: str
    filename: str
    media_type: Literal["video", "photo"]
    normalized_timestamp: str = Field(..., description="ISO-8601 timestamp normalized across cameras")
    in_tc: float = Field(default=0.0, ge=0.0)
    out_tc: float = Field(default=3.0, gt=0.0)
    duration: float = Field(default=3.0, gt=0.0)
    roll_type: Literal["a_roll", "b_roll", "still"] = Field(default="b_roll")
    shot_scale: Literal["wide", "medium", "closeup"] = Field(default="medium")
    camera_movement: Literal["static", "pan_left", "pan_right", "tilt_up", "walk_forward"] = Field(default="static")
    orientation: Literal["landscape", "portrait", "square"] = Field(default="landscape")
    focal_point: tuple[float, float] = Field(default=(0.5, 0.5))
    people_present: list[str] = Field(default_factory=list)
    summary: str = Field(default="")
    spoken_text: str = Field(default="")
    translation: Optional[str] = Field(default=None)
    emotion_score: float = Field(default=8.0, ge=1.0, le=10.0)
    action_peak_tc: float = Field(default=1.5, ge=0.0)
    aesthetic_score: float = Field(default=8.0, ge=1.0, le=10.0)
    brief_match_boost: float = Field(default=0.0, description="Score boost from matching CreativeBrief must-haves")
    annotation_title: Optional[str] = Field(default=None, description="Curated subject/artwork/landmark title")
    annotation_subtitle: Optional[str] = Field(default=None, description="Bilingual curatorial subtitle")
    camera_badge: Optional[str] = Field(default=None, description="Camera & location EXIF badge")
    subject_category: Optional[str] = Field(default=None, description="Curatorial subject category")
    recommended_photo_mode: PhotoModeType = Field(
        default="parallax_gallery",
        description="Curated photo presentation mode (parallax_gallery, cinematic_depth_3d, triptych_cinematic, polaroid_scrapbook, curator_loupe, editorial_magazine, cinema_anamorphic, diptych_frame, etc.)",
    )
    overlay_fx: PhotoOverlayFxType = Field(
        default="none",
        description="Optional composable atmospheric finisher (golden_light_leak, champagne_bokeh, museum_spotlight, nostalgic_film_grain)",
    )
    secondary_source_path: Optional[str] = Field(
        default=None, description="Optional paired second photo path for diptych_frame / polaroid_scrapbook / triptych_cinematic layout"
    )
    tertiary_source_path: Optional[str] = Field(
        default=None, description="Optional third photo path for 3-panel triptych_cinematic layout"
    )
    recommended_transition: Literal["none", "cut", "crossfade", "dissolve", "fade_black", "dip_to_white"] = Field(
        default="crossfade",
        description="AI-recommended transition style based on narrative and visual continuity",
    )
    ai_rationale: str = Field(
        default="",
        description="AI Agent creative rationale for selecting/trimming/styling this clip or photo",
    )
    is_discarded: bool = Field(default=False)
    discard_reason: Optional[str] = Field(default=None)
    words: list[WordTimestamp] = Field(default_factory=list)


class BGMCandidate(BaseModel):
    """A candidate royalty-free / Creative Commons BGM track evaluated by MusicSupervisorAgent."""

    track_id: str
    title: str
    composer_or_artist: str
    genre_mood: str
    license_info: str = "CC0 / Public Domain / Royalty-Free"
    source_url: str = ""
    local_path: str = ""
    tempo_bpm_estimate: float = 96.0
    mood_match_score: float = Field(default=9.0, ge=0.0, le=10.0)
    notes: str = ""


class BGMSelectionReport(BaseModel):
    """Stage 05 Output (bgm_selection.json): Music Supervisor's BGM evaluation & download report."""

    project_title: str
    target_mood_profile: str
    selected_track_path: str
    track_title: str
    composer_or_artist: str
    license_info: str = "CC0 / Public Domain"
    source_url: str = ""
    mood_match_score: float = Field(default=9.5, ge=0.0, le=10.0)
    selection_rationale: str = ""
    rejected_user_bgm_reason: Optional[str] = None
    candidates: list[BGMCandidate] = Field(default_factory=list)

    def to_json_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BGMSelectionReport":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class CuratedPhotoItem(BaseModel):
    """A single photo evaluated and annotated by Stage 15 PhotoCuratorAgent."""

    subclip_id: str
    filename: str
    source_path: str
    orientation: Literal["landscape", "portrait", "square"] = "landscape"
    perceptual_hash: str = ""
    cluster_id: int = 0
    saliency_score: float = Field(default=8.0, ge=1.0, le=10.0)
    composition_score: float = Field(default=8.0, ge=1.0, le=10.0)
    portrait_warmth_score: float = Field(default=7.5, ge=1.0, le=10.0)
    art_architecture_score: float = Field(default=8.0, ge=1.0, le=10.0)
    overall_taste_score: float = Field(default=8.2, ge=1.0, le=10.0)
    is_selected: bool = False
    rejection_reason: Optional[str] = None
    subject_category: str = "landmark_architecture"
    annotation_title: str = ""
    annotation_subtitle: str = ""
    camera_badge: str = ""
    recommended_photo_mode: PhotoModeType = "parallax_gallery"
    overlay_fx: PhotoOverlayFxType = "none"
    paired_with_subclip_id: Optional[str] = None
    tertiary_subclip_id: Optional[str] = None
    curation_rationale: str = ""


class PhotoCurationReport(BaseModel):
    """Stage 15 Output (photo_curation.json): Aesthetic Photo Curator's taste ranking & annotations."""

    project_title: str
    total_evaluated: int = 0
    burst_clusters_found: int = 0
    selected_count: int = 0
    rejected_count: int = 0
    average_taste_score_selected: float = 9.0
    curatorial_statement: str = ""
    items: list[CuratedPhotoItem] = Field(default_factory=list)

    def to_json_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PhotoCurationReport":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class MediaIndex(BaseModel):
    """Agent 1 Output (media_index.json): Normalized, culled, and vision/audio-tagged sub-clips."""

    source_directory: str
    created_at: str
    total_raw_videos: int = 0
    total_raw_photos: int = 0
    kept_selects_count: int = 0
    discarded_count: int = 0
    camera_clock_offsets_sec: dict[str, float] = Field(default_factory=dict)
    subclips: list[CulledSubClip] = Field(default_factory=list)

    def to_json_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "MediaIndex":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class EnergySection(BaseModel):
    """Musical energy section within a background music track."""

    start: float
    end: float
    label: Literal["intro", "verse", "chorus_drop", "outro"] = "verse"
    rms_energy: float = 0.5


class BeatGrid(BaseModel):
    """Agent 4 Output (beat_grid.json): Beat, downbeat, and energy map for music synchronization."""

    source_audio: str
    duration: float
    tempo_bpm: float = 120.0
    beats: list[float] = Field(default_factory=list)
    downbeats: list[float] = Field(default_factory=list)
    energy_sections: list[EnergySection] = Field(default_factory=list)

    def snap_duration_to_beat(self, target_dur: float, min_dur: float = 1.8, max_dur: float = 4.5) -> float:
        """Snap a clip duration to the closest musical beat interval within [min_dur, max_dur]."""
        beat_period = 60.0 / max(40.0, min(220.0, self.tempo_bpm))
        best_beats = max(1, int(round(target_dur / beat_period)))
        snapped = round(best_beats * beat_period, 2)
        while snapped < min_dur:
            snapped = round(snapped + beat_period, 2)
        while snapped > max_dur and snapped - beat_period >= min_dur:
            snapped = round(snapped - beat_period, 2)
        return snapped

    def to_json_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BeatGrid":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class ChapterPlan(BaseModel):
    """A vignette chapter inside paper_edit.json produced by Agent 2 (StoryProducerAgent)."""

    chapter_id: str
    title: str
    subtitle: str = ""
    location_tag: str = "Park"
    time_of_day: Literal["morning", "afternoon", "twilight", "night"] = "afternoon"
    editing_mode: Literal["dialogue_with_broll", "music_montage"] = "dialogue_with_broll"
    photo_integration_mode: Literal["beat_burst", "ken_burns_pause"] = "ken_burns_pause"
    selected_subclip_ids: list[str] = Field(default_factory=list)
    broll_anchor_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps V1 A-roll subclip_id -> list of V2 B-roll/photo cutaway subclip_ids",
    )
    narrative_goal: str = ""


class PaperEdit(BaseModel):
    """Agent 2 Output (paper_edit.json): Chapter vignettes and A-roll/B-roll mode switching plan."""

    project_title: str
    story_arc_summary: str
    cold_open_subclip_id: Optional[str] = Field(
        default=None,
        description="Optional subclip_id selected for the 0.0s-3.8s Cold-Open Teaser Hook before the Title Card",
    )
    cold_open_in_tc: Optional[float] = Field(default=None, description="Optional override in_tc for the Cold-Open Hook")
    cold_open_out_tc: Optional[float] = Field(default=None, description="Optional override out_tc for the Cold-Open Hook")
    cold_open_hook_caption: str = Field(
        default="",
        description="Teaser hook caption or quote displayed during the Cold-Open Hook",
    )
    cold_open_rationale: str = Field(
        default="",
        description="Story Producer / AI Agent rationale for choosing this Cold-Open Hook",
    )
    chapters: list[ChapterPlan] = Field(default_factory=list)

    def to_json_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PaperEdit":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class AgentCoordinationMessage(BaseModel):
    """A message or revision request exchanged on the PipelineBlackboard Message Bus."""

    timestamp: str
    sender_role: str
    recipient_role: str
    message_type: Literal[
        "directive",
        "handoff",
        "revision_request",
        "revision_completed",
        "audit_note",
        "director_directive",
        "director_signoff",
    ] = "handoff"
    subject: str
    payload: dict[str, Any] = Field(default_factory=dict)


class DirectorStageReview(BaseModel):
    """A stage-gate review and corrective directive issued by the Movie Production Director Agent."""

    stage_order: int
    reviewed_agent_role: str
    status: Literal["approved", "directed_improvement", "polished_by_director"] = "approved"
    quality_score: float = Field(default=9.0, ge=0.0, le=10.0)
    critique: str
    directive_sent: Optional[str] = None
    directive_payload: dict[str, Any] = Field(default_factory=dict)


class DirectorReviewReport(BaseModel):
    """Agent 6 Output (director_review.json): Executive Movie Production Director's quality audit & sign-off."""

    director_vision_statement: str
    overall_quality_score: float = Field(default=9.2, ge=0.0, le=10.0)
    quality_grade: str = Field(default="A")
    approved_for_master_render: bool = True
    interventions_count: int = 0
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    stage_reviews: list[DirectorStageReview] = Field(default_factory=list)
    final_director_notes: list[str] = Field(default_factory=list)
    screening_sheet_path: Optional[str] = Field(
        default=None,
        description="Path to the rendered Timeline Visual Screening Contact Sheet inspected by the Director",
    )

    def to_json_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "DirectorReviewReport":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class PacingAuditEntry(BaseModel):
    """A single pacing or brief-compliance intervention recorded by Agent 5 (ShowrunnerAgent)."""

    rule_name: str
    action_taken: Literal[
        "dropped_duplicate",
        "trimmed_long_broll",
        "injected_chapter_title",
        "verified_must_have",
        "requested_revision",
    ]
    target_clip_id: str
    details: str


class PacingAuditReport(BaseModel):
    """Agent 5 Output (pacing_audit.json): Pacing critic audit report and inter-agent coordination log."""

    passed_brief_compliance: bool = True
    revision_rounds_executed: int = 0
    average_shot_length_before: float = 0.0
    average_shot_length_after: float = 0.0
    total_cuts_before: int = 0
    total_cuts_after: int = 0
    entries: list[PacingAuditEntry] = Field(default_factory=list)
    coordination_log: list[AgentCoordinationMessage] = Field(default_factory=list)

    def to_json_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PacingAuditReport":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


# ============================================================================
# Step 2 & Step 3: Edit Timeline Schemas (timeline.yaml)
# ============================================================================


class TimelineSettings(BaseModel):
    """Global output video canvas and encoding settings."""

    width: int = Field(default=3840, description="Output video width in pixels (default 3840 for 4K UHD)")
    height: int = Field(default=2160, description="Output video height in pixels (default 2160 for 4K UHD)")
    fps: int = Field(default=30, description="Output frame rate")
    hdr: bool = Field(
        default=True,
        description="Render in 10-bit HLG HDR (HEVC Main 10, Rec.2020 / arib-std-b67, yuv420p10le)",
    )
    background_color: str = Field(default="#000000")


class KenBurnsConfig(BaseModel):
    """Configuration for photo animation, multi-plane framing, and atmospheric finishers."""

    mode: PhotoModeType = Field(
        default="parallax_gallery",
        description=(
            "'parallax_gallery' renders a 2.5D multi-plane counter-panning backdrop + double museum matte frame; "
            "'cinematic_depth_3d' separates foreground subject from background with bokeh rack-focus parallax; "
            "'triptych_cinematic' animates a 3-panel staggered storyboard reveal; "
            "'polaroid_scrapbook' renders tilted vintage instant-film print(s) with organic rotation; "
            "'curator_loupe' glides an optical 1.65x magnifying glass spotlight across key details; "
            "'editorial_magazine' creates an asymmetric 62/38 luxury magazine spread; "
            "'cinema_anamorphic' frames a 2.39:1 widescreen letterbox with 35mm highlight halation; "
            "'diptych_frame' pairs two photos side-by-side; 'gallery_frame' renders a classic gallery frame; "
            "'full_bleed' crops to fill the canvas."
        ),
    )
    overlay_fx: PhotoOverlayFxType = Field(
        default="none",
        description="Composable atmospheric finisher: 'none', 'golden_light_leak', 'champagne_bokeh', 'museum_spotlight', or 'nostalgic_film_grain'",
    )
    preset: Literal[
        "zoom_in",
        "zoom_out",
        "pan_left_to_right",
        "pan_right_to_left",
        "tilt_up",
        "floating_gallery",
        "static",
        "custom",
    ] = Field(default="floating_gallery")
    start_zoom: float = Field(default=1.0, ge=0.5, le=2.5)
    end_zoom: float = Field(default=1.08, ge=0.5, le=2.5)
    start_center: tuple[float, float] = Field(default=(0.5, 0.5))
    end_center: tuple[float, float] = Field(default=(0.5, 0.48))
    border_px: int = Field(default=16, description="White gallery frame border width in pixels (at 1080p scale)")
    secondary_source: Optional[str] = Field(
        default=None, description="Optional second photo file path for diptych_frame / polaroid_scrapbook / triptych_cinematic"
    )
    tertiary_source: Optional[str] = Field(
        default=None, description="Optional third photo file path for 3-panel triptych_cinematic layout"
    )
    caption_title: Optional[str] = Field(
        default=None, description="Curated subject/artwork/landmark title displayed on the museum plaque"
    )
    caption_subtitle: Optional[str] = Field(
        default=None, description="Bilingual curatorial subtitle displayed on the museum plaque"
    )
    camera_badge: Optional[str] = Field(
        default=None, description="Camera & location EXIF badge (e.g. 'SONY A7C II · LOUVRE PARIS')"
    )


class TransitionConfig(BaseModel):
    """Visual transition into or out of a clip."""

    type: Literal["none", "cut", "crossfade", "dissolve", "fade_black", "dip_to_white"] = Field(default="fade_black")
    duration: float = Field(default=0.35, ge=0.0, le=3.0)


class VisualTrackItem(BaseModel):
    """A single video subclip or animated photo on Track V1 (storyline) or Track V2 (B-roll cutaway)."""

    id: str = Field(..., description="Unique clip ID on the timeline (e.g., clip_001 or broll_001)")
    type: Literal["video", "photo"] = Field(...)
    source: str = Field(..., description="Path to the source video or photo file")
    track_lane: int = Field(default=1, description="1 = Track V1 primary storyline, 2 = Track V2 B-roll cutaway")
    roll_type: Literal["a_roll", "b_roll", "still"] = Field(default="b_roll")
    shot_scale: Literal["wide", "medium", "closeup"] = Field(default="medium")
    act: Optional[str] = Field(default=None, description="Act or chapter section name")
    timeline_start: float = Field(..., ge=0.0, description="Start position on the master timeline (seconds)")
    duration: float = Field(..., gt=0.1, description="Duration on the master timeline (seconds)")
    source_in: float = Field(default=0.0, ge=0.0, description="In-point within source media (seconds)")
    source_out: Optional[float] = Field(default=None, description="Out-point within source media (seconds)")
    speed: float = Field(default=1.0, gt=0.1, le=4.0)
    volume: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Clip audio volume (1.0 for A-roll speech, ~0.10 [-20dB] for B-roll Nat-Sound, 0.0 for photos)",
    )
    j_l_cut_audio_offset: float = Field(
        default=0.0,
        description="Audio lead/lag in seconds (negative = J-cut audio starts early; positive = L-cut audio trails)",
    )
    audio_crossfade_sec: float = Field(
        default=0.33,
        ge=0.0,
        le=2.0,
        description="10-frame (0.33s) audio crossfade to prevent ambient sound clicks/pops",
    )
    ken_burns: Optional[KenBurnsConfig] = Field(default=None, description="Ken Burns / Gallery Frame settings for photos")
    transition_in: TransitionConfig = Field(default_factory=TransitionConfig)
    transition_out: TransitionConfig = Field(default_factory=TransitionConfig)
    narrative_note: str = Field(default="", description="Editorial rationale for why this shot is placed here")

    @property
    def clip_id(self) -> str:
        return self.id

    @property
    def caption_title(self) -> Optional[str]:
        return self.ken_burns.caption_title if self.ken_burns else None

    @property
    def caption_subtitle(self) -> Optional[str]:
        return self.ken_burns.caption_subtitle if self.ken_burns else None


class EffectTrackItem(BaseModel):
    """Extensible Stage-40 VFX / Color / Motion effect item on EditTimeline."""

    id: str
    effect_type: str = Field(
        ...,
        description="Pluggable effect identifier (e.g., 'warm_glow', 'vignette_focus', 'contrast_pop', 'speed_ramp')",
    )
    target_clip_id: Optional[str] = Field(default=None, description="Optional clip ID this effect attaches to")
    timeline_start: float = Field(..., ge=0.0)
    duration: float = Field(..., gt=0.1)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    created_by_agent: str = Field(default="vfx_designer")


class TitleOverlayItem(BaseModel):
    """A graphical title card, chapter banner, or lower-third callout (Track V3)."""

    id: str
    style: Literal["opening_card", "act_banner", "lower_third", "dialogue_badge", "closing_card"] = Field(
        default="lower_third"
    )
    text: str = Field(..., description="Primary title or callout headline")
    subtitle: Optional[str] = Field(default=None, description="Secondary line or English/bilingual subtitle")
    timeline_start: float = Field(..., ge=0.0)
    duration: float = Field(default=3.5, gt=0.2)
    fade_in: float = Field(default=0.35, ge=0.0)
    fade_out: float = Field(default=0.35, ge=0.0)
    accent_color: str = Field(default="#FFD740", description="Hex color for border/title highlight")


class AudioTrackItem(BaseModel):
    """Background music, voiceover, or sound effect track item (Track A3+)."""

    id: str
    role: Literal["music", "voiceover", "sfx"] = Field(default="music")
    source: str = Field(..., description="Path to audio file (.mp3, .wav, .m4a)")
    timeline_start: float = Field(default=0.0, ge=0.0)
    duration: Optional[float] = Field(default=None, description="Duration on timeline; defaults to full timeline length")
    source_in: float = Field(default=0.0, ge=0.0)
    volume: float = Field(default=0.28, ge=0.0, le=2.0, description="Base volume multiplier (e.g. 0.25 ≈ -12 dB)")
    duck_under_dialogue: bool = Field(
        default=True,
        description="Automatically lower volume during spoken dialogue or singing in video clips",
    )
    duck_volume: float = Field(default=0.07, ge=0.0, le=1.0, description="Volume multiplier when ducked (-14 dB)")
    duck_ramp_sec: float = Field(default=1.0, ge=0.1, le=3.0, description="Smooth volume ramp duration in seconds")
    fade_in: float = Field(default=1.0, ge=0.0)
    fade_out: float = Field(default=2.5, ge=0.0)


class SubtitleItem(BaseModel):
    """A single timed caption on the subtitle track."""

    id: str = Field(default="sub")
    start: float = Field(..., ge=0.0, description="Start time on the master timeline (seconds)")
    end: float = Field(..., gt=0.0, description="End time on the master timeline (seconds)")
    speaker: Optional[str] = Field(default=None)
    text: str = Field(..., description="Primary subtitle text")
    secondary_text: Optional[str] = Field(default=None, description="Optional bilingual/secondary subtitle line")


class SubtitleConfig(BaseModel):
    """Subtitle track settings and timed items."""

    enabled: bool = Field(default=True)
    burn_in: bool = Field(default=True, description="Whether to burn subtitles into the rendered video")
    font_name: str = Field(default="Microsoft YaHei")
    font_size: int = Field(default=42)
    items: list[SubtitleItem] = Field(default_factory=list)


class TimelineMarker(BaseModel):
    """Act or chapter marker on the master timeline (exported to DaVinci Resolve / FCPXML)."""

    time_sec: float = Field(..., ge=0.0)
    name: str
    color: str = Field(default="Blue")
    note: str = Field(default="")


class EditTimeline(BaseModel):
    """Complete multi-track editing timeline (timeline.yaml) compiled by the multi-agent pipeline."""

    project_name: str = Field(default="AI_Edited_Story")
    user_prompt: str = Field(default="")
    story_arc_summary: str = Field(default="")
    settings: TimelineSettings = Field(default_factory=TimelineSettings)
    markers: list[TimelineMarker] = Field(default_factory=list)
    visual_track: list[VisualTrackItem] = Field(default_factory=list, description="Track V1 primary storyline")
    broll_track: list[VisualTrackItem] = Field(
        default_factory=list,
        description="Track V2 B-Roll & photo cutaways laid over V1 A-Roll speech",
    )
    title_track: list[TitleOverlayItem] = Field(default_factory=list, description="Track V3 graphic titles & badges")
    effects_track: list[EffectTrackItem] = Field(
        default_factory=list,
        description="Extensible VFX / Color / Motion effects track populated by Stage-40 plugin agents",
    )
    audio_tracks: list[AudioTrackItem] = Field(default_factory=list, description="Track A3 music & SFX beds")
    subtitles: SubtitleConfig = Field(default_factory=SubtitleConfig)

    @property
    def total_duration(self) -> float:
        if not self.visual_track:
            return 0.0
        return max(item.timeline_start + item.duration for item in self.visual_track)

    def recalculate_sequential_starts(self) -> None:
        """Ensure Track V1 visual_track timeline_start values form a clean sequential timeline."""
        cursor = 0.0
        for item in self.visual_track:
            item.timeline_start = round(cursor, 3)
            if item.type == "video" and item.source_out is None:
                item.source_out = round(item.source_in + item.duration, 3)
            cursor += item.duration

    def to_yaml(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                f,
                allow_unicode=True,
                sort_keys=False,
                width=110,
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EditTimeline":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
