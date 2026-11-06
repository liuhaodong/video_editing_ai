"""Pydantic schemas for Step 1 (MediaCatalog) and Step 2/3 (EditTimeline)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import yaml
from pydantic import BaseModel, Field


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
# Step 2 & Step 3: Edit Timeline Schemas (timeline.yaml)
# ============================================================================


class TimelineSettings(BaseModel):
    """Global output video canvas and encoding settings."""

    width: int = Field(default=1920, description="Output video width in pixels (e.g. 1920 or 3840)")
    height: int = Field(default=1080, description="Output video height in pixels (e.g. 1080 or 2160)")
    fps: int = Field(default=30, description="Output frame rate")
    background_color: str = Field(default="#000000")


class KenBurnsConfig(BaseModel):
    """Configuration for photo animation and framing (Ken Burns & Gallery Frame)."""

    mode: Literal["gallery_frame", "full_bleed"] = Field(
        default="gallery_frame",
        description=(
            "'gallery_frame' renders an uncropped photo with a crisp white border & drop shadow over a "
            "Gaussian-blurred ambient background plus subtle floating motion; 'full_bleed' crops to fill the canvas."
        ),
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


class TransitionConfig(BaseModel):
    """Visual transition into or out of a clip."""

    type: Literal["none", "crossfade", "fade_black"] = Field(default="fade_black")
    duration: float = Field(default=0.4, ge=0.0, le=3.0)


class VisualTrackItem(BaseModel):
    """A single video subclip or animated photo on the primary visual storyline (Track V1)."""

    id: str = Field(..., description="Unique clip ID on the timeline (e.g., clip_001)")
    type: Literal["video", "photo"] = Field(...)
    source: str = Field(..., description="Path to the source video or photo file")
    act: Optional[str] = Field(default=None, description="Act or story section name")
    timeline_start: float = Field(..., ge=0.0, description="Start position on the master timeline (seconds)")
    duration: float = Field(..., gt=0.1, description="Duration on the master timeline (seconds)")
    source_in: float = Field(default=0.0, ge=0.0, description="In-point within source media (seconds)")
    source_out: Optional[float] = Field(default=None, description="Out-point within source media (seconds)")
    speed: float = Field(default=1.0, gt=0.1, le=4.0)
    volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Camera/clip native audio volume")
    ken_burns: Optional[KenBurnsConfig] = Field(default=None, description="Ken Burns / Gallery Frame settings for photos")
    transition_in: TransitionConfig = Field(default_factory=TransitionConfig)
    transition_out: TransitionConfig = Field(default_factory=TransitionConfig)
    narrative_note: str = Field(default="", description="Editorial rationale for why this shot is placed here")


class TitleOverlayItem(BaseModel):
    """A graphical title card, chapter banner, or lower-third callout (Tracks V2/V3)."""

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
    """Background music, voiceover, or sound effect track item (Tracks A2+)."""

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
    duck_volume: float = Field(default=0.08, ge=0.0, le=1.0, description="Volume multiplier when ducked")
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
    """Complete text-based editing timeline (timeline.yaml) produced in Step 2 and rendered in Step 3."""

    project_name: str = Field(default="AI_Edited_Story")
    user_prompt: str = Field(default="")
    story_arc_summary: str = Field(default="")
    settings: TimelineSettings = Field(default_factory=TimelineSettings)
    markers: list[TimelineMarker] = Field(default_factory=list)
    visual_track: list[VisualTrackItem] = Field(default_factory=list)
    title_track: list[TitleOverlayItem] = Field(default_factory=list)
    audio_tracks: list[AudioTrackItem] = Field(default_factory=list)
    subtitles: SubtitleConfig = Field(default_factory=SubtitleConfig)

    @property
    def total_duration(self) -> float:
        if not self.visual_track:
            return 0.0
        return max(item.timeline_start + item.duration for item in self.visual_track)

    def recalculate_sequential_starts(self) -> None:
        """Ensure visual_track timeline_start values form a clean sequential timeline."""
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
