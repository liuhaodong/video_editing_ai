"""Modular Multi-Agent Video Editing Architecture (Stage 00 -> Stage 65)."""

from .audio_vibe_agent import AudioVibeAgent
from .base import AgentRegistry, BaseEditingAgent, PipelineBlackboard, register_agent
from .cast_continuity_agent import (
    CastContinuityAgent,
    classify_utterance_language,
    compute_face_safe_ken_burns_centers,
    detect_subject_bbox_normalized,
)
from .culler_agent import (
    CullerAgent,
    apply_video_curation_overrides,
    render_video_culling_contact_sheets,
)
from .director_agent import ProductionDirectorAgent, render_director_screening_sheet
from .dit_ingest_agent import DITIngestAgent
from .intake_agent import (
    ClientIntakeAgent,
    build_ask_question_tool_payload,
    compute_director_duration_recommendation,
    generate_intake_questions,
    inspect_media_library,
    refine_intake_interactively,
)
from .lead_editor_agent import LeadAssemblyEditorAgent
from .music_supervisor_agent import MusicSupervisorAgent
from .nle_export_agent import NLEExportAgent
from .orchestrator import MultiAgentOrchestrator, run_multi_agent_pipeline
from .photo_curator_agent import (
    PhotoCuratorAgent,
    apply_vision_curation_overrides,
    render_curation_contact_sheets,
)
from .picture_editors import (
    DialogueEditorAgent,
    MontageEditorAgent,
    apply_timeline_intelligence_overrides,
)
from .plugins.camera_operator_agent import CameraOperatorAgent
from .plugins.colorist_agent import ColoristAgent, HDRColoristAgent, classify_clip_color_pipeline
from .plugins.motion_graphics_agent import MotionGraphicsAgent, MotionGraphicsTitleAgent
from .plugins.narrator_agent import NarratorVoiceoverAgent
from .plugins.sound_designer_agent import (
    SoundDesignerAgent,
    SoundDesignerMixerAgent,
    build_audio_mix_plan,
    compute_dialogue_normalization_gain,
)
from .plugins.vfx_agent import VFXAgent
from .showrunner_agent import ShowrunnerAgent
from .story_producer_agent import StoryProducerAgent, apply_story_plan_overrides
from .visual_qc_agent import ProxyWatchdownQCAgent, inspect_timeline_proxy_qc

__all__ = [
    "AgentRegistry",
    "AudioVibeAgent",
    "BaseEditingAgent",
    "CameraOperatorAgent",
    "CastContinuityAgent",
    "ClientIntakeAgent",
    "ColoristAgent",
    "CullerAgent",
    "DITIngestAgent",
    "DialogueEditorAgent",
    "HDRColoristAgent",
    "LeadAssemblyEditorAgent",
    "MontageEditorAgent",
    "MotionGraphicsAgent",
    "MotionGraphicsTitleAgent",
    "MultiAgentOrchestrator",
    "MusicSupervisorAgent",
    "NLEExportAgent",
    "NarratorVoiceoverAgent",
    "PhotoCuratorAgent",
    "PipelineBlackboard",
    "ProductionDirectorAgent",
    "ProxyWatchdownQCAgent",
    "ShowrunnerAgent",
    "SoundDesignerAgent",
    "SoundDesignerMixerAgent",
    "StoryProducerAgent",
    "VFXAgent",
    "apply_story_plan_overrides",
    "apply_timeline_intelligence_overrides",
    "apply_video_curation_overrides",
    "apply_vision_curation_overrides",
    "build_ask_question_tool_payload",
    "build_audio_mix_plan",
    "classify_clip_color_pipeline",
    "classify_utterance_language",
    "compute_director_duration_recommendation",
    "compute_dialogue_normalization_gain",
    "compute_face_safe_ken_burns_centers",
    "detect_subject_bbox_normalized",
    "generate_intake_questions",
    "inspect_media_library",
    "inspect_timeline_proxy_qc",
    "refine_intake_interactively",
    "register_agent",
    "render_curation_contact_sheets",
    "render_director_screening_sheet",
    "render_video_culling_contact_sheets",
    "run_multi_agent_pipeline",
]
