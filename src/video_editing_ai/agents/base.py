"""Foundational abstractions for the Modular Multi-Agent Video Editing Framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable, Optional, Type

from ..models.schema import (
    AgentCoordinationMessage,
    BeatGrid,
    BGMSelectionReport,
    CreativeBrief,
    DirectorReviewReport,
    EditTimeline,
    MediaCatalog,
    MediaIndex,
    PacingAuditReport,
    PaperEdit,
    PhotoCurationReport,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json_dict(raw_text: str) -> Optional[dict[str, Any]]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw_text
    s = candidate.find("{")
    e = candidate.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(candidate[s : e + 1])
        except Exception:
            return None
    return None


class PipelineBlackboard:
    """Shared state store and inter-agent message bus for the multi-agent editing pipeline."""

    def __init__(
        self,
        media_dir: str | Path,
        project_dir: str | Path,
        user_prompt: str = "",
        extra_audio_paths: Optional[list[str | Path]] = None,
        interview_answers: Optional[dict[str, str]] = None,
        width: int = 3840,
        height: int = 2160,
        fps: int = 30,
        hdr: bool = True,
        use_antigravity: bool = True,
        whisper_model: str = "small",
    ) -> None:
        self.media_dir = Path(media_dir).resolve()
        self.project_dir = Path(project_dir).resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.user_prompt = user_prompt
        self.extra_audio_paths = [Path(p).resolve() for p in (extra_audio_paths or []) if Path(p).exists()]
        self.interview_answers: dict[str, str] = dict(interview_answers or {})
        self.width = width
        self.height = height
        self.fps = fps
        self.hdr = hdr
        self.use_antigravity = use_antigravity
        self.whisper_model = whisper_model

        # Standard typed contracts
        self.catalog: Optional[MediaCatalog] = None
        self.creative_brief: Optional[CreativeBrief] = None
        self.bgm_selection: Optional[BGMSelectionReport] = None
        self.media_index: Optional[MediaIndex] = None
        self.photo_curation: Optional[PhotoCurationReport] = None
        self.beat_grid: Optional[BeatGrid] = None
        self.paper_edit: Optional[PaperEdit] = None
        self.timeline: Optional[EditTimeline] = None
        self.pacing_audit: Optional[PacingAuditReport] = None
        self.director_review: Optional[DirectorReviewReport] = None

        # Open dictionary for pluggable future agents (e.g., vfx_manifest.json, color_lut.json)
        self.custom_artifacts: dict[str, Any] = {}
        self.messages: list[AgentCoordinationMessage] = []
        self.executed_agents: list[dict[str, Any]] = []
        self.activity_log: list[dict[str, Any]] = []
        self.activity_callback: Optional[Callable[[dict[str, Any]], None]] = None
        self.revision_round: int = 0

    def emit_activity(
        self,
        role_id: str,
        action: str,
        detail: str = "",
        event_type: str = "agent_action",
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Broadcast a real-time sub-step activity or Director thought to the blackboard and live listener."""
        entry: dict[str, Any] = {
            "timestamp": _now_iso(),
            "role_id": role_id,
            "action": action,
            "detail": detail,
            "event_type": event_type,
        }
        if extra:
            entry.update(extra)
        self.activity_log.append(entry)
        if self.activity_callback is not None:
            try:
                self.activity_callback(entry)
            except Exception:
                pass
        return entry

    def send_message(
        self,
        sender_role: str,
        recipient_role: str,
        subject: str,
        payload: Optional[dict[str, Any]] = None,
        message_type: str = "handoff",
    ) -> AgentCoordinationMessage:
        msg = AgentCoordinationMessage(
            timestamp=_now_iso(),
            sender_role=sender_role,
            recipient_role=recipient_role,
            message_type=message_type,  # type: ignore[arg-type]
            subject=subject,
            payload=payload or {},
        )
        self.messages.append(msg)
        self.emit_activity(
            role_id=sender_role,
            action=f"[{message_type.upper()}] -> {recipient_role}: {subject}",
            detail=json.dumps(payload or {}, ensure_ascii=False)[:180] if payload else "",
            event_type="bus_message",
            extra={
                "sender_role": sender_role,
                "recipient_role": recipient_role,
                "message_type": message_type,
                "subject": subject,
            },
        )
        return msg

    def get_messages_for(self, recipient_role: str) -> list[AgentCoordinationMessage]:
        return [
            m
            for m in self.messages
            if m.recipient_role in (recipient_role, "all", "*")
        ]

    def set_custom_artifact(self, filename: str, data: dict[str, Any]) -> Path:
        self.custom_artifacts[filename] = data
        out_path = self.project_dir / filename
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path

    def save_snapshot(self) -> Path:
        snapshot_path = self.project_dir / "blackboard.json"
        doc = {
            "project_dir": str(self.project_dir),
            "media_dir": str(self.media_dir),
            "user_prompt": self.user_prompt,
            "revision_round": self.revision_round,
            "executed_agents": self.executed_agents,
            "custom_artifacts": list(self.custom_artifacts.keys()),
            "messages": [m.model_dump(mode="json") for m in self.messages],
        }
        snapshot_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        return snapshot_path

    def load_existing_contracts(self) -> None:
        """Hydrate blackboard state from existing JSON/YAML contract files in project_dir."""
        p = self.project_dir
        if (p / "annotations.yaml").exists() and self.catalog is None:
            try:
                self.catalog = MediaCatalog.from_yaml(p / "annotations.yaml")
            except Exception:
                pass
        if (p / "creative_brief.json").exists() and self.creative_brief is None:
            try:
                self.creative_brief = CreativeBrief.from_json_file(p / "creative_brief.json")
                if not self.user_prompt and self.creative_brief.raw_user_prompt:
                    self.user_prompt = self.creative_brief.raw_user_prompt
            except Exception:
                pass
        if (p / "bgm_selection.json").exists() and self.bgm_selection is None:
            try:
                self.bgm_selection = BGMSelectionReport.from_json_file(p / "bgm_selection.json")
            except Exception:
                pass
        if (p / "media_index.json").exists() and self.media_index is None:
            try:
                self.media_index = MediaIndex.from_json_file(p / "media_index.json")
            except Exception:
                pass
        if (p / "photo_curation.json").exists() and self.photo_curation is None:
            try:
                self.photo_curation = PhotoCurationReport.from_json_file(p / "photo_curation.json")
            except Exception:
                pass
        if (p / "beat_grid.json").exists() and self.beat_grid is None:
            try:
                self.beat_grid = BeatGrid.from_json_file(p / "beat_grid.json")
            except Exception:
                pass
        if (p / "paper_edit.json").exists() and self.paper_edit is None:
            try:
                self.paper_edit = PaperEdit.from_json_file(p / "paper_edit.json")
            except Exception:
                pass
        if (p / "timeline.yaml").exists() and self.timeline is None:
            try:
                self.timeline = EditTimeline.from_yaml(p / "timeline.yaml")
            except Exception:
                pass
        if (p / "pacing_audit.json").exists() and self.pacing_audit is None:
            try:
                self.pacing_audit = PacingAuditReport.from_json_file(p / "pacing_audit.json")
            except Exception:
                pass
        if (p / "director_review.json").exists() and self.director_review is None:
            try:
                self.director_review = DirectorReviewReport.from_json_file(p / "director_review.json")
            except Exception:
                pass
        for custom_fname in (
            "vfx_manifest.json",
            "narration_script.json",
            "cinematography_report.json",
            "motion_graphics_plan.json",
            "color_grade_plan.json",
            "sound_design_plan.json",
        ):
            if (p / custom_fname).exists() and custom_fname not in self.custom_artifacts:
                try:
                    self.custom_artifacts[custom_fname] = json.loads(
                        (p / custom_fname).read_text(encoding="utf-8")
                    )
                except Exception:
                    pass
        if (p / "blackboard.json").exists() and not self.messages:
            try:
                bb_doc = json.loads((p / "blackboard.json").read_text(encoding="utf-8"))
                self.executed_agents = list(bb_doc.get("executed_agents") or [])
                self.messages = [
                    AgentCoordinationMessage.model_validate(m) for m in (bb_doc.get("messages") or [])
                ]
            except Exception:
                pass


class BaseEditingAgent(ABC):
    """Abstract base class for all core and pluggable video editing sub-agents."""

    role_id: str = "base_agent"
    display_name: str = "Base Editing Agent"
    stage_order: int = 10
    description: str = ""
    enabled_by_default: bool = True
    requires_artifacts: list[str] = []
    produces_artifacts: list[str] = []
    skill_reference_file: str = ""

    def load_skill_instructions(self) -> str:
        """Load the Markdown system instructions from .agent/skills/ai-video-editor/references/."""
        if not self.skill_reference_file:
            return f"You are {self.display_name} ({self.role_id}). {self.description}"

        repo_root = Path(__file__).resolve().parents[3]
        ref_path = repo_root / ".agent" / "skills" / "ai-video-editor" / "references" / self.skill_reference_file
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8")
        return f"You are {self.display_name} ({self.role_id}). {self.description}"

    async def _ask_antigravity_async(self, prompt: str) -> Optional[dict[str, Any]]:
        from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig

        config = LocalAgentConfig(
            system_instructions=self.load_skill_instructions(),
            capabilities=CapabilitiesConfig(),
        )
        chunks: list[str] = []
        async with Agent(config) as agent:
            response = await agent.chat(prompt)
            async for tok in response:
                chunks.append(str(tok))
        return _extract_json_dict("".join(chunks))

    def ask_antigravity_json(self, prompt: str, timeout_sec: float = 45.0) -> Optional[dict[str, Any]]:
        """Invoke the Google Antigravity SDK Agent with this role's skill instructions and return parsed JSON."""
        try:
            return asyncio.run(asyncio.wait_for(self._ask_antigravity_async(prompt), timeout=timeout_sec))
        except Exception:
            return None

    def ask_ai_creative_json(
        self,
        blackboard: "PipelineBlackboard",
        prompt: str,
        image_paths: Optional[list[str | Path]] = None,
        decision_key: Optional[str] = None,
        timeout_sec: float = 45.0,
    ) -> Optional[dict[str, Any]]:
        """
        Unified multimodal AI Creative Intelligence gateway for all 11 editing agents:
        1. Checks `<project_dir>/ai_decisions/<key>.json` for interactive Antigravity Agent overrides.
        2. If `blackboard.use_antigravity` is True, invokes Google Antigravity SDK (with contact-sheet image paths)
           or Gemini API (`GEMINI_API_KEY` / `GOOGLE_API_KEY`) to obtain structured creative JSON decisions.
        3. Persists the decision to `<project_dir>/ai_decisions/<key>.json` and logs an `ai_intelligence` event.
        """
        import base64
        import os
        import urllib.request

        key = decision_key or self.role_id
        decisions_dir = blackboard.project_dir / "ai_decisions"
        decisions_dir.mkdir(parents=True, exist_ok=True)
        decision_file = decisions_dir / f"{key}.json"

        # 1. Check for interactive AI Agent override file written via Skill CLI
        if decision_file.exists():
            try:
                cached = json.loads(decision_file.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached:
                    blackboard.emit_activity(
                        self.role_id,
                        f"Loaded AI Creative Intelligence decision packet ('{decision_file.name}')",
                        detail=str(cached.get("ai_rationale") or cached.get("curatorial_statement") or "Applied AI creative overrides")[:180],
                        event_type="ai_intelligence",
                    )
                    return cached
            except Exception:
                pass

        if not blackboard.use_antigravity:
            return None

        valid_imgs = [str(Path(p).resolve()) for p in (image_paths or []) if p and Path(p).exists()]
        enriched_prompt = prompt
        if valid_imgs:
            enriched_prompt = (
                f"Multimodal Contact Sheet / Keyframe Images available for visual inspection: {valid_imgs}\n\n"
                f"{prompt}"
            )

        # 2a. Try Google Antigravity SDK Agent
        result = self.ask_antigravity_json(enriched_prompt, timeout_sec=timeout_sec)

        # 2b. Optional fallback to GEMINI_API_KEY / GOOGLE_API_KEY if configured in environment
        if result is None:
            api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
            if api_key:
                try:
                    parts: list[dict[str, Any]] = [
                        {"text": self.load_skill_instructions() + "\n\n" + enriched_prompt}
                    ]
                    for img_p in valid_imgs[:3]:
                        raw_bytes = Path(img_p).read_bytes()
                        if len(raw_bytes) <= 4_000_000:
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": base64.b64encode(raw_bytes).decode("ascii"),
                                    }
                                }
                            )
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
                    body = json.dumps(
                        {
                            "contents": [{"parts": parts}],
                            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
                        }
                    ).encode("utf-8")
                    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                        resp_doc = json.loads(resp.read().decode("utf-8"))
                        cand_text = (
                            resp_doc.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        result = _extract_json_dict(cand_text)
                except Exception:
                    result = None

        if isinstance(result, dict) and result:
            try:
                decision_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            blackboard.emit_activity(
                self.role_id,
                f"AI Creative Intelligence synthesized decision packet ('{decision_file.name}')",
                detail=str(result.get("ai_rationale") or "Applied multimodal AI creative intelligence")[:180],
                event_type="ai_intelligence",
            )
        return result

    @classmethod
    def to_metadata_dict(cls) -> dict[str, Any]:
        return {
            "role_id": cls.role_id,
            "display_name": cls.display_name,
            "stage_order": cls.stage_order,
            "description": cls.description,
            "enabled_by_default": cls.enabled_by_default,
            "requires_artifacts": list(cls.requires_artifacts),
            "produces_artifacts": list(cls.produces_artifacts),
            "skill_reference_file": cls.skill_reference_file,
        }

    @abstractmethod
    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        """Execute this sub-agent's logic on the shared PipelineBlackboard."""


class AgentRegistry:
    """Global registry for discovering, listing, and ordering modular editing sub-agents."""

    _agents: dict[str, Type[BaseEditingAgent]] = {}

    @classmethod
    def register(cls, agent_cls: Type[BaseEditingAgent]) -> Type[BaseEditingAgent]:
        cls._agents[agent_cls.role_id] = agent_cls
        return agent_cls

    @classmethod
    def unregister(cls, role_id: str) -> None:
        cls._agents.pop(role_id, None)

    @classmethod
    def get(cls, role_id: str) -> Optional[Type[BaseEditingAgent]]:
        return cls._agents.get(role_id)

    @classmethod
    def list_all(cls) -> list[Type[BaseEditingAgent]]:
        return sorted(cls._agents.values(), key=lambda a: (a.stage_order, a.role_id))

    @classmethod
    def resolve_execution_plan(
        cls,
        enabled_optional_roles: Optional[list[str]] = None,
        disabled_roles: Optional[list[str]] = None,
    ) -> list[BaseEditingAgent]:
        """Return instantiated agents sorted by stage_order according to enabled/disabled filters."""
        opt_set = set(enabled_optional_roles or [])
        dis_set = set(disabled_roles or [])
        plan: list[BaseEditingAgent] = []

        for agent_cls in cls.list_all():
            rid = agent_cls.role_id
            if rid in dis_set:
                continue
            if agent_cls.enabled_by_default or rid in opt_set:
                plan.append(agent_cls())
        return plan


def register_agent(agent_cls: Type[BaseEditingAgent]) -> Type[BaseEditingAgent]:
    """Class decorator to register a sub-agent role in AgentRegistry."""
    return AgentRegistry.register(agent_cls)
