"""Foundational abstractions for the Modular Multi-Agent Video Editing Framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Optional, Type

from ..models.schema import (
    AgentCoordinationMessage,
    BeatGrid,
    CreativeBrief,
    EditTimeline,
    MediaCatalog,
    MediaIndex,
    PacingAuditReport,
    PaperEdit,
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
        self.media_index: Optional[MediaIndex] = None
        self.beat_grid: Optional[BeatGrid] = None
        self.paper_edit: Optional[PaperEdit] = None
        self.timeline: Optional[EditTimeline] = None
        self.pacing_audit: Optional[PacingAuditReport] = None

        # Open dictionary for pluggable future agents (e.g., vfx_manifest.json, color_lut.json)
        self.custom_artifacts: dict[str, Any] = {}
        self.messages: list[AgentCoordinationMessage] = []
        self.executed_agents: list[dict[str, Any]] = []
        self.revision_round: int = 0

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
