"""Introspect AgentRegistry to auto-synchronize README.md and Antigravity skill documentation."""

from __future__ import annotations

import inspect
from pathlib import Path
import re
from typing import Any, Optional

from .agents import AgentRegistry, BaseEditingAgent

TABLE_START_MARKER = "<!-- AGENT_REGISTRY_TABLE_START -->"
TABLE_END_MARKER = "<!-- AGENT_REGISTRY_TABLE_END -->"
REF_LIST_START_MARKER = "<!-- AGENT_REF_LIST_START -->"
REF_LIST_END_MARKER = "<!-- AGENT_REF_LIST_END -->"


def _agent_summary(agent_cls: type[BaseEditingAgent]) -> str:
    """Extract a concise one-line summary from docstring or description."""
    doc = inspect.getdoc(agent_cls) or ""
    first_line = doc.splitlines()[0].strip() if doc.strip() else ""
    summary = (agent_cls.description or first_line or agent_cls.display_name).strip()
    return summary.replace("|", "\\|").replace("\n", " ")


def build_readme_agent_table(agents: Optional[list[type[BaseEditingAgent]]] = None) -> str:
    """Render a Markdown architecture table from AgentRegistry metadata for README.md."""
    roster = agents if agents is not None else AgentRegistry.list_all()
    lines = [
        TABLE_START_MARKER,
        "| Stage | Role ID | Sub-Agent Class | Default | Requires -> Produces | Responsibilities |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for cls in roster:
        aliases = getattr(cls, "legacy_role_aliases", []) or []
        alias_note = f" *(alias: `{', '.join(aliases)}`)*" if aliases else ""
        default_badge = "Core (`True`)" if cls.enabled_by_default else "Optional (`False`)"
        req_str = ", ".join(f"`{r}`" for r in (cls.requires_artifacts or [])) or "*(root)*"
        prod_str = ", ".join(f"`{p}`" for p in (cls.produces_artifacts or [])) or "*(in-memory)*"
        contract_str = f"{req_str} -> {prod_str}"
        lines.append(
            f"| **Stage {cls.stage_order:02d}** | `{cls.role_id}`{alias_note} | `{cls.__name__}` | "
            f"{default_badge} | {contract_str} | {_agent_summary(cls)} |"
        )
    lines.append(TABLE_END_MARKER)
    return "\n".join(lines)


def build_skill_agent_table(agents: Optional[list[type[BaseEditingAgent]]] = None) -> str:
    """Render a Markdown architecture table with skill reference links for SKILL.md."""
    roster = agents if agents is not None else AgentRegistry.list_all()
    lines = [
        TABLE_START_MARKER,
        "| Stage | Role ID | Sub-Agent Class | Default | Contracts (`Requires -> Produces`) | Reference Playbook & Summary |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for cls in roster:
        aliases = getattr(cls, "legacy_role_aliases", []) or []
        alias_note = f" *(alias: `{', '.join(aliases)}`)*" if aliases else ""
        default_badge = "`True`" if cls.enabled_by_default else "`False`"
        req_str = ", ".join(f"`{r}`" for r in (cls.requires_artifacts or [])) or "*(root)*"
        prod_str = ", ".join(f"`{p}`" for p in (cls.produces_artifacts or [])) or "*(in-memory)*"
        ref_file = cls.skill_reference_file or f"{cls.role_id}.md"
        ref_link = f"[references/{ref_file}](references/{ref_file})"
        lines.append(
            f"| **Stage {cls.stage_order:02d}** | `{cls.role_id}`{alias_note} | `{cls.__name__}` | "
            f"{default_badge} | {req_str} -> {prod_str} | {ref_link} — {_agent_summary(cls)} |"
        )
    lines.append(TABLE_END_MARKER)
    return "\n".join(lines)


def build_readme_ref_list(agents: Optional[list[type[BaseEditingAgent]]] = None) -> str:
    """Render deduplicated reference playbook bullet list for README.md."""
    roster = agents if agents is not None else AgentRegistry.list_all()
    seen: list[str] = []
    for cls in roster:
        ref = cls.skill_reference_file or f"{cls.role_id}.md"
        if ref not in seen:
            seen.append(ref)
    if "10_contracts_and_rendering.md" not in seen:
        seen.append("10_contracts_and_rendering.md")
    lines = [REF_LIST_START_MARKER]
    for ref in seen:
        lines.append(f"- `{ref}`")
    lines.append(REF_LIST_END_MARKER)
    return "\n".join(lines)


def _replace_marker_section(content: str, start_marker: str, end_marker: str, new_block: str) -> str:
    if start_marker in content and end_marker in content:
        pattern = re.compile(
            re.escape(start_marker) + r".*?" + re.escape(end_marker),
            flags=re.DOTALL,
        )
        return pattern.sub(lambda _m: new_block, content, count=1)
    return content


def _ensure_reference_file(ref_dir: Path, cls: type[BaseEditingAgent]) -> tuple[Path, bool]:
    """Verify or create a matching .md reference playbook for an agent role."""
    ref_name = cls.skill_reference_file or f"{cls.role_id}.md"
    ref_path = ref_dir / ref_name
    if ref_path.exists():
        return ref_path, False

    ref_dir.mkdir(parents=True, exist_ok=True)
    req_str = ", ".join(f"`{r}`" for r in (cls.requires_artifacts or [])) or "None"
    prod_str = ", ".join(f"`{p}`" for p in (cls.produces_artifacts or [])) or "Timeline state"
    template = (
        f"# Role: Stage {cls.stage_order:02d} — {cls.display_name} (`{cls.role_id}`)\n\n"
        f"## Mission\n"
        f"{_agent_summary(cls)}\n\n"
        f"## Artifact Contracts\n"
        f"- **Class**: `{cls.__name__}`\n"
        f"- **Enabled by Default**: `{cls.enabled_by_default}`\n"
        f"- **Requires Artifacts**: {req_str}\n"
        f"- **Produces Artifacts**: {prod_str}\n\n"
        f"## Skill CLI Invocation\n"
        f"```powershell\n"
        f"python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py run-stage "
        f"--stage {cls.role_id} --project-dir \"<project_dir>\" --media-dir \"<media_dir>\"\n"
        f"```\n"
    )
    ref_path.write_text(template, encoding="utf-8")
    return ref_path, True


def sync_agent_documentation(
    repo_root: Optional[Path] = None,
    create_missing_refs: bool = True,
) -> dict[str, Any]:
    """Synchronize README.md, SKILL.md, and skill reference files from AgentRegistry."""
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    readme_path = root / "README.md"
    skill_dir = root / ".agent" / "skills" / "ai-video-editor"
    skill_md_path = skill_dir / "SKILL.md"
    refs_dir = skill_dir / "references"

    roster = AgentRegistry.list_all()
    count = len(roster)

    verified_refs: list[str] = []
    created_refs: list[str] = []
    for cls in roster:
        if create_missing_refs:
            ref_path, was_created = _ensure_reference_file(refs_dir, cls)
            verified_refs.append(str(ref_path))
            if was_created:
                created_refs.append(str(ref_path))
        else:
            ref_name = cls.skill_reference_file or f"{cls.role_id}.md"
            ref_path = refs_dir / ref_name
            if ref_path.exists():
                verified_refs.append(str(ref_path))

    readme_updated = False
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8")
        new_table = build_readme_agent_table(roster)
        new_refs = build_readme_ref_list(roster)
        updated = readme_text
        if TABLE_START_MARKER in updated and TABLE_END_MARKER in updated:
            updated = _replace_marker_section(updated, TABLE_START_MARKER, TABLE_END_MARKER, new_table)
        else:
            legacy_table_re = re.compile(
                r"\| Stage \| Role ID \|.*?(?=\n---|\n## |\Z)",
                flags=re.DOTALL,
            )
            if legacy_table_re.search(updated):
                updated = legacy_table_re.sub(lambda _m: new_table + "\n", updated, count=1)
        if REF_LIST_START_MARKER in updated and REF_LIST_END_MARKER in updated:
            updated = _replace_marker_section(updated, REF_LIST_START_MARKER, REF_LIST_END_MARKER, new_refs)
        updated = re.sub(
            r"## Modular Multi-Agent Architecture \([^\n]+\)",
            f"## Modular {count}-Agent Architecture (Stages 00 -> 65)",
            updated,
        )
        if updated != readme_text:
            readme_path.write_text(updated, encoding="utf-8")
            readme_updated = True

    skill_updated = False
    if skill_md_path.exists():
        skill_text = skill_md_path.read_text(encoding="utf-8")
        new_skill_table = build_skill_agent_table(roster)
        updated_skill = skill_text
        if TABLE_START_MARKER in updated_skill and TABLE_END_MARKER in updated_skill:
            updated_skill = _replace_marker_section(
                updated_skill, TABLE_START_MARKER, TABLE_END_MARKER, new_skill_table
            )
        updated_skill = re.sub(r"\b(?:11|16|21)-agent\b", f"{count}-agent", updated_skill)
        updated_skill = re.sub(r"\b(?:11|16|21)-Agent\b", f"{count}-Agent", updated_skill)
        updated_skill = re.sub(
            r"## 1\. Progressive Disclosure: \d+ Sub-Agent Role Playbooks",
            f"## 1. Progressive Disclosure: {count} Sub-Agent Role Playbooks",
            updated_skill,
        )
        if updated_skill != skill_text:
            skill_md_path.write_text(updated_skill, encoding="utf-8")
            skill_updated = True

    return {
        "agent_count": count,
        "role_ids": [c.role_id for c in roster],
        "readme_path": str(readme_path),
        "readme_updated": readme_updated,
        "skill_md_path": str(skill_md_path),
        "skill_updated": skill_updated,
        "verified_reference_files": verified_refs,
        "created_reference_files": created_refs,
    }
