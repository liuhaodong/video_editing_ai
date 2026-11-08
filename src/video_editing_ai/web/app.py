"""FastAPI backend for the Guided 3-Step AI Photo & Video Editing Studio with Live Multi-Agent Progress."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agents import (
    AgentRegistry,
    MultiAgentOrchestrator,
    apply_story_plan_overrides,
    apply_vision_curation_overrides,
    generate_intake_questions,
    inspect_media_library,
    refine_intake_interactively,
    render_curation_contact_sheets,
    run_multi_agent_pipeline,
)
from ..analyzer.ag_annotator import analyze_media_directory
from ..analyzer.media_probe import AUDIO_EXTENSIONS, PHOTO_EXTENSIONS, VIDEO_EXTENSIONS
from ..exporters.nle_export import export_all_nle_formats
from ..models.schema import EditTimeline, MediaCatalog
from ..planner.story_director import generate_edit_plan, generate_storyboard_markdown
from ..planner.timeline_surgeon import apply_surgical_edit
from ..renderer.ffmpeg_composer import render_timeline, render_timeline_segment


class AnalyzeRequest(BaseModel):
    media_dir: str
    project_dir: str = "workspace_project"
    whisper_model: str = "small"
    user_hint: str = ""
    extra_audio_path: Optional[str] = None
    use_antigravity: bool = True


class SaveYamlRequest(BaseModel):
    project_dir: str = "workspace_project"
    yaml_text: str


class PlanRequest(BaseModel):
    project_dir: str = "workspace_project"
    user_prompt: str
    target_duration: Optional[float] = None
    width: int = 3840
    height: int = 2160
    fps: int = 30
    hdr: bool = True
    photo_mode: str = "parallax_gallery"
    use_antigravity: bool = True


class MultiAgentRunRequest(BaseModel):
    media_dir: str
    project_dir: str = "active_video_project"
    user_prompt: str = ""
    extra_audio_path: Optional[str] = None
    interview_answers: dict[str, str] = Field(default_factory=dict)
    enabled_optional_roles: list[str] = Field(default_factory=lambda: ["vfx_designer"])
    disabled_roles: list[str] = Field(default_factory=list)
    width: int = 3840
    height: int = 2160
    fps: int = 30
    hdr: bool = True
    use_antigravity: bool = False
    render_video: bool = True


class RenderRequest(BaseModel):
    project_dir: str = "workspace_project"
    output_filename: str = "final_story.mp4"


class IntakeRefineRequest(BaseModel):
    media_dir: str
    user_message: str = ""
    interview_answers: dict[str, str] = Field(default_factory=dict)
    chat_history: list[dict[str, str]] = Field(default_factory=list)


class ProjectCreateRequest(BaseModel):
    name: str = ""
    project_name: str = ""
    project_dir: Optional[str] = None
    media_dir: str = ""
    user_prompt: str = ""


class ProjectDirRequest(BaseModel):
    project_dir: str


class ContactSheetsRequest(BaseModel):
    project_dir: str = "workspace_project"
    max_items: int = 60


class VisionOverridesRequest(BaseModel):
    project_dir: str = "workspace_project"
    overrides: list[dict[str, Any]] = Field(default_factory=list)
    curatorial_statement: Optional[str] = None


class SurgicalEditRequest(BaseModel):
    project_dir: str = "workspace_project"
    operations: list[dict[str, Any]] = Field(default_factory=list)


class SegmentRenderRequest(BaseModel):
    project_dir: str = "workspace_project"
    start_sec: float = 0.0
    end_sec: float = 8.0
    fast_preview: bool = True
    output_filename: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify_project_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", (name or "").strip()).strip("_").lower()
    if not slug:
        slug = "story"
    if not slug.endswith("_project"):
        slug = f"{slug}_project"
    return slug


def _read_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_project_meta(proj_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    proj_dir.mkdir(parents=True, exist_ok=True)
    meta_path = proj_dir / "project_meta.json"
    existing = _read_json_if_exists(meta_path) or {}
    now = _now_iso()
    if "created_at" not in existing:
        existing["created_at"] = now
    existing.update(updates)
    existing["project_dir"] = str(proj_dir.resolve())
    existing["updated_at"] = now
    meta_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return existing


def _inspect_project_dir(proj_dir: Path) -> dict[str, Any]:
    proj = proj_dir.resolve()
    meta = _read_json_if_exists(proj / "project_meta.json") or {}
    brief = _read_json_if_exists(proj / "creative_brief.json") or {}
    bb = _read_json_if_exists(proj / "blackboard.json") or {}
    dr = _read_json_if_exists(proj / "director_review.json") or {}

    mp4_matches = sorted(proj.glob("*.mp4"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    mp4_file = (proj / "final_story.mp4") if (proj / "final_story.mp4").exists() else (mp4_matches[0] if mp4_matches else None)
    has_video = mp4_file is not None and mp4_file.exists()
    video_size_mb = round(mp4_file.stat().st_size / (1024 * 1024), 1) if has_video and mp4_file else 0.0
    video_mtime = int(mp4_file.stat().st_mtime) if has_video and mp4_file else 0

    has_timeline = (proj / "timeline.yaml").exists()
    has_contracts = bool(brief or bb or dr or has_timeline)

    if has_video:
        status = "rendered"
    elif has_contracts:
        status = "planned"
    else:
        status = "empty"

    name = (
        meta.get("name")
        or brief.get("project_title", "").replace("_", " ")
        or proj.name.replace("_project", "").replace("_", " ").title()
    )
    media_dir = meta.get("media_dir") or bb.get("media_dir") or ""
    user_prompt = meta.get("user_prompt") or brief.get("raw_user_prompt") or bb.get("user_prompt") or ""
    target_dur = brief.get("target_duration_sec") or meta.get("target_duration_sec") or 155

    # Updated timestamp fallback
    updated_ts = 0.0
    for candidate_f in (mp4_file, proj / "director_review.json", proj / "timeline.yaml", proj / "project_meta.json"):
        if candidate_f and candidate_f.exists():
            try:
                updated_ts = max(updated_ts, candidate_f.stat().st_mtime)
            except Exception:
                pass

    updated_str = (
        datetime.fromtimestamp(updated_ts).strftime("%Y-%m-%d %H:%M")
        if updated_ts > 0
        else "Just now"
    )
    resolved_video_path = str(mp4_file.resolve()) if has_video and mp4_file else None

    return {
        "name": name,
        "project_name": name,
        "slug": proj.name,
        "project_dir": str(proj),
        "media_dir": media_dir,
        "user_prompt": user_prompt,
        "interview_answers": meta.get("interview_answers") or {},
        "status": status,
        "has_rendered_video": has_video,
        "video_path": resolved_video_path,
        "rendered_video_path": resolved_video_path,
        "video_filename": mp4_file.name if has_video and mp4_file else None,
        "video_size_mb": video_size_mb,
        "rendered_video_size_mb": video_size_mb,
        "video_mtime": video_mtime,
        "rendered_video_mtime": video_mtime,
        "has_timeline": has_timeline,
        "has_contracts": has_contracts,
        "director_grade": dr.get("quality_grade"),
        "quality_grade": dr.get("quality_grade"),
        "director_score": dr.get("overall_quality_score"),
        "target_duration_sec": target_dur,
        "updated_timestamp": updated_ts,
        "updated_at": updated_str,
    }


def _list_workspace_projects(workspace_root: Path, active_proj_dir: str = "") -> list[dict[str, Any]]:
    candidates: list[Path] = []
    if active_proj_dir:
        ap = Path(active_proj_dir)
        if ap.exists() and ap.is_dir():
            candidates.append(ap.resolve())

    for sub in workspace_root.iterdir():
        if sub.is_dir() and (sub.name.endswith("_project") or (sub / "project_meta.json").exists() or (sub / "timeline.yaml").exists()):
            candidates.append(sub.resolve())

    projects_subdir = workspace_root / "projects"
    if projects_subdir.exists() and projects_subdir.is_dir():
        for sub in projects_subdir.iterdir():
            if sub.is_dir():
                candidates.append(sub.resolve())

    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for c in candidates:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        info = _inspect_project_dir(c)
        info["is_active"] = str(c).lower() == str(Path(active_proj_dir).resolve()).lower() if active_proj_dir else False
        results.append(info)

    results.sort(key=lambda r: (r["is_active"], r["has_rendered_video"], r["updated_timestamp"]), reverse=True)
    return results


def _reset_project_directory(proj_dir: Path) -> dict[str, Any]:
    """Delete all generated artifacts, videos, and caches inside proj_dir while keeping project_meta.json."""
    proj = proj_dir.resolve()
    if not proj.exists() or not proj.is_dir():
        proj.mkdir(parents=True, exist_ok=True)
        return _inspect_project_dir(proj)

    meta = _read_json_if_exists(proj / "project_meta.json") or {}
    bb = _read_json_if_exists(proj / "blackboard.json") or {}
    if not meta.get("media_dir") and bb.get("media_dir"):
        meta["media_dir"] = bb["media_dir"]
    if not meta.get("name"):
        meta["name"] = proj.name.replace("_project", "").replace("_", " ").title()

    removed_files: list[str] = []
    exts_to_clear = {".json", ".yaml", ".yml", ".md", ".mp4", ".fcpxml", ".edl", ".otio", ".srt", ".ass", ".wav", ".mp3"}
    for item in list(proj.iterdir()):
        if item.is_file() and item.name != "project_meta.json" and item.suffix.lower() in exts_to_clear:
            try:
                item.unlink()
                removed_files.append(item.name)
            except Exception:
                pass
        elif item.is_dir() and item.name in ("render_cache", "previews", "bgm_library", "keyframes"):
            try:
                shutil.rmtree(item, ignore_errors=True)
                removed_files.append(f"{item.name}/")
            except Exception:
                pass

    _write_project_meta(proj, meta)
    info = _inspect_project_dir(proj)
    info["removed_items"] = removed_files
    return info


def _discover_local_media_libraries(workspace_root: Path) -> list[dict[str, Any]]:
    """Scan common local folders for ready-to-edit media libraries."""
    candidates: list[Path] = []
    projects_parent = workspace_root.parent

    # 1. Check sibling clips/* folders
    clips_root = projects_parent / "clips"
    if clips_root.exists() and clips_root.is_dir():
        for sub in sorted(clips_root.iterdir()):
            if sub.is_dir() and not sub.name.startswith("."):
                candidates.append(sub)
        if not candidates:
            candidates.append(clips_root)

    # 2. Check sibling ai_editor/raw_media
    raw_media = projects_parent / "ai_editor" / "raw_media"
    if raw_media.exists() and raw_media.is_dir():
        candidates.append(raw_media)

    # 3. Check any local media_library or raw_media inside workspace_root
    for local_name in ("raw_media", "media", "clips"):
        p = workspace_root / local_name
        if p.exists() and p.is_dir() and p not in candidates:
            candidates.append(p)

    libraries: list[dict[str, Any]] = []
    seen_resolved: set[str] = set()
    for c in candidates:
        try:
            res_str = str(c.resolve())
        except Exception:
            continue
        if res_str in seen_resolved:
            continue
        seen_resolved.add(res_str)
        info = inspect_media_library(c, workspace_root=workspace_root)
        if info["video_count"] > 0 or info["photo_count"] > 0:
            libraries.append(info)

    return libraries


def create_app(default_project_dir: str | Path = "active_video_project") -> FastAPI:
    app = FastAPI(title="AI Photo & Video Editing Studio (Guided Multi-Agent Edition)", version="0.3.0")
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    workspace_root = Path.cwd().resolve()
    default_proj_resolved = Path(default_project_dir).resolve()

    state: dict[str, Any] = {
        "project_dir": str(default_proj_resolved),
        "last_status": "Ready",
        "progress": 0.0,
    }

    job_lock = threading.Lock()
    job_state: dict[str, Any] = {
        "job_id": None,
        "status": "idle",  # idle | running | completed | error
        "error": None,
        "media_dir": "",
        "project_dir": str(default_proj_resolved),
        "progress": 0.0,
        "progress_pct": 0,
        "current_role_id": "",
        "current_stage_title": "Ready to start",
        "last_status": "Select your media library and click Run to start.",
        "started_at": None,
        "completed_at": None,
        "elapsed_sec": 0.0,
        "stages": [],
        "activity_log": [],
        "agent_activities": [],
        "director_thoughts": [],
    }

    def _elapsed_str() -> tuple[str, float]:
        started = job_state.get("started_at") or time.monotonic()
        elapsed = max(0.0, time.monotonic() - started)
        mm = int(elapsed // 60)
        ss = int(elapsed % 60)
        return f"{mm:02d}:{ss:02d}", round(elapsed, 1)

    def _append_log(role_id: str, message: str, level: str = "info") -> None:
        t_str, elapsed = _elapsed_str()
        entry = {
            "time_str": t_str,
            "elapsed_sec": elapsed,
            "role_id": role_id,
            "message": message,
            "level": level,
        }
        job_state["activity_log"].append(entry)
        if len(job_state["activity_log"]) > 160:
            job_state["activity_log"] = job_state["activity_log"][-160:]

    def _append_agent_activity(
        role_id: str,
        action: str,
        detail: str = "",
        event_type: str = "agent_action",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        t_str, elapsed = _elapsed_str()
        item: dict[str, Any] = {
            "time_str": t_str,
            "elapsed_sec": elapsed,
            "role_id": role_id,
            "action": action,
            "detail": detail,
            "event_type": event_type,
        }
        if extra:
            item.update(extra)
        job_state["agent_activities"].append(item)
        if len(job_state["agent_activities"]) > 180:
            job_state["agent_activities"] = job_state["agent_activities"][-180:]

    def _append_director_thought(ev: dict[str, Any]) -> None:
        t_str, elapsed = _elapsed_str()
        thought = {
            "time_str": t_str,
            "elapsed_sec": elapsed,
            "reviewed_role": ev.get("reviewed_role", ev.get("role_id", "")),
            "reviewed_display_name": ev.get("reviewed_display_name", ev.get("reviewed_role", "")),
            "stage_order": ev.get("stage_order", 0),
            "status": ev.get("status", "approved"),
            "quality_score": ev.get("quality_score", 9.5),
            "observation": ev.get("observation", ""),
            "thinking": ev.get("thinking", ""),
            "decision": ev.get("decision", ev.get("detail", "")),
            "critique": ev.get("critique", ""),
            "directive_sent": ev.get("directive_sent"),
        }
        job_state["director_thoughts"].append(thought)
        if len(job_state["director_thoughts"]) > 80:
            job_state["director_thoughts"] = job_state["director_thoughts"][-80:]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_file = static_dir / "index.html"
        return index_file.read_text(encoding="utf-8")

    @app.get("/api/agents")
    def api_list_agents() -> dict:
        return {
            "agents": [cls.to_metadata_dict() for cls in AgentRegistry.list_all()],
        }

    @app.get("/api/fs/libraries")
    def api_list_media_libraries() -> dict:
        libs = _discover_local_media_libraries(workspace_root)
        return {
            "workspace_root": str(workspace_root),
            "libraries": libs,
        }

    @app.get("/api/projects")
    def api_list_projects() -> dict:
        projects = _list_workspace_projects(workspace_root, state["project_dir"])
        return {
            "active_project_dir": state["project_dir"],
            "projects": projects,
        }

    @app.post("/api/projects/create")
    def api_create_project(req: ProjectCreateRequest) -> dict:
        raw_name = (req.project_name or req.name).strip()
        slug = _slugify_project_name(raw_name or "new_story")
        proj = Path(req.project_dir).resolve() if req.project_dir else (workspace_root / slug).resolve()
        proj.mkdir(parents=True, exist_ok=True)

        media_dir_str = req.media_dir.strip()
        if media_dir_str and Path(media_dir_str).exists():
            media_dir_str = str(Path(media_dir_str).resolve())

        lib_info = inspect_media_library(media_dir_str, workspace_root=workspace_root) if media_dir_str else {}
        prompt_str = req.user_prompt.strip() or lib_info.get("default_prompt", "")
        display_name = raw_name or slug.replace("_project", "").replace("_", " ").title()

        _write_project_meta(
            proj,
            {
                "name": display_name,
                "media_dir": media_dir_str,
                "user_prompt": prompt_str,
                "status": "empty",
            },
        )
        state["project_dir"] = str(proj)
        state["last_status"] = f"Created new project '{display_name}'"
        state["progress"] = 0.0

        with job_lock:
            if job_state["status"] != "running":
                job_state["status"] = "idle"
                job_state["project_dir"] = str(proj)
                job_state["media_dir"] = media_dir_str
                job_state["progress"] = 0.0
                job_state["progress_pct"] = 0
                job_state["activity_log"] = []
                job_state["agent_activities"] = []
                job_state["director_thoughts"] = []

        st_data = get_state(str(proj))
        st_data["project"] = st_data["project_summary"]
        return st_data

    @app.post("/api/projects/load")
    def api_load_project(req: ProjectDirRequest) -> dict:
        proj = Path(req.project_dir).resolve()
        proj.mkdir(parents=True, exist_ok=True)
        state["project_dir"] = str(proj)
        info = _inspect_project_dir(proj)
        state["last_status"] = f"Loaded project '{info['name']}'"
        st_data = get_state(str(proj))
        st_data["project"] = st_data["project_summary"]
        return st_data

    @app.post("/api/projects/reset")
    def api_reset_project(req: ProjectDirRequest) -> dict:
        proj = Path(req.project_dir).resolve()
        with job_lock:
            if job_state["status"] == "running" and str(Path(job_state["project_dir"]).resolve()) == str(proj):
                raise HTTPException(status_code=400, detail="Cannot reset project while a pipeline job is actively running.")
            reset_info = _reset_project_directory(proj)
            if job_state["status"] != "running":
                job_state["status"] = "idle"
                job_state["progress"] = 0.0
                job_state["progress_pct"] = 0
                job_state["current_stage_title"] = "Project Reset — Ready for a fresh edit"
                job_state["last_status"] = f"Cleared generated artifacts from '{reset_info['name']}'"
                job_state["activity_log"] = []
                job_state["agent_activities"] = []
                job_state["director_thoughts"] = []
                for s in job_state.get("stages", []):
                    s["status"] = "pending"
                    s["duration_sec"] = None
                    s["quality_score"] = None

        state["project_dir"] = str(proj)
        removed_count = len(reset_info.get("removed_items", []))
        state["last_status"] = f"Reset project '{reset_info['name']}' ({removed_count} items cleared)"
        state["progress"] = 0.0
        st_data = get_state(str(proj))
        st_data["project"] = st_data["project_summary"]
        st_data["reset_summary"] = reset_info
        st_data["removed_files_count"] = removed_count
        return st_data

    @app.get("/api/projects/download-video")
    def api_download_project_video(project_dir: Optional[str] = Query(default=None)):
        proj = Path(project_dir or state["project_dir"]).resolve()
        mp4_path = proj / "final_story.mp4"
        if not mp4_path.exists():
            matches = sorted(proj.glob("*.mp4"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
            if matches:
                mp4_path = matches[0]
        if not mp4_path.exists() or not mp4_path.is_file():
            raise HTTPException(status_code=404, detail=f"No rendered video (.mp4) found in project: {proj.name}")

        clean_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", proj.name.replace("_project", "")).strip("_") or "story"
        download_name = f"{clean_slug}_{mp4_path.name}"
        return FileResponse(
            str(mp4_path),
            media_type="video/mp4",
            filename=download_name,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.get("/api/fs/inspect")
    def api_inspect_media_library(media_dir: str = Query(..., description="Path to media folder")) -> dict:
        return inspect_media_library(media_dir, workspace_root=workspace_root)

    @app.get("/api/fs/browse")
    def api_browse_directory(path: Optional[str] = Query(default=None)) -> dict:
        target = Path(path).resolve() if path else workspace_root.parent.resolve()
        if not target.exists() or not target.is_dir():
            target = workspace_root.parent.resolve()

        parent_path = str(target.parent.resolve()) if target.parent != target else None
        subdirs: list[dict[str, Any]] = []
        direct_videos = 0
        direct_photos = 0

        try:
            for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
                if entry.name.startswith(".") or entry.name in ("__pycache__", "node_modules", ".git", ".venv"):
                    continue
                if entry.is_dir():
                    v_c = 0
                    p_c = 0
                    try:
                        for child in entry.iterdir():
                            if child.is_file():
                                ext = child.suffix.lower()
                                if ext in VIDEO_EXTENSIONS:
                                    v_c += 1
                                elif ext in PHOTO_EXTENSIONS:
                                    p_c += 1
                            elif child.is_dir() and not child.name.startswith("."):
                                for gchild in child.iterdir():
                                    if gchild.is_file():
                                        ext = gchild.suffix.lower()
                                        if ext in VIDEO_EXTENSIONS:
                                            v_c += 1
                                        elif ext in PHOTO_EXTENSIONS:
                                            p_c += 1
                    except Exception:
                        pass
                    subdirs.append(
                        {
                            "name": entry.name,
                            "path": str(entry.resolve()),
                            "video_count": v_c,
                            "photo_count": p_c,
                        }
                    )
                elif entry.is_file():
                    ext = entry.suffix.lower()
                    if ext in VIDEO_EXTENSIONS:
                        direct_videos += 1
                    elif ext in PHOTO_EXTENSIONS:
                        direct_photos += 1
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Cannot browse folder: {exc}") from exc

        return {
            "current_path": str(target),
            "parent_path": parent_path,
            "direct_videos": direct_videos,
            "direct_photos": direct_photos,
            "subdirectories": subdirs[:80],
        }

    @app.post("/api/fs/pick-folder")
    def api_pick_folder_native(initial_dir: Optional[str] = None) -> dict:
        """Open a native OS folder selection dialog on the local machine."""
        if sys.platform == "darwin":
            try:
                proc = subprocess.run(
                    ["osascript", "-e", 'POSIX path of (choose folder with prompt "Select Media Library Folder")'],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if proc.returncode != 0 or not proc.stdout.strip():
                    return {"cancelled": True, "selected_path": None}
                chosen = proc.stdout.strip()
                info = inspect_media_library(chosen, workspace_root=workspace_root)
                return {"cancelled": False, "selected_path": info["media_dir"], "library": info}
            except Exception:
                pass
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            start_dir = initial_dir if (initial_dir and Path(initial_dir).exists()) else str(workspace_root.parent)
            chosen = filedialog.askdirectory(initialdir=start_dir, title="Select Media Library Folder")
            root.destroy()
            if not chosen:
                return {"cancelled": True, "selected_path": None}
            info = inspect_media_library(chosen, workspace_root=workspace_root)
            return {"cancelled": False, "selected_path": info["media_dir"], "library": info}
        except Exception as exc:
            return {
                "cancelled": True,
                "selected_path": None,
                "error": f"Native folder dialog unavailable ({exc}). Please use the in-browser Folder Explorer.",
            }

    @app.get("/api/intake-questions")
    def api_intake_questions(
        media_dir: str = Query(default=""),
        prompt: str = Query(default=""),
    ) -> dict:
        if not media_dir or not Path(media_dir).exists():
            libs = _discover_local_media_libraries(workspace_root)
            media_dir = libs[0]["media_dir"] if libs else str(workspace_root)
        lib_info = inspect_media_library(media_dir, workspace_root=workspace_root)
        questions = generate_intake_questions(media_dir=media_dir, initial_prompt=prompt)
        initial_refinement = refine_intake_interactively(
            media_dir=media_dir,
            user_message=prompt or lib_info.get("default_prompt", ""),
            interview_answers={q.id: q.default_answer for q in questions},
        )
        return {
            "library": lib_info,
            "questions": [q.model_dump(mode="json") for q in questions],
            "initial_refinement": initial_refinement,
        }

    @app.post("/api/intake/refine")
    def api_intake_refine(req: IntakeRefineRequest) -> dict:
        return refine_intake_interactively(
            media_dir=req.media_dir,
            user_message=req.user_message,
            interview_answers=req.interview_answers,
            chat_history=req.chat_history,
        )

    @app.get("/api/jobs/status")
    def api_job_status() -> dict:
        with job_lock:
            if job_state["status"] == "running" and job_state["started_at"] is not None:
                job_state["elapsed_sec"] = round(time.monotonic() - job_state["started_at"], 1)
            return dict(job_state)

    @app.post("/api/jobs/start")
    def api_job_start(req: MultiAgentRunRequest, wait: bool = Query(default=False)) -> dict:
        media_path = Path(req.media_dir).resolve()
        if not media_path.exists() or not media_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Media directory not found: {req.media_dir}")

        with job_lock:
            if job_state["status"] == "running":
                return dict(job_state)

            proj = Path(req.project_dir).resolve()
            state["project_dir"] = str(proj)
            _write_project_meta(
                proj,
                {
                    "media_dir": str(media_path),
                    "user_prompt": req.user_prompt,
                    "interview_answers": req.interview_answers,
                    "status": "running",
                },
            )

            orchestrator = MultiAgentOrchestrator(
                enabled_optional_roles=req.enabled_optional_roles,
                disabled_roles=req.disabled_roles,
            )
            planned_agents = orchestrator.get_execution_plan()
            stages_init: list[dict[str, Any]] = [
                {
                    "role_id": a.role_id,
                    "display_name": a.display_name,
                    "stage_order": a.stage_order,
                    "status": "pending",
                    "duration_sec": None,
                    "quality_score": None,
                    "critique": a.description,
                    "directive_sent": None,
                }
                for a in planned_agents
            ]
            if req.render_video:
                stages_init.append(
                    {
                        "role_id": "renderer",
                        "display_name": "7. 4K 10-bit HLG HDR Video & NLE Renderer",
                        "stage_order": 70,
                        "status": "pending",
                        "duration_sec": None,
                        "quality_score": None,
                        "critique": "Synthesizes 2.5D parallax & diptych photo plates, titles, and 3-layer audio into 4K HDR MP4.",
                        "directive_sent": None,
                    }
                )

            job_id = f"job_{uuid.uuid4().hex[:8]}"
            now_mono = time.monotonic()
            job_state.update(
                {
                    "job_id": job_id,
                    "status": "running",
                    "error": None,
                    "media_dir": str(media_path),
                    "project_dir": str(proj),
                    "progress": 0.01,
                    "progress_pct": 1,
                    "current_role_id": planned_agents[0].role_id if planned_agents else "client_intake",
                    "current_stage_title": planned_agents[0].display_name if planned_agents else "Initializing...",
                    "last_status": "Initializing Multi-Agent Pipeline...",
                    "started_at": now_mono,
                    "completed_at": None,
                    "elapsed_sec": 0.0,
                    "stages": stages_init,
                    "activity_log": [],
                    "agent_activities": [],
                    "director_thoughts": [],
                }
            )
            _append_log("system", f"Started job {job_id} on media library '{media_path.name}' -> '{proj.name}'")
            _append_agent_activity(
                "orchestrator",
                f"Launched Multi-Agent Pipeline ({len(stages_init)} stages) on '{media_path.name}'",
                detail=f"Output project: {proj.name} | 4K 10-bit HLG HDR: {req.hdr}",
                event_type="stage_started",
            )

        def _worker() -> None:
            extra_audio = (
                [req.extra_audio_path]
                if req.extra_audio_path and Path(req.extra_audio_path).exists()
                else None
            )

            def _on_prog(role_id: str, msg: str, pct: int) -> None:
                with job_lock:
                    state["last_status"] = f"[{role_id}] {msg}"
                    state["progress"] = round(pct / 100.0, 2)
                    job_state["progress"] = round(pct / 100.0, 2)
                    job_state["progress_pct"] = int(pct)
                    job_state["last_status"] = msg
                    if job_state["started_at"] is not None:
                        job_state["elapsed_sec"] = round(time.monotonic() - job_state["started_at"], 1)

            def _on_event(ev: dict[str, Any]) -> None:
                with job_lock:
                    etype = ev.get("event", "")
                    rid = ev.get("role_id", "")
                    pct = int(ev.get("progress_pct", job_state["progress_pct"]))
                    msg = ev.get("message", "")
                    job_state["progress_pct"] = pct
                    job_state["progress"] = round(pct / 100.0, 2)
                    if msg:
                        job_state["last_status"] = msg

                    stage_obj = next((s for s in job_state["stages"] if s["role_id"] == rid), None)
                    if etype == "stage_started":
                        job_state["current_role_id"] = rid
                        job_state["current_stage_title"] = ev.get("display_name", rid)
                        if stage_obj is not None:
                            stage_obj["status"] = "running"
                        _append_log(rid, msg)
                        _append_agent_activity(
                            rid,
                            f"▶ Started {ev.get('display_name', rid)}",
                            detail=stage_obj["critique"] if stage_obj else "",
                            event_type="stage_started",
                        )
                    elif etype in ("agent_action", "bus_message"):
                        act_str = ev.get("action", msg)
                        det_str = ev.get("detail", "")
                        _append_agent_activity(rid, act_str, detail=det_str, event_type=etype)
                        _append_log(rid, f"{act_str}" + (f" ({det_str})" if det_str else ""))
                    elif etype == "director_thought":
                        _append_director_thought(ev)
                        _append_log(
                            "production_director",
                            f"[Director Thought on {ev.get('reviewed_role', '')}] {ev.get('decision', '')}",
                            level="directive" if ev.get("directive_sent") else "info",
                        )
                    elif etype == "director_directive":
                        if stage_obj is not None:
                            stage_obj["status"] = "directed_improvement"
                            stage_obj["directive_sent"] = ev.get("directive_sent")
                            stage_obj["critique"] = ev.get("critique", stage_obj["critique"])
                            stage_obj["quality_score"] = ev.get("quality_score")
                        _append_log("production_director", msg, level="directive")
                        _append_agent_activity(
                            rid,
                            f"🔄 Re-running {ev.get('display_name', rid)} per Executive Director Directive",
                            detail=str(ev.get("directive_sent", "")),
                            event_type="director_directive",
                        )
                    elif etype == "stage_completed":
                        if stage_obj is not None:
                            rev_status = ev.get("review_status", "approved")
                            stage_obj["status"] = (
                                rev_status
                                if rev_status in ("directed_improvement", "polished_by_director")
                                else "completed"
                            )
                            stage_obj["duration_sec"] = ev.get("duration_sec")
                            stage_obj["quality_score"] = ev.get("quality_score")
                            stage_obj["critique"] = ev.get("critique", stage_obj["critique"])
                            if ev.get("directive_sent"):
                                stage_obj["directive_sent"] = ev.get("directive_sent")
                        _append_log(rid, f"{msg} — {ev.get('critique', '')}", level="success")
                    elif etype == "render_progress":
                        job_state["current_role_id"] = "renderer"
                        job_state["current_stage_title"] = "7. 4K 10-bit HLG HDR Video & NLE Renderer"
                        if stage_obj is not None:
                            stage_obj["status"] = "running"
                            stage_obj["critique"] = msg
                        _append_log("renderer", msg)
                        _append_agent_activity("renderer", msg, detail="4K 3840x2160 10-bit HLG HDR synthesis", event_type="render_progress")
                    elif etype == "pipeline_complete":
                        _append_log("complete", msg, level="success")
                        _append_agent_activity("complete", msg, detail="All deliverables & NLE exports ready", event_type="pipeline_complete")

            try:
                bb = run_multi_agent_pipeline(
                    media_dir=media_path,
                    project_dir=proj,
                    user_prompt=req.user_prompt,
                    extra_audio_paths=extra_audio,
                    interview_answers=req.interview_answers,
                    enabled_optional_roles=req.enabled_optional_roles,
                    disabled_roles=req.disabled_roles,
                    width=req.width,
                    height=req.height,
                    fps=req.fps,
                    hdr=req.hdr,
                    use_antigravity=req.use_antigravity,
                    render_video=req.render_video,
                    progress_callback=_on_prog,
                    event_callback=_on_event,
                )
                with job_lock:
                    end_mono = time.monotonic()
                    job_state["status"] = "completed"
                    job_state["progress"] = 1.0
                    job_state["progress_pct"] = 100
                    job_state["completed_at"] = end_mono
                    if job_state["started_at"] is not None:
                        job_state["elapsed_sec"] = round(end_mono - job_state["started_at"], 1)
                    dr = bb.director_review
                    grade_str = f"Grade {dr.quality_grade} ({dr.overall_quality_score}/10)" if dr else "Complete"
                    job_state["current_stage_title"] = f"All Stages Complete • {grade_str}"
                    job_state["last_status"] = (
                        f"Finished in {job_state['elapsed_sec']}s — {grade_str}"
                    )
                    state["last_status"] = job_state["last_status"]
                    state["progress"] = 1.0
            except Exception as exc:
                with job_lock:
                    job_state["status"] = "error"
                    job_state["error"] = str(exc)
                    job_state["last_status"] = f"Error: {exc}"
                    state["last_status"] = f"Error: {exc}"
                    _append_log("error", f"Pipeline failed: {exc}", level="error")

        if wait:
            _worker()
        else:
            t = threading.Thread(target=_worker, daemon=True)
            t.start()

        with job_lock:
            return dict(job_state)

    @app.get("/api/state")
    def get_state(project_dir: Optional[str] = None) -> dict:
        proj = Path(project_dir or state["project_dir"]).resolve()
        state["project_dir"] = str(proj)

        ann_path = proj / "annotations.yaml"
        tl_path = proj / "timeline.yaml"
        sb_path = proj / "storyboard.md"
        mp4_path = proj / "final_story.mp4"

        catalog_data = None
        annotations_yaml = ""
        if ann_path.exists():
            annotations_yaml = ann_path.read_text(encoding="utf-8")
            try:
                catalog_data = MediaCatalog.from_yaml(ann_path).model_dump(mode="json")
            except Exception:
                pass

        timeline_data = None
        timeline_yaml = ""
        if tl_path.exists():
            timeline_yaml = tl_path.read_text(encoding="utf-8")
            try:
                timeline_data = EditTimeline.from_yaml(tl_path).model_dump(mode="json")
            except Exception:
                pass

        storyboard_md = sb_path.read_text(encoding="utf-8") if sb_path.exists() else ""

        exports = {}
        for ext in (".mp4", ".fcpxml", ".edl", ".otio", ".srt", ".ass"):
            matches = list(proj.glob(f"*{ext}"))
            if matches:
                exports[ext.lstrip(".")] = str(matches[0].resolve())

        agent_contracts = {
            "creative_brief": _read_json_if_exists(proj / "creative_brief.json"),
            "bgm_selection": _read_json_if_exists(proj / "bgm_selection.json"),
            "media_index": _read_json_if_exists(proj / "media_index.json"),
            "photo_curation": _read_json_if_exists(proj / "photo_curation.json"),
            "contact_sheets_manifest": _read_json_if_exists(
                proj / "contact_sheets" / "contact_sheets_manifest.json"
            ),
            "beat_grid": _read_json_if_exists(proj / "beat_grid.json"),
            "paper_edit": _read_json_if_exists(proj / "paper_edit.json"),
            "candidate_story_digest": _read_json_if_exists(proj / "candidate_story_digest.json"),
            "pacing_audit": _read_json_if_exists(proj / "pacing_audit.json"),
            "director_review": _read_json_if_exists(proj / "director_review.json"),
            "vfx_manifest": _read_json_if_exists(proj / "vfx_manifest.json"),
            "blackboard": _read_json_if_exists(proj / "blackboard.json"),
        }

        with job_lock:
            current_job = dict(job_state)

        proj_summary = _inspect_project_dir(proj)
        all_projects = _list_workspace_projects(workspace_root, str(proj))

        return {
            "project_dir": str(proj),
            "project_summary": proj_summary,
            "projects": all_projects,
            "last_status": state["last_status"],
            "progress": state["progress"],
            "job": current_job,
            "registered_agents": [cls.to_metadata_dict() for cls in AgentRegistry.list_all()],
            "agent_contracts": agent_contracts,
            "has_annotations": ann_path.exists(),
            "annotations_yaml": annotations_yaml,
            "catalog": catalog_data,
            "has_timeline": tl_path.exists(),
            "timeline_yaml": timeline_yaml,
            "timeline": timeline_data,
            "storyboard_md": storyboard_md,
            "has_rendered_video": mp4_path.exists() or ("mp4" in exports),
            "exports": exports,
        }

    @app.post("/api/agents/run")
    def api_run_multi_agent(req: MultiAgentRunRequest) -> dict:
        media_path = Path(req.media_dir).resolve()
        if not media_path.exists() or not media_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Media directory not found: {req.media_dir}")

        proj = Path(req.project_dir).resolve()
        state["project_dir"] = str(proj)

        extra_audio = [req.extra_audio_path] if req.extra_audio_path and Path(req.extra_audio_path).exists() else None

        def _on_prog(role_id: str, msg: str, pct: int) -> None:
            state["last_status"] = f"[{role_id}] {msg}"
            state["progress"] = round(pct / 100.0, 2)

        blackboard = run_multi_agent_pipeline(
            media_dir=media_path,
            project_dir=proj,
            user_prompt=req.user_prompt,
            extra_audio_paths=extra_audio,
            interview_answers=req.interview_answers,
            enabled_optional_roles=req.enabled_optional_roles,
            disabled_roles=req.disabled_roles,
            width=req.width,
            height=req.height,
            fps=req.fps,
            hdr=req.hdr,
            use_antigravity=req.use_antigravity,
            render_video=req.render_video,
            progress_callback=_on_prog,
        )

        state["last_status"] = (
            f"Multi-Agent Pipeline Complete ({len(blackboard.executed_agents)} agents executed)"
        )
        state["progress"] = 1.0
        return get_state(str(proj))

    @app.post("/api/analyze")
    def api_analyze(req: AnalyzeRequest) -> dict:
        media_path = Path(req.media_dir).resolve()
        if not media_path.exists() or not media_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Media directory not found: {req.media_dir}")

        proj = Path(req.project_dir).resolve()
        state["project_dir"] = str(proj)
        state["last_status"] = "Analyzing videos & photos..."
        state["progress"] = 0.2

        extra_audio = [req.extra_audio_path] if req.extra_audio_path else None
        catalog = analyze_media_directory(
            media_dir=media_path,
            project_dir=proj,
            whisper_model=req.whisper_model,
            use_antigravity=req.use_antigravity,
            user_hint=req.user_hint,
            extra_audio_paths=extra_audio,
        )
        state["last_status"] = "Step 1 Analysis Complete"
        state["progress"] = 1.0
        ann_path = proj / "annotations.yaml"
        return {
            "catalog": catalog.model_dump(mode="json"),
            "annotations_yaml": ann_path.read_text(encoding="utf-8"),
        }

    @app.put("/api/annotations")
    def api_save_annotations(req: SaveYamlRequest) -> dict:
        proj = Path(req.project_dir).resolve()
        ann_path = proj / "annotations.yaml"
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(req.yaml_text, encoding="utf-8")
        try:
            cat = MediaCatalog.from_yaml(ann_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid annotations YAML: {exc}") from exc
        return {"catalog": cat.model_dump(mode="json"), "annotations_yaml": req.yaml_text}

    @app.post("/api/plan")
    def api_plan(req: PlanRequest) -> dict:
        proj = Path(req.project_dir).resolve()
        ann_path = proj / "annotations.yaml"
        if not ann_path.exists():
            raise HTTPException(status_code=400, detail="annotations.yaml not found. Run Step 1 Analyze first.")

        state["last_status"] = "Generating AI Story Plan & Timeline..."
        catalog = MediaCatalog.from_yaml(ann_path)
        timeline, tl_path, sb_path = generate_edit_plan(
            catalog=catalog,
            user_prompt=req.user_prompt,
            project_dir=proj,
            target_duration=req.target_duration,
            width=req.width,
            height=req.height,
            fps=req.fps,
            hdr=req.hdr,
            photo_mode=req.photo_mode,
            use_antigravity=req.use_antigravity,
        )
        export_all_nle_formats(timeline, proj)
        state["last_status"] = "Step 2 Story Timeline Ready"
        return {
            "timeline": timeline.model_dump(mode="json"),
            "timeline_yaml": tl_path.read_text(encoding="utf-8"),
            "storyboard_md": sb_path.read_text(encoding="utf-8"),
        }

    @app.put("/api/timeline")
    def api_save_timeline(req: SaveYamlRequest) -> dict:
        proj = Path(req.project_dir).resolve()
        tl_path = proj / "timeline.yaml"
        sb_path = proj / "storyboard.md"
        tl_path.parent.mkdir(parents=True, exist_ok=True)
        tl_path.write_text(req.yaml_text, encoding="utf-8")
        try:
            timeline = EditTimeline.from_yaml(tl_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid timeline YAML: {exc}") from exc

        sb_md = generate_storyboard_markdown(timeline)
        sb_path.write_text(sb_md, encoding="utf-8")
        export_all_nle_formats(timeline, proj)
        return {
            "timeline": timeline.model_dump(mode="json"),
            "timeline_yaml": req.yaml_text,
            "storyboard_md": sb_md,
        }

    @app.post("/api/render")
    def api_render(req: RenderRequest) -> dict:
        proj = Path(req.project_dir).resolve()
        tl_path = proj / "timeline.yaml"
        if not tl_path.exists():
            raise HTTPException(status_code=400, detail="timeline.yaml not found. Run Step 2 Plan first.")

        timeline = EditTimeline.from_yaml(tl_path)
        out_mp4 = proj / req.output_filename

        def _on_prog(msg: str, frac: float) -> None:
            state["last_status"] = msg
            state["progress"] = round(frac, 2)

        render_outputs = render_timeline(timeline, out_mp4, progress_callback=_on_prog)
        nle_outputs = export_all_nle_formats(timeline, proj)
        state["last_status"] = f"Rendered {out_mp4.name} ({timeline.total_duration:.1f}s)"
        state["progress"] = 1.0

        return {
            "video_path": str(render_outputs["video"]),
            "srt_path": str(render_outputs["srt"]),
            "ass_path": str(render_outputs["ass"]),
            "fcpxml_path": str(nle_outputs["fcpxml"]),
            "edl_path": str(nle_outputs["edl"]),
            "otio_path": str(nle_outputs["otio"]),
        }

    @app.post("/api/curation/contact-sheets")
    def api_render_contact_sheets(req: ContactSheetsRequest) -> dict:
        proj = Path(req.project_dir).resolve()
        return render_curation_contact_sheets(proj, max_items=req.max_items)

    @app.post("/api/curation/vision-overrides")
    def api_apply_vision_overrides(req: VisionOverridesRequest) -> dict:
        proj = Path(req.project_dir).resolve()
        return apply_vision_curation_overrides(
            proj,
            overrides=req.overrides,
            curatorial_statement=req.curatorial_statement,
        )

    @app.post("/api/timeline/surgical-edit")
    def api_surgical_edit(req: SurgicalEditRequest) -> dict:
        proj = Path(req.project_dir).resolve()
        try:
            return apply_surgical_edit(proj, operations=req.operations)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/render/segment")
    def api_render_segment(req: SegmentRenderRequest) -> dict:
        proj = Path(req.project_dir).resolve()
        tl_path = proj / "timeline.yaml"
        if not tl_path.exists():
            raise HTTPException(status_code=400, detail="timeline.yaml not found.")
        timeline = EditTimeline.from_yaml(tl_path)
        fname = (
            req.output_filename
            or f"segment_{int(req.start_sec):03d}s_{int(req.end_sec):03d}s.mp4"
        )
        out_mp4 = proj / "segments" / fname
        return render_timeline_segment(
            timeline=timeline,
            output_mp4_path=out_mp4,
            start_sec=req.start_sec,
            end_sec=req.end_sec,
            fast_preview=req.fast_preview,
        )

    @app.get("/api/file")
    def api_serve_file(
        path: str = Query(..., description="Absolute path to media or export file"),
        download: bool = Query(default=False, description="Force file attachment download"),
    ):
        target = Path(path).resolve()
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        if download:
            return FileResponse(
                str(target),
                filename=target.name,
                headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
            )
        return FileResponse(
            str(target),
            content_disposition_type="inline",
            headers={"Cache-Control": "no-cache"},
        )

    return app
