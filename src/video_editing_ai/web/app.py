"""FastAPI backend for inspecting annotations, editing text timelines, and rendering videos."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..analyzer.ag_annotator import analyze_media_directory
from ..exporters.nle_export import export_all_nle_formats
from ..models.schema import EditTimeline, MediaCatalog
from ..planner.story_director import generate_edit_plan, generate_storyboard_markdown
from ..renderer.ffmpeg_composer import render_timeline


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
    photo_mode: str = "gallery_frame"
    use_antigravity: bool = True


class RenderRequest(BaseModel):
    project_dir: str = "workspace_project"
    output_filename: str = "final_story.mp4"


def create_app(default_project_dir: str | Path = "workspace_project") -> FastAPI:
    app = FastAPI(title="AI Photo & Video Editing Studio", version="0.1.0")
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    state = {
        "project_dir": str(Path(default_project_dir).resolve()),
        "last_status": "Ready",
        "progress": 0.0,
    }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_file = static_dir / "index.html"
        return index_file.read_text(encoding="utf-8")

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

        return {
            "project_dir": str(proj),
            "last_status": state["last_status"],
            "progress": state["progress"],
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

    @app.get("/api/file")
    def api_serve_file(path: str = Query(..., description="Absolute path to media or export file")):
        target = Path(path).resolve()
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        return FileResponse(str(target), filename=target.name)

    return app
