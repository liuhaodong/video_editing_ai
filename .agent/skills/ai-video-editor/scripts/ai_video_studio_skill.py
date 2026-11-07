#!/usr/bin/env python3
"""Antigravity Skill Helper CLI for the 11-Agent AI Photo & Video Editing Studio.

Provides token-efficient, file-backed subcommands for every stage of the editing
workflow:
  - inspect-library : Scan raw media folder, detect cameras, and compute Director's recommended length
  - intake-refine   : Co-refine Story Vision & CreativeBrief with Client Intake Agent + Director
  - run-stage       : Run any individual sub-agent stage(s) with Executive Director stage-gate review
  - run-pipeline    : Run the complete 11-agent pipeline (Stage 00 -> Stage 60) + 4K HLG HDR render
  - director-audit  : Run or inspect the Stage 60 Executive Movie Production Director screening
  - render-video    : Render timeline.yaml to 4K 10-bit HLG HDR MP4 and extract verification keyframes
  - project         : Manage editing projects (list, create, load, reset)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

# Resolve repository src/ path whether invoked from workspace .agent/skills/ or global ~/.gemini/config/skills/
_CANDIDATE_REPO_ROOTS = [
    Path(__file__).resolve().parents[4],
    Path(r"C:\Users\hihao\Documents\projects\video_editing_ai"),
]
for _root in _CANDIDATE_REPO_ROOTS:
    _src = _root / "src"
    if (_src / "video_editing_ai").exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
        break

from video_editing_ai.agents import (  # noqa: E402
    AgentRegistry,
    PipelineBlackboard,
    ProductionDirectorAgent,
    compute_director_duration_recommendation,
    generate_intake_questions,
    inspect_media_library,
    refine_intake_interactively,
    run_multi_agent_pipeline,
)
from video_editing_ai.exporters.nle_export import export_all_nle_formats  # noqa: E402
from video_editing_ai.models.schema import CreativeBrief, EditTimeline  # noqa: E402
from video_editing_ai.planner.story_director import generate_storyboard_markdown  # noqa: E402
from video_editing_ai.renderer.ffmpeg_composer import render_timeline  # noqa: E402
from video_editing_ai.web.app import (  # noqa: E402
    _inspect_project_dir,
    _list_workspace_projects,
    _reset_project_directory,
    _slugify_project_name,
    _write_project_meta,
)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def cmd_inspect_library(args: argparse.Namespace) -> int:
    media_dir = Path(args.media_dir).resolve()
    if not media_dir.exists():
        print(f"Error: media_dir does not exist: {media_dir}", file=sys.stderr)
        return 1

    workspace_root = Path(r"C:\Users\hihao\Documents\projects\video_editing_ai")
    lib_info = inspect_media_library(media_dir, workspace_root=workspace_root)
    questions = [q.model_dump(mode="json") for q in generate_intake_questions(media_dir, args.prompt or "")]
    dur_rec = compute_director_duration_recommendation(lib_info)

    out_path = (
        Path(args.output).resolve()
        if args.output
        else (Path(args.project_dir).resolve() / "library_inspection.json" if args.project_dir else media_dir / "library_inspection.json")
    )
    doc = {
        "library": lib_info,
        "director_duration_recommendation": dur_rec,
        "intake_questions": questions,
    }
    _write_json(out_path, doc)
    print(
        f"Success! Inspected '{lib_info['display_title']}' "
        f"({lib_info['video_count']} videos, {lib_info['photo_count']} photos, "
        f"cameras: {', '.join(lib_info['detected_cameras'])}).\n"
        f"Director Recommends: {dur_rec['director_recommended_duration_sec']}s "
        f"({dur_rec['director_recommended_mmss']}) with ~{dur_rec['recommended_photo_plates']} curated photo plates.\n"
        f"Output written to: {out_path}"
    )
    return 0


def cmd_intake_refine(args: argparse.Namespace) -> int:
    media_dir = Path(args.media_dir).resolve()
    project_dir = Path(args.project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    answers: dict[str, str] = {}
    if args.answers_json:
        try:
            answers.update(json.loads(args.answers_json))
        except Exception as exc:
            print(f"Error parsing --answers-json: {exc}", file=sys.stderr)
            return 1

    if args.narrative_style:
        answers["narrative_style"] = args.narrative_style
    if args.photo_mode:
        answers["photo_mode"] = args.photo_mode
    if args.bgm_mood:
        answers["bgm_mood"] = args.bgm_mood
    if args.dialogue_preference:
        answers["dialogue_preference"] = args.dialogue_preference
    if args.duration:
        answers["target_duration_sec"] = str(args.duration)
    if args.prompt:
        answers["must_include_topics"] = args.prompt

    refinement = refine_intake_interactively(
        media_dir=media_dir,
        user_message=args.prompt or "",
        interview_answers=answers,
    )

    brief_path = project_dir / "creative_brief.json"
    if refinement.get("draft_brief"):
        brief = CreativeBrief.model_validate(refinement["draft_brief"])
        brief.to_json_file(brief_path)

    _write_project_meta(
        project_dir,
        {
            "name": project_dir.name.replace("_project", "").replace("_", " ").title(),
            "media_dir": str(media_dir),
            "user_prompt": args.prompt or "",
            "interview_answers": refinement.get("updated_answers") or answers,
            "target_duration_sec": refinement.get("draft_brief", {}).get("target_duration_sec", 155),
        },
    )

    out_path = Path(args.output).resolve() if args.output else (project_dir / "intake_consultation.json")
    _write_json(out_path, refinement)
    dur_sec = refinement.get("draft_brief", {}).get("target_duration_sec", 155)
    print(
        f"Success! Client Intake & Director refined creative_brief.json (Target Duration: {dur_sec}s).\n"
        f"  - Creative Brief: {brief_path}\n"
        f"  - Consultation:   {out_path}"
    )
    return 0


def cmd_run_stage(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    meta_path = project_dir / "project_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    media_dir_str = args.media_dir or meta.get("media_dir") or ""
    if not media_dir_str or not Path(media_dir_str).exists():
        print("Error: Valid --media-dir is required (or must be saved in project_meta.json).", file=sys.stderr)
        return 1

    blackboard = PipelineBlackboard(
        media_dir=Path(media_dir_str).resolve(),
        project_dir=project_dir,
        user_prompt=args.prompt or meta.get("user_prompt") or "",
        interview_answers=meta.get("interview_answers") or {},
        use_antigravity=False,
    )
    blackboard.load_existing_contracts()

    director = ProductionDirectorAgent()
    requested_roles = [r.strip() for r in args.stage.split(",") if r.strip()]
    stage_results: list[dict[str, Any]] = []

    for role_id in requested_roles:
        agent_cls = AgentRegistry.get(role_id)
        if not agent_cls:
            print(f"Error: Unknown agent role_id '{role_id}'.", file=sys.stderr)
            return 1
        agent = agent_cls()
        summary = agent.run(blackboard)
        blackboard.executed_agents.append(
            {
                "role_id": agent.role_id,
                "display_name": agent.display_name,
                "stage_order": agent.stage_order,
                "summary": summary,
            }
        )
        review_dict: dict[str, Any] = {}
        if agent.role_id != director.role_id:
            review, needs_rerun = director.supervise_stage(agent, blackboard)
            if needs_rerun:
                summary = agent.run(blackboard)
            review_dict = review.model_dump(mode="json")
        stage_results.append(
            {
                "role_id": agent.role_id,
                "display_name": agent.display_name,
                "summary": summary,
                "director_stage_review": review_dict,
            }
        )

    if blackboard.timeline is not None:
        blackboard.timeline.to_yaml(project_dir / "timeline.yaml")
        if not (project_dir / "storyboard.md").exists():
            sb_md = generate_storyboard_markdown(blackboard.timeline)
            (project_dir / "storyboard.md").write_text(sb_md, encoding="utf-8")
        export_all_nle_formats(blackboard.timeline, project_dir)

    blackboard.save_snapshot()
    out_path = Path(args.output).resolve() if args.output else (project_dir / "stage_run_report.json")
    _write_json(
        out_path,
        {
            "project_dir": str(project_dir),
            "executed_stages": stage_results,
            "activity_log_count": len(blackboard.activity_log),
        },
    )
    print(f"Success! Executed stage(s) [{', '.join(requested_roles)}]. Report written to: {out_path}")
    return 0


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    media_dir = Path(args.media_dir).resolve()
    project_dir = Path(args.project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    interview_answers = {
        "narrative_style": args.narrative_style,
        "photo_mode": args.photo_mode,
        "bgm_mood": args.bgm_mood,
        "dialogue_preference": "keep_full_sentences_with_broll_cutaways",
        "must_include_topics": args.prompt or "",
    }
    if args.duration:
        interview_answers["target_duration_sec"] = str(args.duration)

    extra_audio = [Path(args.bgm).resolve()] if args.bgm and Path(args.bgm).exists() else None

    _write_project_meta(
        project_dir,
        {
            "name": project_dir.name.replace("_project", "").replace("_", " ").title(),
            "media_dir": str(media_dir),
            "user_prompt": args.prompt or "",
            "interview_answers": interview_answers,
        },
    )

    bb = run_multi_agent_pipeline(
        media_dir=media_dir,
        project_dir=project_dir,
        user_prompt=args.prompt or "",
        extra_audio_paths=extra_audio,
        interview_answers=interview_answers,
        enabled_optional_roles=["vfx_designer"],
        width=args.width,
        height=args.height,
        fps=args.fps,
        hdr=args.hdr,
        use_antigravity=False,
        render_video=args.render,
    )

    dr = bb.director_review
    tl = bb.timeline
    summary_doc = {
        "project_dir": str(project_dir),
        "media_dir": str(media_dir),
        "executed_agents": [a["role_id"] for a in bb.executed_agents],
        "director_grade": dr.quality_grade if dr else None,
        "director_score": dr.overall_quality_score if dr else None,
        "director_interventions": dr.interventions_count if dr else 0,
        "timeline_duration_sec": round(tl.total_duration, 2) if tl else 0.0,
        "v1_cuts": len(tl.visual_track) if tl else 0,
        "v2_cutaways": len(tl.broll_track) if tl else 0,
        "vfx_items": len(tl.effects_track) if tl else 0,
        "rendered_mp4": str(project_dir / "final_story.mp4") if (project_dir / "final_story.mp4").exists() else None,
    }
    out_path = Path(args.output).resolve() if args.output else (project_dir / "pipeline_summary.json")
    _write_json(out_path, summary_doc)
    print(
        f"Success! 11-Agent Pipeline Complete (Grade: {summary_doc['director_grade']} "
        f"{summary_doc['director_score']}/10, Duration: {summary_doc['timeline_duration_sec']}s).\n"
        f"Summary written to: {out_path}"
    )
    return 0


def cmd_director_audit(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    meta_path = project_dir / "project_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    media_dir_str = args.media_dir or meta.get("media_dir") or str(project_dir)

    bb = PipelineBlackboard(
        media_dir=Path(media_dir_str).resolve(),
        project_dir=project_dir,
        use_antigravity=False,
    )
    bb.load_existing_contracts()
    director = ProductionDirectorAgent()
    result = director.run(bb)
    bb.save_snapshot()

    out_path = Path(args.output).resolve() if args.output else (project_dir / "director_review.json")
    print(
        f"Success! Executive Director Audit Complete: Grade {result.get('quality_grade')} "
        f"({result.get('overall_quality_score')}/10).\n"
        f"Report written to: {out_path}"
    )
    return 0


def cmd_render_video(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    tl_path = project_dir / "timeline.yaml"
    if not tl_path.exists():
        print(f"Error: timeline.yaml not found in {project_dir}", file=sys.stderr)
        return 1

    timeline = EditTimeline.from_yaml(tl_path)
    out_mp4 = Path(args.output_mp4).resolve() if args.output_mp4 else (project_dir / "final_story.mp4")
    outputs = render_timeline(timeline, out_mp4)
    export_all_nle_formats(timeline, project_dir)

    extracted_frames: list[str] = []
    if args.extract_frames and out_mp4.exists():
        frames_dir = project_dir / "verification_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for raw_ts in args.extract_frames.split(","):
            raw_ts = raw_ts.strip()
            if not raw_ts:
                continue
            try:
                ts = float(raw_ts)
                jpg_path = frames_dir / f"verify_{int(ts):03d}s.jpg"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(ts),
                        "-i",
                        str(out_mp4),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(jpg_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if jpg_path.exists():
                    extracted_frames.append(str(jpg_path))
            except ValueError:
                pass

    print(
        f"Success! Rendered video to: {outputs['video']}\n"
        + (f"Extracted verification keyframes: {', '.join(extracted_frames)}" if extracted_frames else "")
    )
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    workspace_root = Path(r"C:\Users\hihao\Documents\projects\video_editing_ai")
    action = args.action

    if action == "list":
        projects = _list_workspace_projects(workspace_root, args.project_dir or "")
        out_path = Path(args.output).resolve() if args.output else (workspace_root / "projects_list.json")
        _write_json(out_path, {"projects": projects})
        print(f"Success! Found {len(projects)} workspace projects. Written to: {out_path}")
        return 0

    if action == "create":
        raw_name = args.name or "New Story Project"
        slug = _slugify_project_name(raw_name)
        proj = Path(args.project_dir).resolve() if args.project_dir else (workspace_root / slug).resolve()
        proj.mkdir(parents=True, exist_ok=True)
        _write_project_meta(
            proj,
            {
                "name": raw_name,
                "media_dir": str(Path(args.media_dir).resolve()) if args.media_dir else "",
                "status": "empty",
            },
        )
        info = _inspect_project_dir(proj)
        out_path = Path(args.output).resolve() if args.output else (proj / "project_status.json")
        _write_json(out_path, info)
        print(f"Success! Created project '{raw_name}' at: {proj}")
        return 0

    if action == "load":
        if not args.project_dir:
            print("Error: --project-dir is required for --action load", file=sys.stderr)
            return 1
        proj = Path(args.project_dir).resolve()
        info = _inspect_project_dir(proj)
        out_path = Path(args.output).resolve() if args.output else (proj / "project_status.json")
        _write_json(out_path, info)
        print(f"Success! Loaded project '{info['name']}' (Status: {info['status']}). Written to: {out_path}")
        return 0

    if action == "reset":
        if not args.project_dir:
            print("Error: --project-dir is required for --action reset", file=sys.stderr)
            return 1
        proj = Path(args.project_dir).resolve()
        reset_info = _reset_project_directory(proj)
        out_path = Path(args.output).resolve() if args.output else (proj / "project_status.json")
        _write_json(out_path, reset_info)
        print(
            f"Success! Reset project '{reset_info['name']}' "
            f"({len(reset_info.get('removed_items', []))} generated files cleared)."
        )
        return 0

    print(f"Error: Unknown project action '{action}'", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai_video_studio_skill",
        description="Antigravity Skill Helper CLI for the 11-Agent AI Photo & Video Editing Studio.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # 1. inspect-library
    p_insp = sub.add_parser("inspect-library", help="Scan raw media folder & get Director's recommended length")
    p_insp.add_argument("--media-dir", required=True, help="Path to raw videos and photos folder")
    p_insp.add_argument("--project-dir", default=None, help="Optional project directory for saving output")
    p_insp.add_argument("--prompt", default="", help="Optional initial user story prompt")
    p_insp.add_argument("--output", default=None, help="Output JSON file path")
    p_insp.set_defaults(func=cmd_inspect_library)

    # 2. intake-refine
    p_int = sub.add_parser("intake-refine", help="Run interactive Client Intake + Director length refinement")
    p_int.add_argument("--media-dir", required=True, help="Path to raw media folder")
    p_int.add_argument("--project-dir", required=True, help="Target project workspace directory")
    p_int.add_argument("--prompt", default="", help="User story vision and feedback")
    p_int.add_argument("--narrative-style", default=None, help="family_keepsake_4act | day_in_the_life_vlog | upbeat_montage | dialogue_story")
    p_int.add_argument("--photo-mode", default=None, help="parallax_gallery | gallery_frame | full_bleed")
    p_int.add_argument("--bgm-mood", default=None, help="auto_match | classical_chamber | acoustic_warm | cinematic_orchestral")
    p_int.add_argument("--dialogue-preference", default=None, help="keep_full_sentences_with_broll_cutaways | highlights_only | music_mostly")
    p_int.add_argument("--duration", type=int, default=None, help="Target duration in seconds (30 to 600)")
    p_int.add_argument("--answers-json", default=None, help="Optional JSON string of interview answers")
    p_int.add_argument("--output", default=None, help="Output consultation JSON path")
    p_int.set_defaults(func=cmd_intake_refine)

    # 3. run-stage
    p_stg = sub.add_parser("run-stage", help="Execute one or more specific sub-agent stages with Director review")
    p_stg.add_argument("--stage", required=True, help="Comma-separated role_ids (e.g. client_intake,music_supervisor,culler,photo_curator)")
    p_stg.add_argument("--project-dir", required=True, help="Target project workspace directory")
    p_stg.add_argument("--media-dir", default=None, help="Raw media folder path")
    p_stg.add_argument("--prompt", default="", help="Optional story prompt override")
    p_stg.add_argument("--output", default=None, help="Output stage report JSON path")
    p_stg.set_defaults(func=cmd_run_stage)

    # 4. run-pipeline
    p_pipe = sub.add_parser("run-pipeline", help="Run the full 11-agent pipeline (Stage 00 -> Stage 60) and optional 4K HDR render")
    p_pipe.add_argument("--media-dir", required=True, help="Raw media folder path")
    p_pipe.add_argument("--project-dir", required=True, help="Target project workspace directory")
    p_pipe.add_argument("--prompt", default="", help="Story vision prompt")
    p_pipe.add_argument("--duration", type=int, default=None, help="Target duration in seconds (30 to 600)")
    p_pipe.add_argument("--photo-mode", default="parallax_gallery", help="parallax_gallery | gallery_frame | full_bleed")
    p_pipe.add_argument("--narrative-style", default="family_keepsake_4act", help="Narrative style")
    p_pipe.add_argument("--bgm-mood", default="auto_match", help="BGM mood preference")
    p_pipe.add_argument("--bgm", default=None, help="Optional custom audio file path")
    p_pipe.add_argument("--width", type=int, default=3840, help="Video width (default 3840)")
    p_pipe.add_argument("--height", type=int, default=2160, help="Video height (default 2160)")
    p_pipe.add_argument("--fps", type=int, default=30, help="Frame rate (default 30)")
    p_pipe.add_argument("--hdr", dest="hdr", action="store_true", default=True, help="Enable 10-bit HLG HDR (default)")
    p_pipe.add_argument("--sdr", dest="hdr", action="store_false", help="Render in 8-bit SDR")
    p_pipe.add_argument("--render", dest="render", action="store_true", default=True, help="Render final_story.mp4 (default)")
    p_pipe.add_argument("--no-render", dest="render", action="store_false", help="Skip MP4 render (contracts + timeline only)")
    p_pipe.add_argument("--output", default=None, help="Output summary JSON path")
    p_pipe.set_defaults(func=cmd_run_pipeline)

    # 5. director-audit
    p_dir = sub.add_parser("director-audit", help="Run Stage 60 Executive Movie Production Director screening")
    p_dir.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_dir.add_argument("--media-dir", default=None, help="Optional media directory")
    p_dir.add_argument("--output", default=None, help="Output director_review.json path")
    p_dir.set_defaults(func=cmd_director_audit)

    # 6. render-video
    p_ren = sub.add_parser("render-video", help="Render timeline.yaml into 4K HDR MP4 and extract verification frames")
    p_ren.add_argument("--project-dir", required=True, help="Project workspace directory containing timeline.yaml")
    p_ren.add_argument("--output-mp4", default=None, help="Optional output MP4 path")
    p_ren.add_argument("--extract-frames", default=None, help="Optional comma-separated timestamps in seconds (e.g. '4,16,30,52')")
    p_ren.set_defaults(func=cmd_render_video)

    # 7. project
    p_proj = sub.add_parser("project", help="Create, load, list, or reset editing projects")
    p_proj.add_argument("--action", required=True, choices=["list", "create", "load", "reset"], help="Project action")
    p_proj.add_argument("--project-dir", default=None, help="Project workspace directory")
    p_proj.add_argument("--name", default=None, help="Project display name (for --action create)")
    p_proj.add_argument("--media-dir", default=None, help="Linked media directory (for --action create)")
    p_proj.add_argument("--output", default=None, help="Output JSON file path")
    p_proj.set_defaults(func=cmd_project)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
