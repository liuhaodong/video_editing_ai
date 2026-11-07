#!/usr/bin/env python3
"""Antigravity Skill Helper CLI for the 11-Agent AI Photo & Video Editing Studio.

Provides token-efficient, file-backed subcommands for every stage of the editing
workflow:
  - inspect-library       : Scan raw media folder, detect cameras, and compute Director's recommended length
  - intake-refine         : Co-refine Story Vision & CreativeBrief with Client Intake Agent + Director
  - run-stage             : Run any individual sub-agent stage(s) with Executive Director stage-gate review
  - run-pipeline          : Run the complete 11-agent pipeline (Stage 00 -> Stage 60) + 4K HLG HDR render
  - render-contact-sheets : Mechanically render numbered photo & video contact sheet JPEGs for AI Agent inspection
  - apply-vision-curation : Persist AI Agent's creative select/reject, diptych & plaque decisions into contracts
  - apply-story-plan      : Persist AI Agent's Cold-Open Teaser Hook & chapter plan overrides into paper_edit.json
  - edit-cut              : Execute ripple-aware surgical timeline revisions (swap, trim, drop, move, update-plaque, set-cold-open, audio-mix)
  - render-segment        : Render a fast 720p or 4K HLG HDR preview of any [start_sec, end_sec] timeline window
  - director-audit        : Run or inspect the Stage 60 Executive Movie Production Director screening
  - render-video          : Render timeline.yaml to 4K 10-bit HLG HDR MP4 and extract verification keyframes
  - project               : Manage editing projects (list, create, load, reset)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

# Resolve repository src/ path whether invoked from workspace .agent/skills/ or global ~/.gemini/config/skills/
def _resolve_workspace_root() -> Path:
    candidates = [
        Path(__file__).resolve().parents[4],
        Path.cwd(),
        Path.home() / "Documents" / "projects" / "video_editing_ai",
        Path(r"C:\Users\hihao\Documents\projects\video_editing_ai"),
    ]
    for root in candidates:
        if (root / "src" / "video_editing_ai").exists():
            return root
    return Path.cwd()


_WORKSPACE_ROOT = _resolve_workspace_root()
_src = _WORKSPACE_ROOT / "src"
if (_src / "video_editing_ai").exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from video_editing_ai.agents import (  # noqa: E402
    AgentRegistry,
    PipelineBlackboard,
    ProductionDirectorAgent,
    apply_story_plan_overrides,
    apply_vision_curation_overrides,
    compute_director_duration_recommendation,
    generate_intake_questions,
    inspect_media_library,
    refine_intake_interactively,
    render_curation_contact_sheets,
    run_multi_agent_pipeline,
)
from video_editing_ai.exporters.nle_export import export_all_nle_formats  # noqa: E402
from video_editing_ai.models.schema import CreativeBrief, EditTimeline  # noqa: E402
from video_editing_ai.planner.story_director import generate_storyboard_markdown  # noqa: E402
from video_editing_ai.planner.timeline_surgeon import apply_surgical_edit  # noqa: E402
from video_editing_ai.renderer.ffmpeg_composer import render_timeline, render_timeline_segment  # noqa: E402
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

    workspace_root = _WORKSPACE_ROOT
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


def cmd_render_contact_sheets(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    manifest = render_curation_contact_sheets(project_dir, max_items=args.max_items)
    out_path = (
        Path(args.output).resolve()
        if args.output
        else (project_dir / "contact_sheets" / "contact_sheets_manifest.json")
    )
    if args.output:
        _write_json(out_path, manifest)
    sheets = manifest.get("photo_contact_sheets", [])
    vid_sheet = manifest.get("video_subclips_sheet")
    print(
        f"Success! Rendered {len(sheets)} photo contact sheet(s)"
        + (f" + video subclips sheet ({vid_sheet})" if vid_sheet else "")
        + f".\nManifest written to: {out_path}"
    )
    return 0


def cmd_apply_vision_curation(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    raw_input = args.overrides_json
    try:
        if Path(raw_input).exists():
            parsed = json.loads(Path(raw_input).read_text(encoding="utf-8"))
        else:
            parsed = json.loads(raw_input)
    except Exception as exc:
        print(f"Error parsing --overrides-json: {exc}", file=sys.stderr)
        return 1

    overrides = parsed.get("overrides", parsed) if isinstance(parsed, dict) else parsed
    statement = args.statement or (parsed.get("curatorial_statement") if isinstance(parsed, dict) else None)
    result = apply_vision_curation_overrides(
        project_dir=project_dir,
        overrides=overrides,
        curatorial_statement=statement,
    )
    out_path = Path(args.output).resolve() if args.output else (project_dir / "vision_curation_applied.json")
    _write_json(out_path, result)
    print(
        f"Success! Applied {result['applied_overrides']} AI Agent vision curation override(s) "
        f"({result['selected_photos']} selected, {result['rejected_photos']} rejected, "
        f"{result['updated_timeline_cuts']} timeline cut(s) updated).\n"
        f"Report written to: {out_path}"
    )
    return 0


def cmd_apply_story_plan(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    raw_input = args.overrides_json
    try:
        if Path(raw_input).exists():
            overrides = json.loads(Path(raw_input).read_text(encoding="utf-8"))
        else:
            overrides = json.loads(raw_input)
    except Exception as exc:
        print(f"Error parsing --overrides-json: {exc}", file=sys.stderr)
        return 1

    result = apply_story_plan_overrides(project_dir=project_dir, overrides=overrides)
    if args.reassemble_timeline:
        meta_path = project_dir / "project_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        media_dir_str = args.media_dir or meta.get("media_dir") or str(project_dir)
        tl_path = project_dir / "timeline.yaml"
        if tl_path.exists():
            tl_path.unlink()
        bb = PipelineBlackboard(
            media_dir=Path(media_dir_str).resolve(),
            project_dir=project_dir,
            use_antigravity=False,
        )
        bb.load_existing_contracts()
        for role_id in ["aroll_editor", "broll_editor", "montage_editor", "vfx_designer", "sound_mixer", "production_director"]:
            cls = AgentRegistry.get(role_id)
            if cls:
                cls().run(bb)
        if bb.timeline is not None:
            bb.timeline.to_yaml(tl_path)
            export_all_nle_formats(bb.timeline, project_dir)
        bb.save_snapshot()
        result["timeline_reassembled"] = True

    out_path = Path(args.output).resolve() if args.output else (project_dir / "story_plan_applied.json")
    _write_json(out_path, result)
    print(
        f"Success! Applied AI Agent Story Plan overrides (Cold-Open Subclip: {result.get('cold_open_subclip_id')}).\n"
        f"Report written to: {out_path}"
    )
    return 0


def cmd_edit_cut(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    ops: list[dict[str, Any]] = []
    if args.ops_json:
        try:
            raw = args.ops_json
            parsed = json.loads(Path(raw).read_text(encoding="utf-8")) if Path(raw).exists() else json.loads(raw)
            ops = parsed if isinstance(parsed, list) else parsed.get("operations", [parsed])
        except Exception as exc:
            print(f"Error parsing --ops-json: {exc}", file=sys.stderr)
            return 1
    else:
        if not args.action:
            print("Error: Specify either --action or --ops-json", file=sys.stderr)
            return 1
        op: dict[str, Any] = {"action": args.action}
        if args.clip_id:
            op["clip_id"] = args.clip_id
        if args.new_source:
            op["new_source"] = args.new_source
        if args.new_duration is not None:
            op["new_duration"] = args.new_duration
        if args.delta_sec is not None:
            op["delta_sec"] = args.delta_sec
        if args.new_source_in is not None:
            op["new_source_in"] = args.new_source_in
        if args.caption_title is not None:
            op["caption_title"] = args.caption_title
        if args.caption_subtitle is not None:
            op["caption_subtitle"] = args.caption_subtitle
        if args.camera_badge is not None:
            op["camera_badge"] = args.camera_badge
        if args.mode:
            op["mode"] = args.mode
        if getattr(args, "overlay_fx", None):
            op["overlay_fx"] = args.overlay_fx
        if getattr(args, "secondary_source", None) is not None:
            op["secondary_source"] = args.secondary_source
        if getattr(args, "tertiary_source", None) is not None:
            op["tertiary_source"] = args.tertiary_source
        if args.preset:
            op["preset"] = args.preset
        if args.before_clip_id:
            op["before_clip_id"] = args.before_clip_id
        if args.after_clip_id:
            op["after_clip_id"] = args.after_clip_id
        if args.subclip_id:
            op["subclip_id"] = args.subclip_id
        if args.bgm_volume is not None:
            op["bgm_volume"] = args.bgm_volume
        if args.duck_volume is not None:
            op["duck_volume"] = args.duck_volume
        ops.append(op)

    result = apply_surgical_edit(project_dir=project_dir, operations=ops)
    out_path = Path(args.output).resolve() if args.output else (project_dir / "surgical_edit_report.json")
    _write_json(out_path, result)
    print(
        f"Success! Executed {result['operations_count']} surgical timeline operation(s). "
        f"Duration: {result['old_duration_sec']}s -> {result['new_duration_sec']}s "
        f"(V1 cuts: {result['v1_cuts']}).\nReport written to: {out_path}"
    )
    return 0


def cmd_render_segment(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    tl_path = project_dir / "timeline.yaml"
    if not tl_path.exists():
        print(f"Error: timeline.yaml not found in {project_dir}", file=sys.stderr)
        return 1

    timeline = EditTimeline.from_yaml(tl_path)
    default_name = f"segment_{int(args.start):03d}s_{int(args.end):03d}s.mp4"
    out_mp4 = Path(args.output_mp4).resolve() if args.output_mp4 else (project_dir / "segments" / default_name)
    res = render_timeline_segment(
        timeline=timeline,
        output_mp4_path=out_mp4,
        start_sec=args.start,
        end_sec=args.end,
        fast_preview=args.fast_preview,
    )

    extracted_frames: list[str] = []
    if args.extract_frames and out_mp4.exists():
        frames_dir = out_mp4.parent / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for raw_ts in args.extract_frames.split(","):
            raw_ts = raw_ts.strip()
            if not raw_ts:
                continue
            try:
                ts = float(raw_ts)
                rel_ts = max(0.0, ts - float(res["start_sec"])) if ts >= float(res["start_sec"]) else ts
                jpg_path = frames_dir / f"seg_frame_{int(ts):03d}s.jpg"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(rel_ts),
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

    res["extracted_frames"] = extracted_frames
    print(
        f"Success! Rendered segment [{res['start_sec']}s - {res['end_sec']}s] "
        f"({res['duration_sec']}s, fast_preview={res['fast_preview']}) to: {res['video']}"
        + (f"\nExtracted segment frames: {', '.join(extracted_frames)}" if extracted_frames else "")
    )
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    workspace_root = _WORKSPACE_ROOT
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


def cmd_scout_bgm(args: argparse.Namespace) -> int:
    """Mechanically search & download royalty-free BGM candidates and measure acoustic diagnostics for the AI BGM Agent."""
    from video_editing_ai.agents.music_supervisor_agent import scout_bgm_candidates_for_project

    proj = Path(args.project_dir).resolve()
    proj.mkdir(parents=True, exist_ok=True)
    keywords = [k.strip() for k in (args.keywords or "").split(",") if k.strip()]
    payload = scout_bgm_candidates_for_project(
        project_dir=proj,
        query_keywords=keywords or ["paris", "accordion", "waltz", "cafe", "piano"],
        mood_profile=args.mood_profile or "parisian_classical_museum",
        target_duration_sec=float(args.target_duration or 240.0),
        download_limit=int(args.download_limit or 5),
    )
    out_path = Path(args.output).resolve() if args.output else (proj / "bgm_candidates.json")
    _write_json(out_path, payload)
    dl_count = sum(1 for c in payload.get("candidates", []) if c.get("downloaded"))
    print(f"Success! Scouted {len(payload.get('candidates', []))} royalty-free BGM candidates ({dl_count} downloaded). Written to: {out_path}")
    return 0


def cmd_apply_bgm_plan(args: argparse.Namespace) -> int:
    """Mechanically master the AI BGM Agent's selected track(s) to full timeline duration at -14 LUFS and update timeline.yaml."""
    from video_editing_ai.agents.music_supervisor_agent import master_bgm_soundtrack, probe_audio_diagnostics
    from video_editing_ai.models.schema import AudioTrackItem, EditTimeline

    proj = Path(args.project_dir).resolve()
    tl_path = proj / "timeline.yaml"
    target_dur = float(args.target_duration or 0.0)
    tl: Optional[EditTimeline] = None
    if tl_path.exists():
        tl = EditTimeline.from_yaml_file(tl_path)
        if target_dur <= 0.0:
            target_dur = max(60.0, float(tl.total_duration))
    if target_dur <= 0.0:
        target_dur = 240.0

    bgm_dir = proj / "bgm_library"
    bgm_dir.mkdir(parents=True, exist_ok=True)
    raw_tracks = [t.strip() for t in (args.tracks or "").split(",") if t.strip()]
    track_paths: list[Path] = []
    for t_str in raw_tracks:
        p = Path(t_str)
        if not p.exists():
            p = bgm_dir / (t_str if t_str.endswith(".mp3") else f"{t_str}.mp3")
        if p.exists():
            track_paths.append(p.resolve())

    if not track_paths:
        print("Error: No valid track files found for --tracks", file=sys.stderr)
        return 1

    mastered_mp3 = bgm_dir / "mastered_soundtrack.mp3"
    out_mp3 = master_bgm_soundtrack(
        track_paths=track_paths,
        target_duration_sec=target_dur,
        output_mp3=mastered_mp3,
        target_lufs=float(args.target_lufs or -14.0),
        crossfade_sec=float(args.crossfade_sec or 3.5),
    )
    diag = probe_audio_diagnostics(out_mp3)

    if tl is not None:
        tl.audio_tracks = [
            AudioTrackItem(
                id="bgm_master_01",
                source=str(out_mp3.resolve()),
                timeline_start=0.0,
                source_in=0.0,
                duration=round(tl.total_duration, 2),
                volume=float(args.bgm_volume or 0.85),
                duck_under_dialogue=True,
                duck_volume=float(args.duck_volume or 0.35),
                fade_in=1.5,
                fade_out=3.0,
            )
        ]
        tl.to_yaml_file(tl_path)

    report_data = {
        "mastered_soundtrack_path": str(out_mp3.resolve()),
        "source_tracks": [str(p) for p in track_paths],
        "target_duration_sec": round(target_dur, 2),
        "mastered_duration_sec": diag["duration_sec"],
        "mean_volume_db": diag["mean_volume_db"],
        "max_volume_db": diag["max_volume_db"],
        "selection_rationale": args.rationale or "AI BGM Agent curated multi-track royalty-free soundtrack suite.",
    }
    out_json = Path(args.output).resolve() if args.output else (proj / "bgm_mastering_report.json")
    _write_json(out_json, report_data)
    print(
        f"Success! Mastered full-length BGM ({diag['duration_sec']:.1f}s, mean {diag['mean_volume_db']:.1f} dB) -> {out_mp3}"
    )
    return 0


def cmd_preview_photo_effects(args: argparse.Namespace) -> int:
    """Mechanically render a 5x2 high-resolution visual showcase sheet of all 10 Photo Modes & 4 Atmospheric Finishers."""
    from video_editing_ai.renderer.ken_burns import render_photo_effects_showcase_sheet

    proj = Path(args.project_dir).resolve()
    proj.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(args.output).resolve() if args.output else (proj / "contact_sheets" / "photo_effects_showcase.jpg")

    photo_path = args.photo
    sec_path = args.secondary_photo
    tert_path = args.tertiary_photo

    # Auto-pick top curated photos from project if --photo is omitted
    if not photo_path:
        pc_path = proj / "photo_curation.json"
        if pc_path.exists():
            doc = json.loads(pc_path.read_text(encoding="utf-8"))
            sel = [it for it in doc.get("items", []) if it.get("is_selected")]
            if sel:
                photo_path = sel[0]["source_path"]
                if not sec_path and len(sel) >= 2:
                    sec_path = sel[1]["source_path"]
                if not tert_path and len(sel) >= 3:
                    tert_path = sel[2]["source_path"]

    if not photo_path or not Path(photo_path).exists():
        print(f"Error: Valid --photo path required (got {photo_path})", file=sys.stderr)
        return 1

    res_jpg = render_photo_effects_showcase_sheet(
        photo_path=photo_path,
        output_jpg=out_jpg,
        secondary_path=sec_path,
        tertiary_path=tert_path,
        caption_title=args.caption_title or "Cathédrale Notre-Dame & Parisian Masterpieces",
        caption_subtitle=args.caption_subtitle or "860 Years of French Gothic Architecture · Île de la Cité, Paris",
        camera_badge=args.camera_badge or "APPLE IPHONE 16 PRO · PARIS 4K HDR",
    )
    print(f"Success! Rendered 10-Mode Premium Photo Effects Showcase Sheet to: {res_jpg}")
    return 0


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

    # 8. render-contact-sheets
    p_cs = sub.add_parser(
        "render-contact-sheets",
        help="Mechanically render numbered photo contact sheet JPEGs & video subclips keyframe sheet for AI Agent inspection",
    )
    p_cs.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_cs.add_argument("--max-items", type=int, default=60, help="Max photos to render across contact sheets")
    p_cs.add_argument("--output", default=None, help="Optional output manifest JSON path")
    p_cs.set_defaults(func=cmd_render_contact_sheets)

    # 9. apply-vision-curation
    p_vc = sub.add_parser(
        "apply-vision-curation",
        help="Persist AI Agent's creative select/reject, diptych & plaque decisions into photo_curation.json & timeline.yaml",
    )
    p_vc.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_vc.add_argument(
        "--overrides-json",
        required=True,
        help="JSON string or file path containing AI Agent's list of photo curation overrides",
    )
    p_vc.add_argument("--statement", default=None, help="Optional AI Agent curatorial statement")
    p_vc.add_argument("--output", default=None, help="Optional output report JSON path")
    p_vc.set_defaults(func=cmd_apply_vision_curation)

    # 10. apply-story-plan
    p_sp = sub.add_parser(
        "apply-story-plan",
        help="Persist AI Agent's Cold-Open Teaser Hook & chapter plan overrides into paper_edit.json",
    )
    p_sp.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_sp.add_argument("--media-dir", default=None, help="Optional media directory when --reassemble-timeline is used")
    p_sp.add_argument(
        "--overrides-json",
        required=True,
        help="JSON string or file path with cold_open_subclip_id, cold_open_hook_caption, chapters, etc.",
    )
    p_sp.add_argument(
        "--reassemble-timeline",
        action="store_true",
        default=False,
        help="Re-run Picture Editors, VFX, Sound Mixer & Director to rebuild timeline.yaml with new Story Plan",
    )
    p_sp.add_argument("--output", default=None, help="Optional output report JSON path")
    p_sp.set_defaults(func=cmd_apply_story_plan)

    # 11. edit-cut
    p_ec = sub.add_parser(
        "edit-cut",
        help="Execute ripple-aware surgical timeline revisions (swap, trim, drop, move, update-plaque, set-cold-open, audio-mix)",
    )
    p_ec.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_ec.add_argument(
        "--action",
        choices=["update-plaque", "set-photo-effect", "trim", "drop", "swap", "move", "set-cold-open", "audio-mix"],
        default=None,
        help="Surgical operation type (or pass a batch via --ops-json)",
    )
    p_ec.add_argument("--clip-id", default=None, help="Target clip_id (e.g. clip_001, clip_000_hook, broll_01)")
    p_ec.add_argument("--new-source", default=None, help="Replacement media path or filename (for --action swap)")
    p_ec.add_argument("--new-duration", type=float, default=None, help="New clip duration in seconds (for --action trim)")
    p_ec.add_argument("--delta-sec", type=float, default=None, help="Relative duration change in seconds (for --action trim)")
    p_ec.add_argument("--new-source-in", type=float, default=None, help="New source in-point in seconds")
    p_ec.add_argument("--caption-title", default=None, help="Updated plaque title or hook caption")
    p_ec.add_argument("--caption-subtitle", default=None, help="Updated plaque subtitle")
    p_ec.add_argument("--camera-badge", default=None, help="Updated camera badge")
    p_ec.add_argument(
        "--mode",
        default=None,
        help="Updated photo mode (cinematic_depth_3d | triptych_cinematic | polaroid_scrapbook | curator_loupe | editorial_magazine | cinema_anamorphic | parallax_gallery | diptych_frame | gallery_frame | full_bleed)",
    )
    p_ec.add_argument(
        "--overlay-fx",
        default=None,
        help="Composable atmospheric finisher (none | golden_light_leak | champagne_bokeh | museum_spotlight | nostalgic_film_grain)",
    )
    p_ec.add_argument("--secondary-source", default=None, help="Optional 2nd photo path for diptych_frame / polaroid_scrapbook / triptych_cinematic")
    p_ec.add_argument("--tertiary-source", default=None, help="Optional 3rd photo path for triptych_cinematic")
    p_ec.add_argument("--preset", default=None, help="Updated Ken Burns preset")
    p_ec.add_argument("--before-clip-id", default=None, help="Move clip before this clip_id (for --action move)")
    p_ec.add_argument("--after-clip-id", default=None, help="Move clip after this clip_id (for --action move)")
    p_ec.add_argument("--subclip-id", default=None, help="Subclip ID for --action set-cold-open")
    p_ec.add_argument("--bgm-volume", type=float, default=None, help="BGM base volume (0.05-1.0) for --action audio-mix")
    p_ec.add_argument("--duck-volume", type=float, default=None, help="BGM ducking volume (0.02-0.5) for --action audio-mix")
    p_ec.add_argument("--ops-json", default=None, help="JSON string or file path containing a list of surgical operations")
    p_ec.add_argument("--output", default=None, help="Optional output report JSON path")
    p_ec.set_defaults(func=cmd_edit_cut)

    # 12. render-segment
    p_rs = sub.add_parser(
        "render-segment",
        help="Render a fast 720p or 4K HLG HDR preview of any [start_sec, end_sec] timeline window in ~2 seconds",
    )
    p_rs.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_rs.add_argument("--start", type=float, required=True, help="Window start time in seconds (e.g. 0.0)")
    p_rs.add_argument("--end", type=float, required=True, help="Window end time in seconds (e.g. 12.0)")
    p_rs.add_argument(
        "--fast-preview",
        dest="fast_preview",
        action="store_true",
        default=True,
        help="Render fast 1280x720 8-bit SDR preview (default)",
    )
    p_rs.add_argument(
        "--full-quality",
        dest="fast_preview",
        action="store_false",
        help="Render segment at full timeline resolution & 10-bit HLG HDR",
    )
    p_rs.add_argument("--output-mp4", default=None, help="Optional output segment MP4 path")
    p_rs.add_argument("--extract-frames", default=None, help="Optional comma-separated timestamps to extract as JPEGs")
    p_rs.set_defaults(func=cmd_render_segment)

    # 13. scout-bgm
    p_sb = sub.add_parser(
        "scout-bgm",
        help="Mechanically search & download royalty-free BGM tracks (Pixabay/Incompetech/CC0) and measure loudness & duration",
    )
    p_sb.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_sb.add_argument("--keywords", default="paris,accordion,waltz,cafe,piano", help="Comma-separated search keywords")
    p_sb.add_argument("--mood-profile", default="parisian_classical_museum", help="Target mood profile")
    p_sb.add_argument("--target-duration", type=float, default=240.0, help="Target video duration in seconds")
    p_sb.add_argument("--download-limit", type=int, default=5, help="Max candidate tracks to download")
    p_sb.add_argument("--output", default=None, help="Optional output bgm_candidates.json path")
    p_sb.set_defaults(func=cmd_scout_bgm)

    # 14. apply-bgm-plan
    p_ab = sub.add_parser(
        "apply-bgm-plan",
        help="Mechanically master the AI BGM Agent's selected track(s) to cover 100% of timeline duration at -14 LUFS",
    )
    p_ab.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_ab.add_argument("--tracks", required=True, help="Comma-separated track filenames or paths to chain/crossfade")
    p_ab.add_argument("--target-duration", type=float, default=0.0, help="Target duration in seconds (defaults to timeline.total_duration)")
    p_ab.add_argument("--target-lufs", type=float, default=-14.0, help="Target EBU R128 loudness in LUFS (default -14.0)")
    p_ab.add_argument("--crossfade-sec", type=float, default=3.5, help="Crossfade duration between tracks in seconds")
    p_ab.add_argument("--bgm-volume", type=float, default=0.85, help="Timeline BGM volume multiplier (default 0.85)")
    p_ab.add_argument("--duck-volume", type=float, default=0.35, help="Timeline BGM ducking volume (default 0.35)")
    p_ab.add_argument("--rationale", default="", help="AI BGM Agent's curatorial rationale")
    p_ab.add_argument("--output", default=None, help="Optional output JSON report path")
    p_ab.set_defaults(func=cmd_apply_bgm_plan)

    # 15. preview-photo-effects
    p_pe = sub.add_parser(
        "preview-photo-effects",
        help="Mechanically render a 5x2 visual comparison showcase sheet of all 10 Photo Modes & 4 Atmospheric Finishers",
    )
    p_pe.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_pe.add_argument("--photo", default=None, help="Primary photo path (auto-picks top curated photo if omitted)")
    p_pe.add_argument("--secondary-photo", default=None, help="Optional 2nd photo path for diptych/polaroid/triptych")
    p_pe.add_argument("--tertiary-photo", default=None, help="Optional 3rd photo path for triptych")
    p_pe.add_argument("--caption-title", default=None, help="Sample caption title")
    p_pe.add_argument("--caption-subtitle", default=None, help="Sample caption subtitle")
    p_pe.add_argument("--camera-badge", default=None, help="Sample camera badge")
    p_pe.add_argument("--output", default=None, help="Optional output JPEG path")
    p_pe.set_defaults(func=cmd_preview_photo_effects)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
