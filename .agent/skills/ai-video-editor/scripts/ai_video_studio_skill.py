#!/usr/bin/env python3
"""Antigravity Skill Helper CLI for the 22-Agent AI Photo & Video Editing Studio.

Provides token-efficient, file-backed subcommands for every stage of the editing
workflow:
  - inspect-library       : Scan raw media folder, detect cameras, and compute Director's recommended length
  - intake-refine         : Co-refine Story Vision & CreativeBrief with Client Intake Agent + Director
  - run-stage             : Run any individual sub-agent stage(s) with Executive Director stage-gate review
  - run-pipeline          : Run the complete multi-agent pipeline (Stage 00 -> Stage 65) + 4K HLG HDR render
  - render-contact-sheets : Mechanically render numbered photo & video contact sheet JPEGs for AI Agent inspection
  - apply-vision-curation : Persist AI Agent's creative select/reject, diptych & plaque decisions into contracts
  - apply-story-plan      : Persist AI Agent's Cold-Open Teaser Hook & chapter plan overrides into paper_edit.json
  - edit-cut              : Execute ripple-aware surgical timeline revisions (swap, trim, drop, move, update-plaque, set-cold-open, audio-mix, set-color-grade)
  - preview-color-grades  : Render 2560x1440 4x2 lookbook sheet across all 8 curated film stocks
  - recommend-color-grade : Analyze project photometry & lookbook sheets via AI VLM/ranker and recommend film stock + chapter looks
  - render-color-scopes   : Render 2560x1440 Split-Wipe & RGB Parade waveform scopes sheet
  - apply-color-grade     : Apply Stage 42 DI film stock look or per-clip balance overrides to timeline.yaml
  - render-segment        : Render a fast 720p or 4K HLG HDR preview of any [start_sec, end_sec] timeline window
  - director-audit        : Run or inspect the Stage 60 Executive Movie Production Director screening
  - render-video          : Render timeline.yaml to 4K 10-bit HLG HDR MP4 and extract verification keyframes
  - project               : Manage editing projects (list, create, load, reset)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional

# Resolve repository src/ path whether invoked from workspace .agent/skills/ or global ~/.gemini/config/skills/
def _resolve_workspace_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get("AI_VIDEO_EDITOR_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())

    marker = Path(__file__).resolve().parents[1] / ".workspace_root"
    if marker.is_file():
        marker_text = marker.read_text(encoding="utf-8").strip()
        if marker_text:
            candidates.append(Path(marker_text).expanduser())

    candidates.extend(
        [
            Path(__file__).resolve().parents[4],
            Path.cwd(),
            Path.home() / "Documents" / "projects" / "video-edit-ai",
            Path.home() / "Documents" / "projects" / "video_editing_ai",
        ]
    )
    for root in candidates:
        if (root / "src" / "video_editing_ai").exists() or (root / ".venv").exists():
            return root.resolve()
    return Path.cwd().resolve()


_WORKSPACE_ROOT = _resolve_workspace_root()
_venv_dir = _WORKSPACE_ROOT / ".venv"
_venv_py = _venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if _venv_py.exists() and Path(sys.prefix).resolve() != _venv_dir.resolve():
    try:
        import video_editing_ai  # noqa: F401
        import torch  # noqa: F401
        import faster_whisper  # noqa: F401
        import pillow_heif  # noqa: F401
    except ImportError:
        os.execv(str(_venv_py), [str(_venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])
_src = _WORKSPACE_ROOT / "src"
if (_src / "video_editing_ai").exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from video_editing_ai.agents import (  # noqa: E402
    AgentRegistry,
    PipelineBlackboard,
    ProductionDirectorAgent,
    apply_story_plan_overrides,
    apply_timeline_intelligence_overrides,
    apply_video_curation_overrides,
    apply_vision_curation_overrides,
    build_ask_question_tool_payload,
    compute_director_duration_recommendation,
    generate_intake_questions,
    inspect_media_library,
    recommend_color_grading,
    refine_intake_interactively,
    render_curation_contact_sheets,
    render_director_screening_sheet,
    render_video_culling_contact_sheets,
    run_multi_agent_pipeline,
)
from video_editing_ai.agents.intake_agent import _recommend_color_look  # noqa: E402
from video_editing_ai.cloud import (  # noqa: E402
    GooglePhotosAuthManager,
    YouTubeAuthManager,
    apply_cloud_pick_overrides,
    build_google_photos_web_search_url,
    build_zero_setup_browser_plan,
    execute_youtube_publish,
    format_picker_date_query,
    prepare_youtube_publish,
    record_browser_publish_receipt,
    run_google_photos_sync,
)
from video_editing_ai.models.schema import YouTubePublishDraft  # noqa: E402
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


def _resolve_cloud_or_local_media_dir(args: argparse.Namespace) -> Path:
    proj_raw = getattr(args, "project_dir", None)
    proj_dir = Path(proj_raw).resolve() if proj_raw else None
    g_takeout = getattr(args, "google_takeout", None) or getattr(args, "takeout", None)
    from_dl = getattr(args, "from_downloads", None)
    g_photos = bool(getattr(args, "google_photos", False))
    date_from = getattr(args, "date_from", None)
    date_to = getattr(args, "date_to", None)
    tz_name = getattr(args, "timezone", None) or "UTC"
    dl_all = bool(getattr(args, "download_all", False))
    dur_hint = float(getattr(args, "duration", None) or getattr(args, "target_duration", None) or 155.0)
    auth_mode = getattr(args, "auth_mode", None) or "auto"

    if g_takeout or from_dl or g_photos:
        if proj_dir is None:
            proj_dir = (_WORKSPACE_ROOT / "cloud_story_project").resolve()

        def _on_browser(url: str, dl_dir: Path) -> None:
            print(f"Open Google Photos Browser URL (zero GCP setup):\n  {url}")
            print(f"Watching downloads folder: {dl_dir}")

        def _on_pick(uri: str, hint: str) -> None:
            print(f"Open Google Photos Picker URL:\n  {uri}")
            if hint:
                print(f"Suggested date search query (copied to clipboard): {hint}")

        manifest = run_google_photos_sync(
            project_dir=proj_dir,
            date_from=date_from,
            date_to=date_to,
            tz_name=tz_name,
            target_duration_sec=dur_hint,
            max_photos=getattr(args, "max_photos", None),
            max_videos=getattr(args, "max_videos", None),
            download_all=dl_all,
            auth_mode=auth_mode,
            from_downloads=from_dl,
            takeout_source=g_takeout,
            downloads_dir=getattr(args, "downloads_dir", None),
            watch_timeout_sec=float(getattr(args, "watch_timeout", None) or 600.0),
            session_id=getattr(args, "session_id", None),
            open_browser=not bool(getattr(args, "no_browser", False)),
            browser_open_callback=_on_browser,
            picker_ready_callback=_on_pick,
            progress_callback=lambda m: print(f"  {m}"),
        )
        _write_project_meta(proj_dir, {"media_dir": manifest.staged_media_dir})
        return Path(manifest.staged_media_dir).resolve()

    if getattr(args, "media_dir", None):
        return Path(args.media_dir).resolve()

    if proj_dir is not None:
        meta_p = proj_dir / "project_meta.json"
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                saved_m = str(meta.get("media_dir") or "").strip()
                if saved_m and Path(saved_m).exists():
                    return Path(saved_m).resolve()
            except Exception:
                pass
        cloud_m = proj_dir / "cloud_media"
        if cloud_m.is_dir():
            return cloud_m.resolve()

    raise ValueError("Provide --media-dir, --google-photos, --from-downloads, --google-takeout, or a synced --project-dir.")


def cmd_google_photos_auth(args: argparse.Namespace) -> int:
    auth = GooglePhotosAuthManager(
        client_secrets_path=getattr(args, "client_secrets", None),
        token_path=getattr(args, "token_path", None),
    )
    if getattr(args, "revoke", False):
        removed = auth.revoke_local_token()
        print(f"Removed Google Photos token: {removed} ({auth.token_path})")
        return 0
    if getattr(args, "status", False):
        print(json.dumps(auth.get_auth_status(), indent=2))
        return 0
    if not auth.client_secrets_path.exists():
        print(auth.format_setup_instructions(), file=sys.stderr)
        return 1
    auth.authenticate_interactive(
        open_browser=not bool(getattr(args, "no_browser", False)),
        auth_url_callback=lambda url: print(f"Open Google Photos consent URL in your browser:\n{url}"),
    )
    print(f"Success! Google Photos OAuth connected -> {auth.token_path}")
    return 0


def cmd_youtube_auth(args: argparse.Namespace) -> int:
    """Inspect or configure YouTube Data API v3 OAuth 2.0 PKCE credentials."""
    auth = YouTubeAuthManager(
        client_secrets_path=getattr(args, "client_secrets", None),
        token_path=getattr(args, "token_path", None),
        include_captions_scope=bool(getattr(args, "include_captions_scope", False)),
    )
    if getattr(args, "revoke", False):
        removed = auth.revoke_local_token()
        print(f"Removed YouTube OAuth token: {removed} ({auth.token_path})")
        return 0
    if getattr(args, "status", False):
        st = auth.get_auth_status()
        st["studio_browser_ready"] = True
        print(json.dumps(st, indent=2))
        return 0
    if not auth.client_secrets_path.exists():
        print(auth.format_setup_instructions(), file=sys.stderr)
        return 1
    auth.authenticate_interactive(
        open_browser=not bool(getattr(args, "no_browser", False)),
        auth_url_callback=lambda url: print(f"Open YouTube consent URL in your browser:\n{url}"),
    )
    print(f"Success! YouTube OAuth connected -> {auth.token_path}")
    return 0


def cmd_publish_youtube(args: argparse.Namespace) -> int:
    """Prepare or execute confirmation-gated YouTube publishing."""
    proj = Path(args.project_dir).expanduser().resolve()
    auth_mgr = YouTubeAuthManager(
        client_secrets_path=getattr(args, "client_secrets", None),
        token_path=getattr(args, "token_path", None),
        include_captions_scope=True,
    )

    draft_path = proj / "youtube_publish_draft.json"
    record_url = getattr(args, "record_browser_receipt", None)
    confirm_tok = getattr(args, "confirm_token", None)
    confirmed_flag = bool(getattr(args, "confirmed", False))

    if record_url:
        if not draft_path.is_file():
            print(f"Error: Draft file missing at {draft_path}. Run --prepare first.", file=sys.stderr)
            return 1
        draft = YouTubePublishDraft.from_json_file(draft_path)
        try:
            receipt = record_browser_publish_receipt(
                project_dir=proj,
                draft=draft,
                confirmed=confirmed_flag,
                confirmation_token=confirm_tok or "",
                video_url_or_id=record_url,
                auth_mode_used="antigravity_browser",
            )
        except PermissionError as exc:
            print(f"Confirmation Error: {exc}", file=sys.stderr)
            return 2
        if args.output:
            _write_json(Path(args.output).resolve(), receipt.model_dump(mode="json"))
        print(json.dumps(receipt.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0

    has_meta_overrides = any(
        getattr(args, f, None) is not None
        for f in ("video", "title", "description", "tags", "category_id", "thumbnail")
    ) or bool(getattr(args, "made_for_kids", False)) or (
        getattr(args, "privacy", "unlisted") != "unlisted"
    ) or (getattr(args, "auth_mode", "auto") != "auto")

    if confirmed_flag and confirm_tok and not has_meta_overrides and draft_path.is_file():
        draft = YouTubePublishDraft.from_json_file(draft_path)
    else:
        draft = prepare_youtube_publish(
            project_dir=proj,
            video_path=getattr(args, "video", None),
            title=getattr(args, "title", None),
            description=getattr(args, "description", None),
            tags=getattr(args, "tags", None),
            privacy_status=getattr(args, "privacy", "unlisted"),
            self_declared_made_for_kids=bool(getattr(args, "made_for_kids", False)),
            category_id=getattr(args, "category_id", None),
            thumbnail_source=getattr(args, "thumbnail", None),
            auth_mode=getattr(args, "auth_mode", "auto"),
            auth_manager=auth_mgr,
        )

    if getattr(args, "prepare", False) or not confirmed_flag:
        payload = draft.model_dump(mode="json")
        if args.output:
            _write_json(Path(args.output).resolve(), payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if not getattr(args, "prepare", False) and not confirmed_flag:
            print(
                "\nAwaiting explicit confirmation. Re-run with --confirmed --confirm-token "
                f"{draft.confirmation_token} after user approval.",
                file=sys.stderr,
            )
            return 2
        return 0

    try:
        receipt = execute_youtube_publish(
            draft=draft,
            confirmed=True,
            confirmation_token=confirm_tok or "",
            auth_manager=auth_mgr,
            open_browser=not bool(getattr(args, "no_browser", False)),
        )
    except PermissionError as exc:
        print(f"Confirmation Error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        _write_json(Path(args.output).resolve(), receipt.model_dump(mode="json"))
    print(json.dumps(receipt.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def cmd_sync_google_photos(args: argparse.Namespace) -> int:
    if getattr(args, "search_url_only", False):
        payload = build_zero_setup_browser_plan(
            date_from=args.date_from,
            date_to=args.date_to,
            album_url=getattr(args, "album_url", None),
            favorites_only=bool(getattr(args, "favorites_only", False)),
        )
        print(json.dumps(payload, indent=2))
        return 0
    if not getattr(args, "project_dir", None):
        print("Error: --project-dir is required unless --search-url-only is passed.", file=sys.stderr)
        return 1

    proj_dir = Path(args.project_dir).resolve()
    proj_dir.mkdir(parents=True, exist_ok=True)

    def _on_browser(url: str, dl_dir: Path) -> None:
        print(f"Open Google Photos Browser URL (zero GCP setup):\n  {url}")
        print(f"Watching downloads folder: {dl_dir}")

    def _on_pick(uri: str, hint: str) -> None:
        print(f"Open Google Photos Picker URL:\n  {uri}")
        if hint:
            print(f"Suggested date search query (copied to clipboard): {hint}")

    manifest = run_google_photos_sync(
        project_dir=proj_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        tz_name=args.timezone or "UTC",
        target_duration_sec=float(args.target_duration or 155.0),
        max_photos=args.max_photos,
        max_videos=args.max_videos,
        download_all=bool(args.download_all),
        headroom_multiplier=float(getattr(args, "headroom_multiplier", None) or 1.6),
        plan_only=bool(getattr(args, "plan_only", False)),
        favorites_only=bool(getattr(args, "favorites_only", False)),
        album_url=getattr(args, "album_url", None),
        auth_mode=getattr(args, "auth_mode", None) or "auto",
        from_downloads=getattr(args, "from_downloads", None),
        takeout_source=getattr(args, "takeout", None),
        downloads_dir=getattr(args, "downloads_dir", None),
        watch_timeout_sec=float(getattr(args, "watch_timeout", None) or 600.0),
        session_id=args.session_id,
        open_browser=not bool(args.no_browser),
        client_secrets_path=args.client_secrets,
        token_path=args.token_path,
        favorite_filenames=getattr(args, "favorites", None),
        browser_open_callback=_on_browser,
        picker_ready_callback=_on_pick,
        progress_callback=lambda m: print(f"  {m}"),
    )

    _write_project_meta(
        proj_dir,
        {
            "name": proj_dir.name.replace("_project", "").replace("_", " ").title(),
            "media_dir": manifest.staged_media_dir,
            "cloud_source_mode": manifest.source_mode,
            "date_from": manifest.date_from,
            "date_to": manifest.date_to,
        },
    )
    if args.output:
        _write_json(Path(args.output).resolve(), manifest.model_dump(mode="json"))
    sheets_list = manifest.scan_contact_sheet_paths or ([manifest.scan_contact_sheet_path] if manifest.scan_contact_sheet_path else [])
    stage_note = "Planned (Pass-1.5 review ready; run apply-cloud-picks to finalize)" if manifest.download_stage == "planned_only" else "Synced"
    print(
        f"Success! {stage_note} {manifest.selected_photos_count} photos and {manifest.selected_videos_count} videos "
        f"(from {manifest.in_date_range_count} in-range candidates across {manifest.event_clusters_count} event clusters) -> {manifest.staged_media_dir}\n"
        f"  - Manifest:       {proj_dir / 'google_photos_manifest.json'}\n"
        f"  - Contact Sheets: {', '.join(sheets_list)}"
    )
    return 0


def cmd_apply_cloud_picks(args: argparse.Namespace) -> int:
    proj_dir = Path(args.project_dir).resolve()
    raw_ov = getattr(args, "overrides_json", None)
    overrides: dict[str, Any] | list[dict[str, Any]] = {}
    if raw_ov:
        ov_p = Path(raw_ov)
        overrides = json.loads(ov_p.read_text(encoding="utf-8")) if ov_p.is_file() else json.loads(raw_ov)

    manifest = apply_cloud_pick_overrides(
        project_dir=proj_dir,
        overrides=overrides,
        execute_pass2=not bool(getattr(args, "no_download", False)),
        progress_callback=lambda m: print(f"  {m}"),
    )
    if args.output:
        _write_json(Path(args.output).resolve(), manifest.model_dump(mode="json"))
    print(
        f"Success! Finalized cloud picks: {manifest.selected_photos_count} photos and {manifest.selected_videos_count} videos "
        f"staged in {manifest.staged_media_dir} (stage={manifest.download_stage})."
    )
    return 0


def cmd_inspect_library(args: argparse.Namespace) -> int:
    try:
        media_dir = _resolve_cloud_or_local_media_dir(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not media_dir.exists():
        print(f"Error: media_dir does not exist: {media_dir}", file=sys.stderr)
        return 1

    workspace_root = _WORKSPACE_ROOT
    lib_info = inspect_media_library(media_dir, workspace_root=workspace_root)
    q_objs = generate_intake_questions(media_dir, args.prompt or "")
    questions = [q.model_dump(mode="json") for q in q_objs]
    ask_q_payload = build_ask_question_tool_payload(q_objs)
    dur_rec = compute_director_duration_recommendation(lib_info)
    col_look, col_rat = _recommend_color_look(lib_info, user_prompt=args.prompt or "")

    out_path = (
        Path(args.output).resolve()
        if args.output
        else (Path(args.project_dir).resolve() / "library_inspection.json" if args.project_dir else media_dir / "library_inspection.json")
    )
    doc = {
        "library": lib_info,
        "director_duration_recommendation": dur_rec,
        "director_color_recommendation": {
            "recommended_color_look": col_look,
            "rationale": col_rat,
        },
        "intake_questions": questions,
        "ask_question_tool_payload": ask_q_payload,
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
    project_dir = Path(args.project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    media_dir = _resolve_cloud_or_local_media_dir(args)

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
    if getattr(args, "photo_modes", None):
        answers["photo_effects_suite"] = args.photo_modes
    if getattr(args, "overlay_fxs", None):
        answers["photo_overlay_finishers"] = args.overlay_fxs
    if args.bgm_mood:
        answers["bgm_mood"] = args.bgm_mood
    if args.dialogue_preference:
        answers["dialogue_preference"] = args.dialogue_preference
    if args.duration:
        answers["target_duration_sec"] = str(args.duration)
    if getattr(args, "travel_map_mode", None):
        answers["travel_map_mode"] = args.travel_map_mode
    if getattr(args, "travel_map_style", None):
        answers["travel_map_style"] = args.travel_map_style
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
        use_antigravity=bool(getattr(args, "use_antigravity", False)),
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
    project_dir = Path(args.project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    media_dir = _resolve_cloud_or_local_media_dir(args)

    meta_path = project_dir / "project_meta.json"
    existing_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    interview_answers = dict(existing_meta.get("interview_answers") or {})
    interview_answers.update(
        {
            "narrative_style": args.narrative_style,
            "photo_mode": args.photo_mode,
            "bgm_mood": args.bgm_mood,
            "dialogue_preference": interview_answers.get(
                "dialogue_preference", "keep_full_sentences_with_broll_cutaways"
            ),
            "must_include_topics": args.prompt or interview_answers.get("must_include_topics", ""),
        }
    )
    if args.duration:
        interview_answers["target_duration_sec"] = str(args.duration)
    if getattr(args, "travel_map_mode", None):
        interview_answers["travel_map_mode"] = args.travel_map_mode
    if getattr(args, "travel_map_style", None):
        interview_answers["travel_map_style"] = args.travel_map_style

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

    skip_whisper = bool(getattr(args, "skip_whisper", False) or getattr(args, "fast_prep", False))
    skip_auto_bgm = bool(getattr(args, "skip_auto_bgm", False) or getattr(args, "fast_prep", False))
    enable_color_grading = bool(getattr(args, "enable_color_grading", False))
    opt_roles = ["vfx_designer"]
    if enable_color_grading:
        opt_roles.append("color_grader")

    bb = run_multi_agent_pipeline(
        media_dir=media_dir,
        project_dir=project_dir,
        user_prompt=args.prompt or "",
        extra_audio_paths=extra_audio,
        interview_answers=interview_answers,
        enabled_optional_roles=opt_roles,
        width=args.width,
        height=args.height,
        fps=args.fps,
        hdr=args.hdr,
        use_antigravity=bool(getattr(args, "use_antigravity", False)),
        render_video=args.render,
        skip_whisper=skip_whisper,
        skip_auto_bgm=skip_auto_bgm,
        enable_color_grading=enable_color_grading,
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
        "timeline_fcpxml": str(project_dir / "timeline.fcpxml") if (project_dir / "timeline.fcpxml").exists() else None,
        "rendered_mp4": str(project_dir / "final_story.mp4") if (project_dir / "final_story.mp4").exists() else None,
    }
    out_path = Path(args.output).resolve() if args.output else (project_dir / "pipeline_summary.json")
    _write_json(out_path, summary_doc)
    print(
        f"Success! Multi-Agent Pipeline Complete (Grade: {summary_doc['director_grade']} "
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
        use_antigravity=bool(getattr(args, "use_antigravity", False)),
    )
    bb.load_existing_contracts()
    director = ProductionDirectorAgent()
    result = director.run(bb)
    bb.save_snapshot()

    out_path = Path(args.output).resolve() if args.output else (project_dir / "director_review.json")
    sheet_path = project_dir / "cache" / "curation" / "director_screening_sheet.jpg"
    sheet_line = f"\nScreening sheet: {sheet_path}" if sheet_path.exists() else ""
    print(
        f"Success! Executive Director Audit Complete: Grade {result.get('quality_grade')} "
        f"({result.get('overall_quality_score')}/10).\n"
        f"Report written to: {out_path}{sheet_line}"
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
    last_cb: list[tuple[str, int]] = [("", -1)]

    def _live_progress(msg: str, frac: float) -> None:
        pct = max(0, min(100, int(round(float(frac) * 100.0))))
        if (msg, pct) != last_cb[0]:
            last_cb[0] = (msg, pct)
            print(f"[{time.strftime('%H:%M:%S')}] [{pct:3d}%] {msg}", flush=True)

    outputs = render_timeline(
        timeline,
        out_mp4,
        work_dir=project_dir / "render_cache",
        progress_callback=_live_progress,
    )
    export_all_nle_formats(timeline, project_dir)

    extracted_frames: list[str] = []
    if args.extract_frames and out_mp4.exists():
        frames_dir = project_dir / "verification_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        ts_list: list[float] = []
        for raw_ts in args.extract_frames.split(","):
            raw_ts = raw_ts.strip()
            if not raw_ts:
                continue
            try:
                ts_list.append(float(raw_ts))
            except ValueError:
                pass

        def _extract_one(ts: float) -> Optional[str]:
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
            return str(jpg_path) if jpg_path.exists() else None

        if len(ts_list) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(ts_list))) as ex:
                extracted_frames = [p for p in ex.map(_extract_one, ts_list) if p is not None]
        elif ts_list:
            one_p = _extract_one(ts_list[0])
            if one_p is not None:
                extracted_frames = [one_p]

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
            use_antigravity=bool(getattr(args, "use_antigravity", False)),
        )
        bb.load_existing_contracts()
        for role_id in [
            "narrator_voiceover",
            "dialogue_editor",
            "montage_editor",
            "camera_operator",
            "lead_editor",
            "motion_graphics",
            "vfx_designer",
            "colorist",
            "sound_designer",
            "showrunner",
            "visual_qc",
            "production_director",
        ]:
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
        if getattr(args, "origin", None):
            op["origin"] = args.origin
        if getattr(args, "destination", None):
            op["destination"] = args.destination
        if getattr(args, "map_style", None):
            op["style"] = args.map_style
        if getattr(args, "placement", None):
            op["placement"] = args.placement
        if getattr(args, "transport_mode", None):
            op["transport_mode"] = args.transport_mode
        if getattr(args, "leg_index", None) is not None:
            op["leg_index"] = int(args.leg_index)
        if getattr(args, "waypoints_json", None):
            raw_wp = args.waypoints_json
            wp_p = Path(raw_wp)
            op["waypoints"] = json.loads(wp_p.read_text(encoding="utf-8")) if wp_p.exists() else json.loads(raw_wp)
        if getattr(args, "look", None):
            op["look"] = args.look
        if getattr(args, "lut_intensity", None) is not None:
            op["intensity"] = float(args.lut_intensity)
        if getattr(args, "exposure_ev", None) is not None:
            op["exposure_ev"] = float(args.exposure_ev)
        if getattr(args, "wb_red", None) is not None:
            op["wb_red"] = float(args.wb_red)
        if getattr(args, "wb_blue", None) is not None:
            op["wb_blue"] = float(args.wb_blue)
        if getattr(args, "saturation", None) is not None:
            op["saturation"] = float(args.saturation)
        if getattr(args, "contrast", None) is not None:
            op["contrast"] = float(args.contrast)
        if getattr(args, "skin_protection", None) is not None:
            op["skin_protection"] = float(args.skin_protection)
        if getattr(args, "grading_strategy", None):
            op["strategy"] = args.grading_strategy
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
        query_keywords=keywords or ["cinematic", "acoustic", "waltz", "piano", "documentary"],
        mood_profile=args.mood_profile or "classical_chamber_documentary",
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
        tl = EditTimeline.from_yaml(tl_path)
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
        non_music = [a for a in tl.audio_tracks if a.role != "music"]
        tl.audio_tracks = [
            *non_music,
            AudioTrackItem(
                id="bgm_master_01",
                source=str(out_mp3.resolve()),
                role="music",
                timeline_start=0.0,
                source_in=0.0,
                duration=round(tl.total_duration, 2),
                volume=float(args.bgm_volume or 0.85),
                duck_under_dialogue=True,
                duck_volume=float(args.duck_volume or 0.35),
                fade_in=1.5,
                fade_out=3.0,
            ),
        ]
        tl.to_yaml(tl_path)

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
        caption_title=args.caption_title or "Curated Architectural & Fine Art Showcase",
        caption_subtitle=args.caption_subtitle or "Cinema-Grade 2.5D Parallax, Loupe, Triptych & Monograph Suite · 4K HDR",
        camera_badge=args.camera_badge or "FULL FRAME · 4K 10-BIT HLG HDR",
    )
    print(f"Success! Rendered 10-Mode Premium Photo Effects Showcase Sheet to: {res_jpg}")
    return 0


def _collect_project_sample_media(project_dir: Path, limit: int = 4) -> list[str]:
    """Select diverse existing visual assets (photos/videos) from timeline, photo_curation, or media_index."""
    picked: list[str] = []
    seen: set[str] = set()

    def _add(p_str: str) -> None:
        if p_str and p_str not in seen and Path(p_str).exists():
            seen.add(p_str)
            picked.append(p_str)

    tl_path = project_dir / "timeline.yaml"
    if tl_path.exists():
        from video_editing_ai.models.schema import EditTimeline
        from video_editing_ai.renderer.color_grading import is_synthetic_graphic_clip

        tl = EditTimeline.from_yaml(tl_path)
        for it in list(tl.visual_track) + list(tl.broll_track):
            if not is_synthetic_graphic_clip(it.source, it.id, getattr(it, "narrative_note", "")):
                _add(it.source)
            if len(picked) >= limit:
                return picked

    pc_path = project_dir / "photo_curation.json"
    if pc_path.exists():
        doc = json.loads(pc_path.read_text(encoding="utf-8"))
        for it in doc.get("items", []):
            if it.get("is_selected"):
                _add(str(it.get("source_path", "")))
            if len(picked) >= limit:
                return picked

    mi_path = project_dir / "media_index.json"
    if mi_path.exists():
        doc = json.loads(mi_path.read_text(encoding="utf-8"))
        for sc in doc.get("subclips", []):
            _add(str(sc.get("source_path", "")))
            if len(picked) >= limit:
                return picked
    return picked


def cmd_preview_color_grades(args: argparse.Namespace) -> int:
    """Mechanically render a 2560x1440 4x2 visual lookbook across all 8 curated film stocks."""
    from video_editing_ai.renderer.color_grading import render_color_lookbook_sheet

    proj = Path(args.project_dir).resolve()
    proj.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(args.output).resolve() if args.output else (proj / "contact_sheets" / "color_lookbook_8stocks.jpg")

    samples: list[str] = []
    if getattr(args, "samples", None):
        samples = [s.strip() for s in str(args.samples).split(",") if s.strip() and Path(s.strip()).exists()]
    if not samples:
        samples = _collect_project_sample_media(proj, limit=4)
    if not samples:
        print("Error: No valid media samples found; pass --samples or populate project_dir.", file=sys.stderr)
        return 1

    res = render_color_lookbook_sheet(
        sample_paths=samples,
        output_path=out_jpg,
        project_title=proj.name.replace("_", " ").title(),
    )
    print(f"Success! Rendered 8-Stock DI Color Lookbook Sheet to: {res}")
    return 0


def cmd_render_color_scopes(args: argparse.Namespace) -> int:
    """Render a 2560x1440 Split-Wipe & RGB Parade waveform scopes sheet for Stage 42 telemetry audit."""
    from video_editing_ai.agents.base import PipelineBlackboard
    from video_editing_ai.agents.plugins.color_grading_agent import (
        CinemaColorGradingAgent,
        pick_representative_scope_entries,
    )
    from video_editing_ai.models.schema import ColorGradeReport, EditTimeline
    from video_editing_ai.renderer.color_grading import (
        is_synthetic_graphic_clip,
        render_color_grading_scopes_sheet,
    )

    proj = Path(args.project_dir).resolve()
    tl_path = proj / "timeline.yaml"
    if not tl_path.exists():
        print(f"Error: timeline.yaml not found in {proj}", file=sys.stderr)
        return 1

    tl = EditTimeline.from_yaml(tl_path)
    plan_path = proj / "color_grade_plan.json"
    if not plan_path.exists():
        bb = PipelineBlackboard(media_dir=proj, project_dir=proj, use_antigravity=False)
        bb.timeline = tl
        bb.hdr = bool(tl.settings.hdr)
        CinemaColorGradingAgent().run(bb)
        tl = EditTimeline.from_yaml(tl_path)

    report = ColorGradeReport.from_json_file(plan_path)
    by_id = {g.clip_id: g for g in report.clip_grades}
    all_items = list(tl.visual_track) + list(tl.broll_track)
    candidates: list[dict[str, Any]] = []
    for it in all_items:
        if is_synthetic_graphic_clip(it.source, it.id, getattr(it, "narrative_note", "")):
            continue
        if not Path(it.source).exists():
            continue
        candidates.append(
            {
                "clip_id": it.id,
                "media_type": it.type,
                "source": it.source,
                "start": it.source_in,
                "duration": it.duration,
                "grade": by_id.get(it.id),
            }
        )

    picked = pick_representative_scope_entries(candidates, limit=6)
    if not picked:
        print("Error: No existing camera video/photo files resolved on timeline.", file=sys.stderr)
        return 1

    out_jpg = Path(args.output).resolve() if args.output else (proj / "contact_sheets" / "color_grading_scopes.jpg")
    res = render_color_grading_scopes_sheet(picked, out_jpg, report=report)
    print(f"Success! Rendered RGB Parade Scopes Sheet to: {res}")
    return 0


def cmd_apply_color_grade(args: argparse.Namespace) -> int:
    """Apply Stage 42 DI color grade overrides or AI decision packet to timeline.yaml and color_grade_plan.json."""
    from video_editing_ai.agents.base import PipelineBlackboard
    from video_editing_ai.agents.plugins.color_grading_agent import (
        CinemaColorGradingAgent,
        apply_color_grade_overrides,
    )
    from video_editing_ai.exporters.nle_export import export_all_nle_formats
    from video_editing_ai.models.schema import EditTimeline

    proj = Path(args.project_dir).resolve()
    tl_path = proj / "timeline.yaml"
    if not tl_path.exists():
        print(f"Error: timeline.yaml not found in {proj}", file=sys.stderr)
        return 1

    if getattr(args, "decision_json", None):
        raw_dec = args.decision_json
        dec_p = Path(raw_dec)
        packet = json.loads(dec_p.read_text(encoding="utf-8")) if dec_p.exists() else json.loads(raw_dec)
        ai_dir = proj / "ai_decisions"
        ai_dir.mkdir(parents=True, exist_ok=True)
        (ai_dir / "color_grader.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")

        tl = EditTimeline.from_yaml(tl_path)
        bb = PipelineBlackboard(media_dir=proj, project_dir=proj, use_antigravity=False)
        bb.load_existing_contracts()
        bb.timeline = tl
        bb.hdr = bool(tl.settings.hdr)
        res = CinemaColorGradingAgent().run(bb)
        export_all_nle_formats(EditTimeline.from_yaml(tl_path), proj)
        print(
            f"Success! Executed Stage 42 color_grader with ai_decisions/color_grader.json: "
            f"look={res.get('primary_film_stock_look')}, shots={res.get('shots_matched_count')}"
        )
        return 0

    tl = EditTimeline.from_yaml(tl_path)
    report = apply_color_grade_overrides(
        timeline=tl,
        project_dir=proj,
        look_name=getattr(args, "look", None),
        clip_id=getattr(args, "clip_id", None),
        intensity=getattr(args, "intensity", None),
        exposure_ev=getattr(args, "exposure_ev", None),
        wb_red=getattr(args, "wb_red", None),
        wb_blue=getattr(args, "wb_blue", None),
        saturation=getattr(args, "saturation", None),
        contrast=getattr(args, "contrast", None),
        skin_protection=getattr(args, "skin_protection", None),
        strategy=getattr(args, "strategy", None),
    )
    tl.to_yaml(tl_path)
    export_all_nle_formats(tl, proj)
    print(
        f"Success! Applied DI Color Grade (look='{report.primary_film_stock_look}', "
        f"strategy='{report.grading_strategy}', shots={report.shots_matched_count}) -> {proj / 'color_grade_plan.json'}"
    )
    return 0


def cmd_apply_links_style(args: argparse.Namespace) -> int:
    """Transform a project into Links pure-white exhibition stills, 880-nit HLG Nordic Clean color & ambient piano BGM."""
    from video_editing_ai.planner.links_style import apply_links_cinema_style

    proj = Path(args.project_dir).resolve()
    res = apply_links_cinema_style(
        project_dir=proj,
        target_bgm_lufs=float(getattr(args, "bgm_lufs", -15.5)),
        bgm_volume=float(getattr(args, "bgm_volume", 0.24)),
        max_diptychs=int(getattr(args, "max_diptychs", 2)),
        max_shutter_cues=int(getattr(args, "max_shutters", 18)),
        offline_bgm_only=bool(getattr(args, "offline_bgm", False)),
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def cmd_recommend_color_grade(args: argparse.Namespace) -> int:
    """Analyze photometry + 8-stock lookbook via AI VLM/ranker and recommend optimal film stock & chapter looks."""
    from video_editing_ai.agents.plugins.color_grading_agent import CinemaColorGradingAgent

    proj = Path(args.project_dir).resolve()
    tl_path = proj / "timeline.yaml"
    if not tl_path.exists():
        print(f"Error: timeline.yaml not found in {proj}", file=sys.stderr)
        return 1

    meta_path = proj / "project_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    media_dir_str = meta.get("media_dir") or str(proj)
    media_dir = Path(media_dir_str).resolve() if Path(media_dir_str).exists() else proj

    bb = PipelineBlackboard(
        media_dir=media_dir,
        project_dir=proj,
        user_prompt=meta.get("user_prompt") or "",
        interview_answers=meta.get("interview_answers") or {},
        use_antigravity=not bool(getattr(args, "no_ai", False)),
    )
    bb.load_existing_contracts()
    if bb.timeline is None:
        bb.timeline = EditTimeline.from_yaml(tl_path)
    bb.hdr = bool(bb.timeline.settings.hdr)

    rec = recommend_color_grading(bb)
    out_path = Path(args.output).resolve() if getattr(args, "output", None) else (proj / "color_grade_recommendation.json")
    if out_path != (proj / "color_grade_recommendation.json"):
        rec.to_json_file(out_path)

    if getattr(args, "apply", False):
        CinemaColorGradingAgent().run(bb)
        export_all_nle_formats(EditTimeline.from_yaml(tl_path), proj)

    print(
        f"Success! Recommended DI Color Grade: '{rec.recommended_primary_look}' "
        f"(runner-up='{rec.runner_up_look}', strategy='{rec.recommended_strategy}', "
        f"intensity={rec.recommended_lut_intensity:.2f}, source='{rec.recommendation_source}').\n"
        f"Rationale: {rec.ai_rationale}\n"
        f"Chapters mapped: {len(rec.recommended_chapter_looks)} | Lookbook: {rec.lookbook_sheet_path or 'n/a'}\n"
        f"Recommendation written to: {out_path}"
    )
    return 0


def cmd_preview_travel_maps(args: argparse.Namespace) -> int:
    """Mechanically render a 2x2 high-resolution cartography comparison sheet across all 4 Travel Map styles."""
    from video_editing_ai.renderer.travel_map import render_travel_map_showcase_sheet

    proj = Path(args.project_dir).resolve()
    proj.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(args.output).resolve() if args.output else (proj / "contact_sheets" / "travel_map_showcase.jpg")

    parsed_wps = None
    if getattr(args, "waypoints_json", None):
        raw_wp = args.waypoints_json
        wp_p = Path(raw_wp)
        parsed_wps = json.loads(wp_p.read_text(encoding="utf-8")) if wp_p.exists() else json.loads(raw_wp)

    res_jpg = render_travel_map_showcase_sheet(output_jpg=out_jpg, waypoints=parsed_wps)
    print(f"Success! Rendered 4-Style Cinema 2.5D Travel Map Showcase Sheet to: {res_jpg}")
    return 0


def cmd_render_travel_map(args: argparse.Namespace) -> int:
    """Mechanically render a standalone animated Cinema 2.5D Travel Map flyover MP4 clip."""
    import os
    import re
    from video_editing_ai.renderer.travel_map import (
        lookup_gazetteer_place,
        render_travel_map_clip,
        resolve_travel_map_style,
    )

    proj = Path(args.project_dir).resolve()
    proj.mkdir(parents=True, exist_ok=True)
    out_mp4 = Path(args.output).resolve() if args.output else (proj / "cache" / "maps" / "travel_map_preview.mp4")

    wps: list[dict[str, Any]] = []
    if getattr(args, "waypoints_json", None):
        raw_wp = args.waypoints_json
        wp_p = Path(raw_wp)
        wps = json.loads(wp_p.read_text(encoding="utf-8")) if wp_p.exists() else json.loads(raw_wp)
        for idx_w, w in enumerate(wps):
            if isinstance(w, dict) and not w.get("label"):
                w["label"] = w.get("name") or f"Stop {idx_w + 1}"
    else:
        for idx_s, raw_place, def_lat, def_lon, def_lbl in [
            (0, args.origin or "Tokyo", 35.6762, 139.6503, "Departure"),
            (1, args.destination or "Kyoto", 35.0116, 135.7681, "Destination"),
        ]:
            m_c = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*(?:,\s*(.*))?$", str(raw_place))
            if m_c:
                wps.append({"label": (m_c.group(3) or def_lbl), "lat": float(m_c.group(1)), "lon": float(m_c.group(2)), "alt_m": 40.0})
            else:
                hit = lookup_gazetteer_place(str(raw_place))
                if hit is not None:
                    wps.append({"label": hit[3], "lat": hit[0], "lon": hit[1], "alt_m": hit[2], "source": "gazetteer"})
                else:
                    wps.append({"label": str(raw_place) or def_lbl, "lat": def_lat, "lon": def_lon, "alt_m": 40.0, "source": "synthetic"})

    leg_modes = [
        str(wps[k + 1].get("transport_mode") or args.transport_mode or "auto")
        for k in range(max(0, len(wps) - 1))
    ] if len(wps) >= 2 and all(isinstance(w, dict) for w in wps) and any(w.get("transport_mode") for w in wps[1:] if isinstance(w, dict)) else None

    allow_tiles = os.environ.get("AI_EDITOR_OFFLINE_MAPS", "0") != "1"
    eff_mode = (leg_modes[0] if leg_modes else None) or args.transport_mode or "auto"
    from video_editing_ai.renderer.travel_map import haversine_km, infer_transport_mode
    focus_i = max(0, min(max(0, len(wps) - 2), int(getattr(args, "leg_index", None) or 0)))
    km_est = haversine_km(float(wps[focus_i]["lat"]), float(wps[focus_i]["lon"]), float(wps[focus_i + 1]["lat"]), float(wps[focus_i + 1]["lon"])) if len(wps) >= 2 else 0.0
    inf_m = infer_transport_mode(km_est, str(wps[focus_i].get("label", "")), str(wps[focus_i + 1].get("label", "")), hint=eff_mode)
    wp_ctx_blob = " ".join(
        filter(
            None,
            [
                args.origin or "",
                args.destination or "",
                *(f"{w.get('label', '')} {w.get('subtitle', '')}" for w in wps if isinstance(w, dict)),
            ],
        )
    )
    tm_style = resolve_travel_map_style(
        args.style or "auto",
        prompt_text=wp_ctx_blob,
        transport_mode=inf_m,
        leg_km=km_est,
    )
    req_leg_idx = getattr(args, "leg_index", None)
    use_leg_idx = int(req_leg_idx) if req_leg_idx is not None else (None if len(wps) > 2 else 0)
    res_mp4 = render_travel_map_clip(
        waypoints=wps,
        output_mp4=out_mp4,
        style=tm_style,
        duration=float(args.duration or 5.2),
        leg_index=use_leg_idx,
        transport_mode=args.transport_mode or "auto",
        allow_tile_fetch=allow_tiles,
        with_audio=bool(args.with_audio),
        cache_dir=proj / "cache" / "maps" / "tiles",
        leg_modes=leg_modes,
        dynamic_zoom=bool(allow_tiles and inf_m == "flight" and km_est >= 250.0),
    )
    print(f"Success! Rendered Cinema 2.5D Travel Map flyover ({tm_style}) to: {res_mp4}")
    return 0


def cmd_apply_video_curation(args: argparse.Namespace) -> int:
    """Persist AI Agent video clip selection, golden-window [in_tc, out_tc] trims, transitions & captions."""
    proj = Path(args.project_dir).resolve()
    raw = args.overrides_json
    ov_path = Path(raw)
    if ov_path.exists():
        overrides = json.loads(ov_path.read_text(encoding="utf-8"))
    else:
        overrides = json.loads(raw)

    res = apply_video_curation_overrides(proj, overrides)
    out_p = Path(args.output).resolve() if args.output else (proj / "video_curation_applied.json")
    _write_json(out_p, res)
    print(f"Success! Applied AI video curation to {res['updated_video_clips']} clips -> {res['media_index_path']}")
    return 0


def cmd_apply_timeline_intelligence(args: argparse.Namespace) -> int:
    """Persist AI Agent transition choreography, hold durations, photo modes & Track V3 lower-third captions."""
    proj = Path(args.project_dir).resolve()
    raw = args.overrides_json
    ov_path = Path(raw)
    if ov_path.exists():
        overrides = json.loads(ov_path.read_text(encoding="utf-8"))
    else:
        overrides = json.loads(raw)

    res = apply_timeline_intelligence_overrides(proj, overrides)
    out_p = Path(args.output).resolve() if args.output else (proj / "timeline_intelligence_applied.json")
    _write_json(out_p, res)
    print(
        f"Success! Applied AI timeline intelligence ({res['updated_items']} updates, total {res['total_duration']}s) -> {res['timeline_path']}"
    )
    return 0


def cmd_render_screening_sheet(args: argparse.Namespace) -> int:
    """Mechanically render the 4x3 Multimodal Director Screening Sheet for AI visual quality inspection."""
    proj = Path(args.project_dir).resolve()
    sheet_path = render_director_screening_sheet(
        proj,
        max_tiles=args.max_tiles,
        force=bool(getattr(args, "force", False)),
    )
    if args.output:
        import shutil

        dest = Path(args.output).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sheet_path, dest)
        sheet_path = dest
    print(f"Success! Rendered Executive Director Timeline Screening Sheet to: {sheet_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai_video_studio_skill",
        description="Antigravity Skill Helper CLI for the 11-Agent AI Photo & Video Editing Studio.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # 1. inspect-library
    p_insp = sub.add_parser("inspect-library", help="Scan raw or cloud-synced media folder & get Director's recommended length")
    p_insp.add_argument("--media-dir", default=None, help="Path to raw videos and photos folder (optional if using --google-photos/--google-takeout/--project-dir)")
    p_insp.add_argument("--project-dir", default=None, help="Optional project directory for saving output")
    p_insp.add_argument("--google-photos", action="store_true", default=False, help="Sync media from Google Photos before inspecting")
    p_insp.add_argument("--auth-mode", default="auto", help="auto (browser login default) | browser | oauth")
    p_insp.add_argument("--from-downloads", default=None, help="Import browser-downloaded Google Photos .zip/media path(s) directly")
    p_insp.add_argument("--google-takeout", default=None, help="Optional Google Takeout folder or .zip archive to sync")
    p_insp.add_argument("--date-from", default=None, help="Inclusive start date (YYYY-MM-DD or ISO-8601) for cloud sync")
    p_insp.add_argument("--date-to", default=None, help="Inclusive end date (YYYY-MM-DD or ISO-8601) for cloud sync")
    p_insp.add_argument("--timezone", default="UTC", help="Timezone for cloud date filtering (UTC, local, or IANA zone)")
    p_insp.add_argument("--download-all", action="store_true", default=False, help="Download all cloud items in date range instead of Two-Pass smart curation")
    p_insp.add_argument("--prompt", default="", help="Optional initial user story prompt")
    p_insp.add_argument("--output", default=None, help="Output JSON file path")
    p_insp.set_defaults(func=cmd_inspect_library)

    # 2. intake-refine
    p_int = sub.add_parser("intake-refine", help="Run interactive Client Intake + Director length refinement")
    p_int.add_argument("--media-dir", default=None, help="Path to raw media folder (defaults to synced media_dir in project_meta.json)")
    p_int.add_argument("--project-dir", required=True, help="Target project workspace directory")
    p_int.add_argument("--prompt", default="", help="User story vision and feedback")
    p_int.add_argument("--narrative-style", default=None, help="family_keepsake_4act | day_in_the_life_vlog | upbeat_montage | dialogue_story")
    p_int.add_argument(
        "--photo-mode",
        default=None,
        help=(
            "Primary photo mode (parallax_gallery | floating_tilt_plate | glassmorphic_pedestal | split_diopter_focus | "
            "contact_sheet_zoom | medium_format_slide | criterion_atelier_split | "
            "editorial_magazine | curator_loupe | polaroid_scrapbook | "
            "triptych_cinematic | cinema_anamorphic | diptych_frame | gallery_frame | full_bleed)"
        ),
    )
    p_int.add_argument("--photo-modes", default=None, help="Multi-select comma-separated photo modes for photo_effects_suite")
    p_int.add_argument("--overlay-fxs", default=None, help="Multi-select comma-separated atmospheric finishers for photo_overlay_finishers")
    p_int.add_argument("--bgm-mood", default=None, help="auto_match | classical_chamber | acoustic_warm | cinematic_orchestral")
    p_int.add_argument("--dialogue-preference", default=None, help="keep_full_sentences_with_broll_cutaways | highlights_only | music_mostly")
    p_int.add_argument("--duration", type=int, default=None, help="Target duration in seconds (30 to 600)")
    p_int.add_argument("--travel-map-mode", default=None, help="auto | flyover_interlude | picture_in_picture | hybrid | off")
    p_int.add_argument("--travel-map-style", default=None, help="auto | midnight_vox_atlas | archival_expedition_parchment | aerial_relief_hybrid | kinfolk_minimal_monograph")
    p_int.add_argument("--answers-json", default=None, help="Optional JSON string of interview answers")
    p_int.add_argument("--output", default=None, help="Output consultation JSON path")
    p_int.set_defaults(func=cmd_intake_refine)

    # 3. run-stage
    p_stg = sub.add_parser("run-stage", help="Execute one or more specific sub-agent stages with Director review")
    p_stg.add_argument("--stage", required=True, help="Comma-separated role_ids (e.g. client_intake,music_supervisor,culler,photo_curator)")
    p_stg.add_argument("--project-dir", required=True, help="Target project workspace directory")
    p_stg.add_argument("--media-dir", default=None, help="Raw media folder path")
    p_stg.add_argument("--prompt", default="", help="Optional story prompt override")
    p_stg.add_argument("--antigravity-sdk", dest="use_antigravity", action="store_true", default=False, help="Enable nested Google Antigravity SDK escalation during stage runs")
    p_stg.add_argument("--no-antigravity-sdk", dest="use_antigravity", action="store_false", help="Run deterministic/local stage pass (default when orchestrated interactively by Antigravity)")
    p_stg.add_argument("--output", default=None, help="Output stage report JSON path")
    p_stg.set_defaults(func=cmd_run_stage)

    # 4. run-pipeline
    p_pipe = sub.add_parser("run-pipeline", help="Run the full 22-agent pipeline (Stage 00 -> Stage 65) and optional 4K HDR render")
    p_pipe.add_argument("--media-dir", default=None, help="Raw media folder path (optional when using --google-photos, --google-takeout, or previously synced --project-dir)")
    p_pipe.add_argument("--project-dir", required=True, help="Target project workspace directory")
    p_pipe.add_argument("--google-photos", action="store_true", default=False, help="Sync media from Google Photos before running pipeline")
    p_pipe.add_argument("--auth-mode", default="auto", help="auto (browser login default) | browser | oauth")
    p_pipe.add_argument("--from-downloads", default=None, help="Import browser-downloaded Google Photos .zip/media path(s) directly")
    p_pipe.add_argument("--google-takeout", default=None, help="Optional Google Takeout folder or .zip archive before running pipeline")
    p_pipe.add_argument("--date-from", default=None, help="Cloud sync inclusive start date (YYYY-MM-DD or ISO-8601)")
    p_pipe.add_argument("--date-to", default=None, help="Cloud sync inclusive end date (YYYY-MM-DD or ISO-8601)")
    p_pipe.add_argument("--timezone", default="UTC", help="Timezone for cloud date filtering (UTC, local, or IANA zone)")
    p_pipe.add_argument("--download-all", action="store_true", default=False, help="Download all cloud items in date range instead of smart Two-Pass curation")
    p_pipe.add_argument("--prompt", default="", help="Story vision prompt")
    p_pipe.add_argument("--duration", type=int, default=None, help="Target duration in seconds (30 to 600)")
    p_pipe.add_argument("--photo-mode", default="parallax_gallery", help="Primary fallback photo mode (any of the 16 PhotoModeType modes)")
    p_pipe.add_argument("--narrative-style", default="family_keepsake_4act", help="Narrative style")
    p_pipe.add_argument("--travel-map-mode", default=None, help="auto | flyover_interlude | picture_in_picture | hybrid | off")
    p_pipe.add_argument("--travel-map-style", default=None, help="auto | midnight_vox_atlas | archival_expedition_parchment | aerial_relief_hybrid | kinfolk_minimal_monograph")
    p_pipe.add_argument("--bgm-mood", default="auto_match", help="BGM mood preference")
    p_pipe.add_argument("--bgm", default=None, help="Optional custom audio file path")
    p_pipe.add_argument("--width", type=int, default=3840, help="Video width (default 3840)")
    p_pipe.add_argument("--height", type=int, default=2160, help="Video height (default 2160)")
    p_pipe.add_argument("--fps", type=int, default=30, help="Frame rate (default 30)")
    p_pipe.add_argument("--hdr", dest="hdr", action="store_true", default=True, help="Enable 10-bit HLG HDR (default)")
    p_pipe.add_argument("--sdr", dest="hdr", action="store_false", help="Render in 8-bit SDR")
    p_pipe.add_argument(
        "--enable-color-grading",
        dest="enable_color_grading",
        action="store_true",
        default=False,
        help="Opt-in: enable Stage 42 DI film-stock color grading (disabled by default to preserve original camera colors)",
    )
    p_pipe.add_argument("--render", dest="render", action="store_true", default=True, help="Render final_story.mp4 (default)")
    p_pipe.add_argument("--no-render", dest="render", action="store_false", help="Skip MP4 render (contracts + timeline only)")
    p_pipe.add_argument("--antigravity-sdk", dest="use_antigravity", action="store_true", default=False, help="Enable nested Google Antigravity SDK escalation inside sub-agents")
    p_pipe.add_argument("--no-antigravity-sdk", dest="use_antigravity", action="store_false", help="Skip nested Antigravity SDK calls (default when orchestrated interactively by Antigravity)")
    p_pipe.add_argument("--skip-whisper", action="store_true", default=False, help="Skip local Whisper speech transcription during prep")
    p_pipe.add_argument("--skip-auto-bgm", action="store_true", default=False, help="Skip online BGM scouting/synthesis during prep (use synthetic beat grid until scout-bgm/apply-bgm-plan)")
    p_pipe.add_argument("--fast-prep", action="store_true", default=False, help="Enable both --skip-whisper and --skip-auto-bgm for rapid structural prep")
    p_pipe.add_argument("--output", default=None, help="Output summary JSON path")
    p_pipe.set_defaults(func=cmd_run_pipeline)

    # 5. director-audit
    p_dir = sub.add_parser("director-audit", help="Run Stage 60 Executive Movie Production Director screening")
    p_dir.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_dir.add_argument("--media-dir", default=None, help="Optional media directory")
    p_dir.add_argument("--antigravity-sdk", dest="use_antigravity", action="store_true", default=False, help="Enable nested Google Antigravity SDK escalation during Director screening")
    p_dir.add_argument("--no-antigravity-sdk", dest="use_antigravity", action="store_false", help="Run local deterministic Director screening (default)")
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
    p_sp.add_argument("--antigravity-sdk", dest="use_antigravity", action="store_true", default=False, help="Enable nested Google Antigravity SDK escalation during timeline reassembly")
    p_sp.add_argument("--no-antigravity-sdk", dest="use_antigravity", action="store_false", help="Run local reassembly pass (default)")
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
        choices=[
            "update-plaque",
            "set-photo-effect",
            "trim",
            "drop",
            "swap",
            "move",
            "set-cold-open",
            "audio-mix",
            "insert-travel-map",
            "set-color-grade",
        ],
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
        help=(
            "Updated photo mode (parallax_gallery | floating_tilt_plate | glassmorphic_pedestal | split_diopter_focus | "
            "contact_sheet_zoom | medium_format_slide | criterion_atelier_split | "
            "triptych_cinematic | polaroid_scrapbook | curator_loupe | editorial_magazine | cinema_anamorphic | "
            "diptych_frame | gallery_frame | full_bleed)"
        ),
    )
    p_ec.add_argument(
        "--overlay-fx",
        default=None,
        help=(
            "Composable atmospheric finisher (none | golden_light_leak | champagne_bokeh | museum_spotlight | "
            "nostalgic_film_grain | anamorphic_streak | prism_chromatic_edge | velvet_vignette_rembrandt | fine_art_rag_texture)"
        ),
    )
    p_ec.add_argument(
        "--secondary-source",
        default=None,
        help="Optional 2nd photo path for diptych_frame / polaroid_scrapbook / triptych_cinematic / contact_sheet_zoom / criterion_atelier_split",
    )
    p_ec.add_argument(
        "--tertiary-source",
        default=None,
        help="Optional 3rd photo path for triptych_cinematic / contact_sheet_zoom / criterion_atelier_split",
    )
    p_ec.add_argument("--preset", default=None, help="Updated Ken Burns preset")
    p_ec.add_argument("--before-clip-id", default=None, help="Move clip before this clip_id (for --action move)")
    p_ec.add_argument("--after-clip-id", default=None, help="Move clip after this clip_id (for --action move)")
    p_ec.add_argument("--subclip-id", default=None, help="Subclip ID for --action set-cold-open")
    p_ec.add_argument("--bgm-volume", type=float, default=None, help="BGM base volume (0.05-1.0) for --action audio-mix")
    p_ec.add_argument("--duck-volume", type=float, default=None, help="BGM ducking volume (0.02-0.5) for --action audio-mix")
    p_ec.add_argument("--origin", default=None, help="Departure city/place or 'lat,lon' for --action insert-travel-map")
    p_ec.add_argument("--destination", default=None, help="Arrival city/place or 'lat,lon' for --action insert-travel-map")
    p_ec.add_argument("--map-style", default=None, help="Cartography style for --action insert-travel-map (auto | satellite_hybrid_4k | google_road_navigator | aerial_relief_hybrid | midnight_vox_atlas | archival_expedition_parchment | kinfolk_minimal_monograph)")
    p_ec.add_argument("--placement", default=None, help="Placement mode: v1_interlude | v2_flyover | v3_pip for --action insert-travel-map")
    p_ec.add_argument("--transport-mode", default=None, help="Transport mode: auto | flight | train | drive | walk | boat")
    p_ec.add_argument("--leg-index", type=int, default=None, help="Optional 0-based focused leg index when passing multi-stop waypoints")
    p_ec.add_argument("--waypoints-json", default=None, help="Optional multi-stop waypoints JSON string or file path for --action insert-travel-map")
    p_ec.add_argument("--look", default=None, help="Film stock look name for --action set-color-grade")
    p_ec.add_argument("--lut-intensity", type=float, default=None, help="3D LUT strength (0.0-1.0) for --action set-color-grade")
    p_ec.add_argument("--exposure-ev", type=float, default=None, help="Exposure compensation in EV (-1.5..1.5) for --action set-color-grade")
    p_ec.add_argument("--wb-red", type=float, default=None, help="Red WB shift (-0.35..0.35) for --action set-color-grade")
    p_ec.add_argument("--wb-blue", type=float, default=None, help="Blue WB shift (-0.35..0.35) for --action set-color-grade")
    p_ec.add_argument("--saturation", type=float, default=None, help="Saturation multiplier for --action set-color-grade")
    p_ec.add_argument("--contrast", type=float, default=None, help="Contrast multiplier for --action set-color-grade")
    p_ec.add_argument("--skin-protection", type=float, default=None, help="Skin tone vector protection (0.0-1.0)")
    p_ec.add_argument("--grading-strategy", default=None, help="chapter_adaptive | uniform_stock | natural_neutral")
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
    p_sb.add_argument("--keywords", default="cinematic,acoustic,waltz,piano,documentary", help="Comma-separated search keywords")
    p_sb.add_argument("--mood-profile", default="classical_chamber_documentary", help="Target mood profile")
    p_sb.add_argument("--target-duration", type=float, default=240.0, help="Target video duration in seconds")
    p_sb.add_argument("--download-limit", type=int, default=5, help="Max candidate tracks to download")
    p_sb.add_argument("--output", default=None, help="Optional output bgm_candidates.json path")
    p_sb.set_defaults(func=cmd_scout_bgm)

    # 14. apply-bgm-plan
    p_ab = sub.add_parser(
        "apply-bgm-plan",
        help="Mechanically master the AI BGM Agent's selected track(s) to cover 100%% of timeline duration at -14 LUFS",
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
        help="Mechanically render a 4x4 visual comparison showcase sheet of all 16 Photo Modes & 8 Atmospheric Finishers",
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

    # 16. apply-video-curation
    p_avc = sub.add_parser(
        "apply-video-curation",
        help="Persist AI Agent video clip selection, golden-window [in_tc, out_tc] trims, transitions & captions",
    )
    p_avc.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_avc.add_argument("--overrides-json", required=True, help="JSON string or file path with video clip overrides")
    p_avc.add_argument("--output", default=None, help="Optional output report JSON path")
    p_avc.set_defaults(func=cmd_apply_video_curation)

    # 17. apply-timeline-intelligence
    p_ati = sub.add_parser(
        "apply-timeline-intelligence",
        help="Persist AI Agent transition choreography, hold durations, photo modes & Track V3 lower-third captions",
    )
    p_ati.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_ati.add_argument("--overrides-json", required=True, help="JSON string or file path with clips / lower_thirds overrides")
    p_ati.add_argument("--output", default=None, help="Optional output report JSON path")
    p_ati.set_defaults(func=cmd_apply_timeline_intelligence)

    # 18. render-screening-sheet
    p_rss = sub.add_parser(
        "render-screening-sheet",
        help="Mechanically render the 4x3 Multimodal Director Screening Sheet for AI visual quality inspection",
    )
    p_rss.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_rss.add_argument("--max-tiles", type=int, default=12, help="Number of timeline cuts to sample (default 12)")
    p_rss.add_argument("--force", action="store_true", default=False, help="Force re-render even if timeline SHA-1 matches cached screening sheet")
    p_rss.add_argument("--output", default=None, help="Optional output JPEG path")
    p_rss.set_defaults(func=cmd_render_screening_sheet)

    # 19. preview-travel-maps
    p_ptm = sub.add_parser(
        "preview-travel-maps",
        help="Mechanically render a 2x2 showcase JPEG comparing all 4 Cinema 2.5D Travel Map cartography styles",
    )
    p_ptm.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_ptm.add_argument("--waypoints-json", default=None, help="Optional JSON string or file path of waypoint dicts")
    p_ptm.add_argument("--output", default=None, help="Optional output JPEG path")
    p_ptm.set_defaults(func=cmd_preview_travel_maps)

    # 20. render-travel-map
    p_rtm = sub.add_parser(
        "render-travel-map",
        help="Mechanically render a standalone animated Cinema 2.5D Travel Map flyover MP4 clip",
    )
    p_rtm.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_rtm.add_argument("--origin", default=None, help="Departure city/place or 'lat,lon'")
    p_rtm.add_argument("--destination", default=None, help="Arrival city/place or 'lat,lon'")
    p_rtm.add_argument("--waypoints-json", default=None, help="Optional multi-stop waypoints JSON string or file")
    p_rtm.add_argument("--style", default="auto", help="auto | satellite_hybrid_4k | google_road_navigator | aerial_relief_hybrid | midnight_vox_atlas | archival_expedition_parchment | kinfolk_minimal_monograph")
    p_rtm.add_argument("--transport-mode", default="auto", help="auto | flight | train | drive | walk | boat")
    p_rtm.add_argument("--leg-index", type=int, default=None, help="Optional 0-based leg index to zoom closely into one stop leg while keeping full multi-stop context visible")
    p_rtm.add_argument("--duration", type=float, default=5.2, help="Animation duration in seconds (default 5.2)")
    p_rtm.add_argument("--with-audio", action="store_true", default=False, help="Synthesize stereo flyover sweep + landing chime audio")
    p_rtm.add_argument("--output", default=None, help="Optional output MP4 path")
    p_rtm.set_defaults(func=cmd_render_travel_map)

    # 21. google-photos-auth
    p_gpa = sub.add_parser(
        "google-photos-auth",
        help="Authenticate or inspect OAuth 2.0 PKCE credentials for Google Photos Picker API",
    )
    p_gpa.add_argument("--client-secrets", default=None, help="Optional path to Desktop OAuth client_secret.json")
    p_gpa.add_argument("--token-path", default=None, help="Optional path to OAuth token.json")
    p_gpa.add_argument("--no-browser", action="store_true", default=False, help="Print consent URL without opening browser")
    p_gpa.add_argument("--status", action="store_true", default=False, help="Print JSON credential & token readiness status")
    p_gpa.add_argument("--revoke", action="store_true", default=False, help="Delete stored OAuth token.json")
    p_gpa.set_defaults(func=cmd_google_photos_auth)

    # 22. sync-google-photos
    p_sgp = sub.add_parser(
        "sync-google-photos",
        help="Scan Google Photos (Browser Login, OAuth Picker, or Takeout) for [date_from, date_to] and stage curated assets",
    )
    p_sgp.add_argument("--project-dir", default=None, help="Target project workspace directory (required unless --search-url-only)")
    p_sgp.add_argument("--date-from", default=None, help="Inclusive start date (YYYY-MM-DD or ISO-8601)")
    p_sgp.add_argument("--date-to", default=None, help="Inclusive end date (YYYY-MM-DD or ISO-8601)")
    p_sgp.add_argument("--timezone", default="UTC", help="Timezone for date filtering (UTC, local, or IANA zone)")
    p_sgp.add_argument("--auth-mode", default="auto", help="Connection mode: auto (browser login default) | browser | oauth")
    p_sgp.add_argument("--from-downloads", default=None, help="Already-downloaded .zip/media file(s) or folder from Google Photos browser")
    p_sgp.add_argument("--downloads-dir", default=None, help="Browser downloads folder to watch (defaults to ~/Downloads)")
    p_sgp.add_argument("--watch-timeout", type=float, default=600.0, help="Seconds to wait for browser download completion")
    p_sgp.add_argument("--search-url-only", action="store_true", default=False, help="Print JSON with Google Photos search URL & date query and exit")
    p_sgp.add_argument("--takeout", default=None, help="Optional Google Takeout folder or .zip archive instead of browser/OAuth")
    p_sgp.add_argument("--target-duration", type=float, default=155.0, help="Target edit duration in seconds for smart selection budget")
    p_sgp.add_argument("--max-photos", type=int, default=None, help="Optional cap on smart-downloaded photos")
    p_sgp.add_argument("--max-videos", type=int, default=None, help="Optional cap on smart-downloaded videos")
    p_sgp.add_argument("--download-all", action="store_true", default=False, help="Download/keep every in-range item without smart curation pruning")
    p_sgp.add_argument("--headroom-multiplier", type=float, default=1.6, help="Recall headroom multiplier over strict timeline count (default 1.6)")
    p_sgp.add_argument("--plan-only", action="store_true", default=False, help="Run Pass-1 curation + paginated 5x4 contact sheets and pause for Pass-1.5 Vision review before staging/Pass-2 download")
    p_sgp.add_argument("--favorites-only", action="store_true", default=False, help="Scope browser search URL to ★ Favorited items only")
    p_sgp.add_argument("--album-url", default=None, help="Optional direct Google Photos album URL to open instead of date search")
    p_sgp.add_argument("--favorites", default=None, help="Comma-separated Favorited filenames or JSON manifest path to mark ★ favorite")
    p_sgp.add_argument("--session-id", default=None, help="Optional existing Google Photos Picker sessionId")
    p_sgp.add_argument("--no-browser", action="store_true", default=False, help="Do not auto-open browser")
    p_sgp.add_argument("--client-secrets", default=None, help="Optional path to OAuth client_secret.json")
    p_sgp.add_argument("--token-path", default=None, help="Optional path to OAuth token.json")
    p_sgp.add_argument("--output", default=None, help="Optional output JSON report path")
    p_sgp.set_defaults(func=cmd_sync_google_photos)

    # 22b. apply-cloud-picks (Pass-1.5 AI Vision / User Rescue Gate)
    p_acp = sub.add_parser(
        "apply-cloud-picks",
        help="Apply Pass-1.5 AI Vision / user rescues & rejections to google_photos_manifest.json and execute Pass-2 selective staging/download",
    )
    p_acp.add_argument("--project-dir", required=True, help="Target project workspace directory containing google_photos_manifest.json")
    p_acp.add_argument("--overrides-json", default=None, help="JSON string or file path with {'select': ['#07'], 'skip': ['#12']} or list of tile overrides")
    p_acp.add_argument("--no-download", action="store_true", default=False, help="Update manifest & contact sheets without executing Pass-2 downloads/staging")
    p_acp.add_argument("--output", default=None, help="Optional output JSON report path")
    p_acp.set_defaults(func=cmd_apply_cloud_picks)

    # 23. youtube-auth
    p_yta = sub.add_parser(
        "youtube-auth",
        help="Inspect YouTube Studio readiness or configure OAuth 2.0 PKCE credentials for YouTube Data API v3",
    )
    p_yta.add_argument("--client-secrets", default=None, help="Optional path to Desktop OAuth client_secret.json")
    p_yta.add_argument("--token-path", default=None, help="Optional path to YouTube OAuth token.json")
    p_yta.add_argument("--include-captions-scope", action="store_true", default=False, help="Include youtube.force-ssl scope for .srt upload")
    p_yta.add_argument("--no-browser", action="store_true", default=False, help="Print consent URL without opening browser")
    p_yta.add_argument("--status", action="store_true", default=False, help="Print JSON credential & token readiness status")
    p_yta.add_argument("--revoke", action="store_true", default=False, help="Delete stored YouTube OAuth token.json")
    p_yta.set_defaults(func=cmd_youtube_auth)

    # 24. publish-youtube
    p_ytp = sub.add_parser(
        "publish-youtube",
        help="Prepare YouTube draft/thumbnail/chapters and publish ONLY with explicit confirmation token",
    )
    p_ytp.add_argument("--project-dir", required=True, help="Project workspace directory containing final_story.mp4")
    p_ytp.add_argument("--video", default=None, help="Optional custom MP4 path")
    p_ytp.add_argument("--title", default=None, help="Custom YouTube video title")
    p_ytp.add_argument("--description", default=None, help="Custom YouTube video description")
    p_ytp.add_argument("--tags", default=None, help="Comma-separated YouTube tags")
    p_ytp.add_argument("--privacy", default="unlisted", help="unlisted (default) | private | public")
    p_ytp.add_argument("--made-for-kids", action="store_true", default=False, help="Mark selfDeclaredMadeForKids=True (default False)")
    p_ytp.add_argument("--category-id", default=None, help="Optional numeric YouTube categoryId")
    p_ytp.add_argument("--thumbnail", default=None, help="Optional custom thumbnail source image")
    p_ytp.add_argument("--auth-mode", default="auto", help="auto | oauth | browser")
    p_ytp.add_argument("--prepare", action="store_true", default=False, help="Generate draft, 00:00 chapters & thumbnail without uploading")
    p_ytp.add_argument("--confirmed", action="store_true", default=False, help="Confirm explicit user approval to publish")
    p_ytp.add_argument("--confirm-token", default=None, help="SHA-256 confirmation token generated by --prepare")
    p_ytp.add_argument("--record-browser-receipt", default=None, help="Record published YouTube URL or ID after browser upload")
    p_ytp.add_argument("--no-browser", action="store_true", default=False, help="Do not auto-open browser")
    p_ytp.add_argument("--client-secrets", default=None, help="Optional path to OAuth client_secret.json")
    p_ytp.add_argument("--token-path", default=None, help="Optional path to YouTube OAuth token.json")
    p_ytp.add_argument("--output", default=None, help="Optional output JSON report path")
    p_ytp.set_defaults(func=cmd_publish_youtube)

    # 25. preview-color-grades
    p_pcg = sub.add_parser(
        "preview-color-grades",
        help="Render a 2560x1440 4x2 visual lookbook sheet across all 8 curated film stocks",
    )
    p_pcg.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_pcg.add_argument("--samples", default=None, help="Optional comma-separated photo/video paths")
    p_pcg.add_argument("--output", default=None, help="Optional output JPEG path")
    p_pcg.set_defaults(func=cmd_preview_color_grades)

    # 26. render-color-scopes
    p_rcs = sub.add_parser(
        "render-color-scopes",
        help="Render a 2560x1440 Before/Graded split-wipe & RGB Parade waveform scopes sheet",
    )
    p_rcs.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_rcs.add_argument("--output", default=None, help="Optional output JPEG path")
    p_rcs.set_defaults(func=cmd_render_color_scopes)

    # 27. apply-color-grade
    p_acg = sub.add_parser(
        "apply-color-grade",
        help="Apply Stage 42 DI film stock look or per-clip primary balance overrides to timeline.yaml",
    )
    p_acg.add_argument("--project-dir", required=True, help="Project workspace directory")
    p_acg.add_argument("--decision-json", default=None, help="Path or JSON string for ai_decisions/color_grader.json")
    p_acg.add_argument("--look", default=None, help="Target film stock name (e.g. archival_kodak_2383)")
    p_acg.add_argument("--clip-id", default=None, help="Optional single clip_id to target")
    p_acg.add_argument("--intensity", type=float, default=None, help="3D LUT strength (0.0-1.0)")
    p_acg.add_argument("--exposure-ev", type=float, default=None, help="Exposure compensation in EV")
    p_acg.add_argument("--wb-red", type=float, default=None, help="Red WB shift (-0.35..0.35)")
    p_acg.add_argument("--wb-blue", type=float, default=None, help="Blue WB shift (-0.35..0.35)")
    p_acg.add_argument("--saturation", type=float, default=None, help="Saturation multiplier")
    p_acg.add_argument("--contrast", type=float, default=None, help="Contrast multiplier")
    p_acg.add_argument("--skin-protection", type=float, default=None, help="Skin tone protection (0.0-1.0)")
    p_acg.add_argument("--strategy", default=None, help="chapter_adaptive | uniform_stock | natural_neutral")
    p_acg.set_defaults(func=cmd_apply_color_grade)

    # 28. recommend-color-grade
    p_rcg = sub.add_parser(
        "recommend-color-grade",
        help="Analyze project photometry & lookbook via AI VLM/ranker and recommend optimal film stock + chapter looks",
    )
    p_rcg.add_argument("--project-dir", required=True, help="Project workspace directory containing timeline.yaml")
    p_rcg.add_argument("--no-ai", action="store_true", default=False, help="Skip external VLM calls and use offline photometric ranking only")
    p_rcg.add_argument("--apply", action="store_true", default=False, help="Also execute Stage 42 color_grader immediately with the recommendation")
    p_rcg.add_argument("--output", default=None, help="Optional output recommendation JSON path")
    p_rcg.set_defaults(func=cmd_recommend_color_grade)

    # 29. apply-links-style
    p_als = sub.add_parser(
        "apply-links-style",
        help="Transform timeline into Links cinema style (pure-white exhibition stills, 880-nit HLG Nordic Clean color & ambient piano/shutter soundscape)",
    )
    p_als.add_argument("--project-dir", required=True, help="Project workspace directory containing timeline.yaml")
    p_als.add_argument("--bgm-lufs", type=float, default=-15.5, help="Target mastered ambient BGM LUFS (default -15.5)")
    p_als.add_argument("--bgm-volume", type=float, default=0.24, help="Linear BGM mix volume (default 0.24)")
    p_als.add_argument("--max-diptychs", type=int, default=2, help="Max 2-photo diptychs to retain before converting stills to pure-white exhibition plates")
    p_als.add_argument("--max-shutters", type=int, default=18, help="Max mechanical camera shutter click SFX cues on still cuts")
    p_als.add_argument("--offline-bgm", action="store_true", default=False, help="Skip network BGM downloads and master from local library/synthesis only")
    p_als.set_defaults(func=cmd_apply_links_style)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
