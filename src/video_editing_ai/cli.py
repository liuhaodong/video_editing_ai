"""Command-Line Interface (ai-editor) for the AI Photo & Video Editing Tool."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Optional

from rich.console import Console
from rich.panel import Panel
import typer
import uvicorn

from .agents import AgentRegistry, generate_intake_questions, run_multi_agent_pipeline
from .analyzer.ag_annotator import analyze_media_directory
from .cloud.google_photos import (
    GooglePhotosAuthManager,
    build_google_photos_web_search_url,
    format_picker_date_query,
    get_default_browser_downloads_dir,
    run_google_photos_sync,
)
from .docs_sync import resolve_repo_root, sync_agent_documentation
from .doctor import collect_environment_diagnostics
from .exporters.nle_export import export_all_nle_formats, export_fcpxml
from .models.schema import EditTimeline, MediaCatalog
from .planner.story_director import generate_edit_plan
from .renderer.ffmpeg_composer import render_timeline
from .web.app import create_app

app = typer.Typer(
    name="ai-editor",
    help="AI-powered photo and video analysis, modular multi-agent story editing, and 4K HDR video renderer.",
    add_completion=False,
)
console = Console()


@app.command("agents-list")
def cmd_agents_list() -> None:
    """List all registered core and pluggable sub-agent roles in the Multi-Agent Architecture."""
    console.print(Panel("[bold cyan]Modular Multi-Agent Editing Roles (AgentRegistry)[/bold cyan]"))
    for agent_cls in AgentRegistry.list_all():
        meta = agent_cls.to_metadata_dict()
        status = "[green]ENABLED BY DEFAULT[/green]" if meta["enabled_by_default"] else "[yellow]OPTIONAL PLUGIN[/yellow]"
        console.print(
            f"  - [bold]{meta['role_id']}[/bold] (Stage {meta['stage_order']:02d}) - {meta['display_name']} [{status}]\n"
            f"    Description: {meta['description']}\n"
            f"    Requires: {meta['requires_artifacts']} -> Produces: {meta['produces_artifacts']}\n"
            f"    Skill Prompt: .agent/skills/ai-video-editor/references/{meta['skill_reference_file']}"
        )


@app.command("google-photos-auth")
def cmd_google_photos_auth(
    client_secrets: Optional[Path] = typer.Option(None, "--client-secrets", help="Path to Google Cloud Desktop OAuth client_secret.json"),
    token_path: Optional[Path] = typer.Option(None, "--token-path", help="Optional custom path to store OAuth token.json"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not auto-open browser during OAuth consent"),
    status: bool = typer.Option(False, "--status", help="Inspect current Browser Login & OAuth credential readiness"),
    revoke: bool = typer.Option(False, "--revoke", help="Delete the local stored OAuth token.json"),
) -> None:
    """Inspect zero-setup Browser Login status or configure optional Google Photos OAuth 2.0 PKCE credentials."""
    auth = GooglePhotosAuthManager(client_secrets_path=client_secrets, token_path=token_path)
    if revoke:
        removed = auth.revoke_local_token()
        console.print(f"[green][OK][/green] Local Google Photos OAuth token removed: {removed} ({auth.token_path})")
        return
    if status:
        st = auth.get_auth_status()
        oauth_badge = "[green]READY[/green]" if st["ready"] else "[yellow]OPTIONAL / NOT CONFIGURED[/yellow]"
        console.print(
            Panel(
                "[bold cyan]Google Photos Connection Status[/bold cyan]\n"
                f"  - Browser Login (Zero-Setup Default): [green]READY[/green] (watches {get_default_browser_downloads_dir()})\n"
                f"  - Picker API OAuth (Optional):        {oauth_badge}\n"
                f"  - Config Dir:     {st['config_dir']}\n"
                f"  - Client Secrets: {st['client_secrets_path']} (exists={st['client_secrets_configured']})\n"
                f"  - Token File:     {st['token_path']} (present={st['token_present']}, valid={st['token_valid']}, refresh={st['refresh_token_present']})\n"
                f"  - Scope:          {st['scope']}"
            )
        )
        return

    if not auth.client_secrets_path.exists():
        console.print(
            Panel(
                "[bold green]Zero-Setup Tip:[/bold green] Browser Login works right now with NO Google Cloud setup!\n"
                "Simply run: [bold cyan]ai-editor agents-run --google-photos --date-from YYYY-MM-DD --date-to YYYY-MM-DD -p \"...\"[/bold cyan]\n\n"
                + auth.format_setup_instructions(),
                title="[bold yellow]Optional Google Cloud OAuth Setup[/bold yellow]",
            )
        )
        raise typer.Exit(code=1)

    def _show_url(url: str) -> None:
        console.print(f"[bold cyan]Open this Google sign-in URL in your browser:[/bold cyan]\n{url}\n")
        if not no_browser:
            import webbrowser

            webbrowser.open(url, new=2)

    auth.authenticate_interactive(open_browser=not no_browser, auth_url_callback=_show_url)
    console.print(f"[bold green][OK] Google Photos OAuth connected![/bold green] Token saved to: {auth.token_path}")


def _make_cli_cloud_callbacks(no_browser: bool = False):
    def _on_browser(search_url: str, dl_dir: Path) -> None:
        console.print(
            Panel(
                "[bold cyan]Zero-Setup Google Photos Browser Login[/bold cyan]\n"
                f"  1. Opened [underline]{search_url}[/underline] in your browser (no API keys needed).\n"
                "  2. Sign into Google Photos if prompted, Shift-select items in your date range, and press [bold]Shift+D[/bold] (Download).\n"
                f"  3. Watching [underline]{dl_dir}[/underline] — burst deduplication & editing start automatically once finished!"
            )
        )
        if not no_browser and search_url:
            import webbrowser

            webbrowser.open(search_url, new=2)

    def _on_picker(uri: str, hint: str) -> None:
        hint_line = f"\nSearch tip copied to clipboard: [bold yellow]{hint}[/bold yellow]" if hint else ""
        console.print(
            Panel(
                f"[bold cyan]Select items in Google Photos Picker:[/bold cyan]\n{uri}{hint_line}\n"
                "After clicking Done in Google Photos, downloads will start automatically."
            )
        )
        if not no_browser and uri:
            import webbrowser

            webbrowser.open(uri, new=2)

    return _on_browser, _on_picker, (lambda msg: console.print(f"  [cyan]{msg}[/cyan]"))


@app.command("google-photos-sync")
def cmd_google_photos_sync(
    project_dir: Path = typer.Option(Path("workspace_project"), "-o", "--project-dir", help="Project workspace directory"),
    date_from: Optional[str] = typer.Option(None, "--date-from", help="Inclusive start date (YYYY-MM-DD or ISO-8601)"),
    date_to: Optional[str] = typer.Option(None, "--date-to", help="Inclusive end date (YYYY-MM-DD or ISO-8601)"),
    timezone_name: str = typer.Option("UTC", "--timezone", help="Date interpretation timezone (UTC, local, or IANA zone)"),
    auth_mode: str = typer.Option("auto", "--auth-mode", help="Connection mode: auto (browser login default) | browser | oauth"),
    from_downloads: Optional[str] = typer.Option(None, "--from-downloads", help="Import already-downloaded Google Photos .zip/media file(s) or folder"),
    downloads_dir: Optional[Path] = typer.Option(None, "--downloads-dir", help="Browser downloads folder to watch (defaults to ~/Downloads)"),
    watch_timeout: float = typer.Option(600.0, "--watch-timeout", help="Seconds to wait for browser download completion"),
    takeout: Optional[Path] = typer.Option(None, "--takeout", help="Optional Google Takeout folder or .zip archive instead of browser/OAuth"),
    target_duration: float = typer.Option(155.0, "--target-duration", help="Target edit duration in seconds for smart selection budget"),
    max_photos: Optional[int] = typer.Option(None, "--max-photos", help="Optional override max photos to download"),
    max_videos: Optional[int] = typer.Option(None, "--max-videos", help="Optional override max videos to download"),
    download_all: bool = typer.Option(False, "--download-all", help="Bypass Two-Pass Smart Curation and download/keep all in-range items"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Attach to an existing Google Photos Picker sessionId"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not auto-open browser"),
    client_secrets: Optional[Path] = typer.Option(None, "--client-secrets", help="Optional custom client_secret.json path"),
    token_path: Optional[Path] = typer.Option(None, "--token-path", help="Optional custom OAuth token.json path"),
) -> None:
    """Connect to Google Photos via Browser Login (or OAuth/Takeout) in [date_from, date_to] and stage smart-curated assets."""
    on_browser, on_picker, on_log = _make_cli_cloud_callbacks(no_browser=no_browser)
    manifest = run_google_photos_sync(
        project_dir=project_dir,
        date_from=date_from,
        date_to=date_to,
        tz_name=timezone_name,
        target_duration_sec=target_duration,
        max_photos=max_photos,
        max_videos=max_videos,
        download_all=download_all,
        auth_mode=auth_mode,
        from_downloads=from_downloads,
        takeout_source=takeout,
        downloads_dir=downloads_dir,
        watch_timeout_sec=watch_timeout,
        session_id=session_id,
        open_browser=not no_browser,
        client_secrets_path=client_secrets,
        token_path=token_path,
        browser_open_callback=on_browser,
        picker_ready_callback=on_picker,
        progress_callback=on_log,
    )

    console.print(
        f"[bold green][OK] Cloud Sync Complete ({manifest.source_mode})![/bold green]\n"
        f"  - In Date Range:    {manifest.in_date_range_count} / {manifest.total_scanned_in_session} scanned\n"
        f"  - Selected Media:   {manifest.selected_photos_count} photos, {manifest.selected_videos_count} videos "
        f"(skipped {manifest.skipped_count}, cache hits {manifest.cache_hits_count})\n"
        f"  - Staged Folder:    [underline]{manifest.staged_media_dir}[/underline]\n"
        f"  - Manifest JSON:    [underline]{project_dir / 'google_photos_manifest.json'}[/underline]"
    )


@app.command("agents-run")
def cmd_agents_run(
    media_dir: Optional[Path] = typer.Argument(None, help="Directory containing raw videos (.MOV, .MP4) and photos (.HEIC, .JPG)"),
    prompt: str = typer.Option(..., "-p", "--prompt", help="Client story idea and must-have moments"),
    project_dir: Path = typer.Option(Path("workspace_project"), "-o", "--project-dir", help="Project output folder"),
    bgm: Optional[Path] = typer.Option(None, "--bgm", help="Optional background music file (.mp3, .wav)"),
    google_photos: bool = typer.Option(False, "--google-photos", help="Pull media directly from Google Photos via Browser Login (or configured OAuth) before editing"),
    auth_mode: str = typer.Option("auto", "--auth-mode", help="Google Photos auth mode: auto (browser login default) | browser | oauth"),
    from_downloads: Optional[str] = typer.Option(None, "--from-downloads", help="Import browser-downloaded Google Photos .zip/media file(s) directly"),
    downloads_dir: Optional[Path] = typer.Option(None, "--downloads-dir", help="Browser downloads folder to watch (defaults to ~/Downloads)"),
    watch_timeout: float = typer.Option(600.0, "--watch-timeout", help="Seconds to wait for browser download completion"),
    google_takeout: Optional[Path] = typer.Option(None, "--google-takeout", help="Pull media from a Google Takeout folder or .zip archive"),
    date_from: Optional[str] = typer.Option(None, "--date-from", help="Cloud sync inclusive start date (YYYY-MM-DD or ISO-8601)"),
    date_to: Optional[str] = typer.Option(None, "--date-to", help="Cloud sync inclusive end date (YYYY-MM-DD or ISO-8601)"),
    timezone_name: str = typer.Option("UTC", "--timezone", help="Timezone for --date-from / --date-to filtering"),
    max_cloud_photos: Optional[int] = typer.Option(None, "--max-cloud-photos", help="Optional cap on smart-downloaded cloud photos"),
    max_cloud_videos: Optional[int] = typer.Option(None, "--max-cloud-videos", help="Optional cap on smart-downloaded cloud videos"),
    download_all: bool = typer.Option(False, "--download-all", help="Bypass Two-Pass Smart Curation and download all cloud items in range"),
    target_duration: Optional[float] = typer.Option(None, "--target-duration", help="Optional target duration hint (s) for cloud budget & story length"),
    enable_agent: Optional[list[str]] = typer.Option(
        None,
        "--enable-agent",
        help="Enable optional plugin agent role_id (e.g. --enable-agent vfx_designer)",
    ),
    disable_agent: Optional[list[str]] = typer.Option(
        None,
        "--disable-agent",
        help="Disable specific agent role_id",
    ),
    story_arc_mode: Optional[str] = typer.Option(
        None,
        "--story-arc-mode",
        help="Storyline mode: chronological_cinema (default), hero_journey, thematic_essay, or high_energy_reel",
    ),
    title_theme: Optional[str] = typer.Option(
        None,
        "--title-theme",
        help="Cinema title theme: criterion_atelier, a24_minimal, apple_tv_glass, kinfolk_magazine, archival_35mm, or kinetic_travelogue",
    ),
    travel_map_mode: Optional[str] = typer.Option(
        None,
        "--travel-map-mode",
        help="Travel map mode: auto, flyover_interlude, picture_in_picture, hybrid, or off",
    ),
    travel_map_style: Optional[str] = typer.Option(
        None,
        "--travel-map-style",
        help="Cartography style: auto, midnight_vox_atlas, archival_expedition_parchment, aerial_relief_hybrid, or kinfolk_minimal_monograph",
    ),
    burn_in_subtitles: Optional[bool] = typer.Option(
        None,
        "--burn-in-subtitles/--no-burn-in-subtitles",
        help="Burn styled ASS subtitles into rendered MP4 in addition to SRT/ASS sidecars",
    ),
    interactive_intake: bool = typer.Option(
        False,
        "--interactive-intake",
        help="Interactively answer ClientIntakeAgent discovery questions in the terminal",
    ),
    width: int = typer.Option(3840, "--width", help="Output video width (default 3840 for 4K UHD)"),
    height: int = typer.Option(2160, "--height", help="Output video height (default 2160 for 2160p 4K UHD)"),
    fps: int = typer.Option(30, "--fps", help="Output frame rate"),
    hdr: bool = typer.Option(True, "--hdr/--sdr", help="Render in 10-bit HLG HDR (Rec.2020 / arib-std-b67)"),
    export_fcpxml_flag: bool = typer.Option(
        True,
        "--export-fcpxml/--no-export-fcpxml",
        help="Emit frame-accurate timeline.fcpxml (FCPXML 1.11 for DaVinci Resolve & Final Cut Pro)",
    ),
    render: bool = typer.Option(False, "--render/--no-render", help="Also render final_story.mp4 after Director sign-off"),
    use_antigravity: bool = typer.Option(True, "--antigravity/--no-antigravity", help="Use Google Antigravity SDK"),
) -> None:
    """Run the Coordinated Multi-Agent Post-Production Pipeline (Stage 00 -> Stage 65)."""
    if google_photos or google_takeout is not None or from_downloads is not None:
        on_browser, on_picker, on_log = _make_cli_cloud_callbacks(no_browser=False)
        sync_manifest = run_google_photos_sync(
            project_dir=project_dir,
            date_from=date_from,
            date_to=date_to,
            tz_name=timezone_name,
            target_duration_sec=float(target_duration or 155.0),
            max_photos=max_cloud_photos,
            max_videos=max_cloud_videos,
            download_all=download_all,
            auth_mode=auth_mode,
            from_downloads=from_downloads,
            takeout_source=google_takeout,
            downloads_dir=downloads_dir,
            watch_timeout_sec=watch_timeout,
            open_browser=True,
            browser_open_callback=on_browser,
            picker_ready_callback=on_picker,
            progress_callback=on_log,
        )
        media_dir = Path(sync_manifest.staged_media_dir)

    if media_dir is None:
        console.print("[bold red]Error:[/bold red] Provide MEDIA_DIR or pass --google-photos / --from-downloads / --google-takeout.")
        raise typer.Exit(code=2)

    console.print(
        Panel(
            f"[bold cyan]Running Modular Multi-Agent Video Editing Pipeline[/bold cyan]\n"
            f"Media Dir: {media_dir}\n"
            f"Project Dir: {project_dir}\n"
            f"Client Prompt: {prompt}"
        )
    )

    answers: dict[str, str] = {}
    if target_duration is not None:
        answers["target_duration_sec"] = str(int(round(target_duration)))
    if story_arc_mode:
        answers["story_arc_mode"] = story_arc_mode
    if title_theme:
        answers["title_theme"] = title_theme
    if travel_map_mode:
        answers["travel_map_mode"] = travel_map_mode
    if travel_map_style:
        answers["travel_map_style"] = travel_map_style
    if burn_in_subtitles is not None:
        answers["burn_in_subtitles"] = "true" if burn_in_subtitles else "false"
    if interactive_intake:
        questions = generate_intake_questions(media_dir, prompt)
        console.print("[bold yellow]Stage 0 Client Intake Interview:[/bold yellow]")
        for q in questions:
            console.print(f"\n[bold]{q.question}[/bold]")
            for opt in q.options:
                console.print(f"  * {opt}")
            ans = typer.prompt("Your answer", default=q.default_answer)
            answers[q.id] = ans

    extra_audio = [bgm] if bgm and bgm.exists() else None

    def _on_progress(role_id: str, msg: str, pct: int) -> None:
        console.print(f"  [{pct:3d}%] [{role_id}] {msg}")

    blackboard = run_multi_agent_pipeline(
        media_dir=media_dir,
        project_dir=project_dir,
        user_prompt=prompt,
        extra_audio_paths=extra_audio,
        interview_answers=answers,
        enabled_optional_roles=list(enable_agent or []),
        disabled_roles=list(disable_agent or []),
        width=width,
        height=height,
        fps=fps,
        hdr=hdr,
        use_antigravity=use_antigravity,
        render_video=render,
        export_fcpxml=export_fcpxml_flag,
        progress_callback=_on_progress,
    )

    tl = blackboard.timeline
    dr = blackboard.director_review
    dr_note = (
        f"Grade {dr.quality_grade} ({dr.overall_quality_score}/10, {dr.interventions_count} active directives)"
        if dr
        else "N/A"
    )
    console.print(
        f"[bold green][OK] Multi-Agent Pipeline Complete![/bold green]\n"
        f"  - Executed Roles:   {', '.join(a['role_id'] for a in blackboard.executed_agents)}\n"
        f"  - Creative Brief:   [underline]{project_dir / 'creative_brief.json'}[/underline]\n"
        f"  - DIT Manifest:     [underline]{project_dir / 'dit_manifest.json'}[/underline]\n"
        f"  - Media Index:      [underline]{project_dir / 'media_index.json'}[/underline]\n"
        f"  - Cast Manifest:    [underline]{project_dir / 'cast_manifest.json'}[/underline]\n"
        f"  - Beat Grid:        [underline]{project_dir / 'beat_grid.json'}[/underline]\n"
        f"  - Paper Edit:       [underline]{project_dir / 'paper_edit.json'}[/underline]\n"
        f"  - Color Manifest:   [underline]{project_dir / 'color_transform_manifest.json'}[/underline]\n"
        f"  - Audio Mix:        [underline]{project_dir / 'audio_mix_manifest.json'}[/underline]\n"
        f"  - Graphics Plan:    [underline]{project_dir / 'graphics_manifest.json'}[/underline]\n"
        f"  - Pacing Audit:     [underline]{project_dir / 'pacing_audit.json'}[/underline]\n"
        f"  - Visual QC:        [underline]{project_dir / 'visual_qc_report.json'}[/underline]\n"
        f"  - Director Review:  [underline]{project_dir / 'director_review.json'}[/underline] [{dr_note}]\n"
        f"  - Final Timeline:   [underline]{project_dir / 'timeline.yaml'}[/underline] "
        f"({len(tl.visual_track) if tl else 0} V1 cuts, {len(tl.broll_track) if tl else 0} V2 cutaways, "
        f"{len(tl.effects_track) if tl else 0} VFX items)\n"
        f"  - FCPXML 1.11:      [underline]{project_dir / 'timeline.fcpxml'}[/underline]"
    )


@app.command("analyze")
def cmd_analyze(
    media_dir: Path = typer.Argument(..., help="Directory containing raw videos (.MOV, .MP4) and photos (.HEIC, .JPG)"),
    project_dir: Path = typer.Option(Path("workspace_project"), "-o", "--project-dir", help="Project output folder"),
    bgm: Optional[Path] = typer.Option(None, "--bgm", help="Optional background music file (.mp3, .wav)"),
    whisper_model: str = typer.Option("small", "--whisper-model", help="faster-whisper model size (base, small, medium)"),
    language: Optional[str] = typer.Option(None, "--language", help="Optional ISO language code (e.g. zh, en) or auto"),
    hint: str = typer.Option("", "--hint", help="Optional story context hint for multimodal annotation"),
    use_antigravity: bool = typer.Option(True, "--antigravity/--no-antigravity", help="Use Google Antigravity SDK"),
) -> None:
    """Step 1: Extract video scenes, keyframes, timestamped dialogues, and photo metadata into annotations.yaml."""
    console.print(Panel(f"[bold cyan]Step 1: Analyzing media in[/bold cyan] {media_dir}"))
    extra_audio = [bgm] if bgm and bgm.exists() else None
    catalog = analyze_media_directory(
        media_dir=media_dir,
        project_dir=project_dir,
        whisper_model=whisper_model,
        language=language,
        use_antigravity=use_antigravity,
        user_hint=hint,
        extra_audio_paths=extra_audio,
    )
    ann_file = Path(project_dir).resolve() / "annotations.yaml"
    console.print(
        f"[bold green][OK] Step 1 Complete![/bold green] Analyzed [bold]{len(catalog.videos)}[/bold] videos, "
        f"[bold]{len(catalog.photos)}[/bold] photos, and [bold]{len(catalog.audio_files)}[/bold] audio files.\n"
        f"Saved catalog to: [underline]{ann_file}[/underline]"
    )


@app.command("plan")
def cmd_plan(
    project_dir: Path = typer.Argument(Path("workspace_project"), help="Project folder containing annotations.yaml"),
    prompt: str = typer.Option(..., "-p", "--prompt", help="Rough story idea for the final video"),
    duration: Optional[float] = typer.Option(None, "-d", "--duration", help="Optional target duration in seconds"),
    width: int = typer.Option(3840, "--width", help="Output video width (default 3840 for 4K UHD)"),
    height: int = typer.Option(2160, "--height", help="Output video height (default 2160 for 4K UHD)"),
    fps: int = typer.Option(30, "--fps", help="Output video frame rate"),
    hdr: bool = typer.Option(True, "--hdr/--sdr", help="Render in 10-bit HLG HDR (Rec.2020 / arib-std-b67)"),
    photo_mode: str = typer.Option("gallery_frame", "--photo-mode", help="gallery_frame or full_bleed"),
    use_antigravity: bool = typer.Option(True, "--antigravity/--no-antigravity", help="Use Google Antigravity SDK"),
) -> None:
    """Step 2: Generate a story-driven timeline.yaml and storyboard.md from annotations.yaml."""
    ann_path = Path(project_dir).resolve() / "annotations.yaml"
    if not ann_path.exists():
        console.print(f"[bold red]Error:[/bold red] {ann_path} not found. Run 'ai-editor analyze' first.")
        raise typer.Exit(code=1)

    console.print(Panel(f"[bold cyan]Step 2: Generating Story Plan & Timeline[/bold cyan]\nPrompt: {prompt}"))
    catalog = MediaCatalog.from_yaml(ann_path)
    timeline, tl_path, sb_path = generate_edit_plan(
        catalog=catalog,
        user_prompt=prompt,
        project_dir=project_dir,
        target_duration=duration,
        width=width,
        height=height,
        fps=fps,
        hdr=hdr,
        photo_mode=photo_mode,
        use_antigravity=use_antigravity,
    )
    nle_paths = export_all_nle_formats(timeline, project_dir)
    console.print(
        f"[bold green][OK] Step 2 Complete![/bold green] Total timeline duration: [bold]{timeline.total_duration:.1f}s[/bold] "
        f"({len(timeline.visual_track)} visual cuts, {len(timeline.title_track)} titles, "
        f"{len(timeline.subtitles.items)} dialogue captions).\n"
        f"  - Editable Timeline: [underline]{tl_path}[/underline]\n"
        f"  - Storyboard Markdown: [underline]{sb_path}[/underline]\n"
        f"  - NLE Exports: [underline]{nle_paths['fcpxml'].name}[/underline], "
        f"[underline]{nle_paths['edl'].name}[/underline], [underline]{nle_paths['otio'].name}[/underline]"
    )


@app.command("render")
def cmd_render(
    timeline_file: Path = typer.Argument(..., help="Path to timeline.yaml or project directory"),
    output_mp4: Optional[Path] = typer.Option(None, "-o", "--output", help="Output MP4 file path"),
) -> None:
    """Step 3: Render timeline.yaml into an MP4 video with Ken Burns/Gallery Frame, titles, and auto-ducked BGM."""
    tl_path = timeline_file / "timeline.yaml" if timeline_file.is_dir() else timeline_file
    tl_path = tl_path.resolve()
    if not tl_path.exists():
        console.print(f"[bold red]Error:[/bold red] {tl_path} not found.")
        raise typer.Exit(code=1)

    timeline = EditTimeline.from_yaml(tl_path)
    out_file = (output_mp4 or (tl_path.parent / "final_story.mp4")).resolve()

    console.print(
        Panel(
            f"[bold cyan]Step 3: Rendering Video[/bold cyan] ({timeline.settings.width}x{timeline.settings.height} "
            f"@ {timeline.settings.fps}fps, {timeline.total_duration:.1f}s)"
        )
    )

    def _prog(msg: str, frac: float) -> None:
        console.print(f"  [{int(frac * 100):3d}%] {msg}")

    outputs = render_timeline(timeline, out_file, progress_callback=_prog)
    nle_outputs = export_all_nle_formats(timeline, out_file.parent)
    console.print(
        f"[bold green][OK] Step 3 Render Complete![/bold green]\n"
        f"  - Final Video: [underline]{outputs['video']}[/underline]\n"
        f"  - Subtitles:   [underline]{outputs['srt']}[/underline] & [underline]{outputs['ass']}[/underline]\n"
        f"  - NLE Files:   [underline]{nle_outputs['fcpxml']}[/underline]"
    )


@app.command("export")
def cmd_export(
    timeline_file: Path = typer.Argument(..., help="Path to timeline.yaml or project directory"),
) -> None:
    """Export timeline.yaml to FCPXML (.fcpxml), CMX 3600 EDL (.edl), and OpenTimelineIO (.otio)."""
    tl_path = timeline_file / "timeline.yaml" if timeline_file.is_dir() else timeline_file
    timeline = EditTimeline.from_yaml(tl_path.resolve())
    paths = export_all_nle_formats(timeline, tl_path.resolve().parent)
    for fmt, p in paths.items():
        console.print(f"[green][OK][/green] Exported {fmt.upper()}: {p}")


@app.command("export-fcpxml")
def cmd_export_fcpxml(
    project_dir: Path = typer.Argument(..., help="Project directory containing timeline.yaml (or path to timeline.yaml)"),
    output_file: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Optional output .fcpxml path (defaults to <project_dir>/timeline.fcpxml)",
    ),
) -> None:
    """Export a frame-accurate FCPXML 1.11 timeline (timeline.fcpxml) for DaVinci Resolve and Final Cut Pro."""
    tl_path = (project_dir / "timeline.yaml") if project_dir.is_dir() else project_dir
    tl_path = tl_path.resolve()
    if not tl_path.exists():
        console.print(f"[bold red]Error:[/bold red] {tl_path} not found.")
        raise typer.Exit(code=1)

    timeline = EditTimeline.from_yaml(tl_path)
    target_out = (output_file or (tl_path.parent / "timeline.fcpxml")).resolve()
    written = export_fcpxml(timeline, target_out)
    console.print(f"[bold green][OK] Exported FCPXML 1.11:[/bold green] [underline]{written}[/underline]")


@app.command("docs-sync")
def cmd_docs_sync(
    sync_global: bool = typer.Option(
        True,
        "--sync-global/--no-sync-global",
        help="Also synchronize updated skill files to ~/.gemini/config/skills/ai-video-editor",
    ),
) -> None:
    """Introspect AgentRegistry and auto-update README.md and .agent/skills/ai-video-editor/ documentation."""
    result = sync_agent_documentation()
    installed_note = ""
    if sync_global:
        try:
            installed_dir = sync_skill_to_global()
            installed_note = f"\n  - Global Skill:    [underline]{installed_dir}[/underline]"
        except Exception:
            installed_note = ""
    console.print(
        f"[bold green][OK] Documentation synchronized with AgentRegistry ({result['agent_count']} agents)![/bold green]\n"
        f"  - README.md:       [underline]{result['readme_path']}[/underline] (updated={result['readme_updated']})\n"
        f"  - SKILL.md:        [underline]{result['skill_md_path']}[/underline] (updated={result['skill_updated']})\n"
        f"  - Playbooks:       {len(result['verified_reference_files'])} verified, "
        f"{len(result['created_reference_files'])} created{installed_note}"
    )


@app.command("run")
def cmd_run(
    media_dir: Path = typer.Argument(..., help="Directory containing raw videos and photos"),
    prompt: str = typer.Option(..., "-p", "--prompt", help="Rough story idea for the final video"),
    project_dir: Path = typer.Option(Path("workspace_project"), "-o", "--project-dir", help="Project output folder"),
    bgm: Optional[Path] = typer.Option(None, "--bgm", help="Optional background music file"),
    width: int = typer.Option(3840, "--width"),
    height: int = typer.Option(2160, "--height"),
    hdr: bool = typer.Option(True, "--hdr/--sdr"),
    use_antigravity: bool = typer.Option(True, "--antigravity/--no-antigravity"),
) -> None:
    """Run Steps 1 -> 2 -> 3 end-to-end (Analyze -> Plan -> Render)."""
    cmd_analyze(
        media_dir=media_dir,
        project_dir=project_dir,
        bgm=bgm,
        whisper_model="small",
        language=None,
        hint=prompt,
        use_antigravity=use_antigravity,
    )
    cmd_plan(
        project_dir=project_dir,
        prompt=prompt,
        duration=None,
        width=width,
        height=height,
        fps=30,
        hdr=hdr,
        photo_mode="gallery_frame",
        use_antigravity=use_antigravity,
    )
    cmd_render(timeline_file=project_dir / "timeline.yaml", output_mp4=project_dir / "final_story.mp4")


@app.command("serve")
def cmd_serve(
    project_dir: Path = typer.Option(Path("active_video_project"), "--project-dir", help="Default project folder"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Launch the Local Web Studio UI in your browser."""
    console.print(
        Panel(
            f"[bold green]Launching AI Photo & Video Story Studio[/bold green]\n"
            f"Open [bold underline]http://{host}:{port}[/bold underline] in your browser."
        )
    )
    web_app = create_app(default_project_dir=project_dir)
    uvicorn.run(web_app, host=host, port=port)


def sync_skill_to_global(dest_root: Optional[Path] = None, repo_root: Optional[Path] = None) -> Path:
    """Synchronize the workspace .agent/skills/ai-video-editor skill to the global Antigravity skills folder."""
    resolved_root = resolve_repo_root(repo_root)
    src_skill_dir = resolved_root / ".agent" / "skills" / "ai-video-editor"
    if not src_skill_dir.exists():
        raise FileNotFoundError(f"Workspace skill directory not found: {src_skill_dir}")

    target_base = dest_root or (Path.home() / ".gemini" / "config" / "skills")
    target_skill_dir = target_base / "ai-video-editor"
    target_base.mkdir(parents=True, exist_ok=True)
    if (
        not target_skill_dir.is_symlink()
        and target_skill_dir.exists()
        and target_skill_dir.resolve() == src_skill_dir.resolve()
    ):
        return target_skill_dir
    if target_skill_dir.is_symlink() or target_skill_dir.is_file():
        target_skill_dir.unlink()
    elif target_skill_dir.exists():
        shutil.rmtree(target_skill_dir)
    shutil.copytree(src_skill_dir, target_skill_dir)
    (target_skill_dir / ".workspace_root").write_text(str(resolved_root), encoding="utf-8")
    return target_skill_dir


@app.command("sync-skill")
def cmd_sync_skill(
    dest_dir: Optional[Path] = typer.Option(
        None,
        "--dest-dir",
        help="Optional custom global skills parent directory (defaults to ~/.gemini/config/skills)",
    ),
) -> None:
    """Install / synchronize the ai-video-editor skill globally for Antigravity (~/.gemini/config/skills/ai-video-editor)."""
    resolved_root = resolve_repo_root()
    installed_path = sync_skill_to_global(dest_dir, repo_root=resolved_root)
    console.print(
        f"[bold green][OK] Installed & synchronized Antigravity Skill globally![/bold green]\n"
        f"  - Workspace Skill: [underline]{resolved_root / '.agent/skills/ai-video-editor'}[/underline]\n"
        f"  - Global Skill:    [underline]{installed_path}[/underline]"
    )


@app.command("doctor")
def cmd_doctor(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON diagnostics",
    ),
) -> None:
    """Verify Python compatibility, FFmpeg/FFprobe binaries, GPU encoders, and all required dependencies."""
    report = collect_environment_diagnostics()
    if json_output:
        typer.echo(json.dumps(report, indent=2))
        if not report["healthy"]:
            raise typer.Exit(code=1)
        return

    py = report["python"]
    ff = report["ffmpeg"]
    fp = report["ffprobe"]
    hw = report["hardware"]
    gp = report.get("google_photos") or {}
    heif = report["heif"]
    ag = report["antigravity"]
    pkgs = report["packages"]
    failed_pkgs = [p for p in pkgs if not p["ok"]]

    py_badge = "[green]OK[/green]" if py["compatible"] else "[red]FAIL[/red]"
    ff_badge = "[green]OK[/green]" if ff["ok"] else "[red]MISSING[/red]"
    fp_badge = "[green]OK[/green]" if fp["ok"] else "[red]MISSING[/red]"
    heif_badge = "[green]OK[/green]" if heif["ok"] else "[red]FAIL[/red]"
    ag_badge = "[green]OK[/green]" if ag["sdk_ok"] else "[red]FAIL[/red]"
    gp_badge = (
        "[green]READY (OAuth + Browser)[/green]"
        if gp.get("ready")
        else "[green]READY (Browser Login)[/green]"
    )

    console.print(
        Panel(
            f"[bold cyan]video-editing-ai Environment Doctor[/bold cyan]\n"
            f"  - Python:    {py_badge} {py['version']} ({py['platform']} / {py['arch']}) — {py['executable']}"
            + (f"\n    [yellow]Note: {py['note']}[/yellow]" if py["note"] else "")
            + f"\n  - ffmpeg:    {ff_badge} {ff['version_line'] or ff['path'] or 'Not found on PATH'}"
            + f"\n  - ffprobe:   {fp_badge} {fp['path'] or 'Not found on PATH'}"
            + f"\n  - HEIF/HEIC: {heif_badge} {'pillow-heif registered' if heif['ok'] else heif['error']}"
            + f"\n  - Google Photos: {gp_badge} (secrets={gp.get('client_secrets_configured', False)}, token={gp.get('token_valid', False)})"
            + f"\n  - Antigravity: {ag_badge} sdk={ag['sdk_version'] or 'missing'}, "
            + f"skill={'OK' if ag['workspace_skill_ok'] else 'missing'}, "
            + f"rules={'OK' if ag['workspace_rules_ok'] else 'missing'}, "
            + f"global_skill={'OK' if ag['global_skill_ok'] else 'not synced (run ai-editor sync-skill)'}"
            + f"\n  - GPU/Accel: torch={hw['torch_device']} (mps={hw['has_mps']}), "
            + f"opencl={hw['has_amd_opencl']} ({hw['opencl_device_spec'] or 'default'}), "
            + f"discrete_gpu={hw['discrete_gpu_name'] or 'none'}"
            + f"\n  - Encoders:  HDR={hw['best_hdr_encoder']}, SDR={hw['best_sdr_encoder']}, "
            + f"videotoolbox_hw={hw['has_videotoolbox_hwaccel']}, scale_vt={hw['has_scale_vt']}, zscale={hw['has_zscale']}"
        )
    )
    console.print(f"[bold]Python Packages ({len(pkgs) - len(failed_pkgs)}/{len(pkgs)} OK):[/bold]")
    for pkg in pkgs:
        status = "[green]OK[/green]" if pkg["ok"] else "[red]FAIL[/red]"
        ver = pkg["version"] or "not installed"
        err = f" — [red]{pkg['error']}[/red]" if pkg["error"] else ""
        console.print(f"  - {status} {pkg['distribution']} ({pkg['import_name']}): {ver}{err}")

    if not report["healthy"]:
        console.print("[bold red][FAIL] Environment check found missing or incompatible components.[/bold red]")
        raise typer.Exit(code=1)
    console.print("[bold green][OK] Environment ready for Antigravity + 4K 10-bit HLG HDR analysis & rendering![/bold green]")


if __name__ == "__main__":
    app()
