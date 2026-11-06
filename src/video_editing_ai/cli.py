"""Command-Line Interface (ai-editor) for the AI Photo & Video Editing Tool."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
import typer
import uvicorn

from .analyzer.ag_annotator import analyze_media_directory
from .exporters.nle_export import export_all_nle_formats
from .models.schema import EditTimeline, MediaCatalog
from .planner.story_director import generate_edit_plan
from .renderer.ffmpeg_composer import render_timeline
from .web.app import create_app

app = typer.Typer(
    name="ai-editor",
    help="AI-powered photo and video analysis, story timeline planning, and open-source programmatic video renderer.",
    add_completion=False,
)
console = Console()


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


@app.command("run")
def cmd_run(
    media_dir: Path = typer.Argument(..., help="Directory containing raw videos and photos"),
    prompt: str = typer.Option(..., "-p", "--prompt", help="Rough story idea for the final video"),
    project_dir: Path = typer.Option(Path("workspace_project"), "-o", "--project-dir", help="Project output folder"),
    bgm: Optional[Path] = typer.Option(None, "--bgm", help="Optional background music file"),
    width: int = typer.Option(1920, "--width"),
    height: int = typer.Option(1080, "--height"),
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
        photo_mode="gallery_frame",
        use_antigravity=use_antigravity,
    )
    cmd_render(timeline_file=project_dir / "timeline.yaml", output_mp4=project_dir / "final_story.mp4")


@app.command("serve")
def cmd_serve(
    project_dir: Path = typer.Option(Path("gilroy_halloween_project"), "--project-dir", help="Default project folder"),
    host: str = typer.Option("127.0.0.1", "--host"),
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


if __name__ == "__main__":
    app()
