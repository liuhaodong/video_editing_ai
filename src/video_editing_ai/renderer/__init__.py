"""Step 3: Open-Source Programmatic Video Renderer (FFmpeg + Pillow)."""

from .ffmpeg_composer import render_timeline
from .ken_burns import build_gallery_frame_image, render_photo_clip
from .subtitles import export_ass_subtitles, export_srt_subtitles
from .titles import render_title_png

__all__ = [
    "build_gallery_frame_image",
    "export_ass_subtitles",
    "export_srt_subtitles",
    "render_photo_clip",
    "render_timeline",
    "render_title_png",
]
