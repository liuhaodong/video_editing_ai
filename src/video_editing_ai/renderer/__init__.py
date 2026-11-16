"""Step 3: Open-Source Programmatic Video Renderer (FFmpeg + Pillow)."""

from .ffmpeg_composer import render_timeline
from .ken_burns import build_gallery_frame_image, render_photo_clip
from .kinetic_titles import render_glass_mask_png, render_kinetic_title_mov
from .subtitles import export_ass_subtitles, export_srt_subtitles
from .titles import render_title_frame, render_title_png, sample_shot_accent_hex
from .travel_map import (
    CART_THEME_SPECS,
    haversine_km,
    infer_transport_mode,
    initial_bearing_deg,
    lookup_gazetteer_place,
    render_travel_map_clip,
    render_travel_map_frame,
    render_travel_map_showcase_sheet,
    resolve_journey_waypoints,
    resolve_travel_map_style,
)

__all__ = [
    "CART_THEME_SPECS",
    "build_gallery_frame_image",
    "export_ass_subtitles",
    "export_srt_subtitles",
    "haversine_km",
    "infer_transport_mode",
    "initial_bearing_deg",
    "lookup_gazetteer_place",
    "render_glass_mask_png",
    "render_kinetic_title_mov",
    "render_photo_clip",
    "render_timeline",
    "render_title_frame",
    "render_title_png",
    "render_travel_map_clip",
    "render_travel_map_frame",
    "render_travel_map_showcase_sheet",
    "resolve_journey_waypoints",
    "resolve_travel_map_style",
    "sample_shot_accent_hex",
]
