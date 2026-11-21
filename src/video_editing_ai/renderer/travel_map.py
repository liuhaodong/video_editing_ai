"""Studio-grade Cinema 2.5D Travel Map & Cartography Engine.

Renders multi-stop geographic route flyovers across 4 bespoke cinema styles
(midnight_vox_atlas, archival_expedition_parchment, aerial_relief_hybrid,
kinfolk_minimal_monograph) with Web Mercator projection, cached slippy-tile +
offline 2.5D hillshade topography, parabolic flight arcs, tangent-aligned
vector transport silhouettes, arrival hero photo callouts, virtual 2.5D camera
homography, synchronized elevation sparklines, and ~203-nit HDR clamping.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import math
import os
from pathlib import Path
import re
import ssl
import subprocess
import threading
from typing import Any, Optional
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFilter, ImageOps

from ..models.schema import (
    ChapterPlan,
    CulledSubClip,
    MapWaypointSpec,
    MediaCatalog,
    VALID_TRANSPORT_MODES,
    VALID_TRAVEL_MAP_STYLES,
)
from .titles import (
    HDR_DIFFUSE_WHITE_PEAK_RGB,
    HDR_MAX_SHADOW_ALPHA,
    _load_font_tier,
    clamp_rgba_for_hdr,
    draw_tracked_text,
    measure_tracked_text,
)

# ---------------------------------------------------------------------------
# 4 Art-Directed Cinema Cartography Themes
# ---------------------------------------------------------------------------

CART_THEME_SPECS: dict[str, dict[str, Any]] = {
    "midnight_vox_atlas": {
        "display_name": "Midnight Vox Atlas",
        "water_deep": (9, 15, 28),
        "water_shallow": (18, 36, 58),
        "bathymetry_rgb": (36, 76, 112),
        "land_low": (24, 34, 52),
        "land_mid": (36, 50, 74),
        "land_high": (64, 84, 114),
        "coast_rgb": (92, 168, 214),
        "contour_rgba": (72, 98, 134, 86),
        "graticule_rgba": (86, 122, 162, 62),
        "route_glow": (44, 194, 238),
        "route_core": (122, 232, 252),
        "route_casing": (8, 24, 44),
        "route_future": (132, 162, 198),
        "route_history": (86, 178, 214),
        "accent": (244, 182, 82),
        "pin_fill": (18, 26, 42),
        "pin_text": (244, 248, 252),
        "hud_fill": (10, 16, 28, 218),
        "hud_border": (110, 220, 246, 125),
        "hud_title": (244, 248, 252),
        "hud_sub": (176, 198, 222),
        "hud_metric": (122, 232, 252),
        "title_tier": "sans",
        "sub_tier": "mono",
        "light_paper": False,
        "neatline": False,
        "compass_color": (160, 198, 228),
        "horizon_mist": (12, 22, 38),
        "tile_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "labels_overlay_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        "hillshade_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        "hillshade_strength": 0.22,
        "tile_blend": 0.90,
    },
    "google_road_navigator": {
        "display_name": "Google-Style Road & Terrain Navigator",
        "water_deep": (162, 204, 240),
        "water_shallow": (182, 218, 246),
        "bathymetry_rgb": (146, 190, 230),
        "land_low": (236, 234, 226),
        "land_mid": (228, 224, 212),
        "land_high": (212, 206, 192),
        "coast_rgb": (118, 152, 188),
        "contour_rgba": (168, 160, 146, 52),
        "graticule_rgba": (132, 146, 164, 26),
        "route_glow": (26, 115, 232),
        "route_core": (66, 133, 244),
        "route_casing": (20, 56, 132),
        "route_future": (110, 126, 146),
        "route_history": (52, 168, 83),
        "accent": (234, 67, 53),
        "pin_fill": (255, 255, 255),
        "pin_text": (32, 33, 36),
        "hud_fill": (255, 255, 255, 238),
        "hud_border": (26, 115, 232, 175),
        "hud_title": (32, 33, 36),
        "hud_sub": (95, 99, 104),
        "hud_metric": (26, 115, 232),
        "title_tier": "sans",
        "sub_tier": "mono",
        "light_paper": True,
        "neatline": False,
        "compass_color": (60, 64, 67),
        "horizon_mist": (244, 246, 248),
        "tile_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "hillshade_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        "hillshade_strength": 0.28,
        "tile_blend": 0.97,
    },
    "archival_expedition_parchment": {
        "display_name": "Archival Expedition Parchment",
        "water_deep": (176, 182, 170),
        "water_shallow": (198, 202, 188),
        "bathymetry_rgb": (138, 138, 122),
        "land_low": (224, 213, 186),
        "land_mid": (210, 194, 158),
        "land_high": (182, 154, 114),
        "coast_rgb": (96, 76, 54),
        "contour_rgba": (128, 102, 72, 72),
        "graticule_rgba": (112, 88, 62, 30),
        "route_glow": (192, 68, 52),
        "route_core": (164, 42, 36),
        "route_casing": (78, 32, 24),
        "route_future": (118, 92, 66),
        "route_history": (146, 72, 56),
        "accent": (178, 126, 44),
        "pin_fill": (236, 228, 208),
        "pin_text": (42, 32, 22),
        "hud_fill": (234, 226, 206, 230),
        "hud_border": (112, 88, 58, 165),
        "hud_title": (38, 30, 22),
        "hud_sub": (90, 72, 52),
        "hud_metric": (164, 42, 36),
        "title_tier": "serif",
        "sub_tier": "mono",
        "light_paper": True,
        "neatline": True,
        "compass_color": (82, 64, 44),
        "horizon_mist": (218, 210, 190),
        "tile_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "hillshade_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        "hillshade_strength": 0.24,
        "tile_blend": 0.92,
    },
    "aerial_relief_hybrid": {
        "display_name": "Aerial Relief Hybrid",
        "water_deep": (12, 34, 48),
        "water_shallow": (26, 78, 96),
        "bathymetry_rgb": (48, 116, 130),
        "land_low": (56, 78, 58),
        "land_mid": (118, 108, 82),
        "land_high": (172, 162, 144),
        "coast_rgb": (146, 206, 202),
        "contour_rgba": (228, 212, 174, 42),
        "graticule_rgba": (196, 216, 220, 14),
        "route_glow": (244, 184, 74),
        "route_core": (250, 216, 120),
        "route_casing": (14, 22, 32),
        "route_future": (216, 226, 230),
        "route_history": (208, 166, 88),
        "accent": (246, 190, 80),
        "pin_fill": (20, 34, 42),
        "pin_text": (248, 246, 240),
        "hud_fill": (14, 24, 30, 218),
        "hud_border": (246, 190, 80, 140),
        "hud_title": (248, 246, 240),
        "hud_sub": (192, 210, 212),
        "hud_metric": (248, 204, 102),
        "title_tier": "sans",
        "sub_tier": "mono",
        "light_paper": False,
        "neatline": False,
        "compass_color": (212, 226, 224),
        "horizon_mist": (24, 40, 50),
        "tile_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "hillshade_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        "hillshade_strength": 0.30,
        "roads_overlay_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
        "labels_overlay_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        "tile_blend": 0.97,
    },
    "satellite_hybrid_4k": {
        "display_name": "Satellite Hybrid 4K",
        "water_deep": (8, 24, 38),
        "water_shallow": (18, 58, 78),
        "bathymetry_rgb": (38, 92, 114),
        "land_low": (44, 64, 48),
        "land_mid": (98, 90, 70),
        "land_high": (154, 146, 128),
        "coast_rgb": (132, 198, 212),
        "contour_rgba": (214, 202, 168, 36),
        "graticule_rgba": (180, 208, 224, 14),
        "route_glow": (56, 189, 248),
        "route_core": (255, 214, 102),
        "route_casing": (12, 18, 28),
        "route_future": (222, 232, 242),
        "route_history": (56, 189, 248),
        "accent": (245, 158, 11),
        "pin_fill": (14, 22, 34),
        "pin_text": (250, 250, 252),
        "hud_fill": (12, 18, 26, 222),
        "hud_border": (56, 189, 248, 155),
        "hud_title": (250, 252, 255),
        "hud_sub": (186, 204, 220),
        "hud_metric": (255, 214, 102),
        "title_tier": "sans",
        "sub_tier": "mono",
        "light_paper": False,
        "neatline": False,
        "compass_color": (216, 230, 238),
        "horizon_mist": (14, 24, 34),
        "tile_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "hillshade_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        "hillshade_strength": 0.32,
        "roads_overlay_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
        "labels_overlay_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        "tile_blend": 0.98,
    },
    "kinfolk_minimal_monograph": {
        "display_name": "Kinfolk Minimal Monograph",
        "water_deep": (200, 210, 216),
        "water_shallow": (216, 224, 229),
        "bathymetry_rgb": (174, 188, 196),
        "land_low": (238, 234, 226),
        "land_mid": (226, 220, 208),
        "land_high": (208, 200, 186),
        "coast_rgb": (138, 148, 154),
        "contour_rgba": (170, 164, 152, 62),
        "graticule_rgba": (152, 146, 138, 26),
        "route_glow": (196, 102, 74),
        "route_core": (178, 76, 52),
        "route_casing": (88, 42, 30),
        "route_future": (146, 138, 130),
        "route_history": (176, 112, 90),
        "accent": (178, 76, 52),
        "pin_fill": (244, 240, 234),
        "pin_text": (38, 34, 30),
        "hud_fill": (242, 238, 230, 228),
        "hud_border": (166, 156, 142, 145),
        "hud_title": (34, 32, 28),
        "hud_sub": (100, 94, 86),
        "hud_metric": (178, 76, 52),
        "title_tier": "serif",
        "sub_tier": "mono",
        "light_paper": True,
        "neatline": True,
        "compass_color": (108, 102, 94),
        "horizon_mist": (228, 226, 220),
        "tile_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "labels_overlay_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        "hillshade_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        "hillshade_strength": 0.20,
        "tile_blend": 0.56,
    },
}


def resolve_travel_map_style(
    requested_style: str = "auto",
    title_theme: str = "apple_tv_glass",
    prompt_text: str = "",
    transport_mode: str = "auto",
    leg_km: Optional[float] = None,
) -> str:
    """Intelligently resolve explicit, theme-matched, landscape-matched, or mode-adaptive cartography style."""
    req = (requested_style or "auto").strip().lower()
    if req in CART_THEME_SPECS:
        return req

    low = (prompt_text or "").lower()
    # 1. Explicit cartography style mentions in user prompt
    if any(k in low for k in ("satellite hybrid", "earth studio", "satellite 4k")):
        return "satellite_hybrid_4k"
    if any(k in low for k in ("google", "road navigator", "street map", "highway map", "driving map")):
        return "google_road_navigator"
    if any(k in low for k in ("parchment", "vintage", "archival", "expedition", "indiana", "sepia", "treasure")):
        return "archival_expedition_parchment"
    if any(k in low for k in ("aerial", "satellite", "terrain", "alpine", "mountain", "relief", "topograph")):
        return "aerial_relief_hybrid"
    if any(k in low for k in ("kinfolk", "minimal", "monograph", "alabaster", "magazine")):
        return "kinfolk_minimal_monograph"
    if any(k in low for k in ("vox", "midnight", "neon", "cyan", "dark atlas")):
        return "midnight_vox_atlas"

    # 2. Flights & coastal/ocean crossings shine best on high-res orbital satellite imagery
    t_mode = (transport_mode or "auto").strip().lower()
    if t_mode in ("flight", "boat"):
        return "satellite_hybrid_4k"

    # 3. Context-intelligent destination, mood, and film-stock style selection (avoid blind midnight_vox_atlas default!)
    heritage_cues = (
        "paris", "versailles", "chateau", "château", "palace", "museum", "cathedral", "castle",
        "rome", "florence", "venice", "prague", "vienna", "kyoto", "temple", "shrine",
        "historic", "heritage", "monument", "kodak", "classic", "europe", "louvre", "triomphe",
    )
    nature_cues = (
        "national park", "yosemite", "big sur", "coast", "coastal", "canyon", "zion", "bryce",
        "banff", "alps", "hiking", "trail", "lake", "island", "forest", "glacier", "desert",
        "road trip", "scenic drive", "pacific", "highway 1", "valley", "beach",
    )
    editorial_cues = (
        "bistro", "cafe", "café", "garden", "botanical", "winery", "napa", "carmel",
        "gallery", "boutique", "fuji", "eterna", "pastel", "soft", "artisan", "culinary",
    )
    modern_night_cues = (
        "shinjuku", "shibuya", "seoul", "manhattan", "hong kong", "singapore", "las vegas",
        "night", "dusk", "twilight", "skyline", "cyber", "modern", "tech", "metro", "subway",
    )
    day_road_cues = ("freeway", "interstate", "commute", "suburb", "street navigation", "highway")

    if any(k in low for k in heritage_cues):
        return "archival_expedition_parchment"
    if any(k in low for k in nature_cues):
        return "aerial_relief_hybrid"
    if any(k in low for k in editorial_cues):
        return "kinfolk_minimal_monograph"
    if any(k in low for k in modern_night_cues):
        return "midnight_vox_atlas"
    if any(k in low for k in day_road_cues):
        return "google_road_navigator"

    # 4. Respect explicit cinema title themes before mode fallback
    theme_map = {
        "criterion_atelier": "archival_expedition_parchment",
        "archival_35mm": "archival_expedition_parchment",
        "kinfolk_magazine": "kinfolk_minimal_monograph",
        "kinetic_travelogue": "aerial_relief_hybrid",
        "a24_minimal": "midnight_vox_atlas",
    }
    if title_theme in theme_map:
        return theme_map[title_theme]

    # 5. Long overland drives (> 80 km) favor terrain relief; neutral urban hops use sleek dark Vox atlas
    if t_mode == "drive" and leg_km is not None and leg_km >= 80.0:
        return "aerial_relief_hybrid"
    if t_mode in ("drive", "train", "walk"):
        return "midnight_vox_atlas"

    return "archival_expedition_parchment" if title_theme != "apple_tv_glass" else "midnight_vox_atlas"


# ---------------------------------------------------------------------------
# Offline Global Gazetteer (~210 Cities, Parks, Islands & Landmarks, ASCII keys)
# ---------------------------------------------------------------------------

# Format: lowercase_key -> (lat, lon, alt_m, display_name)
OFFLINE_GAZETTEER: dict[str, tuple[float, float, float, str]] = {
    # North America
    "san francisco": (37.7749, -122.4194, 16.0, "San Francisco"),
    "golden gate bridge": (37.8199, -122.4783, 67.0, "Golden Gate"),
    "golden gate": (37.8199, -122.4783, 67.0, "Golden Gate"),
    "sausalito": (37.8591, -122.4853, 4.0, "Sausalito"),
    "napa valley": (38.2975, -122.2869, 24.0, "Napa Valley"),
    "sonoma": (38.2919, -122.4580, 26.0, "Sonoma"),
    "yosemite": (37.8651, -119.5383, 1210.0, "Yosemite Valley"),
    "yosemite valley": (37.7456, -119.5936, 1218.0, "Yosemite Valley"),
    "lake tahoe": (39.0968, -120.0324, 1897.0, "Lake Tahoe"),
    "monterey": (36.6002, -121.8947, 8.0, "Monterey"),
    "carmel": (36.5552, -121.9233, 68.0, "Carmel-by-the-Sea"),
    "big sur": (36.2704, -121.8081, 145.0, "Big Sur"),
    "santa barbara": (34.4208, -119.6982, 15.0, "Santa Barbara"),
    "los angeles": (34.0522, -118.2437, 89.0, "Los Angeles"),
    "santa monica": (34.0195, -118.4912, 32.0, "Santa Monica"),
    "malibu": (34.0259, -118.7798, 25.0, "Malibu"),
    "san diego": (32.7157, -117.1611, 20.0, "San Diego"),
    "joshua tree": (33.8734, -115.9010, 900.0, "Joshua Tree"),
    "palm springs": (33.8303, -116.5453, 146.0, "Palm Springs"),
    "gilroy": (37.0058, -121.5683, 61.0, "Gilroy"),
    "silicon valley": (37.3875, -122.0575, 22.0, "Silicon Valley"),
    "stanford": (37.4275, -122.1697, 29.0, "Stanford"),
    "half moon bay": (37.4636, -122.4286, 21.0, "Half Moon Bay"),
    "sedona": (34.8697, -111.7610, 1319.0, "Sedona"),
    "grand canyon": (36.1069, -112.1129, 2070.0, "Grand Canyon"),
    "page arizona": (36.9147, -111.4558, 1255.0, "Page / Antelope"),
    "zion": (37.2982, -113.0263, 1210.0, "Zion Canyon"),
    "bryce canyon": (37.5930, -112.1871, 2400.0, "Bryce Canyon"),
    "moab": (38.5733, -109.5498, 1227.0, "Moab / Arches"),
    "las vegas": (36.1699, -115.1398, 610.0, "Las Vegas"),
    "salt lake city": (40.7608, -111.8910, 1288.0, "Salt Lake City"),
    "yellowstone": (44.4280, -110.5885, 2400.0, "Yellowstone"),
    "grand teton": (43.7904, -110.6818, 2080.0, "Grand Teton"),
    "jackson hole": (43.4799, -110.7624, 1901.0, "Jackson Hole"),
    "denver": (39.7392, -104.9903, 1609.0, "Denver"),
    "aspen": (39.1911, -106.8175, 2438.0, "Aspen"),
    "seattle": (47.6062, -122.3321, 56.0, "Seattle"),
    "mount rainier": (46.8523, -121.7603, 1650.0, "Mount Rainier"),
    "olympic peninsula": (47.8021, -123.6044, 620.0, "Olympic Peninsula"),
    "portland": (45.5152, -122.6784, 15.0, "Portland"),
    "crater lake": (42.8684, -122.1685, 1883.0, "Crater Lake"),
    "vancouver": (49.2827, -123.1207, 18.0, "Vancouver"),
    "whistler": (50.1163, -122.9574, 670.0, "Whistler"),
    "banff": (51.1784, -115.5708, 1383.0, "Banff"),
    "lake louise": (51.4254, -116.1773, 1750.0, "Lake Louise"),
    "jasper": (52.8737, -118.0814, 1062.0, "Jasper"),
    "anchorage": (61.2181, -149.9003, 31.0, "Anchorage"),
    "honolulu": (21.3069, -157.8583, 6.0, "Honolulu"),
    "waikiki": (21.2793, -157.8292, 4.0, "Waikiki"),
    "maui": (20.7984, -156.3319, 120.0, "Maui"),
    "haleakala": (20.7097, -156.2533, 3055.0, "Haleakala"),
    "kauai": (22.0964, -159.5261, 85.0, "Kauai"),
    "new york": (40.7128, -74.0060, 10.0, "New York"),
    "manhattan": (40.7831, -73.9712, 18.0, "Manhattan"),
    "brooklyn": (40.6782, -73.9442, 14.0, "Brooklyn"),
    "boston": (42.3601, -71.0589, 14.0, "Boston"),
    "washington": (38.9072, -77.0369, 12.0, "Washington DC"),
    "chicago": (41.8781, -87.6298, 182.0, "Chicago"),
    "miami": (25.7617, -80.1918, 2.0, "Miami"),
    "key west": (24.5551, -81.7800, 3.0, "Key West"),
    "new orleans": (29.9511, -90.0715, 2.0, "New Orleans"),
    "austin": (30.2672, -97.7431, 149.0, "Austin"),
    "toronto": (43.6532, -79.3832, 76.0, "Toronto"),
    "montreal": (45.5017, -73.5673, 36.0, "Montreal"),
    "quebec": (46.8139, -71.2080, 98.0, "Quebec City"),
    "mexico city": (19.4326, -99.1332, 2240.0, "Mexico City"),
    "oaxaca": (17.0732, -96.7266, 1555.0, "Oaxaca"),
    "cancun": (21.1619, -86.8515, 8.0, "Cancun"),
    # South America
    "cusco": (-13.5319, -71.9675, 3399.0, "Cusco"),
    "machu picchu": (-13.1631, -72.5450, 2430.0, "Machu Picchu"),
    "lima": (-12.0464, -77.0428, 154.0, "Lima"),
    "patagonia": (-50.9423, -73.4068, 220.0, "Patagonia"),
    "ushuaia": (-54.8019, -68.3030, 23.0, "Ushuaia"),
    "santiago": (-33.4489, -70.6693, 570.0, "Santiago"),
    "buenos aires": (-34.6037, -58.3816, 25.0, "Buenos Aires"),
    "rio de janeiro": (-22.9068, -43.1729, 11.0, "Rio de Janeiro"),
    # Europe
    "london": (51.5072, -0.1276, 11.0, "London"),
    "oxford": (51.7520, -1.2577, 62.0, "Oxford"),
    "cotswolds": (51.8330, -1.8433, 185.0, "Cotswolds"),
    "edinburgh": (55.9533, -3.1883, 47.0, "Edinburgh"),
    "isle of skye": (57.2736, -6.2155, 88.0, "Isle of Skye"),
    "dublin": (53.3498, -6.2603, 20.0, "Dublin"),
    "galway": (53.2707, -9.0568, 14.0, "Galway"),
    "paris": (48.8566, 2.3522, 35.0, "Paris"),
    "louvre": (48.8606, 2.3376, 36.0, "Louvre Pyramid"),
    "versailles": (48.8049, 2.1204, 132.0, "Versailles"),
    "eiffel tower": (48.8584, 2.2945, 33.0, "Champ de Mars"),
    "montmartre": (48.8867, 2.3431, 130.0, "Montmartre"),
    "nice": (43.7102, 7.2620, 10.0, "Nice"),
    "cannes": (43.5528, 7.0174, 8.0, "Cannes"),
    "chamonix": (45.9237, 6.8694, 1035.0, "Chamonix"),
    "lyon": (45.7640, 4.8357, 173.0, "Lyon"),
    "bordeaux": (44.8378, -0.5792, 16.0, "Bordeaux"),
    "amsterdam": (52.3676, 4.9041, 2.0, "Amsterdam"),
    "bruges": (51.2093, 3.2247, 8.0, "Bruges"),
    "brussels": (50.8503, 4.3517, 28.0, "Brussels"),
    "zurich": (47.3769, 8.5417, 408.0, "Zurich"),
    "lucerne": (47.0502, 8.3093, 436.0, "Lucerne"),
    "interlaken": (46.6863, 7.8632, 568.0, "Interlaken"),
    "grindelwald": (46.6242, 8.0414, 1034.0, "Grindelwald"),
    "zermatt": (46.0207, 7.7491, 1608.0, "Zermatt"),
    "st moritz": (46.4908, 9.8355, 1822.0, "St. Moritz"),
    "geneva": (46.2044, 6.1432, 375.0, "Geneva"),
    "munich": (48.1351, 11.5820, 519.0, "Munich"),
    "berlin": (52.5200, 13.4050, 34.0, "Berlin"),
    "vienna": (48.2082, 16.3738, 171.0, "Vienna"),
    "salzburg": (47.8095, 13.0550, 424.0, "Salzburg"),
    "hallstatt": (47.5622, 13.6493, 511.0, "Hallstatt"),
    "prague": (50.0755, 14.4378, 235.0, "Prague"),
    "rome": (41.9028, 12.4964, 21.0, "Rome"),
    "florence": (43.7696, 11.2558, 50.0, "Florence"),
    "venice": (45.4408, 12.3155, 2.0, "Venice"),
    "milan": (45.4642, 9.1900, 120.0, "Milan"),
    "como": (45.8081, 9.0852, 201.0, "Lake Como"),
    "dolomites": (46.5405, 11.8825, 1560.0, "Dolomites"),
    "amalfi": (40.6340, 14.6027, 42.0, "Amalfi Coast"),
    "positano": (40.6281, 14.4850, 38.0, "Positano"),
    "naples": (40.8518, 14.2681, 17.0, "Naples"),
    "tuscany": (43.3188, 11.3308, 322.0, "Tuscany"),
    "barcelona": (41.3874, 2.1686, 12.0, "Barcelona"),
    "madrid": (40.4168, -3.7038, 667.0, "Madrid"),
    "seville": (37.3891, -5.9845, 7.0, "Seville"),
    "granada": (37.1773, -3.5986, 738.0, "Granada"),
    "lisbon": (38.7223, -9.1393, 35.0, "Lisbon"),
    "porto": (41.1579, -8.6291, 75.0, "Porto"),
    "sintra": (38.8029, -9.3817, 210.0, "Sintra"),
    "athens": (37.9838, 23.7275, 70.0, "Athens"),
    "santorini": (36.3932, 25.4615, 220.0, "Santorini"),
    "mykonos": (37.4467, 25.3289, 32.0, "Mykonos"),
    "dubrovnik": (42.6507, 18.0944, 28.0, "Dubrovnik"),
    "reykjavik": (64.1466, -21.9426, 14.0, "Reykjavik"),
    "tromso": (69.6492, 18.9553, 18.0, "Tromso"),
    "oslo": (59.9139, 10.7522, 23.0, "Oslo"),
    "bergen": (60.3913, 5.3221, 15.0, "Bergen"),
    "stockholm": (59.3293, 18.0686, 28.0, "Stockholm"),
    "copenhagen": (55.6761, 12.5683, 14.0, "Copenhagen"),
    "istanbul": (41.0082, 28.9784, 39.0, "Istanbul"),
    "cappadocia": (38.6431, 34.8289, 1070.0, "Cappadocia"),
    # Asia & Middle East
    "tokyo": (35.6762, 139.6503, 40.0, "Tokyo"),
    "shinjuku": (35.6938, 139.7034, 42.0, "Shinjuku"),
    "shibuya": (35.6595, 139.7004, 28.0, "Shibuya"),
    "kamakura": (35.3192, 139.5467, 18.0, "Kamakura"),
    "hakone": (35.2324, 139.1069, 725.0, "Hakone"),
    "mount fuji": (35.3606, 138.7274, 2305.0, "Mount Fuji"),
    "kawaguchiko": (35.5162, 138.7518, 830.0, "Kawaguchiko"),
    "nagoya": (35.1815, 136.9066, 18.0, "Nagoya"),
    "takayama": (36.1461, 137.2522, 573.0, "Takayama"),
    "kanazawa": (36.5613, 136.6562, 26.0, "Kanazawa"),
    "kyoto": (35.0116, 135.7681, 50.0, "Kyoto"),
    "arashiyama": (35.0094, 135.6668, 84.0, "Arashiyama"),
    "nara": (34.6851, 135.8048, 86.0, "Nara"),
    "osaka": (34.6937, 135.5023, 20.0, "Osaka"),
    "hiroshima": (34.3853, 132.4553, 14.0, "Hiroshima"),
    "miyajima": (34.2960, 132.3198, 22.0, "Miyajima"),
    "sapporo": (43.0618, 141.3545, 29.0, "Sapporo"),
    "otaru": (43.1907, 140.9947, 15.0, "Otaru"),
    "okinawa": (26.2124, 127.6809, 25.0, "Okinawa"),
    "seoul": (37.5665, 126.9780, 38.0, "Seoul"),
    "busan": (35.1796, 129.0756, 22.0, "Busan"),
    "jeju": (33.4996, 126.5312, 48.0, "Jeju Island"),
    "taipei": (25.0330, 121.5654, 15.0, "Taipei"),
    "jiufen": (25.1099, 121.8452, 295.0, "Jiufen"),
    "taroko": (24.1583, 121.6226, 480.0, "Taroko Gorge"),
    "hong kong": (22.3193, 114.1694, 32.0, "Hong Kong"),
    "macau": (22.1987, 113.5439, 18.0, "Macau"),
    "beijing": (39.9042, 116.4074, 44.0, "Beijing"),
    "shanghai": (31.2304, 121.4737, 12.0, "Shanghai"),
    "hangzhou": (30.2741, 120.1551, 19.0, "Hangzhou"),
    "suzhou": (31.2989, 120.5853, 8.0, "Suzhou"),
    "chengdu": (30.5728, 104.0668, 500.0, "Chengdu"),
    "chongqing": (29.5630, 106.5516, 244.0, "Chongqing"),
    "xi an": (34.3416, 108.9398, 405.0, "Xi'an"),
    "guilin": (25.2736, 110.2902, 153.0, "Guilin"),
    "lijiang": (26.8721, 100.2299, 2418.0, "Lijiang"),
    "singapore": (1.3521, 103.8198, 15.0, "Singapore"),
    "bangkok": (13.7563, 100.5018, 2.0, "Bangkok"),
    "chiang mai": (18.7883, 98.9853, 310.0, "Chiang Mai"),
    "phuket": (7.8804, 98.3923, 18.0, "Phuket"),
    "hanoi": (21.0278, 105.8342, 16.0, "Hanoi"),
    "ha long": (20.9101, 107.1839, 8.0, "Ha Long Bay"),
    "ho chi minh": (10.8231, 106.6297, 19.0, "Ho Chi Minh"),
    "bali": (-8.4095, 115.1889, 120.0, "Bali"),
    "ubud": (-8.5069, 115.2625, 210.0, "Ubud"),
    "kathmandu": (27.7172, 85.3240, 1350.0, "Kathmandu"),
    "dubai": (25.2048, 55.2708, 8.0, "Dubai"),
    "petra": (30.3285, 35.4444, 810.0, "Petra"),
    # Africa & Oceania
    "cairo": (30.0444, 31.2357, 23.0, "Cairo"),
    "giza": (29.9792, 31.1342, 65.0, "Giza"),
    "luxor": (25.6872, 32.6396, 78.0, "Luxor"),
    "marrakech": (31.6295, -7.9811, 466.0, "Marrakech"),
    "cape town": (-33.9249, 18.4241, 25.0, "Cape Town"),
    "serengeti": (-2.3333, 34.8333, 1450.0, "Serengeti"),
    "sydney": (-33.8688, 151.2093, 19.0, "Sydney"),
    "melbourne": (-37.8136, 144.9631, 31.0, "Melbourne"),
    "cairns": (-16.9186, 145.7781, 6.0, "Cairns"),
    "auckland": (-36.8509, 174.7645, 26.0, "Auckland"),
    "queenstown": (-45.0312, 168.6626, 310.0, "Queenstown"),
    "milford sound": (-44.6716, 167.9256, 12.0, "Milford Sound"),
}

_GAZETTEER_KEYS_SORTED: tuple[str, ...] = tuple(
    sorted(OFFLINE_GAZETTEER.keys(), key=lambda k: (-len(k), k))
)

_NON_GEO_WORDS: set[str] = {
    "park", "garden", "gardens", "museum", "gallery", "chapter", "act",
    "arrival", "departure", "finale", "story", "home", "stage", "main",
    "morning", "afternoon", "twilight", "night", "golden", "moment", "moments",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 coordinates."""
    r_earth = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    c = 2.0 * math.atan2(math.sqrt(max(0.0, a)), math.sqrt(max(0.0, 1.0 - a)))
    return r_earth * c


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing (0..360 deg clockwise from North)."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    brg = math.degrees(math.atan2(y, x))
    return (brg + 360.0) % 360.0


def lookup_gazetteer_place(text: str) -> Optional[tuple[float, float, float, str]]:
    """Resolve the first unambiguous geographic destination mentioned in text."""
    raw = re.sub(r"[_/\-]+", " ", (text or "").strip().lower())
    if not raw or raw in _NON_GEO_WORDS:
        return None
    for key in _GAZETTEER_KEYS_SORTED:
        if re.search(rf"\b{re.escape(key)}\b", raw):
            return OFFLINE_GAZETTEER[key]
    return None


def find_gazetteer_sequence(text: str) -> list[tuple[float, float, float, str]]:
    """Extract ordered, non-overlapping geographic destinations mentioned in prompt text."""
    raw = re.sub(r"[_/\-]+", " ", (text or "").strip().lower())
    if not raw:
        return []
    hits: list[tuple[int, int, tuple[float, float, float, str]]] = []
    occupied: list[tuple[int, int]] = []
    for key in _GAZETTEER_KEYS_SORTED:
        for m in re.finditer(rf"\b{re.escape(key)}\b", raw):
            s, e = m.span()
            if any(not (e <= os_ or s >= oe_) for os_, oe_ in occupied):
                continue
            occupied.append((s, e))
            hits.append((s, e, OFFLINE_GAZETTEER[key]))
    hits.sort(key=lambda r: r[0])
    out: list[tuple[float, float, float, str]] = []
    seen_labels: set[str] = set()
    for _, _, geo in hits:
        if geo[3] not in seen_labels:
            seen_labels.add(geo[3])
            out.append(geo)
    return out


def infer_transport_mode(
    distance_km: float,
    origin_label: str = "",
    dest_label: str = "",
    hint: str = "auto",
) -> str:
    """Infer transit vehicle icon ('flight', 'train', 'drive', 'boat', 'walk')."""
    clean_hint = (hint or "auto").strip().lower()
    if clean_hint in VALID_TRANSPORT_MODES and clean_hint != "auto":
        return clean_hint
    blob = f"{origin_label} {dest_label}".lower()
    if any(k in blob for k in ("shinkansen", "rail", "train", "station", "eurostar", "tgv", "hakone", "kyoto")):
        if 35.0 <= distance_km <= 900.0:
            return "train"
    # Require explicit marine transit cues ("ferry", "island", "cruise", ...) rather than landing near a bay/harbor by car
    orig_low = (origin_label or "").lower()
    dest_low = (dest_label or "").lower()
    explicit_marine = any(k in blob for k in ("ferry", "island", "cruise", "catamaran", "santorini", "miyajima", "venice", "como", "fiord", "catalina", "archipelago"))
    both_coastal = any(k in orig_low for k in ("harbor", "harbour", "bay", "pier", "marina")) and any(k in dest_low for k in ("harbor", "harbour", "bay", "pier", "marina"))
    if (explicit_marine or both_coastal) and distance_km <= 350.0:
        return "boat"
    if distance_km >= 450.0:
        return "flight"
    if distance_km >= 15.0:
        return "drive"
    return "walk"


def resolve_journey_waypoints(
    chapters: list[ChapterPlan],
    subclip_by_id: Optional[dict[str, CulledSubClip]] = None,
    catalog: Optional[MediaCatalog] = None,
    user_prompt: str = "",
    force_synthetic_if_missing: bool = False,
) -> list[MapWaypointSpec]:
    """
    Build deduplicated, chronologically ordered journey waypoints across chapters.

    In default 'auto' mode (`force_synthetic_if_missing=False`), returns `< 2` waypoints
    if footage lacks real EXIF GPS stops or explicit named destinations in OFFLINE_GAZETTEER,
    preserving deterministic non-map tests.
    """
    sc_map = subclip_by_id or {}
    asset_gps: dict[str, tuple[Optional[float], Optional[float], Optional[float]]] = {}
    if catalog is not None:
        for a in (*catalog.videos, *catalog.photos):
            if a.gps_lat is not None and a.gps_lon is not None:
                asset_gps[a.id] = (a.gps_lat, a.gps_lon, getattr(a, "gps_alt", None))

    prompt_seq = find_gazetteer_sequence(user_prompt)
    raw_stops: list[MapWaypointSpec] = []

    for c_idx, chap in enumerate(chapters):
        lat = chap.waypoint_lat
        lon = chap.waypoint_lon
        alt = chap.waypoint_alt
        src_kind = "exif_gps"

        if lat is None or lon is None:
            lats: list[float] = []
            lons: list[float] = []
            alts: list[float] = []
            for sid in chap.selected_subclip_ids:
                sc = sc_map.get(sid)
                if sc is None:
                    continue
                if sc.gps_lat is not None and sc.gps_lon is not None:
                    lats.append(float(sc.gps_lat))
                    lons.append(float(sc.gps_lon))
                    if getattr(sc, "gps_alt", None) is not None:
                        alts.append(float(sc.gps_alt))
                elif sc.source_asset_id in asset_gps:
                    alat, alon, aalt = asset_gps[sc.source_asset_id]
                    if alat is not None and alon is not None:
                        lats.append(float(alat))
                        lons.append(float(alon))
                        if aalt is not None:
                            alts.append(float(aalt))
            if lats and lons:
                lat = float(np.median(lats))
                lon = float(np.median(lons))
                alt = float(np.median(alts)) if alts else 0.0
                src_kind = "exif_gps"

        clean_loc = (chap.location_tag or "").strip()
        if clean_loc.lower() in _NON_GEO_WORDS or clean_loc.lower() in ("featured highlight", "overview", "timeline"):
            clean_loc = ""
        clean_title = re.sub(r"^(?:Chapter|Act)\s*\d+\s*[:.\-\u00b7]*\s*", "", chap.title, flags=re.IGNORECASE).strip()
        label = clean_loc or clean_title or f"Stop {c_idx + 1}"
        if (lat is None or lon is None):
            geo = (
                lookup_gazetteer_place(chap.location_tag)
                or lookup_gazetteer_place(chap.title)
                or lookup_gazetteer_place(chap.subtitle)
            )
            if geo is not None:
                lat, lon, alt, g_label = geo
                if not clean_loc:
                    label = g_label
                src_kind = "gazetteer"

        if lat is not None and lon is not None:
            raw_stops.append(
                MapWaypointSpec(
                    waypoint_id=f"wp_{c_idx + 1:02d}",
                    chapter_id=chap.chapter_id,
                    label=label[:28],
                    subtitle=(chap.subtitle or "")[:44],
                    lat=round(float(lat), 5),
                    lon=round(float(lon), 5),
                    alt_m=round(float(alt or 0.0), 1),
                    source=src_kind,  # type: ignore[arg-type]
                )
            )

    # If user explicitly named a multi-stop itinerary in the prompt and chapters had < 2 real stops
    if len(raw_stops) < 2 and len(prompt_seq) >= 2:
        raw_stops = []
        for idx_p, (plat, plon, palt, plabel) in enumerate(prompt_seq):
            ch_id = chapters[idx_p].chapter_id if idx_p < len(chapters) else f"ch_{idx_p + 1:02d}"
            raw_stops.append(
                MapWaypointSpec(
                    waypoint_id=f"wp_{idx_p + 1:02d}",
                    chapter_id=ch_id,
                    label=plabel[:28],
                    subtitle="",
                    lat=round(plat, 5),
                    lon=round(plon, 5),
                    alt_m=round(palt, 1),
                    source="gazetteer",
                )
            )

    # Deduplicate stops that sit within 0.35 km of previous stop (single-venue shoots stay < 2 stops)
    deduped: list[MapWaypointSpec] = []
    for wp in raw_stops:
        if not deduped:
            deduped.append(wp)
            continue
        prev = deduped[-1]
        if haversine_km(prev.lat, prev.lon, wp.lat, wp.lon) >= 0.35:
            deduped.append(wp)

    if len(deduped) >= 2:
        for i, wp in enumerate(deduped, start=1):
            wp.waypoint_id = f"wp_{i:02d}"
        return deduped

    if not force_synthetic_if_missing:
        return deduped

    # Explicit map mode requested on untagged test footage: build plausible scenic regional itinerary
    seed_pts = [
        (37.7749, -122.4194, 24.0, "Coastal Departure"),
        (36.2704, -121.8081, 148.0, "Headland Overlook"),
        (37.7456, -119.5936, 1218.0, "Alpine Sanctuary"),
        (38.2975, -122.2869, 64.0, "Golden Ridge Finale"),
    ]
    n_target = max(2, min(4, len(chapters) if len(chapters) >= 2 else 3))
    out: list[MapWaypointSpec] = []
    for i in range(n_target):
        lat, lon, alt, def_lbl = seed_pts[i % len(seed_pts)]
        ch_id = chapters[i].chapter_id if i < len(chapters) else f"ch_{i + 1:02d}"
        ch_title = (
            re.sub(r"^(?:Chapter|Act)\s*\d+\s*[:.\-\u00b7]*\s*", "", chapters[i].title, flags=re.IGNORECASE).strip()
            if i < len(chapters)
            else ""
        )
        out.append(
            MapWaypointSpec(
                waypoint_id=f"wp_{i + 1:02d}",
                chapter_id=ch_id,
                label=(ch_title or def_lbl)[:28],
                subtitle="",
                lat=lat,
                lon=lon,
                alt_m=alt,
                source="synthetic",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Web Mercator Projection, Cached Slippy Tiles & Procedural 2.5D Topography
# ---------------------------------------------------------------------------

@dataclass
class _MapViewport:
    width: int
    height: int
    zoom: int
    x0_world: float
    y0_world: float
    x1_world: float
    y1_world: float

    def latlon_to_px(self, lat: float, lon: float) -> tuple[float, float]:
        wx, wy = _latlon_to_world_px(lat, lon, self.zoom)
        span_x = max(1.0, self.x1_world - self.x0_world)
        span_y = max(1.0, self.y1_world - self.y0_world)
        px = (wx - self.x0_world) / span_x * self.width
        py = (wy - self.y0_world) / span_y * self.height
        return px, py

    def px_to_latlon(self, px: float, py: float) -> tuple[float, float]:
        wx = self.x0_world + (px / max(1.0, self.width)) * (self.x1_world - self.x0_world)
        wy = self.y0_world + (py / max(1.0, self.height)) * (self.y1_world - self.y0_world)
        return _world_px_to_latlon(wx, wy, self.zoom)


def _latlon_to_world_px(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    lat_c = max(-85.0511, min(85.0511, float(lat)))
    lon_c = max(-180.0, min(180.0, float(lon)))
    scale = 256.0 * (2.0 ** zoom)
    wx = (lon_c + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(lat_c))
    wy = (0.5 - math.log((1.0 + sin_lat) / max(1e-9, 1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    return wx, wy


def _world_px_to_latlon(wx: float, wy: float, zoom: int) -> tuple[float, float]:
    scale = 256.0 * (2.0 ** zoom)
    lon = (wx / scale) * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * (wy / scale)
    lat = math.degrees(math.atan(0.5 * (math.exp(n) - math.exp(-n))))
    return lat, lon


def _compute_viewport(
    waypoints: list[MapWaypointSpec],
    world_w: int,
    world_h: int,
    focus_leg_index: int = -1,
) -> _MapViewport:
    if 0 <= focus_leg_index < len(waypoints) - 1:
        focus_wps = [waypoints[focus_leg_index], waypoints[focus_leg_index + 1]]
    else:
        focus_wps = waypoints
    lats = [w.lat for w in focus_wps]
    lons = [w.lon for w in focus_wps]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    mid_lat = 0.5 * (min_lat + max_lat)
    mid_lon = 0.5 * (min_lon + max_lon)

    best_zoom = 5
    for z in range(15, 1, -1):
        x0, y1 = _latlon_to_world_px(min_lat, min_lon, z)
        x1, y0 = _latlon_to_world_px(max_lat, max_lon, z)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx <= world_w * 0.46 and dy <= world_h * 0.42:
            best_zoom = z
            break

    cx, cy = _latlon_to_world_px(mid_lat, mid_lon, best_zoom)
    half_w = max(world_w * 0.5, 16.0)
    half_h = max(world_h * 0.5, 16.0)
    return _MapViewport(
        width=world_w,
        height=world_h,
        zoom=best_zoom,
        x0_world=cx - half_w,
        y0_world=cy - half_h,
        x1_world=cx + half_w,
        y1_world=cy + half_h,
    )


def _format_coord_dms(val: float, is_lat: bool) -> str:
    hemi = ("N" if val >= 0 else "S") if is_lat else ("E" if val >= 0 else "W")
    v = abs(val)
    deg = int(v)
    mins = int((v - deg) * 60.0)
    return f"{deg}\u00b0{mins:02d}'{hemi}"


_SSL_CTX_CACHE: Optional[ssl.SSLContext] = None
_SSL_CTX_LOCK = threading.Lock()
_HTTP_TLS = threading.local()
_OSRM_ROUTE_MEM: dict[tuple[str, str, str], list[tuple[float, float]]] = {}


def _build_ssl_ctx() -> ssl.SSLContext:
    global _SSL_CTX_CACHE
    if _SSL_CTX_CACHE is not None:
        return _SSL_CTX_CACHE
    with _SSL_CTX_LOCK:
        if _SSL_CTX_CACHE is not None:
            return _SSL_CTX_CACHE
        try:
            import certifi

            _SSL_CTX_CACHE = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            _SSL_CTX_CACHE = ssl.create_default_context()
        return _SSL_CTX_CACHE


def _http_fetch_bytes(url: str, timeout: float = 3.5) -> Optional[bytes]:
    """Fetch URL bytes using a thread-local HTTP/1.1 Keep-Alive connection pool."""
    try:
        sess = getattr(_HTTP_TLS, "session", None)
        if sess is None:
            import requests
            from requests.adapters import HTTPAdapter

            sess = requests.Session()
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            sess.headers.update({"User-Agent": "video_editing_ai/1.0"})
            _HTTP_TLS.session = sess
        resp = sess.get(url, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            return resp.content
        return None
    except ImportError:
        pass
    except Exception:
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "video_editing_ai/1.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=_build_ssl_ctx()) as resp:
            return resp.read()
    except Exception:
        return None


def _compute_tile_grid_for_zoom(
    viewport: _MapViewport,
    z_fetch: int,
) -> tuple[int, int, int, int, int, int, float]:
    scale_f = 2.0 ** (z_fetch - viewport.zoom)
    n_tiles = 1 << z_fetch
    x0_s = viewport.x0_world * scale_f
    x1_s = viewport.x1_world * scale_f
    y0_s = viewport.y0_world * scale_f
    y1_s = viewport.y1_world * scale_f
    tx_min = int(math.floor(x0_s / 256.0))
    tx_max = int(math.floor(x1_s / 256.0))
    ty_min = max(0, int(math.floor(y0_s / 256.0)))
    ty_max = min(n_tiles - 1, int(math.floor(y1_s / 256.0)))
    total_tiles = max(1, (tx_max - tx_min + 1) * (ty_max - ty_min + 1))
    return n_tiles, tx_min, tx_max, ty_min, ty_max, total_tiles, scale_f


def _fetch_tile_mosaic_pil(
    viewport: _MapViewport,
    url_tpl: str,
    layer_tag: str,
    mode: str = "RGB",
    fill_color: tuple[int, ...] = (0, 0, 0),
    cache_dir: Optional[Path] = None,
    retina_boost: bool = True,
) -> Optional[Image.Image]:
    """Fetch parallel disk-cached slippy tiles up to high-DPI (z+3) density within tile budget."""
    max_zoom = 14 if "labels" in layer_tag or "places" in layer_tag else 15
    max_tiles = 420 if viewport.width >= 3000 else 324
    z_fetch = viewport.zoom
    n_tiles, tx_min, tx_max, ty_min, ty_max, total_tiles, scale_f = _compute_tile_grid_for_zoom(viewport, z_fetch)
    if retina_boost and viewport.zoom < max_zoom:
        for z_cand in range(min(max_zoom, viewport.zoom + 3), viewport.zoom - 1, -1):
            cand = _compute_tile_grid_for_zoom(viewport, z_cand)
            if cand[5] <= max_tiles:
                z_fetch = z_cand
                n_tiles, tx_min, tx_max, ty_min, ty_max, total_tiles, scale_f = cand
                break
    if total_tiles > max_tiles:
        return None

    tile_root = (cache_dir or Path("cache")) / "map_tiles" / layer_tag / str(z_fetch)
    tile_root.mkdir(parents=True, exist_ok=True)
    ctx = _build_ssl_ctx()

    canvas_w = max(256, (tx_max - tx_min + 1) * 256)
    canvas_h = max(256, (ty_max - ty_min + 1) * 256)
    mosaic = Image.new(mode, (canvas_w, canvas_h), fill_color)
    fetched_ok = 0

    def _fetch_one(tx_raw: int, ty: int) -> tuple[int, int, Optional[Path]]:
        tx_wrap = tx_raw % n_tiles
        local_png = tile_root / f"{tx_wrap}_{ty}.png"
        if not local_png.exists():
            url = url_tpl.format(z=z_fetch, x=tx_wrap, y=ty)
            data = _http_fetch_bytes(url, timeout=2.8)
            if data is not None and len(data) > 160:
                try:
                    local_png.write_bytes(data)
                except Exception:
                    return tx_raw, ty, None
            else:
                return tx_raw, ty, None
        return tx_raw, ty, local_png if local_png.exists() else None

    coords = [(tx_raw, ty) for tx_raw in range(tx_min, tx_max + 1) for ty in range(ty_min, ty_max + 1)]
    results: list[tuple[int, int, Optional[Path]]] = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        futs = [pool.submit(_fetch_one, tx_r, ty_v) for tx_r, ty_v in coords]
        for fut in as_completed(futs):
            results.append(fut.result())

    for tx_raw, ty, local_png in results:
        if local_png is None:
            continue
        try:
            with Image.open(local_png) as t_im:
                conv = t_im.convert(mode)
                mosaic.paste(conv, ((tx_raw - tx_min) * 256, (ty - ty_min) * 256))
                fetched_ok += 1
        except Exception:
            continue

    if fetched_ok < max(1, total_tiles // 2):
        return None

    ox = int(round(viewport.x0_world * scale_f - tx_min * 256.0))
    oy = int(round(viewport.y0_world * scale_f - ty_min * 256.0))
    ow = int(round((viewport.x1_world - viewport.x0_world) * scale_f))
    oh = int(round((viewport.y1_world - viewport.y0_world) * scale_f))
    crop = mosaic.crop((ox, oy, ox + max(8, ow), oy + max(8, oh)))
    crop = crop.resize((viewport.width, viewport.height), Image.Resampling.LANCZOS)
    if mode == "RGB":
        crop = crop.filter(ImageFilter.UnsharpMask(radius=0.85, percent=36, threshold=2))
    return crop


def _try_fetch_slippy_mosaic(
    viewport: _MapViewport,
    style: str,
    cache_dir: Optional[Path] = None,
    url_override: Optional[str] = None,
    layer_tag: str = "",
    retina_boost: bool = True,
) -> Optional[np.ndarray]:
    """Fetch and disk-cache high-DPI Retina (z+1) slippy tiles covering viewport."""
    spec = CART_THEME_SPECS[style]
    url_tpl = url_override or spec["tile_url"]
    img = _fetch_tile_mosaic_pil(
        viewport=viewport,
        url_tpl=url_tpl,
        layer_tag=layer_tag or style,
        mode="RGB",
        fill_color=tuple(spec["water_deep"]),
        cache_dir=cache_dir,
        retina_boost=retina_boost,
    )
    if img is None:
        return None
    return np.asarray(img, dtype=np.float32)


def _try_fetch_rgba_tile_layer(
    viewport: _MapViewport,
    url_tpl: str,
    layer_tag: str,
    cache_dir: Optional[Path] = None,
    retina_boost: bool = True,
) -> Optional[Image.Image]:
    """Fetch a transparent RGBA slippy reference overlay (roads, places, boundaries)."""
    return _fetch_tile_mosaic_pil(
        viewport=viewport,
        url_tpl=url_tpl,
        layer_tag=layer_tag,
        mode="RGBA",
        fill_color=(0, 0, 0, 0),
        cache_dir=cache_dir,
        retina_boost=retina_boost,
    )


def _synth_fbm_2d(h: int, w: int, seed: int) -> np.ndarray:
    """Fast vectorized multi-octave fractional Brownian motion field in [0, 1]."""
    rng = np.random.default_rng(seed)
    field = np.zeros((h, w), dtype=np.float32)
    amp = 0.55
    weight_sum = 0.0
    for grid_sz in (5, 11, 23, 47, 96):
        gh = max(3, int(round(grid_sz * (h / max(1.0, w)))))
        gw = max(3, grid_sz)
        coarse = rng.uniform(0.0, 1.0, (gh, gw)).astype(np.float32)
        up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        field += up * amp
        weight_sum += amp
        amp *= 0.5
    field = field / max(1e-6, weight_sum)
    fmin, fmax = float(field.min()), float(field.max())
    return (field - fmin) / max(1e-6, fmax - fmin)


def _draw_compass_rose(
    img: Image.Image,
    cx: int,
    cy: int,
    radius: int,
    color_rgb: tuple[int, int, int],
    sub_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    light_paper: bool,
) -> None:
    """Draw a crisp 8-point nautical/cartographic compass rose with true alpha compositing."""
    r = max(18, radius)
    pad = r + max(24, r // 2)
    patch = Image.new("RGBA", (pad * 2, pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch, "RGBA")
    lx, ly = pad, pad
    ring_c = (*color_rgb, 145)
    dark_c = (*color_rgb, 215)
    fill_lite = (color_rgb[0], color_rgb[1], color_rgb[2], 78 if light_paper else 60)

    draw.ellipse([lx - r, ly - r, lx + r, ly + r], outline=ring_c, width=max(1, r // 24))
    inner_r = int(round(r * 0.78))
    draw.ellipse([lx - inner_r, ly - inner_r, lx + inner_r, ly + inner_r], outline=(*color_rgb, 85), width=1)

    for angle_deg in range(0, 360, 45):
        rad = math.radians(angle_deg - 90.0)
        is_cardinal = angle_deg % 90 == 0
        tip_len = r * (0.94 if is_cardinal else 0.62)
        side_len = r * (0.20 if is_cardinal else 0.13)
        perp = rad + math.pi * 0.5
        tip = (lx + math.cos(rad) * tip_len, ly + math.sin(rad) * tip_len)
        left_pt = (lx + math.cos(perp) * side_len, ly + math.sin(perp) * side_len)
        right_pt = (lx - math.cos(perp) * side_len, ly - math.sin(perp) * side_len)
        draw.polygon([(lx, ly), tip, right_pt], fill=dark_c if is_cardinal else fill_lite)
        draw.polygon([(lx, ly), tip, left_pt], fill=fill_lite, outline=(*color_rgb, 145))

    draw.text((lx - max(4, r // 9), ly - r - max(12, r // 3)), "N", font=sub_font, fill=dark_c)
    px0, py0 = int(cx - pad), int(cy - pad)
    if img.mode == "RGBA":
        img.alpha_composite(patch, (px0, py0))
    else:
        tmp = img.convert("RGBA")
        tmp.alpha_composite(patch, (px0, py0))
        img.paste(tmp.convert("RGB"))


def _fetch_osrm_route_latlon(
    orig: MapWaypointSpec,
    dest: MapWaypointSpec,
    cache_dir: Optional[Path] = None,
    profile: str = "driving",
    via_latlons: Optional[list[tuple[float, float]]] = None,
) -> Optional[list[tuple[float, float]]]:
    """Fetch and disk-cache real road/pedestrian network geometry between (orig, [*via], dest) via OSRM."""
    import json as _json

    clean_prof = "foot" if str(profile).lower() in ("walk", "foot", "pedestrian") else "driving"
    vias = [(float(p[0]), float(p[1])) for p in (via_latlons or []) if len(p) >= 2]
    via_tag = ""
    if vias:
        via_sig = hashlib.sha1(";".join(f"{lat:.4f},{lon:.4f}" for lat, lon in vias).encode("utf-8")).hexdigest()[:8]
        via_tag = f"_via_{via_sig}"
    key = f"{orig.lat:.4f}_{orig.lon:.4f}_{dest.lat:.4f}_{dest.lon:.4f}{via_tag}"
    r_dir = (cache_dir or Path("cache")) / "map_routes"
    mem_key = (clean_prof, key, str(r_dir))
    cached_pts = _OSRM_ROUTE_MEM.get(mem_key)
    if cached_pts is not None:
        return cached_pts
    r_dir.mkdir(parents=True, exist_ok=True)
    # Prefer profile-specific cache when present, falling back to legacy coordinate key (only when no vias)
    r_file_prof = r_dir / f"{clean_prof}_{key}.json"
    candidates = [r_file_prof] if vias else [r_dir / f"{key}.json", r_file_prof]
    for candidate in candidates:
        if candidate.exists():
            try:
                pts = _json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(pts, list) and len(pts) >= 2:
                    res = [(float(p[0]), float(p[1])) for p in pts]
                    _OSRM_ROUTE_MEM[mem_key] = res
                    return res
            except Exception:
                pass
    coord_chain = [(orig.lon, orig.lat)] + [(lon, lat) for lat, lon in vias] + [(dest.lon, dest.lat)]
    coord_str = ";".join(f"{lon:.5f},{lat:.5f}" for lon, lat in coord_chain)
    url = (
        f"https://router.project-osrm.org/route/v1/{clean_prof}/"
        f"{coord_str}"
        "?overview=full&geometries=geojson"
    )
    try:
        raw_bytes = _http_fetch_bytes(url, timeout=2.5)
        if not raw_bytes:
            return None
        payload = _json.loads(raw_bytes.decode("utf-8"))
        routes = payload.get("routes") or []
        coords = routes[0].get("geometry", {}).get("coordinates", []) if routes else []
        if len(coords) >= 2:
            # GeoJSON coordinates are [lon, lat]; thin slightly to ~180 points max for smooth AA drawing
            step = max(1, len(coords) // 180)
            latlons = [(float(c[1]), float(c[0])) for c in coords[::step]]
            if latlons[-1] != (float(coords[-1][1]), float(coords[-1][0])):
                latlons.append((float(coords[-1][1]), float(coords[-1][0])))
            r_file_prof.write_text(_json.dumps(latlons), encoding="utf-8")
            _OSRM_ROUTE_MEM[mem_key] = latlons
            return latlons
    except Exception:
        return None
    return None


@lru_cache(maxsize=24)
def _cached_world_baseplate(
    style: str,
    world_w: int,
    world_h: int,
    wp_sig: tuple[tuple[float, float, float, str], ...],
    allow_tile_fetch: bool,
    cache_dir_str: str,
    focus_leg_index: int = -1,
) -> Image.Image:
    """Build or retrieve cached oversampled RGB cartographic world baseplate with 2.5D relief."""
    waypoints = [
        MapWaypointSpec(waypoint_id=f"wp_{i+1:02d}", label=lbl, lat=lat, lon=lon, alt_m=alt)
        for i, (lat, lon, alt, lbl) in enumerate(wp_sig)
    ]
    spec = CART_THEME_SPECS[style]
    vp = _compute_viewport(waypoints, world_w, world_h, focus_leg_index=focus_leg_index)

    # Synthesize terrain field at fast intermediate resolution and upscale
    sim_w = min(960, max(320, world_w // 2))
    sim_h = max(180, int(round(sim_w * (world_h / max(1.0, world_w)))))
    seed_src = "|".join(f"{w.lat:.2f},{w.lon:.2f}" for w in waypoints)
    seed_int = int(hashlib.sha1(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    noise_fbm = _synth_fbm_2d(sim_h, sim_w, seed_int)

    # Shape elevation around geodesic route corridors so stops sit on credible coastal/highland land
    px_stops = [vp.latlon_to_px(w.lat, w.lon) for w in waypoints]
    yy, xx = np.mgrid[0:sim_h, 0:sim_w].astype(np.float32)
    xx_w = xx * (world_w / float(sim_w))
    yy_w = yy * (world_h / float(sim_h))

    max_dist_km = 0.0
    for i in range(1, len(waypoints)):
        max_dist_km = max(max_dist_km, haversine_km(waypoints[i - 1].lat, waypoints[i - 1].lon, waypoints[i].lat, waypoints[i].lon))
    is_transoceanic = max_dist_km >= 750.0

    corridor_field = np.zeros((sim_h, sim_w), dtype=np.float32)
    sigma_px = world_w * (0.14 if is_transoceanic else 0.24)
    if is_transoceanic:
        # Distinct landmasses around each terminal cluster separated by deep water basin
        for i, (sx, sy) in enumerate(px_stops):
            alt_boost = 0.20 * min(1.0, max(0.0, waypoints[i].alt_m / 2000.0))
            d2 = ((xx_w - sx) ** 2 + (yy_w - sy) ** 2) / max(100.0, sigma_px ** 2)
            corridor_field = np.maximum(corridor_field, (0.76 + alt_boost) * np.exp(-0.5 * d2))
    else:
        for i in range(max(1, len(px_stops) - 1)):
            ax, ay = px_stops[i]
            bx, by = px_stops[min(i + 1, len(px_stops) - 1)]
            vx, vy = bx - ax, by - ay
            denom = max(1e-3, vx * vx + vy * vy)
            t_proj = np.clip(((xx_w - ax) * vx + (yy_w - ay) * vy) / denom, 0.0, 1.0)
            cx_p = ax + vx * t_proj
            cy_p = ay + vy * t_proj
            d2 = ((xx_w - cx_p) ** 2 + (yy_w - cy_p) ** 2) / max(100.0, sigma_px ** 2)
            corridor_field = np.maximum(corridor_field, 0.80 * np.exp(-0.5 * d2))

    # Asymmetric directional continent shelf + organic fractal coastlines
    coast_bias = (xx / float(sim_w) - 0.5) * 0.14 + (yy / float(sim_h) - 0.5) * 0.08
    elev_sim = np.clip(0.56 * corridor_field + 0.52 * (noise_fbm - 0.42) + coast_bias + 0.14, 0.0, 1.0)
    sea_level = 0.44
    land_mask_sim = elev_sim >= sea_level

    # Directional 315 deg NW Sobel hillshade
    gx = cv2.Sobel(elev_sim, cv2.CV_32F, 1, 0, ksize=3) * 18.0
    gy = cv2.Sobel(elev_sim, cv2.CV_32F, 0, 1, ksize=3) * 18.0
    slope = np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gy, gx)
    sun_alt = math.radians(42.0)
    sun_az = math.radians(315.0)
    shade_sim = (
        math.cos(sun_alt) * np.cos(slope)
        + math.sin(sun_alt) * np.sin(slope) * np.cos(sun_az - aspect)
    )
    shade_sim = np.clip(0.68 + 0.46 * (shade_sim - 0.5), 0.52, 1.32)

    # Upscale fields smoothly to world canvas size
    elev = cv2.resize(elev_sim, (world_w, world_h), interpolation=cv2.INTER_CUBIC)
    shade = cv2.resize(shade_sim, (world_w, world_h), interpolation=cv2.INTER_CUBIC)
    land_dist = cv2.distanceTransform((~land_mask_sim).astype(np.uint8), cv2.DIST_L2, 5)
    land_dist_up = cv2.resize(land_dist, (world_w, world_h), interpolation=cv2.INTER_LINEAR)

    # Color-map water bathymetry & land elevation relief
    water_d = np.array(spec["water_deep"], dtype=np.float32)
    water_s = np.array(spec["water_shallow"], dtype=np.float32)
    bathy_c = np.array(spec["bathymetry_rgb"], dtype=np.float32)
    land_l = np.array(spec["land_low"], dtype=np.float32)
    land_m = np.array(spec["land_mid"], dtype=np.float32)
    land_h = np.array(spec["land_high"], dtype=np.float32)

    shore_glow = np.exp(-land_dist_up / max(4.0, sim_w * 0.045))[..., None]
    water_rgb = water_d * (1.0 - shore_glow) + water_s * shore_glow

    # Three crisp coastal bathymetric echo rings
    for ring_d, ring_w in ((6.0, 0.35), (15.0, 0.22), (28.0, 0.14)):
        ring_mask = np.exp(-((land_dist_up - ring_d) ** 2) / 3.2)[..., None] * ring_w
        water_rgb = water_rgb * (1.0 - ring_mask) + bathy_c * ring_mask

    h_norm = np.clip((elev - sea_level) / max(1e-4, 1.0 - sea_level), 0.0, 1.0)
    t1 = np.clip(h_norm * 2.0, 0.0, 1.0)[..., None]
    t2 = np.clip((h_norm - 0.5) * 2.0, 0.0, 1.0)[..., None]
    land_rgb = np.where(
        h_norm[..., None] < 0.5,
        land_l * (1.0 - t1) + land_m * t1,
        land_m * (1.0 - t2) + land_h * t2,
    )
    land_rgb = land_rgb * shade[..., None]

    # Smooth antialiased coastline blend
    coast_alpha = np.clip((elev - sea_level) * 55.0 + 0.5, 0.0, 1.0)[..., None]
    rgb = water_rgb * (1.0 - coast_alpha) + land_rgb * coast_alpha

    # Optional online cached Retina slippy-tile base + 3D DEM World_Hillshade relief
    has_real_tiles = False
    c_dir_obj = Path(cache_dir_str) if cache_dir_str else None
    if allow_tile_fetch and os.environ.get("AI_EDITOR_OFFLINE_MAPS") != "1":
        tile_arr = _try_fetch_slippy_mosaic(vp, style=style, cache_dir=c_dir_obj, retina_boost=True)
        if tile_arr is not None and tile_arr.shape == rgb.shape:
            has_real_tiles = True
            blend = float(spec.get("tile_blend", 0.52))
            if style == "archival_expedition_parchment":
                # Premium museum parchment clarity wash: calm noisy topo clutter so crimson route & Polaroids pop
                soft_t = cv2.GaussianBlur(tile_arr.astype(np.float32), (3, 3), 0.55)
                clean_t = tile_arr.astype(np.float32) * 0.68 + soft_t * 0.32
                luma_t = (0.299 * clean_t[..., 0] + 0.587 * clean_t[..., 1] + 0.114 * clean_t[..., 2])[..., None]
                desat_t = luma_t + (clean_t - luma_t) * 0.72
                paper_veil = np.array([234.0, 225.0, 206.0], dtype=np.float32)
                washed = np.clip(desat_t * 0.80 + paper_veil * 0.20, 0, 255)
                warm = np.array([1.015, 0.992, 0.955], dtype=np.float32)
                rgb = np.clip(washed * warm, 0, 255) * blend + rgb * (1.0 - blend)
            elif style == "kinfolk_minimal_monograph":
                soft_t = cv2.GaussianBlur(tile_arr.astype(np.float32), (3, 3), 0.60)
                clean_t = tile_arr.astype(np.float32) * 0.62 + soft_t * 0.38
                luma_t = (0.299 * clean_t[..., 0] + 0.587 * clean_t[..., 1] + 0.114 * clean_t[..., 2])[..., None]
                desat_t = luma_t + (clean_t - luma_t) * 0.58
                alabaster = np.array([240.0, 236.0, 228.0], dtype=np.float32)
                washed = np.clip(desat_t * 0.78 + alabaster * 0.22, 0, 255)
                rgb = washed * blend + rgb * (1.0 - blend)
            elif style == "midnight_vox_atlas":
                rgb = np.clip(tile_arr * 1.15, 0, 255) * blend + rgb * (1.0 - blend)
            else:
                rgb = np.clip(tile_arr, 0, 255) * blend + rgb * (1.0 - blend)

            hs_url = spec.get("hillshade_url")
            hs_strength = float(spec.get("hillshade_strength", 0.0) or 0.0)
            if hs_url and hs_strength > 0.01:
                hs_arr = _try_fetch_slippy_mosaic(
                    vp,
                    style=style,
                    cache_dir=c_dir_obj,
                    url_override=hs_url,
                    layer_tag="world_hillshade",
                    retina_boost=False,
                )
                if hs_arr is not None and hs_arr.shape == rgb.shape:
                    # Esri World_Hillshade flat sea-level neutral luminance is ~0.706 (RGB 180)
                    eff_hs = hs_strength * (0.42 if vp.zoom >= 10 else 1.0)
                    hs_luma = (0.299 * hs_arr[:, :, 0] + 0.587 * hs_arr[:, :, 1] + 0.114 * hs_arr[:, :, 2]) / 255.0
                    relief = np.clip(1.0 + eff_hs * ((hs_luma - 0.706) * 2.2), 0.74, 1.16)[..., None]
                    rgb = np.clip(rgb * relief, 0, 255)

            if style in ("satellite_hybrid_4k", "aerial_relief_hybrid"):
                # Google Earth clarity grade with protective highlight rolloff on sunny sand & reef whites
                rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
                lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                mean_l = float(l_ch.mean())
                clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(12, 12))
                l_eq = clahe.apply(l_ch)
                l_mix = cv2.addWeighted(l_ch, 0.78, l_eq, 0.22, 0).astype(np.float32)
                if mean_l > 130.0:
                    # Tame blown-out sunny aerial whites (beaches, rooftops, surf) while preserving deep blues
                    l_norm = np.clip(l_mix / 255.0, 0.0, 1.0)
                    l_mix = np.power(l_norm, 1.12) * 246.0
                l_u8 = np.clip(l_mix, 0, 240).astype(np.uint8)
                lab_clean = cv2.merge((l_u8, a_ch, b_ch))
                hsv = cv2.cvtColor(cv2.cvtColor(lab_clean, cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.18, 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.94 if mean_l > 130.0 else 1.0), 0, 232)
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    # Quantized marching iso-elevation contour hairlines on land (only when using offline synthetic terrain)
    if not has_real_tiles:
        land_u8 = (elev >= sea_level).astype(np.uint8)
        edges = cv2.Canny(land_u8 * 255, 80, 180)
        edges_d = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)
        coast_mask = (edges_d > 0)[..., None]
        coast_arr = np.array(spec["coast_rgb"], dtype=np.float32)
        rgb = np.where(coast_mask, rgb * 0.35 + coast_arr * 0.65, rgb)

        contour_col = np.array(spec["contour_rgba"][:3], dtype=np.float32)
        contour_a = spec["contour_rgba"][3] / 255.0
        step_levels = np.floor(h_norm * 7.0).astype(np.uint8)
        iso_gx = np.abs(np.diff(step_levels, axis=1, prepend=step_levels[:, :1])) > 0
        iso_gy = np.abs(np.diff(step_levels, axis=0, prepend=step_levels[:1, :])) > 0
        iso_mask = ((iso_gx | iso_gy) & (elev >= sea_level + 0.02))[..., None]
        rgb = np.where(iso_mask, rgb * (1.0 - contour_a) + contour_col * contour_a, rgb)

    base_img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")
    if has_real_tiles and spec.get("roads_overlay_url") and (vp.zoom >= 7 or not is_transoceanic):
        roads_im = _try_fetch_rgba_tile_layer(
            vp,
            url_tpl=str(spec["roads_overlay_url"]),
            layer_tag="world_transportation",
            cache_dir=c_dir_obj,
            retina_boost=True,
        )
        if roads_im is not None:
            r_alpha = roads_im.getchannel("A").point(lambda v: int(round(v * 0.85)))
            roads_im.putalpha(r_alpha)
            base_img.alpha_composite(roads_im)
    grat_layer = Image.new("RGBA", (world_w, world_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grat_layer, "RGBA")
    s = world_w / 1920.0
    f_mono = _load_font_tier(max(10, int(round(12 * s))), tier="mono", bold=False)

    # Lat/Lon cartographic graticule grid (true translucent archival hairlines via RGBA layer composite)
    grat_col = tuple(spec["graticule_rgba"])
    if grat_col[3] >= 20:
        for gx_frac in (0.18, 0.34, 0.50, 0.66, 0.82):
            px = int(round(world_w * gx_frac))
            draw.line([(px, 0), (px, world_h)], fill=grat_col, width=max(1, int(round(1.0 * s))))
            _, g_lon = vp.px_to_latlon(px, world_h * 0.5)
            draw.text((px + int(round(4 * s)), world_h - int(round(20 * s))), _format_coord_dms(g_lon, False), font=f_mono, fill=(*spec["coast_rgb"], 118))
        for gy_frac in (0.20, 0.38, 0.56, 0.74):
            py = int(round(world_h * gy_frac))
            draw.line([(0, py), (world_w, py)], fill=grat_col, width=max(1, int(round(1.0 * s))))
            g_lat, _ = vp.px_to_latlon(world_w * 0.5, py)
            draw.text((int(round(14 * s)), py + int(round(3 * s))), _format_coord_dms(g_lat, True), font=f_mono, fill=(*spec["coast_rgb"], 118))

    # Optional engraved alternating cartographic neatline border (Archival / Kinfolk)
    if spec.get("neatline"):
        bw = max(6, int(round(8 * s)))
        dark_b = (*spec["coast_rgb"], 155)
        light_b = (*spec["land_low"], 195)
        seg = max(48, int(round(96 * s)))
        for x0 in range(0, world_w, seg):
            c = dark_b if ((x0 // seg) % 2 == 0) else light_b
            draw.rectangle([x0, 0, min(world_w, x0 + seg), bw], fill=c)
            draw.rectangle([x0, world_h - bw, min(world_w, x0 + seg), world_h], fill=c)
        for y0 in range(0, world_h, seg):
            c = dark_b if ((y0 // seg) % 2 == 0) else light_b
            draw.rectangle([0, y0, bw, min(world_h, y0 + seg)], fill=c)
            draw.rectangle([world_w - bw, y0, world_w, min(world_h, y0 + seg)], fill=c)

    base_img.alpha_composite(grat_layer)
    # Compass rose is rendered cleanly in post-warp screen space inside _render_screen_hud
    # so it never drifts across routes or Polaroids during camera pans.
    return base_img.convert("RGB")


@lru_cache(maxsize=16)
def _cached_labels_overlay(
    style: str,
    world_w: int,
    world_h: int,
    wp_sig: tuple[tuple[float, float, float, str], ...],
    allow_tile_fetch: bool,
    cache_dir_str: str,
    focus_leg_index: int = -1,
) -> Optional[Image.Image]:
    """Fetch optional transparent RGBA reference labels overlay so city names float crisp above routes."""
    if not allow_tile_fetch or os.environ.get("AI_EDITOR_OFFLINE_MAPS") == "1":
        return None
    spec = CART_THEME_SPECS.get(style, {})
    lbl_url = spec.get("labels_overlay_url")
    if not lbl_url:
        return None
    waypoints = [
        MapWaypointSpec(waypoint_id=f"wp_{i+1:02d}", label=lbl, lat=lat, lon=lon, alt_m=alt)
        for i, (lat, lon, alt, lbl) in enumerate(wp_sig)
    ]
    vp = _compute_viewport(waypoints, world_w, world_h, focus_leg_index=focus_leg_index)
    lbl_tag = "world_dark_labels" if "Dark_Gray" in str(lbl_url) else ("world_light_labels" if "Light_Gray" in str(lbl_url) else "world_places")
    return _try_fetch_rgba_tile_layer(
        vp,
        url_tpl=str(lbl_url),
        layer_tag=lbl_tag,
        cache_dir=Path(cache_dir_str) if cache_dir_str else None,
        retina_boost=True,
    )


# ---------------------------------------------------------------------------
# Great-Circle Flight Geodesics, Arc-Length Path Engine & Sub-Pixel Shaders
# ---------------------------------------------------------------------------

def _great_circle_latlon_pts(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    n_samples: int = 160,
) -> list[tuple[float, float]]:
    """Sample points along a spherical 3D Great-Circle SLERP geodesic between (lat1, lon1) and (lat2, lon2)."""
    phi1, lam1 = math.radians(float(lat1)), math.radians(float(lon1))
    phi2, lam2 = math.radians(float(lat2)), math.radians(float(lon2))
    p0 = np.array([math.cos(phi1) * math.cos(lam1), math.cos(phi1) * math.sin(lam1), math.sin(phi1)], dtype=np.float64)
    p1 = np.array([math.cos(phi2) * math.cos(lam2), math.cos(phi2) * math.sin(lam2), math.sin(phi2)], dtype=np.float64)
    dot = float(np.clip(np.dot(p0, p1), -1.0, 1.0))
    omega = math.acos(dot)
    n = max(12, int(n_samples))
    out: list[tuple[float, float]] = []
    if omega < 1e-6:
        for i in range(n):
            t = i / float(n - 1)
            out.append((lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t))
        return out
    sin_om = math.sin(omega)
    for i in range(n):
        t = i / float(n - 1)
        a = math.sin((1.0 - t) * omega) / sin_om
        b = math.sin(t * omega) / sin_om
        v = a * p0 + b * p1
        v_norm = float(np.linalg.norm(v))
        if v_norm > 1e-12:
            v = v / v_norm
        lat_t = math.degrees(math.asin(max(-1.0, min(1.0, float(v[2])))))
        lon_t = math.degrees(math.atan2(float(v[1]), float(v[0])))
        out.append((lat_t, lon_t))
    return out


def _project_flight_latlons_to_px(
    vp: _MapViewport,
    orig: MapWaypointSpec,
    dest: MapWaypointSpec,
    gc_latlons: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Project Great-Circle lat/lon points into viewport pixel coords safely across antimeridian arcs."""
    crosses_dateline = abs(dest.lon - orig.lon) > 180.0
    if not crosses_dateline:
        return [vp.latlon_to_px(lat, lon) for lat, lon in gc_latlons]
    # Preserve the curved high-latitude Great-Circle profile without horizontal [-180..+180] wrap snaps
    x0, _ = vp.latlon_to_px(orig.lat, orig.lon)
    x1, _ = vp.latlon_to_px(dest.lat, dest.lon)
    n = max(1, len(gc_latlons) - 1)
    pts: list[tuple[float, float]] = []
    for i, (lat_t, _) in enumerate(gc_latlons):
        frac = i / float(n)
        _, py = vp.latlon_to_px(lat_t, orig.lon)
        pts.append((x0 + (x1 - x0) * frac, py))
    return pts


class _ArcLengthPath:
    """Cumulative Euclidean arc-length reparameterizer for jitter-free constant-speed vehicle motion."""

    def __init__(self, pts: list[tuple[float, float]]) -> None:
        clean: list[tuple[float, float]] = []
        for x, y in pts:
            if not clean or math.hypot(x - clean[-1][0], y - clean[-1][1]) >= 0.05:
                clean.append((float(x), float(y)))
        if len(clean) < 2:
            p0 = clean[0] if clean else (0.0, 0.0)
            clean = [p0, (p0[0] + 1.0, p0[1])]
        self.pts = np.asarray(clean, dtype=np.float64)
        diffs = np.diff(self.pts, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        self.cum_d = np.concatenate(([0.0], np.cumsum(seg_lens)))
        self.total_len = float(self.cum_d[-1]) if float(self.cum_d[-1]) > 1e-6 else 1.0

    def point_at_s(self, s: float) -> tuple[float, float]:
        s_c = max(0.0, min(self.total_len, float(s)))
        if s_c <= 0.0:
            return (float(self.pts[0, 0]), float(self.pts[0, 1]))
        if s_c >= self.total_len:
            return (float(self.pts[-1, 0]), float(self.pts[-1, 1]))
        idx = int(np.searchsorted(self.cum_d, s_c, side="right")) - 1
        idx = max(0, min(len(self.pts) - 2, idx))
        s0 = float(self.cum_d[idx])
        s1 = float(self.cum_d[idx + 1])
        frac = (s_c - s0) / max(1e-6, s1 - s0)
        x = float(self.pts[idx, 0] + (self.pts[idx + 1, 0] - self.pts[idx, 0]) * frac)
        y = float(self.pts[idx, 1] + (self.pts[idx + 1, 1] - self.pts[idx, 1]) * frac)
        return (x, y)

    def sample_prefix(self, u: float) -> list[tuple[float, float]]:
        s_target = max(0.0, min(1.0, float(u))) * self.total_len
        if s_target <= 1e-4:
            p0 = (float(self.pts[0, 0]), float(self.pts[0, 1]))
            return [p0, self.point_at_s(min(self.total_len, 1.0))]
        idx = int(np.searchsorted(self.cum_d, s_target, side="right"))
        prefix = [(float(self.pts[k, 0]), float(self.pts[k, 1])) for k in range(max(1, min(len(self.pts), idx)))]
        tip = self.point_at_s(s_target)
        if math.hypot(tip[0] - prefix[-1][0], tip[1] - prefix[-1][1]) >= 0.01:
            prefix.append(tip)
        return prefix

    def sample_window(self, s_start: float, s_end: float, step_px: float = 4.0) -> list[tuple[float, float]]:
        s0 = max(0.0, min(self.total_len, float(s_start)))
        s1 = max(0.0, min(self.total_len, float(s_end)))
        if s1 - s0 < 0.25:
            return []
        n = max(3, int(math.ceil((s1 - s0) / max(1.0, step_px))) + 1)
        return [self.point_at_s(s0 + (s1 - s0) * (k / float(n - 1))) for k in range(n)]

    def tangent_and_curvature(self, u: float, window_px: float = 22.0) -> tuple[float, float]:
        s_cur = max(0.0, min(self.total_len, float(u) * self.total_len))
        half_w = max(4.0, float(window_px))
        # Gaussian-weighted local tangent samples to eliminate highway-ramp heading snap
        offsets = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
        weights = np.array([0.05, 0.09, 0.13, 0.17, 0.20, 0.17, 0.13, 0.09, 0.05], dtype=np.float64)
        angles: list[float] = []
        delta = max(2.0, half_w * 0.22)
        for off in offsets:
            sc = max(0.0, min(self.total_len, s_cur + off * half_w))
            sa = max(0.0, sc - delta)
            sb = min(self.total_len, sc + delta)
            if sb - sa < 0.5:
                sa = max(0.0, self.total_len - 2.0)
                sb = self.total_len
            pa = self.point_at_s(sa)
            pb = self.point_at_s(sb)
            ang = math.atan2(pb[1] - pa[1], pb[0] - pa[0]) if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 1e-4 else 0.0
            angles.append(ang)
        unwrapped = np.unwrap(np.array(angles, dtype=np.float64))
        weights /= np.sum(weights)
        smooth_heading = float(np.sum(unwrapped * weights))
        turn_diff = float(unwrapped[-2] - unwrapped[1])
        bank_factor = float(np.clip(turn_diff / 0.45, -1.0, 1.0))
        return smooth_heading, bank_factor

    def smoothed_camera_point(self, u: float, window_px: float = 160.0) -> tuple[float, float]:
        s_cur = max(0.0, min(self.total_len, float(u) * self.total_len))
        half_w = max(8.0, float(window_px))
        xs, ys = [], []
        for k in range(11):
            frac = (k / 10.0) * 2.0 - 1.0
            px, py = self.point_at_s(s_cur + frac * half_w)
            xs.append(px)
            ys.append(py)
        return (float(np.mean(xs)), float(np.mean(ys)))

def _ease_in_out_cubic(u: float) -> float:
    x = max(0.0, min(1.0, float(u)))
    return 4.0 * x * x * x if x < 0.5 else 1.0 - ((-2.0 * x + 2.0) ** 3) * 0.5


def _ease_in_out_quintic(u: float) -> float:
    """C2-continuous quintic smootherstep (zero initial/final velocity & acceleration) for serene camera motion."""
    x = max(0.0, min(1.0, float(u)))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _cartography_travel_progress(t_norm: float) -> float:
    """Calm departure dwell (0..10%) -> fluid quintic glide (10..86%) -> serene arrival settle (86..100%)."""
    t = max(0.0, min(1.0, float(t_norm)))
    if t <= 0.10:
        return 0.0
    if t >= 0.86:
        return 1.0
    return _ease_in_out_quintic((t - 0.10) / (0.86 - 0.10))


def _catmull_rom_xy(pts: list[tuple[float, float]], samples_per_seg: int = 120) -> list[tuple[float, float]]:
    if len(pts) < 2:
        return list(pts)
    ext = [pts[0]] + list(pts) + [pts[-1]]
    out: list[tuple[float, float]] = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for j in range(samples_per_seg):
            t = j / float(samples_per_seg)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2.0 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2.0 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
            )
            out.append((x, y))
    out.append(pts[-1])
    return out


def _draw_aa_polyline(
    overlay_bgr_a: np.ndarray,
    pts: list[tuple[float, float]],
    rgba: tuple[int, int, int, int],
    thickness: int,
) -> None:
    clean_pts = [(x, y) for x, y in pts if math.isfinite(x) and math.isfinite(y) and abs(x) < 1e6 and abs(y) < 1e6]
    if len(clean_pts) < 2:
        return
    arr = np.round(np.asarray(clean_pts, dtype=np.float64) * 16.0).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        overlay_bgr_a,
        [arr],
        False,
        color=(rgba[0], rgba[1], rgba[2], rgba[3]),
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_AA,
        shift=4,
    )


def _draw_dashed_aa_polyline(
    overlay_rgba: np.ndarray,
    pts: list[tuple[float, float]],
    rgba: tuple[int, int, int, int],
    thickness: int,
    dash_px: float,
    gap_px: float,
) -> None:
    if len(pts) < 2:
        return
    draw_on = True
    rem = dash_px
    for i in range(1, len(pts)):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg < 1e-4:
            continue
        pos = 0.0
        while pos < seg:
            step = min(rem, seg - pos)
            if draw_on:
                t0 = pos / seg
                t1 = (pos + step) / seg
                cv2.line(
                    overlay_rgba,
                    (int(round((x1 + (x2 - x1) * t0) * 16.0)), int(round((y1 + (y2 - y1) * t0) * 16.0))),
                    (int(round((x1 + (x2 - x1) * t1) * 16.0)), int(round((y1 + (y2 - y1) * t1) * 16.0))),
                    color=rgba,
                    thickness=max(1, thickness),
                    lineType=cv2.LINE_AA,
                    shift=4,
                )
            pos += step
            rem -= step
            if rem <= 1e-3:
                draw_on = not draw_on
                rem = dash_px if draw_on else gap_px


def _draw_cased_route(
    route_arr: np.ndarray,
    pts: list[tuple[float, float]],
    spec: dict[str, Any],
    stroke_w: int,
    s_w: float,
    is_flight: bool = False,
    ground_pts: Optional[list[tuple[float, float]]] = None,
) -> None:
    """Render a 4-pass sub-pixel AA route shader: shadow -> contrasting casing -> core -> comet tip."""
    if len(pts) < 2:
        return
    if is_flight and ground_pts and len(ground_pts) >= 2:
        _draw_dashed_aa_polyline(route_arr, ground_pts, (8, 12, 18, 92), max(2, stroke_w - 1), dash_px=10.0 * s_w, gap_px=8.0 * s_w)

    # 1. Soft ground shadow / atmospheric bloom
    if is_flight or not spec.get("light_paper", False):
        _draw_aa_polyline(route_arr, pts, (*spec["route_glow"], 96), max(4, stroke_w * 3))
    else:
        off_x, off_y = 1.6 * s_w, 2.4 * s_w
        shadow_pts = [(px + off_x, py + off_y) for px, py in pts]
        _draw_aa_polyline(route_arr, shadow_pts, (12, 18, 28, 72), max(4, stroke_w * 2 + int(round(4 * s_w))))

    # 2. High-contrast outer casing
    casing_rgb = spec.get("route_casing", (18, 48, 112))
    casing_w = max(4, stroke_w + max(3, int(round(4.5 * s_w))))
    _draw_aa_polyline(route_arr, pts, (*casing_rgb, 240), casing_w)

    # 3. Saturated inner core
    _draw_aa_polyline(route_arr, pts, (*spec["route_core"], 255), max(2, stroke_w + 1))

    # 4. Multi-stage tapered luminous comet tail leading smoothly into the moving beacon
    arc_p = _ArcLengthPath(pts)
    tail_len = min(arc_p.total_len * 0.35, max(56.0 * s_w, 128.0 * s_w))
    if tail_len >= 6.0 * s_w:
        rc = spec["route_core"]
        ac = spec["accent"]
        glow_c = spec["route_glow"]
        # 4 stepped segments from trailing tail -> blazing tip
        stages = (
            (1.00, 0.62, glow_c, 95, max(2, int(round(stroke_w * 1.45)))),
            (0.66, 0.34, ac, 155, max(2, int(round(stroke_w * 1.15)))),
            (0.38, 0.12, (min(255, int(rc[0] * 0.45 + 255 * 0.55)), min(255, int(rc[1] * 0.45 + 255 * 0.55)), min(255, int(rc[2] * 0.45 + 255 * 0.55))), 215, max(2, stroke_w)),
            (0.15, 0.00, (255, 250, 238), 245, max(1, stroke_w - 1)),
        )
        for frac_back, frac_front, col_rgb, alpha_v, seg_w in stages:
            s0 = max(0.0, arc_p.total_len - tail_len * frac_back)
            s1 = max(s0, arc_p.total_len - tail_len * frac_front)
            if s1 - s0 >= 1.5:
                seg_pts = arc_p.sample_window(s0, s1, step_px=max(2.0, 3.5 * s_w))
                if len(seg_pts) >= 2:
                    _draw_aa_polyline(route_arr, seg_pts, (*col_rgb, alpha_v), seg_w)


def _draw_aircraft_ground_shadow(
    canvas: Image.Image,
    gx: float,
    gy: float,
    angle_rad: float,
    scale: float,
    alpha: int = 90,
) -> None:
    """Render a soft Gaussian-blurred ground silhouette shadow beneath an airborne aircraft."""
    if alpha <= 4:
        return
    box = max(44, int(round(80 * scale)))
    ss = box * 2
    sub = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(sub, "RGBA")
    m = ss * 0.5
    r = ss * 0.34
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)

    def _rot(lx: float, ly: float) -> tuple[float, float]:
        return (m + lx * ca - ly * sa, m + lx * sa + ly * ca)

    poly = [
        _rot(r * 1.04, 0.0),
        _rot(r * 0.32, r * 0.14),
        _rot(-r * 0.26, r * 0.94),
        _rot(-r * 0.46, r * 0.88),
        _rot(-r * 0.12, r * 0.15),
        _rot(-r * 0.72, r * 0.10),
        _rot(-r * 0.92, r * 0.42),
        _rot(-r * 1.02, r * 0.36),
        _rot(-r * 0.86, 0.0),
        _rot(-r * 1.02, -r * 0.36),
        _rot(-r * 0.92, -r * 0.42),
        _rot(-r * 0.72, -r * 0.10),
        _rot(-r * 0.12, -r * 0.15),
        _rot(-r * 0.46, -r * 0.88),
        _rot(-r * 0.26, -r * 0.94),
        _rot(r * 0.32, -r * 0.14),
    ]
    d.polygon(poly, fill=(6, 10, 16, max(12, min(200, int(alpha)))))
    sub = sub.filter(ImageFilter.GaussianBlur(radius=max(1.5, ss * 0.055)))
    down = sub.resize((box, box), Image.Resampling.LANCZOS)
    canvas.alpha_composite(down, (int(round(gx - box * 0.5)), int(round(gy - box * 0.5))))


def _draw_flight_contrails(
    canvas: Image.Image,
    arc_pts: list[tuple[float, float]],
    scale: float,
    alt_sin: float,
) -> None:
    """Draw twin fading turbofan vapor contrails along the elevated flight path behind wings."""
    if len(arc_pts) < 4 or alt_sin <= 0.08:
        return
    arc_path = _ArcLengthPath(arc_pts)
    trail_len = min(arc_path.total_len * 0.72, 175.0 * scale)
    if trail_len < 8.0:
        return
    s0 = max(0.0, arc_path.total_len - trail_len)
    samples = arc_path.sample_window(s0, arc_path.total_len, step_px=max(2.0, 4.0 * scale))
    if len(samples) < 3:
        return

    overlay = np.zeros((canvas.height, canvas.width, 4), dtype=np.uint8)
    wing_sep = 11.5 * scale * (0.78 + 0.34 * alt_sin)
    base_thick = max(2, int(round(2.4 * scale)))
    # Build port and starboard contrail ribbons
    for k in range(1, len(samples)):
        frac = k / float(max(1, len(samples) - 1))
        x0, y0 = samples[k - 1]
        x1, y1 = samples[k]
        dx, dy = x1 - x0, y1 - y0
        seg_l = math.hypot(dx, dy)
        if seg_l < 0.05:
            continue
        nx, ny = -dy / seg_l, dx / seg_l
        fade_a = int(round(145.0 * (frac ** 1.45) * min(1.0, alt_sin * 1.35)))
        if fade_a < 4:
            continue
        for sign in (-1.0, 1.0):
            ox, oy = nx * wing_sep * sign, ny * wing_sep * sign
            cv2.line(
                overlay,
                (int(round((x0 + ox) * 16.0)), int(round((y0 + oy) * 16.0))),
                (int(round((x1 + ox) * 16.0)), int(round((y1 + oy) * 16.0))),
                color=(238, 246, 255, fade_a),
                thickness=base_thick,
                lineType=cv2.LINE_AA,
                shift=4,
            )
    contrail_im = Image.fromarray(overlay).filter(ImageFilter.GaussianBlur(radius=max(0.6, 1.1 * scale)))
    canvas.alpha_composite(contrail_im)


def _draw_boat_kelvin_wake(
    canvas: Image.Image,
    arc_path: _ArcLengthPath,
    u: float,
    scale: float,
) -> None:
    """Render an animated foaming V-shaped Kelvin bow wake and center prop wash behind a vessel."""
    s_cur = max(0.0, min(arc_path.total_len, float(u) * arc_path.total_len))
    wake_len = min(s_cur, 115.0 * scale)
    if wake_len < 6.0:
        return
    samples = arc_path.sample_window(s_cur - wake_len, s_cur, step_px=max(2.0, 3.5 * scale))
    if len(samples) < 3:
        return

    overlay = np.zeros((canvas.height, canvas.width, 4), dtype=np.uint8)
    for k in range(1, len(samples)):
        frac = k / float(max(1, len(samples) - 1))  # 0=oldest tail, 1=at hull
        dist_back = 1.0 - frac
        x0, y0 = samples[k - 1]
        x1, y1 = samples[k]
        dx, dy = x1 - x0, y1 - y0
        seg_l = math.hypot(dx, dy)
        if seg_l < 0.05:
            continue
        nx, ny = -dy / seg_l, dx / seg_l
        arm_spread = (4.5 + 24.0 * dist_back) * scale
        arm_alpha = int(round(170.0 * (frac ** 1.3)))
        center_alpha = int(round(115.0 * (frac ** 1.6)))
        w_arm = max(1, int(round((1.5 + 2.0 * dist_back) * scale)))
        for sign in (-1.0, 1.0):
            ox, oy = nx * arm_spread * sign, ny * arm_spread * sign
            cv2.line(
                overlay,
                (int(round((x0 + ox) * 16.0)), int(round((y0 + oy) * 16.0))),
                (int(round((x1 + ox) * 16.0)), int(round((y1 + oy) * 16.0))),
                color=(242, 250, 255, arm_alpha),
                thickness=w_arm,
                lineType=cv2.LINE_AA,
                shift=4,
            )
        cv2.line(
            overlay,
            (int(round(x0 * 16.0)), int(round(y0 * 16.0))),
            (int(round(x1 * 16.0)), int(round(y1 * 16.0))),
            color=(210, 240, 252, center_alpha),
            thickness=max(1, int(round(2.5 * scale))),
            lineType=cv2.LINE_AA,
            shift=4,
        )
    canvas.alpha_composite(Image.fromarray(overlay))


def _draw_glow_beacon_dot(
    canvas: Image.Image,
    cx: float,
    cy: float,
    glow_rgb: tuple[int, int, int],
    core_rgb: tuple[int, int, int],
    stroke_rgb: tuple[int, int, int],
    scale: float,
    pulse_phase: float = 0.0,
) -> None:
    """Render a sleek 4x-supersampled glowing beacon dot with gentle rhythmic pulse."""
    pulse = 1.0 + 0.07 * math.sin(float(pulse_phase) * 2.0 * math.pi)
    box = max(48, int(round(68 * scale)))
    ss = box * 4
    sub = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(sub, "RGBA")
    m = ss * 0.5
    for mul, a_val in ((0.42 * pulse, 36), (0.31 * pulse, 78), (0.22, 128)):
        r_g = min(m - 2.0, ss * mul)
        d.ellipse([m - r_g, m - r_g, m + r_g, m + r_g], fill=(*glow_rgb, a_val))
    r_out = ss * 0.165
    d.ellipse([m - r_out, m - r_out, m + r_out, m + r_out], fill=(*stroke_rgb, 250))
    r_in = ss * 0.118
    d.ellipse([m - r_in, m - r_in, m + r_in, m + r_in], fill=(*core_rgb, 255))
    r_spec = ss * 0.046
    d.ellipse([m - r_spec, m - r_spec, m + r_spec, m + r_spec], fill=(255, 253, 246, 252))
    tile = sub.resize((box, box), Image.Resampling.LANCZOS)
    canvas.alpha_composite(tile, dest=(int(round(cx - box * 0.5)), int(round(cy - box * 0.5))))


def _draw_articulated_train(
    canvas: Image.Image,
    arc_path: _ArcLengthPath,
    u: float,
    fill_rgb: tuple[int, int, int],
    stroke_rgb: tuple[int, int, int],
    scale: float,
) -> None:
    """Render an articulated 3-car bullet train whose coaches follow curved railway geometry."""
    s_lead = max(0.0, min(arc_path.total_len, float(u) * arc_path.total_len))
    car_spacing = 30.0 * scale
    # Draw rear coach -> middle coach -> streamlined lead locomotive
    for car_idx in (2, 1, 0):
        s_car = s_lead - car_idx * car_spacing
        if s_car < 0.0 and car_idx > 0:
            continue
        s_car = max(0.0, s_car)
        cx, cy = arc_path.point_at_s(s_car)
        u_car = s_car / max(1e-6, arc_path.total_len)
        ang, _ = arc_path.tangent_and_curvature(u_car, window_px=14.0 * scale)
        sub_mode = "train" if car_idx == 0 else "train_coach"
        _draw_transport_icon(
            canvas,
            cx=cx,
            cy=cy,
            angle_rad=ang,
            mode=sub_mode,
            fill_rgb=fill_rgb,
            stroke_rgb=stroke_rgb,
            scale=scale * 0.94,
        )


def _draw_transport_icon(
    canvas: Image.Image,
    cx: float,
    cy: float,
    angle_rad: float,
    mode: str,
    fill_rgb: tuple[int, int, int],
    stroke_rgb: tuple[int, int, int],
    scale: float,
    bank_factor: float = 0.0,
) -> None:
    """Render a crisp 4x-supersampled multi-layer vehicle rotated along heading."""
    box = max(56, int(round(84 * scale)))
    ss = box * 4
    sub = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(sub, "RGBA")
    m = ss * 0.5
    r = ss * 0.34
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)

    def _rot(lx: float, ly: float, off_x: float = 0.0, off_y: float = 0.0) -> tuple[float, float]:
        return (m + off_x + lx * ca - ly * sa, m + off_y + lx * sa + ly * ca)

    if mode == "flight":
        # Widebody swept-wing jetliner pointing along +X (with turbofans, flaps, cockpit & nav lights)
        bank_sq = 1.0 - 0.16 * min(1.0, abs(float(bank_factor)))
        wing_y = r * 0.98 * bank_sq
        outline_c = (24, 34, 48, 238)
        lw_o = max(2, int(ss * 0.014))

        # Horizontal tailplane
        tailplane = [
            _rot(-r * 0.62, 0.0),
            _rot(-r * 0.92, wing_y * 0.43),
            _rot(-r * 1.03, wing_y * 0.38),
            _rot(-r * 0.84, 0.0),
            _rot(-r * 1.03, -wing_y * 0.38),
            _rot(-r * 0.92, -wing_y * 0.43),
        ]
        d.polygon(tailplane, fill=(224, 230, 238, 255), outline=outline_c, width=lw_o)

        # Swept main wings + trailing flap shadow
        main_wings = [
            _rot(r * 0.34, r * 0.14),
            _rot(-r * 0.24, wing_y),
            _rot(-r * 0.45, wing_y * 0.95),
            _rot(-r * 0.12, r * 0.16),
            _rot(-r * 0.12, -r * 0.16),
            _rot(-r * 0.45, -wing_y * 0.95),
            _rot(-r * 0.24, -wing_y),
            _rot(r * 0.34, -r * 0.14),
        ]
        d.polygon(main_wings, fill=(236, 240, 246, 255), outline=outline_c, width=lw_o)
        for sign in (-1.0, 1.0):
            flap = [
                _rot(-r * 0.14, sign * r * 0.18),
                _rot(-r * 0.42, sign * wing_y * 0.86),
                _rot(-r * 0.34, sign * wing_y * 0.86),
                _rot(-r * 0.06, sign * r * 0.18),
            ]
            d.polygon(flap, fill=(176, 188, 204, 215))

            # Under-wing turbofan engine pods
            eng = [
                _rot(r * 0.24, sign * r * 0.36 * bank_sq),
                _rot(r * 0.22, sign * r * 0.48 * bank_sq),
                _rot(-r * 0.04, sign * r * 0.47 * bank_sq),
                _rot(-r * 0.04, sign * r * 0.37 * bank_sq),
            ]
            d.polygon(eng, fill=(208, 216, 228, 255), outline=outline_c, width=max(1, lw_o // 2))
            intake = [
                _rot(r * 0.25, sign * r * 0.38 * bank_sq),
                _rot(r * 0.25, sign * r * 0.46 * bank_sq),
                _rot(r * 0.19, sign * r * 0.46 * bank_sq),
                _rot(r * 0.19, sign * r * 0.38 * bank_sq),
            ]
            d.polygon(intake, fill=(30, 40, 54, 245))

        # Tapered fuselage + dorsal specular highlight + livery stripe
        fuselage = [
            _rot(r * 1.08, 0.0),
            _rot(r * 0.84, r * 0.115),
            _rot(r * 0.48, r * 0.145),
            _rot(-r * 0.54, r * 0.135),
            _rot(-r * 0.98, r * 0.045),
            _rot(-r * 1.02, 0.0),
            _rot(-r * 0.98, -r * 0.045),
            _rot(-r * 0.54, -r * 0.135),
            _rot(r * 0.48, -r * 0.145),
            _rot(r * 0.84, -r * 0.115),
        ]
        d.polygon(fuselage, fill=(250, 252, 255, 255), outline=outline_c, width=lw_o)
        dorsal_hi = [
            _rot(r * 0.88, -r * 0.03),
            _rot(-r * 0.72, -r * 0.03),
            _rot(-r * 0.72, r * 0.02),
            _rot(r * 0.88, r * 0.02),
        ]
        d.polygon(dorsal_hi, fill=(255, 255, 255, 210))

        # Livery vertical tailfin accent
        tailfin = [_rot(-r * 0.64, 0.0), _rot(-r * 0.96, r * 0.035), _rot(-r * 1.02, 0.0), _rot(-r * 0.96, -r * 0.035)]
        d.polygon(tailfin, fill=(*fill_rgb, 245))

        # Cockpit wraparound glass
        cockpit = [
            _rot(r * 0.94, 0.0),
            _rot(r * 0.78, r * 0.075),
            _rot(r * 0.69, r * 0.065),
            _rot(r * 0.69, -r * 0.065),
            _rot(r * 0.78, -r * 0.075),
        ]
        d.polygon(cockpit, fill=(24, 38, 56, 248))

        # Wingtip nav strobes (port -Y red, starboard +Y green)
        nav_r = max(2.5, ss * 0.016)
        px_p, py_p = _rot(-r * 0.34, -wing_y * 0.96)
        px_s, py_s = _rot(-r * 0.34, wing_y * 0.96)
        d.ellipse([px_p - nav_r, py_p - nav_r, px_p + nav_r, py_p + nav_r], fill=(239, 68, 68, 255))
        d.ellipse([px_s - nav_r, py_s - nav_r, px_s + nav_r, py_s + nav_r], fill=(34, 197, 94, 255))

    elif mode in ("train", "train_coach"):
        # Streamlined Shinkansen/Eurostar coach (nose cone for lead, articulated flat ends for trailing coaches)
        outline_c = (24, 34, 48, 245)
        lw_o = max(2, int(ss * 0.016))
        sh_off_x, sh_off_y = ss * 0.022, ss * 0.026
        is_lead = mode == "train"
        body_pts = (
            [
                _rot(r * 0.98, 0.0),
                _rot(r * 0.72, r * 0.23),
                _rot(r * 0.42, r * 0.28),
                _rot(-r * 0.76, r * 0.28),
                _rot(-r * 0.80, 0.0),
                _rot(-r * 0.76, -r * 0.28),
                _rot(r * 0.42, -r * 0.28),
                _rot(r * 0.72, -r * 0.23),
            ]
            if is_lead
            else [
                _rot(r * 0.78, r * 0.27),
                _rot(-r * 0.78, r * 0.27),
                _rot(-r * 0.78, -r * 0.27),
                _rot(r * 0.78, -r * 0.27),
            ]
        )
        # Tight railbed shadow
        sh_im = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
        sh_d = ImageDraw.Draw(sh_im, "RGBA")
        sh_poly = [(px + sh_off_x, py + sh_off_y) for px, py in body_pts]
        sh_d.polygon(sh_poly, fill=(10, 14, 22, 118))
        sh_im = sh_im.filter(ImageFilter.GaussianBlur(radius=ss * 0.025))
        sub.alpha_composite(sh_im)
        d = ImageDraw.Draw(sub, "RGBA")

        d.polygon(body_pts, fill=(246, 249, 252, 255), outline=outline_c, width=lw_o)
        # Roof accent livery stripe & recessed side window bands
        x_lo, x_hi = (-r * 0.70, r * 0.56) if is_lead else (-r * 0.72, r * 0.72)
        d.polygon([_rot(x_hi, -r * 0.14), _rot(x_hi, r * 0.14), _rot(x_lo, r * 0.14), _rot(x_lo, -r * 0.14)], fill=(*fill_rgb, 232))
        d.polygon([_rot(x_hi * 0.86, -r * 0.06), _rot(x_hi * 0.86, r * 0.06), _rot(x_lo * 0.86, r * 0.06), _rot(x_lo * 0.86, -r * 0.06)], fill=(245, 248, 252, 195))
        if is_lead:
            canopy = [_rot(r * 0.82, 0.0), _rot(r * 0.58, r * 0.16), _rot(r * 0.42, r * 0.16), _rot(r * 0.42, -r * 0.16), _rot(r * 0.58, -r * 0.16)]
            d.polygon(canopy, fill=(24, 36, 52, 246))
            for hs in (-1.0, 1.0):
                d.ellipse([_rot(r * 0.88, hs * r * 0.09)[0] - nav_r_t(ss), _rot(r * 0.88, hs * r * 0.09)[1] - nav_r_t(ss),
                           _rot(r * 0.88, hs * r * 0.09)[0] + nav_r_t(ss), _rot(r * 0.88, hs * r * 0.09)[1] + nav_r_t(ss)], fill=(255, 236, 156, 255))

    elif mode == "boat":
        # Sculpted coastal yacht cruiser with white stepped bow, warm teak deck & tinted bridge glass
        outline_c = (20, 32, 46, 245)
        lw_o = max(2, int(ss * 0.016))
        hull = [
            _rot(r * 0.98, 0.0),
            _rot(r * 0.52, r * 0.29),
            _rot(-r * 0.18, r * 0.34),
            _rot(-r * 0.82, r * 0.27),
            _rot(-r * 0.84, 0.0),
            _rot(-r * 0.82, -r * 0.27),
            _rot(-r * 0.18, -r * 0.34),
            _rot(r * 0.52, -r * 0.29),
        ]
        d.polygon(hull, fill=(248, 250, 252, 255), outline=outline_c, width=lw_o)
        # Warm teak aft cockpit & bow walkway trim
        teak_aft = [_rot(-r * 0.22, -r * 0.22), _rot(-r * 0.22, r * 0.22), _rot(-r * 0.74, r * 0.19), _rot(-r * 0.74, -r * 0.19)]
        d.polygon(teak_aft, fill=(192, 144, 92, 248))
        # Cabin superstructure + tinted bridge windscreen
        cabin = [_rot(r * 0.56, 0.0), _rot(r * 0.28, r * 0.22), _rot(-r * 0.24, r * 0.22), _rot(-r * 0.24, -r * 0.22), _rot(r * 0.28, -r * 0.22)]
        d.polygon(cabin, fill=(238, 242, 246, 255), outline=(170, 182, 196, 220))
        bridge_glass = [_rot(r * 0.42, 0.0), _rot(r * 0.24, r * 0.18), _rot(r * 0.10, r * 0.18), _rot(r * 0.10, -r * 0.18), _rot(r * 0.24, -r * 0.18)]
        d.polygon(bridge_glass, fill=(22, 48, 72, 245))
        # Radar arch stripe
        d.line([_rot(-r * 0.08, -r * 0.20), _rot(-r * 0.08, r * 0.20)], fill=(*fill_rgb, 240), width=max(2, int(ss * 0.022)))

    elif mode == "walk":
        # Crisp directional GPS radar beacon with forward view-cone wedge
        cone = [_rot(0.0, 0.0), _rot(r * 1.15, -r * 0.62), _rot(r * 1.15, r * 0.62)]
        d.polygon(cone, fill=(*fill_rgb, 82))
        pr_o = r * 0.42
        pr_i = r * 0.28
        d.ellipse([m - pr_o, m - pr_o, m + pr_o, m + pr_o], fill=(255, 255, 255, 248), outline=(24, 34, 48, 220), width=max(2, int(ss * 0.015)))
        d.ellipse([m - pr_i, m - pr_i, m + pr_i, m + pr_i], fill=(*fill_rgb, 255))

    else:
        # Sculpted pearl-white Touring SUV / Car with directional ground shadow (no grey circle puck)
        outline_c = (22, 30, 44, 255)
        lw_o = max(2, int(ss * 0.020))

        # Tight chassis-shaped directional drop shadow
        chassis_pts = [
            _rot(r * 0.96, -r * 0.24),
            _rot(r * 1.02, 0.0),
            _rot(r * 0.96, r * 0.24),
            _rot(r * 0.56, r * 0.38),
            _rot(-r * 0.80, r * 0.37),
            _rot(-r * 0.94, r * 0.25),
            _rot(-r * 0.94, -r * 0.25),
            _rot(-r * 0.80, -r * 0.37),
            _rot(r * 0.56, -r * 0.38),
        ]
        sh_im = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
        sh_d = ImageDraw.Draw(sh_im, "RGBA")
        sh_off_x, sh_off_y = ss * 0.024, ss * 0.030
        sh_d.polygon([(px + sh_off_x, py + sh_off_y) for px, py in chassis_pts], fill=(8, 12, 20, 128))
        sh_im = sh_im.filter(ImageFilter.GaussianBlur(radius=ss * 0.030))
        sub.alpha_composite(sh_im)
        d = ImageDraw.Draw(sub, "RGBA")

        # Soft layered LED headlight beams
        for alpha_f, reach, spread in ((42, 1.56, 0.44), (72, 1.24, 0.34)):
            beam_l = [_rot(r * 0.88, -r * 0.22), _rot(r * reach, -r * spread), _rot(r * reach, -r * 0.05)]
            beam_r = [_rot(r * 0.88, r * 0.22), _rot(r * reach, r * 0.05), _rot(r * reach, r * spread)]
            d.polygon(beam_l, fill=(255, 244, 190, alpha_f))
            d.polygon(beam_r, fill=(255, 244, 190, alpha_f))

        # 4 dark tucked tires
        tire_col = (22, 26, 34, 245)
        for tx, ty in ((r * 0.50, -r * 0.42), (r * 0.50, r * 0.42), (-r * 0.48, -r * 0.42), (-r * 0.48, r * 0.42)):
            tire = [
                _rot(tx - r * 0.16, ty - r * 0.065),
                _rot(tx + r * 0.16, ty - r * 0.065),
                _rot(tx + r * 0.16, ty + r * 0.065),
                _rot(tx - r * 0.16, ty + r * 0.065),
            ]
            d.polygon(tire, fill=tire_col)

        # Side mirrors
        for my_sign in (-1.0, 1.0):
            mir = [
                _rot(r * 0.27, my_sign * r * 0.37),
                _rot(r * 0.17, my_sign * r * 0.51),
                _rot(r * 0.09, my_sign * r * 0.48),
                _rot(r * 0.16, my_sign * r * 0.37),
            ]
            d.polygon(mir, fill=(242, 245, 250, 255), outline=outline_c, width=max(1, lw_o // 2))

        # Sculpted pearl-white SUV body + hood crease lines
        d.polygon(chassis_pts, fill=(249, 250, 253, 255), outline=outline_c, width=lw_o)
        for crease_y in (-r * 0.16, r * 0.16):
            d.line([_rot(r * 0.88, crease_y * 0.82), _rot(r * 0.48, crease_y)], fill=(204, 212, 224, 215), width=max(2, int(ss * 0.010)))

        # Tinted wraparound windshield + panoramic glass roof + diagonal sky reflection
        windshield = [
            _rot(r * 0.45, -r * 0.29),
            _rot(r * 0.53, 0.0),
            _rot(r * 0.45, r * 0.29),
            _rot(r * 0.16, r * 0.26),
            _rot(r * 0.16, -r * 0.26),
        ]
        d.polygon(windshield, fill=(24, 38, 54, 246))
        glint = [_rot(r * 0.43, -r * 0.18), _rot(r * 0.47, -r * 0.02), _rot(r * 0.21, -r * 0.14), _rot(r * 0.19, -r * 0.23)]
        d.polygon(glint, fill=(165, 200, 236, 92))
        sunroof = [
            _rot(r * 0.08, -r * 0.24),
            _rot(r * 0.08, r * 0.24),
            _rot(-r * 0.53, r * 0.24),
            _rot(-r * 0.53, -r * 0.24),
        ]
        d.polygon(sunroof, fill=(32, 48, 68, 236))
        rear_glass = [
            _rot(-r * 0.61, -r * 0.25),
            _rot(-r * 0.61, r * 0.25),
            _rot(-r * 0.80, r * 0.23),
            _rot(-r * 0.80, -r * 0.23),
        ]
        d.polygon(rear_glass, fill=(24, 38, 54, 246))

        # Crisp LED headlights & ruby taillight bars
        for hy_sign in (-1.0, 1.0):
            d.polygon(
                [_rot(r * 0.90, hy_sign * r * 0.15), _rot(r * 0.99, hy_sign * r * 0.23), _rot(r * 0.86, hy_sign * r * 0.29)],
                fill=(255, 232, 140, 255),
            )
            d.polygon(
                [_rot(-r * 0.88, hy_sign * r * 0.15), _rot(-r * 0.93, hy_sign * r * 0.25), _rot(-r * 0.84, hy_sign * r * 0.29)],
                fill=(228, 42, 46, 255),
            )

    down = sub.resize((box, box), Image.Resampling.LANCZOS)
    canvas.alpha_composite(down, (int(round(cx - box * 0.5)), int(round(cy - box * 0.5))))


def nav_r_t(ss: int) -> float:
    return max(2.0, ss * 0.015)


def _load_clean_callout_rgb(path: Path) -> Image.Image:
    """Load an inset map photo with HEIC support, Display-P3->sRGB ICC conversion, and gentle highlight knee."""
    if path.suffix.lower() in (".heic", ".heif"):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass
    with Image.open(path) as raw:
        im = ImageOps.exif_transpose(raw)
        icc = im.info.get("icc_profile")
        if icc:
            try:
                src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                dst_prof = ImageCms.createProfile("sRGB")
                im = ImageCms.profileToProfile(im.convert("RGB"), src_prof, dst_prof, outputMode="RGB")
            except Exception:
                im = im.convert("RGB")
        else:
            im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.float32)
    # Soft photographic shoulder above 208 so inset photos never clip white highlights when composited onto HDR timelines
    arr = np.where(arr > 208.0, 208.0 + (arr - 208.0) * 0.48, arr)
    return Image.fromarray(np.clip(np.round(arr), 0, 232).astype(np.uint8), mode="RGB")


def _polaroid_candidate_slots(
    dx: float,
    dy: float,
    card_w: float,
    card_h: float,
    s: float,
) -> list[tuple[float, float]]:
    """Canonical 8-quadrant Polaroid placement candidates anchored cleanly around a map pin."""
    margin = int(round(32 * s))
    return [
        (dx - card_w - margin, dy + int(round(34 * s))),                   # 0: Lower-left
        (dx + margin, dy + int(round(34 * s))),                            # 1: Lower-right
        (dx - card_w - margin, dy - card_h - int(round(30 * s))),          # 2: Upper-left
        (dx + margin, dy - card_h - int(round(30 * s))),                   # 3: Upper-right
        (dx - card_w * 0.50, dy + int(round(62 * s))),                     # 4: Directly below
        (dx - card_w * 0.50, dy - card_h - int(round(58 * s))),            # 5: Directly above
        (dx - card_w - int(round(52 * s)), dy - card_h * 0.50),            # 6: Due left
        (dx + int(round(52 * s)), dy - card_h * 0.50),                     # 7: Due right
    ]


def _score_rect_against_obstacles(
    px_c: float,
    py_c: float,
    rw: float,
    rh: float,
    world_w: int,
    world_h: int,
    s: float,
    others: list[tuple[float, float]],
    rects: list[tuple[float, float, float, float]],
    route_pts: list[tuple[float, float]],
) -> float:
    """Score a candidate card/callout rectangle: strongly avoids HUD, labels, pins, edges, and route line."""
    pad_rect = 14.0 * s
    cx_mid = px_c + rw * 0.5
    cy_mid = py_c + rh * 0.5
    min_d = min((math.hypot(cx_mid - ax, cy_mid - ay) for ax, ay in others), default=9999.0)
    ov_area = 0.0
    for rx0, ry0, rx1, ry1 in rects:
        ix = max(0.0, min(px_c + rw + pad_rect, rx1) - max(px_c - pad_rect, rx0))
        iy = max(0.0, min(py_c + rh + pad_rect, ry1) - max(py_c - pad_rect, ry0))
        ov_area += ix * iy
    # Penalize any active route spline point passing inside or right beside the card!
    route_pen = 0.0
    route_pad = 16.0 * s
    for rpx, rpy in route_pts:
        if (px_c - route_pad) <= rpx <= (px_c + rw + route_pad) and (py_c - route_pad) <= rpy <= (py_c + rh + route_pad):
            # Stronger penalty deeper inside the card
            dx_in = min(rpx - (px_c - route_pad), (px_c + rw + route_pad) - rpx)
            dy_in = min(rpy - (py_c - route_pad), (py_c + rh + route_pad) - rpy)
            route_pen += 900.0 + min(dx_in, dy_in) * 28.0
    # Penalize clipping against frame bounds
    edge_pen = 0.0
    if px_c < 18.0 * s or (px_c + rw) > (world_w - 18.0 * s):
        edge_pen += 6000.0
    if py_c < world_h * 0.18 or (py_c + rh) > (world_h - 18.0 * s):
        edge_pen += 6000.0
    return min_d - ov_area * 3.0 - route_pen - edge_pen


def _paste_hero_photo_callout(
    world_rgba: Image.Image,
    dest_px: tuple[float, float],
    thumb_path: Optional[str],
    label: str,
    bloom_p: float,
    spec: dict[str, Any],
    world_w: int,
    world_h: int,
    avoid_pts: Optional[list[tuple[float, float]]] = None,
    caption_prefix: str = "TO",
    tilt_deg: float = 2.2,
    accent_override: Optional[tuple[int, int, int]] = None,
    avoid_rects: Optional[list[tuple[float, float, float, float]]] = None,
    route_avoid_pts: Optional[list[tuple[float, float]]] = None,
    locked_slot_idx: Optional[int] = None,
    preloaded_thumb_rgb: Optional[Image.Image] = None,
) -> Optional[tuple[float, float]]:
    """Draw a tilted gallery photo card with caption & pin leader beside an origin/destination stop."""
    if bloom_p <= 0.01:
        return None
    if preloaded_thumb_rgb is not None:
        im = preloaded_thumb_rgb
    else:
        if not thumb_path:
            return None
        p = Path(thumb_path)
        if not p.exists():
            return None
        try:
            im = _load_clean_callout_rgb(p)
        except Exception:
            return None

    s = world_w / 1920.0
    accent_rgb = accent_override or spec["accent"]
    bloom_eased = _ease_in_out_cubic(min(1.0, max(0.0, float(bloom_p))))
    card_w = int(round(256 * s * (0.90 + 0.10 * bloom_eased)))
    img_h = int(round(card_w * 0.60))
    cap_h = max(14, int(round(24 * s)))
    card_h = img_h + cap_h + int(round(12 * s))
    pad = max(6, int(round(8 * s)))
    inner_w = max(24, card_w - pad * 2)
    inner_h = max(20, img_h - pad)
    im = ImageOps.fit(im, (inner_w, inner_h), method=Image.Resampling.LANCZOS)

    alpha_f = max(0.0, min(1.0, float(bloom_p)))
    matte_bg = (*spec["pin_fill"], int(round(244 * alpha_f))) if not spec["light_paper"] else (248, 245, 238, int(round(246 * alpha_f)))
    border_c = (*accent_rgb, int(round(225 * alpha_f)))
    card = Image.new("RGBA", (card_w + 18, card_h + 18), (0, 0, 0, 0))
    dc = ImageDraw.Draw(card, "RGBA")
    # Soft layered gallery drop shadow
    dc.rounded_rectangle([11, 12, card_w + 11, card_h + 12], radius=max(4, int(round(8 * s))), fill=(6, 10, 16, int(round(105 * alpha_f))))
    dc.rounded_rectangle([9, 10, card_w + 9, card_h + 10], radius=max(4, int(round(8 * s))), fill=(6, 10, 16, int(round(75 * alpha_f))))
    dc.rounded_rectangle([4, 4, card_w + 4, card_h + 4], radius=max(4, int(round(8 * s))), fill=matte_bg, outline=border_c, width=max(1, int(round(2 * s))))
    im_rgba = im.convert("RGBA")
    if alpha_f < 0.99:
        alpha = im_rgba.getchannel("A").point(lambda v: int(round(v * alpha_f)))
        im_rgba.putalpha(alpha)
    card.alpha_composite(im_rgba, (4 + pad, 4 + pad))

    f_cap = _load_font_tier(max(9, int(round(12 * s))), tier="mono", bold=True)
    cap_txt = f"{caption_prefix} · {label}"[:28].upper()
    cap_col = (*spec["hud_title"], int(round(245 * alpha_f)))
    dc.text((4 + pad + 2, 4 + pad + inner_h + max(3, int(round(5 * s)))), cap_txt, font=f_cap, fill=cap_col)

    card = card.rotate(tilt_deg, resample=Image.Resampling.BICUBIC, expand=True)
    dx, dy = dest_px
    cand_slots = _polaroid_candidate_slots(dx, dy, float(card.width), float(card.height), s)
    others = list(avoid_pts or [])
    rects = list(avoid_rects or [])
    route_pts = list(route_avoid_pts or [])

    if locked_slot_idx is not None and 0 <= locked_slot_idx < len(cand_slots):
        cx_raw, cy_raw = cand_slots[locked_slot_idx]
        place_x = int(round(max(14.0, min(float(world_w - card.width - 14), cx_raw))))
        place_y = int(round(max(float(world_h * 0.19), min(float(world_h - card.height - 14), cy_raw))))
    else:
        best_xy: Optional[tuple[int, int]] = None
        best_score = -1e12
        for cx_raw, cy_raw in cand_slots:
            px_c = int(round(max(14.0, min(float(world_w - card.width - 14), cx_raw))))
            py_c = int(round(max(float(world_h * 0.19), min(float(world_h - card.height - 14), cy_raw))))
            score = _score_rect_against_obstacles(
                float(px_c), float(py_c), float(card.width), float(card.height),
                world_w, world_h, s, others, rects, route_pts,
            )
            if score > best_score:
                best_score = score
                best_xy = (px_c, py_c)
        assert best_xy is not None
        place_x, place_y = best_xy

    # Leader connector hairline linking the Polaroid corner back to the exact map pin
    wd_conn = ImageDraw.Draw(world_rgba, "RGBA")
    anc_x = max(float(place_x + 12), min(float(place_x + card.width - 12), dx))
    anc_y = max(float(place_y + 12), min(float(place_y + card.height - 12), dy))
    wd_conn.line([(dx, dy), (anc_x, anc_y)], fill=(250, 248, 242, int(round(205 * alpha_f))), width=max(2, int(round(3.5 * s))))
    wd_conn.line([(dx, dy), (anc_x, anc_y)], fill=(*accent_rgb, int(round(235 * alpha_f))), width=max(1, int(round(1.8 * s))))
    r_dot = max(2, int(round(3.5 * s)))
    wd_conn.ellipse([anc_x - r_dot, anc_y - r_dot, anc_x + r_dot, anc_y + r_dot], fill=(*accent_rgb, int(round(240 * alpha_f))))

    world_rgba.alpha_composite(card, (place_x, place_y))
    if avoid_rects is not None:
        avoid_rects.append((float(place_x), float(place_y), float(place_x + card.width), float(place_y + card.height)))
    return (place_x + card.width * 0.5, place_y + card.height * 0.5)


def _draw_route_annotations_screen(
    out_rgba: Image.Image,
    annotations: Optional[list[dict[str, Any]]],
    vp: _MapViewport,
    proj_fn: Any,
    u: float,
    spec: dict[str, Any],
    width: int,
    height: int,
    occupied_rects: list[tuple[float, float, float, float]],
    route_avoid_pts: Optional[list[tuple[float, float]]] = None,
    locked_offsets: Optional[dict[int, tuple[float, float]]] = None,
) -> None:
    """Draw razor-sharp post-warp screen-space corridor/landmark callouts with collision avoidance."""
    if not annotations:
        return
    ann_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    wd = ImageDraw.Draw(ann_layer, "RGBA")
    s = width / 1920.0
    f_t = _load_font_tier(max(11, int(round(15 * s))), tier=spec["title_tier"], bold=True)
    f_s = _load_font_tier(max(9, int(round(11 * s))), tier="mono", bold=True)
    route_pts = list(route_avoid_pts or [])
    shadow_rgb = (68, 54, 38) if spec.get("light_paper") else (6, 10, 16)
    drawn_any = False

    for idx_a, ann in enumerate(annotations):
        reveal_u = float(ann.get("reveal_u", ann.get("u_min", 0.0)))
        hide_u = float(ann.get("u_max", 1.15))
        if u < reveal_u:
            continue
        raw_in = min(1.0, (u - reveal_u) / 0.14) if reveal_u > 0.0 else 1.0
        raw_out = 1.0 if u <= hide_u else max(0.0, 1.0 - (u - hide_u) / 0.14)
        fade = _ease_in_out_cubic(raw_in) * _ease_in_out_cubic(raw_out)
        if fade <= 0.04 or "lat" not in ann or "lon" not in ann:
            continue
        wx, wy = vp.latlon_to_px(float(ann["lat"]), float(ann["lon"]))
        ax, ay = proj_fn(wx, wy)
        if ax < -40 or ax > width + 40 or ay < height * 0.12 or ay > height + 40:
            continue
        title = str(ann.get("title") or "").strip()
        sub = str(ann.get("sub") or "").strip()
        if not title:
            continue
        bb_t = wd.textbbox((0, 0), title, font=f_t)
        wt, ht = bb_t[2] - bb_t[0], bb_t[3] - bb_t[1]
        ws, hs = (0, 0)
        if sub:
            bb_s = wd.textbbox((0, 0), sub, font=f_s)
            ws, hs = bb_s[2] - bb_s[0], bb_s[3] - bb_s[1]
        bw = max(wt, ws) + int(round(22 * s))
        bh = ht + (hs + int(round(10 * s)) if sub else 0) + int(round(15 * s))

        # Use locked per-leg offset when precomputed; else evaluate caller offset + 6 cardinal fallbacks
        if locked_offsets is not None and idx_a in locked_offsets:
            dx, dy = locked_offsets[idx_a]
        else:
            raw_off = ann.get("offset")
            cand_offs: list[tuple[float, float]] = []
            if raw_off is not None and len(raw_off) >= 2:
                # Scale caller offsets (authored in ~1920p space) cleanly to current screen scale
                cand_offs.append((float(raw_off[0]) * (s / max(0.5, 1.333 if abs(float(raw_off[0])) > 120 else 1.0)), float(raw_off[1]) * (s / max(0.5, 1.333 if abs(float(raw_off[1])) > 120 else 1.0))))
            cand_offs.extend([
                (22 * s, -bh - 20 * s),
                (-bw - 22 * s, -bh - 20 * s),
                (22 * s, 20 * s),
                (-bw - 22 * s, 20 * s),
                (-bw * 0.5, -bh - 26 * s),
                (-bw * 0.5, 26 * s),
            ])
            dx, dy = cand_offs[0]
            best_sc = -1e12
            for ox_c, oy_c in cand_offs:
                tx_c = max(16.0, min(float(width - bw - 16), ax + ox_c))
                ty_c = max(float(height * 0.19), min(float(height - bh - 16), ay + oy_c))
                sc = _score_rect_against_obstacles(tx_c, ty_c, float(bw), float(bh), width, height, s, [], occupied_rects, route_pts)
                if sc > best_sc:
                    best_sc = sc
                    dx, dy = ox_c, oy_c

        rise_px = (1.0 - fade) * 7.0 * s
        tx = int(round(max(16.0, min(float(width - bw - 16), ax + dx))))
        ty = int(round(max(float(height * 0.19), min(float(height - bh - 16), ay + dy + rise_px))))
        bx = max(float(tx + 8), min(float(tx + bw - 8), ax))
        by = max(float(ty + 8), min(float(ty + bh - 8), ay))
        wd.line([(ax, ay), (bx, by)], fill=(250, 248, 242, int(round(165 * fade))), width=max(2, int(round(2.8 * s))))
        wd.line([(ax, ay), (bx, by)], fill=(*spec["hud_border"][:3], int(round(205 * fade))), width=max(1, int(round(1.5 * s))))
        r_a = max(3, int(round(4.5 * s)))
        wd.ellipse([ax - r_a - 2, ay - r_a - 2, ax + r_a + 2, ay + r_a + 2], fill=(*spec["accent"], int(round(92 * fade))))
        wd.ellipse([ax - r_a, ay - r_a, ax + r_a, ay + r_a], fill=(*spec["pin_fill"], int(round(245 * fade))), outline=(*spec["accent"], int(round(240 * fade))), width=max(1, int(round(2 * s))))
        # Soft translucent shadow + frosted museum pill
        wd.rounded_rectangle([tx + 2, ty + 4, tx + bw + 3, ty + bh + 5], radius=max(4, int(round(7 * s))), fill=(*shadow_rgb, int(round(48 * fade))))
        wd.rounded_rectangle([tx, ty, tx + bw, ty + bh], radius=max(4, int(round(7 * s))), fill=(*spec["hud_fill"][:3], int(round(234 * fade))), outline=(*spec["hud_border"][:3], int(round(168 * fade))), width=max(1, int(round(1.4 * s))))
        wd.text((tx + int(round(11 * s)), ty + int(round(6 * s))), title, font=f_t, fill=(*spec["hud_title"], int(round(252 * fade))))
        if sub:
            wd.text((tx + int(round(11 * s)), ty + int(round(6 * s)) + ht + int(round(7 * s))), sub, font=f_s, fill=(*spec["hud_metric"], int(round(238 * fade))))
        occupied_rects.append((float(tx), float(ty), float(tx + bw), float(ty + bh)))
        drawn_any = True
    if drawn_any:
        out_rgba.alpha_composite(ann_layer)


def _draw_route_annotations(
    world_rgba: Image.Image,
    annotations: Optional[list[dict[str, Any]]],
    vp: _MapViewport,
    u: float,
    spec: dict[str, Any],
    world_w: int,
    world_h: int,
) -> None:
    """Draw optional corridor / landmark callout badges along the active travel route."""
    if not annotations:
        return
    wd = ImageDraw.Draw(world_rgba, "RGBA")
    s = world_w / 1920.0
    f_t = _load_font_tier(max(10, int(round(14 * s))), tier=spec["title_tier"], bold=True)
    f_s = _load_font_tier(max(8, int(round(11 * s))), tier="mono", bold=True)
    for ann in annotations:
        reveal_u = float(ann.get("reveal_u", 0.0))
        if u < reveal_u:
            continue
        fade = min(1.0, (u - reveal_u) / 0.16) if reveal_u > 0 else 1.0
        if fade <= 0.05 or "lat" not in ann or "lon" not in ann:
            continue
        ax, ay = vp.latlon_to_px(float(ann["lat"]), float(ann["lon"]))
        dx, dy = ann.get("offset", (14 * s, -48 * s))
        title = str(ann.get("title") or "").strip()
        sub = str(ann.get("sub") or "").strip()
        if not title:
            continue
        bb_t = wd.textbbox((0, 0), title, font=f_t)
        wt, ht = bb_t[2] - bb_t[0], bb_t[3] - bb_t[1]
        ws, hs = (0, 0)
        if sub:
            bb_s = wd.textbbox((0, 0), sub, font=f_s)
            ws, hs = bb_s[2] - bb_s[0], bb_s[3] - bb_s[1]
        bw = max(wt, ws) + int(round(20 * s))
        bh = ht + (hs + int(round(10 * s)) if sub else 0) + int(round(14 * s))
        tx = int(round(max(16, min(world_w - bw - 16, ax + float(dx)))))
        ty = int(round(max(16, min(world_h - bh - 16, ay + float(dy)))))
        bx = tx + (bw if float(dx) < 0 else 0)
        by = ty + bh * 0.5
        wd.line([(ax, ay), (bx, by)], fill=(*spec["hud_border"][:3], int(round(200 * fade))), width=max(1, int(round(2 * s))))
        r_a = max(3, int(round(4 * s)))
        wd.ellipse([ax - r_a, ay - r_a, ax + r_a, ay + r_a], fill=(*spec["pin_fill"], int(round(235 * fade))), outline=(*spec["accent"], int(round(235 * fade))), width=2)
        wd.rounded_rectangle([tx, ty, tx + bw, ty + bh], radius=max(4, int(round(6 * s))), fill=(*spec["hud_fill"][:3], int(round(228 * fade))), outline=(*spec["hud_border"][:3], int(round(170 * fade))), width=1)
        wd.text((tx + int(round(10 * s)), ty + int(round(6 * s))), title, font=f_t, fill=(*spec["hud_title"], int(round(248 * fade))))
        if sub:
            wd.text((tx + int(round(10 * s)), ty + int(round(6 * s)) + ht + int(round(7 * s))), sub, font=f_s, fill=(*spec["hud_metric"], int(round(235 * fade))))


_CPU_GLOBE_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
_CPU_GLOBE_GRID_LOCK = threading.Lock()
_CUDA_MAP_LOCK = threading.Lock()
_CUDA_MAP_GRID_CACHE: dict[tuple[int, int], tuple[Any, Any, Any, Any, Any, Any]] = {}


def _get_cpu_globe_grid(
    out_w: int,
    out_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = (out_w, out_h)
    cached = _CPU_GLOBE_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    with _CPU_GLOBE_GRID_LOCK:
        cached = _CPU_GLOBE_GRID_CACHE.get(key)
        if cached is not None:
            return cached
        gx_grid = np.arange(out_w, dtype=np.float32)[None, :].repeat(out_h, axis=0)
        gy_grid = np.arange(out_h, dtype=np.float32)[:, None].repeat(out_w, axis=1)
        nx_norm = (2.0 * gx_grid / max(1.0, float(out_w - 1))) - 1.0
        scale_h = max(0.5, out_h / 1080.0)
        rng = np.random.default_rng(1972)
        n_stars = max(80, int(140 * scale_h))
        star_xs = rng.integers(4, max(5, out_w - 4), size=n_stars)
        star_ys = rng.integers(2, max(4, int(out_h * 0.24)), size=n_stars)
        star_b = rng.uniform(95.0, 205.0, size=n_stars).astype(np.float32)
        val = (gx_grid, gy_grid, nx_norm, star_xs, star_ys, star_b)
        if len(_CPU_GLOBE_GRID_CACHE) >= 3:
            _CPU_GLOBE_GRID_CACHE.clear()
        _CPU_GLOBE_GRID_CACHE[key] = val
        return val


def _cuda_map_enabled() -> bool:
    try:
        from ..hardware import select_torch_device

        return select_torch_device() == "cuda"
    except Exception:
        return False


def _get_cuda_map_grid(out_w: int, out_h: int) -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch

    key = (out_w, out_h)
    cached = _CUDA_MAP_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_CUDA_MAP_GRID_CACHE) >= 2:
        _CUDA_MAP_GRID_CACHE.clear()
    gx_1d = torch.arange(out_w, device="cuda", dtype=torch.float32).view(1, out_w)
    gy_1d = torch.arange(out_h, device="cuda", dtype=torch.float32).view(out_h, 1)
    gx_g = gx_1d.expand(out_h, out_w)
    gy_g = gy_1d.expand(out_h, out_w)
    nx_g = (2.0 * gx_g / max(1.0, float(out_w - 1))) - 1.0
    _, _, _, s_xs, s_ys, s_b = _get_cpu_globe_grid(out_w, out_h)
    val = (
        gx_g,
        gy_g,
        nx_g,
        torch.from_numpy(np.ascontiguousarray(s_xs, dtype=np.int64)).to("cuda"),
        torch.from_numpy(np.ascontiguousarray(s_ys, dtype=np.int64)).to("cuda"),
        torch.from_numpy(np.ascontiguousarray(s_b, dtype=np.float32)).to("cuda"),
    )
    _CUDA_MAP_GRID_CACHE[key] = val
    return val


def _upload_world_to_cuda(arr: np.ndarray) -> Optional[Any]:
    """Upload static world RGB array (H, W, 3) once to CUDA VRAM as (1, 3, H, W) float32 tensor."""
    if not _cuda_map_enabled():
        return None
    try:
        import torch

        with _CUDA_MAP_LOCK, torch.no_grad():
            return (
                torch.from_numpy(np.array(arr, dtype=np.uint8, copy=True))
                .to("cuda", dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous()
            )
    except Exception:
        return None


def _cuda_warp_flat(
    arr: np.ndarray,
    h_inv: np.ndarray,
    out_w: int,
    out_h: int,
    src_cuda_t: Optional[Any] = None,
    vignette_strength: float = 0.0,
    sharpen_amount: float = 0.0,
) -> Optional[np.ndarray]:
    if not _cuda_map_enabled() or (out_w * out_h) < 400_000:
        return None
    try:
        import torch
        import torch.nn.functional as F

        wh, ww = int(arr.shape[0]), int(arr.shape[1])
        with _CUDA_MAP_LOCK, torch.no_grad():
            gx_g, gy_g, nx_g, _, _, _ = _get_cuda_map_grid(out_w, out_h)
            h00, h01, h02 = float(h_inv[0, 0]), float(h_inv[0, 1]), float(h_inv[0, 2])
            h10, h11, h12 = float(h_inv[1, 0]), float(h_inv[1, 1]), float(h_inv[1, 2])
            h20, h21, h22 = float(h_inv[2, 0]), float(h_inv[2, 1]), float(h_inv[2, 2])

            denom = h20 * gx_g + h21 * gy_g + h22
            denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
            map_x = (h00 * gx_g + h01 * gy_g + h02) / denom
            map_y = (h10 * gx_g + h11 * gy_g + h12) / denom
            norm_x = (2.0 * map_x / max(1.0, float(ww - 1))) - 1.0
            norm_y = (2.0 * map_y / max(1.0, float(wh - 1))) - 1.0
            grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)

            owned_src = src_cuda_t is None
            src_t = (
                src_cuda_t
                if src_cuda_t is not None
                else torch.from_numpy(np.array(arr, dtype=np.uint8, copy=True))
                .to("cuda", dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            warped = F.grid_sample(src_t, grid, mode="bicubic", padding_mode="border", align_corners=True)
            if sharpen_amount > 1e-3:
                blurred = F.avg_pool2d(warped, kernel_size=3, stride=1, padding=1)
                warped = warped + (warped - blurred) * float(sharpen_amount)
                del blurred
            if vignette_strength > 1e-3:
                ny_g = (2.0 * gy_g / max(1.0, float(out_h - 1))) - 1.0
                r2 = nx_g * nx_g * 0.72 + ny_g * ny_g * 0.95
                vig = torch.clamp(1.0 - float(vignette_strength) * torch.clamp(r2 - 0.35, min=0.0), 0.72, 1.0)
                warped = warped * vig.view(1, 1, out_h, out_w)
                del ny_g, r2, vig
            out_u8 = warped.squeeze(0).permute(1, 2, 0).clamp_(0.0, 255.0).round().to(torch.uint8).cpu().numpy()
            del denom, map_x, map_y, norm_x, norm_y, grid, warped
            if owned_src:
                del src_t
            return out_u8
    except Exception:
        return None


def _cuda_warp_globe(
    arr: np.ndarray,
    h_inv: np.ndarray,
    out_w: int,
    out_h: int,
    g_str: float,
    y_h0: float,
    curve_drop: float,
    bank_slope: float,
    src_cuda_t: Optional[Any] = None,
) -> Optional[np.ndarray]:
    if not _cuda_map_enabled() or (out_w * out_h) < 400_000:
        return None
    try:
        import torch
        import torch.nn.functional as F

        wh, ww = int(arr.shape[0]), int(arr.shape[1])
        scale_h = max(0.5, out_h / 1080.0)
        with _CUDA_MAP_LOCK, torch.no_grad():
            gx_g, gy_g, nx_g, star_xs, star_ys, star_b = _get_cuda_map_grid(out_w, out_h)
            y_limb = y_h0 + curve_drop * (nx_g * nx_g) + bank_slope * nx_g
            span_h = torch.clamp(float(out_h) - y_limb, min=1.0)
            t_y = torch.clamp((gy_g - y_limb) / span_h, 0.0, 1.0)
            sy_planar = t_y * float(out_h)
            sx_planar = out_w * 0.5 + (gx_g - out_w * 0.5) * (1.0 + 0.08 * g_str * torch.pow(1.0 - t_y, 1.4))

            h00, h01, h02 = float(h_inv[0, 0]), float(h_inv[0, 1]), float(h_inv[0, 2])
            h10, h11, h12 = float(h_inv[1, 0]), float(h_inv[1, 1]), float(h_inv[1, 2])
            h20, h21, h22 = float(h_inv[2, 0]), float(h_inv[2, 1]), float(h_inv[2, 2])
            denom = h20 * sx_planar + h21 * sy_planar + h22
            denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
            map_x = (h00 * sx_planar + h01 * sy_planar + h02) / denom
            map_y = (h10 * sx_planar + h11 * sy_planar + h12) / denom
            norm_x = (2.0 * map_x / max(1.0, float(ww - 1))) - 1.0
            norm_y = (2.0 * map_y / max(1.0, float(wh - 1))) - 1.0
            grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)

            owned_src = src_cuda_t is None
            src_t = (
                src_cuda_t
                if src_cuda_t is not None
                else torch.from_numpy(np.array(arr, dtype=np.uint8, copy=True))
                .to("cuda", dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            earth_w = (
                F.grid_sample(src_t, grid, mode="bicubic", padding_mode="border", align_corners=True)
                .clamp_(0.0, 255.0)
                .squeeze(0)
                .permute(1, 2, 0)
            )

            d_limb = gy_g - y_limb
            zero = torch.zeros_like(d_limb)
            haze_w = torch.where(d_limb >= 0.0, torch.exp(-d_limb / (58.0 * scale_h)) * 0.52, zero).unsqueeze(-1)
            atmo_rgb = torch.tensor([92.0, 178.0, 250.0], device="cuda", dtype=torch.float32)
            earth_lit = earth_w * (1.0 - haze_w) + atmo_rgb * haze_w

            space_bg = torch.tensor([3.0, 6.0, 14.0], device="cuda", dtype=torch.float32).view(1, 1, 3).expand(out_h, out_w, 3).clone()
            limb_at_stars = y_limb[star_ys, star_xs]
            vis = star_ys.to(torch.float32) < (limb_at_stars - 6.0)
            if bool(vis.any()):
                v_xs = star_xs[vis]
                v_ys = star_ys[vis]
                v_b = star_b[vis]
                space_bg[v_ys, v_xs, 0] = v_b * 0.88
                space_bg[v_ys, v_xs, 1] = v_b * 0.94
                space_bg[v_ys, v_xs, 2] = v_b

            rim_out = torch.where(d_limb < 0.0, torch.exp(d_limb / (14.0 * scale_h)) * 0.82, zero).unsqueeze(-1)
            rim_rgb = torch.tensor([82.0, 172.0, 252.0], device="cuda", dtype=torch.float32)
            space_lit = space_bg * (1.0 - rim_out) + rim_rgb * rim_out

            earth_mask = torch.clamp((d_limb + 2.5) / 5.0, 0.0, 1.0).unsqueeze(-1)
            crest = (torch.exp(-((d_limb / (5.5 * scale_h)) ** 2)) * 0.45).unsqueeze(-1)
            crest_rgb = torch.tensor([128.0, 208.0, 255.0], device="cuda", dtype=torch.float32)
            comp = space_lit * (1.0 - earth_mask) + earth_lit * earth_mask + crest * crest_rgb
            out_u8 = comp.clamp_(0.0, 255.0).round().to(torch.uint8).cpu().numpy()
            del y_limb, span_h, t_y, sy_planar, sx_planar, denom, map_x, map_y, norm_x, norm_y, grid
            del earth_w, d_limb, zero, haze_w, earth_lit, space_bg, rim_out, space_lit, earth_mask, crest, comp
            if owned_src:
                del src_t
            return out_u8
    except Exception:
        return None


def _compute_camera_projection(
    ww: int,
    wh: int,
    out_w: int,
    out_h: int,
    center_xy: tuple[float, float],
    zoom: float,
    pitch_deg: float,
    bank_deg: float,
    globe_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, Any, Any, float, float, float]:
    """Compute exact camera homography matrices and world->screen projector without warping pixels."""
    cx, cy = center_xy
    pitch = math.radians(max(0.0, min(42.0, pitch_deg)))
    bank = math.radians(max(-12.0, min(12.0, bank_deg)))
    g_str = max(0.0, min(1.0, float(globe_strength)))

    dst_pts = np.array(
        [[0.0, 0.0], [float(out_w), 0.0], [float(out_w), float(out_h)], [0.0, float(out_h)]],
        dtype=np.float32,
    )
    src_pts: list[list[float]] = []
    half_sw = out_w * 0.5
    half_sh = out_h * (0.58 if g_str > 0.1 else 0.50)
    cb, sb = math.cos(bank), math.sin(bank)
    cp = math.cos(pitch)

    base_frac = 0.56 if g_str > 0.1 else 0.72
    base_scale = (ww / max(1.0, float(out_w))) * (base_frac / max(0.55, zoom))
    max_rx = abs(half_sw * cb) + abs(half_sh * sb)
    max_ry = abs(half_sw * sb) + abs(half_sh * cb)
    far_persp = 1.0 / max(0.42, 1.0 - 0.5 * math.tan(pitch) * 0.72)
    req_hx = max_rx * far_persp * base_scale
    req_hy = max_ry * far_persp * (base_scale / max(0.52, cp))
    if ww > 2.0 * req_hx + 4.0:
        cx = max(req_hx + 2.0, min(float(ww - 3) - req_hx, cx))
    if wh > 2.0 * req_hy + 4.0:
        cy = max(req_hy + 2.0, min(float(wh - 3) - req_hy, cy))

    for sx_c, sy_c in dst_pts:
        nx = sx_c - half_sw
        ny = sy_c - half_sh
        v_norm = ny / max(1.0, float(out_h))
        persp = 1.0 / max(0.42, 1.0 + v_norm * math.tan(pitch) * 0.72)
        rx = (nx * cb - ny * sb) * persp * base_scale
        ry = (nx * sb + ny * cb) * persp * (base_scale / max(0.52, cp))
        wx = max(0.0, min(float(ww - 1), cx + rx))
        wy = max(0.0, min(float(wh - 1), cy + ry))
        src_pts.append([wx, wy])

    src_arr = np.array(src_pts, dtype=np.float32)
    h_inv = cv2.getPerspectiveTransform(dst_pts, src_arr)  # screen -> world
    h_fwd = cv2.getPerspectiveTransform(src_arr, dst_pts)  # world -> screen

    if g_str <= 0.02:
        def _proj_flat(wx: float, wy: float) -> tuple[float, float]:
            pt = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
            res = cv2.perspectiveTransform(pt, h_fwd)
            return (float(res[0, 0, 0]), float(res[0, 0, 1]))

        return h_inv, h_fwd, _proj_flat, (lambda _sx: -100.0), 0.0, 0.0, 0.0

    y_h0 = out_h * (0.105 * g_str)
    curve_drop = out_h * (0.095 * g_str)
    bank_slope = math.tan(bank) * (out_w * 0.08)

    def _horizon_y(sx_val: float) -> float:
        nx_v = (2.0 * sx_val / max(1.0, float(out_w - 1))) - 1.0
        return y_h0 + curve_drop * (nx_v * nx_v) + bank_slope * nx_v

    def _proj_globe(wx: float, wy: float) -> tuple[float, float]:
        pt = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, h_fwd)
        sx_p, sy_p = float(res[0, 0, 0]), float(res[0, 0, 1])
        t_v = sy_p / max(1.0, float(out_h))
        nx_approx = (2.0 * sx_p / max(1.0, float(out_w - 1))) - 1.0
        yl_approx = y_h0 + curve_drop * (nx_approx * nx_approx) + bank_slope * nx_approx
        sy_g = yl_approx + t_v * max(1.0, float(out_h) - yl_approx)
        span_f = 1.0 + 0.08 * g_str * (max(0.0, 1.0 - t_v) ** 1.4)
        sx_g = out_w * 0.5 + (sx_p - out_w * 0.5) / max(0.5, span_f)
        return (sx_g, sy_g)

    return h_inv, h_fwd, _proj_globe, _horizon_y, y_h0, curve_drop, bank_slope


def _apply_camera_homography(
    world_rgb: Image.Image,
    out_w: int,
    out_h: int,
    center_xy: tuple[float, float],
    zoom: float,
    pitch_deg: float,
    bank_deg: float,
    globe_strength: float = 0.0,
    src_cuda_t: Optional[Any] = None,
    vignette_strength: float = 0.0,
    sharpen_amount: float = 0.0,
) -> tuple[np.ndarray, Any, Any]:
    """
    Project oversampled 2D world atlas through a 2.5D virtual drone / Google Earth orbital camera.

    Supports pre-uploaded `src_cuda_t` VRAM tensor reuse + GPU sharpening/vignette for ~3-5x faster renders.
    """
    ww, wh = world_rgb.size
    arr = np.asarray(world_rgb, dtype=np.uint8)
    g_str = max(0.0, min(1.0, float(globe_strength)))
    h_inv, h_fwd, proj_fn, horizon_y_fn, y_h0, curve_drop, bank_slope = _compute_camera_projection(
        ww, wh, out_w, out_h, center_xy, zoom, pitch_deg, bank_deg, g_str
    )

    if g_str <= 0.02:
        warped = _cuda_warp_flat(
            arr, h_inv, out_w, out_h,
            src_cuda_t=src_cuda_t,
            vignette_strength=vignette_strength,
            sharpen_amount=sharpen_amount,
        )
        if warped is None:
            warped = cv2.warpPerspective(
                arr,
                h_fwd,
                (out_w, out_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            if sharpen_amount > 1e-3:
                w_f = warped.astype(np.float32)
                blur_f = cv2.GaussianBlur(w_f, (3, 3), 0.7)
                warped = np.clip(np.round(w_f + (w_f - blur_f) * float(sharpen_amount)), 0, 255).astype(np.uint8)
            if vignette_strength > 1e-3:
                yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
                nx_v = (2.0 * xx / max(1.0, float(out_w - 1))) - 1.0
                ny_v = (2.0 * yy / max(1.0, float(out_h - 1))) - 1.0
                r2 = nx_v * nx_v * 0.72 + ny_v * ny_v * 0.95
                vig = np.clip(1.0 - float(vignette_strength) * np.maximum(0.0, r2 - 0.35), 0.72, 1.0)[..., None]
                warped = np.clip(np.round(warped.astype(np.float32) * vig), 0, 255).astype(np.uint8)
        return warped, proj_fn, horizon_y_fn

    comp = _cuda_warp_globe(
        arr, h_inv, out_w, out_h, g_str, y_h0, curve_drop, bank_slope,
        src_cuda_t=src_cuda_t,
    )
    if comp is None:
        gx_grid, gy_grid, nx_norm, star_xs, star_ys, star_b = _get_cpu_globe_grid(out_w, out_h)
        y_limb = y_h0 + curve_drop * (nx_norm * nx_norm) + bank_slope * nx_norm

        span_h = np.maximum(1.0, float(out_h) - y_limb)
        t_y = np.clip((gy_grid - y_limb) / span_h, 0.0, 1.0)
        sy_planar = t_y * float(out_h)
        sx_planar = out_w * 0.5 + (gx_grid - out_w * 0.5) * (1.0 + 0.08 * g_str * np.power(1.0 - t_y, 1.4))

        denom = h_inv[2, 0] * sx_planar + h_inv[2, 1] * sy_planar + h_inv[2, 2]
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        map_x = ((h_inv[0, 0] * sx_planar + h_inv[0, 1] * sy_planar + h_inv[0, 2]) / denom).astype(np.float32)
        map_y = ((h_inv[1, 0] * sx_planar + h_inv[1, 1] * sy_planar + h_inv[1, 2]) / denom).astype(np.float32)

        earth_warped = cv2.remap(
            arr,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)

        d_limb = gy_grid - y_limb
        scale_h = max(0.5, out_h / 1080.0)
        haze_w = np.where(d_limb >= 0.0, np.exp(-d_limb / (58.0 * scale_h)) * 0.52, 0.0).astype(np.float32)[..., None]
        atmo_rgb = np.array([92.0, 178.0, 250.0], dtype=np.float32)
        earth_lit = earth_warped * (1.0 - haze_w) + atmo_rgb * haze_w

        space_bg = np.full((out_h, out_w, 3), np.array([3.0, 6.0, 14.0], dtype=np.float32), dtype=np.float32)
        for st_x, st_y, br in zip(star_xs, star_ys, star_b):
            if st_y < y_limb[st_y, st_x] - 6.0:
                space_bg[st_y, st_x] = [br * 0.88, br * 0.94, br]

        rim_out = np.where(d_limb < 0.0, np.exp(d_limb / (14.0 * scale_h)) * 0.82, 0.0).astype(np.float32)[..., None]
        space_lit = space_bg * (1.0 - rim_out) + np.array([82.0, 172.0, 252.0], dtype=np.float32) * rim_out

        earth_mask = np.clip((d_limb + 2.5) / 5.0, 0.0, 1.0).astype(np.float32)[..., None]
        crest = np.exp(-(d_limb / (5.5 * scale_h)) ** 2) * 0.45
        comp = space_lit * (1.0 - earth_mask) + earth_lit * earth_mask
        comp = np.clip(comp + crest[..., None] * np.array([128.0, 208.0, 255.0], dtype=np.float32), 0, 255).astype(np.uint8)

    return comp, proj_fn, horizon_y_fn


def _draw_hud_mode_glyph(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    mode: str,
    color_rgba: tuple[int, int, int, int],
) -> None:
    """Draw a tiny crisp geometric vector icon (plane/car/train/boat/walk) inside the HUD header."""
    r = max(5.0, float(size))
    if mode == "flight":
        poly = [
            (cx + r * 0.9, cy),
            (cx + r * 0.2, cy - r * 0.16),
            (cx - r * 0.2, cy - r * 0.85),
            (cx - r * 0.42, cy - r * 0.78),
            (cx - r * 0.15, cy - r * 0.16),
            (cx - r * 0.68, cy - r * 0.12),
            (cx - r * 0.88, cy - r * 0.36),
            (cx - r * 0.95, cy),
            (cx - r * 0.88, cy + r * 0.36),
            (cx - r * 0.68, cy + r * 0.12),
            (cx - r * 0.15, cy + r * 0.16),
            (cx - r * 0.42, cy + r * 0.78),
            (cx - r * 0.2, cy + r * 0.85),
            (cx + r * 0.2, cy + r * 0.16),
        ]
        draw.polygon(poly, fill=color_rgba)
    elif mode == "train":
        draw.rounded_rectangle([cx - r * 0.85, cy - r * 0.34, cx + r * 0.85, cy + r * 0.34], radius=max(2, int(r * 0.25)), fill=color_rgba)
        draw.line([(cx - r * 0.95, cy + r * 0.52), (cx + r * 0.95, cy + r * 0.52)], fill=color_rgba, width=max(1, int(r * 0.15)))
    elif mode == "boat":
        hull = [(cx + r * 0.9, cy), (cx + r * 0.35, cy + r * 0.35), (cx - r * 0.75, cy + r * 0.35), (cx - r * 0.85, cy - r * 0.10), (cx + r * 0.35, cy - r * 0.10)]
        draw.polygon(hull, fill=color_rgba)
        draw.polygon([(cx - r * 0.15, cy - r * 0.12), (cx + r * 0.15, cy - r * 0.65), (cx + r * 0.25, cy - r * 0.12)], fill=color_rgba)
    elif mode == "walk":
        draw.ellipse([cx - r * 0.32, cy - r * 0.80, cx + r * 0.32, cy - r * 0.18], fill=color_rgba)
        draw.line([(cx, cy - r * 0.18), (cx, cy + r * 0.35)], fill=color_rgba, width=max(2, int(r * 0.22)))
        draw.line([(cx, cy + r * 0.35), (cx - r * 0.42, cy + r * 0.85)], fill=color_rgba, width=max(2, int(r * 0.20)))
        draw.line([(cx, cy + r * 0.35), (cx + r * 0.42, cy + r * 0.85)], fill=color_rgba, width=max(2, int(r * 0.20)))
    else:
        draw.rounded_rectangle([cx - r * 0.90, cy - r * 0.22, cx + r * 0.90, cy + r * 0.28], radius=max(2, int(r * 0.22)), fill=color_rgba)
        cabin = [(cx - r * 0.48, cy - r * 0.20), (cx - r * 0.25, cy - r * 0.56), (cx + r * 0.32, cy - r * 0.56), (cx + r * 0.55, cy - r * 0.20)]
        draw.polygon(cabin, fill=color_rgba)
        for wx_o in (-r * 0.48, r * 0.48):
            draw.ellipse([cx + wx_o - r * 0.20, cy + r * 0.18, cx + wx_o + r * 0.20, cy + r * 0.56], fill=color_rgba)


def _abbreviate_ribbon_label(label: str, max_chars: int = 12) -> str:
    """Shorten a waypoint label cleanly for the journey ribbon without mid-word truncation."""
    base = str(label or "").split("·")[0].strip()
    if len(base) <= max_chars:
        return base
    # Strip common leading articles first ("Les Affranchis" -> "Affranchis", "Le Louvre" -> "Louvre")
    words = base.split()
    if len(words) >= 2 and words[0].lower() in ("the", "les", "la", "le", "l'", "el", "los", "las", "der", "die", "das"):
        stripped = " ".join(words[1:])
        if len(stripped) <= max_chars:
            return stripped
        words = words[1:]
    # For multi-word monument names like "Arc de Triomphe", pick the distinctive headline noun ("Triomphe")
    if len(words) >= 2 and len(words[-1]) >= 5 and len(words[-1]) <= max_chars:
        return words[-1]
    # Otherwise fit as many whole words as possible before hard fallback
    acc = words[0]
    for w_tok in words[1:]:
        if len(acc) + 1 + len(w_tok) <= max_chars:
            acc = f"{acc} {w_tok}"
        else:
            break
    return acc if len(acc) <= max_chars else (acc[: max_chars - 1].rstrip() + ".")


def _render_screen_hud(
    frame_rgba: Image.Image,
    style: str,
    waypoints: list[MapWaypointSpec],
    leg_index: int,
    t_norm: float,
    transport_mode: str,
    compact_hud: bool,
    hdr_safe: bool,
    route_total_km: Optional[float] = None,
    cur_latlon: Optional[tuple[float, float]] = None,
    draw_compass: bool = True,
) -> None:
    """Composite crisp post-warp frosted-glass route telemetry card, compass rose & journey/elevation card."""
    spec = CART_THEME_SPECS[style]
    w, h = frame_rgba.size
    s = w / 1920.0
    shadow_rgb = (68, 54, 38) if spec.get("light_paper") else (6, 10, 16)

    i0 = max(0, min(len(waypoints) - 1, leg_index))
    i1 = min(len(waypoints) - 1, i0 + 1)
    orig = waypoints[i0]
    dest = waypoints[i1]
    u = _cartography_travel_progress(t_norm)
    leg_km = float(route_total_km) if (route_total_km is not None and route_total_km > 0.0) else haversine_km(orig.lat, orig.lon, dest.lat, dest.lon)
    cur_km = leg_km * u
    if cur_latlon is not None:
        cur_lat, cur_lon = float(cur_latlon[0]), float(cur_latlon[1])
    else:
        cur_lat = orig.lat + (dest.lat - orig.lat) * u
        cur_lon = orig.lon + (dest.lon - orig.lon) * u
    base_alt = orig.alt_m + (dest.alt_m - orig.alt_m) * u
    if transport_mode == "flight":
        cruise_peak = min(10950.0, max(1800.0, leg_km * 14.0))
        cur_alt = base_alt + cruise_peak * math.sin(math.pi * u)
    else:
        cur_alt = base_alt

    # Top horizon mist strip for 2.5D atmospheric depth (subtle edge fade)
    if not compact_hud:
        mist_h = max(12, int(round(h * 0.14)))
        mist_rgb = spec["horizon_mist"]
        strip = np.zeros((mist_h, 1, 4), dtype=np.uint8)
        for y in range(mist_h):
            k = 1.0 - (y / max(1, mist_h - 1))
            strip[y, 0] = (mist_rgb[0], mist_rgb[1], mist_rgb[2], int(round(130 * (k * k))))
        mist_img = Image.fromarray(strip).resize((w, mist_h), Image.Resampling.BILINEAR)
        frame_rgba.alpha_composite(mist_img, (0, 0))

    hud_fill = tuple(spec["hud_fill"])
    hud_border = tuple(spec["hud_border"])
    title_c = (*spec["hud_title"], 248)
    sub_c = (*spec["hud_sub"], 226)
    metric_c = (*spec["hud_metric"], 248)

    hud_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hud_layer, "RGBA")

    if compact_hud:
        ph = max(24, int(round(h * 0.18)))
        pt = h - ph - max(6, int(round(10 * s)))
        pl = max(8, int(round(14 * s)))
        pr = w - pl
        draw.rounded_rectangle([pl, pt, pr, pt + ph], radius=max(6, int(round(10 * s))), fill=hud_fill, outline=hud_border, width=max(1, int(round(1.5 * s))))
        frame_rgba.alpha_composite(hud_layer)
        t_sz = max(11, int(round(24 * s)))
        m_sz = max(10, int(round(18 * s)))
        f_t = _load_font_tier(t_sz, tier=spec["title_tier"], bold=True)
        f_m = _load_font_tier(m_sz, tier=spec["sub_tier"], bold=True)
        draw_tracked_text(
            frame_rgba,
            (pl + max(8, int(round(14 * s))), pt + max(3, (ph - t_sz) // 2)),
            f"LEG {i0 + 1:02d}/{max(1, len(waypoints) - 1):02d}  |  {transport_mode.upper()}",
            font=f_t,
            fill=title_c,
            shadow=False,
            hdr_safe=hdr_safe,
        )
        km_lbl = f"{cur_km:.0f} KM" if leg_km >= 2.0 else f"{cur_km * 1000.0:.0f} M"
        km_w, _ = measure_tracked_text(draw, km_lbl, f_m)
        draw_tracked_text(frame_rgba, (pr - km_w - max(8, int(round(14 * s))), pt + max(3, (ph - m_sz) // 2)), km_lbl, font=f_m, fill=metric_c, shadow=False, hdr_safe=hdr_safe)
        return

    # Full-frame documentary HUD telemetry pill (upper-left) with crisp vector arrow separator
    px = int(round(w * 0.045))
    py = int(round(h * 0.060))
    f_over = _load_font_tier(max(10, int(round(13 * s))), tier=spec["sub_tier"], bold=True)
    f_head = _load_font_tier(max(13, int(round(29 * s))), tier=spec["title_tier"], bold=True, text=f"{orig.label} {dest.label}")
    f_stat = _load_font_tier(max(10, int(round(15 * s))), tier=spec["sub_tier"], bold=False)

    orig_w, orig_h = measure_tracked_text(draw, orig.label, f_head)
    dest_w, _ = measure_tracked_text(draw, dest.label, f_head)
    arrow_span = int(round(36 * s))
    rw1 = orig_w + arrow_span + dest_w
    km_disp = f"{cur_km:.1f} / {leg_km:.1f} KM" if leg_km >= 10.0 else f"{cur_km:.2f} / {leg_km:.1f} KM"
    stat_line = f"{transport_mode.upper()}  |  {km_disp}  |  {_format_coord_dms(cur_lat, True)} {_format_coord_dms(cur_lon, False)}"
    over_line = f"EXPEDITION LEG {i0 + 1:02d} OF {max(1, len(waypoints) - 1):02d}"

    rw2, _ = measure_tracked_text(draw, stat_line, f_stat)
    rw3, _ = measure_tracked_text(draw, over_line, f_over, letter_spacing_px=2.0 * s)
    glyph_pad = int(round(28 * s))
    box_w = min(int(w * 0.68), max(int(round(370 * s)), max(rw1, rw2 + glyph_pad, rw3) + int(round(62 * s))))
    box_h = max(68, int(round(106 * s)))
    # Soft true-alpha drop shadow + frosted glass card
    draw.rounded_rectangle(
        [px + 3, py + 5, px + box_w + 4, py + box_h + 6],
        radius=max(8, int(round(14 * s))),
        fill=(*shadow_rgb, 48),
    )
    draw.rounded_rectangle(
        [px, py, px + box_w, py + box_h],
        radius=max(8, int(round(14 * s))),
        fill=hud_fill,
        outline=hud_border,
        width=max(1, int(round(1.6 * s))),
    )
    bar_w = max(2, int(round(3.5 * s)))
    draw.rounded_rectangle(
        [px + int(round(14 * s)), py + int(round(14 * s)), px + int(round(14 * s)) + bar_w, py + box_h - int(round(14 * s))],
        radius=max(1, bar_w // 2),
        fill=(*spec["accent"], 240),
    )
    tx = px + int(round(28 * s))
    head_y = py + int(round(33 * s))
    ax0 = tx + orig_w + int(round(10 * s))
    ax1 = tx + orig_w + arrow_span - int(round(10 * s))
    ay_mid = head_y + int(round(max(12.0, orig_h * 0.54)))
    ar_col = (*spec["accent"], 245)
    draw.line([(ax0, ay_mid), (ax1 - int(round(4 * s)), ay_mid)], fill=ar_col, width=max(2, int(round(2.4 * s))))
    ah = max(4.0, 5.5 * s)
    draw.polygon([(ax1, ay_mid), (ax1 - ah * 1.45, ay_mid - ah * 0.85), (ax1 - ah * 1.45, ay_mid + ah * 0.85)], fill=ar_col)
    _draw_hud_mode_glyph(draw, tx + int(round(9 * s)), py + int(round(81 * s)), 8.5 * s, transport_mode, (*spec["accent"], 242))

    # Smart lower-right telemetry card:
    # - Flights & mountain legs (>=180m relief or >=500m altitude): Altitude / Elevation Sparkline
    # - Urban / flat multi-stop journeys: Multi-Stop Journey Progress Ribbon
    alt_a = max(0.0, float(orig.alt_m))
    alt_b = max(0.0, float(dest.alt_m))
    show_elev_profile = (
        transport_mode == "flight"
        or abs(alt_b - alt_a) >= 180.0
        or max(alt_a, alt_b) >= 500.0
        or len(waypoints) < 2
    )

    sp_w = max(160, int(round((396 if not show_elev_profile else 310) * s)))
    sp_h = max(60, int(round(94 * s)))
    sp_x = w - int(round(w * 0.045)) - sp_w
    sp_y = h - int(round(h * 0.060)) - sp_h
    draw.rounded_rectangle(
        [sp_x + 3, sp_y + 5, sp_x + sp_w + 4, sp_y + sp_h + 6],
        radius=max(6, int(round(12 * s))),
        fill=(*shadow_rgb, 48),
    )
    draw.rounded_rectangle(
        [sp_x, sp_y, sp_x + sp_w, sp_y + sp_h],
        radius=max(6, int(round(12 * s))),
        fill=hud_fill,
        outline=hud_border,
        width=max(1, int(round(1.5 * s))),
    )
    f_sp = _load_font_tier(max(9, int(round(12 * s))), tier=spec["sub_tier"], bold=True)

    if not show_elev_profile:
        # Multi-stop Journey Progress Ribbon (all vector graphics on true-alpha hud_layer!)
        n_wp = len(waypoints)
        n_legs = max(1, n_wp - 1)
        overall_pct = int(round(((i0 + u) / float(n_legs)) * 100.0))
        rx0 = sp_x + int(round(32 * s))
        rx1 = sp_x + sp_w - int(round(32 * s))
        ry_line = sp_y + int(round(48 * s))
        draw.line([(rx0, ry_line), (rx1, ry_line)], fill=(*spec["route_future"], 145), width=max(2, int(round(3.0 * s))))
        prog_x = rx0 + (rx1 - rx0) * ((i0 + u) / float(n_legs))
        draw.line([(rx0, ry_line), (prog_x, ry_line)], fill=(*spec["route_core"], 250), width=max(3, int(round(4.0 * s))))
        f_node = _load_font_tier(max(8, int(round(10.5 * s))), tier="mono", bold=True)
        for k_s in range(n_wp):
            nx_s = rx0 + (rx1 - rx0) * (k_s / float(n_legs))
            is_done = k_s <= i0
            is_target = k_s == i1
            nr = max(5, int(round((7.5 if (k_s in (i0, i1)) else 5.5) * s)))
            ring_c = (*spec["accent"], 250) if (is_done or is_target) else (*spec["route_future"], 180)
            fill_c = (*spec["route_core"], 255) if is_done else (*spec["pin_fill"], 248)
            draw.ellipse([nx_s - nr - 1.5, ry_line - nr - 1.5, nx_s + nr + 1.5, ry_line + nr + 1.5], fill=ring_c)
            draw.ellipse([nx_s - nr, ry_line - nr, nx_s + nr, ry_line + nr], fill=fill_c)
            lbl_short = _abbreviate_ribbon_label(waypoints[k_s].label, max_chars=12)
            if k_s in (i0, i1) or n_wp <= 4:
                lw, _ = measure_tracked_text(draw, lbl_short, f_node)
                lx = max(sp_x + int(round(10 * s)), min(sp_x + sp_w - lw - int(round(10 * s)), nx_s - lw * 0.5))
                draw.text((lx, ry_line + int(round(13 * s))), lbl_short, font=f_node, fill=title_c if k_s in (i0, i1) else sub_c)
        pr_r = max(3, int(round(5.0 * s)))
        draw.ellipse([prog_x - pr_r * 2.0, ry_line - pr_r * 2.0, prog_x + pr_r * 2.0, ry_line + pr_r * 2.0], fill=(*spec["route_glow"], 88))
        draw.ellipse([prog_x - pr_r, ry_line - pr_r, prog_x + pr_r, ry_line + pr_r], fill=(*spec["accent"], 255))
        frame_rgba.alpha_composite(hud_layer)

        # Crisp text on top of composited HUD
        draw_tracked_text(frame_rgba, (tx, py + int(round(12 * s))), over_line, font=f_over, fill=metric_c, letter_spacing_px=2.0 * s, shadow=False, hdr_safe=hdr_safe)
        draw_tracked_text(frame_rgba, (tx, head_y), orig.label, font=f_head, fill=title_c, shadow=False, hdr_safe=hdr_safe)
        draw_tracked_text(frame_rgba, (tx + orig_w + arrow_span, head_y), dest.label, font=f_head, fill=title_c, shadow=False, hdr_safe=hdr_safe)
        draw_tracked_text(frame_rgba, (tx + glyph_pad, py + int(round(73 * s))), stat_line, font=f_stat, fill=sub_c, shadow=False, hdr_safe=hdr_safe)
        draw_tracked_text(frame_rgba, (sp_x + int(round(14 * s)), sp_y + int(round(10 * s))), f"JOURNEY PROGRESS  |  {overall_pct:d}%", font=f_sp, fill=sub_c, shadow=False, hdr_safe=hdr_safe)
    else:
        gx0 = sp_x + int(round(14 * s))
        gx1 = sp_x + sp_w - int(round(14 * s))
        gy_base = sp_y + sp_h - int(round(14 * s))
        gy_top = sp_y + int(round(34 * s))
        alt_max = max(alt_a, alt_b, 200.0)
        n_sp = 32
        prof_pts: list[tuple[float, float]] = []
        for k in range(n_sp):
            frac = k / float(n_sp - 1)
            if transport_mode == "flight":
                e_norm = 0.14 + 0.78 * math.sin(math.pi * frac)
            else:
                ridge = math.sin(math.pi * frac) * 0.14
                e_norm = max(0.08, min(0.95, ((alt_a * (1.0 - frac) + alt_b * frac) / alt_max) * 0.72 + ridge))
            prof_pts.append((gx0 + (gx1 - gx0) * frac, gy_base - (gy_base - gy_top) * e_norm))

        poly_fill = [(gx0, gy_base)] + prof_pts + [(gx1, gy_base)]
        draw.polygon(poly_fill, fill=(*spec["route_glow"], 54))
        active_cnt = max(2, int(round(u * (len(prof_pts) - 1))) + 1)
        hx, hy = prof_pts[min(len(prof_pts) - 1, active_cnt - 1)]
        pr_r = max(3, int(round(4.5 * s)))
        draw.ellipse([hx - pr_r * 1.9, hy - pr_r * 1.9, hx + pr_r * 1.9, hy + pr_r * 1.9], fill=(*spec["accent"], 85))
        draw.ellipse([hx - pr_r, hy - pr_r, hx + pr_r, hy + pr_r], fill=(*spec["accent"], 255))
        frame_rgba.alpha_composite(hud_layer)

        sp_layer = np.zeros((sp_h + 2, sp_w + 2, 4), dtype=np.uint8)
        local_prof = [(px_p - sp_x, py_p - sp_y) for (px_p, py_p) in prof_pts]
        _draw_aa_polyline(sp_layer, local_prof, (*spec["route_future"], 145), max(1, int(round(2.0 * s))))
        _draw_aa_polyline(sp_layer, local_prof[:active_cnt], (*spec["route_core"], 250), max(2, int(round(3.0 * s))))
        frame_rgba.alpha_composite(Image.fromarray(sp_layer, mode="RGBA"), dest=(sp_x, sp_y))

        draw_tracked_text(frame_rgba, (tx, py + int(round(12 * s))), over_line, font=f_over, fill=metric_c, letter_spacing_px=2.0 * s, shadow=False, hdr_safe=hdr_safe)
        draw_tracked_text(frame_rgba, (tx, head_y), orig.label, font=f_head, fill=title_c, shadow=False, hdr_safe=hdr_safe)
        draw_tracked_text(frame_rgba, (tx + orig_w + arrow_span, head_y), dest.label, font=f_head, fill=title_c, shadow=False, hdr_safe=hdr_safe)
        draw_tracked_text(frame_rgba, (tx + glyph_pad, py + int(round(73 * s))), stat_line, font=f_stat, fill=sub_c, shadow=False, hdr_safe=hdr_safe)
        draw_tracked_text(
            frame_rgba,
            (sp_x + int(round(14 * s)), sp_y + int(round(10 * s))),
            f"ALT PROFILE  |  {cur_alt:.0f}M ASL" if transport_mode == "flight" else f"ELEV PROFILE  |  {cur_alt:.0f}M ASL",
            font=f_sp,
            fill=sub_c,
            shadow=False,
            hdr_safe=hdr_safe,
        )

    # Crisp anchored screen-space compass rose (bottom-left safe corner, never drifts across route!)
    if draw_compass:
        f_comp = _load_font_tier(max(9, int(round(11 * s))), tier="mono", bold=True)
        _draw_compass_rose(
            frame_rgba,
            cx=int(round(w * 0.055)),
            cy=int(round(h * 0.892)),
            radius=max(22, int(round(32 * s))),
            color_rgb=spec["compass_color"],
            sub_font=f_comp,
            light_paper=bool(spec["light_paper"]),
        )


@dataclass
class _LegLayoutPlan:
    """Locked per-leg screen-space slots preventing mid-clip Polaroid, label, or annotation teleports."""

    orig_polaroid_slot: Optional[int] = None
    dest_polaroid_slot: Optional[int] = None
    label_slots: dict[int, int] = None  # type: ignore[assignment]
    annotation_offsets: dict[int, tuple[float, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.label_slots is None:
            self.label_slots = {}
        if self.annotation_offsets is None:
            self.annotation_offsets = {}


@dataclass
class _PreparedLegContext:
    """Precomputed static geometry, VRAM-cached world baseplate, thumbs & locked layout for a map leg."""

    use_style: str
    spec: dict[str, Any]
    norm_wps: list[MapWaypointSpec]
    i0: int
    i1: int
    width: int
    height: int
    world_w: int
    world_h: int
    vp: _MapViewport
    px_stops: list[tuple[float, float]]
    all_leg_splines: list[list[tuple[float, float]]]
    ground_arc_path: _ArcLengthPath
    leg_km: float
    route_total_km: float
    resolved_mode: str
    use_globe: bool
    pitch_deg: float
    compact_hud: bool
    static_world_rgb: Image.Image
    static_cuda_t: Optional[Any]
    orig_thumb_rgb: Optional[Image.Image]
    dest_thumb_rgb: Optional[Image.Image]
    layout_plan: _LegLayoutPlan

    def release_cuda(self) -> None:
        if self.static_cuda_t is not None:
            try:
                with _CUDA_MAP_LOCK:
                    self.static_cuda_t = None
            except Exception:
                self.static_cuda_t = None


def _compute_leg_camera_state(
    orig_px: tuple[float, float],
    dest_px: tuple[float, float],
    ground_arc_path: _ArcLengthPath,
    u: float,
    world_w: int,
    leg_km: float,
    use_globe: bool,
    compact_hud: bool,
) -> tuple[float, float, float]:
    """Compute serene overview-breathing camera center & zoom at progress u in [0..1]."""
    s_w = world_w / 1920.0
    smooth_cam_pt = ground_arc_path.smoothed_camera_point(u, window_px=165.0 * s_w)
    mid_x = 0.5 * (orig_px[0] + dest_px[0])
    mid_y = 0.5 * (orig_px[1] + dest_px[1])
    # Keep balanced corridor framing (46% midpoint, 54% smoothed head) so origin/dest pins never get pushed off-edge
    cam_cx = 0.46 * mid_x + 0.54 * smooth_cam_pt[0]
    cam_cy = 0.46 * mid_y + 0.54 * smooth_cam_pt[1]
    e_u = _ease_in_out_cubic(u)
    breath = math.sin(math.pi * u)  # Gentle wide overview mid-journey -> serene arrival settle
    if compact_hud:
        cam_zoom = 1.06 + 0.04 * e_u
    elif use_globe:
        cam_zoom = 0.98 - 0.04 * breath + 0.06 * e_u
    elif leg_km < 15.0:
        cam_zoom = 1.12 - 0.08 * breath + 0.05 * e_u
    else:
        cam_zoom = 1.05 - 0.07 * breath + 0.05 * e_u
    return cam_cx, cam_cy, cam_zoom


def _base_occupied_rects(width: int, height: int, horizon_y_fn: Any, compact_hud: bool) -> list[tuple[float, float, float, float]]:
    if compact_hud:
        return [(0.0, height * 0.78, float(width), float(height))]
    return [
        (0.0, 0.0, width * 0.47, height * 0.21),                                          # Top-left HUD card
        (width * 0.67, height * 0.74, float(width), float(height)),                        # Bottom-right Journey/Elev card
        (0.0, height * 0.78, width * 0.14, float(height)),                                 # Bottom-left Compass rose
        (0.0, 0.0, float(width), max(height * 0.14, horizon_y_fn(width * 0.5) + 12.0)),   # Top horizon band
    ]


def _waypoint_label_candidates(
    sx_w: float,
    sy_w: float,
    pin_r: float,
    tw: float,
    th: float,
    tb0: tuple[int, int, int, int],
    s_out: float,
    is_dest: bool,
) -> list[tuple[float, float]]:
    cands = [
        (sx_w + pin_r + 14 * s_out, sy_w - th * 0.55 - tb0[1]),
        (sx_w - tw * 0.5 - tb0[0], sy_w + pin_r + 18 * s_out - tb0[1]),
        (sx_w - tw * 0.5 - tb0[0], sy_w - pin_r - th - 18 * s_out - tb0[1]),
        (sx_w - pin_r - tw - 24 * s_out - tb0[0], sy_w - th * 0.55 - tb0[1]),
        (sx_w + pin_r + 14 * s_out, sy_w + pin_r + 14 * s_out - tb0[1]),
        (sx_w + pin_r + 14 * s_out, sy_w - pin_r - th - 14 * s_out - tb0[1]),
    ]
    if is_dest:
        return [cands[0], cands[1], cands[3], cands[2], cands[4], cands[5]]
    return cands


def _solve_leg_layout_plan(
    norm_wps: list[MapWaypointSpec],
    i0: int,
    i1: int,
    px_stops: list[tuple[float, float]],
    ground_arc_path: _ArcLengthPath,
    vp: _MapViewport,
    spec: dict[str, Any],
    width: int,
    height: int,
    world_w: int,
    world_h: int,
    leg_km: float,
    use_globe: bool,
    pitch_deg: float,
    compact_hud: bool,
    annotations: Optional[list[dict[str, Any]]],
    has_orig_thumb: bool,
    has_dest_thumb: bool,
) -> _LegLayoutPlan:
    """Solve collision-free Polaroid quadrants, label slots & annotation offsets ONCE across start/mid/end frames."""
    plan = _LegLayoutPlan()
    if compact_hud:
        return plan

    s_out = width / 1920.0
    dummy = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    od = ImageDraw.Draw(dummy, "RGBA")
    all_lbl_txt = " ".join(w.label for w in norm_wps)
    f_wp_lbl = _load_font_tier(max(11, int(round(17 * s_out))), tier=spec["title_tier"], bold=True, text=all_lbl_txt)

    # Project anchor frames (departure u=0.05, mid u=0.50, arrival u=0.95)
    sample_us = (0.05, 0.50, 0.95)
    proj_states: list[tuple[Any, Any, list[tuple[float, float]], list[tuple[float, float]]]] = []
    world_route_samples = [
        ground_arc_path.point_at_s((idx_r / 23.0) * ground_arc_path.total_len) for idx_r in range(24)
    ]
    for u_s in sample_us:
        cam_cx, cam_cy, cam_zoom = _compute_leg_camera_state(
            px_stops[i0], px_stops[i1], ground_arc_path, u_s, world_w, leg_km, use_globe, compact_hud
        )
        _, _, p_fn, h_fn, _, _, _ = _compute_camera_projection(
            world_w, world_h, width, height, (cam_cx, cam_cy), cam_zoom, pitch_deg, 0.0, 1.0 if use_globe else 0.0
        )
        scr_stops = [p_fn(wx, wy) for wx, wy in px_stops]
        scr_route = [p_fn(rx, ry) for rx, ry in world_route_samples]
        proj_states.append((p_fn, h_fn, scr_stops, scr_route))

    # Track per-anchor occupied rects so labels, Polaroids, and annotations never overlap each other
    occ_by_t: list[list[tuple[float, float, float, float]]] = []
    for _, h_fn, scr_stops, _ in proj_states:
        rects = _base_occupied_rects(width, height, h_fn, compact_hud=False)
        for sx_w, sy_w in scr_stops:
            if -40 <= sx_w <= width + 40 and -40 <= sy_w <= height + 40:
                pr = max(12.0, 22.0 * s_out)
                rects.append((sx_w - pr, sy_w - pr, sx_w + pr, sy_w + pr))
        occ_by_t.append(rects)

    # 1. Lock waypoint label slots jointly across departure, mid & arrival frames
    pad_x = int(round(10 * s_out))
    pad_y = int(round(7 * s_out))
    tag_order = [i0, i1] + [k for k in range(len(px_stops)) if k not in (i0, i1)]
    for k in tag_order:
        is_orig = k == i0
        is_dest = k == i1
        pin_r = max(6.0, float(round((11 if (is_orig or is_dest) else 8) * s_out)))
        tag = norm_wps[k].label
        tb0 = od.textbbox((0, 0), tag, font=f_wp_lbl)
        tw, th = float(tb0[2] - tb0[0]), float(tb0[3] - tb0[1])
        best_slot = 0
        best_sc = -1e12
        for slot_i in range(6):
            tot_sc = 0.0
            for t_idx in (0, 1, 2):
                _, _, scr_stops, scr_route = proj_states[t_idx]
                sx_w, sy_w = scr_stops[k]
                cands = _waypoint_label_candidates(sx_w, sy_w, pin_r, tw, th, tb0, s_out, is_dest)
                cx_c, cy_c = cands[slot_i]
                tx_c = max(18.0, min(float(width - tw - 22), cx_c))
                ty_c = max(float(height * 0.19), min(float(height - th - 20), cy_c))
                tot_sc += _score_rect_against_obstacles(
                    tx_c - pad_x, ty_c - pad_y, tw + 2 * pad_x, th + 2 * pad_y,
                    width, height, s_out, [], occ_by_t[t_idx], scr_route,
                )
            if tot_sc > best_sc:
                best_sc = tot_sc
                best_slot = slot_i
        plan.label_slots[k] = best_slot
        for t_idx in (0, 1, 2):
            _, _, scr_stops, _ = proj_states[t_idx]
            sx_w, sy_w = scr_stops[k]
            cands = _waypoint_label_candidates(sx_w, sy_w, pin_r, tw, th, tb0, s_out, is_dest)
            cx_c, cy_c = cands[best_slot]
            tx_c = max(18.0, min(float(width - tw - 22), cx_c))
            ty_c = max(float(height * 0.19), min(float(height - th - 20), cy_c))
            occ_by_t[t_idx].append((tx_c - pad_x, ty_c - pad_y, tx_c + tw + pad_x, ty_c + th + pad_y))

    # Approximate rotated Polaroid dimensions at full bloom
    est_cw = 256.0 * s_out + 24.0
    est_ch = (256.0 * 0.60 + 36.0) * s_out + 24.0

    # 2. Lock FROM Polaroid quadrant (evaluated over departure & mid frames when FROM is visible)
    if has_orig_thumb:
        best_slot = 0
        best_sc = -1e12
        for slot_i in range(8):
            tot_sc = 0.0
            for t_idx in (0, 1):
                _, _, scr_stops, scr_route = proj_states[t_idx]
                dx, dy = scr_stops[i0]
                cx_r, cy_r = _polaroid_candidate_slots(dx, dy, est_cw, est_ch, s_out)[slot_i]
                px_c = max(14.0, min(float(width - est_cw - 14), cx_r))
                py_c = max(float(height * 0.19), min(float(height - est_ch - 14), cy_r))
                others = [p for idx_p, p in enumerate(scr_stops) if idx_p != i0]
                tot_sc += _score_rect_against_obstacles(
                    px_c, py_c, est_cw, est_ch, width, height, s_out, others, occ_by_t[t_idx], scr_route
                )
            if tot_sc > best_sc:
                best_sc = tot_sc
                best_slot = slot_i
        plan.orig_polaroid_slot = best_slot
        for t_idx in (0, 1):
            _, _, scr_stops, _ = proj_states[t_idx]
            dx, dy = scr_stops[i0]
            cx_r, cy_r = _polaroid_candidate_slots(dx, dy, est_cw, est_ch, s_out)[best_slot]
            px_c = max(14.0, min(float(width - est_cw - 14), cx_r))
            py_c = max(float(height * 0.19), min(float(height - est_ch - 14), cy_r))
            occ_by_t[t_idx].append((px_c, py_c, px_c + est_cw, py_c + est_ch))

    # 3. Lock TO Polaroid quadrant (evaluated over mid & arrival frames when TO is visible)
    if has_dest_thumb:
        best_slot = 0
        best_sc = -1e12
        for slot_i in range(8):
            tot_sc = 0.0
            for t_idx in (1, 2):
                _, _, scr_stops, scr_route = proj_states[t_idx]
                dx, dy = scr_stops[i1]
                cx_r, cy_r = _polaroid_candidate_slots(dx, dy, est_cw, est_ch, s_out)[slot_i]
                px_c = max(14.0, min(float(width - est_cw - 14), cx_r))
                py_c = max(float(height * 0.19), min(float(height - est_ch - 14), cy_r))
                others = [p for idx_p, p in enumerate(scr_stops) if idx_p != i1]
                tot_sc += _score_rect_against_obstacles(
                    px_c, py_c, est_cw, est_ch, width, height, s_out, others, occ_by_t[t_idx], scr_route
                )
            if tot_sc > best_sc:
                best_sc = tot_sc
                best_slot = slot_i
        plan.dest_polaroid_slot = best_slot
        for t_idx in (1, 2):
            _, _, scr_stops, _ = proj_states[t_idx]
            dx, dy = scr_stops[i1]
            cx_r, cy_r = _polaroid_candidate_slots(dx, dy, est_cw, est_ch, s_out)[best_slot]
            px_c = max(14.0, min(float(width - est_cw - 14), cx_r))
            py_c = max(float(height * 0.19), min(float(height - est_ch - 14), cy_r))
            occ_by_t[t_idx].append((px_c, py_c, px_c + est_cw, py_c + est_ch))

    # 4. Lock collision-free screen-space offset for each corridor annotation
    if annotations:
        f_t = _load_font_tier(max(11, int(round(15 * s_out))), tier=spec["title_tier"], bold=True)
        f_s = _load_font_tier(max(9, int(round(11 * s_out))), tier="mono", bold=True)
        for idx_a, ann in enumerate(annotations):
            if "lat" not in ann or "lon" not in ann:
                continue
            title = str(ann.get("title") or "").strip()
            sub = str(ann.get("sub") or "").strip()
            if not title:
                continue
            bb_t = od.textbbox((0, 0), title, font=f_t)
            wt, ht = bb_t[2] - bb_t[0], bb_t[3] - bb_t[1]
            ws, hs = (0, 0)
            if sub:
                bb_s = od.textbbox((0, 0), sub, font=f_s)
                ws, hs = bb_s[2] - bb_s[0], bb_s[3] - bb_s[1]
            bw = float(max(wt, ws) + int(round(22 * s_out)))
            bh = float(ht + (hs + int(round(10 * s_out)) if sub else 0) + int(round(15 * s_out)))
            wx_a, wy_a = vp.latlon_to_px(float(ann["lat"]), float(ann["lon"]))
            cand_offs: list[tuple[float, float]] = []
            raw_off = ann.get("offset")
            if raw_off is not None and len(raw_off) >= 2:
                cand_offs.append((float(raw_off[0]) * 0.78 * s_out, float(raw_off[1]) * 0.78 * s_out))
            cand_offs.extend([
                (22 * s_out, -bh - 20 * s_out),
                (-bw - 22 * s_out, -bh - 20 * s_out),
                (22 * s_out, 20 * s_out),
                (-bw - 22 * s_out, 20 * s_out),
                (-bw * 0.5, -bh - 28 * s_out),
                (-bw * 0.5, 28 * s_out),
            ])
            best_off = cand_offs[0]
            best_sc = -1e12
            for ox_c, oy_c in cand_offs:
                tot_sc = 0.0
                for t_idx in (1, 2):
                    p_fn, _, _, scr_route = proj_states[t_idx]
                    ax, ay = p_fn(wx_a, wy_a)
                    tx_c = max(16.0, min(float(width - bw - 16), ax + ox_c))
                    ty_c = max(float(height * 0.19), min(float(height - bh - 16), ay + oy_c))
                    tot_sc += _score_rect_against_obstacles(
                        tx_c, ty_c, bw, bh, width, height, s_out, [], occ_by_t[t_idx], scr_route
                    )
                if tot_sc > best_sc:
                    best_sc = tot_sc
                    best_off = (ox_c, oy_c)
            plan.annotation_offsets[idx_a] = best_off
            for t_idx in (1, 2):
                p_fn, _, _, _ = proj_states[t_idx]
                ax, ay = p_fn(wx_a, wy_a)
                tx_c = max(16.0, min(float(width - bw - 16), ax + best_off[0]))
                ty_c = max(float(height * 0.19), min(float(height - bh - 16), ay + best_off[1]))
                occ_by_t[t_idx].append((tx_c, ty_c, tx_c + bw, ty_c + bh))

    return plan


def _prepare_travel_map_leg(
    waypoints: list[MapWaypointSpec | dict[str, Any]],
    style: str = "archival_expedition_parchment",
    width: int = 1920,
    height: int = 1080,
    leg_index: int = 0,
    transport_mode: str = "auto",
    compact_hud: bool = False,
    allow_tile_fetch: bool = False,
    cache_dir: Optional[str | Path] = None,
    annotations: Optional[list[dict[str, Any]]] = None,
    leg_via_points: Optional[dict[int, list[tuple[float, float]]]] = None,
    upload_cuda: bool = False,
) -> _PreparedLegContext:
    """Precompute static world atlas, OSRM routes, VRAM tensor & zero-teleport layout once per leg."""
    norm_wps: list[MapWaypointSpec] = []
    for idx_w, w in enumerate(waypoints):
        if isinstance(w, MapWaypointSpec):
            norm_wps.append(w)
        else:
            raw_vias = w.get("via_points") or []
            vias = [(float(vp_i[0]), float(vp_i[1])) for vp_i in raw_vias if len(vp_i) >= 2]
            norm_wps.append(
                MapWaypointSpec(
                    waypoint_id=str(w.get("waypoint_id") or f"wp_{idx_w + 1:02d}"),
                    chapter_id=str(w.get("chapter_id") or ""),
                    label=str(w.get("label") or w.get("name") or f"Stop {idx_w + 1}"),
                    subtitle=str(w.get("subtitle") or ""),
                    lat=float(w.get("lat", 37.77 + idx_w * 0.35)),
                    lon=float(w.get("lon", -122.42 + idx_w * 0.45)),
                    alt_m=float(w.get("alt_m", 40.0)),
                    source=w.get("source", "exif_gps") if w.get("source") in ("exif_gps", "gazetteer", "synthetic") else "exif_gps",
                    hero_thumbnail_path=w.get("hero_thumbnail_path"),
                    via_points=vias,
                )
            )
    if len(norm_wps) < 2:
        norm_wps = [
            MapWaypointSpec(waypoint_id="wp_01", label="Origin", lat=35.6762, lon=139.6503, alt_m=40.0),
            MapWaypointSpec(waypoint_id="wp_02", label="Destination", lat=35.0116, lon=135.7681, alt_m=50.0),
        ]

    i0 = max(0, min(len(norm_wps) - 2, leg_index))
    i1 = i0 + 1
    orig = norm_wps[i0]
    dest = norm_wps[i1]
    leg_km = haversine_km(orig.lat, orig.lon, dest.lat, dest.lon)
    resolved_mode = infer_transport_mode(leg_km, orig.label, dest.label, hint=transport_mode)

    if style in CART_THEME_SPECS:
        use_style = style
    else:
        lbl_blob = " ".join(f"{w.label} {w.subtitle}" for w in norm_wps)
        use_style = resolve_travel_map_style(
            style or "auto",
            prompt_text=lbl_blob,
            transport_mode=resolved_mode,
            leg_km=leg_km,
        )
    spec = CART_THEME_SPECS[use_style]

    oversample = 1.30 if width > 2000 else 1.60
    world_w = max(640, min(5184, int(round(width * oversample))))
    world_h = max(360, min(2916, int(round(height * oversample))))
    wp_sig = tuple((round(w.lat, 5), round(w.lon, 5), round(w.alt_m, 1), w.label) for w in norm_wps)

    focus_leg = max(0, min(len(norm_wps) - 2, leg_index)) if len(norm_wps) >= 2 else -1
    cache_dir_s = str(Path(cache_dir).resolve()) if cache_dir else ""
    base_rgb = _cached_world_baseplate(
        use_style, world_w, world_h, wp_sig, bool(allow_tile_fetch), cache_dir_s, focus_leg
    )
    vp = _compute_viewport(norm_wps, world_w, world_h, focus_leg_index=focus_leg)

    px_stops = [vp.latlon_to_px(w.lat, w.lon) for w in norm_wps]
    c_dir_p = Path(cache_dir).resolve() if cache_dir else None
    all_leg_splines: list[list[tuple[float, float]]] = []
    active_latlons: list[tuple[float, float]] = []
    for k in range(len(px_stops) - 1):
        leg_mode_k = resolved_mode if k == i0 else infer_transport_mode(
            haversine_km(norm_wps[k].lat, norm_wps[k].lon, norm_wps[k + 1].lat, norm_wps[k + 1].lon),
            norm_wps[k].label, norm_wps[k + 1].label, hint=transport_mode,
        )
        vias_k = list((leg_via_points or {}).get(k) or norm_wps[k + 1].via_points or [])
        road_lls: Optional[list[tuple[float, float]]] = None
        if allow_tile_fetch and os.environ.get("AI_EDITOR_OFFLINE_MAPS") != "1":
            if leg_mode_k in ("drive", "walk", "train"):
                road_lls = _fetch_osrm_route_latlon(
                    norm_wps[k], norm_wps[k + 1],
                    cache_dir=c_dir_p,
                    profile="foot" if leg_mode_k == "walk" else "driving",
                    via_latlons=vias_k or None,
                )
        if road_lls:
            all_leg_splines.append([vp.latlon_to_px(lat, lon) for lat, lon in road_lls])
            if k == i0:
                active_latlons = road_lls
        elif leg_mode_k == "flight":
            gc_lls = _great_circle_latlon_pts(norm_wps[k].lat, norm_wps[k].lon, norm_wps[k + 1].lat, norm_wps[k + 1].lon, n_samples=160)
            all_leg_splines.append(_project_flight_latlons_to_px(vp, norm_wps[k], norm_wps[k + 1], gc_lls))
            if k == i0:
                active_latlons = gc_lls
        else:
            ctrl_pts = [px_stops[k]] + [vp.latlon_to_px(la, lo) for la, lo in vias_k] + [px_stops[k + 1]]
            spl = _catmull_rom_xy(ctrl_pts, samples_per_seg=100 if vias_k else 140)
            all_leg_splines.append(spl)
            if k == i0:
                active_latlons = [vp.px_to_latlon(px_s, py_s) for px_s, py_s in spl[::4]]

    # Compute exact total path length in km along the actual spline (instead of crow-flies haversine!)
    route_total_km = leg_km
    if len(active_latlons) >= 2:
        km_sum = sum(
            haversine_km(active_latlons[j][0], active_latlons[j][1], active_latlons[j + 1][0], active_latlons[j + 1][1])
            for j in range(len(active_latlons) - 1)
        )
        if km_sum > 0.01:
            route_total_km = km_sum

    # Bake static history legs + dashed active future track + place labels overlay ONCE onto static_world_rgb
    world_rgba = base_rgb.convert("RGBA")
    route_arr = np.zeros((world_h, world_w, 4), dtype=np.uint8)
    s_w = world_w / 1920.0
    stroke_w = max(2, int(round(5.0 * s_w)))
    for k, seg_pts in enumerate(all_leg_splines):
        if k < i0:
            hist_km = haversine_km(norm_wps[k].lat, norm_wps[k].lon, norm_wps[k + 1].lat, norm_wps[k + 1].lon)
            if leg_km < 120.0 and hist_km > max(150.0, leg_km * 30.0):
                continue
            _draw_aa_polyline(route_arr, seg_pts, (*spec["route_history"], 165), max(2, stroke_w - 1))
        elif k == i0:
            _draw_dashed_aa_polyline(
                route_arr,
                seg_pts,
                (*spec["route_future"], 145 if resolved_mode != "flight" else 118),
                max(1, stroke_w // 2 + 1),
                dash_px=11.0 * s_w,
                gap_px=8.0 * s_w,
            )
    world_rgba = Image.alpha_composite(world_rgba, Image.fromarray(route_arr))

    labels_ov = _cached_labels_overlay(use_style, world_w, world_h, wp_sig, bool(allow_tile_fetch), cache_dir_s, focus_leg)
    if labels_ov is not None:
        l_rgb = np.asarray(labels_ov.convert("RGB"), dtype=np.float32)
        l_a = np.asarray(labels_ov.getchannel("A"), dtype=np.float32) / 255.0
        if float(l_a.max()) > 0.01:
            shadow_a = np.clip(cv2.dilate((l_a * 255.0).astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.float32) / 255.0 * 0.72, 0.0, 0.72)
            base_w_arr = np.asarray(world_rgba.convert("RGB"), dtype=np.float32) * (1.0 - shadow_a[..., None])
            lbl_mix = np.clip(l_a * 0.96, 0.0, 1.0)[..., None]
            world_rgba = Image.fromarray(np.clip(base_w_arr * (1.0 - lbl_mix) + l_rgb * lbl_mix, 0, 255).astype(np.uint8)).convert("RGBA")

    static_world_rgb = world_rgba.convert("RGB")
    static_cuda_t = _upload_world_to_cuda(np.asarray(static_world_rgb, dtype=np.uint8)) if (upload_cuda and (width * height) >= 400_000) else None

    ground_arc_path = _ArcLengthPath(all_leg_splines[i0])
    use_globe = (
        not compact_hud
        and resolved_mode == "flight"
        and leg_km >= 180.0
        and use_style in ("satellite_hybrid_4k", "aerial_relief_hybrid", "midnight_vox_atlas", "archival_expedition_parchment")
    )
    crisp_real_tiles = bool(allow_tile_fetch and os.environ.get("AI_EDITOR_OFFLINE_MAPS") != "1")
    if compact_hud:
        pitch_deg = 0.0 if crisp_real_tiles else 14.0
    elif use_globe:
        pitch_deg = 24.0
    elif crisp_real_tiles:
        pitch_deg = 7.5
    else:
        pitch_deg = 20.0

    # Pre-decode Polaroid thumbnails ONCE per leg (instead of decoding 33MP JPEGs 180 times!)
    orig_thumb_rgb: Optional[Image.Image] = None
    dest_thumb_rgb: Optional[Image.Image] = None
    if orig.hero_thumbnail_path and Path(orig.hero_thumbnail_path).exists():
        try:
            orig_thumb_rgb = _load_clean_callout_rgb(Path(orig.hero_thumbnail_path))
        except Exception:
            orig_thumb_rgb = None
    if dest.hero_thumbnail_path and Path(dest.hero_thumbnail_path).exists():
        try:
            dest_thumb_rgb = _load_clean_callout_rgb(Path(dest.hero_thumbnail_path))
        except Exception:
            dest_thumb_rgb = None

    layout_plan = _solve_leg_layout_plan(
        norm_wps=norm_wps, i0=i0, i1=i1, px_stops=px_stops,
        ground_arc_path=ground_arc_path, vp=vp, spec=spec,
        width=width, height=height, world_w=world_w, world_h=world_h,
        leg_km=leg_km, use_globe=use_globe, pitch_deg=pitch_deg,
        compact_hud=compact_hud, annotations=annotations,
        has_orig_thumb=orig_thumb_rgb is not None,
        has_dest_thumb=dest_thumb_rgb is not None,
    )

    return _PreparedLegContext(
        use_style=use_style, spec=spec, norm_wps=norm_wps, i0=i0, i1=i1,
        width=width, height=height, world_w=world_w, world_h=world_h,
        vp=vp, px_stops=px_stops, all_leg_splines=all_leg_splines,
        ground_arc_path=ground_arc_path, leg_km=leg_km, route_total_km=route_total_km,
        resolved_mode=resolved_mode, use_globe=use_globe, pitch_deg=pitch_deg,
        compact_hud=compact_hud, static_world_rgb=static_world_rgb,
        static_cuda_t=static_cuda_t, orig_thumb_rgb=orig_thumb_rgb,
        dest_thumb_rgb=dest_thumb_rgb, layout_plan=layout_plan,
    )


def render_travel_map_frame(
    waypoints: list[MapWaypointSpec | dict[str, Any]],
    style: str = "auto",
    width: int = 1920,
    height: int = 1080,
    leg_index: int = 0,
    t_norm: float = 0.65,
    transport_mode: str = "auto",
    hdr_safe: bool = False,
    compact_hud: bool = False,
    allow_tile_fetch: bool = False,
    cache_dir: Optional[str | Path] = None,
    annotations: Optional[list[dict[str, Any]]] = None,
    leg_via_points: Optional[dict[int, list[tuple[float, float]]]] = None,
    _prepared: Optional[_PreparedLegContext] = None,
) -> Image.Image:
    """
    Render a single keyframe of the Cinema 2.5D Travel Map engine as an RGB PIL Image.

    Uses GPU-accelerated VRAM-cached baseplate warping with crisp post-warp screen-space routes,
    tapered comet-tail beacon, zero-teleport Polaroids, post-warp annotations, and accurate HUD telemetry.
    """
    ctx = _prepared if _prepared is not None else _prepare_travel_map_leg(
        waypoints=waypoints, style=style, width=width, height=height,
        leg_index=leg_index, transport_mode=transport_mode, compact_hud=compact_hud,
        allow_tile_fetch=allow_tile_fetch, cache_dir=cache_dir,
        annotations=annotations, leg_via_points=leg_via_points, upload_cuda=False,
    )

    spec = ctx.spec
    norm_wps = ctx.norm_wps
    i0, i1 = ctx.i0, ctx.i1
    orig, dest = norm_wps[i0], norm_wps[i1]
    vp = ctx.vp
    px_stops = ctx.px_stops
    ground_arc_path = ctx.ground_arc_path
    resolved_mode = ctx.resolved_mode

    u = _cartography_travel_progress(t_norm)
    prog_ground = ground_arc_path.sample_prefix(u)
    ground_head_pt = ground_arc_path.point_at_s(u * ground_arc_path.total_len)
    cur_latlon = vp.px_to_latlon(ground_head_pt[0], ground_head_pt[1])

    # 1. Smooth overview-breathing 2.5D GPU camera warp of static world baseplate
    cam_cx, cam_cy, cam_zoom = _compute_leg_camera_state(
        px_stops[i0], px_stops[i1], ground_arc_path, u, ctx.world_w, ctx.leg_km, ctx.use_globe, compact_hud
    )
    warped_rgb, proj_fn, horizon_y_fn = _apply_camera_homography(
        ctx.static_world_rgb,
        out_w=width,
        out_h=height,
        center_xy=(cam_cx, cam_cy),
        zoom=cam_zoom,
        pitch_deg=ctx.pitch_deg,
        bank_deg=0.0,
        globe_strength=1.0 if ctx.use_globe else 0.0,
        src_cuda_t=ctx.static_cuda_t,
        vignette_strength=0.0 if (compact_hud or ctx.use_globe) else 0.10,
        sharpen_amount=0.22 if not ctx.use_globe else 0.0,
    )
    out_rgba = Image.fromarray(warped_rgb).convert("RGBA")
    s_out = width / 1920.0
    screen_stops = [proj_fn(wx, wy) for wx, wy in px_stops]
    scr_route_all = [
        proj_fn(*ground_arc_path.point_at_s((idx_r / 23.0) * ground_arc_path.total_len))
        for idx_r in range(24)
    ]

    # 2. Razor-sharp post-warp active route + tapered comet tail + arrival radar + beacon/vehicle
    scr_ground = [proj_fn(gx, gy) for gx, gy in prog_ground]
    scr_head = proj_fn(ground_head_pt[0], ground_head_pt[1])
    stroke_w_scr = max(2, int(round(5.2 * s_out)))
    stroke_col = spec["pin_fill"] if not spec["light_paper"] else (248, 244, 238)
    veh_scr_scale = s_out * (0.90 if compact_hud else 1.08)

    route_scr_arr = np.zeros((height, width, 4), dtype=np.uint8)
    if resolved_mode == "flight":
        if len(scr_ground) >= 2:
            _draw_aa_polyline(route_scr_arr, scr_ground, (12, 18, 28, 128), max(2, stroke_w_scr))
            _draw_aa_polyline(route_scr_arr, scr_ground, (*spec["route_glow"], 115), max(1, stroke_w_scr // 2))
            out_rgba = Image.alpha_composite(out_rgba, Image.fromarray(route_scr_arr))
        alt_sin = math.sin(math.pi * u)
        g_heading, _ = ground_arc_path.tangent_and_curvature(u, window_px=24.0 * (ctx.world_w / 1920.0))
        _draw_aircraft_ground_shadow(
            out_rgba,
            gx=scr_head[0] + 10.0 * s_out * alt_sin,
            gy=scr_head[1] + 12.0 * s_out * alt_sin,
            angle_rad=g_heading,
            scale=veh_scr_scale * (0.92 - 0.28 * alt_sin),
            alpha=int(round(125 - 65 * alt_sin)),
        )
    else:
        if len(scr_ground) >= 2:
            _draw_cased_route(route_scr_arr, scr_ground, spec, stroke_w_scr, s_out, is_flight=False)
            out_rgba = Image.alpha_composite(out_rgba, Image.fromarray(route_scr_arr))

    # 3. Crisp post-warp surface pins, destination arrival radar pulse & pin numerals (true alpha layer)
    pin_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    od = ImageDraw.Draw(pin_layer, "RGBA")
    shadow_rgb = (68, 54, 38) if spec.get("light_paper") else (6, 10, 16)
    f_pin = _load_font_tier(max(10, int(round(13 * s_out))), tier="mono", bold=True)
    for k in [idx for idx in range(len(screen_stops)) if idx not in (i0, i1)] + [i0, i1]:
        sx_w, sy_w = screen_stops[k]
        if sx_w < -50 or sx_w > width + 50 or sy_w < -50 or sy_w > height + 50:
            continue
        is_orig, is_dest = (k == i0), (k == i1)
        if not (is_orig or is_dest):
            if min(math.hypot(sx_w - screen_stops[i0][0], sy_w - screen_stops[i0][1]), math.hypot(sx_w - screen_stops[i1][0], sy_w - screen_stops[i1][1])) < 26.0 * s_out:
                continue
        pin_r = max(6, int(round((12 if (is_orig or is_dest) else 8.5) * s_out)))
        if is_dest and u >= 0.70:
            pulse_p = (u - 0.70) / 0.30
            r_pulse = pin_r + int(round(32 * s_out * pulse_p))
            p_alpha = int(round(215 * (1.0 - pulse_p)))
            od.ellipse([sx_w - r_pulse, sy_w - r_pulse, sx_w + r_pulse, sy_w + r_pulse], outline=(*spec["accent"], p_alpha), width=max(1, int(round(2.5 * s_out))))
            r_pulse2 = pin_r + int(round(16 * s_out * pulse_p))
            od.ellipse([sx_w - r_pulse2, sy_w - r_pulse2, sx_w + r_pulse2, sy_w + r_pulse2], outline=(*spec["route_core"], int(round(p_alpha * 0.85))), width=max(1, int(round(2.0 * s_out))))
        ring_col = (*spec["accent"], 255) if (is_orig or is_dest) else (*spec["route_history"], 220)
        # Soft translucent shadow under pin
        od.ellipse([sx_w - pin_r - 2, sy_w - pin_r, sx_w + pin_r + 3, sy_w + pin_r + 5], fill=(*shadow_rgb, 62))
        od.ellipse([sx_w - pin_r - 3, sy_w - pin_r - 3, sx_w + pin_r + 3, sy_w + pin_r + 3], fill=ring_col)
        od.ellipse([sx_w - pin_r, sy_w - pin_r, sx_w + pin_r, sy_w + pin_r], fill=(*spec["pin_fill"], 250))
        num_str = f"{k + 1:02d}"
        nw, nh = measure_tracked_text(od, num_str, f_pin)
        od.text((sx_w - nw * 0.5, sy_w - nh * 0.62), num_str, font=f_pin, fill=(*spec["pin_text"], 252))
    out_rgba.alpha_composite(pin_layer)

    # 4. Pulsing glowing beacon (ground modes) OR boat wake OR elevated 3D flight arc + jetliner
    if resolved_mode == "flight":
        alt_sin = math.sin(math.pi * u)
        if len(scr_ground) >= 2:
            gx0, gy0 = screen_stops[i0]
            gx1, gy1 = screen_stops[i1]
            scr_span = math.hypot(gx1 - gx0, gy1 - gy0)
            lift_px = min(height * 0.25, max(54.0 * s_out, scr_span * 0.26))
            cum_s = [0.0]
            for j in range(1, len(scr_ground)):
                cum_s.append(cum_s[-1] + math.hypot(scr_ground[j][0] - scr_ground[j - 1][0], scr_ground[j][1] - scr_ground[j - 1][1]))
            tot_s = max(1e-6, cum_s[-1] / max(1e-4, u))
            scr_arc_pts = [(sx_g, sy_g - lift_px * math.sin(math.pi * min(1.0, cum_s[j] / tot_s))) for j, (sx_g, sy_g) in enumerate(scr_ground)]

            sky_arr = np.zeros((height, width, 4), dtype=np.uint8)
            if len(scr_ground) >= 12:
                step_teth = max(8, len(scr_ground) // 6)
                for j in range(step_teth, len(scr_ground) - 2, step_teth):
                    _draw_dashed_aa_polyline(sky_arr, [scr_ground[j], scr_arc_pts[j]], (*spec["route_future"], 92), max(1, int(round(1.4 * s_out))), dash_px=4.0 * s_out, gap_px=5.0 * s_out)
            _draw_cased_route(sky_arr, scr_arc_pts, spec, max(2, int(round(5.5 * s_out))), s_out, is_flight=False)
            out_rgba = Image.alpha_composite(out_rgba, Image.fromarray(sky_arr))

            elev_scr_path = _ArcLengthPath(scr_arc_pts)
            e_head = elev_scr_path.point_at_s(elev_scr_path.total_len)
            scr_heading, scr_bank = elev_scr_path.tangent_and_curvature(1.0, window_px=26.0 * s_out)
            jet_scale = s_out * (0.95 if compact_hud else 1.12) * (0.86 + 0.44 * alt_sin)
            _draw_flight_contrails(out_rgba, scr_arc_pts, scale=jet_scale, alt_sin=alt_sin)
            _draw_transport_icon(out_rgba, cx=e_head[0], cy=e_head[1], angle_rad=scr_heading, mode="flight", fill_rgb=spec["route_core"], stroke_rgb=stroke_col, scale=jet_scale, bank_factor=scr_bank)
    elif resolved_mode in ("train", "walk", "drive"):
        _draw_glow_beacon_dot(
            out_rgba,
            cx=scr_head[0],
            cy=scr_head[1],
            glow_rgb=spec["route_glow"],
            core_rgb=spec["accent"],
            stroke_rgb=stroke_col,
            scale=veh_scr_scale,
            pulse_phase=float(t_norm) * 2.5,
        )
    elif resolved_mode == "boat":
        scr_arc_p = _ArcLengthPath(scr_ground) if len(scr_ground) >= 2 else _ArcLengthPath([scr_head, (scr_head[0] + 1.0, scr_head[1])])
        scr_heading, scr_bank = scr_arc_p.tangent_and_curvature(1.0, window_px=22.0 * s_out)
        _draw_boat_kelvin_wake(out_rgba, arc_path=scr_arc_p, u=1.0, scale=veh_scr_scale)
        _draw_transport_icon(out_rgba, cx=scr_head[0], cy=scr_head[1], angle_rad=scr_heading, mode="boat", fill_rgb=spec["route_core"], stroke_rgb=stroke_col, scale=veh_scr_scale, bank_factor=scr_bank)
    else:
        scr_arc_p = _ArcLengthPath(scr_ground) if len(scr_ground) >= 2 else _ArcLengthPath([scr_head, (scr_head[0] + 1.0, scr_head[1])])
        scr_heading, scr_bank = scr_arc_p.tangent_and_curvature(1.0, window_px=22.0 * s_out)
        _draw_transport_icon(out_rgba, cx=scr_head[0], cy=scr_head[1], angle_rad=scr_heading, mode=resolved_mode, fill_rgb=spec["route_core"], stroke_rgb=stroke_col, scale=veh_scr_scale, bank_factor=scr_bank)

    # 5. Zero-Teleport Collision-Free Waypoint Labels, Smooth FROM/TO Polaroids & Crisp Screen-Space Annotations
    od = ImageDraw.Draw(out_rgba, "RGBA")
    all_lbl_txt = " ".join(w.label for w in norm_wps)
    f_wp_lbl = _load_font_tier(max(11, int(round(17 * s_out))), tier=spec["title_tier"], bold=True, text=all_lbl_txt)

    occupied_rects = _base_occupied_rects(width, height, horizon_y_fn, compact_hud=compact_hud)
    for sx_w, sy_w in screen_stops:
        if -40 <= sx_w <= width + 40 and -40 <= sy_w <= height + 40:
            pr = max(12.0, 22.0 * s_out)
            occupied_rects.append((sx_w - pr, sy_w - pr, sx_w + pr, sy_w + pr))

    def _overlaps_any(box: tuple[float, float, float, float], pad: float = 4.0) -> bool:
        bx0, by0, bx1, by1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
        return any(not (bx1 < rx0 or bx0 > rx1 or by1 < ry0 or by0 > ry1) for rx0, ry0, rx1, ry1 in occupied_rects)

    pending_labels: list[tuple[tuple[int, int, int, int], int, int, str, tuple[int, int, int, int], bool, bool]] = []
    tag_order = [i0, i1] + [k for k in range(len(screen_stops)) if k not in (i0, i1)]
    for k in tag_order:
        sx_w, sy_w = screen_stops[k]
        if sx_w < -60 or sx_w > width + 60 or sy_w < -60 or sy_w > height + 60:
            continue
        is_orig, is_dest, is_done = (k == i0), (k == i1), (k <= i0)
        if not (is_orig or is_dest):
            if min(math.hypot(sx_w - screen_stops[i0][0], sy_w - screen_stops[i0][1]), math.hypot(sx_w - screen_stops[i1][0], sy_w - screen_stops[i1][1])) < 72.0 * s_out:
                continue

        pin_r = max(6.0, float(round((12 if (is_orig or is_dest) else 8.5) * s_out)))
        ring_col = (*spec["accent"], 255) if (is_orig or is_dest) else (*spec["route_history"], 220)
        tag = norm_wps[k].label
        tb0 = od.textbbox((0, 0), tag, font=f_wp_lbl)
        tw, th = float(tb0[2] - tb0[0]), float(tb0[3] - tb0[1])
        pad_x, pad_y = int(round(10 * s_out)), int(round(7 * s_out))
        candidates = _waypoint_label_candidates(sx_w, sy_w, pin_r, tw, th, tb0, s_out, is_dest)

        locked_idx = ctx.layout_plan.label_slots.get(k)
        if locked_idx is not None and 0 <= locked_idx < len(candidates):
            cx_c, cy_c = candidates[locked_idx]
            tx = int(round(max(18.0, min(float(width - tw - 22), cx_c))))
            ty = int(round(max(float(height * 0.19), min(float(height - th - 20), cy_c))))
            tb_l = od.textbbox((tx, ty), tag, font=f_wp_lbl)
            occupied_rects.append((tb_l[0] - pad_x, tb_l[1] - pad_y, tb_l[2] + pad_x, tb_l[3] + pad_y))
        else:
            chosen: Optional[tuple[int, int]] = None
            for cx_c, cy_c in candidates:
                tx_c = int(round(max(18.0, min(float(width - tw - 22), cx_c))))
                ty_c = int(round(max(float(height * 0.19), min(float(height - th - 20), cy_c))))
                tb_c = od.textbbox((tx_c, ty_c), tag, font=f_wp_lbl)
                cand_box = (tb_c[0] - pad_x, tb_c[1] - pad_y, tb_c[2] + pad_x, tb_c[3] + pad_y)
                if not _overlaps_any(cand_box):
                    chosen = (tx_c, ty_c)
                    occupied_rects.append(cand_box)
                    break
            if chosen is None:
                if not (is_orig or is_dest):
                    continue
                tx = int(round(max(18.0, min(float(width - tw - 22), candidates[0][0]))))
                ty = int(round(max(float(height * 0.21), min(float(height - th - 20), candidates[0][1]))))
                tb_f = od.textbbox((tx, ty), tag, font=f_wp_lbl)
                occupied_rects.append((tb_f[0] - pad_x, tb_f[1] - pad_y, tb_f[2] + pad_x, tb_f[3] + pad_y))
            else:
                tx, ty = chosen

        tb = od.textbbox((tx, ty), tag, font=f_wp_lbl)
        pending_labels.append(
            ((tb[0] - pad_x, tb[1] - pad_y, tb[2] + pad_x, tb[3] + pad_y), tx, ty, tag, ring_col, is_orig or is_dest, is_done or is_dest)
        )

    # Smooth FROM -> TO Polaroid Choreography (zero teleport, route-line avoidance, graceful FROM fade-out!)
    orig_card_center: Optional[tuple[float, float]] = None
    if not compact_hud and (ctx.orig_thumb_rgb is not None or orig.hero_thumbnail_path):
        intro_p = min(1.0, 0.78 + float(t_norm) * 2.4)
        # Gracefully fade FROM out as arrival approaches so it never crowds or pins against screen edges
        fade_out_p = 1.0 if u <= 0.48 else max(0.0, 1.0 - (u - 0.48) / 0.24)
        orig_bloom = intro_p * fade_out_p
        if orig_bloom > 0.02:
            orig_card_center = _paste_hero_photo_callout(
                world_rgba=out_rgba,
                dest_px=screen_stops[i0],
                thumb_path=orig.hero_thumbnail_path,
                label=orig.label,
                bloom_p=orig_bloom,
                spec=spec,
                world_w=width,
                world_h=height,
                avoid_pts=[p for idx_p, p in enumerate(screen_stops) if idx_p != i0],
                caption_prefix="FROM",
                tilt_deg=-1.6,
                accent_override=spec["route_history"],
                avoid_rects=occupied_rects,
                route_avoid_pts=scr_route_all,
                locked_slot_idx=ctx.layout_plan.orig_polaroid_slot,
                preloaded_thumb_rgb=ctx.orig_thumb_rgb,
            )

    if not compact_hud and u >= 0.44 and (ctx.dest_thumb_rgb is not None or dest.hero_thumbnail_path):
        bloom_p = min(1.0, (u - 0.44) / 0.28)
        dest_avoid = [p for idx_p, p in enumerate(screen_stops) if idx_p != i1]
        if orig_card_center is not None:
            dest_avoid.append(orig_card_center)
        _paste_hero_photo_callout(
            world_rgba=out_rgba,
            dest_px=screen_stops[i1],
            thumb_path=dest.hero_thumbnail_path,
            label=dest.label,
            bloom_p=bloom_p,
            spec=spec,
            world_w=width,
            world_h=height,
            avoid_pts=dest_avoid,
            caption_prefix="TO",
            tilt_deg=2.0,
            avoid_rects=occupied_rects,
            route_avoid_pts=scr_route_all,
            locked_slot_idx=ctx.layout_plan.dest_polaroid_slot,
            preloaded_thumb_rgb=ctx.dest_thumb_rgb,
        )

    # Crisp post-warp screen-space corridor annotations (sharing unified collision layout!)
    if not compact_hud and annotations:
        _draw_route_annotations_screen(
            out_rgba=out_rgba,
            annotations=annotations,
            vp=vp,
            proj_fn=proj_fn,
            u=u,
            spec=spec,
            width=width,
            height=height,
            occupied_rects=occupied_rects,
            route_avoid_pts=scr_route_all,
            locked_offsets=ctx.layout_plan.annotation_offsets,
        )

    # Draw waypoint pills cleanly on top (true alpha shadow & frosted pill layer)
    if pending_labels:
        pill_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        od_top = ImageDraw.Draw(pill_layer, "RGBA")
        for pill_box, tx, ty, tag, ring_col, is_active_pin, is_bright in pending_labels:
            od_top.rounded_rectangle(
                [pill_box[0] + 2, pill_box[1] + 4, pill_box[2] + 3, pill_box[3] + 5],
                radius=max(4, int(round(6 * s_out))),
                fill=(*shadow_rgb, 48 if is_bright else 28),
            )
            od_top.rounded_rectangle(
                list(pill_box),
                radius=max(4, int(round(6 * s_out))),
                fill=(*spec["hud_fill"][:3], 240 if is_bright else 188),
                outline=ring_col if is_active_pin else (*spec["hud_border"][:3], 130),
                width=max(1, int(round(1.4 * s_out))) if is_active_pin else 1,
            )
            od_top.text((tx, ty), tag, font=f_wp_lbl, fill=(*spec["hud_title"], 252))
        out_rgba.alpha_composite(pill_layer)

    # 6. Composite crisp screen-space HUD, anchored compass rose & Journey/Elevation telemetry card
    _render_screen_hud(
        frame_rgba=out_rgba,
        style=ctx.use_style,
        waypoints=norm_wps,
        leg_index=i0,
        t_norm=t_norm,
        transport_mode=resolved_mode,
        compact_hud=compact_hud,
        hdr_safe=hdr_safe,
        route_total_km=ctx.route_total_km,
        cur_latlon=cur_latlon,
        draw_compass=not compact_hud,
    )

    out_arr = np.asarray(out_rgba.convert("RGB"), dtype=np.uint8)
    if hdr_safe:
        max_ch = np.max(out_arr, axis=2, keepdims=True).astype(np.float32)
        scale = np.where(max_ch > float(HDR_DIFFUSE_WHITE_PEAK_RGB), float(HDR_DIFFUSE_WHITE_PEAK_RGB) / np.maximum(1.0, max_ch), 1.0)
        out_arr = np.clip(np.round(out_arr.astype(np.float32) * scale), 0, HDR_DIFFUSE_WHITE_PEAK_RGB).astype(np.uint8)
    return Image.fromarray(out_arr)


def _synthesize_flyover_wav(output_wav: Path, duration_sec: float, sample_rate: int = 48000) -> Path:
    """Synthesize subtle stereo cartography wind sweep + arrival harmonic ping for standalone reels."""
    import wave

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.8, float(duration_sec))
    n = int(round(dur * sample_rate))
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(42)
    w_l = rng.normal(0.0, 1.0, n).astype(np.float32)
    w_r = rng.normal(0.0, 1.0, n).astype(np.float32)
    kern = np.ones(48, dtype=np.float32) / 48.0
    b_l = np.convolve(w_l, kern, mode="same")
    b_r = np.convolve(w_r, kern, mode="same")

    pan_r = np.clip(0.25 + 0.50 * (t / dur), 0.15, 0.85)
    pan_l = 1.0 - pan_r
    swell = np.exp(-((t - dur * 0.42) ** 2) / (2.0 * (dur * 0.22) ** 2))
    ping_t0 = dur * 0.76
    dt_p = np.maximum(0.0, t - ping_t0)
    ping_env = np.where(t >= ping_t0, np.exp(-dt_p * 3.8), 0.0)
    ping = 0.09 * np.sin(2.0 * math.pi * 660.0 * dt_p) + 0.045 * np.sin(2.0 * math.pi * 1320.0 * dt_p)

    sig_l = b_l * swell * pan_l * 0.32 + ping * ping_env * 0.6
    sig_r = b_r * swell * pan_r * 0.32 + ping * ping_env * 0.6
    stereo = np.stack([sig_l, sig_r], axis=-1)
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return output_wav


# ---------------------------------------------------------------------------
# Multi-Resolution Google Earth Dynamic-Zoom Pyramid Engine
# ---------------------------------------------------------------------------

_PYRAMID_TILE_MEM: dict[tuple[str, int, int, int], Optional[np.ndarray]] = {}


def _load_or_fetch_pyramid_tile(
    layer_tag: str,
    url_tpl: str,
    z: int,
    tx: int,
    ty: int,
    mode: str = "RGB",
    cache_dir: Optional[Path] = None,
    allow_fetch: bool = True,
) -> Optional[np.ndarray]:
    """Fetch and disk/memory-cache a single 256x256 slippy tile for dynamic multi-zoom sampling."""
    n = 1 << z
    if ty < 0 or ty >= n:
        return None
    tx_w = tx % n
    key = (layer_tag, z, tx_w, ty)
    if key in _PYRAMID_TILE_MEM:
        return _PYRAMID_TILE_MEM[key]

    zdir = (cache_dir or Path("cache")) / "map_tiles" / layer_tag / str(z)
    zdir.mkdir(parents=True, exist_ok=True)
    local_p = zdir / f"{tx_w}_{ty}.png"
    if not local_p.exists() and allow_fetch and os.environ.get("AI_EDITOR_OFFLINE_MAPS") != "1":
        url = url_tpl.format(z=z, x=tx_w, y=ty)
        data = _http_fetch_bytes(url, timeout=3.5)
        if data is not None and len(data) > 120:
            try:
                local_p.write_bytes(data)
            except Exception:
                _PYRAMID_TILE_MEM[key] = None
                return None
        else:
            _PYRAMID_TILE_MEM[key] = None
            return None
    if not local_p.exists():
        _PYRAMID_TILE_MEM[key] = None
        return None
    try:
        with Image.open(local_p) as im:
            arr = np.asarray(im.convert(mode), dtype=np.uint8)
        _PYRAMID_TILE_MEM[key] = arr
        return arr
    except Exception:
        _PYRAMID_TILE_MEM[key] = None
        return None


def _sample_pyramid_mosaic(
    layer_tag: str,
    url_tpl: str,
    z: int,
    wx0: float,
    wy0: float,
    wx1: float,
    wy1: float,
    out_w: int,
    out_h: int,
    mode: str = "RGB",
    fill_val: tuple[int, ...] = (12, 22, 36),
    cache_dir: Optional[Path] = None,
    allow_fetch: bool = True,
) -> np.ndarray:
    """Assemble and sub-pixel warp exact Web Mercator region [wx0..wx1, wy0..wy1] at zoom z."""
    n = 1 << z
    tx_min = int(math.floor(wx0 / 256.0))
    tx_max = int(math.floor(wx1 / 256.0))
    ty_min = max(0, int(math.floor(wy0 / 256.0)))
    ty_max = min(n - 1, int(math.floor(wy1 / 256.0)))

    ch = 3 if mode == "RGB" else 4
    cw = max(256, (tx_max - tx_min + 1) * 256)
    ch_px = max(256, (ty_max - ty_min + 1) * 256)
    canvas = np.zeros((ch_px, cw, ch), dtype=np.uint8)
    canvas[:, :] = fill_val

    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            t_arr = _load_or_fetch_pyramid_tile(
                layer_tag, url_tpl, z, tx, ty, mode=mode, cache_dir=cache_dir, allow_fetch=allow_fetch
            )
            if t_arr is not None:
                oy = (ty - ty_min) * 256
                ox = (tx - tx_min) * 256
                canvas[oy : oy + 256, ox : ox + 256] = t_arr

    ox = wx0 - tx_min * 256.0
    oy = wy0 - ty_min * 256.0
    ow = max(4.0, wx1 - wx0)
    oh = max(4.0, wy1 - wy0)
    src_pts = np.array([[ox, oy], [ox + ow, oy], [ox, oy + oh]], dtype=np.float32)
    dst_pts = np.array([[0.0, 0.0], [float(out_w), 0.0], [0.0, float(out_h)]], dtype=np.float32)
    m_aff = cv2.getAffineTransform(src_pts, dst_pts)
    return cv2.warpAffine(canvas, m_aff, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def _normalize_flyover_waypoints(waypoints: list[MapWaypointSpec | dict[str, Any]]) -> list[MapWaypointSpec]:
    norm_wps: list[MapWaypointSpec] = []
    for idx_w, w in enumerate(waypoints):
        if isinstance(w, MapWaypointSpec):
            norm_wps.append(w)
        else:
            norm_wps.append(
                MapWaypointSpec(
                    waypoint_id=str(w.get("waypoint_id") or f"wp_{idx_w + 1:02d}"),
                    chapter_id=str(w.get("chapter_id") or ""),
                    label=str(w.get("label") or w.get("name") or f"Stop {idx_w + 1}"),
                    subtitle=str(w.get("subtitle") or ""),
                    lat=float(w.get("lat", 37.77 + idx_w * 0.35)),
                    lon=float(w.get("lon", -122.42 + idx_w * 0.45)),
                    alt_m=float(w.get("alt_m", 40.0)),
                    source=w.get("source", "exif_gps") if w.get("source") in ("exif_gps", "gazetteer", "synthetic") else "exif_gps",
                    hero_thumbnail_path=w.get("hero_thumbnail_path"),
                )
            )
    if len(norm_wps) < 2:
        norm_wps = [
            MapWaypointSpec(waypoint_id="wp_01", label="Origin", lat=35.6762, lon=139.6503, alt_m=40.0),
            MapWaypointSpec(waypoint_id="wp_02", label="Destination", lat=35.0116, lon=135.7681, alt_m=50.0),
        ]
    return norm_wps


def _compute_google_earth_camera(
    norm_wps: list[MapWaypointSpec],
    leg_splines_ll: list[list[tuple[float, float]]],
    resolved_modes: list[str],
    global_t: float,
    width: int,
    height: int,
    leg_index: Optional[int] = None,
) -> tuple[int, float, float, float, float, float, float]:
    """Compute `(active_leg, leg_u, center_lat, center_lon, zoom_float, pitch_deg, globe_strength)`."""
    n_legs = len(norm_wps) - 1
    g_u = _cartography_travel_progress(global_t)
    focused = (n_legs == 1) or (leg_index is not None)
    if leg_index is not None:
        active_leg = max(0, min(n_legs - 1, int(leg_index)))
        leg_u = g_u
    else:
        scaled = g_u * n_legs
        active_leg = min(n_legs - 1, int(math.floor(scaled)))
        leg_u = 1.0 if g_u >= 1.0 else (scaled - active_leg)

    cur_mode = resolved_modes[active_leg]
    leg_span_km = haversine_km(
        norm_wps[active_leg].lat, norm_wps[active_leg].lon,
        norm_wps[active_leg + 1].lat, norm_wps[active_leg + 1].lon,
    )
    total_span_km = max(
        haversine_km(norm_wps[0].lat, norm_wps[0].lon, norm_wps[-1].lat, norm_wps[-1].lon),
        max((haversine_km(norm_wps[k].lat, norm_wps[k].lon, norm_wps[k + 1].lat, norm_wps[k + 1].lon) for k in range(n_legs)), default=1.0),
    )

    if cur_mode == "flight" and leg_span_km >= 250.0 and focused:
        pts_ll = leg_splines_ll[active_leg]
        idx_f = leg_u * (len(pts_ll) - 1)
        ia = min(len(pts_ll) - 2, int(math.floor(idx_f)))
        f = idx_f - ia
        center_lat = pts_ll[ia][0] + (pts_ll[ia + 1][0] - pts_ll[ia][0]) * f
        center_lon = pts_ll[ia][1] + (pts_ll[ia + 1][1] - pts_ll[ia][1]) * f
        cruise_dip = math.sin(math.pi * leg_u) ** 0.72
        z_cruise = 3.6 if leg_span_km > 2500.0 else 5.4
        zoom_float = (8.8 * (1.0 - leg_u) + 9.6 * leg_u) * (1.0 - cruise_dip) + z_cruise * cruise_dip
        globe_strength = float(np.clip((6.8 - zoom_float) / 2.6, 0.0, 1.0))
        pitch_deg = 16.0 + 10.0 * globe_strength
    elif focused and leg_span_km < 8.0:
        # Serene urban walk/transit glide: keep both origin & destination gracefully composed with quintic easing
        fit_vp = _compute_viewport(
            [norm_wps[active_leg], norm_wps[active_leg + 1]],
            int(round(width * 1.35)),
            int(round(height * 1.35)),
            focus_leg_index=0,
        )
        start_z = float(min(15.2, max(13.85, fit_vp.zoom + 0.25)))
        e = _ease_in_out_quintic(leg_u)
        mid_lat = 0.5 * (norm_wps[active_leg].lat + norm_wps[active_leg + 1].lat)
        mid_lon = 0.5 * (norm_wps[active_leg].lon + norm_wps[active_leg + 1].lon)
        c_start_lat = 0.62 * norm_wps[active_leg].lat + 0.38 * mid_lat
        c_start_lon = 0.62 * norm_wps[active_leg].lon + 0.38 * mid_lon
        c_end_lat = 0.20 * mid_lat + 0.80 * norm_wps[active_leg + 1].lat
        c_end_lon = 0.20 * mid_lon + 0.80 * norm_wps[active_leg + 1].lon
        center_lat = c_start_lat + (c_end_lat - c_start_lat) * e
        center_lon = c_start_lon + (c_end_lon - c_start_lon) * e
        zoom_float = start_z + 0.55 * e
        pitch_deg = 6.5 + 5.0 * e
        globe_strength = 0.0
    elif focused and leg_span_km >= 8.0:
        # Regional rail/drive/boat corridor + Google Earth destination zoom-in over the final 28%
        fit_vp = _compute_viewport([norm_wps[active_leg], norm_wps[active_leg + 1]], int(round(width * 1.35)), int(round(height * 1.35)), focus_leg_index=0)
        base_z = float(min(14.5, max(7.0, fit_vp.zoom + 0.25)))
        mid_lat = 0.5 * (norm_wps[active_leg].lat + norm_wps[active_leg + 1].lat)
        mid_lon = 0.5 * (norm_wps[active_leg].lon + norm_wps[active_leg + 1].lon)
        pts_ll = leg_splines_ll[active_leg]
        if global_t < 0.72:
            u_c = _cartography_travel_progress(global_t / 0.72)
            ia = min(len(pts_ll) - 2, int(math.floor(u_c * (len(pts_ll) - 1))))
            center_lat = 0.65 * pts_ll[ia][0] + 0.35 * mid_lat
            center_lon = 0.65 * pts_ll[ia][1] + 0.35 * mid_lon
            zoom_float = base_z
            pitch_deg = 8.0
        else:
            e = _ease_in_out_cubic((global_t - 0.72) / 0.28)
            c0_lat = 0.65 * pts_ll[-1][0] + 0.35 * mid_lat
            c0_lon = 0.65 * pts_ll[-1][1] + 0.35 * mid_lon
            center_lat = c0_lat + (norm_wps[active_leg + 1].lat - c0_lat) * e
            center_lon = c0_lon + (norm_wps[active_leg + 1].lon - c0_lon) * e
            zoom_float = base_z + min(2.8, max(0.3, 14.3 - base_z)) * e
            pitch_deg = 8.0 + 10.0 * e
        globe_strength = 0.0
    else:
        fit_vp = _compute_viewport(norm_wps, int(round(width * 1.35)), int(round(height * 1.35)), focus_leg_index=-1)
        base_z = float(min(15.2, max(6.0, fit_vp.zoom + (0.6 if total_span_km < 5.0 else 0.0))))
        mid_lat = sum(w.lat for w in norm_wps) / len(norm_wps)
        mid_lon = sum(w.lon for w in norm_wps) / len(norm_wps)
        e_u = _ease_in_out_cubic(leg_u)
        c_lat_cur = norm_wps[active_leg].lat + (norm_wps[active_leg + 1].lat - norm_wps[active_leg].lat) * e_u
        c_lon_cur = norm_wps[active_leg].lon + (norm_wps[active_leg + 1].lon - norm_wps[active_leg].lon) * e_u
        center_lat = 0.55 * c_lat_cur + 0.45 * mid_lat
        center_lon = 0.55 * c_lon_cur + 0.45 * mid_lon
        zoom_float = base_z + 0.45 * _ease_in_out_cubic(global_t)
        pitch_deg = 4.0 if cur_mode == "walk" else 9.0
        globe_strength = 0.0

    return active_leg, leg_u, center_lat, center_lon, zoom_float, pitch_deg, globe_strength


def prefetch_google_earth_tiles(
    waypoints: list[MapWaypointSpec | dict[str, Any]],
    style: str = "satellite_hybrid_4k",
    width: int = 1920,
    height: int = 1080,
    leg_index: Optional[int] = None,
    leg_modes: Optional[list[str]] = None,
    transport_mode: str = "auto",
    cache_dir: Optional[str | Path] = None,
) -> None:
    """Pre-warm tile pyramid cache in parallel along the dynamic-zoom camera path."""
    if os.environ.get("AI_EDITOR_OFFLINE_MAPS") == "1":
        return
    norm_wps = _normalize_flyover_waypoints(waypoints)
    use_style = style if style in CART_THEME_SPECS else "satellite_hybrid_4k"
    spec = CART_THEME_SPECS[use_style]
    c_dir_p = Path(cache_dir).resolve() if cache_dir else None
    n_legs = len(norm_wps) - 1
    resolved_modes: list[str] = []
    leg_splines_ll: list[list[tuple[float, float]]] = []
    for k in range(n_legs):
        km_k = haversine_km(norm_wps[k].lat, norm_wps[k].lon, norm_wps[k + 1].lat, norm_wps[k + 1].lon)
        hint_k = leg_modes[k] if (leg_modes and k < len(leg_modes)) else transport_mode
        m_k = infer_transport_mode(km_k, norm_wps[k].label, norm_wps[k + 1].label, hint=hint_k)
        resolved_modes.append(m_k)
        if m_k in ("drive", "walk", "train"):
            r_pts = _fetch_osrm_route_latlon(norm_wps[k], norm_wps[k + 1], cache_dir=c_dir_p, profile="foot" if m_k == "walk" else "driving")
            if r_pts:
                leg_splines_ll.append(r_pts)
                continue
        if m_k == "flight":
            leg_splines_ll.append(_great_circle_latlon_pts(norm_wps[k].lat, norm_wps[k].lon, norm_wps[k + 1].lat, norm_wps[k + 1].lon, n_samples=120))
        else:
            leg_splines_ll.append([(norm_wps[k].lat, norm_wps[k].lon), (norm_wps[k + 1].lat, norm_wps[k + 1].lon)])

    layers = [(use_style, spec["tile_url"], "RGB")]
    if spec.get("hillshade_url"):
        layers.append(("world_hillshade", spec["hillshade_url"], "RGB"))
    if spec.get("roads_overlay_url"):
        layers.append(("world_transportation", spec["roads_overlay_url"], "RGBA"))
    lbl_cache_tag = f"labels_{use_style}"
    if spec.get("labels_overlay_url"):
        layers.append((lbl_cache_tag, spec["labels_overlay_url"], "RGBA"))

    seen: set[tuple[str, int, int, int]] = set()
    jobs: list[tuple[str, str, int, int, int, str]] = []
    ww, wh = int(round(width * 1.35)), int(round(height * 1.35))
    for step in range(17):
        gt = step / 16.0
        _, _, clat, clon, z_f, _, _ = _compute_google_earth_camera(
            norm_wps, leg_splines_ll, resolved_modes, gt, width, height, leg_index=leg_index
        )
        z_max_prefetch = 17 if use_style == "satellite_hybrid_4k" else 16
        z_tile = max(3, min(z_max_prefetch, int(math.floor(z_f)) + 1))
        w_scale = 2.0 ** (z_tile - z_f)
        span_x = ww * w_scale * 0.72
        span_y = wh * w_scale * 0.72
        cx_w, cy_w = _latlon_to_world_px(clat, clon, z_tile)
        for ltag, utpl, mode in layers:
            if ltag == "world_transportation" and z_tile < 8:
                continue
            if ltag == "world_hillshade":
                zt = min(14, z_tile)
            elif ltag == lbl_cache_tag:
                zt = min(16, z_tile)
            elif ltag == "world_transportation":
                zt = min(16 if use_style == "satellite_hybrid_4k" else 15, z_tile)
            else:
                zt = z_tile
            sc = 2.0 ** (zt - z_tile)
            n = 1 << zt
            tx0 = int(math.floor((cx_w - span_x * 0.52) * sc / 256.0))
            tx1 = int(math.floor((cx_w + span_x * 0.52) * sc / 256.0))
            ty0 = max(0, int(math.floor((cy_w - span_y * 0.52) * sc / 256.0)))
            ty1 = min(n - 1, int(math.floor((cy_w + span_y * 0.52) * sc / 256.0)))
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    tx_w = tx % n
                    k_tile = (ltag, zt, tx_w, ty)
                    if k_tile not in seen and k_tile not in _PYRAMID_TILE_MEM:
                        seen.add(k_tile)
                        jobs.append((ltag, utpl, zt, tx_w, ty, mode))
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=24) as pool:
        futs = [pool.submit(_load_or_fetch_pyramid_tile, ltag, utpl, zt, tx_w, ty, mode, c_dir_p, True) for (ltag, utpl, zt, tx_w, ty, mode) in jobs]
        for fut in futs:
            try:
                fut.result()
            except Exception:
                pass


def render_google_earth_flyover_frame(
    waypoints: list[MapWaypointSpec | dict[str, Any]],
    style: str = "satellite_hybrid_4k",
    width: int = 1920,
    height: int = 1080,
    global_t: float = 0.5,
    leg_index: Optional[int] = None,
    leg_modes: Optional[list[str]] = None,
    transport_mode: str = "auto",
    hdr_safe: bool = False,
    allow_tile_fetch: bool = True,
    cache_dir: Optional[str | Path] = None,
) -> Image.Image:
    """
    Render a Google Earth Studio-style dynamic-zoom frame with zero camera roll/shake
    (`bank_deg = 0.0`), multi-resolution Retina tile streaming, destination radar pulse,
    collision-free waypoint labels, and unobstructed corner photo callouts.
    """
    norm_wps = _normalize_flyover_waypoints(waypoints)
    use_style = style if style in CART_THEME_SPECS else "satellite_hybrid_4k"
    spec = CART_THEME_SPECS[use_style]
    c_dir_p = Path(cache_dir).resolve() if cache_dir else None
    fetch_ok = bool(allow_tile_fetch and os.environ.get("AI_EDITOR_OFFLINE_MAPS") != "1")

    n_legs = len(norm_wps) - 1
    resolved_modes: list[str] = []
    leg_splines_ll: list[list[tuple[float, float]]] = []
    for k in range(n_legs):
        km_k = haversine_km(norm_wps[k].lat, norm_wps[k].lon, norm_wps[k + 1].lat, norm_wps[k + 1].lon)
        hint_k = leg_modes[k] if (leg_modes and k < len(leg_modes)) else transport_mode
        m_k = infer_transport_mode(km_k, norm_wps[k].label, norm_wps[k + 1].label, hint=hint_k)
        resolved_modes.append(m_k)
        if m_k in ("drive", "walk", "train") and fetch_ok:
            r_pts = _fetch_osrm_route_latlon(
                norm_wps[k],
                norm_wps[k + 1],
                cache_dir=c_dir_p,
                profile="foot" if m_k == "walk" else "driving",
            )
            if r_pts:
                leg_splines_ll.append(r_pts)
                continue
        if m_k == "flight":
            leg_splines_ll.append(_great_circle_latlon_pts(norm_wps[k].lat, norm_wps[k].lon, norm_wps[k + 1].lat, norm_wps[k + 1].lon, n_samples=200))
        else:
            leg_splines_ll.append([(norm_wps[k].lat, norm_wps[k].lon), (norm_wps[k + 1].lat, norm_wps[k + 1].lon)])

    active_leg, leg_u, center_lat, center_lon, zoom_float, pitch_deg, globe_strength = _compute_google_earth_camera(
        norm_wps, leg_splines_ll, resolved_modes, global_t, width, height, leg_index=leg_index
    )
    cur_mode = resolved_modes[active_leg]

    ww = int(round(width * 1.35))
    wh = int(round(height * 1.35))
    z_max_frame = 17 if use_style == "satellite_hybrid_4k" else 16
    z_tile = max(3, min(z_max_frame, int(math.floor(zoom_float)) + 1))
    world_px_scale = 2.0 ** (z_tile - zoom_float)
    span_x = ww * world_px_scale * 0.72
    span_y = wh * world_px_scale * 0.72
    cx_w, cy_w = _latlon_to_world_px(center_lat, center_lon, z_tile)
    wx0, wx1 = cx_w - span_x * 0.5, cx_w + span_x * 0.5
    wy0, wy1 = cy_w - span_y * 0.5, cy_w + span_y * 0.5

    def _ll_to_world(lat: float, lon: float) -> tuple[float, float]:
        px, py = _latlon_to_world_px(lat, lon, z_tile)
        world_full = 256.0 * (2.0 ** z_tile)
        if px - cx_w > world_full * 0.5:
            px -= world_full
        elif cx_w - px > world_full * 0.5:
            px += world_full
        return ((px - wx0) / span_x * ww, (py - wy0) / span_y * wh)

    base_rgb = _sample_pyramid_mosaic(
        use_style, spec["tile_url"], z_tile, wx0, wy0, wx1, wy1, ww, wh,
        mode="RGB", fill_val=tuple(spec["water_deep"]), cache_dir=c_dir_p, allow_fetch=fetch_ok,
    ).astype(np.float32)

    hs_url = spec.get("hillshade_url")
    hs_str = float(spec.get("hillshade_strength", 0.0) or 0.0)
    if hs_url and hs_str > 0.01 and not (use_style == "midnight_vox_atlas" and z_tile >= 11):
        z_hs = min(14, z_tile)
        hs_sc = 2.0 ** (z_hs - z_tile)
        hs_arr = _sample_pyramid_mosaic(
            "world_hillshade", hs_url, z_hs, wx0 * hs_sc, wy0 * hs_sc, wx1 * hs_sc, wy1 * hs_sc,
            ww, wh, mode="RGB", fill_val=(180, 180, 180), cache_dir=c_dir_p, allow_fetch=fetch_ok,
        )
        eff_hs = hs_str * (0.38 if z_tile >= 11 else 0.85)
        hs_luma = (0.299 * hs_arr[:, :, 0] + 0.587 * hs_arr[:, :, 1] + 0.114 * hs_arr[:, :, 2]) / 255.0
        relief = np.clip(1.0 + eff_hs * ((hs_luma - 0.706) * 2.2), 0.74, 1.14)[..., None]
        base_rgb = np.clip(base_rgb * relief, 0, 255)

    if use_style in ("satellite_hybrid_4k", "aerial_relief_hybrid"):
        rgb_u8 = np.clip(base_rgb, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_mix = cv2.addWeighted(l_ch, 0.90, cv2.createCLAHE(clipLimit=1.05, tileGridSize=(12, 12)).apply(l_ch), 0.10, 0)
        hsv = cv2.cvtColor(cv2.cvtColor(cv2.merge((l_mix, a_ch, b_ch)), cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.92, 0, 255)
        base_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    if spec.get("roads_overlay_url") and z_tile >= 8:
        z_rd = min(16 if use_style == "satellite_hybrid_4k" else 15, z_tile)
        rd_sc = 2.0 ** (z_rd - z_tile)
        rd_arr = _sample_pyramid_mosaic(
            "world_transportation", spec["roads_overlay_url"], z_rd,
            wx0 * rd_sc, wy0 * rd_sc, wx1 * rd_sc, wy1 * rd_sc, ww, wh,
            mode="RGBA", fill_val=(0, 0, 0, 0), cache_dir=c_dir_p, allow_fetch=fetch_ok,
        )
        rd_a = (rd_arr[:, :, 3:4].astype(np.float32) * (0.78 / 255.0))
        if float(rd_a.max()) > 0.005:
            base_rgb = base_rgb * (1.0 - rd_a) + rd_arr[:, :, :3].astype(np.float32) * rd_a
    world_rgba = Image.fromarray(np.clip(base_rgb, 0, 255).astype(np.uint8)).convert("RGBA")

    route_arr = np.zeros((wh, ww, 4), dtype=np.uint8)
    s_w = ww / 1920.0
    stroke_w = max(2, int(round(5.0 * s_w)))
    px_splines = [[_ll_to_world(lat, lon) for lat, lon in seg] for seg in leg_splines_ll]
    px_stops = [_ll_to_world(w.lat, w.lon) for w in norm_wps]

    for k, seg_pts in enumerate(px_splines):
        if k > active_leg or k == active_leg:
            _draw_dashed_aa_polyline(route_arr, seg_pts, (*spec["route_future"], 135), max(1, stroke_w // 2 + 1), 10.0 * s_w, 8.0 * s_w)
        else:
            _draw_cased_route(route_arr, seg_pts, spec, stroke_w, s_w, is_flight=False)

    active_pts = px_splines[active_leg]
    ground_arc_path = _ArcLengthPath(active_pts)
    prog_ground = ground_arc_path.sample_prefix(leg_u)
    ground_head_pt = ground_arc_path.point_at_s(leg_u * ground_arc_path.total_len)
    if cur_mode == "flight":
        _draw_aa_polyline(route_arr, prog_ground, (12, 18, 28, 128), max(2, stroke_w))
        _draw_aa_polyline(route_arr, prog_ground, (*spec["route_glow"], 115), max(1, stroke_w // 2))
    else:
        _draw_cased_route(route_arr, prog_ground, spec, stroke_w, s_w, is_flight=False)
        head_pt = ground_head_pt
        heading, _ = ground_arc_path.tangent_and_curvature(leg_u, window_px=26.0 * s_w)
    world_rgba = Image.alpha_composite(world_rgba, Image.fromarray(route_arr))

    if spec.get("labels_overlay_url"):
        z_lb = min(16, z_tile)
        lb_sc = 2.0 ** (z_lb - z_tile)
        lb_arr = _sample_pyramid_mosaic(
            f"labels_{use_style}", spec["labels_overlay_url"], z_lb,
            wx0 * lb_sc, wy0 * lb_sc, wx1 * lb_sc, wy1 * lb_sc, ww, wh,
            mode="RGBA", fill_val=(0, 0, 0, 0), cache_dir=c_dir_p, allow_fetch=fetch_ok,
        )
        l_rgb = lb_arr[:, :, :3].astype(np.float32)
        l_a = lb_arr[:, :, 3].astype(np.float32) / 255.0
        if float(l_a.max()) > 0.01:
            shadow_a = np.clip(cv2.dilate((l_a * 255.0).astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.float32) / 255.0 * 0.70, 0.0, 0.70)
            base_w_arr = np.asarray(world_rgba.convert("RGB"), dtype=np.float32) * (1.0 - shadow_a[..., None])
            lbl_mix = np.clip(l_a * 0.96, 0.0, 1.0)[..., None]
            world_rgba = Image.fromarray(np.clip(base_w_arr * (1.0 - lbl_mix) + l_rgb * lbl_mix, 0, 255).astype(np.uint8)).convert("RGBA")

    wd = ImageDraw.Draw(world_rgba, "RGBA")
    i0, i1 = active_leg, active_leg + 1
    for k in [idx for idx in range(len(px_stops)) if idx not in (i0, i1)] + [i0, i1]:
        wx, wy = px_stops[k]
        if wx < -80 or wx > ww + 80 or wy < -80 or wy > wh + 80:
            continue
        is_orig, is_dest = k == i0, k == i1
        pin_r = max(6, int(round((11 if (is_orig or is_dest) else 8) * s_w)))
        if is_dest and leg_u >= 0.72:
            pulse_p = (leg_u - 0.72) / 0.28
            r_pulse = pin_r + int(round(26 * s_w * pulse_p))
            p_alpha = int(round(200 * (1.0 - pulse_p)))
            wd.ellipse([wx - r_pulse, wy - r_pulse, wx + r_pulse, wy + r_pulse], outline=(*spec["accent"], p_alpha), width=max(1, int(round(2.5 * s_w))))
        ring_col = (*spec["accent"], 255) if (is_orig or is_dest) else (*spec["route_history"], 220)
        wd.ellipse([wx - pin_r - 3, wy - pin_r - 3, wx + pin_r + 3, wy + pin_r + 3], fill=ring_col)
        wd.ellipse([wx - pin_r, wy - pin_r, wx + pin_r, wy + pin_r], fill=(*spec["pin_fill"], 248))

    veh_scale_w = s_w * 1.08
    stroke_col = spec["pin_fill"] if not spec["light_paper"] else (248, 244, 238)
    if cur_mode == "flight":
        alt_sin = math.sin(math.pi * leg_u)
        g_heading, _ = ground_arc_path.tangent_and_curvature(leg_u, window_px=24.0 * s_w)
        _draw_aircraft_ground_shadow(world_rgba, ground_head_pt[0] + 10.0 * s_w * alt_sin, ground_head_pt[1] + 12.0 * s_w * alt_sin, g_heading, veh_scale_w * (0.92 - 0.28 * alt_sin), int(round(125 - 65 * alt_sin)))
    elif cur_mode in ("train", "walk", "drive"):
        _draw_glow_beacon_dot(
            world_rgba,
            cx=head_pt[0],
            cy=head_pt[1],
            glow_rgb=spec["route_glow"],
            core_rgb=spec["accent"],
            stroke_rgb=stroke_col,
            scale=veh_scale_w,
        )
    elif cur_mode == "boat":
        _draw_boat_kelvin_wake(world_rgba, arc_path=ground_arc_path, u=leg_u, scale=veh_scale_w)
        _draw_transport_icon(world_rgba, cx=head_pt[0], cy=head_pt[1], angle_rad=heading, mode="boat", fill_rgb=spec["route_core"], stroke_rgb=stroke_col, scale=veh_scale_w, bank_factor=0.0)
    else:
        _draw_transport_icon(world_rgba, cx=head_pt[0], cy=head_pt[1], angle_rad=heading, mode=cur_mode, fill_rgb=spec["route_core"], stroke_rgb=stroke_col, scale=veh_scale_w, bank_factor=0.0)

    warped_rgb, proj_fn, horizon_y_fn = _apply_camera_homography(
        world_rgba.convert("RGB"), out_w=width, out_h=height,
        center_xy=(ww * 0.5, wh * 0.5), zoom=1.02, pitch_deg=pitch_deg, bank_deg=0.0, globe_strength=globe_strength,
    )
    warped_im = Image.fromarray(warped_rgb).filter(ImageFilter.UnsharpMask(radius=0.7, percent=24, threshold=2))
    out_rgba = warped_im.convert("RGBA")
    s_out = width / 1920.0
    screen_stops = [proj_fn(wx, wy) for wx, wy in px_stops]

    if cur_mode == "flight":
        alt_sin = math.sin(math.pi * leg_u)
        scr_ground = [proj_fn(gx, gy) for gx, gy in prog_ground]
        if len(scr_ground) >= 2:
            gx0, gy0 = screen_stops[i0]
            gx1, gy1 = screen_stops[i1]
            scr_span = math.hypot(gx1 - gx0, gy1 - gy0)
            lift_px = min(height * 0.24, max(42.0 * s_out, scr_span * 0.24)) * (0.35 + 0.65 * alt_sin)
            cum_s = [0.0]
            for j in range(1, len(scr_ground)):
                cum_s.append(cum_s[-1] + math.hypot(scr_ground[j][0] - scr_ground[j - 1][0], scr_ground[j][1] - scr_ground[j - 1][1]))
            tot_s = max(1e-6, cum_s[-1] / max(1e-4, leg_u))
            scr_arc_pts = [(sx_g, sy_g - lift_px * math.sin(math.pi * min(1.0, cum_s[j] / tot_s))) for j, (sx_g, sy_g) in enumerate(scr_ground)]
            sky_arr = np.zeros((height, width, 4), dtype=np.uint8)
            _draw_cased_route(sky_arr, scr_arc_pts, spec, max(2, int(round(5.5 * s_out))), s_out, is_flight=False)
            out_rgba = Image.alpha_composite(out_rgba, Image.fromarray(sky_arr))
            elev_scr_path = _ArcLengthPath(scr_arc_pts)
            scr_head = elev_scr_path.point_at_s(elev_scr_path.total_len)
            scr_heading, _ = elev_scr_path.tangent_and_curvature(1.0, window_px=26.0 * s_out)
            veh_scr_scale = s_out * 1.15 * (0.88 + 0.40 * alt_sin)
            _draw_flight_contrails(out_rgba, scr_arc_pts, scale=veh_scr_scale, alt_sin=alt_sin)
            _draw_transport_icon(out_rgba, cx=scr_head[0], cy=scr_head[1], angle_rad=scr_heading, mode="flight", fill_rgb=spec["route_core"], stroke_rgb=stroke_col, scale=veh_scr_scale, bank_factor=0.0)

    # Pin index numerals & collision-free stop labels
    od = ImageDraw.Draw(out_rgba, "RGBA")
    f_pin = _load_font_tier(max(10, int(round(13 * s_out))), tier="mono", bold=True)
    all_lbl_txt = " ".join(w.label for w in norm_wps)
    f_wp_lbl = _load_font_tier(max(11, int(round(17 * s_out))), tier=spec["title_tier"], bold=True, text=all_lbl_txt)
    for k, (sx_w, sy_w) in enumerate(screen_stops):
        if -40 <= sx_w <= width + 40 and -40 <= sy_w <= height + 40:
            num_str = f"{k + 1:02d}"
            nw, _ = measure_tracked_text(od, num_str, f_pin)
            od.text((sx_w - nw * 0.5, sy_w - 8.0 * s_out), num_str, font=f_pin, fill=(*spec["pin_text"], 250))

    active_card_wp = norm_wps[i0] if leg_u < 0.45 else norm_wps[i1]
    has_card = bool(active_card_wp.hero_thumbnail_path and Path(active_card_wp.hero_thumbnail_path).exists())
    card_x = int(round(width * 0.045))
    cw_est, ch_est = int(round(360 * s_out)), int(round(269 * s_out))
    occupied_rects: list[tuple[float, float, float, float]] = [
        (0.0, 0.0, width * 0.45, height * 0.22),
        (width * 0.69, height * 0.74, float(width), float(height)),
        (0.0, 0.0, float(width), max(height * 0.14, horizon_y_fn(width * 0.5) + 12.0)),
    ]
    if has_card:
        occupied_rects.append((float(card_x), float(height - int(round(height * 0.065)) - ch_est), float(card_x + cw_est), float(height)))
    for sx_w, sy_w in screen_stops:
        if -40 <= sx_w <= width + 40 and -40 <= sy_w <= height + 40:
            pr = max(12.0, 22.0 * s_out)
            occupied_rects.append((sx_w - pr, sy_w - pr, sx_w + pr, sy_w + pr))

    def _overlaps_any(box: tuple[float, float, float, float], pad: float = 4.0) -> bool:
        bx0, by0, bx1, by1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
        return any(not (bx1 < rx0 or bx0 > rx1 or by1 < ry0 or by0 > ry1) for rx0, ry0, rx1, ry1 in occupied_rects)

    pending_labels = []
    for k in [i0, i1] + [idx for idx in range(len(screen_stops)) if idx not in (i0, i1)]:
        sx_w, sy_w = screen_stops[k]
        if sx_w < -60 or sx_w > width + 60 or sy_w < -60 or sy_w > height + 60:
            continue
        is_orig, is_dest, is_done = k == i0, k == i1, k <= i0
        pin_r = max(6, int(round((11 if (is_orig or is_dest) else 8) * s_out)))
        ring_col = (*spec["accent"], 255) if (is_orig or is_dest) else (*spec["route_history"], 220)
        tag = norm_wps[k].label
        tb0 = od.textbbox((0, 0), tag, font=f_wp_lbl)
        tw, th = tb0[2] - tb0[0], tb0[3] - tb0[1]
        pad_x, pad_y = int(round(10 * s_out)), int(round(7 * s_out))
        for cx_c, cy_c in [
            (sx_w + pin_r + 14 * s_out, sy_w - th * 0.55 - tb0[1]),
            (sx_w - tw * 0.5 - tb0[0], sy_w + pin_r + 18 * s_out - tb0[1]),
            (sx_w - tw * 0.5 - tb0[0], sy_w - pin_r - th - 18 * s_out - tb0[1]),
            (sx_w - pin_r - tw - 24 * s_out - tb0[0], sy_w - th * 0.55 - tb0[1]),
        ]:
            tx_c = int(round(max(18, min(width - tw - 22, cx_c))))
            ty_c = int(round(max(int(round(height * 0.21)), min(height - th - 20, cy_c))))
            tb_c = od.textbbox((tx_c, ty_c), tag, font=f_wp_lbl)
            cbox = (tb_c[0] - pad_x, tb_c[1] - pad_y, tb_c[2] + pad_x, tb_c[3] + pad_y)
            if not _overlaps_any(cbox):
                occupied_rects.append(cbox)
                pending_labels.append((cbox, tx_c, ty_c, tag, ring_col, is_orig or is_dest, is_done or is_dest))
                break

    if has_card and active_card_wp.hero_thumbnail_path:
        try:
            pim = _load_clean_callout_rgb(Path(active_card_wp.hero_thumbnail_path))
            cw, ch_img, cap_h, pad_c = int(round(360 * s_out)), int(round(215 * s_out)), int(round(34 * s_out)), int(round(10 * s_out))
            pim = ImageOps.fit(pim, (cw - 2 * pad_c, ch_img), method=Image.Resampling.LANCZOS)
            card_im = Image.new("RGBA", (cw, ch_img + cap_h + 2 * pad_c), (0, 0, 0, 0))
            dc = ImageDraw.Draw(card_im, "RGBA")
            dc.rounded_rectangle([0, 0, cw - 1, card_im.height - 1], radius=int(round(12 * s_out)), fill=(*spec["hud_fill"][:3], 238), outline=(*spec["accent"], 220), width=max(2, int(round(2.5 * s_out))))
            card_im.alpha_composite(pim.convert("RGBA"), (pad_c, pad_c))
            f_card = _load_font_tier(max(11, int(round(15 * s_out))), tier=spec["title_tier"], bold=True, text=active_card_wp.label)
            dc.text((pad_c + 4, pad_c + ch_img + int(round(7 * s_out))), active_card_wp.label[:30], font=f_card, fill=(*spec["hud_title"], 250))
            out_rgba.alpha_composite(card_im, (card_x, height - int(round(height * 0.065)) - card_im.height))
        except Exception:
            pass

    od_top = ImageDraw.Draw(out_rgba, "RGBA")
    for pbox, tx, ty, tag, ring_col, is_active_pin, is_bright in pending_labels:
        od_top.rounded_rectangle(list(pbox), radius=max(4, int(round(6 * s_out))), fill=(*spec["hud_fill"][:3], 238 if is_bright else 185), outline=ring_col if is_active_pin else (*spec["hud_border"][:3], 125), width=1)
        od_top.text((tx, ty), tag, font=f_wp_lbl, fill=(*spec["hud_title"], 252))

    _render_screen_hud(frame_rgba=out_rgba, style=use_style, waypoints=norm_wps, leg_index=active_leg, t_norm=0.06 + leg_u * 0.76, transport_mode=cur_mode, compact_hud=False, hdr_safe=hdr_safe)
    out_arr = np.asarray(out_rgba.convert("RGB"), dtype=np.uint8)
    if hdr_safe:
        max_ch = np.max(out_arr, axis=2, keepdims=True).astype(np.float32)
        scale = np.where(max_ch > float(HDR_DIFFUSE_WHITE_PEAK_RGB), float(HDR_DIFFUSE_WHITE_PEAK_RGB) / np.maximum(1.0, max_ch), 1.0)
        out_arr = np.clip(np.round(out_arr.astype(np.float32) * scale), 0, HDR_DIFFUSE_WHITE_PEAK_RGB).astype(np.uint8)
    return Image.fromarray(out_arr)


def render_travel_map_clip(
    waypoints: list[MapWaypointSpec | dict[str, Any]],
    output_mp4: str | Path,
    style: str = "auto",
    duration: float = 5.2,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    leg_index: Optional[int] = 0,
    transport_mode: str = "auto",
    hdr_safe: bool = False,
    allow_tile_fetch: bool = False,
    with_audio: bool = False,
    cache_dir: Optional[str | Path] = None,
    annotations: Optional[list[dict[str, Any]]] = None,
    leg_modes: Optional[list[str]] = None,
    dynamic_zoom: bool = False,
    leg_via_points: Optional[dict[int, list[tuple[float, float]]]] = None,
) -> Path:
    """
    Encode a multi-frame Cinema 2.5D Travel Map sequence to an H.264 MP4 clip.

    Pass `leg_index=0..N` for a single chapter leg transition, or `leg_index=None` to
    animate across all legs sequentially inside a standalone demo reel.
    Automatically precomputes per-leg routes, zero-teleport layout plans, and CUDA VRAM baseplates.
    """
    out_path = Path(output_mp4).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_even = max(320, int(width) - (int(width) % 2))
    h_even = max(180, int(height) - (int(height) % 2))
    dur = max(0.8, float(duration))
    total_frames = max(12, int(round(dur * fps)))

    norm_count = max(2, len(waypoints))
    n_legs = max(1, norm_count - 1)

    # Estimate leg distance & mode for intelligent auto-style and orbital-flight routing
    norm_preview = _normalize_flyover_waypoints(waypoints)
    focus_i_clip = max(0, min(len(norm_preview) - 2, int(leg_index or 0)))
    est_leg_km = haversine_km(
        norm_preview[focus_i_clip].lat,
        norm_preview[focus_i_clip].lon,
        norm_preview[focus_i_clip + 1].lat,
        norm_preview[focus_i_clip + 1].lon,
    )
    eff_mode_hint = (
        leg_modes[focus_i_clip]
        if (leg_modes and leg_index is not None and focus_i_clip < len(leg_modes))
        else transport_mode
    )
    eff_resolved_mode = infer_transport_mode(
        est_leg_km,
        norm_preview[focus_i_clip].label,
        norm_preview[focus_i_clip + 1].label,
        hint=eff_mode_hint,
    )

    # Intelligently resolve style when 'auto' is passed (never blindly hardcodes midnight_vox_atlas!)
    if str(style or "auto").strip().lower() == "auto":
        lbl_blob = " ".join(f"{w.label} {w.subtitle}" for w in norm_preview)
        style = resolve_travel_map_style(
            "auto",
            prompt_text=lbl_blob,
            transport_mode=eff_resolved_mode,
            leg_km=est_leg_km,
        )

    wav_tmp: Optional[Path] = None
    if with_audio:
        wav_tmp = out_path.with_suffix(".tmp_sfx.wav")
        _synthesize_flyover_wav(wav_tmp, duration_sec=dur)

    from ..hardware import detect_best_video_encoder, get_io_worker_count

    best_sdr = detect_best_video_encoder(hdr=False)

    def _build_map_cmd(force_sw: bool) -> list[str]:
        cq_val = "14" if w_even >= 3000 else "17"
        vb_val = "35M" if w_even >= 3000 else "18M"
        crf_val = "15" if w_even >= 3000 else "17"
        if best_sdr == "h264_nvenc" and not force_sw:
            enc_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", cq_val, "-b:v", vb_val]
        elif best_sdr == "h264_videotoolbox" and not force_sw:
            enc_args = ["-c:v", "h264_videotoolbox", "-b:v", vb_val, "-allow_sw", "1"]
        else:
            enc_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", crf_val]
        c = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w_even}x{h_even}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        if wav_tmp is not None and wav_tmp.exists():
            c.extend(["-i", str(wav_tmp)])
        else:
            c.extend(["-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000:d={dur}"])
        c.extend(
            [
                *enc_args,
                "-pix_fmt",
                "yuv420p",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-t",
                f"{dur:.3f}",
                "-shortest",
                str(out_path),
            ]
        )
        return c

    # Precompute static per-leg baseplates, routes, Polaroid thumbs, zero-teleport layouts & CUDA VRAM tensors ONCE!
    # Reserve multi-zoom Google Earth pyramid streaming strictly for long-haul orbital flights (>= 250 km) without corridor annotations;
    # all ground/regional/heritage/editorial legs automatically get the GPU-VRAM-cached post-warp screen-space engine by default!
    use_ge_dynamic = bool(
        dynamic_zoom
        and allow_tile_fetch
        and os.environ.get("AI_EDITOR_OFFLINE_MAPS") != "1"
        and eff_resolved_mode == "flight"
        and est_leg_km >= 250.0
        and style == "satellite_hybrid_4k"
        and not annotations
    )
    prepared_legs: dict[int, _PreparedLegContext] = {}
    if not use_ge_dynamic:
        needed_legs = list(range(n_legs)) if (leg_index is None and n_legs > 1) else [max(0, min(n_legs - 1, int(leg_index or 0)))]
        for k_leg in needed_legs:
            lm = leg_modes[k_leg] if (leg_modes and k_leg < len(leg_modes)) else transport_mode
            prepared_legs[k_leg] = _prepare_travel_map_leg(
                waypoints=waypoints,
                style=style,
                width=w_even,
                height=h_even,
                leg_index=k_leg,
                transport_mode=lm,
                compact_hud=False,
                allow_tile_fetch=allow_tile_fetch,
                cache_dir=cache_dir,
                annotations=annotations,
                leg_via_points=leg_via_points,
                upload_cuda=True,
            )

    def _render_frame_bytes(f_idx: int) -> bytes:
        global_p = f_idx / float(max(1, total_frames - 1))
        if use_ge_dynamic:
            im = render_google_earth_flyover_frame(
                waypoints=waypoints,
                style=style,
                width=w_even,
                height=h_even,
                global_t=global_p,
                leg_index=leg_index,
                leg_modes=leg_modes,
                transport_mode=transport_mode,
                hdr_safe=hdr_safe,
                allow_tile_fetch=True,
                cache_dir=cache_dir,
            )
            if im.size != (w_even, h_even):
                im = im.resize((w_even, h_even), Image.Resampling.BILINEAR)
            return im.tobytes()

        if leg_index is None and n_legs > 1:
            scaled = global_p * n_legs
            cur_leg = min(n_legs - 1, int(math.floor(scaled)))
            leg_p = 1.0 if f_idx == total_frames - 1 else (scaled - cur_leg)
        else:
            cur_leg = max(0, min(n_legs - 1, int(leg_index or 0)))
            leg_p = global_p

        leg_mode = leg_modes[cur_leg] if (leg_modes and cur_leg < len(leg_modes)) else transport_mode
        im = render_travel_map_frame(
            waypoints=waypoints,
            style=style,
            width=w_even,
            height=h_even,
            leg_index=cur_leg,
            t_norm=leg_p,
            transport_mode=leg_mode,
            hdr_safe=hdr_safe,
            compact_hud=False,
            allow_tile_fetch=allow_tile_fetch,
            cache_dir=cache_dir,
            annotations=annotations,
            leg_via_points=leg_via_points,
            _prepared=prepared_legs.get(cur_leg),
        )
        if im.size != (w_even, h_even):
            im = im.resize((w_even, h_even), Image.Resampling.BILINEAR)
        return im.tobytes()

    if use_ge_dynamic:
        prefetch_google_earth_tiles(
            waypoints=waypoints,
            style=style,
            width=w_even,
            height=h_even,
            leg_index=leg_index,
            leg_modes=leg_modes,
            transport_mode=transport_mode,
            cache_dir=cache_dir,
        )

    # Pre-warm tile/font caches on frame 0, then render remaining frames in parallel chunks
    try:
        frame0_bytes = _render_frame_bytes(0)
        map_workers = max(1, min(get_io_worker_count(12), total_frames - 1))

        def _encode_all(force_sw: bool) -> tuple[int, bytes]:
            prev_cv_threads = cv2.getNumThreads()
            if map_workers > 1 and prev_cv_threads != 1:
                cv2.setNumThreads(1)
            proc = subprocess.Popen(_build_map_cmd(force_sw=force_sw), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            assert proc.stdin is not None and proc.stderr is not None
            try:
                proc.stdin.write(frame0_bytes)
                if total_frames > 1:
                    if map_workers == 1:
                        for f_i in range(1, total_frames):
                            proc.stdin.write(_render_frame_bytes(f_i))
                    else:
                        chunk_sz = max(map_workers * 2, 12)
                        with ThreadPoolExecutor(max_workers=map_workers) as pool:
                            for c_start in range(1, total_frames, chunk_sz):
                                c_end = min(total_frames, c_start + chunk_sz)
                                for fb in pool.map(_render_frame_bytes, range(c_start, c_end)):
                                    proc.stdin.write(fb)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                err_b = proc.stderr.read()
                rc_val = proc.wait()
                return (rc_val if rc_val != 0 else 1), err_b
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise
            finally:
                if map_workers > 1 and prev_cv_threads != 1:
                    cv2.setNumThreads(prev_cv_threads)
            err_b = proc.stderr.read()
            rc_val = proc.wait()
            return rc_val, err_b

        from ..hardware import nvenc_encode_guard

        with nvenc_encode_guard(["-c:v", best_sdr]):
            rc, stderr_bytes = _encode_all(force_sw=False)
        if rc != 0 and best_sdr == "h264_nvenc":
            rc, stderr_bytes = _encode_all(force_sw=True)
    finally:
        for p_ctx in prepared_legs.values():
            p_ctx.release_cuda()
        if wav_tmp is not None and wav_tmp.exists():
            try:
                wav_tmp.unlink()
            except Exception:
                pass

    if rc != 0:
        raise RuntimeError(f"Travel map FFmpeg encode failed (rc={rc}): {stderr_bytes.decode('utf-8', errors='ignore')}")
    return out_path


def render_travel_map_showcase_sheet(
    output_jpg: str | Path,
    waypoints: Optional[list[MapWaypointSpec | dict[str, Any]]] = None,
    cell_w: int = 960,
    cell_h: int = 540,
    allow_tile_fetch: bool = False,
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """Render a 2x2 showcase JPEG comparing all 4 Cinema Cartography styles across transport modes."""
    out_p = Path(output_jpg).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    demo_wps = waypoints or [
        {"label": "Tokyo", "lat": 35.6762, "lon": 139.6503, "alt_m": 40.0},
        {"label": "Hakone", "lat": 35.2324, "lon": 139.1069, "alt_m": 725.0},
        {"label": "Kyoto", "lat": 35.0116, "lon": 135.7681, "alt_m": 120.0},
    ]

    cells = [
        ("midnight_vox_atlas", "01. MIDNIGHT VOX ATLAS (TRAIN)", "train", 0.68),
        ("archival_expedition_parchment", "02. ARCHIVAL EXPEDITION PARCHMENT (FLIGHT)", "flight", 0.62),
        ("aerial_relief_hybrid", "03. AERIAL RELIEF HYBRID (DRIVE)", "drive", 0.74),
        ("kinfolk_minimal_monograph", "04. KINFOLK MINIMAL MONOGRAPH (BOAT)", "boat", 0.58),
    ]
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 2), (10, 12, 16))
    f_hdr = _load_font_tier(max(13, int(cell_w * 0.018)), tier="mono", bold=True)
    draw_s = ImageDraw.Draw(sheet)

    for idx, (st_name, badge, mode_name, t_n) in enumerate(cells):
        panel = render_travel_map_frame(
            waypoints=demo_wps,
            style=st_name,
            width=cell_w,
            height=cell_h,
            leg_index=0 if len(demo_wps) < 3 or idx % 2 == 0 else 1,
            t_norm=t_n,
            transport_mode=mode_name,
            hdr_safe=False,
            allow_tile_fetch=allow_tile_fetch,
            cache_dir=cache_dir,
        )
        col = idx % 2
        row = idx // 2
        ox = col * cell_w
        oy = row * cell_h
        sheet.paste(panel, (ox, oy))
        draw_s.rounded_rectangle([ox + 16, oy + cell_h - 40, ox + 460, oy + cell_h - 12], radius=6, fill=(10, 12, 16))
        draw_s.text((ox + 26, oy + cell_h - 33), badge, font=f_hdr, fill=(238, 236, 230))

    sheet.save(out_p, "JPEG", quality=93)
    return out_p
