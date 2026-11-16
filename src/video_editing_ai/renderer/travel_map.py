"""Studio-grade Cinema 2.5D Travel Map & Cartography Engine.

Renders multi-stop geographic route flyovers across 4 bespoke cinema styles
(midnight_vox_atlas, archival_expedition_parchment, aerial_relief_hybrid,
kinfolk_minimal_monograph) with Web Mercator projection, cached slippy-tile +
offline 2.5D hillshade topography, parabolic flight arcs, tangent-aligned
vector transport silhouettes, arrival hero photo callouts, virtual 2.5D camera
homography, synchronized elevation sparklines, and ~203-nit HDR clamping.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
import ssl
import subprocess
from typing import Any, Optional
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

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
        "route_future": (132, 162, 198),
        "route_history": (86, 178, 214),
        "accent": (244, 182, 82),
        "pin_fill": (18, 26, 42),
        "pin_text": (244, 248, 252),
        "hud_fill": (10, 16, 28, 212),
        "hud_border": (110, 220, 246, 120),
        "hud_title": (244, 248, 252),
        "hud_sub": (176, 198, 222),
        "hud_metric": (122, 232, 252),
        "title_tier": "sans",
        "sub_tier": "mono",
        "light_paper": False,
        "neatline": False,
        "compass_color": (160, 198, 228),
        "horizon_mist": (12, 22, 38),
        "tile_url": "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
        "tile_blend": 0.62,
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
        "contour_rgba": (128, 102, 72, 92),
        "graticule_rgba": (112, 88, 62, 74),
        "route_glow": (192, 68, 52),
        "route_core": (164, 42, 36),
        "route_future": (118, 92, 66),
        "route_history": (146, 72, 56),
        "accent": (178, 126, 44),
        "pin_fill": (236, 228, 208),
        "pin_text": (42, 32, 22),
        "hud_fill": (234, 226, 206, 228),
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
        "tile_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
        "tile_blend": 0.44,
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
        "contour_rgba": (228, 212, 174, 60),
        "graticule_rgba": (196, 216, 220, 52),
        "route_glow": (244, 184, 74),
        "route_core": (248, 216, 128),
        "route_future": (216, 226, 230),
        "route_history": (208, 166, 88),
        "accent": (246, 190, 80),
        "pin_fill": (20, 34, 42),
        "pin_text": (248, 246, 240),
        "hud_fill": (14, 24, 30, 208),
        "hud_border": (246, 190, 80, 125),
        "hud_title": (248, 246, 240),
        "hud_sub": (192, 210, 212),
        "hud_metric": (248, 204, 102),
        "title_tier": "sans",
        "sub_tier": "mono",
        "light_paper": False,
        "neatline": False,
        "compass_color": (212, 226, 224),
        "horizon_mist": (38, 62, 72),
        "tile_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "tile_blend": 0.58,
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
        "contour_rgba": (170, 164, 152, 78),
        "graticule_rgba": (152, 146, 138, 64),
        "route_glow": (196, 102, 74),
        "route_core": (178, 76, 52),
        "route_future": (146, 138, 130),
        "route_history": (176, 112, 90),
        "accent": (178, 76, 52),
        "pin_fill": (244, 240, 234),
        "pin_text": (38, 34, 30),
        "hud_fill": (242, 238, 230, 226),
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
        "tile_url": "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
        "tile_blend": 0.48,
    },
}


def resolve_travel_map_style(
    requested_style: str = "auto",
    title_theme: str = "apple_tv_glass",
    prompt_text: str = "",
) -> str:
    """Resolve explicit or brief-matched cartography style from the 4 curated palettes."""
    req = (requested_style or "auto").strip().lower()
    if req in CART_THEME_SPECS:
        return req

    low = (prompt_text or "").lower()
    if any(k in low for k in ("parchment", "vintage", "archival", "expedition", "indiana", "sepia", "treasure")):
        return "archival_expedition_parchment"
    if any(k in low for k in ("aerial", "satellite", "terrain", "alpine", "mountain", "relief", "topograph")):
        return "aerial_relief_hybrid"
    if any(k in low for k in ("kinfolk", "minimal", "monograph", "alabaster", "gallery", "magazine")):
        return "kinfolk_minimal_monograph"
    if any(k in low for k in ("vox", "midnight", "neon", "cyan", "dark atlas")):
        return "midnight_vox_atlas"

    theme_map = {
        "criterion_atelier": "archival_expedition_parchment",
        "archival_35mm": "archival_expedition_parchment",
        "kinfolk_magazine": "kinfolk_minimal_monograph",
        "a24_minimal": "midnight_vox_atlas",
        "kinetic_travelogue": "aerial_relief_hybrid",
        "apple_tv_glass": "midnight_vox_atlas",
    }
    return theme_map.get(title_theme, "midnight_vox_atlas")


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
    if any(k in blob for k in ("ferry", "harbor", "harbour", "island", "bay", "santorini", "miyajima", "venice", "como", "fiord", "sound")):
        if distance_km <= 350.0:
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
) -> _MapViewport:
    lats = [w.lat for w in waypoints]
    lons = [w.lon for w in waypoints]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    mid_lat = 0.5 * (min_lat + max_lat)
    mid_lon = 0.5 * (min_lon + max_lon)

    best_zoom = 5
    for z in range(13, 1, -1):
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


def _build_ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _try_fetch_slippy_mosaic(
    viewport: _MapViewport,
    style: str,
    cache_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Fetch and disk-cache Open/Esri slippy tiles covering viewport, graded into the theme."""
    spec = CART_THEME_SPECS[style]
    url_tpl = spec["tile_url"]
    n_tiles = 1 << viewport.zoom
    tx_min = int(math.floor(viewport.x0_world / 256.0))
    tx_max = int(math.floor(viewport.x1_world / 256.0))
    ty_min = max(0, int(math.floor(viewport.y0_world / 256.0)))
    ty_max = min(n_tiles - 1, int(math.floor(viewport.y1_world / 256.0)))
    if (tx_max - tx_min + 1) * (ty_max - ty_min + 1) > 48:
        return None

    tile_root = (cache_dir or Path("cache")) / "map_tiles" / style / str(viewport.zoom)
    tile_root.mkdir(parents=True, exist_ok=True)
    ctx = _build_ssl_ctx()

    canvas_w = max(256, (tx_max - tx_min + 1) * 256)
    canvas_h = max(256, (ty_max - ty_min + 1) * 256)
    mosaic = Image.new("RGB", (canvas_w, canvas_h), spec["water_deep"])
    fetched_ok = 0

    for tx_raw in range(tx_min, tx_max + 1):
        tx_wrap = tx_raw % n_tiles
        for ty in range(ty_min, ty_max + 1):
            local_png = tile_root / f"{tx_wrap}_{ty}.png"
            if not local_png.exists():
                url = url_tpl.format(z=viewport.zoom, x=tx_wrap, y=ty)
                req = urllib.request.Request(url, headers={"User-Agent": "video_editing_ai/1.0"})
                try:
                    with urllib.request.urlopen(req, timeout=1.6, context=ctx) as resp:
                        data = resp.read()
                    if len(data) > 200:
                        local_png.write_bytes(data)
                except Exception:
                    return None if fetched_ok == 0 else None
            try:
                with Image.open(local_png) as t_im:
                    mosaic.paste(t_im.convert("RGB"), ((tx_raw - tx_min) * 256, (ty - ty_min) * 256))
                    fetched_ok += 1
            except Exception:
                return None

    if fetched_ok == 0:
        return None

    # Crop exact pixel window and resize cleanly into (viewport.width, viewport.height)
    ox = int(round(viewport.x0_world - tx_min * 256.0))
    oy = int(round(viewport.y0_world - ty_min * 256.0))
    ow = int(round(viewport.x1_world - viewport.x0_world))
    oh = int(round(viewport.y1_world - viewport.y0_world))
    crop = mosaic.crop((ox, oy, ox + max(8, ow), oy + max(8, oh)))
    crop = crop.resize((viewport.width, viewport.height), Image.Resampling.BICUBIC)
    return np.asarray(crop, dtype=np.float32)


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
    """Draw a crisp 8-point nautical/cartographic compass rose with N/E/S/W ticks."""
    draw = ImageDraw.Draw(img, "RGBA")
    r = max(18, radius)
    ring_c = (*color_rgb, 140)
    dark_c = (*color_rgb, 210)
    fill_lite = (color_rgb[0], color_rgb[1], color_rgb[2], 70 if light_paper else 55)

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ring_c, width=max(1, r // 24))
    inner_r = int(round(r * 0.78))
    draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], outline=(*color_rgb, 80), width=1)

    for angle_deg in range(0, 360, 45):
        rad = math.radians(angle_deg - 90.0)
        is_cardinal = angle_deg % 90 == 0
        tip_len = r * (0.94 if is_cardinal else 0.62)
        side_len = r * (0.20 if is_cardinal else 0.13)
        perp = rad + math.pi * 0.5
        tip = (cx + math.cos(rad) * tip_len, cy + math.sin(rad) * tip_len)
        left_pt = (cx + math.cos(perp) * side_len, cy + math.sin(perp) * side_len)
        right_pt = (cx - math.cos(perp) * side_len, cy - math.sin(perp) * side_len)
        draw.polygon([(cx, cy), tip, right_pt], fill=dark_c if is_cardinal else fill_lite)
        draw.polygon([(cx, cy), tip, left_pt], fill=fill_lite, outline=(*color_rgb, 150))

    draw.text((cx - max(4, r // 9), cy - r - max(12, r // 3)), "N", font=sub_font, fill=dark_c)


@lru_cache(maxsize=24)
def _cached_world_baseplate(
    style: str,
    world_w: int,
    world_h: int,
    wp_sig: tuple[tuple[float, float, float, str], ...],
    allow_tile_fetch: bool,
    cache_dir_str: str,
) -> Image.Image:
    """Build or retrieve cached oversampled RGB cartographic world baseplate with 2.5D relief."""
    waypoints = [
        MapWaypointSpec(waypoint_id=f"wp_{i+1:02d}", label=lbl, lat=lat, lon=lon, alt_m=alt)
        for i, (lat, lon, alt, lbl) in enumerate(wp_sig)
    ]
    spec = CART_THEME_SPECS[style]
    vp = _compute_viewport(waypoints, world_w, world_h)

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

    # Optional online cached slippy-tile blend
    if allow_tile_fetch and os.environ.get("AI_EDITOR_OFFLINE_MAPS") != "1":
        tile_arr = _try_fetch_slippy_mosaic(vp, style=style, cache_dir=Path(cache_dir_str) if cache_dir_str else None)
        if tile_arr is not None and tile_arr.shape == rgb.shape:
            # Preserve real coastline/street detail while tinting smoothly into cinema palette
            blend = float(spec.get("tile_blend", 0.52))
            tile_luma = (0.299 * tile_arr[:, :, 0] + 0.587 * tile_arr[:, :, 1] + 0.114 * tile_arr[:, :, 2])[..., None] / 255.0
            if style in ("midnight_vox_atlas", "archival_expedition_parchment", "kinfolk_minimal_monograph"):
                tinted = water_d + (land_h - water_d) * tile_luma
                rgb = rgb * (1.0 - blend) + tinted * blend
            else:
                # aerial_relief_hybrid keeps full warm natural imagery + subtle terrain hillshade
                rgb = np.clip(tile_arr * (0.85 + 0.25 * (shade[..., None] - 0.5)), 0, 255) * blend + rgb * (1.0 - blend)

    # Quantized marching iso-elevation contour hairlines on land
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
    draw = ImageDraw.Draw(base_img, "RGBA")
    s = world_w / 1920.0
    f_mono = _load_font_tier(max(10, int(round(12 * s))), tier="mono", bold=False)

    # Lat/Lon cartographic graticule grid + coordinate edge ticks
    grat_col = tuple(spec["graticule_rgba"])
    for gx_frac in (0.18, 0.34, 0.50, 0.66, 0.82):
        px = int(round(world_w * gx_frac))
        draw.line([(px, 0), (px, world_h)], fill=grat_col, width=max(1, int(round(1.0 * s))))
        _, g_lon = vp.px_to_latlon(px, world_h * 0.5)
        draw.text((px + int(round(4 * s)), world_h - int(round(20 * s))), _format_coord_dms(g_lon, False), font=f_mono, fill=(*spec["coast_rgb"], 150))
    for gy_frac in (0.20, 0.38, 0.56, 0.74):
        py = int(round(world_h * gy_frac))
        draw.line([(0, py), (world_w, py)], fill=grat_col, width=max(1, int(round(1.0 * s))))
        g_lat, _ = vp.px_to_latlon(world_w * 0.5, py)
        draw.text((int(round(14 * s)), py + int(round(3 * s))), _format_coord_dms(g_lat, True), font=f_mono, fill=(*spec["coast_rgb"], 150))

    # Optional engraved alternating cartographic neatline border (Archival / Kinfolk)
    if spec.get("neatline"):
        bw = max(6, int(round(8 * s)))
        dark_b = (*spec["coast_rgb"], 185)
        light_b = (*spec["land_low"], 210)
        seg = max(48, int(round(96 * s)))
        for x0 in range(0, world_w, seg):
            c = dark_b if ((x0 // seg) % 2 == 0) else light_b
            draw.rectangle([x0, 0, min(world_w, x0 + seg), bw], fill=c)
            draw.rectangle([x0, world_h - bw, min(world_w, x0 + seg), world_h], fill=c)
        for y0 in range(0, world_h, seg):
            c = dark_b if ((y0 // seg) % 2 == 0) else light_b
            draw.rectangle([0, y0, bw, min(world_h, y0 + seg)], fill=c)
            draw.rectangle([world_w - bw, y0, world_w, min(world_h, y0 + seg)], fill=c)

    # 8-point cartographic compass rose in bottom-left safe area
    _draw_compass_rose(
        base_img,
        cx=int(round(world_w * 0.14)),
        cy=int(round(world_h * 0.80)),
        radius=max(24, int(round(38 * s))),
        color_rgb=spec["compass_color"],
        sub_font=f_mono,
        light_paper=bool(spec["light_paper"]),
    )
    return base_img.convert("RGB")


# ---------------------------------------------------------------------------
# Vector Splines, Flight Parabolic Arcs, Vehicle Silhouettes & Photo Callouts
# ---------------------------------------------------------------------------

def _ease_in_out_cubic(u: float) -> float:
    x = max(0.0, min(1.0, float(u)))
    return 4.0 * x * x * x if x < 0.5 else 1.0 - ((-2.0 * x + 2.0) ** 3) * 0.5


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
    if len(pts) < 2:
        return
    arr = np.round(np.array(pts, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        overlay_bgr_a,
        [arr],
        False,
        color=(rgba[0], rgba[1], rgba[2], rgba[3]),
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_AA,
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
                    (int(round(x1 + (x2 - x1) * t0)), int(round(y1 + (y2 - y1) * t0))),
                    (int(round(x1 + (x2 - x1) * t1)), int(round(y1 + (y2 - y1) * t1))),
                    color=rgba,
                    thickness=max(1, thickness),
                    lineType=cv2.LINE_AA,
                )
            pos += step
            rem -= step
            if rem <= 1e-3:
                draw_on = not draw_on
                rem = dash_px if draw_on else gap_px


def _draw_transport_icon(
    canvas: Image.Image,
    cx: float,
    cy: float,
    angle_rad: float,
    mode: str,
    fill_rgb: tuple[int, int, int],
    stroke_rgb: tuple[int, int, int],
    scale: float,
) -> None:
    """Render a crisp 2x-supersampled geometric vector silhouette rotated along heading."""
    box = max(48, int(round(76 * scale)))
    ss = box * 2
    sub = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(sub, "RGBA")
    m = ss * 0.5
    r = ss * 0.34

    def _rot(lx: float, ly: float) -> tuple[float, float]:
        ca = math.cos(angle_rad)
        sa = math.sin(angle_rad)
        return (m + lx * ca - ly * sa, m + lx * sa + ly * ca)

    # Soft shadow halo
    d.ellipse([m - r * 0.9, m - r * 0.9, m + r * 0.9, m + r * 0.9], fill=(8, 12, 18, 95))

    if mode == "flight":
        # Swept-wing jetliner pointing along +X
        poly = [
            _rot(r * 1.05, 0.0),
            _rot(r * 0.35, r * 0.16),
            _rot(-r * 0.25, r * 0.92),
            _rot(-r * 0.48, r * 0.85),
            _rot(-r * 0.12, r * 0.18),
            _rot(-r * 0.72, r * 0.12),
            _rot(-r * 0.92, r * 0.42),
            _rot(-r * 1.02, r * 0.36),
            _rot(-r * 0.86, 0.0),
            _rot(-r * 1.02, -r * 0.36),
            _rot(-r * 0.92, -r * 0.42),
            _rot(-r * 0.72, -r * 0.12),
            _rot(-r * 0.12, -r * 0.18),
            _rot(-r * 0.48, -r * 0.85),
            _rot(-r * 0.25, -r * 0.92),
            _rot(r * 0.35, -r * 0.16),
        ]
        d.polygon(poly, fill=(*fill_rgb, 250), outline=(*stroke_rgb, 255))
    elif mode == "train":
        # Aerodynamic high-speed train capsule
        poly = [
            _rot(r * 0.98, 0.0),
            _rot(r * 0.52, r * 0.34),
            _rot(-r * 0.75, r * 0.34),
            _rot(-r * 0.88, 0.0),
            _rot(-r * 0.75, -r * 0.34),
            _rot(r * 0.52, -r * 0.34),
        ]
        d.polygon(poly, fill=(*fill_rgb, 248), outline=(*stroke_rgb, 255))
        nose_glass = [_rot(r * 0.82, 0.0), _rot(r * 0.45, r * 0.20), _rot(r * 0.45, -r * 0.20)]
        d.polygon(nose_glass, fill=(24, 34, 48, 220))
    elif mode == "boat":
        # Sleek vessel hull + triangular wake lines
        poly = [
            _rot(r * 0.96, 0.0),
            _rot(r * 0.28, r * 0.36),
            _rot(-r * 0.80, r * 0.28),
            _rot(-r * 0.72, 0.0),
            _rot(-r * 0.80, -r * 0.28),
            _rot(r * 0.28, -r * 0.36),
        ]
        d.polygon(poly, fill=(*fill_rgb, 248), outline=(*stroke_rgb, 255))
        d.line([_rot(-r * 0.85, 0.0), _rot(-r * 1.25, r * 0.38)], fill=(*stroke_rgb, 185), width=max(2, int(ss * 0.03)))
        d.line([_rot(-r * 0.85, 0.0), _rot(-r * 1.25, -r * 0.38)], fill=(*stroke_rgb, 185), width=max(2, int(ss * 0.03)))
    else:
        # Touring chevron / grand-tourer beacon puck
        d.ellipse([m - r * 0.68, m - r * 0.68, m + r * 0.68, m + r * 0.68], fill=(*fill_rgb, 248), outline=(*stroke_rgb, 255), width=max(3, int(ss * 0.04)))
        chev = [
            _rot(r * 0.46, 0.0),
            _rot(-r * 0.28, r * 0.34),
            _rot(-r * 0.10, 0.0),
            _rot(-r * 0.28, -r * 0.34),
        ]
        d.polygon(chev, fill=(*stroke_rgb, 245))

    down = sub.resize((box, box), Image.Resampling.LANCZOS)
    canvas.alpha_composite(down, (int(round(cx - box * 0.5)), int(round(cy - box * 0.5))))


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
) -> None:
    """Draw a tilted double-matte gallery thumbnail card beside destination waypoint."""
    if not thumb_path or bloom_p <= 0.01:
        return
    p = Path(thumb_path)
    if not p.exists():
        return
    try:
        with Image.open(p) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception:
        return

    s = world_w / 1920.0
    card_w = int(round(220 * s * (0.78 + 0.22 * bloom_p)))
    card_h = int(round(card_w * 0.66))
    pad = max(6, int(round(8 * s)))
    inner_w = max(24, card_w - pad * 2)
    inner_h = max(20, card_h - pad * 2)
    im = ImageOps.fit(im, (inner_w, inner_h), method=Image.Resampling.LANCZOS)

    matte_bg = (*spec["pin_fill"], int(round(240 * bloom_p))) if not spec["light_paper"] else (246, 242, 234, int(round(242 * bloom_p)))
    border_c = (*spec["accent"], int(round(215 * bloom_p)))
    card = Image.new("RGBA", (card_w + 16, card_h + 16), (0, 0, 0, 0))
    dc = ImageDraw.Draw(card, "RGBA")
    dc.rounded_rectangle([10, 10, card_w + 10, card_h + 10], radius=max(4, int(round(8 * s))), fill=(6, 10, 16, int(round(135 * bloom_p))))
    dc.rounded_rectangle([4, 4, card_w + 4, card_h + 4], radius=max(4, int(round(8 * s))), fill=matte_bg, outline=border_c, width=max(1, int(round(2 * s))))
    im_rgba = im.convert("RGBA")
    if bloom_p < 0.99:
        alpha = im_rgba.getchannel("A").point(lambda v: int(round(v * bloom_p)))
        im_rgba.putalpha(alpha)
    card.alpha_composite(im_rgba, (4 + pad, 4 + pad))

    card = card.rotate(2.4, resample=Image.Resampling.BICUBIC, expand=True)
    dx, dy = dest_px
    margin = int(round(26 * s))
    cand_slots = [
        (dx - card.width - margin, dy - card.height - int(round(18 * s))),
        (dx + margin, dy - card.height - int(round(18 * s))),
        (dx - card.width - margin, dy + int(round(28 * s))),
        (dx + margin, dy + int(round(28 * s))),
    ]
    others = list(avoid_pts or [])
    best_xy: Optional[tuple[int, int]] = None
    best_score = -1.0
    for cx_raw, cy_raw in cand_slots:
        px_c = int(round(cx_raw))
        py_c = int(round(cy_raw))
        in_bounds = (
            px_c >= int(round(world_w * 0.05))
            and py_c >= int(round(world_h * 0.14))
            and (px_c + card.width) <= int(round(world_w * 0.95))
            and (py_c + card.height) <= int(round(world_h * 0.88))
        )
        cx_mid = px_c + card.width * 0.5
        cy_mid = py_c + card.height * 0.5
        min_d = min((math.hypot(cx_mid - ax, cy_mid - ay) for ax, ay in others), default=9999.0)
        score = min_d + (2000.0 if in_bounds else 0.0)
        if score > best_score:
            best_score = score
            best_xy = (px_c, py_c)
    assert best_xy is not None
    place_x = max(8, min(world_w - card.width - 8, best_xy[0]))
    place_y = max(8, min(world_h - card.height - 8, best_xy[1]))
    world_rgba.alpha_composite(card, (place_x, place_y))


def _apply_camera_homography(
    world_rgb: Image.Image,
    out_w: int,
    out_h: int,
    center_xy: tuple[float, float],
    zoom: float,
    pitch_deg: float,
    bank_deg: float,
) -> np.ndarray:
    """Project oversampled 2D world atlas through a 2.5D virtual drone camera homography."""
    ww, wh = world_rgb.size
    arr = np.asarray(world_rgb, dtype=np.uint8)

    # 3D pinhole projection mapping world plane z=0 around center_xy -> screen pixels
    cx, cy = center_xy
    focal = float(max(out_w, out_h)) * 1.15 * max(0.65, zoom)
    pitch = math.radians(max(0.0, min(42.0, pitch_deg)))
    bank = math.radians(max(-12.0, min(12.0, bank_deg)))

    # 4 target screen corners -> corresponding 4 world coordinates
    # Compute via inverse planar homography from screen plane
    dist = focal / max(0.35, zoom)
    # Screen coordinates centered at (out_w/2, out_h/2)
    dst_pts = np.array(
        [[0.0, 0.0], [float(out_w), 0.0], [float(out_w), float(out_h)], [0.0, float(out_h)]],
        dtype=np.float32,
    )
    src_pts: list[list[float]] = []
    half_sw = out_w * 0.5
    half_sh = out_h * 0.5
    cb, sb = math.cos(bank), math.sin(bank)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # Scale factor converting screen pixel offsets back to oversampled world pixels
    base_scale = (ww / max(1.0, float(out_w))) * (0.72 / max(0.55, zoom))
    for sx_c, sy_c in dst_pts:
        nx = (sx_c - half_sw)
        ny = (sy_c - half_sh)
        # Apply subtle perspective trapezoid: top edges look farther (wider world span), bottom closer
        # Normalized vertical [-0.5..0.5]
        v_norm = ny / max(1.0, float(out_h))
        persp = 1.0 / max(0.45, 1.0 + v_norm * math.tan(pitch) * 0.72)
        rx = (nx * cb - ny * sb) * persp * base_scale
        ry = (nx * sb + ny * cb) * persp * (base_scale / max(0.55, cp))
        wx = max(0.0, min(float(ww - 1), cx + rx))
        wy = max(0.0, min(float(wh - 1), cy + ry))
        src_pts.append([wx, wy])

    h_mat = cv2.getPerspectiveTransform(np.array(src_pts, dtype=np.float32), dst_pts)
    warped = cv2.warpPerspective(
        arr,
        h_mat,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def _render_screen_hud(
    frame_rgba: Image.Image,
    style: str,
    waypoints: list[MapWaypointSpec],
    leg_index: int,
    t_norm: float,
    transport_mode: str,
    compact_hud: bool,
    hdr_safe: bool,
) -> None:
    """Composite crisp post-warp frosted-glass route telemetry card & elevation sparkline."""
    spec = CART_THEME_SPECS[style]
    w, h = frame_rgba.size
    s = w / 1920.0
    draw = ImageDraw.Draw(frame_rgba, "RGBA")

    i0 = max(0, min(len(waypoints) - 1, leg_index))
    i1 = min(len(waypoints) - 1, i0 + 1)
    orig = waypoints[i0]
    dest = waypoints[i1]
    u = _ease_in_out_cubic(t_norm)
    leg_km = haversine_km(orig.lat, orig.lon, dest.lat, dest.lon)
    cur_km = leg_km * u
    cur_lat = orig.lat + (dest.lat - orig.lat) * u
    cur_lon = orig.lon + (dest.lon - orig.lon) * u
    cur_alt = orig.alt_m + (dest.alt_m - orig.alt_m) * u

    # Top horizon mist strip for 2.5D atmospheric depth (subtle edge fade)
    if not compact_hud:
        mist_h = max(12, int(round(h * 0.14)))
        mist_rgb = spec["horizon_mist"]
        strip = np.zeros((mist_h, 1, 4), dtype=np.uint8)
        for y in range(mist_h):
            k = 1.0 - (y / max(1, mist_h - 1))
            strip[y, 0] = (mist_rgb[0], mist_rgb[1], mist_rgb[2], int(round(135 * (k * k))))
        mist_img = Image.fromarray(strip).resize((w, mist_h), Image.Resampling.BILINEAR)
        frame_rgba.alpha_composite(mist_img, (0, 0))

    hud_fill = tuple(spec["hud_fill"])
    hud_border = tuple(spec["hud_border"])
    title_c = (*spec["hud_title"], 248)
    sub_c = (*spec["hud_sub"], 226)
    metric_c = (*spec["hud_metric"], 248)

    if compact_hud:
        # Compact bottom telemetry strip inside travel_map_pip_card
        ph = max(24, int(round(h * 0.18)))
        pt = h - ph - max(6, int(round(10 * s)))
        pl = max(8, int(round(14 * s)))
        pr = w - pl
        draw.rounded_rectangle([pl, pt, pr, pt + ph], radius=max(6, int(round(10 * s))), fill=hud_fill, outline=hud_border, width=max(1, int(round(1.5 * s))))
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

    # Full-frame documentary HUD telemetry pill (upper-left)
    px = int(round(w * 0.045))
    py = int(round(h * 0.065))
    f_over = _load_font_tier(max(10, int(round(13 * s))), tier=spec["sub_tier"], bold=True)
    f_head = _load_font_tier(max(13, int(round(30 * s))), tier=spec["title_tier"], bold=True)
    f_stat = _load_font_tier(max(10, int(round(16 * s))), tier=spec["sub_tier"], bold=False)

    route_line = f"{orig.label}  ->  {dest.label}"
    km_disp = f"{cur_km:.1f} KM" if leg_km >= 10.0 else f"{cur_km:.2f} KM"
    stat_line = f"{transport_mode.upper()}  |  {km_disp}  |  {_format_coord_dms(cur_lat, True)} {_format_coord_dms(cur_lon, False)}  |  {cur_alt:.0f}M"
    over_line = f"EXPEDITION LEG {i0 + 1:02d} OF {max(1, len(waypoints) - 1):02d}"

    rw1, _ = measure_tracked_text(draw, route_line, f_head)
    rw2, _ = measure_tracked_text(draw, stat_line, f_stat)
    rw3, _ = measure_tracked_text(draw, over_line, f_over, letter_spacing_px=2.0 * s)
    box_w = min(int(w * 0.68), max(int(round(360 * s)), max(rw1, rw2, rw3) + int(round(56 * s))))
    box_h = max(68, int(round(108 * s)))
    draw.rounded_rectangle(
        [px, py, px + box_w, py + box_h],
        radius=max(8, int(round(14 * s))),
        fill=hud_fill,
        outline=hud_border,
        width=max(1, int(round(2.0 * s))),
    )
    bar_w = max(2, int(round(3.5 * s)))
    draw.rounded_rectangle(
        [px + int(round(14 * s)), py + int(round(14 * s)), px + int(round(14 * s)) + bar_w, py + box_h - int(round(14 * s))],
        radius=max(1, bar_w // 2),
        fill=(*spec["accent"], 240),
    )
    tx = px + int(round(28 * s))
    draw_tracked_text(frame_rgba, (tx, py + int(round(13 * s))), over_line, font=f_over, fill=metric_c, letter_spacing_px=2.0 * s, shadow=False, hdr_safe=hdr_safe)
    draw_tracked_text(frame_rgba, (tx, py + int(round(34 * s))), route_line, font=f_head, fill=title_c, shadow=False, hdr_safe=hdr_safe)
    draw_tracked_text(frame_rgba, (tx, py + int(round(74 * s))), stat_line, font=f_stat, fill=sub_c, shadow=False, hdr_safe=hdr_safe)

    # Lower-right synchronized elevation sparkline card
    sp_w = max(140, int(round(310 * s)))
    sp_h = max(56, int(round(92 * s)))
    sp_x = w - int(round(w * 0.045)) - sp_w
    sp_y = h - int(round(h * 0.065)) - sp_h
    draw.rounded_rectangle(
        [sp_x, sp_y, sp_x + sp_w, sp_y + sp_h],
        radius=max(6, int(round(12 * s))),
        fill=hud_fill,
        outline=hud_border,
        width=max(1, int(round(1.5 * s))),
    )
    f_sp = _load_font_tier(max(9, int(round(12 * s))), tier=spec["sub_tier"], bold=True)
    draw_tracked_text(
        frame_rgba,
        (sp_x + int(round(14 * s)), sp_y + int(round(10 * s))),
        f"ELEV PROFILE  |  {cur_alt:.0f}M ASL",
        font=f_sp,
        fill=sub_c,
        shadow=False,
        hdr_safe=hdr_safe,
    )

    gx0 = sp_x + int(round(14 * s))
    gx1 = sp_x + sp_w - int(round(14 * s))
    gy_base = sp_y + sp_h - int(round(14 * s))
    gy_top = sp_y + int(round(34 * s))
    alt_a = max(0.0, float(orig.alt_m))
    alt_b = max(0.0, float(dest.alt_m))
    alt_max = max(alt_a, alt_b, 200.0)
    n_sp = 28
    prof_pts: list[tuple[float, float]] = []
    for k in range(n_sp):
        frac = k / float(n_sp - 1)
        ridge = math.sin(math.pi * frac) * (0.28 if transport_mode == "flight" else 0.14)
        e_norm = ((alt_a * (1.0 - frac) + alt_b * frac) / alt_max) * 0.72 + ridge
        e_norm = max(0.08, min(0.95, e_norm))
        prof_pts.append((gx0 + (gx1 - gx0) * frac, gy_base - (gy_base - gy_top) * e_norm))

    poly_fill = [(gx0, gy_base)] + prof_pts + [(gx1, gy_base)]
    draw.polygon(poly_fill, fill=(*spec["route_glow"], 54))
    draw.line(prof_pts, fill=(*spec["route_future"], 145), width=max(1, int(round(2 * s))))
    active_cnt = max(2, int(round(u * (len(prof_pts) - 1))) + 1)
    draw.line(prof_pts[:active_cnt], fill=(*spec["route_core"], 250), width=max(2, int(round(3 * s))))
    hx, hy = prof_pts[min(len(prof_pts) - 1, active_cnt - 1)]
    pr_r = max(3, int(round(4.5 * s)))
    draw.ellipse([hx - pr_r * 1.8, hy - pr_r * 1.8, hx + pr_r * 1.8, hy + pr_r * 1.8], fill=(*spec["accent"], 90))
    draw.ellipse([hx - pr_r, hy - pr_r, hx + pr_r, hy + pr_r], fill=(*spec["accent"], 255))


def render_travel_map_frame(
    waypoints: list[MapWaypointSpec | dict[str, Any]],
    style: str = "midnight_vox_atlas",
    width: int = 1920,
    height: int = 1080,
    leg_index: int = 0,
    t_norm: float = 0.65,
    transport_mode: str = "auto",
    hdr_safe: bool = False,
    compact_hud: bool = False,
    allow_tile_fetch: bool = False,
    cache_dir: Optional[str | Path] = None,
) -> Image.Image:
    """
    Render a single keyframe of the Cinema 2.5D Travel Map engine as an RGB PIL Image.

    Combines cached/procedural hillshaded terrain, geodesic splines + parabolic 3D flight arcs,
    2x-supersampled vector vehicles, destination hero thumbnail callouts, 2.5D virtual drone
    camera homography, synchronized elevation sparklines, and strict hdr_safe clamping.
    """
    norm_wps: list[MapWaypointSpec] = []
    for idx_w, w in enumerate(waypoints):
        if isinstance(w, MapWaypointSpec):
            norm_wps.append(w)
        else:
            norm_wps.append(
                MapWaypointSpec(
                    waypoint_id=str(w.get("waypoint_id") or f"wp_{idx_w + 1:02d}"),
                    chapter_id=str(w.get("chapter_id") or ""),
                    label=str(w.get("label") or f"Stop {idx_w + 1}"),
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

    use_style = style if style in CART_THEME_SPECS else "midnight_vox_atlas"
    spec = CART_THEME_SPECS[use_style]

    # Oversample world canvas by 1.45x so 2.5D perspective warp preserves razor-sharp detail
    world_w = max(640, min(2880, int(round(width * 1.45))))
    world_h = max(360, min(1620, int(round(height * 1.45))))
    wp_sig = tuple((round(w.lat, 5), round(w.lon, 5), round(w.alt_m, 1), w.label) for w in norm_wps)

    base_rgb = _cached_world_baseplate(
        use_style,
        world_w,
        world_h,
        wp_sig,
        bool(allow_tile_fetch),
        str(Path(cache_dir).resolve()) if cache_dir else "",
    )
    vp = _compute_viewport(norm_wps, world_w, world_h)

    i0 = max(0, min(len(norm_wps) - 2, leg_index))
    i1 = i0 + 1
    orig = norm_wps[i0]
    dest = norm_wps[i1]
    leg_km = haversine_km(orig.lat, orig.lon, dest.lat, dest.lon)
    resolved_mode = infer_transport_mode(leg_km, orig.label, dest.label, hint=transport_mode)

    px_stops = [vp.latlon_to_px(w.lat, w.lon) for w in norm_wps]
    all_leg_splines = [
        _catmull_rom_xy([px_stops[k], px_stops[k + 1]], samples_per_seg=140)
        for k in range(len(px_stops) - 1)
    ]

    # Composite dynamic route layer onto world canvas
    world_rgba = base_rgb.convert("RGBA")
    route_arr = np.zeros((world_h, world_w, 4), dtype=np.uint8)
    s_w = world_w / 1920.0
    stroke_w = max(2, int(round(5.0 * s_w)))

    # 1. History legs (< i0) + subtle dashed planned future path
    for k, seg_pts in enumerate(all_leg_splines):
        if k < i0:
            _draw_aa_polyline(route_arr, seg_pts, (*spec["route_history"], 175), max(2, stroke_w - 1))
        elif k >= i0:
            _draw_dashed_aa_polyline(route_arr, seg_pts, (*spec["route_future"], 140), max(1, stroke_w // 2 + 1), dash_px=11.0 * s_w, gap_px=8.0 * s_w)

    # 2. Active animated leg progress
    u = _ease_in_out_cubic(t_norm)
    active_pts = all_leg_splines[i0]
    head_idx = max(1, min(len(active_pts) - 1, int(round(u * (len(active_pts) - 1)))))
    prog_ground = active_pts[: head_idx + 1]

    if resolved_mode == "flight":
        span_px = math.hypot(active_pts[-1][0] - active_pts[0][0], active_pts[-1][1] - active_pts[0][1])
        lift_max = min(world_h * 0.22, max(28.0 * s_w, span_px * 0.24))
        arc_pts: list[tuple[float, float]] = []
        n_total = max(1, len(active_pts) - 1)
        for idx_p, (gx, gy) in enumerate(prog_ground):
            frac = idx_p / float(n_total)
            arc_pts.append((gx, gy - lift_max * math.sin(math.pi * frac)))
        # Ground drop shadow beneath elevated flight arc
        _draw_aa_polyline(route_arr, prog_ground, (8, 12, 18, 95), max(2, stroke_w))
        _draw_aa_polyline(route_arr, arc_pts, (*spec["route_glow"], 115), max(4, stroke_w * 3))
        _draw_aa_polyline(route_arr, arc_pts, (*spec["route_core"], 255), max(2, stroke_w + 1))
        head_pt = arc_pts[-1]
        prev_pt = arc_pts[max(0, len(arc_pts) - 3)]
    else:
        arc_pts = prog_ground
        _draw_aa_polyline(route_arr, prog_ground, (*spec["route_glow"], 115), max(4, stroke_w * 3))
        _draw_aa_polyline(route_arr, prog_ground, (*spec["route_core"], 255), max(2, stroke_w + 1))
        head_pt = prog_ground[-1]
        prev_pt = prog_ground[max(0, len(prog_ground) - 3)]

    world_rgba = Image.alpha_composite(world_rgba, Image.fromarray(route_arr))

    # 3. Waypoint pins + destination arrival pulse rings + hero thumbnail callout
    wd = ImageDraw.Draw(world_rgba, "RGBA")
    f_pin = _load_font_tier(max(10, int(round(13 * s_w))), tier="mono", bold=True)
    f_wp_lbl = _load_font_tier(max(11, int(round(17 * s_w))), tier=spec["title_tier"], bold=True)
    for k, (wx, wy) in enumerate(px_stops):
        is_orig = k == i0
        is_dest = k == i1
        is_done = k <= i0
        pin_r = max(6, int(round((11 if (is_orig or is_dest) else 8) * s_w)))
        if is_dest and u >= 0.72:
            pulse_p = (u - 0.72) / 0.28
            r_pulse = pin_r + int(round(26 * s_w * pulse_p))
            p_alpha = int(round(190 * (1.0 - pulse_p)))
            wd.ellipse([wx - r_pulse, wy - r_pulse, wx + r_pulse, wy + r_pulse], outline=(*spec["accent"], p_alpha), width=max(1, int(round(2.5 * s_w))))
            r_pulse2 = pin_r + int(round(14 * s_w * pulse_p))
            wd.ellipse([wx - r_pulse2, wy - r_pulse2, wx + r_pulse2, wy + r_pulse2], outline=(*spec["route_core"], int(round(p_alpha * 0.8))), width=max(1, int(round(2.0 * s_w))))

        ring_col = (*spec["accent"], 255) if (is_orig or is_dest) else (*spec["route_history"], 220)
        fill_col = (*spec["pin_fill"], 248)
        wd.ellipse([wx - pin_r - 3, wy - pin_r - 3, wx + pin_r + 3, wy + pin_r + 3], fill=ring_col)
        wd.ellipse([wx - pin_r, wy - pin_r, wx + pin_r, wy + pin_r], fill=fill_col)
        num_str = f"{k + 1:02d}"
        nw, nh = measure_tracked_text(wd, num_str, f_pin)
        wd.text((wx - nw * 0.5, wy - nh * 0.62), num_str, font=f_pin, fill=(*spec["pin_text"], 250))

        # Crisp waypoint city tag (place active destination pill below pin so incoming vehicle stays clear)
        tag = norm_wps[k].label
        tw, th = measure_tracked_text(wd, tag, f_wp_lbl)
        if is_dest:
            tx = int(round(wx - tw * 0.5))
            ty = int(round(wy + pin_r + int(round(14 * s_w))))
        else:
            tx = int(round(wx + pin_r + int(round(9 * s_w))))
            ty = int(round(wy - th * 0.55))
            if tx + tw + 16 > world_w:
                tx = int(round(wx - pin_r - tw - int(round(18 * s_w))))
        tx = max(16, min(world_w - int(tw) - 18, tx))
        ty = max(14, min(world_h - int(th) - 14, ty))
        wd.rounded_rectangle(
            [tx - int(round(7 * s_w)), ty - int(round(3 * s_w)), tx + tw + int(round(9 * s_w)), ty + th + int(round(5 * s_w))],
            radius=max(3, int(round(5 * s_w))),
            fill=(*spec["hud_fill"][:3], 188 if is_done or is_dest else 138),
            outline=ring_col if (is_orig or is_dest) else (*spec["hud_border"][:3], 90),
            width=1,
        )
        wd.text((tx, ty), tag, font=f_wp_lbl, fill=(*spec["hud_title"], 246))

    # Hero photo pop-up callout near destination (select clear quadrant away from other waypoints)
    if u >= 0.68 and dest.hero_thumbnail_path:
        bloom_p = min(1.0, (u - 0.68) / 0.22)
        _paste_hero_photo_callout(
            world_rgba=world_rgba,
            dest_px=px_stops[i1],
            thumb_path=dest.hero_thumbnail_path,
            label=dest.label,
            bloom_p=bloom_p,
            spec=spec,
            world_w=world_w,
            world_h=world_h,
            avoid_pts=[p for idx_p, p in enumerate(px_stops) if idx_p != i1],
        )

    # 4. Tangent-aligned vector transport silhouette
    heading = math.atan2(head_pt[1] - prev_pt[1], head_pt[0] - prev_pt[0])
    _draw_transport_icon(
        world_rgba,
        cx=head_pt[0],
        cy=head_pt[1],
        angle_rad=heading,
        mode=resolved_mode,
        fill_rgb=spec["route_core"],
        stroke_rgb=spec["pin_fill"] if not spec["light_paper"] else (248, 244, 238),
        scale=s_w * (0.92 if compact_hud else 1.05),
    )

    # 5. Virtual 2.5D drone camera pan/zoom/pitch/bank homography
    orig_px = px_stops[i0]
    dest_px = px_stops[i1]
    mid_cam_x = 0.5 * (orig_px[0] + dest_px[0])
    mid_cam_y = 0.5 * (orig_px[1] + dest_px[1])
    follow_x = orig_px[0] + (dest_px[0] - orig_px[0]) * u
    follow_y = orig_px[1] + (dest_px[1] - orig_px[1]) * u
    cam_cx = 0.42 * mid_cam_x + 0.58 * follow_x
    cam_cy = 0.42 * mid_cam_y + 0.58 * follow_y
    cam_zoom = 0.94 + 0.14 * math.sin(math.pi * u) + 0.08 * u
    pitch_deg = 16.0 if compact_hud else (21.0 + 5.0 * math.sin(math.pi * u))
    bank_deg = 0.0 if compact_hud else (-2.4 + 4.8 * u)

    warped_rgb = _apply_camera_homography(
        world_rgba.convert("RGB"),
        out_w=width,
        out_h=height,
        center_xy=(cam_cx, cam_cy),
        zoom=cam_zoom,
        pitch_deg=pitch_deg,
        bank_deg=bank_deg,
    )

    # 6. Composite crisp screen-space HUD over 2.5D warped plate
    out_rgba = Image.fromarray(warped_rgb).convert("RGBA")
    _render_screen_hud(
        frame_rgba=out_rgba,
        style=use_style,
        waypoints=norm_wps,
        leg_index=i0,
        t_norm=t_norm,
        transport_mode=resolved_mode,
        compact_hud=compact_hud,
        hdr_safe=hdr_safe,
    )

    out_arr = np.asarray(out_rgba.convert("RGB"), dtype=np.uint8)
    if hdr_safe:
        # Smooth proportional luminance ceiling <= 192 (~203 nits diffuse white) preserving hue & contrast
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


def render_travel_map_clip(
    waypoints: list[MapWaypointSpec | dict[str, Any]],
    output_mp4: str | Path,
    style: str = "midnight_vox_atlas",
    duration: float = 3.2,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    leg_index: Optional[int] = 0,
    transport_mode: str = "auto",
    hdr_safe: bool = False,
    allow_tile_fetch: bool = False,
    with_audio: bool = False,
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """
    Encode a multi-frame Cinema 2.5D Travel Map sequence to an H.264 MP4 clip.

    Pass `leg_index=0..N` for a single chapter leg transition, or `leg_index=None` to
    animate across all legs sequentially inside a standalone demo reel.
    """
    out_path = Path(output_mp4).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_even = max(320, int(width) - (int(width) % 2))
    h_even = max(180, int(height) - (int(height) % 2))
    dur = max(0.8, float(duration))
    total_frames = max(12, int(round(dur * fps)))

    norm_count = max(2, len(waypoints))
    n_legs = max(1, norm_count - 1)

    wav_tmp: Optional[Path] = None
    if with_audio:
        wav_tmp = out_path.with_suffix(".tmp_sfx.wav")
        _synthesize_flyover_wav(wav_tmp, duration_sec=dur)

    cmd = [
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
        cmd.extend(["-i", str(wav_tmp)])
    else:
        cmd.extend(["-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000:d={dur}"])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
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

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for f_idx in range(total_frames):
            global_p = f_idx / float(max(1, total_frames - 1))
            if leg_index is None and n_legs > 1:
                scaled = global_p * n_legs
                cur_leg = min(n_legs - 1, int(math.floor(scaled)))
                leg_p = 1.0 if f_idx == total_frames - 1 else (scaled - cur_leg)
            else:
                cur_leg = max(0, min(n_legs - 1, int(leg_index or 0)))
                leg_p = global_p

            im = render_travel_map_frame(
                waypoints=waypoints,
                style=style,
                width=w_even,
                height=h_even,
                leg_index=cur_leg,
                t_norm=leg_p,
                transport_mode=transport_mode,
                hdr_safe=hdr_safe,
                compact_hud=False,
                allow_tile_fetch=allow_tile_fetch,
                cache_dir=cache_dir,
            )
            if im.size != (w_even, h_even):
                im = im.resize((w_even, h_even), Image.Resampling.BILINEAR)
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        raise
    finally:
        if wav_tmp is not None and wav_tmp.exists():
            try:
                wav_tmp.unlink()
            except Exception:
                pass

    stderr_bytes = proc.stderr.read()
    rc = proc.wait()
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
