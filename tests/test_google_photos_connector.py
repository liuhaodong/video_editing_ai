"""Gate 15: Direct Google Photos Picker OAuth 2.0 PKCE, Two-Pass Smart Curation, and Takeout Connector."""

from __future__ import annotations

from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
import zipfile

import httpx
from PIL import Image, ImageDraw
from PIL.TiffImagePlugin import IFDRational
import pytest

from video_editing_ai.agents.base import PipelineBlackboard
from video_editing_ai.agents.dit_ingest_agent import DITIngestAgent
from video_editing_ai.agents.orchestrator import run_multi_agent_pipeline
from video_editing_ai.cloud.google_photos import (
    GooglePhotosAuthManager,
    build_autoclose_picker_uri,
    build_google_photos_web_search_url,
    filter_items_by_date_range,
    format_picker_date_query,
    ingest_google_photos_browser_downloads,
    snapshot_downloads_directory,
    sync_google_photos_media,
    sync_google_photos_via_browser,
    sync_google_takeout_media,
    watch_browser_downloads,
)
from video_editing_ai.models.schema import GooglePhotosSyncManifest


def _make_jpeg_bytes(
    width: int = 960,
    height: int = 640,
    bg_rgb: tuple[int, int, int] = (24, 52, 86),
    shape: str = "mountain",
    seed_offset: int = 0,
) -> bytes:
    im = Image.new("RGB", (width, height), bg_rgb)
    draw = ImageDraw.Draw(im)
    if shape == "mountain":
        draw.polygon(
            [(0, height), (int(width * 0.28), int(height * 0.36)), (int(width * 0.56), height)],
            fill=(54, 92, 118),
        )
        draw.polygon(
            [(int(width * 0.32), height), (int(width * 0.68), int(height * 0.25)), (width, height)],
            fill=(72, 114, 136),
        )
        draw.ellipse([int(width * 0.72), int(height * 0.14), int(width * 0.86), int(height * 0.35)], fill=(238, 198, 112))
    elif shape == "arch":
        draw.rectangle([int(width * 0.18), int(height * 0.22), int(width * 0.82), int(height * 0.85)], fill=(176, 128, 76))
        draw.rectangle([int(width * 0.36), int(height * 0.42), int(width * 0.64), int(height * 0.85)], fill=(38, 46, 62))
    elif shape == "burst":
        draw.rectangle([int(width * 0.15) + seed_offset * 4, int(height * 0.20), int(width * 0.85), int(height * 0.80)], fill=(192, 148, 86))
        draw.ellipse([int(width * 0.38) + seed_offset * 3, int(height * 0.28), int(width * 0.62) + seed_offset * 3, int(height * 0.68)], fill=(240, 210, 150))
    else:
        draw.rectangle([int(width * 0.12), int(height * 0.16), int(width * 0.88), int(height * 0.84)], fill=(86, 142, 108))
        draw.line([(0, height // 2), (width, height // 2)], fill=(232, 224, 196), width=8)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_stripped_mp4_bytes(tmp_path: Path, name: str, color: str = "steelblue", freq: int = 440, duration: float = 2.6) -> bytes:
    """Create a real MP4 without creation_time metadata so manifest hydration is genuinely tested."""
    out = tmp_path / name
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=640x360:r=15:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:sample_rate=48000:d={duration}",
            "-map_metadata",
            "-1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
    )
    return out.read_bytes()


class _MockGoogleState:
    def __init__(self, items: list[dict[str, Any]], media_payloads: dict[str, bytes], page_size_cap: int = 4) -> None:
        self.items = items
        self.media_payloads = media_payloads
        self.page_size_cap = page_size_cap
        self.token_requests: list[dict[str, str]] = []
        self.session_polls: dict[str, int] = {}
        self.deleted_sessions: list[str] = []
        self.download_requests: list[tuple[str, str]] = []
        self.created_sessions: list[dict[str, Any]] = []


class _MockGoogleHandler(BaseHTTPRequestHandler):
    def __init__(self, state: _MockGoogleState, *args: Any, **kwargs: Any) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length)
        if self.path == "/oauth/token":
            form = {k: v[0] for k, v in parse_qs(raw_body.decode("utf-8")).items()}
            self.state.token_requests.append(form)
            grant = form.get("grant_type")
            if grant == "authorization_code":
                assert form.get("code_verifier")
                self._send_json(
                    200,
                    {
                        "access_token": "mock_access_1",
                        "refresh_token": "mock_refresh_1",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    },
                )
                return
            if grant == "refresh_token":
                assert form.get("refresh_token") == "mock_refresh_1"
                self._send_json(
                    200,
                    {
                        "access_token": "mock_access_2",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    },
                )
                return
            self._send_json(400, {"error": "unsupported_grant_type"})
            return

        if self.path == "/v1/sessions":
            auth_h = self.headers.get("Authorization", "")
            if not auth_h.startswith("Bearer "):
                self._send_json(401, {"error": "unauthorized"})
                return
            req_json = json.loads(raw_body.decode("utf-8") or "{}")
            self.state.created_sessions.append(req_json)
            sess_id = f"sess_{len(self.state.created_sessions):03d}"
            self.state.session_polls[sess_id] = 0
            host_base = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
            self._send_json(
                200,
                {
                    "id": sess_id,
                    "pickerUri": f"{host_base}/picker/{sess_id}",
                    "pollingConfig": {"pollInterval": "0.05s", "timeoutIn": "60s"},
                    "pickingConfig": req_json.get("pickingConfig", {"maxItemCount": 2000}),
                },
            )
            return
        self._send_json(404, {"error": "not_found"})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/v1/sessions/"):
            sess_id = parsed.path.rsplit("/", 1)[-1]
            cnt = self.state.session_polls.get(sess_id, 0) + 1
            self.state.session_polls[sess_id] = cnt
            self._send_json(
                200,
                {
                    "id": sess_id,
                    "mediaItemsSet": cnt >= 2,
                    "pollingConfig": {"pollInterval": "0.05s", "timeoutIn": "60s"},
                },
            )
            return

        if parsed.path == "/v1/mediaItems":
            qs = parse_qs(parsed.query)
            offset = int((qs.get("pageToken") or ["0"])[0])
            req_ps = int((qs.get("pageSize") or ["100"])[0])
            step = min(req_ps, self.state.page_size_cap)
            chunk = self.state.items[offset : offset + step]
            next_tok = str(offset + step) if (offset + step) < len(self.state.items) else ""
            payload: dict[str, Any] = {"mediaItems": chunk}
            if next_tok:
                payload["nextPageToken"] = next_tok
            self._send_json(200, payload)
            return

        if parsed.path.startswith("/media/"):
            auth_h = self.headers.get("Authorization", "")
            if not auth_h.startswith("Bearer "):
                self._send_json(401, {"error": "unauthorized"})
                return
            tail = parsed.path[len("/media/") :]
            item_id, sep, mod = tail.partition("=")
            modifier = mod if sep else ""
            self.state.download_requests.append((item_id, modifier))
            blob = self.state.media_payloads.get(f"{item_id}={modifier}") or self.state.media_payloads.get(item_id)
            if blob is None:
                self._send_json(404, {"error": f"missing payload for {tail}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4" if modifier == "dv" else "image/jpeg")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return

        self._send_json(404, {"error": f"unknown path {parsed.path}"})

    def do_DELETE(self) -> None:  # noqa: N802
        if self.path.startswith("/v1/sessions/"):
            sess_id = self.path.rsplit("/", 1)[-1]
            self.state.deleted_sessions.append(sess_id)
            self._send_json(200, {})
            return
        self._send_json(404, {"error": "not_found"})


def _start_mock_server(state: _MockGoogleState) -> tuple[ThreadingHTTPServer, str]:
    handler = partial(_MockGoogleHandler, state)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _write_mock_credentials(tmp_path: Path, base_url: str, with_valid_token: bool = True) -> tuple[Path, Path]:
    secrets_p = tmp_path / "client_secret.json"
    token_p = tmp_path / "token.json"
    secrets_p.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "test-client-id.apps.googleusercontent.com",
                    "client_secret": "test-client-secret",
                    "auth_uri": f"{base_url}/oauth/authorize",
                    "token_uri": f"{base_url}/oauth/token",
                }
            }
        ),
        encoding="utf-8",
    )
    if with_valid_token:
        token_p.write_text(
            json.dumps(
                {
                    "access_token": "mock_access_1",
                    "refresh_token": "mock_refresh_1",
                    "expires_at": time.time() + 3600.0,
                    "scope": "https://www.googleapis.com/auth/photospicker.mediaitems.readonly",
                }
            ),
            encoding="utf-8",
        )
    return secrets_p, token_p


def test_oauth_pkce_loopback_and_token_refresh(tmp_path: Path) -> None:
    state = _MockGoogleState(items=[], media_payloads={})
    srv, base_url = _start_mock_server(state)
    try:
        secrets_p, token_p = _write_mock_credentials(tmp_path, base_url, with_valid_token=False)
        auth = GooglePhotosAuthManager(client_secrets_path=secrets_p, token_path=token_p)
        captured_urls: list[str] = []

        def _simulate_browser_consent(auth_url: str) -> None:
            captured_urls.append(auth_url)
            qs = parse_qs(urlparse(auth_url).query)
            assert qs.get("code_challenge_method") == ["S256"]
            assert len(qs["code_challenge"][0]) >= 40
            redir = qs["redirect_uri"][0]
            cb_url = f"{redir}?{urlencode({'code': 'auth_code_xyz', 'state': qs['state'][0]})}"
            r = httpx.get(cb_url, timeout=5.0)
            assert r.status_code == 200

        tok = auth.authenticate_interactive(open_browser=False, timeout_sec=10.0, auth_url_callback=_simulate_browser_consent)
        assert len(captured_urls) == 1
        assert tok["access_token"] == "mock_access_1"
        assert tok["refresh_token"] == "mock_refresh_1"
        assert token_p.is_file()
        assert auth.get_auth_status()["ready"] is True

        # Force expiration -> get_valid_access_token must refresh cleanly via mock_refresh_1
        expired = json.loads(token_p.read_text(encoding="utf-8"))
        expired["expires_at"] = time.time() - 30.0
        token_p.write_text(json.dumps(expired), encoding="utf-8")

        refreshed = auth.get_valid_access_token()
        assert refreshed == "mock_access_2"
        assert state.token_requests[-1]["grant_type"] == "refresh_token"
        assert json.loads(token_p.read_text(encoding="utf-8"))["refresh_token"] == "mock_refresh_1"
    finally:
        srv.shutdown()
        srv.server_close()


def test_picker_session_polling_and_strict_date_range_filtering(tmp_path: Path) -> None:
    assert build_autoclose_picker_uri("https://photos.google.com/picker/abc") == "https://photos.google.com/picker/abc/autoclose"
    assert format_picker_date_query("2026-07-01", "2026-07-07") == "2026-07-01 to 2026-07-07"

    raw_items = [
        {"id": "jun_30", "createTime": "2026-06-30T23:59:59Z", "type": "PHOTO", "mediaFile": {"filename": "pre.jpg"}},
        {"id": "jul_01", "createTime": "2026-07-01T00:00:00Z", "type": "PHOTO", "mediaFile": {"filename": "in1.jpg"}},
        {"id": "jul_07", "createTime": "2026-07-07T23:59:59Z", "type": "PHOTO", "mediaFile": {"filename": "in2.jpg"}},
        {"id": "jul_08", "createTime": "2026-07-08T00:00:00Z", "type": "PHOTO", "mediaFile": {"filename": "post.jpg"}},
    ]
    filtered = filter_items_by_date_range(raw_items, "2026-07-01", "2026-07-07", tz_name="UTC")
    assert [it["id"] for it in filtered] == ["jul_01", "jul_07"]

    # Local timezone shift check: 2026-07-01T01:00Z is still June 30 in UTC-7!
    filtered_la = filter_items_by_date_range(
        [
            {"id": "utc_early_jul1", "createTime": "2026-07-01T02:00:00Z"},
            {"id": "la_morning_jul1", "createTime": "2026-07-01T16:00:00Z"},
        ],
        "2026-07-01",
        "2026-07-01",
        tz_name="America/Los_Angeles",
    )
    assert [it["id"] for it in filtered_la] == ["la_morning_jul1"]

    # Mock paginated session poll + cleanup delete
    img_bytes = _make_jpeg_bytes(480, 320, shape="mountain")
    mock_items = []
    payloads: dict[str, bytes] = {}
    for row in raw_items:
        iid = row["id"]
        mock_items.append(
            {
                **row,
                "mediaFile": {
                    "baseUrl": f"PLACEHOLDER/media/{iid}",
                    "mimeType": "image/jpeg",
                    "filename": row["mediaFile"]["filename"],
                    "mediaFileMetadata": {"width": 960, "height": 640, "cameraMake": "Sony", "cameraModel": "A7"},
                },
            }
        )
        payloads[f"{iid}=w512-h512"] = img_bytes
        payloads[f"{iid}=d"] = img_bytes

    state = _MockGoogleState(items=mock_items, media_payloads=payloads, page_size_cap=2)
    srv, base_url = _start_mock_server(state)
    try:
        for it in mock_items:
            it["mediaFile"]["baseUrl"] = it["mediaFile"]["baseUrl"].replace("PLACEHOLDER", base_url)
        secrets_p, token_p = _write_mock_credentials(tmp_path, base_url, with_valid_token=True)
        manifest = sync_google_photos_media(
            project_dir=tmp_path / "proj",
            date_from="2026-07-01",
            date_to="2026-07-07",
            open_browser=False,
            client_secrets_path=secrets_p,
            token_path=token_p,
            cache_dir=tmp_path / "gp_cache",
            api_base_url=f"{base_url}/v1",
        )
        assert manifest.total_scanned_in_session == 4
        assert manifest.in_date_range_count == 2
        assert {it.cloud_item_id for it in manifest.items} == {"jul_01", "jul_07"}
        assert state.deleted_sessions == ["sess_001"]
        assert state.session_polls["sess_001"] >= 2
    finally:
        srv.shutdown()
        srv.server_close()


def test_two_pass_smart_curation_selective_downloads_and_cache_hits(tmp_path: Path) -> None:
    vid_bytes = _make_stripped_mp4_bytes(tmp_path, "src_vid.mp4", color="teal", duration=2.4)
    burst_jpg = _make_jpeg_bytes(640, 480, shape="burst", seed_offset=0)
    burst_jpg_2 = _make_jpeg_bytes(640, 480, shape="burst", seed_offset=1)
    distinct_a = _make_jpeg_bytes(640, 480, bg_rgb=(20, 44, 78), shape="mountain")
    distinct_b = _make_jpeg_bytes(640, 480, bg_rgb=(110, 62, 42), shape="arch")
    distinct_c = _make_jpeg_bytes(640, 480, bg_rgb=(42, 88, 62), shape="grid")

    specs = [
        ("p_burst_1", "2026-07-02T10:00:00Z", "PHOTO", "READY", "IMG_101.JPG", burst_jpg),
        ("p_burst_2", "2026-07-02T10:00:02Z", "PHOTO", "READY", "IMG_102.JPG", burst_jpg_2),
        ("p_burst_3", "2026-07-02T10:00:04Z", "PHOTO", "READY", "IMG_103.JPG", burst_jpg),
        ("p_dist_1", "2026-07-03T11:00:00Z", "PHOTO", "READY", "IMG_201.JPG", distinct_a),
        ("p_dist_2", "2026-07-04T15:00:00Z", "PHOTO", "READY", "IMG_202.JPG", distinct_b),
        ("p_dist_3", "2026-07-05T18:00:00Z", "PHOTO", "READY", "IMG_203.JPG", distinct_c),
        ("v_proc", "2026-07-02T09:00:00Z", "VIDEO", "PROCESSING", "VID_WAIT.MP4", vid_bytes),
        ("v_day1", "2026-07-02T12:00:00Z", "VIDEO", "READY", "VID_01.MP4", vid_bytes),
        ("v_day2", "2026-07-03T12:00:00Z", "VIDEO", "READY", "VID_02.MP4", vid_bytes),
        ("v_day3", "2026-07-04T12:00:00Z", "VIDEO", "READY", "VID_03.MP4", vid_bytes),
        ("v_day4", "2026-07-05T12:00:00Z", "VIDEO", "READY", "VID_04.MP4", vid_bytes),
    ]

    mock_items: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    state = _MockGoogleState(items=mock_items, media_payloads=payloads, page_size_cap=6)
    srv, base_url = _start_mock_server(state)
    try:
        for iid, ts, mtype, v_stat, fname, blob in specs:
            is_vid = mtype == "VIDEO"
            mf: dict[str, Any] = {
                "baseUrl": f"{base_url}/media/{iid}",
                "mimeType": "video/mp4" if is_vid else "image/jpeg",
                "filename": fname,
                "mediaFileMetadata": {
                    "width": 1920 if is_vid else 4000,
                    "height": 1080 if is_vid else 3000,
                    "cameraMake": "Google",
                    "cameraModel": "Pixel",
                    "videoMetadata": {"fps": 30.0, "processingStatus": v_stat} if is_vid else {},
                },
            }
            mock_items.append({"id": iid, "createTime": ts, "type": mtype, "mediaFile": mf})
            payloads[f"{iid}=w512-h512"] = distinct_a if is_vid else blob
            payloads[f"{iid}={'dv' if is_vid else 'd'}"] = blob

        secrets_p, token_p = _write_mock_credentials(tmp_path, base_url, with_valid_token=True)
        cache_dir = tmp_path / "persistent_cache"
        proj_dir = tmp_path / "smart_proj"

        manifest = sync_google_photos_media(
            project_dir=proj_dir,
            date_from="2026-07-01",
            date_to="2026-07-06",
            max_photos=3,
            max_videos=2,
            open_browser=False,
            client_secrets_path=secrets_p,
            token_path=token_p,
            cache_dir=cache_dir,
            api_base_url=f"{base_url}/v1",
        )
        assert manifest.selected_photos_count <= 3
        assert manifest.selected_videos_count == 2
        assert manifest.skipped_count >= 6
        assert any(it.skip_reason == "burst_duplicate" for it in manifest.items)
        assert any(it.cloud_item_id == "v_proc" and it.skip_reason == "video_processing_not_ready" for it in manifest.items)
        assert Path(manifest.scan_contact_sheet_path or "").is_file()

        # Assert full-res downloads (=d / =dv) happened ONLY for selected_for_download=True items!
        sel_ids = {it.cloud_item_id for it in manifest.items if it.selected_for_download}
        skipped_ids = {it.cloud_item_id for it in manifest.items if not it.selected_for_download}
        full_reqs = {(iid, mod) for iid, mod in state.download_requests if mod in ("d", "dv")}
        assert {iid for iid, _ in full_reqs} == sel_ids
        assert not (skipped_ids & {iid for iid, _ in full_reqs})

        staged_files = list(Path(manifest.staged_media_dir).iterdir())
        assert len(staged_files) == len(sel_ids)

        # Second sync with same cache must incur ZERO =d/=dv network downloads
        state.download_requests.clear()
        manifest2 = sync_google_photos_media(
            project_dir=tmp_path / "smart_proj_2",
            date_from="2026-07-01",
            date_to="2026-07-06",
            max_photos=3,
            max_videos=2,
            open_browser=False,
            client_secrets_path=secrets_p,
            token_path=token_p,
            cache_dir=cache_dir,
            api_base_url=f"{base_url}/v1",
        )
        assert manifest2.cache_hits_count == len(sel_ids)
        assert [r for r in state.download_requests if r[1] in ("d", "dv")] == []
    finally:
        srv.shutdown()
        srv.server_close()


def test_google_takeout_zip_date_filter_and_dit_gps_hydration(tmp_path: Path) -> None:
    takeout_zip = tmp_path / "takeout-export.zip"
    vid_bytes = _make_stripped_mp4_bytes(tmp_path, "clip.mp4", color="navy", duration=2.2)
    jpg_1 = _make_jpeg_bytes(640, 480, shape="mountain")
    jpg_2 = _make_jpeg_bytes(640, 480, shape="arch")
    jpg_old = _make_jpeg_bytes(640, 480, shape="grid")

    # Unix timestamps: 2026-07-02T10:00Z = 1782986400, 2026-07-04T14:00Z = 1783173600, 2026-05-01T12:00Z = 1777636800
    with zipfile.ZipFile(takeout_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        base = "Takeout/Google Photos/Summer 2026/"
        zf.writestr(base + "IMG_OLD.JPG", jpg_old)
        zf.writestr(
            base + "IMG_OLD.JPG.supplemental-metadata.json",
            json.dumps({"photoTakenTime": {"timestamp": "1777636800"}, "geoData": {"latitude": 0.0, "longitude": 0.0}}),
        )
        zf.writestr(base + "TOKYO_GATE.JPG", jpg_1)
        zf.writestr(
            base + "TOKYO_GATE.JPG.supplemental-metadata.json",
            json.dumps(
                {
                    "photoTakenTime": {"timestamp": "1782986400"},
                    "geoData": {"latitude": 35.6762, "longitude": 139.6503, "altitude": 42.0},
                }
            ),
        )
        zf.writestr(base + "KYOTO_SHRINE.JPG", jpg_2)
        zf.writestr(
            base + "KYOTO_SHRINE.JPG.json",
            json.dumps(
                {
                    "photoTakenTime": {"timestamp": "1783173600"},
                    "geoData": {"latitude": 35.0116, "longitude": 135.7681, "altitude": 118.5},
                }
            ),
        )
        zf.writestr(base + "STRIPPED_CLIP.MP4", vid_bytes)
        zf.writestr(
            base + "STRIPPED_CLIP.MP4.supplemental-metadata.json",
            json.dumps(
                {
                    "photoTakenTime": {"timestamp": "1782990000"},
                    "geoData": {"latitude": 35.2324, "longitude": 139.1069, "altitude": 725.0},
                }
            ),
        )

    proj_dir = tmp_path / "takeout_proj"
    manifest = sync_google_takeout_media(
        takeout_source=takeout_zip,
        project_dir=proj_dir,
        date_from="2026-07-01",
        date_to="2026-07-07",
    )
    assert manifest.in_date_range_count == 3
    assert manifest.selected_photos_count == 2
    assert manifest.selected_videos_count == 1
    assert all("IMG_OLD" not in it.filename for it in manifest.items)

    # Hydrate into DITIngestAgent -> GPS and capture timestamp must populate even though MP4 is stripped
    bb = PipelineBlackboard(media_dir=manifest.staged_media_dir, project_dir=proj_dir, use_antigravity=False)
    res = DITIngestAgent().run(bb)
    assert res["entries_count"] == 3
    assert bb.dit_manifest is not None
    by_name = {ent.filename: ent for ent in bb.dit_manifest.entries}
    vid_ent = by_name["STRIPPED_CLIP.MP4"]
    assert vid_ent.creation_time_raw is not None and vid_ent.creation_time_raw.startswith("2026-07-02")
    assert vid_ent.gps_lat is not None and abs(vid_ent.gps_lat - 35.2324) < 0.01
    assert vid_ent.gps_alt is not None and abs(vid_ent.gps_alt - 725.0) < 0.5
    tok_ent = by_name["TOKYO_GATE.JPG"]
    assert tok_ent.gps_lat is not None and abs(tok_ent.gps_lat - 35.6762) < 0.01


def test_end_to_end_mock_google_photos_to_rendered_pipeline(tmp_path: Path) -> None:
    vid_1 = _make_stripped_mp4_bytes(tmp_path, "v1.mp4", color="steelblue", freq=330, duration=3.0)
    vid_2 = _make_stripped_mp4_bytes(tmp_path, "v2.mp4", color="seagreen", freq=440, duration=3.0)
    p1 = _make_jpeg_bytes(640, 480, shape="mountain")
    p2 = _make_jpeg_bytes(640, 480, shape="arch")

    mock_items: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    state = _MockGoogleState(items=mock_items, media_payloads=payloads)
    srv, base_url = _start_mock_server(state)
    try:
        raw_specs = [
            ("gp_v1", "2026-07-02T09:00:00Z", "VIDEO", "GP_CLIP_01.MP4", vid_1, p1),
            ("gp_p1", "2026-07-02T10:30:00Z", "PHOTO", "GP_IMG_01.JPG", p1, p1),
            ("gp_v2", "2026-07-04T15:00:00Z", "VIDEO", "GP_CLIP_02.MP4", vid_2, p2),
            ("gp_p2", "2026-07-04T17:15:00Z", "PHOTO", "GP_IMG_02.JPG", p2, p2),
        ]
        for iid, ts, mtype, fname, blob, thumb_blob in raw_specs:
            is_vid = mtype == "VIDEO"
            mock_items.append(
                {
                    "id": iid,
                    "createTime": ts,
                    "type": mtype,
                    "mediaFile": {
                        "baseUrl": f"{base_url}/media/{iid}",
                        "mimeType": "video/mp4" if is_vid else "image/jpeg",
                        "filename": fname,
                        "mediaFileMetadata": {
                            "width": 640,
                            "height": 360 if is_vid else 480,
                            "cameraMake": "Nikon",
                            "cameraModel": "Zf",
                            "videoMetadata": {"fps": 15.0, "processingStatus": "READY"} if is_vid else {},
                        },
                    },
                }
            )
            payloads[f"{iid}=w512-h512"] = thumb_blob
            payloads[f"{iid}={'dv' if is_vid else 'd'}"] = blob

        secrets_p, token_p = _write_mock_credentials(tmp_path, base_url, with_valid_token=True)
        proj_dir = tmp_path / "e2e_gp_proj"
        manifest = sync_google_photos_media(
            project_dir=proj_dir,
            date_from="2026-07-01",
            date_to="2026-07-05",
            open_browser=False,
            client_secrets_path=secrets_p,
            token_path=token_p,
            cache_dir=tmp_path / "e2e_cache",
            api_base_url=f"{base_url}/v1",
        )
        bb = run_multi_agent_pipeline(
            media_dir=manifest.staged_media_dir,
            project_dir=proj_dir,
            user_prompt="Create a scenic July road trip documentary from Google Photos.",
            width=640,
            height=360,
            fps=15,
            hdr=False,
            render_video=True,
            use_antigravity=False,
            skip_whisper=True,
            skip_auto_bgm=True,
        )
        assert bb.google_photos_manifest is not None
        assert bb.dit_manifest is not None
        hydrated_vid = next(e for e in bb.dit_manifest.entries if e.filename == "GP_CLIP_01.MP4")
        assert hydrated_vid.camera_make_model == "Nikon Zf"
        assert hydrated_vid.creation_time_raw == "2026-07-02T09:00:00Z"
        assert (proj_dir / "final_story.mp4").is_file()
        assert (proj_dir / "timeline.fcpxml").is_file()
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_and_skill_google_photos_commands(tmp_path: Path) -> None:
    from typer.testing import CliRunner
    from video_editing_ai.cli import app

    runner = CliRunner()
    res_auth = runner.invoke(app, ["google-photos-auth", "--status"])
    assert res_auth.exit_code == 0
    assert "Google Photos Connection Status" in res_auth.output
    assert "Browser Login (Zero-Setup Default)" in res_auth.output

    res_doc = runner.invoke(app, ["doctor", "--json"])
    assert res_doc.exit_code == 0
    doc_json = json.loads(res_doc.output)
    assert "google_photos" in doc_json
    assert doc_json["google_photos"]["browser_login_ready"] is True

    # Skill helper sync-google-photos + inspect-library using Takeout folder without repeating --media-dir
    takeout_dir = tmp_path / "TakeoutFolder"
    takeout_dir.mkdir(parents=True, exist_ok=True)
    img_path = takeout_dir / "PARK.JPG"
    img_path.write_bytes(_make_jpeg_bytes(480, 320, shape="arch"))
    (takeout_dir / "PARK.JPG.supplemental-metadata.json").write_text(
        json.dumps({"photoTakenTime": {"timestamp": "1782986400"}, "geoData": {"latitude": 35.68, "longitude": 139.76}}),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parent.parent
    skill_script = repo_root / ".agent" / "skills" / "ai-video-editor" / "scripts" / "ai_video_studio_skill.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proj_dir = tmp_path / "skill_cloud_proj"

    subprocess.run(
        [
            sys.executable,
            str(skill_script),
            "sync-google-photos",
            "--project-dir",
            str(proj_dir),
            "--takeout",
            str(takeout_dir),
            "--date-from",
            "2026-07-01",
            "--date-to",
            "2026-07-05",
        ],
        env=env,
        check=True,
    )
    loaded = GooglePhotosSyncManifest.from_json_file(proj_dir / "google_photos_manifest.json")
    assert loaded.selected_photos_count == 1

    # inspect-library without --media-dir resolves staged cloud_media from project_meta.json
    subprocess.run(
        [
            sys.executable,
            str(skill_script),
            "inspect-library",
            "--project-dir",
            str(proj_dir),
            "--prompt",
            "Quick park story",
        ],
        env=env,
        check=True,
    )
    assert (proj_dir / "library_inspection.json").is_file()


def _make_exif_jpeg_bytes(
    dt_str: str,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    shape: str = "mountain",
    bg_rgb: tuple[int, int, int] = (24, 52, 86),
    seed_offset: int = 0,
) -> bytes:
    im = Image.open(BytesIO(_make_jpeg_bytes(640, 480, bg_rgb=bg_rgb, shape=shape, seed_offset=seed_offset)))
    exif = Image.Exif()
    exif[306] = dt_str
    exif[36867] = dt_str
    exif[0x8825] = {
        1: "N" if lat_deg >= 0 else "S",
        2: (
            IFDRational(int(abs(lat_deg)), 1),
            IFDRational(int((abs(lat_deg) % 1) * 60), 1),
            IFDRational(int((((abs(lat_deg) * 60) % 1) * 60) * 100), 100),
        ),
        3: "E" if lon_deg >= 0 else "W",
        4: (
            IFDRational(int(abs(lon_deg)), 1),
            IFDRational(int((abs(lon_deg) % 1) * 60), 1),
            IFDRational(int((((abs(lon_deg) * 60) % 1) * 60) * 100), 100),
        ),
        5: b"\x00" if alt_m >= 0 else b"\x01",
        6: IFDRational(int(round(abs(alt_m) * 10)), 10),
    }
    out = BytesIO()
    im.save(out, format="JPEG", quality=92, exif=exif)
    return out.getvalue()


def test_google_photos_web_search_url_and_live_crdownload_watcher(tmp_path: Path) -> None:
    assert (
        build_google_photos_web_search_url("2026-07-01", "2026-07-07")
        == "https://photos.google.com/search/2026-07-01%20to%202026-07-07"
    )
    assert build_google_photos_web_search_url(None, None) == "https://photos.google.com"

    dl_dir = tmp_path / "Downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    pre_existing = dl_dir / "OLD_PHOTO.JPG"
    pre_existing.write_bytes(_make_jpeg_bytes(320, 240))

    baseline = snapshot_downloads_directory(dl_dir)
    started_at = time.time()

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("IMG_NEW.JPG", _make_jpeg_bytes(480, 320, shape="arch"))
    valid_zip_bytes = zip_buf.getvalue()

    def _simulate_chrome_download() -> None:
        temp_part = dl_dir / "Photos-001.zip.crdownload"
        temp_part.write_bytes(valid_zip_bytes[:128])
        time.sleep(0.35)
        final_zip = dl_dir / "Photos-001.zip"
        final_zip.write_bytes(valid_zip_bytes)
        temp_part.unlink()

    th = threading.Thread(target=_simulate_chrome_download, daemon=True)
    th.start()
    found = watch_browser_downloads(
        dl_dir,
        started_after_epoch=started_at,
        baseline_files=baseline,
        timeout_sec=10.0,
        settle_quiet_sec=0.2,
        poll_interval_sec=0.05,
    )
    th.join(timeout=5.0)
    assert [p.name for p in found] == ["Photos-001.zip"]


def test_browser_login_zip_preserves_exif_gps_bursts_and_dit_hydration(tmp_path: Path) -> None:
    dl_dir = tmp_path / "Downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    photos_zip_src = tmp_path / "staged_browser_download.zip"

    # Create a real MP4 carrying QuickTime ISO-6709 GPS & creation_time metadata (as downloaded via Shift+D from photos.google.com)
    mp4_path = tmp_path / "KYOTO_RAIL.MP4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=teal:s=640x360:r=15:d=2.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:d=2.2",
            "-metadata",
            "creation_time=2026-07-04T12:00:00Z",
            "-metadata",
            "location=+35.0116+135.7681+118.0/",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(mp4_path),
        ],
        check=True,
    )

    with zipfile.ZipFile(photos_zip_src, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("OUT_OF_RANGE.JPG", _make_exif_jpeg_bytes("2026:06:15 09:00:00", 35.67, 139.65, 20.0, shape="grid"))
        zf.writestr("BURST_01.JPG", _make_exif_jpeg_bytes("2026:07:02 10:00:00", 35.6762, 139.6503, 42.0, shape="burst", seed_offset=0))
        zf.writestr("BURST_02.JPG", _make_exif_jpeg_bytes("2026:07:02 10:00:02", 35.6762, 139.6503, 42.0, shape="burst", seed_offset=1))
        zf.writestr("TOKYO_TOWER.JPG", _make_exif_jpeg_bytes("2026:07:03 16:30:00", 35.6586, 139.7454, 88.0, shape="arch", bg_rgb=(110, 62, 42)))
        zf.writestr("KYOTO_RAIL.MP4", mp4_path.read_bytes())

    opened_urls: list[str] = []

    def _on_browser_open(url: str, watched_dir: Path) -> None:
        opened_urls.append(url)
        (watched_dir / "Photos-001.zip").write_bytes(photos_zip_src.read_bytes())

    proj_dir = tmp_path / "browser_proj"
    manifest = sync_google_photos_via_browser(
        project_dir=proj_dir,
        date_from="2026-07-01",
        date_to="2026-07-06",
        max_photos=2,
        downloads_dir=dl_dir,
        timeout_sec=10.0,
        settle_quiet_sec=0.15,
        open_browser=False,
        browser_open_callback=_on_browser_open,
    )
    assert opened_urls == ["https://photos.google.com/search/2026-07-01%20to%202026-07-06"]
    assert manifest.source_mode == "google_photos_browser"
    assert manifest.in_date_range_count == 4
    assert all(it.filename != "OUT_OF_RANGE.JPG" for it in manifest.items)
    assert manifest.selected_photos_count <= 2
    assert manifest.selected_videos_count == 1
    assert any(it.skip_reason == "burst_duplicate" for it in manifest.items)

    bb = PipelineBlackboard(media_dir=manifest.staged_media_dir, project_dir=proj_dir, use_antigravity=False)
    DITIngestAgent().run(bb)
    assert bb.dit_manifest is not None
    by_name = {ent.filename: ent for ent in bb.dit_manifest.entries}
    vid_ent = by_name["KYOTO_RAIL.MP4"]
    assert vid_ent.gps_lat is not None and abs(vid_ent.gps_lat - 35.0116) < 0.01
    assert vid_ent.gps_alt is not None and abs(vid_ent.gps_alt - 118.0) < 1.0
    tok_ent = by_name["TOKYO_TOWER.JPG"]
    assert tok_ent.gps_lat is not None and abs(tok_ent.gps_lat - 35.6586) < 0.01
    assert tok_ent.gps_alt is not None and abs(tok_ent.gps_alt - 88.0) < 1.0


def test_cli_and_skill_zero_setup_browser_default_and_from_downloads(tmp_path: Path) -> None:
    from typer.testing import CliRunner
    from video_editing_ai.cli import app

    repo_root = Path(__file__).resolve().parent.parent
    skill_script = repo_root / ".agent" / "skills" / "ai-video-editor" / "scripts" / "ai_video_studio_skill.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    # 1. Skill --search-url-only returns browser deep-link with zero OAuth secrets
    proc = subprocess.run(
        [
            sys.executable,
            str(skill_script),
            "sync-google-photos",
            "--search-url-only",
            "--date-from",
            "2026-07-01",
            "--date-to",
            "2026-07-07",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(proc.stdout)
    assert info["search_url"] == "https://photos.google.com/search/2026-07-01%20to%202026-07-07"
    assert info["date_query"] == "2026-07-01 to 2026-07-07"

    # 2. CLI google-photos-sync --from-downloads imports browser ZIP cleanly with zero OAuth config
    photos_zip = tmp_path / "Photos-from-browser.zip"
    with zipfile.ZipFile(photos_zip, "w") as zf:
        zf.writestr("BAY.JPG", _make_exif_jpeg_bytes("2026:07:03 11:00:00", 37.808, -122.417, 15.0, shape="mountain"))

    proj_dir = tmp_path / "cli_browser_proj"
    res = CliRunner().invoke(
        app,
        [
            "google-photos-sync",
            "-o",
            str(proj_dir),
            "--from-downloads",
            str(photos_zip),
            "--date-from",
            "2026-07-01",
            "--date-to",
            "2026-07-07",
        ],
    )
    assert res.exit_code == 0, res.output
    manifest = GooglePhotosSyncManifest.from_json_file(proj_dir / "google_photos_manifest.json")
    assert manifest.source_mode == "google_photos_browser"
    assert manifest.selected_photos_count == 1


def test_favorites_boost_and_guaranteed_selection(tmp_path: Path) -> None:
    dl_dir = tmp_path / "dl_fav"
    dl_dir.mkdir(parents=True, exist_ok=True)
    # Create a burst pair + two alternate scenes; star BURST_STARRED.JPG and ALT_STARRED.JPG
    (dl_dir / "BURST_A.JPG").write_bytes(_make_exif_jpeg_bytes("2025:12:19 17:53:10", 33.83, -116.54, 140.0, shape="burst", seed_offset=0))
    (dl_dir / "BURST_STARRED.JPG").write_bytes(_make_exif_jpeg_bytes("2025:12:19 17:53:11", 33.83, -116.54, 140.0, shape="burst", seed_offset=1))
    (dl_dir / "PLAIN_SCENE.JPG").write_bytes(_make_exif_jpeg_bytes("2025:12:20 12:00:00", 33.09, -116.99, 120.0, shape="arch"))
    (dl_dir / "ALT_STARRED.JPG").write_bytes(_make_exif_jpeg_bytes("2025:12:25 16:00:00", 33.87, -115.90, 900.0, shape="mountain"))

    proj_dir = tmp_path / "fav_proj"
    manifest = ingest_google_photos_browser_downloads(
        download_paths=dl_dir,
        project_dir=proj_dir,
        date_from="2025-12-18",
        date_to="2025-12-25",
        tz_name="local",
        max_photos=2,
        favorite_filenames="BURST_STARRED.JPG,ALT_STARRED.JPG",
    )
    by_name = {it.filename: it for it in manifest.items}
    assert by_name["BURST_STARRED.JPG"].is_favorite is True
    assert by_name["BURST_STARRED.JPG"].selected_for_download is True
    assert by_name["BURST_A.JPG"].selected_for_download is False
    assert by_name["BURST_A.JPG"].skip_reason == "burst_duplicate"
    assert by_name["ALT_STARRED.JPG"].is_favorite is True
    assert by_name["ALT_STARRED.JPG"].selected_for_download is True
    assert by_name["PLAIN_SCENE.JPG"].selected_for_download is False

    bb = PipelineBlackboard(media_dir=manifest.staged_media_dir, project_dir=proj_dir, use_antigravity=False)
    DITIngestAgent().run(bb)
    assert bb.dit_manifest is not None
    dit_by_name = {ent.filename: ent for ent in bb.dit_manifest.entries}
    assert dit_by_name["BURST_STARRED.JPG"].is_favorite is True
    assert dit_by_name["ALT_STARRED.JPG"].is_favorite is True


def test_screenshot_live_photo_and_duplicate_video_rejection(tmp_path: Path) -> None:
    """Verify phone screenshots, Live Photo micro-.MOVs & duplicate videos are skipped while camera signs & favs stay."""
    from video_editing_ai.cloud.google_photos import select_necessary_cloud_items
    from video_editing_ai.models.schema import CloudMediaCandidateItem

    # Build real thumbnail files on disk
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    flat_ui = Image.new("RGB", (320, 640), (245, 245, 248))
    d_ui = ImageDraw.Draw(flat_ui)
    d_ui.rectangle([0, 0, 320, 64], fill=(28, 32, 42))
    d_ui.rectangle([20, 90, 300, 150], fill=(255, 255, 255))
    ss_p = thumbs / "ss.png"
    flat_ui.save(ss_p, "PNG")

    cam_p = thumbs / "cam_sign.jpg"
    cam_p.write_bytes(_make_jpeg_bytes(640, 480, shape="arch", bg_rgb=(96, 72, 48)))
    vid_t = thumbs / "vid_t.jpg"
    vid_t.write_bytes(_make_jpeg_bytes(640, 360, shape="mountain", bg_rgb=(30, 64, 96)))

    cands = [
        # 1. Phone screenshot by filename -> skipped
        CloudMediaCandidateItem(
            cloud_item_id="c_ss_name", filename="Screenshot_20260702-101500.png", mime_type="image/png",
            media_type="photo", creation_time="2026-07-02T10:15:00Z", width=1080, height=2400,
            proxy_thumbnail_path=str(ss_p),
        ),
        # 2. Unnamed flat-UI PNG without camera EXIF -> skipped
        CloudMediaCandidateItem(
            cloud_item_id="c_ss_flat", filename="IMG_9991.png", mime_type="image/png",
            media_type="photo", creation_time="2026-07-02T10:20:00Z", width=1170, height=2532,
            proxy_thumbnail_path=str(ss_p),
        ),
        # 3. Starred screenshot -> protected!
        CloudMediaCandidateItem(
            cloud_item_id="c_ss_fav", filename="Screenshot_important_ticket.png", mime_type="image/png",
            media_type="photo", creation_time="2026-07-02T10:25:00Z", width=1080, height=2400,
            proxy_thumbnail_path=str(ss_p), is_favorite=True,
        ),
        # 4. Camera-captured museum plaque / sign -> KEPT!
        CloudMediaCandidateItem(
            cloud_item_id="c_cam_sign", filename="PXL_20260702_103000.jpg", mime_type="image/jpeg",
            media_type="photo", creation_time="2026-07-02T10:30:00Z", width=4032, height=3024,
            camera_make_model="Google Pixel 9 Pro", proxy_thumbnail_path=str(cam_p),
        ),
        # 5. Paired still + Apple Live Photo companion micro-.MOV (1.9s) -> .MOV skipped!
        CloudMediaCandidateItem(
            cloud_item_id="c_live_jpg", filename="IMG_5001.JPG", mime_type="image/jpeg",
            media_type="photo", creation_time="2026-07-02T11:00:00Z", width=4032, height=3024,
            camera_make_model="Apple iPhone 16 Pro", proxy_thumbnail_path=str(vid_t),
        ),
        CloudMediaCandidateItem(
            cloud_item_id="c_live_mov", filename="IMG_5001.MOV", mime_type="video/quicktime",
            media_type="video", creation_time="2026-07-02T11:00:00Z", width=1920, height=1440,
            fps=30.0, duration_sec=1.9, proxy_thumbnail_path=str(vid_t),
        ),
        # 6. Duplicate re-uploaded 4K videos at identical timestamp -> _1 skipped!
        CloudMediaCandidateItem(
            cloud_item_id="c_vid_orig", filename="A7C2_20260702_0618.MP4", mime_type="video/mp4",
            media_type="video", creation_time="2026-07-02T14:00:00Z", width=3840, height=2160,
            fps=29.97, duration_sec=12.0, proxy_thumbnail_path=str(vid_t),
        ),
        CloudMediaCandidateItem(
            cloud_item_id="c_vid_dup", filename="A7C2_20260702_0618_1.MP4", mime_type="video/mp4",
            media_type="video", creation_time="2026-07-02T14:00:00Z", width=3840, height=2160,
            fps=29.97, duration_sec=12.0, proxy_thumbnail_path=str(vid_t),
        ),
    ]
    select_necessary_cloud_items(cands, target_duration_sec=120.0)
    by_id = {c.cloud_item_id: c for c in cands}

    assert by_id["c_ss_name"].selected_for_download is False and by_id["c_ss_name"].skip_reason == "screenshot_or_junk"
    assert by_id["c_ss_flat"].selected_for_download is False and by_id["c_ss_flat"].skip_reason == "screenshot_or_junk"
    assert by_id["c_ss_fav"].selected_for_download is True
    assert by_id["c_cam_sign"].selected_for_download is True
    assert by_id["c_live_mov"].selected_for_download is False and by_id["c_live_mov"].skip_reason == "live_photo_companion"
    assert by_id["c_vid_orig"].selected_for_download is True
    assert by_id["c_vid_dup"].selected_for_download is False and by_id["c_vid_dup"].skip_reason == "duplicate_video"


def test_non_adjacent_cross_camera_dedup_and_gps_inheritance(tmp_path: Path) -> None:
    """Verify Sony (no GPS) + Pixel (GPS) shots 6 minutes apart dedup into Sony hero AND inherit Pixel GPS."""
    from video_editing_ai.cloud.google_photos import select_necessary_cloud_items
    from video_editing_ai.models.schema import CloudMediaCandidateItem

    thumbs = tmp_path / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    same_scene_a = thumbs / "sony_gate.jpg"
    same_scene_b = thumbs / "pxl_gate.jpg"
    other_scene = thumbs / "other.jpg"
    same_scene_a.write_bytes(_make_jpeg_bytes(640, 426, bg_rgb=(36, 62, 94), shape="burst", seed_offset=0))
    same_scene_b.write_bytes(_make_jpeg_bytes(640, 426, bg_rgb=(36, 62, 94), shape="burst", seed_offset=1))
    other_scene.write_bytes(_make_jpeg_bytes(640, 426, bg_rgb=(120, 48, 36), shape="mountain"))

    cands = [
        CloudMediaCandidateItem(
            cloud_item_id="sony_01", filename="_DSC4512.JPG", media_type="photo",
            creation_time="2026-04-21T09:10:00Z", width=7008, height=4672,
            camera_make_model="SONY ILCE-7CM2", gps_lat=None, gps_lon=None,
            proxy_thumbnail_path=str(same_scene_a),
        ),
        # Unrelated shot taken in between (non-adjacent!)
        CloudMediaCandidateItem(
            cloud_item_id="other_01", filename="_DSC4513.JPG", media_type="photo",
            creation_time="2026-04-21T09:13:00Z", width=7008, height=4672,
            camera_make_model="SONY ILCE-7CM2", gps_lat=None, gps_lon=None,
            proxy_thumbnail_path=str(other_scene),
        ),
        # Pixel phone shot of same landmark 6 minutes later carrying Versailles GPS
        CloudMediaCandidateItem(
            cloud_item_id="pxl_01", filename="PXL_20260421_091600.jpg", media_type="photo",
            creation_time="2026-04-21T09:16:00Z", width=4032, height=2688,
            camera_make_model="Google Pixel 9 Pro", gps_lat=48.8049, gps_lon=2.1204, gps_alt=135.0,
            proxy_thumbnail_path=str(same_scene_b),
        ),
    ]
    select_necessary_cloud_items(cands, target_duration_sec=120.0)
    by_id = {c.cloud_item_id: c for c in cands}

    # Sony wins over Pixel in cross-camera cluster and inherits Pixel's GPS!
    assert by_id["sony_01"].selected_for_download is True
    assert by_id["sony_01"].burst_cluster_size == 2
    assert by_id["pxl_01"].selected_for_download is False and by_id["pxl_01"].skip_reason == "burst_duplicate"
    assert by_id["sony_01"].gps_lat is not None and abs(by_id["sony_01"].gps_lat - 48.8049) < 1e-4
    assert by_id["sony_01"].gps_lon is not None and abs(by_id["sony_01"].gps_lon - 2.1204) < 1e-4
    # Nearby Sony shot within 180s also inherits GPS!
    assert by_id["other_01"].gps_lat is not None and abs(by_id["other_01"].gps_lat - 48.8049) < 1e-4


def test_event_clustered_mmr_orientation_pair_balance_and_headroom(tmp_path: Path) -> None:
    """Verify multi-stop event floor, landscape + vertical diptych pairing, and 1.6x recall headroom."""
    from video_editing_ai.cloud.google_photos import select_necessary_cloud_items
    from video_editing_ai.models.schema import CloudMediaCandidateItem

    thumbs = tmp_path / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    palettes = [
        (24, 52, 86), (110, 62, 42), (42, 88, 62), (96, 44, 88),
        (132, 98, 42), (38, 92, 108), (76, 62, 112), (144, 72, 64),
    ]
    shapes = ["mountain", "arch", "grid"]

    cands: list[CloudMediaCandidateItem] = []
    # 3 distinct event stops separated by >2 hours & >5km:
    # Event 1 (08:00): 4 photos (2 landscape, 2 portrait)
    # Event 2 (11:00): 8 photos (4 landscape, 4 portrait)
    # Event 3 (19:00): 4 photos (2 landscape, 2 portrait)
    stops = [
        ("08", 48.845, 2.366, 4),
        ("11", 48.802, 2.128, 8),
        ("19", 48.874, 2.295, 4),
    ]
    idx_g = 0
    for hr, lat, lon, count in stops:
        for k in range(count):
            is_port = (k % 2 == 1)
            w, h = (3000, 4000) if is_port else (4000, 3000)
            tp = thumbs / f"ev_{hr}_{k}.jpg"
            tp.write_bytes(
                _make_jpeg_bytes(
                    360 if is_port else 480,
                    480 if is_port else 360,
                    bg_rgb=palettes[idx_g % len(palettes)],
                    shape=shapes[(idx_g // 2) % len(shapes)],
                )
            )
            cands.append(
                CloudMediaCandidateItem(
                    cloud_item_id=f"p_{hr}_{k}",
                    filename=f"IMG_{hr}_{k:02d}.JPG",
                    media_type="photo",
                    creation_time=f"2026-04-21T{hr}:{k * 3:02d}:00Z",
                    width=w,
                    height=h,
                    camera_make_model="SONY ILCE-7CM2",
                    gps_lat=lat,
                    gps_lon=lon,
                    proxy_thumbnail_path=str(tp),
                )
            )
            idx_g += 1

    # Constrained budget = 8 across 3 events -> every event must get >= 2 picks including >= 1 landscape!
    select_necessary_cloud_items(cands, target_duration_sec=155.0, max_photos=8)
    selected = [c for c in cands if c.selected_for_download]
    assert len(selected) == 8
    events_found = {c.event_cluster_id for c in cands if c.event_cluster_id}
    assert len(events_found) == 3
    for ev in events_found:
        ev_sel = [c for c in selected if c.event_cluster_id == ev]
        assert len(ev_sel) >= 2
        assert any(c.orientation == "landscape" for c in ev_sel)


def test_paginated_cloud_sheets_and_pass15_vision_rescue_gate(tmp_path: Path) -> None:
    """Verify plan_only=True emits paginated 5x4 sheets (>20 tiles) with ZERO Pass-2 downloads until apply_cloud_pick_overrides."""
    from video_editing_ai.cloud.google_photos import apply_cloud_pick_overrides

    mock_items: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    state = _MockGoogleState(items=mock_items, media_payloads=payloads, page_size_cap=15)
    srv, base_url = _start_mock_server(state)
    try:
        palettes = [(20 + (k * 17) % 180, 40 + (k * 29) % 160, 60 + (k * 37) % 160) for k in range(26)]
        shapes = ["mountain", "arch", "grid"]
        for k in range(26):
            iid = f"pick_{k + 1:02d}"
            jpg = _make_jpeg_bytes(480, 360, bg_rgb=palettes[k], shape=shapes[k % 3])
            day = 1 + (k // 9)
            hr = 9 + (k % 9)
            mock_items.append(
                {
                    "id": iid,
                    "createTime": f"2026-07-0{day}T{hr:02d}:00:00Z",
                    "type": "PHOTO",
                    "mediaFile": {
                        "baseUrl": f"{base_url}/media/{iid}",
                        "mimeType": "image/jpeg",
                        "filename": f"TRIP_{k + 1:02d}.JPG",
                        "mediaFileMetadata": {"width": 4000, "height": 3000, "cameraMake": "Sony", "cameraModel": "A7C2"},
                    },
                }
            )
            payloads[f"{iid}=w512-h512"] = jpg
            payloads[f"{iid}=d"] = jpg

        secrets_p, token_p = _write_mock_credentials(tmp_path, base_url, with_valid_token=True)
        proj = tmp_path / "pass15_proj"
        manifest_plan = sync_google_photos_media(
            project_dir=proj,
            date_from="2026-07-01",
            date_to="2026-07-05",
            max_photos=10,
            plan_only=True,
            open_browser=False,
            client_secrets_path=secrets_p,
            token_path=token_p,
            cache_dir=tmp_path / "gp_cache",
            api_base_url=f"{base_url}/v1",
        )
        # Zero full-res downloads happened during plan_only!
        assert manifest_plan.download_stage == "planned_only"
        assert [r for r in state.download_requests if r[1] in ("d", "dv")] == []
        # Paginated 5x4 sheets cover ALL 26 items (2 pages: 20 + 6)
        assert len(manifest_plan.scan_contact_sheet_paths) == 2
        assert all(Path(p).is_file() for p in manifest_plan.scan_contact_sheet_paths)

        # Rescue 1 skipped runner-up & drop 1 selected item via Pass-1.5 tile IDs
        tile_map = json.loads((proj / "contact_sheets" / "cloud_scan_tile_map.json").read_text(encoding="utf-8"))
        skipped_tile = next(tk for tk, tv in tile_map.items() if not tv["selected"])
        selected_tile = next(tk for tk, tv in tile_map.items() if tv["selected"])
        rescued_cid = tile_map[skipped_tile]["cloud_item_id"]
        dropped_cid = tile_map[selected_tile]["cloud_item_id"]

        final_m = apply_cloud_pick_overrides(
            proj,
            overrides={"select": [skipped_tile], "skip": [selected_tile]},
            execute_pass2=True,
        )
        assert final_m.download_stage == "completed"
        assert final_m.selected_photos_count == 10
        full_dls = {iid for iid, mod in state.download_requests if mod == "d"}
        assert rescued_cid in full_dls
        assert dropped_cid not in full_dls
        assert len(list(Path(final_m.staged_media_dir).iterdir())) == 10
    finally:
        srv.shutdown()
        srv.server_close()


def test_takeout_lazy_zip_skips_unselected_video_extraction(tmp_path: Path) -> None:
    """Verify Takeout ZIP ingest NEVER extracts skipped videos to disk, and rescues lazily on demand."""
    from video_editing_ai.cloud.google_photos import apply_cloud_pick_overrides

    takeout_zip = tmp_path / "takeout_lazy.zip"
    v_bytes = _make_stripped_mp4_bytes(tmp_path, "v_src.mp4", color="navy", duration=2.0)
    p_bytes = _make_jpeg_bytes(480, 360, shape="arch")

    with zipfile.ZipFile(takeout_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        base = "Takeout/Google Photos/Trip/"
        zf.writestr(base + "HERO.JPG", p_bytes)
        zf.writestr(base + "HERO.JPG.json", json.dumps({"photoTakenTime": {"timestamp": "1782986400"}}))
        for i_v, (fname, ts, fav) in enumerate(
            [
                ("VID_FAV.MP4", "1782987000", True),
                ("VID_SKIP_1.MP4", "1782994000", False),
                ("VID_SKIP_2.MP4", "1783001000", False),
            ]
        ):
            zf.writestr(base + fname, v_bytes)
            zf.writestr(
                base + f"{fname}.supplemental-metadata.json",
                json.dumps({"photoTakenTime": {"timestamp": ts}, "favorited": fav}),
            )

    proj = tmp_path / "lazy_takeout_proj"
    manifest = sync_google_takeout_media(
        takeout_source=takeout_zip,
        project_dir=proj,
        date_from="2026-07-01",
        date_to="2026-07-05",
        max_videos=1,
    )
    assert manifest.selected_videos_count == 1
    extracted_dir = proj / "cache" / "takeout_scan" / "extracted"
    extracted_mp4s = [p.name for p in extracted_dir.glob("*.MP4")]
    assert len(extracted_mp4s) == 1 and "VID_FAV.MP4" in extracted_mp4s[0]

    # Rescuing VID_SKIP_1.MP4 via Pass-1.5 lazily extracts ONLY that rescued video
    rescued_m = apply_cloud_pick_overrides(proj, overrides={"select": ["VID_SKIP_1.MP4"]}, execute_pass2=True)
    assert rescued_m.selected_videos_count == 2
    extracted_after = [p.name for p in extracted_dir.glob("*.MP4")]
    assert len(extracted_after) == 2
    assert any("VID_SKIP_1.MP4" in n for n in extracted_after)
    assert not any("VID_SKIP_2.MP4" in n for n in extracted_after)


def test_zero_setup_browser_search_urls_and_multi_zip_ingest(tmp_path: Path) -> None:
    """Verify zero-setup browser deep-links (favorites/videos/album), multi-ZIP day-batch merge & apply-cloud-picks CLI."""
    from video_editing_ai.cloud.google_photos import build_zero_setup_browser_plan

    plan = build_zero_setup_browser_plan("2026-04-21", "2026-04-22")
    assert plan["search_url"] == "https://photos.google.com/search/2026-04-21%20to%202026-04-22"
    assert plan["favorites_search_url"] == "https://photos.google.com/search/2026-04-21%20to%202026-04-22%20favorites"
    assert plan["videos_search_url"] == "https://photos.google.com/search/2026-04-21%20to%202026-04-22%20videos"
    assert plan["max_items_per_zip"] == 450

    album_u = "https://photos.google.com/album/AF1QipTestAlbum"
    assert build_google_photos_web_search_url("2026-04-21", "2026-04-22", album_url=album_u) == album_u

    # Multi-ZIP day batches merged cleanly in plan_only -> apply-cloud-picks CLI
    z1 = tmp_path / "Photos-day1.zip"
    z2 = tmp_path / "Photos-day2.zip"
    with zipfile.ZipFile(z1, "w") as zf:
        zf.writestr("DAY1_SEINE.JPG", _make_exif_jpeg_bytes("2026:04:21 09:00:00", 48.845, 2.366, 35.0, shape="mountain"))
    with zipfile.ZipFile(z2, "w") as zf:
        zf.writestr("DAY2_LOUVRE.JPG", _make_exif_jpeg_bytes("2026:04:22 11:00:00", 48.861, 2.337, 42.0, shape="arch", bg_rgb=(110, 62, 42)))

    repo_root = Path(__file__).resolve().parent.parent
    skill_script = repo_root / ".agent" / "skills" / "ai-video-editor" / "scripts" / "ai_video_studio_skill.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proj = tmp_path / "multi_zip_proj"

    subprocess.run(
        [
            sys.executable, str(skill_script), "sync-google-photos",
            "--project-dir", str(proj),
            "--from-downloads", f"{z1},{z2}",
            "--date-from", "2026-04-21", "--date-to", "2026-04-22",
            "--max-photos", "1", "--plan-only",
        ],
        env=env,
        check=True,
    )
    m_plan = GooglePhotosSyncManifest.from_json_file(proj / "google_photos_manifest.json")
    assert m_plan.download_stage == "planned_only"
    assert m_plan.in_date_range_count == 2
    assert m_plan.event_clusters_count == 2

    # Finalize via apply-cloud-picks rescuing both days
    subprocess.run(
        [
            sys.executable, str(skill_script), "apply-cloud-picks",
            "--project-dir", str(proj),
            "--overrides-json", json.dumps({"select": ["DAY1_SEINE.JPG", "DAY2_LOUVRE.JPG"]}),
        ],
        env=env,
        check=True,
    )
    m_done = GooglePhotosSyncManifest.from_json_file(proj / "google_photos_manifest.json")
    assert m_done.download_stage == "completed"
    assert m_done.selected_photos_count == 2
    assert len(list(Path(m_done.staged_media_dir).iterdir())) == 2

