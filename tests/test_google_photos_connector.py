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
