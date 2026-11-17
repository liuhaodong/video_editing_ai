"""Confirmation-gated YouTube publishing tests (OAuth PKCE, Resumable Upload, Web UI, CLI & Skill)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest
from typer.testing import CliRunner

from video_editing_ai.cli import app as cli_app
from video_editing_ai.cloud.youtube_publisher import (
    YouTubeAuthManager,
    execute_youtube_publish,
    prepare_youtube_publish,
)
from video_editing_ai.models.schema import (
    EditTimeline,
    TimelineMarker,
    TimelineSettings,
    VisualTrackItem,
    YouTubePublishDraft,
    YouTubePublishReceipt,
)
from video_editing_ai.web.app import create_app


def _make_real_mp4(path: Path, duration: float = 3.0, bitrate: str = "1800k", noisy: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    v_args = ["-vf", "noise=alls=48:allf=t"] if noisy else []
    size_str = "1280x720" if noisy else "640x360"
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
            f"testsrc=size={size_str}:rate=15:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration}",
            *v_args,
            "-c:v",
            "libx264",
            "-b:v",
            bitrate,
            "-minrate",
            bitrate,
            "-maxrate",
            bitrate,
            "-bufsize",
            bitrate,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    return path


def _make_photo(path: Path, color: tuple[int, int, int] = (46, 98, 148)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (1600, 1000), color)
    draw = ImageDraw.Draw(im)
    draw.rectangle([220, 180, 1380, 820], fill=(184, 136, 78))
    im.save(path, format="JPEG", quality=92)
    return path


class _MockYouTubeState:
    def __init__(self, fail_once_mid_upload: bool = True) -> None:
        self.fail_once_mid_upload = fail_once_mid_upload
        self.failed_once = False
        self.token_requests: list[dict[str, str]] = []
        self.init_payloads: list[dict[str, Any]] = []
        self.uploaded_video_bytes = bytearray()
        self.total_expected_size = 0
        self.uploaded_thumb_bytes = b""
        self.uploaded_caption_bytes = b""
        self.range_queries = 0
        self.video_id = "dQw4w9WgXcQ"


def _start_mock_youtube_server(state: _MockYouTubeState) -> tuple[ThreadingHTTPServer, str]:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length > 0 else b""

            if parsed.path == "/token":
                form = {k: v[0] for k, v in parse_qs(raw.decode("utf-8")).items()}
                state.token_requests.append(form)
                body = json.dumps(
                    {
                        "access_token": "mock_yt_access_tok",
                        "refresh_token": "mock_yt_refresh_tok",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/upload/youtube/v3/videos":
                if qs.get("uploadType") == ["resumable"]:
                    meta = json.loads(raw.decode("utf-8"))
                    state.init_payloads.append(meta)
                    state.total_expected_size = int(self.headers.get("X-Upload-Content-Length") or "0")
                    state.uploaded_video_bytes = bytearray()
                    self.send_response(200)
                    self.send_header("Location", "/upload_session/yt_sess_01")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

            if parsed.path == "/upload/youtube/v3/thumbnails/set":
                state.uploaded_thumb_bytes = raw
                body = json.dumps({"kind": "youtube#thumbnailSetResponse"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/upload/youtube/v3/captions":
                self.send_response(200)
                self.send_header("Location", "/upload_session/cap_sess_01")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            self.send_response(404)
            self.end_headers()

        def do_PUT(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length > 0 else b""
            content_range = self.headers.get("Content-Range") or ""

            if parsed.path == "/upload_session/cap_sess_01":
                state.uploaded_caption_bytes = raw
                body = json.dumps({"id": "cap_01", "snippet": {"language": "en"}}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path != "/upload_session/yt_sess_01":
                self.send_response(404)
                self.end_headers()
                return

            if content_range.startswith("bytes */"):
                state.range_queries += 1
                committed = len(state.uploaded_video_bytes)
                self.send_response(308)
                if committed > 0:
                    self.send_header("Range", f"bytes=0-{committed - 1}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if state.fail_once_mid_upload and not state.failed_once and len(state.uploaded_video_bytes) > 0:
                state.failed_once = True
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if content_range.startswith("bytes "):
                span = content_range.split(" ", 1)[1].split("/", 1)[0]
                start_b, end_b = (int(x) for x in span.split("-", 1))
                assert start_b == len(state.uploaded_video_bytes), (
                    f"Unexpected chunk offset {start_b}, expected {len(state.uploaded_video_bytes)}"
                )
                assert len(raw) == (end_b - start_b + 1)

            state.uploaded_video_bytes.extend(raw)
            committed = len(state.uploaded_video_bytes)
            if committed < state.total_expected_size:
                self.send_response(308)
                self.send_header("Range", f"bytes=0-{committed - 1}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            privacy = (
                state.init_payloads[-1].get("status", {}).get("privacyStatus", "unlisted")
                if state.init_payloads
                else "unlisted"
            )
            body = json.dumps(
                {"id": state.video_id, "status": {"privacyStatus": privacy}}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _write_mock_oauth_files(cfg_dir: Path, base_url: str, *, expires_in: int = 3600) -> tuple[Path, Path]:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    secrets_p = cfg_dir / "client_secret.json"
    token_p = cfg_dir / "token.json"
    secrets_p.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "mock_yt_client_id",
                    "client_secret": "mock_yt_secret",
                    "auth_uri": f"{base_url}/auth",
                    "token_uri": f"{base_url}/token",
                }
            }
        ),
        encoding="utf-8",
    )
    token_p.write_text(
        json.dumps(
            {
                "access_token": "expired_or_valid_tok",
                "refresh_token": "mock_refresh_tok",
                "expires_at": time.time() + expires_in,
                "scope": "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.force-ssl",
            }
        ),
        encoding="utf-8",
    )
    return secrets_p, token_p


def test_confirmation_gate_blocks_unconfirmed_or_tampered_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOUTUBE_CONFIG_DIR", str(tmp_path / "yt_cfg"))
    proj = tmp_path / "proj"
    vid = _make_real_mp4(proj / "final_story.mp4", duration=1.5)

    draft = prepare_youtube_publish(project_dir=proj, privacy_status="unlisted")
    assert draft.status == "awaiting_confirmation"
    assert draft.privacy_status == "unlisted"
    assert draft.self_declared_made_for_kids is False
    assert (proj / "youtube_publish_draft.json").is_file()
    assert draft.thumbnail_path and Path(draft.thumbnail_path).is_file()

    # 1. Unconfirmed must fail
    with pytest.raises(PermissionError, match="explicit user confirmation"):
        execute_youtube_publish(draft, confirmed=False, confirmation_token=draft.confirmation_token, open_browser=False)

    # 2. Blank or wrong token must fail
    with pytest.raises(PermissionError, match="missing confirmation_token"):
        execute_youtube_publish(draft, confirmed=True, confirmation_token="", open_browser=False)
    with pytest.raises(PermissionError, match="does not match"):
        execute_youtube_publish(draft, confirmed=True, confirmation_token="0" * 64, open_browser=False)

    # 3. Mutating privacy or title after prepare invalidates the bound token
    tampered_priv = draft.model_copy(update={"privacy_status": "public"})
    with pytest.raises(PermissionError, match="does not match"):
        execute_youtube_publish(tampered_priv, confirmed=True, confirmation_token=draft.confirmation_token, open_browser=False)

    tampered_title = draft.model_copy(update={"title": "Sneaky Title Change"})
    with pytest.raises(PermissionError, match="does not match"):
        execute_youtube_publish(tampered_title, confirmed=True, confirmation_token=draft.confirmation_token, open_browser=False)

    # 4. Modifying final_story.mp4 on disk after prepare invalidates confirmation
    with open(vid, "ab") as fh:
        fh.write(b"\x00\x01\x02")
    with pytest.raises(PermissionError, match="Video file changed"):
        execute_youtube_publish(draft, confirmed=True, confirmation_token=draft.confirmation_token, open_browser=False)


def test_youtube_chapters_metadata_and_hdr_safe_thumbnail_under_2mb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOUTUBE_CONFIG_DIR", str(tmp_path / "yt_cfg"))
    proj = tmp_path / "trip_proj"
    hero_jpg = _make_photo(proj / "hero_vista.jpg")
    _make_real_mp4(proj / "final_story.mp4", duration=2.0)

    tl = EditTimeline(
        project_name="Coast <Trip> Adventure",
        user_prompt="Drive up the scenic coast and mountain overlook",
        story_arc_summary="From misty coastal headlands to alpine peaks.",
        settings=TimelineSettings(width=3840, height=2160, fps=30, hdr=True),
        markers=[
            TimelineMarker(time_sec=2.5, name="Coast Departure"),
            TimelineMarker(time_sec=6.0, name="Too Close Micro Marker"),
            TimelineMarker(time_sec=45.0, name="Redwood Ridge <View>"),
            TimelineMarker(time_sec=95.0, name="Summit Finale"),
        ],
        visual_track=[
            VisualTrackItem(
                id="clip_001",
                type="photo",
                source=str(hero_jpg),
                timeline_start=0.0,
                duration=140.0,
            )
        ],
    )
    tl.to_yaml(proj / "timeline.yaml")
    (proj / "travel_map_manifest.json").write_text(
        json.dumps(
            {
                "project_title": "Coast Trip",
                "total_distance_km": 320.5,
                "waypoints": [{"label": "Monterey"}, {"label": "Big Sur"}, {"label": "Carmel"}],
            }
        ),
        encoding="utf-8",
    )
    (proj / "photo_curation.json").write_text(
        json.dumps(
            {
                "project_title": "Coast Trip",
                "items": [
                    {
                        "subclip_id": "p1",
                        "filename": "hero_vista.jpg",
                        "source_path": str(hero_jpg),
                        "orientation": "landscape",
                        "overall_taste_score": 9.6,
                        "is_selected": True,
                        "is_favorite": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    draft = prepare_youtube_publish(project_dir=proj)
    assert "<" not in draft.title and ">" not in draft.title
    assert "<" not in draft.description and ">" not in draft.description
    assert draft.category_id == "19"  # Travel & Events from multi-stop route
    assert len(draft.chapters) == 3
    assert draft.chapters[0].timecode == "00:00"
    assert [c.title for c in draft.chapters] == ["Coast Departure", "Redwood Ridge View", "Summit Finale"]
    assert "00:00 Coast Departure" in draft.description

    thumb = Path(draft.thumbnail_path or "")
    assert thumb.is_file()
    assert thumb.stat().st_size < 2_000_000
    with Image.open(thumb) as im:
        assert im.size == (1280, 720)


def test_oauth_pkce_and_resumable_chunked_upload_with_503_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _MockYouTubeState(fail_once_mid_upload=True)
    srv, base_url = _start_mock_youtube_server(state)
    try:
        cfg_dir = tmp_path / "yt_cfg"
        # Use expires_in=5s to test automatic refresh_token renewal
        secrets_p, token_p = _write_mock_oauth_files(cfg_dir, base_url, expires_in=5)
        monkeypatch.setenv("YOUTUBE_CONFIG_DIR", str(cfg_dir))
        monkeypatch.setenv("YOUTUBE_API_BASE", base_url)

        auth_mgr = YouTubeAuthManager(client_secrets_path=secrets_p, token_path=token_p, include_captions_scope=True)

        # Verify PKCE loopback auth exchange
        def _auto_approve(auth_url: str) -> None:
            qs = parse_qs(urlparse(auth_url).query)
            cb = f"{qs['redirect_uri'][0]}?code=mock_code_123&state={qs['state'][0]}"
            import httpx

            r = httpx.get(cb, timeout=10.0)
            assert r.status_code == 200

        auth_mgr.authenticate_interactive(open_browser=False, auth_url_callback=_auto_approve)
        assert len(state.token_requests) == 1
        assert state.token_requests[0]["grant_type"] == "authorization_code"

        # Now expire access_token to verify refresh_token rotation during execute_youtube_publish
        tok_data = json.loads(token_p.read_text(encoding="utf-8"))
        tok_data["expires_at"] = time.time() + 10.0
        token_p.write_text(json.dumps(tok_data), encoding="utf-8")

        proj = tmp_path / "oauth_proj"
        vid = _make_real_mp4(proj / "final_story.mp4", duration=3.0, bitrate="2200k", noisy=True)
        assert vid.stat().st_size > 400_000, "Need >256KB video to exercise multi-chunk 308 + 503 recovery"
        _make_photo(proj / "cover.jpg")
        (proj / "final_story.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nHello Canyon\n", encoding="utf-8")

        draft = prepare_youtube_publish(
            project_dir=proj,
            title="Desert Drive Test",
            privacy_status="unlisted",
            thumbnail_source=proj / "cover.jpg",
            auth_mode="oauth",
            auth_manager=auth_mgr,
        )
        assert draft.auth_mode_resolved == "oauth"

        progress_events: list[tuple[int, int]] = []
        receipt = execute_youtube_publish(
            draft=draft,
            confirmed=True,
            confirmation_token=draft.confirmation_token,
            auth_manager=auth_mgr,
            api_base=base_url,
            chunk_size=256 * 1024,
            open_browser=False,
            progress_callback=lambda sent, total, _msg: progress_events.append((sent, total)),
        )

        assert receipt.video_id == "dQw4w9WgXcQ"
        assert receipt.watch_url == "https://youtu.be/dQw4w9WgXcQ"
        assert receipt.thumbnail_uploaded is True
        assert receipt.captions_uploaded is True
        assert state.failed_once is True
        assert state.range_queries >= 1
        assert bytes(state.uploaded_video_bytes) == vid.read_bytes()
        assert state.uploaded_thumb_bytes == Path(draft.thumbnail_path or "").read_bytes()
        assert b"Hello Canyon" in state.uploaded_caption_bytes
        assert any(tr.get("grant_type") == "refresh_token" for tr in state.token_requests)
        assert (proj / "youtube_publish_receipt.json").is_file()
        assert YouTubePublishDraft.from_json_file(proj / "youtube_publish_draft.json").status == "published"
        assert progress_events[-1][0] == vid.stat().st_size
    finally:
        srv.shutdown()
        srv.server_close()


def test_web_api_youtube_prepare_and_confirmation_enforced_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _MockYouTubeState(fail_once_mid_upload=False)
    srv, base_url = _start_mock_youtube_server(state)
    try:
        cfg_dir = tmp_path / "yt_cfg"
        _write_mock_oauth_files(cfg_dir, base_url, expires_in=3600)
        monkeypatch.setenv("YOUTUBE_CONFIG_DIR", str(cfg_dir))
        monkeypatch.setenv("YOUTUBE_API_BASE", base_url)

        proj = tmp_path / "web_proj"
        _make_real_mp4(proj / "final_story.mp4", duration=1.5)

        client = TestClient(create_app(default_project_dir=proj))
        st_res = client.get("/api/youtube/status", params={"project_dir": str(proj)})
        assert st_res.status_code == 200
        assert st_res.json()["auth"]["ready"] is True

        prep_res = client.post(
            "/api/youtube/prepare",
            json={"project_dir": str(proj), "title": "Web Studio Test", "privacy_status": "unlisted"},
        )
        assert prep_res.status_code == 200
        draft = prep_res.json()["draft"]
        tok = draft["confirmation_token"]

        # Blocked without confirmation / token
        r_unconf = client.post(
            "/api/youtube/publish",
            json={"project_dir": str(proj), "confirmed": False, "confirmation_token": tok},
        )
        assert r_unconf.status_code == 403

        r_badtok = client.post(
            "/api/youtube/publish",
            json={"project_dir": str(proj), "confirmed": True, "confirmation_token": "wrong"},
        )
        assert r_badtok.status_code == 403

        # Allowed with valid confirmation token
        r_ok = client.post(
            "/api/youtube/publish",
            json={"project_dir": str(proj), "confirmed": True, "confirmation_token": tok, "open_browser": False},
        )
        assert r_ok.status_code == 200
        job_id = r_ok.json()["job_id"]

        deadline = time.time() + 10.0
        job_data: dict[str, Any] = {}
        while time.time() < deadline:
            jr = client.get(f"/api/youtube/job/{job_id}")
            assert jr.status_code == 200
            job_data = jr.json()
            if job_data["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert job_data["status"] == "completed", job_data
        assert job_data["receipt"]["watch_url"] == "https://youtu.be/dQw4w9WgXcQ"
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_and_skill_youtube_prepare_and_publish_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _MockYouTubeState(fail_once_mid_upload=False)
    srv, base_url = _start_mock_youtube_server(state)
    try:
        cfg_dir = tmp_path / "yt_cfg"
        _write_mock_oauth_files(cfg_dir, base_url, expires_in=3600)
        monkeypatch.setenv("YOUTUBE_CONFIG_DIR", str(cfg_dir))
        monkeypatch.setenv("YOUTUBE_API_BASE", base_url)

        proj = tmp_path / "cli_proj"
        _make_real_mp4(proj / "final_story.mp4", duration=1.5)

        runner = CliRunner()
        res_auth = runner.invoke(cli_app, ["youtube-auth", "--status"])
        assert res_auth.exit_code == 0
        assert "YouTube Publishing Connection Status" in res_auth.output

        res_prep = runner.invoke(
            cli_app,
            ["youtube-publish", "--project-dir", str(proj), "--privacy", "private", "--prepare-only"],
        )
        assert res_prep.exit_code == 0
        draft = YouTubePublishDraft.from_json_file(proj / "youtube_publish_draft.json")
        assert draft.privacy_status == "private"

        # Non-TTY subprocess without --confirm-token must refuse with exit code 2
        repo_root = Path(__file__).resolve().parent.parent
        env = dict(os.environ)
        env["YOUTUBE_CONFIG_DIR"] = str(cfg_dir)
        env["YOUTUBE_API_BASE"] = base_url
        env["PYTHONPATH"] = str(repo_root / "src")

        refuse_proc = subprocess.run(
            [sys.executable, "-m", "video_editing_ai.cli", "youtube-publish", "--project-dir", str(proj), "--privacy", "private"],
            cwd=str(repo_root),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        assert refuse_proc.returncode == 2
        assert "requires explicit --confirm-token" in refuse_proc.stdout

        # Confirmed CLI publish succeeds
        res_pub = runner.invoke(
            cli_app,
            [
                "youtube-publish",
                "--project-dir",
                str(proj),
                "--privacy",
                "private",
                "--confirm-token",
                draft.confirmation_token,
                "--no-browser",
            ],
        )
        assert res_pub.exit_code == 0, res_pub.output
        assert "https://youtu.be/dQw4w9WgXcQ" in res_pub.output

        # Skill CLI --prepare and --record-browser-receipt
        skill_script = repo_root / ".agent" / "skills" / "ai-video-editor" / "scripts" / "ai_video_studio_skill.py"
        proj_browser = tmp_path / "skill_browser_proj"
        _make_real_mp4(proj_browser / "final_story.mp4", duration=1.5)

        prep_skill = subprocess.run(
            [
                sys.executable,
                str(skill_script),
                "publish-youtube",
                "--project-dir",
                str(proj_browser),
                "--auth-mode",
                "browser",
                "--prepare",
            ],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        skill_draft = json.loads(prep_skill.stdout)
        assert skill_draft["auth_mode_resolved"] == "browser"

        rec_skill = subprocess.run(
            [
                sys.executable,
                str(skill_script),
                "publish-youtube",
                "--project-dir",
                str(proj_browser),
                "--record-browser-receipt",
                "https://studio.youtube.com/video/AbC123xYz09/edit",
                "--confirmed",
                "--confirm-token",
                skill_draft["confirmation_token"],
            ],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        rec = YouTubePublishReceipt.from_json_file(proj_browser / "youtube_publish_receipt.json")
        assert rec.video_id == "AbC123xYz09"
        assert rec.watch_url == "https://youtu.be/AbC123xYz09"
        assert rec.auth_mode_used == "antigravity_browser"
        assert json.loads(rec_skill.stdout)["video_id"] == "AbC123xYz09"
    finally:
        srv.shutdown()
        srv.server_close()
