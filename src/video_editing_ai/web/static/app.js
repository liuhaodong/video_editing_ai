const statusPill = document.getElementById("statusPill");

function setStatus(text) {
  statusPill.textContent = text;
}

// Tab switching
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

function renderCatalogView(catalog) {
  const box = document.getElementById("catalogSummary");
  if (!catalog) {
    box.innerHTML = "<p class='muted'>No catalog loaded yet.</p>";
    return;
  }
  const vids = catalog.videos || [];
  const photos = catalog.photos || [];
  const audios = catalog.audio_files || [];

  let html = `<p><strong>${vids.length} Videos</strong> | <strong>${photos.length} Photos</strong> | <strong>${audios.length} Audio Files</strong></p>`;

  vids.forEach((v) => {
    const dialoguesHtml = (v.dialogues || [])
      .map((d) => `<span class="dialogue-chip">[${d.start_time.toFixed(1)}s-${d.end_time.toFixed(1)}s] “${d.text}”</span>`)
      .join("");
    html += `
      <div class="asset-item">
        <div><strong>🎬 ${v.filename}</strong> (${v.duration.toFixed(1)}s, ${v.width}x${v.height}, ${v.color_transfer})</div>
        <div class="muted" style="font-size:0.83rem;">${v.overall_summary || ""}</div>
        <div>${dialoguesHtml || "<span class='muted' style='font-size:0.78rem;'>No speech segments</span>"}</div>
      </div>
    `;
  });

  if (photos.length > 0) {
    html += `<div class="asset-item"><strong>📸 Photos (${photos.length} total):</strong><div style="margin-top:6px; font-size:0.82rem;" class="muted">`;
    html += photos
      .slice(0, 24)
      .map((p) => `${p.filename} (${p.orientation[0].toUpperCase()}, score ${p.quality_score})`)
      .join(" • ");
    if (photos.length > 24) html += ` ... +${photos.length - 24} more`;
    html += `</div></div>`;
  }

  box.innerHTML = html;
}

function renderTimelineTracks(timeline) {
  const container = document.getElementById("visualTimelineTracks");
  if (!timeline || !timeline.visual_track || timeline.visual_track.length === 0) {
    container.innerHTML = "<p class='muted'>No timeline loaded yet.</p>";
    return;
  }

  const totalDur = Math.max(
    1,
    ...timeline.visual_track.map((c) => c.timeline_start + c.duration)
  );

  const vBlocks = timeline.visual_track
    .map((c) => {
      const left = ((c.timeline_start / totalDur) * 100).toFixed(2);
      const width = Math.max(1.2, (c.duration / totalDur) * 100).toFixed(2);
      const name = c.source.split(/[\\/]/).pop();
      return `<div class="track-block ${c.type}" style="left:${left}%; width:${width}%;" title="${c.id}: ${name} (${c.duration}s) - ${c.narrative_note}">${name}</div>`;
    })
    .join("");

  const tBlocks = (timeline.title_track || [])
    .map((t) => {
      const left = ((t.timeline_start / totalDur) * 100).toFixed(2);
      const width = Math.max(2.0, (t.duration / totalDur) * 100).toFixed(2);
      return `<div class="track-block title" style="left:${left}%; width:${width}%;" title="${t.style}: ${t.text}">${t.text}</div>`;
    })
    .join("");

  const aBlocks = (timeline.audio_tracks || [])
    .map((a) => {
      const dur = a.duration || totalDur;
      const left = (((a.timeline_start || 0) / totalDur) * 100).toFixed(2);
      const width = Math.min(100, (dur / totalDur) * 100).toFixed(2);
      const name = a.source.split(/[\\/]/).pop();
      return `<div class="track-block audio" style="left:${left}%; width:${width}%;" title="${name} (duck=${a.duck_under_dialogue})">🎵 ${name} (Auto-Duck)</div>`;
    })
    .join("");

  container.innerHTML = `
    <div style="font-size:0.82rem; margin-bottom:8px; color:#929db3;">Total Timeline Duration: <strong>${totalDur.toFixed(1)}s</strong> (${timeline.visual_track.length} cuts)</div>
    <div class="track-row"><div class="track-label">V2/V3 Titles</div><div class="track-lane">${tBlocks}</div></div>
    <div class="track-row"><div class="track-label">V1 Storyline</div><div class="track-lane">${vBlocks}</div></div>
    <div class="track-row"><div class="track-label">A2 Music/BGM</div><div class="track-lane">${aBlocks}</div></div>
  `;
}

function renderExportsView(exportsMap) {
  const exBox = document.getElementById("exportsList");
  const vidBox = document.getElementById("videoPlayerContainer");
  if (!exportsMap || Object.keys(exportsMap).length === 0) {
    exBox.innerHTML = "<p class='muted'>No exports generated yet.</p>";
    return;
  }

  const labels = {
    mp4: "🎬 Final Rendered Video (.mp4)",
    fcpxml: "🎞️ DaVinci Resolve / Final Cut Pro XML (.fcpxml)",
    edl: "📋 CMX 3600 Edit Decision List (.edl)",
    otio: "📐 OpenTimelineIO Project (.otio)",
    srt: "💬 SubRip Subtitles (.srt)",
    ass: "✨ Styled SubStation Alpha Subtitles (.ass)",
  };

  exBox.innerHTML = Object.entries(exportsMap)
    .map(([ext, fullPath]) => {
      const url = `/api/file?path=${encodeURIComponent(fullPath)}`;
      return `<a class="export-link" href="${url}" target="_blank" download><span>${labels[ext] || ext.toUpperCase()}</span><span>Download</span></a>`;
    })
    .join("");

  if (exportsMap.mp4) {
    const videoUrl = `/api/file?path=${encodeURIComponent(exportsMap.mp4)}&t=${Date.now()}`;
    vidBox.innerHTML = `<video controls src="${videoUrl}"></video>`;
  }
}

async function refreshProjectState() {
  const projDir = document.getElementById("projectDirInput").value.trim();
  setStatus("Loading state...");
  const res = await fetch(`/api/state?project_dir=${encodeURIComponent(projDir)}`);
  const data = await res.json();
  setStatus(data.last_status || "Ready");

  if (data.annotations_yaml) {
    document.getElementById("annotationsYamlEditor").value = data.annotations_yaml;
  }
  if (data.catalog) {
    renderCatalogView(data.catalog);
  }
  if (data.timeline_yaml) {
    document.getElementById("timelineYamlEditor").value = data.timeline_yaml;
  }
  if (data.timeline) {
    renderTimelineTracks(data.timeline);
  }
  if (data.storyboard_md) {
    document.getElementById("storyboardPreview").textContent = data.storyboard_md;
  }
  if (data.exports) {
    renderExportsView(data.exports);
  }
}

document.getElementById("btnRefreshState").addEventListener("click", refreshProjectState);

document.getElementById("btnAnalyze").addEventListener("click", async () => {
  setStatus("Step 1: Analyzing media & transcribing dialogues...");
  const payload = {
    media_dir: document.getElementById("mediaDirInput").value.trim(),
    project_dir: document.getElementById("projectDirInput").value.trim(),
    whisper_model: document.getElementById("whisperModelSelect").value,
    extra_audio_path: document.getElementById("bgmPathInput").value.trim() || null,
    use_antigravity: document.getElementById("useAgStep1").checked,
    user_hint: document.getElementById("storyPromptInput").value.trim(),
  };
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    setStatus(`Error: ${err.detail}`);
    return;
  }
  const data = await res.json();
  document.getElementById("annotationsYamlEditor").value = data.annotations_yaml;
  renderCatalogView(data.catalog);
  setStatus("Step 1 Complete!");
});

document.getElementById("btnSaveAnnotations").addEventListener("click", async () => {
  const payload = {
    project_dir: document.getElementById("projectDirInput").value.trim(),
    yaml_text: document.getElementById("annotationsYamlEditor").value,
  };
  const res = await fetch("/api/annotations", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.ok) {
    const data = await res.json();
    renderCatalogView(data.catalog);
    setStatus("Saved annotations.yaml");
  }
});

document.getElementById("btnGeneratePlan").addEventListener("click", async () => {
  setStatus("Step 2: Generating AI Story Plan & Timeline...");
  const [w, h] = document.getElementById("resolutionSelect").value.split("x").map(Number);
  const payload = {
    project_dir: document.getElementById("projectDirInput").value.trim(),
    user_prompt: document.getElementById("storyPromptInput").value.trim(),
    width: w,
    height: h,
    fps: 30,
    photo_mode: document.getElementById("photoModeSelect").value,
    use_antigravity: document.getElementById("useAgStep1").checked,
  };
  const res = await fetch("/api/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    setStatus(`Error: ${err.detail}`);
    return;
  }
  const data = await res.json();
  document.getElementById("timelineYamlEditor").value = data.timeline_yaml;
  document.getElementById("storyboardPreview").textContent = data.storyboard_md;
  renderTimelineTracks(data.timeline);
  setStatus("Step 2 Timeline Generated!");
});

document.getElementById("btnSaveTimeline").addEventListener("click", async () => {
  const payload = {
    project_dir: document.getElementById("projectDirInput").value.trim(),
    yaml_text: document.getElementById("timelineYamlEditor").value,
  };
  const res = await fetch("/api/timeline", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.ok) {
    const data = await res.json();
    document.getElementById("storyboardPreview").textContent = data.storyboard_md;
    renderTimelineTracks(data.timeline);
    setStatus("Saved & validated timeline.yaml!");
  } else {
    const err = await res.json();
    setStatus(`Validation Error: ${err.detail}`);
  }
});

document.getElementById("btnRenderVideo").addEventListener("click", async () => {
  setStatus("Step 3: Rendering MP4 with FFmpeg & Pillow...");
  const payload = {
    project_dir: document.getElementById("projectDirInput").value.trim(),
    output_filename: "final_story.mp4",
  };
  const res = await fetch("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    setStatus(`Render Error: ${err.detail}`);
    return;
  }
  await refreshProjectState();
  setStatus("Step 3 Render Complete!");
});

window.addEventListener("DOMContentLoaded", refreshProjectState);
