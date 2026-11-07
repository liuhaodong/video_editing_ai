const statusPill = document.getElementById("statusPill");
let latestAgentContracts = {};
let registeredAgentsList = [];
let discoveredLibraries = [];
let currentIntakeAnswers = {
  narrative_style: "family_keepsake_4act",
  photo_mode: "parallax_gallery",
  bgm_mood: "auto_match",
  dialogue_preference: "keep_full_sentences_with_broll_cutaways",
  target_duration_sec: "155",
  must_include_topics: "",
};
let intakeChatHistory = [];
let intakeDebounceTimer = null;
let latestDirectorRecommendedSec = 155;
let activePollTimer = null;
let currentBrowserPath = "";
let currentBrowserParent = null;

function formatMmSs(seconds) {
  const sec = Math.max(1, Math.round(Number(seconds) || 90));
  const mins = Math.floor(sec / 60);
  const rem = sec % 60;
  if (mins > 0 && rem > 0) return `${mins}m ${String(rem).padStart(2, "0")}s`;
  if (mins > 0) return `${mins}m 00s`;
  return `${rem}s`;
}

function setStatus(text) {
  if (statusPill) statusPill.textContent = text;
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Tab switching
function switchToTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-${tabId}`);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    switchToTab(btn.dataset.tab);
  });
});

document.getElementById("btnNextToIntake")?.addEventListener("click", () => switchToTab("intake"));
document.getElementById("btnBackToProjects")?.addEventListener("click", () => switchToTab("projects"));
document.getElementById("btnGoToTheaterTab")?.addEventListener("click", () => switchToTab("deliverables"));
document.getElementById("btnTopNewProject")?.addEventListener("click", () => {
  switchToTab("projects");
  const input = document.getElementById("newProjectNameInput");
  if (input) input.focus();
});
document.getElementById("btnStartFromIntake")?.addEventListener("click", () => {
  switchToTab("run");
  document.getElementById("btnRunMultiAgent")?.click();
});

// ---------------------------------------------------------------------------
// PROJECT MANAGEMENT (CREATE, LOAD, CLEAR / RESET, LIST)
// ---------------------------------------------------------------------------
let workspaceProjectsList = [];

function renderProjectsList(projects, activeProjectDir = null) {
  workspaceProjectsList = projects || [];
  const currentProj = (
    activeProjectDir ||
    document.getElementById("projectDirInput")?.value ||
    ""
  )
    .trim()
    .toLowerCase();

  // 1. Update Topbar Project Selector
  const selector = document.getElementById("topProjectSelector");
  if (selector) {
    if (workspaceProjectsList.length === 0) {
      selector.innerHTML = `<option value="">No projects yet</option>`;
    } else {
      selector.innerHTML = workspaceProjectsList
        .map((p) => {
          const isSelected = p.project_dir.toLowerCase() === currentProj;
          const badge = p.has_rendered_video
            ? `✓ MP4 Ready${p.quality_grade ? ` (${p.quality_grade})` : ""}`
            : p.has_timeline
            ? "📋 Plan Ready"
            : "🆕 Empty";
          return `<option value="${escapeHtml(p.project_dir)}" ${isSelected ? "selected" : ""}>${escapeHtml(
            p.project_name
          )} — ${badge}</option>`;
        })
        .join("");
    }
  }

  // 2. Update Tab 1 Existing Projects Grid
  const listEl = document.getElementById("existingProjectsList");
  if (!listEl) return;
  if (workspaceProjectsList.length === 0) {
    listEl.innerHTML = `<div class="muted">No projects found in workspace. Create one above!</div>`;
    return;
  }

  listEl.innerHTML = workspaceProjectsList
    .map((p, idx) => {
      const isActive = p.project_dir.toLowerCase() === currentProj;
      let statusBadge = `<span class="stage-badge-mini">🆕 Empty Workspace</span>`;
      if (p.has_rendered_video) {
        statusBadge = `<span class="stage-badge-mini" style="color:#34d399; border-color:rgba(16,185,129,0.4);">🎬 MP4 Ready (${
          p.rendered_video_size_mb || 0
        } MB)${p.quality_grade ? ` • Grade ${escapeHtml(p.quality_grade)}` : ""}</span>`;
      } else if (p.has_timeline) {
        statusBadge = `<span class="stage-badge-mini" style="color:var(--accent);">📋 Timeline Ready</span>`;
      }

      const dlBtnHtml = p.has_rendered_video
        ? `<a class="btn small success-btn" href="/api/projects/download-video?project_dir=${encodeURIComponent(
            p.project_dir
          )}&t=${p.rendered_video_mtime || Date.now()}" download style="text-decoration:none;">⬇️ MP4</a>`
        : "";

      return `
        <div class="project-card ${isActive ? "active" : ""}" data-proj-idx="${idx}">
          <div class="project-card-info">
            <div class="project-card-title">
              <span>🗂️ ${escapeHtml(p.project_name)}</span>
              ${isActive ? `<span class="stage-badge-mini" style="background:rgba(0,229,255,0.16); color:var(--accent-cyan);">ACTIVE</span>` : ""}
              ${statusBadge}
            </div>
            <div class="project-card-meta">
              Updated: ${escapeHtml(p.updated_at || "Just now")}
              ${p.media_dir ? ` &bull; Media: <code>${escapeHtml(p.media_dir)}</code>` : ""}
            </div>
            <div class="library-card-path" title="${escapeHtml(p.project_dir)}">${escapeHtml(p.project_dir)}</div>
          </div>
          <div class="project-card-actions">
            <button type="button" class="btn small ${isActive ? "secondary" : "primary"} btn-load-proj" data-dir="${escapeHtml(
        p.project_dir
      )}">
              ${isActive ? "✓ Loaded" : "📂 Load"}
            </button>
            ${dlBtnHtml}
            <button type="button" class="btn small danger-outline btn-reset-proj" data-dir="${escapeHtml(
              p.project_dir
            )}" title="Clear all generated files in this project">
              🧹 Reset
            </button>
          </div>
        </div>
      `;
    })
    .join("");

  listEl.querySelectorAll(".btn-load-proj").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const dir = btn.dataset.dir;
      if (dir) await loadProject(dir);
    });
  });

  listEl.querySelectorAll(".btn-reset-proj").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const dir = btn.dataset.dir;
      if (dir) await resetProject(dir);
    });
  });

  listEl.querySelectorAll(".project-card").forEach((card) => {
    card.addEventListener("click", async () => {
      const idx = Number(card.dataset.projIdx);
      const p = workspaceProjectsList[idx];
      if (p) await loadProject(p.project_dir);
    });
  });
}

async function createNewProject() {
  const nameInput = document.getElementById("newProjectNameInput");
  const mediaInput = document.getElementById("mediaDirInput");
  const rawName = (nameInput?.value || "").trim() || `New Story Project ${new Date().toLocaleTimeString()}`;
  const mediaDir = (mediaInput?.value || "").trim();

  setStatus(`Creating new project "${rawName}"...`);
  const res = await fetch("/api/projects/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_name: rawName, media_dir: mediaDir }),
  });
  if (!res.ok) {
    const err = await res.json();
    setStatus(`Error creating project: ${err.detail}`);
    return;
  }
  const data = await res.json();
  if (nameInput) nameInput.value = "";
  const projInput = document.getElementById("projectDirInput");
  if (projInput) projInput.value = data.project_dir;

  clearProjectUiPanels();
  if (data.projects) renderProjectsList(data.projects, data.project_dir);
  await refreshProjectState();
  setStatus(`Created & loaded new project: ${data.project?.project_name || rawName}`);
}

async function loadProject(projectDir) {
  if (!projectDir) return;
  setStatus(`Loading project ${projectDir}...`);
  const res = await fetch("/api/projects/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_dir: projectDir }),
  });
  if (!res.ok) {
    const err = await res.json();
    setStatus(`Error loading project: ${err.detail}`);
    return;
  }
  const data = await res.json();
  const projInput = document.getElementById("projectDirInput");
  if (projInput) projInput.value = data.project_dir;

  clearProjectUiPanels();
  if (data.projects) renderProjectsList(data.projects, data.project_dir);

  if (data.project && data.project.media_dir) {
    await selectMediaLibrary(data.project.media_dir, null, false);
  } else {
    await refreshProjectState();
  }
  setStatus(`Loaded project: ${data.project?.project_name || projectDir}`);
}

function clearProjectUiPanels() {
  const doingEl = document.getElementById("liveAgentDoingStream");
  const doingCountEl = document.getElementById("agentDoingCount");
  if (doingEl) {
    doingEl.innerHTML = `<div class="muted">Click "Start Full AI Workflow" to watch each sub-agent's live actions, calculations, and blackboard handoffs...</div>`;
  }
  if (doingCountEl) doingCountEl.textContent = "0 actions";

  const thinkingEl = document.getElementById("liveDirectorThinkingStream");
  const thinkingCountEl = document.getElementById("directorThinkingCount");
  if (thinkingEl) {
    thinkingEl.innerHTML = `<div class="muted">The Executive Movie Production Director's stage-by-stage observations, aesthetic thoughts, and corrective directives will appear here live...</div>`;
  }
  if (thinkingCountEl) thinkingCountEl.textContent = "0 reviews";

  const logEl = document.getElementById("liveActivityLog");
  const logCountEl = document.getElementById("activityLogCount");
  if (logEl) logEl.innerHTML = `<div class="muted">Pipeline telemetry will stream here in real time as each agent runs...</div>`;
  if (logCountEl) logCountEl.textContent = "0 events";

  const annEl = document.getElementById("annotationsYamlEditor");
  if (annEl) annEl.value = "";
  const tlEl = document.getElementById("timelineYamlEditor");
  if (tlEl) tlEl.value = "";
  const sbEl = document.getElementById("storyboardPreview");
  if (sbEl) sbEl.textContent = "No storyboard generated yet.";

  renderDefaultAgentStagesFromRegistry(registeredAgentsList);
  updateVideoDownloadButtons({}, null);
}

async function resetProject(projectDir = null) {
  const targetDir = (projectDir || document.getElementById("projectDirInput")?.value || "").trim();
  if (!targetDir) return;

  const confirmed = window.confirm(
    `Clear and reset project workspace:\n${targetDir}\n\nThis removes generated timelines, JSON contracts, and rendered MP4s in this project folder so you can start fresh. Your source media files will NOT be touched.`
  );
  if (!confirmed) return;

  setStatus(`Resetting project ${targetDir}...`);
  const res = await fetch("/api/projects/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_dir: targetDir }),
  });
  if (!res.ok) {
    const err = await res.json();
    setStatus(`Error resetting project: ${err.detail}`);
    return;
  }
  const data = await res.json();
  clearProjectUiPanels();
  if (data.projects) renderProjectsList(data.projects, targetDir);
  await refreshProjectState();
  setStatus(`Project reset & cleared: ${data.project?.project_name || targetDir}`);
}

document.getElementById("btnCreateProject")?.addEventListener("click", createNewProject);
document.getElementById("newProjectNameInput")?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    createNewProject();
  }
});
document.getElementById("topProjectSelector")?.addEventListener("change", async (e) => {
  const dir = e.target.value;
  if (dir) await loadProject(dir);
});
document.getElementById("btnTopResetProject")?.addEventListener("click", () => resetProject());
document.getElementById("btnResetProjectCard")?.addEventListener("click", () => resetProject());
document.getElementById("btnRefreshProjects")?.addEventListener("click", async () => {
  const res = await fetch("/api/projects");
  if (res.ok) {
    const data = await res.json();
    renderProjectsList(data.projects || []);
  }
});

// ---------------------------------------------------------------------------
// STEP 1: MEDIA LIBRARY DISCOVERY & FOLDER SELECTION
// ---------------------------------------------------------------------------
function renderDiscoveredLibraries(libs) {
  discoveredLibraries = libs || [];
  const grid = document.getElementById("discoveredLibrariesGrid");
  if (!grid) return;

  if (!discoveredLibraries.length) {
    grid.innerHTML = `<div class="muted">No pre-populated media libraries found. Click "📂 Browse Folders" or "🪟 OS Folder Picker" below.</div>`;
    return;
  }

  const currentMediaDir = (document.getElementById("mediaDirInput")?.value || "").trim().toLowerCase();

  grid.innerHTML = discoveredLibraries
    .map((lib, idx) => {
      const isSelected = lib.media_dir.toLowerCase() === currentMediaDir;
      const cams = (lib.detected_cameras || []).join(" • ");
      return `
        <div class="library-card ${isSelected ? "selected" : ""}" data-lib-idx="${idx}">
          <div class="library-card-title">
            <span>📁 ${escapeHtml(lib.display_title)}</span>
            <span class="stage-badge-mini">${lib.total_size_mb} MB</span>
          </div>
          <div class="library-card-counts">
            🎬 ${lib.video_count} Videos &bull; 📸 ${lib.photo_count} Photos
          </div>
          <div class="muted" style="font-size:0.76rem;">📷 ${escapeHtml(cams)}</div>
          <div class="library-card-path" title="${escapeHtml(lib.media_dir)}">${escapeHtml(lib.media_dir)}</div>
        </div>
      `;
    })
    .join("");

  grid.querySelectorAll(".library-card").forEach((card) => {
    card.addEventListener("click", async () => {
      const idx = Number(card.dataset.libIdx);
      const lib = discoveredLibraries[idx];
      if (!lib) return;
      await selectMediaLibrary(lib.media_dir, lib.suggested_project_dir, true);
    });
  });
}

function updateLibraryBanner(lib) {
  if (!lib) return;
  const titleEl = document.getElementById("libTitleText");
  const countsEl = document.getElementById("libCountsText");
  const camsEl = document.getElementById("libCamerasText");
  const sizeEl = document.getElementById("libSizeText");

  if (titleEl) titleEl.textContent = lib.display_title || "Selected Media Library";
  if (countsEl) {
    countsEl.textContent = `🎬 ${lib.video_count || 0} Videos • 📸 ${lib.photo_count || 0} Photos${
      lib.audio_count ? ` • 🎵 ${lib.audio_count} Audio` : ""
    }`;
  }
  if (camsEl) camsEl.textContent = (lib.detected_cameras || ["Multi-Cam"]).join(" • ");
  if (sizeEl) sizeEl.textContent = `${lib.total_size_mb || 0} MB`;
}

async function selectMediaLibrary(mediaDir, suggestedProjectDir = null, updatePrompt = true) {
  const mediaInput = document.getElementById("mediaDirInput");
  const projInput = document.getElementById("projectDirInput");
  if (mediaInput) mediaInput.value = mediaDir;

  setStatus("Scanning media library & preparing interactive intake...");
  const res = await fetch(`/api/intake-questions?media_dir=${encodeURIComponent(mediaDir)}`);
  if (res.ok) {
    const data = await res.json();
    const lib = data.library;
    if (lib) {
      updateLibraryBanner(lib);
      if (projInput && !projInput.value.trim() && (suggestedProjectDir || lib.suggested_project_dir)) {
        projInput.value = suggestedProjectDir || lib.suggested_project_dir;
      }
      if (updatePrompt && lib.default_prompt) {
        const promptEl = document.getElementById("agentPromptInput");
        if (promptEl) promptEl.value = lib.default_prompt;
        currentIntakeAnswers.must_include_topics = lib.default_prompt;
      }
    }
    if (data.questions) {
      renderInteractiveIntakeQuestions(data.questions);
    }
    if (data.initial_refinement) {
      intakeChatHistory = [];
      applyIntakeRefinementResult(data.initial_refinement, null);
    }
  }

  renderDiscoveredLibraries(discoveredLibraries);
  await refreshProjectState();
}

// ---------------------------------------------------------------------------
// STEP 2: LIVE INTERACTIVE CLIENT INTAKE & DIRECTOR CONSULTATION
// ---------------------------------------------------------------------------
function updateDurationControlsUI(targetSec, recInfo = null) {
  const sec = Math.max(30, Math.min(600, Math.round(Number(targetSec) || 155)));
  currentIntakeAnswers.target_duration_sec = String(sec);

  const sliderEl = document.getElementById("customDurationSlider");
  const labelEl = document.getElementById("customDurationLabel");
  const hiddenInput = document.getElementById("intakeTargetDuration");
  if (sliderEl) sliderEl.value = String(sec);
  if (labelEl) labelEl.textContent = `${sec}s (${formatMmSs(sec)})`;
  if (hiddenInput) hiddenInput.value = String(sec);

  if (recInfo) {
    latestDirectorRecommendedSec = recInfo.director_recommended_duration_sec || sec;
    const titleEl = document.getElementById("directorRecDurationTitle");
    const ratEl = document.getElementById("directorRecRationaleText");
    if (titleEl) {
      titleEl.textContent = `Director Recommends: ${recInfo.director_recommended_duration_sec}s (${recInfo.director_recommended_mmss}) — ~${recInfo.recommended_photo_plates} Curated Gallery Plates`;
    }
    if (ratEl) {
      ratEl.textContent = recInfo.director_duration_rationale || "";
    }
  }

  // Highlight matching duration radio pill if present
  const container = document.getElementById("interactiveIntakeContainer");
  if (container) {
    container.querySelectorAll(`.intake-option-pill[data-qid="target_duration_sec"]`).forEach((p) => {
      const isMatch = Number(p.dataset.val) === sec;
      p.classList.toggle("active", isMatch);
      const r = p.querySelector("input[type='radio']");
      if (r) r.checked = isMatch;
    });
  }
}

function applyIntakeRefinementResult(refData, userMsgText = null) {
  if (!refData) return;

  if (refData.updated_answers) {
    Object.assign(currentIntakeAnswers, refData.updated_answers);
  }
  const durRec = refData.duration_recommendation;
  const draft = refData.draft_brief || {};
  const chosenDur = draft.target_duration_sec || currentIntakeAnswers.target_duration_sec || 155;
  updateDurationControlsUI(chosenDur, durRec);

  // Sync underlying selects & radio pills
  ["narrative_style", "photo_mode", "bgm_mood", "dialogue_preference"].forEach((qId) => {
    const val = (currentIntakeAnswers[qId] || "").split(" ")[0].trim();
    if (!val) return;
    const container = document.getElementById("interactiveIntakeContainer");
    if (container) {
      container.querySelectorAll(`.intake-option-pill[data-qid="${qId}"]`).forEach((p) => {
        const isMatch = p.dataset.val === val;
        p.classList.toggle("active", isMatch);
        const r = p.querySelector("input[type='radio']");
        if (r) r.checked = isMatch;
      });
    }
  });

  // Update conversation feed
  if (userMsgText) {
    intakeChatHistory.push({ role: "user", text: userMsgText });
  }
  if (refData.agent_reply) {
    intakeChatHistory.push({ role: "intake", text: refData.agent_reply });
  }
  if (refData.director_note) {
    intakeChatHistory.push({ role: "director", text: refData.director_note });
  }
  if (intakeChatHistory.length > 9) {
    intakeChatHistory = intakeChatHistory.slice(-9);
  }

  const chatBox = document.getElementById("intakeConversationBox");
  if (chatBox) {
    chatBox.innerHTML = intakeChatHistory
      .map((item) => {
        const cleanHtml = escapeHtml(item.text)
          .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
          .replace(/`(.*?)`/g, "<code>$1</code>");
        return `<div class="chat-bubble ${escapeHtml(item.role)}">${cleanHtml}</div>`;
      })
      .join("");
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // Render Live Draft CreativeBrief Card
  const briefCard = document.getElementById("liveDraftBriefCard");
  if (briefCard && draft.project_title) {
    const topicsHtml = (draft.must_include_topics || [])
      .slice(0, 6)
      .map((t) => `<span class="brief-tag">${escapeHtml(t)}</span>`)
      .join("");
    briefCard.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
        <strong style="color:var(--accent-cyan);">📄 Live Draft <code>creative_brief.json</code></strong>
        <span class="stage-badge-mini">⏱️ ${Math.round(draft.target_duration_sec)}s (${formatMmSs(draft.target_duration_sec)}) • ~${refData.estimated_photo_plates || 22} Photos</span>
      </div>
      <div style="font-size:0.76rem; color:#cbd5e1; margin-bottom:5px;">
        <strong>Arc:</strong> <code>${escapeHtml(draft.narrative_style)}</code> &bull;
        <strong>Photos:</strong> <code>${escapeHtml(draft.photo_mode)}</code> &bull;
        <strong>Audio:</strong> <code>${escapeHtml(currentIntakeAnswers.bgm_mood || "auto_match")}</code> (-14dB Ducking)
      </div>
      <div>${topicsHtml}</div>
    `;
  }

  // Render 1-Click Suggested Refinement Chips
  const chipsBox = document.getElementById("intakeSuggestedChips");
  if (chipsBox && Array.isArray(refData.suggested_chips)) {
    chipsBox.innerHTML = refData.suggested_chips
      .map((chip) => `<button type="button" class="intake-chip-btn" data-chip="${escapeHtml(chip)}">${escapeHtml(chip)}</button>`)
      .join("");
    chipsBox.querySelectorAll(".intake-chip-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const chipText = btn.dataset.chip || "";
        const promptEl = document.getElementById("agentPromptInput");
        if (promptEl && !promptEl.value.includes(chipText)) {
          promptEl.value = `${promptEl.value.trim()} — ${chipText}`;
        }
        await runLiveIntakeRefinement(chipText);
      });
    });
  }
}

async function runLiveIntakeRefinement(customUserAction = null) {
  const mediaDir = document.getElementById("mediaDirInput")?.value.trim();
  const promptEl = document.getElementById("agentPromptInput");
  if (!mediaDir || !promptEl) return;

  const pillEl = document.getElementById("intakeAgentStatusPill");
  if (pillEl) {
    pillEl.textContent = "⏳ Intake Agent & Director Refining...";
    pillEl.classList.add("busy");
  }

  const msgToSend = customUserAction || promptEl.value.trim();
  currentIntakeAnswers.must_include_topics = promptEl.value.trim();

  try {
    const res = await fetch("/api/intake/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        media_dir: mediaDir,
        user_message: msgToSend,
        interview_answers: currentIntakeAnswers,
        chat_history: intakeChatHistory,
      }),
    });
    if (res.ok) {
      const refData = await res.json();
      applyIntakeRefinementResult(refData, customUserAction || "Updated story vision & intake preferences");
    }
  } catch (err) {
    console.warn("Live intake refine error:", err);
  } finally {
    if (pillEl) {
      pillEl.textContent = "🟢 Intake Agent & Director Synchronized";
      pillEl.classList.remove("busy");
    }
  }
}
function renderInteractiveIntakeQuestions(questions) {
  const container = document.getElementById("interactiveIntakeContainer");
  if (!container) return;

  container.innerHTML = (questions || [])
    .map((q) => {
      const qId = q.id;
      const defaultCode = (q.default_answer || "").split(" ")[0].trim();
      if (!currentIntakeAnswers[qId]) {
        currentIntakeAnswers[qId] = qId === "must_include_topics" ? q.default_answer : defaultCode;
      }
      const selectedVal = currentIntakeAnswers[qId];
      const isFullSpan = qId === "must_include_topics" || qId === "photo_mode";

      const optionsHtml = (q.options || [])
        .map((optStr, optIdx) => {
          const optCode = qId === "must_include_topics" ? optStr : optStr.split(" ")[0].trim();
          const isChecked =
            qId === "must_include_topics"
              ? optIdx === 0
              : selectedVal === optCode || (optIdx === 0 && !selectedVal);
          return `
            <label class="intake-option-pill ${isChecked ? "active" : ""}" data-qid="${escapeHtml(qId)}" data-val="${escapeHtml(optCode)}">
              <input type="radio" name="intake_${escapeHtml(qId)}" value="${escapeHtml(optCode)}" ${isChecked ? "checked" : ""} />
              <span>${escapeHtml(optStr)}</span>
            </label>
          `;
        })
        .join("");

      return `
        <div class="intake-q-card ${isFullSpan ? "full-span" : ""}">
          <div class="intake-q-title">${escapeHtml(q.question)}</div>
          <div class="intake-options-list">${optionsHtml}</div>
        </div>
      `;
    })
    .join("");

  container.querySelectorAll(".intake-option-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      const qId = pill.dataset.qid;
      const val = pill.dataset.val;
      currentIntakeAnswers[qId] = val;

      // Update active styling within this question group
      container.querySelectorAll(`.intake-option-pill[data-qid="${qId}"]`).forEach((p) => {
        p.classList.remove("active");
      });
      pill.classList.add("active");
      const radio = pill.querySelector("input[type='radio']");
      if (radio) radio.checked = true;

      // Sync with underlying selects / prompt box
      if (qId === "narrative_style") {
        const el = document.getElementById("intakeNarrativeStyle");
        if (el) el.value = val;
      } else if (qId === "photo_mode") {
        const el = document.getElementById("intakePhotoMode");
        if (el) el.value = val;
      } else if (qId === "dialogue_preference") {
        const el = document.getElementById("intakeDialoguePref");
        if (el) el.value = val;
      } else if (qId === "target_duration_sec") {
        updateDurationControlsUI(val);
        runLiveIntakeRefinement(`Set target duration to ${val}s`);
      } else if (qId === "must_include_topics") {
        const promptEl = document.getElementById("agentPromptInput");
        if (promptEl) promptEl.value = val;
        runLiveIntakeRefinement(`Prioritize highlights: ${val}`);
      }
      if (qId !== "target_duration_sec" && qId !== "must_include_topics") {
        runLiveIntakeRefinement(`Selected ${qId} = ${val}`);
      }
    });
  });
}

// Live listeners for Step 2 interactive prompt box, refine button, and duration slider
document.getElementById("agentPromptInput")?.addEventListener("input", () => {
  const pillEl = document.getElementById("intakeAgentStatusPill");
  if (pillEl) {
    pillEl.textContent = "✍️ Listening... Intake Agent preparing refinement";
    pillEl.classList.add("busy");
  }
  if (intakeDebounceTimer) clearTimeout(intakeDebounceTimer);
  intakeDebounceTimer = setTimeout(() => {
    runLiveIntakeRefinement();
  }, 650);
});

document.getElementById("btnRefineIntakeNow")?.addEventListener("click", () => {
  if (intakeDebounceTimer) clearTimeout(intakeDebounceTimer);
  runLiveIntakeRefinement();
});

document.getElementById("customDurationSlider")?.addEventListener("input", (e) => {
  updateDurationControlsUI(e.target.value);
});

document.getElementById("customDurationSlider")?.addEventListener("change", (e) => {
  const sec = Number(e.target.value);
  updateDurationControlsUI(sec);
  runLiveIntakeRefinement(`Set custom target duration to ${sec} seconds (${formatMmSs(sec)})`);
});

document.getElementById("btnApplyDirectorDuration")?.addEventListener("click", () => {
  updateDurationControlsUI(latestDirectorRecommendedSec);
  runLiveIntakeRefinement(
    `Use Director's Recommended Length of ${latestDirectorRecommendedSec} seconds (${formatMmSs(latestDirectorRecommendedSec)})`
  );
});

// ---------------------------------------------------------------------------
// STEP 3: LIVE PIPELINE PROGRESS & 12-STAGE AGENT STEPPER
// ---------------------------------------------------------------------------
function formatElapsed(sec) {
  const s = Math.max(0, Math.floor(sec || 0));
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `⏱️ ${mm}:${ss}`;
}

function renderJobProgress(job) {
  if (!job) return;

  const badgeEl = document.getElementById("liveStageBadge");
  const titleEl = document.getElementById("liveStageTitle");
  const elapsedEl = document.getElementById("liveElapsedText");
  const pctEl = document.getElementById("livePctText");
  const fillEl = document.getElementById("liveProgressBarFill");
  const subEl = document.getElementById("liveStatusSubtext");
  const gridEl = document.getElementById("liveAgentStagesGrid");
  const logEl = document.getElementById("liveActivityLog");
  const logCountEl = document.getElementById("activityLogCount");
  const runBtn = document.getElementById("btnRunMultiAgent");

  const st = job.status || "idle";
  const pct = job.progress_pct || 0;

  if (badgeEl) {
    badgeEl.textContent = st.toUpperCase();
    badgeEl.className = `progress-stage-pill ${st}`;
  }
  if (titleEl) {
    titleEl.textContent = job.current_stage_title || "Ready";
  }
  if (elapsedEl) {
    elapsedEl.textContent = formatElapsed(job.elapsed_sec || 0);
  }
  if (pctEl) {
    pctEl.textContent = `${pct}%`;
  }
  if (fillEl) {
    fillEl.style.width = `${pct}%`;
  }
  if (subEl) {
    subEl.textContent = job.last_status || "";
  }
  if (runBtn) {
    if (st === "running") {
      runBtn.disabled = true;
      runBtn.textContent = `⏳ Running Multi-Agent Workflow (${pct}%)...`;
    } else {
      runBtn.disabled = false;
      runBtn.textContent = "🚀 Start Full AI Workflow & Render Video";
    }
  }

  if (gridEl && Array.isArray(job.stages) && job.stages.length > 0) {
    gridEl.innerHTML = job.stages
      .map((s) => {
        const sStatus = s.status || "pending";
        let badgeLabel = "WAITING";
        if (sStatus === "running") badgeLabel = "RUNNING...";
        else if (sStatus === "completed") {
          badgeLabel = s.quality_score ? `✓ ${s.quality_score.toFixed(1)}/10` : "✓ DONE";
        } else if (sStatus === "polished_by_director" || sStatus === "directed_improvement") {
          badgeLabel = `🎬 DIRECTED (${s.quality_score ? s.quality_score.toFixed(1) : "9.5"})`;
        } else if (sStatus === "error") {
          badgeLabel = "ERROR";
        }
        const durText = s.duration_sec != null ? ` (${s.duration_sec}s)` : "";
        return `
          <div class="stage-step-card ${escapeHtml(sStatus)}">
            <div class="stage-step-top">
              <span>${escapeHtml(s.display_name)}${durText}</span>
              <span class="stage-badge-mini">${escapeHtml(badgeLabel)}</span>
            </div>
            <div class="stage-step-desc" title="${escapeHtml(s.critique || "")}">
              ${escapeHtml(s.critique || "")}
            </div>
          </div>
        `;
      })
      .join("");
  }

  // Render Left Stream: What the Agents Are Doing
  const doingEl = document.getElementById("liveAgentDoingStream");
  const doingCountEl = document.getElementById("agentDoingCount");
  if (doingEl && Array.isArray(job.agent_activities) && job.agent_activities.length > 0) {
    if (doingCountEl) doingCountEl.textContent = `${job.agent_activities.length} actions`;
    doingEl.innerHTML = job.agent_activities
      .map(
        (act) => `
        <div class="agent-action-card ${escapeHtml(act.event_type || "agent_action")}">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="color:var(--accent-cyan);">[${escapeHtml(act.role_id)}]</strong>
            <span class="muted" style="font-size:0.7rem;">${escapeHtml(act.time_str || "")}</span>
          </div>
          <div style="margin-top:2px; font-weight:600;">${escapeHtml(act.action)}</div>
          ${act.detail ? `<div class="muted" style="font-size:0.73rem; margin-top:2px;">${escapeHtml(act.detail)}</div>` : ""}
        </div>`
      )
      .join("");
    doingEl.scrollTop = doingEl.scrollHeight;
  }

  // Render Right Stream: What the Director Is Thinking
  const thinkingEl = document.getElementById("liveDirectorThinkingStream");
  const thinkingCountEl = document.getElementById("directorThinkingCount");
  if (thinkingEl && Array.isArray(job.director_thoughts) && job.director_thoughts.length > 0) {
    if (thinkingCountEl) thinkingCountEl.textContent = `${job.director_thoughts.length} stage reviews`;
    thinkingEl.innerHTML = job.director_thoughts
      .map(
        (th) => `
        <div class="director-thought-card ${escapeHtml(th.status || "approved")}">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong>🎬 Reviewing <code>${escapeHtml(th.reviewed_role)}</code></strong>
            <span class="stage-badge-mini">${escapeHtml(String(th.status || "approved").toUpperCase())} • ${Number(th.quality_score || 9.5).toFixed(1)}/10</span>
          </div>
          ${th.observation ? `<div class="thought-row"><span class="thought-label">👁️ Observation:</span> ${escapeHtml(th.observation)}</div>` : ""}
          ${th.thinking ? `<div class="thought-row"><span class="thought-label">🧠 Thinking:</span> ${escapeHtml(th.thinking)}</div>` : ""}
          <div class="thought-row"><span class="thought-label">⚖️ Verdict:</span> ${escapeHtml(th.decision || th.critique || "")}</div>
        </div>`
      )
      .join("");
    thinkingEl.scrollTop = thinkingEl.scrollHeight;
  }

  if (logEl && Array.isArray(job.activity_log)) {
    if (logCountEl) logCountEl.textContent = `${job.activity_log.length} events`;
    if (job.activity_log.length > 0) {
      logEl.innerHTML = job.activity_log
        .map(
          (entry) => `
          <div class="log-line ${escapeHtml(entry.level || "info")}">
            <span class="log-time">[${escapeHtml(entry.time_str)}]</span>
            <span class="log-role">[${escapeHtml(entry.role_id)}]</span>
            <span>${escapeHtml(entry.message)}</span>
          </div>`
        )
        .join("");
      logEl.scrollTop = logEl.scrollHeight;
    }
  }
}

function renderDefaultAgentStagesFromRegistry(agents) {
  const gridEl = document.getElementById("liveAgentStagesGrid");
  if (!gridEl || !agents || !agents.length) return;
  if (gridEl.children.length > 0) return;

  const stages = [
    ...agents.map((a) => ({
      display_name: a.display_name,
      status: "pending",
      critique: a.description,
    })),
    {
      display_name: "7. 4K 10-bit HLG HDR Video & NLE Renderer",
      status: "pending",
      critique: "Synthesizes 2.5D parallax & diptych photo plates, titles, and 3-layer audio into 4K HDR MP4.",
    },
  ];
  renderJobProgress({
    status: "idle",
    progress_pct: 0,
    current_stage_title: "Ready to run all 11 AI editing agents + 4K HDR Renderer",
    last_status: "Select your media library above and click 'Run Full AI Edit & Render Video'.",
    elapsed_sec: 0,
    stages,
    activity_log: [],
  });
}

function renderDirectorHighlights(contracts) {
  const box = document.getElementById("directorHighlightsBox");
  if (!box) return;

  const dr = contracts?.director_review;
  const bgm = contracts?.bgm_selection;
  const pc = contracts?.photo_curation;

  if (!dr && !bgm && !pc) {
    box.innerHTML = `<p class="muted">Run the pipeline to inspect the Director's Sign-Off, BGM selection, and curated museum photo annotations.</p>`;
    return;
  }

  let html = "";
  if (dr) {
    html += `
      <div class="highlight-banner">
        <div style="font-size:0.95rem; font-weight:700; color:var(--accent);">
          🎬 Executive Director Sign-Off: Grade ${escapeHtml(dr.quality_grade)} (${dr.overall_quality_score}/10)
        </div>
        <div style="font-size:0.8rem; margin-top:4px;">${escapeHtml(dr.director_vision_statement)}</div>
      </div>
    `;
  }

  if (bgm) {
    html += `
      <div class="curated-photo-pill" style="border-color:rgba(168,85,247,0.45); background:rgba(168,85,247,0.08);">
        <strong>🎵 Selected Soundtrack (Stage 05 Music Supervisor):</strong><br/>
        <span style="color:#c084fc; font-weight:600;">${escapeHtml(bgm.track_title)}</span> — ${escapeHtml(bgm.composer_or_artist)}
        <span class="stage-badge-mini" style="margin-left:6px;">Mood Match ${bgm.mood_match_score}/10</span>
        ${
          bgm.rejected_user_bgm_reason
            ? `<div class="muted" style="font-size:0.74rem; margin-top:3px;">🛡️ ${escapeHtml(bgm.rejected_user_bgm_reason)}</div>`
            : ""
        }
      </div>
    `;
  }

  if (pc && Array.isArray(pc.items)) {
    const selectedPhotos = pc.items.filter((it) => it.is_selected);
    html += `
      <div style="margin:10px 0 6px; font-weight:700; font-size:0.84rem; color:var(--accent-cyan);">
        🖼️ Curated Gallery Plates (${selectedPhotos.length} Selected from ${pc.total_evaluated} Photos • Avg Taste ${pc.average_taste_score_selected}/10)
      </div>
    `;
    html += selectedPhotos
      .slice(0, 12)
      .map(
        (p) => `
        <div class="curated-photo-pill">
          <div style="display:flex; justify-content:space-between;">
            <strong>${escapeHtml(p.annotation_title)}</strong>
            <span class="stage-badge-mini">${escapeHtml(p.recommended_photo_mode)} • ${p.overall_taste_score}/10</span>
          </div>
          <div style="color:var(--accent); font-size:0.76rem;">${escapeHtml(p.annotation_subtitle)}</div>
          <div class="muted" style="font-size:0.71rem;">${escapeHtml(p.filename)} • ${escapeHtml(p.camera_badge)}</div>
        </div>`
      )
      .join("");
  }

  box.innerHTML = html;
}

// ---------------------------------------------------------------------------
// REGISTRY, CONTRACTS, CATALOG, TIMELINE & EXPORTS RENDERERS
// ---------------------------------------------------------------------------
function renderAgentRegistry(agents) {
  registeredAgentsList = agents || [];
  const box = document.getElementById("agentRegistryList");
  if (!box) return;
  if (!registeredAgentsList.length) {
    box.innerHTML = "<span class='muted'>No registered sub-agents found.</span>";
    return;
  }

  box.innerHTML = registeredAgentsList
    .map((a) => {
      const isPlugin = !a.enabled_by_default;
      const checked = "checked"; // Enable vfx_designer plugin by default in the studio UI
      const tag = isPlugin
        ? `<span style="color:#00e5ff; font-weight:700;">[PLUGIN]</span>`
        : `<span style="color:#10b981;">[CORE]</span>`;
      return `
        <label class="agent-toggle-card ${isPlugin ? "plugin" : ""}">
          <input type="checkbox" class="agent-role-checkbox" data-role="${a.role_id}" data-default="${a.enabled_by_default}" ${checked} />
          <div>
            <div><strong>${a.display_name}</strong> ${tag}</div>
            <div class="muted" style="font-size:0.74rem; margin-top:2px;">Stage ${a.stage_order} • <code>${a.role_id}</code> &rarr; ${a.produces_artifacts.join(", ") || "EditTimeline"}</div>
          </div>
        </label>
      `;
    })
    .join("");

  renderDefaultAgentStagesFromRegistry(registeredAgentsList);
}

function renderContractsAndBus(contracts) {
  latestAgentContracts = contracts || {};
  renderDirectorHighlights(latestAgentContracts);

  const busBox = document.getElementById("messageBusSummary");
  const viewer = document.getElementById("contractJsonViewer");
  const selector = document.getElementById("contractSelector");

  const bb = latestAgentContracts.blackboard;
  const dr = latestAgentContracts.director_review;
  let busHtml = "";
  if (dr) {
    busHtml += `
      <div class="bus-msg" style="background:rgba(255,200,59,0.12); border-left:3px solid #ffc83b; margin-bottom:6px;">
        <strong>🎬 Movie Production Director Sign-Off: Grade ${dr.quality_grade} (${dr.overall_quality_score}/10)</strong>
        • <span>${dr.interventions_count} Active Agent Improvements Directed</span>
      </div>
    `;
  }
  if (bb && bb.messages && bb.messages.length > 0) {
    busHtml += bb.messages
      .map(
        (m) => `
        <div class="bus-msg">
          <strong style="color:#ffc83b;">${escapeHtml(m.sender_role)}</strong> &rarr;
          <strong style="color:#00e5ff;">${escapeHtml(m.recipient_role)}</strong>
          <span class="muted">[${escapeHtml(m.message_type)}]</span>: ${escapeHtml(m.subject)}
        </div>`
      )
      .join("");
    if (busBox) busBox.innerHTML = busHtml;
  } else if (busBox) {
    busBox.innerHTML = "<span class='muted'>No inter-agent messages recorded yet.</span>";
  }

  const key = selector ? selector.value : "director_review";
  const doc = latestAgentContracts[key];
  if (viewer) {
    viewer.textContent = doc
      ? JSON.stringify(doc, null, 2)
      : `// ${key}.json has not been generated yet in this project workspace.`;
  }

  // If Dual-Stream Console is currently empty (e.g. loaded existing completed project), populate from contracts
  const doingEl = document.getElementById("liveAgentDoingStream");
  const doingCountEl = document.getElementById("agentDoingCount");
  if (doingEl && bb && Array.isArray(bb.executed_agents) && doingCountEl?.textContent === "0 actions") {
    const synthActions = [];
    bb.executed_agents.forEach((ea) => {
      synthActions.push({
        role_id: ea.role_id,
        action: `Executed ${ea.display_name} (Stage ${ea.stage_order})`,
        detail: JSON.stringify(ea.summary || {}).slice(0, 140),
        event_type: "agent_action",
      });
    });
    (bb.messages || []).forEach((m) => {
      synthActions.push({
        role_id: m.sender_role,
        action: `[${String(m.message_type).toUpperCase()}] -> ${m.recipient_role}: ${m.subject}`,
        detail: JSON.stringify(m.payload || {}).slice(0, 140),
        event_type: "bus_message",
      });
    });
    if (synthActions.length > 0) {
      if (doingCountEl) doingCountEl.textContent = `${synthActions.length} actions (loaded)`;
      doingEl.innerHTML = synthActions
        .map(
          (act) => `
          <div class="agent-action-card ${escapeHtml(act.event_type)}">
            <div><strong style="color:var(--accent-cyan);">[${escapeHtml(act.role_id)}]</strong> ${escapeHtml(act.action)}</div>
            <div class="muted" style="font-size:0.72rem; margin-top:2px;">${escapeHtml(act.detail)}</div>
          </div>`
        )
        .join("");
    }
  }

  const thinkingEl = document.getElementById("liveDirectorThinkingStream");
  const thinkingCountEl = document.getElementById("directorThinkingCount");
  if (thinkingEl && dr && Array.isArray(dr.stage_reviews) && thinkingCountEl?.textContent === "0 reviews") {
    if (thinkingCountEl) thinkingCountEl.textContent = `${dr.stage_reviews.length + 1} reviews (loaded)`;
    const cardsHtml = dr.stage_reviews
      .map(
        (sr) => `
        <div class="director-thought-card ${escapeHtml(sr.status)}">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong>🎬 Stage ${sr.stage_order} • <code>${escapeHtml(sr.reviewed_agent_role)}</code></strong>
            <span class="stage-badge-mini">${escapeHtml(String(sr.status).toUpperCase())} • ${Number(sr.quality_score).toFixed(1)}/10</span>
          </div>
          <div class="thought-row"><span class="thought-label">🧠 Critique:</span> ${escapeHtml(sr.critique)}</div>
          ${sr.directive_sent ? `<div class="thought-row"><span class="thought-label">⚡ Directive:</span> ${escapeHtml(sr.directive_sent)}</div>` : ""}
        </div>`
      )
      .join("");
    const finalCard = `
      <div class="director-thought-card approved">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <strong>🏆 Final Executive Screening Sign-Off</strong>
          <span class="stage-badge-mini">GRADE ${escapeHtml(dr.quality_grade)} • ${dr.overall_quality_score}/10</span>
        </div>
        <div class="thought-row"><span class="thought-label">🧠 Vision:</span> ${escapeHtml(dr.director_vision_statement)}</div>
      </div>`;
    thinkingEl.innerHTML = cardsHtml + finalCard;
  }
}

const contractSelectorEl = document.getElementById("contractSelector");
if (contractSelectorEl) {
  contractSelectorEl.addEventListener("change", () => {
    renderContractsAndBus(latestAgentContracts);
  });
}

function renderCatalogView(catalog) {
  const box = document.getElementById("catalogSummary");
  if (!box) return;
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
      .map((d) => `<span class="dialogue-chip">[${d.start_time.toFixed(1)}s-${d.end_time.toFixed(1)}s] “${escapeHtml(d.text)}”</span>`)
      .join("");
    html += `
      <div class="asset-item">
        <div><strong>🎬 ${escapeHtml(v.filename)}</strong> (${v.duration.toFixed(1)}s, ${v.width}x${v.height}, ${escapeHtml(v.color_transfer)})</div>
        <div class="muted" style="font-size:0.83rem;">${escapeHtml(v.overall_summary || "")}</div>
        <div>${dialoguesHtml || "<span class='muted' style='font-size:0.78rem;'>No speech segments</span>"}</div>
      </div>
    `;
  });

  if (photos.length > 0) {
    html += `<div class="asset-item"><strong>📸 Photos (${photos.length} total):</strong><div style="margin-top:6px; font-size:0.82rem;" class="muted">`;
    html += photos
      .slice(0, 24)
      .map((p) => `${escapeHtml(p.filename)} (${p.orientation[0].toUpperCase()}, score ${p.quality_score})`)
      .join(" • ");
    if (photos.length > 24) html += ` ... +${photos.length - 24} more`;
    html += `</div></div>`;
  }

  box.innerHTML = html;
}

function renderTimelineTracks(timeline) {
  const container = document.getElementById("visualTimelineTracks");
  if (!container) return;
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
      const roleTag = c.roll_type ? `[${c.roll_type.toUpperCase()}] ` : "";
      return `<div class="track-block ${c.type}" style="left:${left}%; width:${width}%;" title="${escapeHtml(c.id)}: ${escapeHtml(roleTag + name)} (${c.duration}s) - ${escapeHtml(c.narrative_note)}">${escapeHtml(roleTag + name)}</div>`;
    })
    .join("");

  const brollBlocks = (timeline.broll_track || [])
    .map((b) => {
      const left = ((b.timeline_start / totalDur) * 100).toFixed(2);
      const width = Math.max(1.8, (b.duration / totalDur) * 100).toFixed(2);
      const name = b.source.split(/[\\/]/).pop();
      return `<div class="track-block broll" style="left:${left}%; width:${width}%;" title="Track V2 Cutaway ${escapeHtml(b.id)}: ${escapeHtml(name)} (${b.duration}s) - ${escapeHtml(b.narrative_note)}">V2: ${escapeHtml(name)}</div>`;
    })
    .join("");

  const fxBlocks = (timeline.effects_track || [])
    .map((fx) => {
      const left = ((fx.timeline_start / totalDur) * 100).toFixed(2);
      const width = Math.max(1.4, (fx.duration / totalDur) * 100).toFixed(2);
      return `<div class="track-block vfx" style="left:${left}%; width:${width}%;" title="Stage-40 VFX (${escapeHtml(fx.effect_type)}) on ${escapeHtml(fx.target_clip_id)}">✨ ${escapeHtml(fx.effect_type)}</div>`;
    })
    .join("");

  const tBlocks = (timeline.title_track || [])
    .map((t) => {
      const left = ((t.timeline_start / totalDur) * 100).toFixed(2);
      const width = Math.max(2.0, (t.duration / totalDur) * 100).toFixed(2);
      return `<div class="track-block title" style="left:${left}%; width:${width}%;" title="${escapeHtml(t.style)}: ${escapeHtml(t.text)}">${escapeHtml(t.text)}</div>`;
    })
    .join("");

  const aBlocks = (timeline.audio_tracks || [])
    .map((a) => {
      const dur = a.duration || totalDur;
      const left = (((a.timeline_start || 0) / totalDur) * 100).toFixed(2);
      const width = Math.min(100, (dur / totalDur) * 100).toFixed(2);
      const name = a.source.split(/[\\/]/).pop();
      return `<div class="track-block audio" style="left:${left}%; width:${width}%;" title="${escapeHtml(name)}">🎵 ${escapeHtml(name)} (-14dB Auto-Duck)</div>`;
    })
    .join("");

  const vfxRowHtml =
    (timeline.effects_track || []).length > 0
      ? `<div class="track-row"><div class="track-label">Stage-40 VFX</div><div class="track-lane">${fxBlocks}</div></div>`
      : "";

  container.innerHTML = `
    <div style="font-size:0.82rem; margin-bottom:8px; color:#929db3;">
      Total Timeline Duration: <strong>${totalDur.toFixed(1)}s</strong>
      (${timeline.visual_track.length} V1 cuts, ${(timeline.broll_track || []).length} V2 B-Roll cutaways, ${(timeline.effects_track || []).length} VFX items)
    </div>
    <div class="track-row"><div class="track-label">V3 Titles</div><div class="track-lane">${tBlocks}</div></div>
    <div class="track-row"><div class="track-label">V2 B-Roll Cutaways</div><div class="track-lane">${brollBlocks}</div></div>
    <div class="track-row"><div class="track-label">V1 Storyline</div><div class="track-lane">${vBlocks}</div></div>
    ${vfxRowHtml}
    <div class="track-row"><div class="track-label">A3 Music Bed</div><div class="track-lane">${aBlocks}</div></div>
  `;
}

function updateVideoDownloadButtons(exportsMap, projectSummary = null) {
  const topBtn = document.getElementById("btnTopDownloadVideo");
  const bannerBox = document.getElementById("runCompleteDownloadBanner");
  const bannerBtn = document.getElementById("btnBannerDownloadMp4");
  const bannerSub = document.getElementById("runCompleteBannerSubtext");
  const heroBtn = document.getElementById("btnHeroDownloadMp4");

  const hasMp4 = Boolean(exportsMap?.mp4 || projectSummary?.has_rendered_video);
  const projDir = (
    document.getElementById("projectDirInput")?.value ||
    projectSummary?.project_dir ||
    ""
  ).trim();

  if (!hasMp4 || !projDir) {
    if (topBtn) topBtn.style.display = "none";
    if (bannerBox) bannerBox.style.display = "none";
    if (heroBtn) heroBtn.style.display = "none";
    return null;
  }

  const mtime = projectSummary?.rendered_video_mtime || Date.now();
  const sizeMb = projectSummary?.rendered_video_size_mb
    ? ` (${projectSummary.rendered_video_size_mb} MB)`
    : "";
  const dlUrl = `/api/projects/download-video?project_dir=${encodeURIComponent(projDir)}&t=${mtime}`;

  if (topBtn) {
    topBtn.href = dlUrl;
    topBtn.textContent = `⬇️ Download Video (.mp4)${sizeMb}`;
    topBtn.style.display = "inline-flex";
  }
  if (bannerBtn) {
    bannerBtn.href = dlUrl;
    bannerBtn.textContent = `⬇️ Download Newly Rendered Video (.mp4)${sizeMb}`;
  }
  if (bannerSub) {
    const fileLabel = exportsMap?.mp4 || projectSummary?.rendered_video_path || "final_story.mp4";
    bannerSub.textContent = `Ready to download: ${fileLabel}${sizeMb}`;
  }
  if (bannerBox) {
    bannerBox.style.display = "flex";
  }
  if (heroBtn) {
    heroBtn.href = dlUrl;
    heroBtn.textContent = `⬇️ Download Newly Rendered Video (.mp4)${sizeMb}`;
    heroBtn.style.display = "inline-flex";
  }
  return dlUrl;
}

function renderExportsView(exportsMap, projectSummary = null) {
  const exBox = document.getElementById("exportsList");
  const vidBox = document.getElementById("videoPlayerContainer");
  const mp4DlUrl = updateVideoDownloadButtons(exportsMap, projectSummary);
  if (!exBox || !vidBox) return;

  if (!exportsMap || Object.keys(exportsMap).length === 0) {
    exBox.innerHTML = "<p class='muted'>No exports generated yet.</p>";
    vidBox.innerHTML =
      "<p class='muted'>Run the pipeline in Tab 3 or load a rendered project in Tab 1 to preview and download the 4K HDR MP4 video here.</p>";
    return;
  }

  const mtime = projectSummary?.rendered_video_mtime || Date.now();
  const sizeLabel = projectSummary?.rendered_video_size_mb
    ? ` (${projectSummary.rendered_video_size_mb} MB)`
    : "";

  const labels = {
    mp4: `🎬 Newly Rendered 4K HDR Video (.mp4)${sizeLabel}`,
    fcpxml: "🎞️ DaVinci Resolve / Final Cut Pro XML (.fcpxml)",
    edl: "📋 CMX 3600 Edit Decision List (.edl)",
    otio: "📐 OpenTimelineIO Project (.otio)",
    srt: "💬 SubRip Subtitles (.srt)",
    ass: "✨ Styled SubStation Alpha Subtitles (.ass)",
  };

  exBox.innerHTML = Object.entries(exportsMap)
    .map(([ext, fullPath]) => {
      const isMp4 = ext === "mp4";
      const url =
        isMp4 && mp4DlUrl
          ? mp4DlUrl
          : `/api/file?path=${encodeURIComponent(fullPath)}&download=true&t=${mtime}`;
      return `<a class="export-link ${isMp4 ? "primary-mp4" : ""}" href="${url}" download><span>${
        labels[ext] || ext.toUpperCase()
      }</span><span>⬇️ Download ${ext.toUpperCase()}</span></a>`;
    })
    .join("");

  if (exportsMap.mp4) {
    const videoUrl = `/api/file?path=${encodeURIComponent(exportsMap.mp4)}&t=${mtime}`;
    vidBox.innerHTML = `<video controls preload="metadata" src="${videoUrl}"></video>`;
  } else {
    vidBox.innerHTML =
      "<p class='muted'>Timeline exports are ready. Click 'Render 4K HDR Video' to generate the MP4 video file.</p>";
  }
}

async function refreshProjectState() {
  const projDir = document.getElementById("projectDirInput").value.trim();
  const res = await fetch(`/api/state?project_dir=${encodeURIComponent(projDir)}`);
  if (!res.ok) return;
  const data = await res.json();
  setStatus(data.last_status || "Ready");

  if (data.projects) {
    renderProjectsList(data.projects, projDir);
  }
  if (data.registered_agents && registeredAgentsList.length === 0) {
    renderAgentRegistry(data.registered_agents);
  }
  if (data.job && data.job.stages && data.job.stages.length > 0) {
    renderJobProgress(data.job);
  }
  if (data.agent_contracts) {
    renderContractsAndBus(data.agent_contracts);
  }
  document.getElementById("annotationsYamlEditor").value = data.annotations_yaml || "";
  renderCatalogView(data.catalog || null);
  document.getElementById("timelineYamlEditor").value = data.timeline_yaml || "";
  renderTimelineTracks(data.timeline || null);
  document.getElementById("storyboardPreview").textContent =
    data.storyboard_md || "No storyboard generated yet.";
  renderExportsView(data.exports || {}, data.project_summary || null);
}

// ---------------------------------------------------------------------------
// FOLDER BROWSER MODAL & NATIVE PICKER HANDLERS
// ---------------------------------------------------------------------------
async function openBrowserModal(path = null) {
  const modal = document.getElementById("folderBrowserModal");
  if (modal) modal.style.display = "flex";
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  const res = await fetch(`/api/fs/browse${query}`);
  if (!res.ok) return;
  const data = await res.json();
  currentBrowserPath = data.current_path;
  currentBrowserParent = data.parent_path;

  document.getElementById("browserCurrentPath").textContent = currentBrowserPath;
  document.getElementById("browserFolderStats").textContent = `Direct files in this folder: 🎬 ${data.direct_videos} videos • 📸 ${data.direct_photos} photos`;

  const listEl = document.getElementById("browserSubdirList");
  if (!data.subdirectories || data.subdirectories.length === 0) {
    listEl.innerHTML = `<div class="muted" style="padding:10px;">No subfolders inside this directory. Click "✓ Use This Folder" above to select it.</div>`;
    return;
  }

  listEl.innerHTML = data.subdirectories
    .map(
      (sub) => `
      <div class="browser-folder-row" data-subpath="${escapeHtml(sub.path)}">
        <span>📁 <strong>${escapeHtml(sub.name)}</strong></span>
        <span class="muted">🎬 ${sub.video_count} videos • 📸 ${sub.photo_count} photos &rarr;</span>
      </div>`
    )
    .join("");

  listEl.querySelectorAll(".browser-folder-row").forEach((row) => {
    row.addEventListener("click", () => {
      openBrowserModal(row.dataset.subpath);
    });
  });
}

document.getElementById("btnBrowseModal")?.addEventListener("click", () => {
  const cur = document.getElementById("mediaDirInput").value.trim();
  openBrowserModal(cur || null);
});

document.getElementById("btnCloseBrowserModal")?.addEventListener("click", () => {
  document.getElementById("folderBrowserModal").style.display = "none";
});

document.getElementById("btnBrowserUp")?.addEventListener("click", () => {
  if (currentBrowserParent) openBrowserModal(currentBrowserParent);
});

document.getElementById("btnSelectCurrentBrowserFolder")?.addEventListener("click", async () => {
  document.getElementById("folderBrowserModal").style.display = "none";
  if (currentBrowserPath) {
    await selectMediaLibrary(currentBrowserPath, null, true);
  }
});

document.getElementById("btnPickNativeFolder")?.addEventListener("click", async () => {
  setStatus("Opening OS Folder Picker...");
  const cur = document.getElementById("mediaDirInput").value.trim();
  const res = await fetch(`/api/fs/pick-folder?initial_dir=${encodeURIComponent(cur)}`, { method: "POST" });
  const data = await res.json();
  if (!data.cancelled && data.selected_path) {
    await selectMediaLibrary(data.selected_path, data.library?.suggested_project_dir, true);
  } else if (data.error) {
    setStatus(data.error);
    openBrowserModal(cur || null);
  } else {
    setStatus("Ready");
  }
});

document.getElementById("btnInspectFolder")?.addEventListener("click", async () => {
  const cur = document.getElementById("mediaDirInput").value.trim();
  if (cur) await selectMediaLibrary(cur, null, true);
});

document.getElementById("btnRefreshState")?.addEventListener("click", refreshProjectState);
document.getElementById("btnRefreshAgents")?.addEventListener("click", refreshProjectState);

// ---------------------------------------------------------------------------
// START ASYNC PIPELINE JOB & POLL PROGRESS
// ---------------------------------------------------------------------------
async function pollActiveJob() {
  try {
    const res = await fetch("/api/jobs/status");
    if (!res.ok) return;
    const job = await res.json();
    renderJobProgress(job);
    setStatus(job.last_status || "Running...");

    if (job.status === "completed" || job.status === "error") {
      if (activePollTimer) {
        clearInterval(activePollTimer);
        activePollTimer = null;
      }
      await refreshProjectState();
      if (job.status === "completed") {
        const banner = document.getElementById("runCompleteDownloadBanner");
        if (banner && banner.style.display !== "none") {
          banner.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }
    }
  } catch (err) {
    console.error("Poll error:", err);
  }
}

document.getElementById("btnRunMultiAgent")?.addEventListener("click", async () => {
  const enabledOptional = [];
  const disabledRoles = [];
  document.querySelectorAll(".agent-role-checkbox").forEach((cb) => {
    const role = cb.dataset.role;
    const isDefault = cb.dataset.default === "true";
    if (cb.checked && !isDefault) {
      enabledOptional.push(role);
    } else if (!cb.checked && isDefault) {
      disabledRoles.push(role);
    }
  });

  const [w, h] = document.getElementById("resolutionSelect").value.split("x").map(Number);
  const promptText = document.getElementById("agentPromptInput").value.trim();
  const shouldRender = document.getElementById("renderVideoCheckbox")?.checked ?? true;

  const payload = {
    media_dir: document.getElementById("mediaDirInput").value.trim(),
    project_dir: document.getElementById("projectDirInput").value.trim(),
    user_prompt: promptText,
    extra_audio_path: document.getElementById("bgmPathInput").value.trim() || null,
    interview_answers: {
      narrative_style: currentIntakeAnswers.narrative_style || document.getElementById("intakeNarrativeStyle").value,
      dialogue_preference:
        currentIntakeAnswers.dialogue_preference || document.getElementById("intakeDialoguePref").value,
      photo_mode: currentIntakeAnswers.photo_mode || document.getElementById("intakePhotoMode").value,
      bgm_mood: currentIntakeAnswers.bgm_mood || "auto_match",
      target_duration_sec:
        currentIntakeAnswers.target_duration_sec || document.getElementById("intakeTargetDuration").value,
      must_include_topics: promptText,
    },
    enabled_optional_roles: enabledOptional.length ? enabledOptional : ["vfx_designer"],
    disabled_roles: disabledRoles,
    width: w,
    height: h,
    fps: 30,
    hdr: true,
    use_antigravity: document.getElementById("useAgStep1")?.checked ?? false,
    render_video: shouldRender,
  };

  setStatus("Starting Multi-Agent Pipeline...");
  const res = await fetch("/api/jobs/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    setStatus(`Error: ${err.detail}`);
    return;
  }
  const job = await res.json();
  renderJobProgress(job);

  if (activePollTimer) clearInterval(activePollTimer);
  activePollTimer = setInterval(pollActiveJob, 500);
});

// ---------------------------------------------------------------------------
// STANDALONE INSPECTOR TAB HANDLERS
// ---------------------------------------------------------------------------
document.getElementById("btnAnalyze")?.addEventListener("click", async () => {
  setStatus("Analyzing media & transcribing dialogues...");
  const payload = {
    media_dir: document.getElementById("mediaDirInput").value.trim(),
    project_dir: document.getElementById("projectDirInput").value.trim(),
    whisper_model: document.getElementById("whisperModelSelect").value,
    extra_audio_path: document.getElementById("bgmPathInput").value.trim() || null,
    use_antigravity: document.getElementById("useAgStep1").checked,
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
  setStatus("Media Analysis Complete!");
});

document.getElementById("btnSaveAnnotations")?.addEventListener("click", async () => {
  setStatus("Saving annotations.yaml...");
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
    setStatus("Saved annotations.yaml!");
  }
});

document.getElementById("btnGeneratePlan")?.addEventListener("click", async () => {
  setStatus("Generating baseline timeline...");
  const [w, h] = document.getElementById("resolutionSelect").value.split("x").map(Number);
  const payload = {
    project_dir: document.getElementById("projectDirInput").value.trim(),
    user_prompt: document.getElementById("agentPromptInput").value.trim(),
    width: w,
    height: h,
    fps: 30,
    hdr: true,
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
  setStatus("Baseline Timeline Ready!");
});

document.getElementById("btnSaveTimeline")?.addEventListener("click", async () => {
  setStatus("Validating & saving timeline.yaml...");
  const payload = {
    project_dir: document.getElementById("projectDirInput").value.trim(),
    yaml_text: document.getElementById("timelineYamlEditor").value,
  };
  const res = await fetch("/api/timeline", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    setStatus(`Validation Error: ${err.detail}`);
    return;
  }
  const data = await res.json();
  document.getElementById("storyboardPreview").textContent = data.storyboard_md;
  renderTimelineTracks(data.timeline);
  setStatus("Saved timeline.yaml & updated NLE exports!");
});

document.getElementById("btnRenderVideo")?.addEventListener("click", async () => {
  setStatus("Rendering 4K HDR MP4 video...");
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
  setStatus("Render Complete!");
});

// ---------------------------------------------------------------------------
// INITIALIZE STUDIO ON PAGE LOAD
// ---------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", async () => {
  try {
    const libRes = await fetch("/api/fs/libraries");
    if (libRes.ok) {
      const libData = await libRes.json();
      renderDiscoveredLibraries(libData.libraries || []);
      if (libData.libraries && libData.libraries.length > 0) {
        const firstLib = libData.libraries[0];
        await selectMediaLibrary(firstLib.media_dir, firstLib.suggested_project_dir, true);
        return;
      }
    }
  } catch (e) {
    console.warn("Could not load discovered libraries:", e);
  }
  const defaultMedia = document.getElementById("mediaDirInput")?.value.trim();
  if (defaultMedia) {
    await selectMediaLibrary(defaultMedia, null, false);
  } else {
    await refreshProjectState();
  }
});
