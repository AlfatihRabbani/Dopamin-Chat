// Dopamine Chat — Discord-style client
// State in window.S. Server is authoritative for session+model state.

const S = {
  selectedModel: null,
  selectedPersonality: null,
  modelLoaded: false,
  sessionActive: false,
  sessionId: null,
  generating: false,
  pendingApproval: null,
  thinkingHidden: false,
  models: [],
  personalities: [],
  history: [],
  settings: {},
  pipers: [],
  rvcs: [],
};

const $ = (id) => document.getElementById(id);

// ---- Toast -----------------------------------------------------------------
let toastEl = null, toastTimer = null;
function toast(msg, ms = 2500) {
  if (!toastEl) { toastEl = document.createElement("div"); toastEl.id = "toast"; document.body.appendChild(toastEl); }
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), ms);
}

// ---- HTTP ------------------------------------------------------------------
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
async function apiJSON(path, body, method = "POST") {
  return api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function escapeAttr(s) { return escapeHtml(s).replace(/"/g, "&quot;"); }

// ---- Theme -----------------------------------------------------------------
function applyTheme(name) {
  document.body.setAttribute("data-theme", name || "dark");
  document.querySelectorAll(".theme-card").forEach(b => b.classList.toggle("active", b.dataset.theme === name));
}

// ---- Mood ------------------------------------------------------------------
function moodColor(d) { return d > 75 ? "var(--green)" : d >= 50 ? "var(--yellow)" : d >= 30 ? "var(--orange)" : "var(--red)"; }
function moodLabel(d) { return d > 75 ? "HAPPY" : d >= 50 ? "MILD" : d >= 30 ? "STRESSED" : "DEPRESSED"; }
function updateMood(d) {
  $("mood-wrap").hidden = false;
  $("mood-fill").style.width = `${Math.max(0, Math.min(100, d))}%`;
  $("mood-fill").style.background = moodColor(d);
  $("mood-label").textContent = `${moodLabel(d)} ${d}`;
}

// ---- Multi-emotion mood panel ----------------------------------------------
// Lazy-loaded order + meta from /api/emotions/meta on first session load.
async function ensureEmotionMeta() {
  if (S.emotionMeta) return S.emotionMeta;
  try {
    const r = await fetch("/api/emotions/meta");
    S.emotionMeta = await r.json();
  } catch (_) {
    S.emotionMeta = { order: [], meta: {} };
  }
  return S.emotionMeta;
}

async function renderEmotionPanel(emotions) {
  const panel = $("emo-panel");
  if (!panel) return;
  if (!emotions || typeof emotions !== "object") return;
  const meta = await ensureEmotionMeta();
  const order = meta.order && meta.order.length ? meta.order : Object.keys(emotions);
  // Build rows once, then just patch widths/colors thereafter.
  if (panel.dataset.built !== "1") {
    panel.innerHTML = order.map(k => {
      const m = (meta.meta && meta.meta[k]) || {};
      const color = m.color || "#888";
      return `<div class="emo-row" data-k="${k}">
        <div class="emo-head"><span class="emo-name">${k}</span><span class="emo-val">0</span></div>
        <div class="emo-bar"><div class="emo-fill" style="background:${color}"></div></div>
      </div>`;
    }).join("");
    panel.dataset.built = "1";
  }
  for (const k of order) {
    const v = Math.max(0, Math.min(100, Number(emotions[k] ?? 0)));
    const row = panel.querySelector(`.emo-row[data-k="${k}"]`);
    if (!row) continue;
    row.querySelector(".emo-val").textContent = Math.round(v);
    row.querySelector(".emo-fill").style.width = `${v}%`;
    row.classList.toggle("muted", v < 15);
  }
}

function toggleEmotionPanel(force) {
  const panel = $("emo-panel");
  const btn = $("emo-toggle");
  if (!panel || !btn) return;
  const open = typeof force === "boolean" ? force : panel.hidden;
  panel.hidden = !open;
  btn.textContent = open ? "mood ▴" : "mood ▾";
}

// ---- Server rail (personalities as servers) --------------------------------
function renderRail() {
  const rail = $("rail-personalities");
  rail.innerHTML = S.personalities.map(p => {
    const pic = p.has_pfp ? `<img src="/api/personalities/${encodeURIComponent(p.id)}/pfp" alt=""/>` : escapeHtml((p.name || "?")[0]);
    return `<div class="rail-item" data-pid="${escapeAttr(p.id)}" title="${escapeAttr(p.name)}">${pic}</div>`;
  }).join("");
  for (const it of rail.querySelectorAll(".rail-item")) {
    it.onclick = () => railClick(it.dataset.pid);
  }
  refreshRailActive();
}

function refreshRailActive() {
  for (const el of $("rail-personalities").querySelectorAll(".rail-item")) {
    el.classList.toggle("active", el.dataset.pid === S.selectedPersonality);
  }
}

// Rail click: same flow as the settings persona card. If a chat is in
// progress and the user clicks a different character, selectPersonality()
// will prompt to start a new chat under that character.
function railClick(id) {
  if (S.sessionActive && id === S.selectedPersonality) return;          // already showing
  selectPersonality(id);
}

function selectPersonality(id, opts) {
  opts = opts || {};
  // If a chat is already active and the user is *switching to a different
  // character*, generation + saving must move to that character's session.
  // Otherwise the UI shows B's name/pfp while the server keeps generating as
  // A and writing to A's history file (the "wrong-character-saved" bug).
  // No confirm dialog — picking a different card is the explicit intent.
  if (S.sessionActive && S.selectedPersonality && id !== S.selectedPersonality
      && !opts.fromLoad) {
    S.selectedPersonality = id;
    refreshRailActive();
    const p = S.personalities.find(x => x.id === id);
    if (p) {
      $("active-persona-name").textContent = p.name;
      $("active-persona-desc").textContent = p.description || "";
    }
    newChat();
    return;
  }
  S.selectedPersonality = id;
  refreshRailActive();
  for (const el of document.querySelectorAll("#personality-grid .persona-card")) {
    el.classList.toggle("active", el.dataset.pid === id);
  }
  const p = S.personalities.find(x => x.id === id);
  if (p) {
    $("active-persona-name").textContent = p.name;
    $("active-persona-desc").textContent = p.description || "";
    $("persona-name").textContent = p.name;
    if (p.has_pfp) {
      const img = $("bot-pfp");
      img.src = `/api/personalities/${encodeURIComponent(id)}/pfp?t=${Date.now()}`;
      img.hidden = false;
    } else {
      $("bot-pfp").hidden = true;
    }
  }
  // Refresh chat list so the "select a character" placeholder gets replaced
  if (typeof loadHistory === "function") loadHistory();
}

// ---- Loaders ---------------------------------------------------------------
async function loadPersonalities() {
  S.personalities = await api("/api/personalities");
  renderRail();
  renderPersonalityGrid();
}

function renderPersonalityGrid() {
  const el = $("personality-grid");
  if (!el) return;
  el.innerHTML = S.personalities.map(p => `
    <div class="persona-card" data-pid="${escapeAttr(p.id)}">
      ${p.has_pfp
        ? `<img class="pfp" src="/api/personalities/${encodeURIComponent(p.id)}/pfp"/>`
        : `<div class="pfp"></div>`}
      <div>
        <div class="p-name">${escapeHtml(p.name)}</div>
        <div class="p-desc">${escapeHtml(p.description || "")}</div>
      </div>
    </div>`).join("");
  for (const it of el.querySelectorAll(".persona-card")) {
    it.onclick = () => selectPersonality(it.dataset.pid);
  }
}

async function loadModels() {
  S.models = await api("/api/models");
  const sel = $("model-select");
  sel.innerHTML = S.models.length
    ? S.models.map(m => `<option value="${escapeAttr(m.name)}">${escapeHtml(m.name)} (${m.format})</option>`).join("")
    : `<option value="">— no models in models/ —</option>`;
  if (S.selectedModel) sel.value = S.selectedModel;
}

async function loadHistory() {
  const el = $("history");
  if (!S.selectedPersonality) {
    S.history = [];
    el.innerHTML = `<div class="ch-item" style="opacity:.6;cursor:pointer" id="ch-pick-char">select a character →</div>`;
    const b = $("ch-pick-char");
    if (b) b.onclick = () => openSettings("personality");
    return;
  }
  // Backend filters to this personality only — each character sees its own chats.
  S.history = await api("/api/history?personality_id=" + encodeURIComponent(S.selectedPersonality));
  if (!S.history.length) {
    el.innerHTML = `<div class="ch-item" style="opacity:.5;cursor:default">no chats yet</div>`;
    return;
  }
  el.innerHTML = S.history.map(h => {
    const title = h.title || h.personality_name || h.session_id.slice(0, 8);
    const meta = `${h.messages} msgs • dop ${h.dopamine} • ${(h.updated_at || "").slice(0, 10)}`;
    return `<div class="ch-item" data-sid="${escapeAttr(h.session_id)}">
      <div class="ch-title-row"><span class="ch-hash">#</span><span>${escapeHtml(title)}</span></div>
      <div class="ch-meta">${escapeHtml(meta)}</div>
      <div class="row-actions">
        <button data-act="del" class="warn">delete</button>
      </div>
    </div>`;
  }).join("");
  for (const it of el.querySelectorAll(".ch-item")) {
    const sid = it.dataset.sid;
    it.onclick = () => loadSessionById(sid);
    const delBtn = it.querySelector("[data-act=del]");
    if (delBtn) delBtn.onclick = (e) => { e.stopPropagation(); deleteSession(sid); };
  }
}

async function loadSettings() {
  S.settings = await api("/api/settings");
  applyTheme(S.settings.theme || "dark");
  $("auto-rename").checked = !!S.settings.auto_rename_chats;
  $("user-name-input").value = S.settings.user_name || "";
  $("user-desc-input").value = S.settings.user_description || "";
  $("tts-enabled").checked = !!S.settings.tts_enabled;
  $("rvc-enabled").checked = !!S.settings.rvc_enabled;
  $("rvc-pitch").value = S.settings.rvc_pitch || 0;
  $("rvc-index-rate").value = S.settings.rvc_index_rate || 0.75;
  $("rvc-extractor").value = S.settings.rvc_pitch_extractor || "rmvpe";
  $("rvc-index").value = S.settings.rvc_index || "";
  $("user-strip-name").textContent = S.settings.user_name || "You";
  if (S.settings.user_pfp) {
    const img = $("user-pfp-img");
    img.src = `/api/user_pfp?t=${Date.now()}`;
    img.hidden = false;
    $("user-pfp-preview").src = img.src; $("user-pfp-preview").hidden = false;
  }
}

// Blender Cycles-style GPU picker:
//   top: single-select backend tabs (None / CUDA / OptiX / HIP / oneAPI / Metal / Vulkan)
//   bottom: per-backend device list with multi-select checkboxes (+ CPU row always)
S.gpuData = null;
S.gpuBackend = "none";
S.gpuEnabled = new Set();

const GPU_TAB_ORDER = ["none", "cuda", "rocm", "intel", "metal", "vulkan"];
const GPU_TAB_LABEL = {
  none: "None", cuda: "CUDA", rocm: "HIP",
  intel: "oneAPI", metal: "Metal", vulkan: "Vulkan",
};

function renderLoraList() {
  const el = $("imggen-loras");
  if (!el) return;
  const list = S.loras || [];
  if (!list.length) {
    el.innerHTML = `<div class="hint">No LoRAs found. Drop .safetensors files into ImageGen_LoRAs/.</div>`;
    return;
  }
  const active = new Map((S.settings.imggen_loras || []).map(l => [l.path, l.scale]));
  el.innerHTML = list.map(l => {
    const checked = active.has(l.path) ? "checked" : "";
    const scale = active.get(l.path) ?? 0.8;
    return `<div class="gpu-dev-row" data-path="${escapeAttr(l.path)}">
      <input type="checkbox" ${checked} class="lora-on" />
      <span style="flex:1">${escapeHtml(l.name)}</span>
      <input type="number" class="lora-scale" value="${scale}" step="0.05" min="0" max="2" style="width:70px" />
    </div>`;
  }).join("");
}

function collectLoras() {
  const out = [];
  for (const r of document.querySelectorAll("#imggen-loras .gpu-dev-row")) {
    if (!r.querySelector(".lora-on").checked) continue;
    out.push({
      path: r.dataset.path,
      scale: Number(r.querySelector(".lora-scale").value) || 1.0,
    });
  }
  return out;
}

function setImgGenMode(mode) {
  S.imggenMode = mode;
  document.querySelectorAll("#imggen-mode-tabs .gpu-bt-tab").forEach(t =>
    t.classList.toggle("active", t.dataset.mode === mode));
  $("imggen-local-pane").hidden = mode !== "local";
  $("imggen-server-pane").hidden = mode !== "server";
}

// Map server-side `kind` strings to sub-tab buckets.
const IMGGEN_KIND_MAP = {
  "hf-diffusers":                    "diffusers",
  "safetensors":                     "safetensors",
  "hf-transformers-NOT-DIFFUSERS":   "transformers",
  "gguf":                            "gguf",
  "gguf-text-llm-WRONG-FOLDER":      "gguf",
};

function filterModelSelect() {
  const sel = $("imggen-model");
  if (!sel) return;
  const want = S.imggenKind || "diffusers";
  const all = S.imggenAllModels || [];
  const filt = all.filter(m => (IMGGEN_KIND_MAP[m.kind] || m.kind) === want);
  sel.innerHTML = filt.length
    ? filt.map(m => {
        const warn = m.warning ? "  ⚠" : "";
        return `<option value="${escapeAttr(m.path)}" title="${escapeAttr(m.warning || "")}">${escapeHtml(m.name)} [${m.kind}]${warn}</option>`;
      }).join("")
    : `<option value="">— no ${want} models in ImageGen_Models/ —</option>`;
  if (S.settings.imggen_model_path) {
    const has = filt.some(m => m.path === S.settings.imggen_model_path);
    if (has) sel.value = S.settings.imggen_model_path;
  }
}

function setImgGenKind(kind) {
  S.imggenKind = kind;
  document.querySelectorAll(".imggen-kind-tabs .gpu-bt-tab").forEach(t =>
    t.classList.toggle("active", t.dataset.kind === kind));
  for (const pane of document.querySelectorAll(".imggen-kind-pane")) {
    const kinds = (pane.dataset.paneKind || "").split(/\s+/);
    pane.hidden = !kinds.includes(kind);
  }
  // Per-kind hint + download row visibility within the model sub-pane
  for (const el of document.querySelectorAll("[data-hint-kind]")) {
    const kinds = el.dataset.hintKind.split(/\s+/);
    el.hidden = !kinds.includes(kind);
  }
  for (const el of document.querySelectorAll("[data-dl-kind]")) {
    const kinds = el.dataset.dlKind.split(/\s+/);
    el.style.display = kinds.includes(kind) ? "" : "none";
  }
  if (kind !== "lora") filterModelSelect();
  // LoRA tab: surface currently-saved base model so user knows what LoRAs stack onto.
  const baseHint = $("imggen-lora-basehint");
  if (baseHint) {
    const base = S.settings.imggen_model_path || "";
    baseHint.innerHTML = base
      ? `Current base: <code>${escapeHtml(base.split('/').pop())}</code>`
      : `<b>No base model picked yet.</b> Switch to Diffusers/SafeTensors/Transformers/GGUF tab and pick one first.`;
  }
}

async function loadImgGen() {
  let info;
  try { info = await api("/api/imggen/info"); } catch (e) { return; }
  S.imggenAllModels = info.diffusers_models || [];
  const warnings = S.imggenAllModels.filter(m => m.warning).map(m => `${m.name}: ${m.warning}`);
  if (warnings.length) {
    setTimeout(() => { $("imggen-status").textContent += "  ⚠ " + warnings.join(" | "); }, 0);
  }
  // LoRA list
  S.loras = info.loras || [];
  renderLoraList();
  setImgGenMode(S.settings.imggen_mode || "local");
  // Auto-pick the kind that matches the saved model, else default to diffusers
  let initKind = S.settings.imggen_kind || "diffusers";
  if (S.settings.imggen_model_path) {
    const m = S.imggenAllModels.find(x => x.path === S.settings.imggen_model_path);
    if (m) initKind = IMGGEN_KIND_MAP[m.kind] || initKind;
  }
  setImgGenKind(initKind);
  $("imggen-backend").value = (S.settings.imggen_backend === "comfyui" || S.settings.imggen_backend === "sd_cpp")
    ? S.settings.imggen_backend : "comfyui";
  $("imggen-comfy-url").value = S.settings.imggen_comfy_url || "http://127.0.0.1:8188";
  $("imggen-comfy-model").value = S.settings.imggen_comfy_model || "";
  $("imggen-negative").value = S.settings.imggen_negative || "";
  $("imggen-width").value = S.settings.imggen_width || 512;
  $("imggen-height").value = S.settings.imggen_height || 512;
  $("imggen-steps").value = S.settings.imggen_steps || 20;
  $("imggen-strength").value = S.settings.imggen_strength || 0.75;
  $("imggen-offload").value = S.settings.imggen_offload_mode || "auto";
  const lines = [
    `diffusers: ${info.diffusers[0] ? "ok" : info.diffusers[1]}`,
    `comfyui:   ${info.comfyui[0] ? "ok" : info.comfyui[1]}`,
    `sd.cpp:    ${info.sd_cpp[0] ? "ok (" + info.sd_cpp[1] + ")" : info.sd_cpp[1]}`,
  ];
  $("imggen-status").textContent = lines.join("  |  ");
}

async function saveImgGen() {
  const mode = S.imggenMode || "local";
  // backend resolves to: 'local' when local mode, else the server-pane choice
  const backend = (mode === "local") ? "local" : ($("imggen-backend").value || "comfyui");
  // LoRA tab hides the model select; fall back to last saved path.
  const modelPath = ($("imggen-model").value || S.settings.imggen_model_path || "");
  await saveSettingsPatch({
    imggen_mode:       mode,
    imggen_kind:       S.imggenKind || "diffusers",
    imggen_backend:    backend,
    imggen_model_path: modelPath,
    imggen_comfy_url:  $("imggen-comfy-url").value,
    imggen_comfy_model: $("imggen-comfy-model").value,
    imggen_negative:   $("imggen-negative").value,
    imggen_width:      Number($("imggen-width").value),
    imggen_height:     Number($("imggen-height").value),
    imggen_steps:      Number($("imggen-steps").value),
    imggen_strength:   Number($("imggen-strength").value),
    imggen_offload_mode: $("imggen-offload").value,
    imggen_loras:        collectLoras(),
  });
  toast(`Settings saved. Loading model… (see terminal for progress)`);
  $("imggen-status").textContent = "loading model + LoRAs… (terminal shows progress)";
  let info;
  try {
    const r = await fetch("/api/imggen/load", { method: "POST" });
    info = await r.json();
  } catch (e) {
    toast(`Load failed: ${e}`);
    $("imggen-status").textContent = `load failed: ${e}`;
    return;
  }
  if (info.error) {
    toast(`Load error: ${info.error}`);
    $("imggen-status").textContent = `error: ${info.error}`;
    return;
  }
  const tag = info.loaded
    ? `loaded ${info.model_path ? info.model_path.split('/').pop() : ''} on ${info.device || '?'}`
    : (info.note || "no preload needed");
  const loraTag = (info.loras_applied || []).length
    ? `  +${info.loras_applied.length} LoRA(s)` : "";
  toast(`Image gen ${tag}${loraTag}`);
  $("imggen-status").textContent = `${tag}${loraTag}`;
}

async function testImgGen() {
  $("imggen-test-out").hidden = true;
  toast("Generating…");
  const fInit = $("imggen-test-init").files[0];
  let r;
  if (fInit) {
    const fd = new FormData();
    fd.append("prompt", $("imggen-test-prompt").value);
    fd.append("init_image", fInit);
    r = await fetch("/api/imggen/test", { method: "POST", body: fd });
  } else {
    r = await fetch("/api/imggen/test", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: $("imggen-test-prompt").value }),
    });
  }
  if (!r.ok) { const j = await r.json().catch(() => ({})); toast(`imggen: ${j.error || r.statusText}`); return; }
  const j = await r.json();
  $("imggen-test-out").src = j.url; $("imggen-test-out").hidden = false;
  toast(`Generated via ${j.mode || j.backend}`);
}

async function loadGPU() {
  let g;
  try {
    g = await api("/api/gpu/list");
  } catch (e) {
    console.warn("GPU list fetch failed:", e);
    const el = $("gpu-device-list");
    if (el) el.innerHTML = `<div class="gpu-dev-empty">GPU probe failed (retry: reopen tab). ${escapeHtml(e.message || "")}</div>`;
    return;
  }
  S.gpuData = g;
  S.gpuBackend = g.compute_backend || "none";
  S.gpuEnabled = new Set(g.devices_enabled || []);
  renderGPUTabs();
  renderGPUDevices();
  // Show backend notes (e.g. "torch is CPU-only, install cu124 wheel") under tabs.
  const noteEl = $("gpu-status");
  const notes = (g.backends || []).filter(b => b.note && !b.available)
                                  .map(b => `${b.name}: ${b.note}`);
  noteEl.textContent = notes.join("  •  ");
  renderGPUDiagnostic();
}

async function renderGPUDiagnostic() {
  try {
    const d = await api("/api/gpu/diagnose");
    const lines = Object.entries(d).map(([k, v]) =>
      `${k}: ${Array.isArray(v) ? v.join("; ") : v}`);
    $("gpu-status").textContent = lines.join("  •  ");
  } catch (_) {}
}

function renderGPUTabs() {
  const el = $("gpu-backend-tabs");
  const avail = {};
  for (const b of (S.gpuData?.backends || [])) avail[b.kind] = b.available;
  el.innerHTML = GPU_TAB_ORDER.map(k => {
    const isNone = k === "none";
    const ok = isNone || avail[k];
    const active = S.gpuBackend === k;
    const cls = ["gpu-bt-tab", active ? "active" : "", ok ? "" : "unavailable"].join(" ");
    return `<button class="${cls}" data-kind="${k}">${escapeHtml(GPU_TAB_LABEL[k])}</button>`;
  }).join("");
  for (const t of el.querySelectorAll(".gpu-bt-tab")) {
    t.onclick = () => {
      if (t.classList.contains("unavailable")) { toast(`${GPU_TAB_LABEL[t.dataset.kind]} not available on this system`); return; }
      S.gpuBackend = t.dataset.kind;
      renderGPUTabs();
      renderGPUDevices();
    };
  }
}

function renderGPUDevices() {
  const el = $("gpu-device-list");
  const k = S.gpuBackend;
  let gpuDevs = [];
  if (k !== "none") {
    const b = (S.gpuData?.backends || []).find(x => x.kind === k);
    gpuDevs = b ? (b.devices || []) : [];
  }
  // CPU row from CPU backend
  const cpuB = (S.gpuData?.backends || []).find(x => x.kind === "cpu");
  const cpuDev = cpuB ? (cpuB.devices[0] || null) : null;

  const rows = [];
  for (const d of gpuDevs) {
    const id = `${k}:${d.index}`;
    const meta = d.memory_gib ? `${d.memory_gib} GB` : "";
    rows.push(devRow(id, d.name, meta));
  }
  if (cpuDev) {
    rows.push(devRow("cpu", cpuDev.name, cpuDev.cores ? `${cpuDev.cores} cores` : ""));
  }
  if (!rows.length) {
    el.innerHTML = `<div class="gpu-dev-empty">No devices for this backend</div>`;
    return;
  }
  el.innerHTML = rows.join("");
  for (const r of el.querySelectorAll(".gpu-dev-row")) {
    const id = r.dataset.id;
    const cb = r.querySelector("input[type=checkbox]");
    cb.checked = S.gpuEnabled.has(id);
    cb.onchange = () => {
      if (cb.checked) S.gpuEnabled.add(id);
      else S.gpuEnabled.delete(id);
    };
    r.onclick = (e) => {
      if (e.target === cb) return;
      cb.checked = !cb.checked; cb.onchange();
    };
  }
}

function devRow(id, name, meta) {
  return `<div class="gpu-dev-row" data-id="${escapeAttr(id)}">
    <input type="checkbox" />
    <span>${escapeHtml(name)}</span>
    <span class="gpu-dev-meta">${escapeHtml(meta)}</span>
  </div>`;
}

async function applyGPU() {
  const devs = [...S.gpuEnabled];
  try {
    const r = await apiJSON("/api/gpu/apply", {
      compute_backend: S.gpuBackend,
      devices_enabled: devs,
    });
    $("gpu-status").textContent = `applied: backend=${S.gpuBackend}, devices=[${devs.join(", ")}]. ${r.note || ""}`;
    toast("GPU selection saved. Reload the model to use it.");
  } catch (e) { toast(`apply failed: ${e.message}`); }
}

async function loadVoiceLists() {
  let v;
  try { v = await api("/api/voice/list"); }
  catch (e) { return; }
  S.pipers = v.piper_voices || [];
  S.rvcs = v.rvc_models || [];
  $("tts-voice").innerHTML = S.pipers.length
    ? S.pipers.map(p => `<option value="${escapeAttr(p.name)}">${escapeHtml(p.name)}</option>`).join("")
    : `<option value="">— drop .onnx files in voices/piper/ —</option>`;
  if (S.settings.tts_voice) $("tts-voice").value = S.settings.tts_voice;
  $("rvc-pth").innerHTML = S.rvcs.length
    ? S.rvcs.map(r => `<option value="${escapeAttr(r.pth)}" data-index="${escapeAttr(r.index)}">${escapeHtml(r.name)}</option>`).join("")
    : `<option value="">— no .pth in voices/rvc/ or applio/ —</option>`;
  if (S.settings.rvc_pth) {
    $("rvc-pth").value = S.settings.rvc_pth;
  }
  // auto-pair index when selecting pth
  $("rvc-pth").onchange = () => {
    const opt = $("rvc-pth").selectedOptions[0];
    if (opt && opt.dataset.index) $("rvc-index").value = opt.dataset.index;
  };
  $("tts-status").textContent = v.piper_ok && v.piper_ok[0] ? "piper available" : (v.piper_ok ? v.piper_ok[1] : "");
  if (v.rvc_ok && v.rvc_ok[0]) {
    const d = v.rvc_device || {};
    let line = "Applio RVC available";
    if (d.device === "cuda:0" && d.gpu_name) {
      line += ` — GPU: ${d.gpu_name} (${d.vram_gb} GB)`;
    } else if (d.device === "cpu") {
      line += " — running on CPU (slow)";
    } else if (d.error) {
      line += ` — device probe: ${d.error}`;
    }
    $("rvc-status").textContent = line;
  } else {
    $("rvc-status").textContent = v.rvc_ok ? v.rvc_ok[1] : "";
  }
}

// ---- Load model ------------------------------------------------------------
async function loadModel() {
  const name = $("model-select").value;
  if (!name) { toast("no model selected"); return; }
  S.selectedModel = name;
  $("model-status").textContent = `loading ${name}…`;
  $("diag-load").textContent = `model: loading ${name}…`;
  toast(`Loading ${name}…`);
  try {
    const r = await apiJSON("/api/load_model", { name });
    S.modelLoaded = true;
    $("model-status").textContent = `loaded: ${r.backend} • vision ${r.has_vision ? "yes" : "no"} • ${r.load_time_s}s`;
    $("diag-load").textContent = `model: ${r.backend} (${r.load_time_s}s)`;
    $("user-strip-status").textContent = `${r.backend}`;
    toast(`Model loaded in ${r.load_time_s}s`);
  } catch (e) {
    $("model-status").textContent = `load failed: ${e.message}`;
    toast(`Load failed: ${e.message}`);
  }
}

// ---- Session ---------------------------------------------------------------
async function newChat() {
  if (!S.modelLoaded || !S.selectedPersonality) {
    openSettings(!S.modelLoaded ? "models" : "personality");
    toast(!S.modelLoaded ? "Pick a model first" : "Pick a character first");
    return;
  }
  const s = await apiJSON("/api/new_session", { personality_id: S.selectedPersonality });
  renderSession(s);
  S.sessionActive = true;
  S.sessionId = s.session_id;
  loadHistory();
  toast(`New session: ${s.personality.name}`);
}

async function loadSessionById(sid) {
  if (!S.modelLoaded) { openSettings("models"); toast("Pick a model first"); return; }
  const s = await apiJSON("/api/load_session", { session_id: sid });
  renderSession(s);
  S.sessionActive = true;
  S.sessionId = s.session_id;
  selectPersonality(s.personality.id, { fromLoad: true });
  toast(`Resumed ${s.personality.name}`);
}

async function deleteSession(sid) {
  if (!confirm(`Delete ${sid}?`)) return;
  await fetch(`/api/history/${encodeURIComponent(sid)}`, { method: "DELETE" });
  loadHistory();
}

async function resetSession() {
  if (!S.sessionActive) { toast("No active session"); return; }
  if (!confirm("Reset dopamine + clear THIS chat's memory?")) return;
  const s = await apiJSON("/api/reset", {});
  renderSession(s);
  toast("Reset");
}

// How many message bubbles the user sees at once. The full chat is held
// in S.allMessages; scrolling to the top loads PAGE_SIZE older messages
// from that array. None of this paging affects what the LLM sees — the
// backend manages its own context window from s.messages on the server.
const CHAT_PAGE_SIZE = 20;
S.allMessages = [];
S.shownStart = 0;

function renderSession(s) {
  $("persona-name").textContent = s.personality.name;
  updateMood(s.dopamine);
  renderEmotionPanel(s.emotions || {});
  $("messages").innerHTML = "";
  S.allMessages = (s.messages || []).filter(m => m.role !== "system");
  const total = S.allMessages.length;
  S.shownStart = Math.max(0, total - CHAT_PAGE_SIZE);
  for (let i = S.shownStart; i < total; i++) {
    const m = S.allMessages[i];
    const bubble = appendMessage(m.role, m.content || "");
    if (m.followup) bubble.parentElement.classList.add("followup");
    // Replay any tool calls that were attached to this assistant message
    // (generated images, run_command output, etc.) so they survive refresh.
    if (Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) renderToolCall(tc);
    }
  }
  if (S.shownStart > 0) prependLoadMoreSentinel();
  scrollToBottom();
}

function prependLoadMoreSentinel() {
  const m = $("messages");
  if (!m) return;
  let s = document.getElementById("chat-loadmore");
  if (s) s.remove();
  s = document.createElement("div");
  s.id = "chat-loadmore";
  s.className = "ch-loadmore";
  s.textContent = `↑ load previous ${Math.min(CHAT_PAGE_SIZE, S.shownStart)} messages (only you can see them — the AI's context is unchanged)`;
  s.onclick = loadMoreHistory;
  m.prepend(s);
}

function loadMoreHistory() {
  const m = $("messages");
  if (!m || S.shownStart <= 0) return;
  const prevH = m.scrollHeight;
  const newStart = Math.max(0, S.shownStart - CHAT_PAGE_SIZE);
  const slice = S.allMessages.slice(newStart, S.shownStart);
  S.shownStart = newStart;
  // Build new wrappers and insert before any existing bubble (and the sentinel)
  const sentinel = document.getElementById("chat-loadmore");
  for (let i = slice.length - 1; i >= 0; i--) {
    const mm = slice[i];
    const tmp = document.createElement("div");
    document.body.appendChild(tmp);
    appendMessage(mm.role, mm.content || "");
    if (Array.isArray(mm.tool_calls)) {
      for (const tc of mm.tool_calls) renderToolCall(tc);
    }
    const inserted = [];
    let n = $("messages").lastElementChild;
    while (n && !inserted.includes(n)) {
      inserted.unshift(n);
      if (inserted.length > 1 + (mm.tool_calls?.length || 0)) break;
      n = n.previousElementSibling;
    }
    for (const node of inserted) {
      if (sentinel) m.insertBefore(node, sentinel.nextSibling);
      else m.insertBefore(node, m.firstChild);
    }
    tmp.remove();
  }
  if (S.shownStart > 0) prependLoadMoreSentinel();
  else if (sentinel) sentinel.remove();
  // Preserve scroll offset so the user stays anchored at their read position
  m.scrollTop = m.scrollHeight - prevH;
}

// ---- Messages --------------------------------------------------------------
function appendMessage(role, raw) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  let pfpSrc = "";
  if (role === "user" && S.settings.user_pfp) pfpSrc = `/api/user_pfp`;
  if (role === "assistant" && S.selectedPersonality) {
    const p = S.personalities.find(x => x.id === S.selectedPersonality);
    if (p && p.has_pfp) pfpSrc = `/api/personalities/${encodeURIComponent(p.id)}/pfp`;
  }
  const pfp = pfpSrc
    ? `<img class="pfp" src="${pfpSrc}"/>`
    : `<div class="pfp"></div>`;
  const who = role === "user" ? (S.settings.user_name || "you") : (S.personalities.find(x => x.id === S.selectedPersonality)?.name || "ai");
  wrap.innerHTML = `${pfp}<div><div class="who">${escapeHtml(who)}</div><div class="bubble"></div></div>`;
  const bubble = wrap.querySelector(".bubble");
  bubble.innerHTML = formatContent(raw);
  $("messages").appendChild(wrap);
  return bubble;
}

function attachProgressBar(wrap, meta) {
  let bar = wrap.querySelector(".gen-progress");
  if (!bar) {
    bar = document.createElement("div");
    bar.className = "gen-progress";
    bar.innerHTML = `
      <div class="gen-progress-row">
        <div class="gen-progress-track"><div class="gen-progress-fill"></div></div>
        <div class="gen-progress-label">0% • 0/${meta.max_new} tok • 0.00 tok/s</div>
      </div>
      <div class="gen-progress-meta">context ${meta.context_tokens} • seed ${meta.seed}</div>`;
    wrap.querySelector("div:last-child").appendChild(bar);
  }
  return bar;
}
function updateProgressBar(wrap, p) {
  const bar = wrap.querySelector(".gen-progress");
  if (!bar) return;
  const fill = bar.querySelector(".gen-progress-fill");
  const lab  = bar.querySelector(".gen-progress-label");
  fill.style.width = `${p.pct}%`;
  const phaseTag = (p.phase === "prefill") ? " · prefilling" : "";
  const rate = (typeof p.tok_per_s === "number") ? p.tok_per_s : 0;
  lab.textContent = `${p.pct}% • ${p.tokens}/${p.max} tok • ${rate.toFixed(2)} tok/s • ${p.elapsed_s}s${phaseTag}`;
  if (p.phase === "prefill") bar.classList.add("prefilling");
  else bar.classList.remove("prefilling");
}
function finalizeProgressBar(wrap, statsLine) {
  const bar = wrap.querySelector(".gen-progress");
  if (!bar) return;
  bar.classList.add("done");
  bar.innerHTML = `<div class="gen-stats">${escapeHtml(statsLine)}</div>`;
}

// Extract fenced ``` blocks BEFORE escaping so we can render them properly
// and not let inline markdown rules chew them up. Returns {text, blocks}.
function extractCodeBlocks(raw) {
  const blocks = [];
  // ```lang\n...\n```  (lang optional)
  const re = /```([a-zA-Z0-9_+\-]*)\n?([\s\S]*?)```/g;
  const text = raw.replace(re, (m, lang, body) => {
    const idx = blocks.length;
    blocks.push({ lang: (lang || "code").trim(), body: body.replace(/\n$/, "") });
    return `\u200BCBLK${idx}\u200B`;
  });
  return { text, blocks };
}

function renderCodeBlock(b) {
  const label = b.lang ? b.lang.charAt(0).toUpperCase() + b.lang.slice(1) : "Code";
  const body = highlightCode(b.body, b.lang);
  // download data-URI is built lazily at click time via dataset
  return `<div class="codeblock" data-lang="${escapeAttr(b.lang)}">
    <div class="codeblock-head">
      <span class="codeblock-lang">${escapeHtml(label)}</span>
      <span class="codeblock-actions">
        <button class="codeblock-btn" data-act="download" title="Download">⬇</button>
        <button class="codeblock-btn" data-act="copy" title="Copy">⧉</button>
      </span>
    </div>
    <pre class="codeblock-body"><code>${body}</code></pre>
    <textarea class="codeblock-src" hidden>${escapeHtml(b.body)}</textarea>
  </div>`;
}

// Very lightweight syntax highlighter — handles strings, comments, numbers,
// and a few keywords per language. Not a full parser; good enough to look
// like the screenshot.
function highlightCode(code, lang) {
  let s = escapeHtml(code);
  const L = (lang || "").toLowerCase();
  const KEYWORDS = {
    lua:        ["local","function","end","if","then","else","elseif","for","while","do","return","in","nil","true","false","not","and","or","break","repeat","until"],
    python:     ["def","class","return","if","else","elif","for","while","import","from","as","with","try","except","finally","raise","yield","lambda","None","True","False","and","or","not","in","is","pass","break","continue","global","nonlocal"],
    javascript: ["function","const","let","var","return","if","else","for","while","switch","case","break","continue","new","class","extends","this","super","import","export","from","as","async","await","try","catch","finally","throw","typeof","instanceof","null","undefined","true","false"],
    typescript: ["function","const","let","var","return","if","else","for","while","switch","case","break","continue","new","class","extends","this","super","import","export","from","as","async","await","try","catch","finally","throw","typeof","instanceof","null","undefined","true","false","interface","type","enum","public","private","protected"],
    bash:       ["if","then","else","fi","for","do","done","while","case","esac","function","return","local","echo","exit","export"],
    sh:         ["if","then","else","fi","for","do","done","while","case","esac","function","return","local","echo","exit","export"],
  };
  const kws = KEYWORDS[L] || [];
  // Pull strings + comments out FIRST so keyword/number regex don't touch them
  const slots = [];
  const stash = (cls, body) => { slots.push({ cls, body }); return `\u200CHS${slots.length-1}\u200C`; };
  // Comments
  if (L === "python" || L === "bash" || L === "sh") {
    s = s.replace(/(#.*)$/gm, (_, c) => stash("c-com", c));
  }
  if (L === "lua") {
    s = s.replace(/(--\[\[[\s\S]*?\]\])/g, (_, c) => stash("c-com", c));
    s = s.replace(/(--.*)$/gm, (_, c) => stash("c-com", c));
  }
  if (L === "javascript" || L === "typescript" || L === "c" || L === "cpp" || L === "java" || L === "go" || L === "rust") {
    s = s.replace(/(\/\*[\s\S]*?\*\/|\/\/.*$)/gm, (_, c) => stash("c-com", c));
  }
  // Strings
  s = s.replace(/("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')/g, (_, c) => stash("c-str", c));
  // Numbers
  s = s.replace(/\b(\d+(?:\.\d+)?)\b/g, (_, n) => stash("c-num", n));
  // Keywords
  if (kws.length) {
    const re = new RegExp("\\b(" + kws.join("|") + ")\\b", "g");
    s = s.replace(re, (_, k) => stash("c-kw", k));
  }
  // Restore slots with classes
  s = s.replace(/\u200CHS(\d+)\u200C/g, (_, i) => {
    const o = slots[Number(i)];
    return `<span class="${o.cls}">${o.body}</span>`;
  });
  return s;
}

// Discord-style inline markdown: ***bold-italic***, **bold**, *italic*,
// _italic_, __underline__, ~~strikethrough~~, `inline code`, and the
// roleplay convention *action* (handled by *italic* rule above). Applied
// AFTER escapeHtml + AFTER code blocks are stashed.
function applyInlineMarkdown(s) {
  // Inline code first so its contents are protected
  const codeSlots = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => {
    codeSlots.push(c);
    return `\u200DIC${codeSlots.length-1}\u200D`;
  });
  // ***bold italic*** and ___bold italic___
  s = s.replace(/\*\*\*([^*\n]+)\*\*\*/g, "<b><i>$1</i></b>");
  s = s.replace(/___([^_\n]+)___/g, "<b><i>$1</i></b>");
  // __underline__
  s = s.replace(/__([^_\n]+)__/g, "<u>$1</u>");
  // **bold**
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>");
  // ~~strikethrough~~
  s = s.replace(/~~([^~\n]+)~~/g, "<s>$1</s>");
  // *italic* (also handles roleplay actions like *blushes*, *pets hair*)
  s = s.replace(/(^|[^\w*])\*([^*\n]+)\*(?!\*)/g, '$1<i class="rp">$2</i>');
  // _italic_
  s = s.replace(/(^|[^\w_])_([^_\n]+)_(?!_)/g, "$1<i>$2</i>");
  // Restore inline code
  s = s.replace(/\u200DIC(\d+)\u200D/g, (_, i) =>
    `<code class="inline-code">${escapeHtml(codeSlots[Number(i)])}</code>`);
  return s;
}

function formatContent(raw, streaming = false) {
  // 1) Pull fenced code blocks out before any escaping
  const { text: t1, blocks } = extractCodeBlocks(raw);
  // 2) Pull ALL <think> regions (closed + trailing open) so we can collect
  //    them into ONE pill at the top of the bubble. Main text never shows
  //    raw <think> markup.
  let thinkText = "";
  let main = t1;
  // Closed <think>...</think>
  const CLOSED_RAW = /(?:<\|think\|>|<think>)([\s\S]*?)(?:<\|\/think\|>|<\/think>)/g;
  main = main.replace(CLOSED_RAW, (_, body) => { thinkText += body + "\n"; return ""; });
  // Trailing open <think> (no close yet, streaming)
  if (streaming) {
    const OPEN_RAW = /(?:<\|think\|>|<think>)([\s\S]*)$/;
    main = main.replace(OPEN_RAW, (_, body) => { thinkText += body; return ""; });
  }
  thinkText = thinkText.trim();
  // Pick a short "summary" — the last non-empty line of the think text.
  let thinkSummary = "";
  if (thinkText) {
    const lines = thinkText.split(/\n/).map(l => l.trim()).filter(Boolean);
    thinkSummary = (lines[lines.length - 1] || "").slice(0, 80);
  }
  // 3) Escape main body
  let s = escapeHtml(main);
  s = stripOrphanThinkCloses(s);
  // 4) Tool-call markup
  s = s.replace(/&lt;tool&gt;\s*([\w\-]+)\s*&lt;\/tool&gt;(?:\s*&lt;args&gt;([\s\S]*?)&lt;\/args&gt;)?/g,
    (_, name, args) => `<div class="tool-block">🔧 tool call: <b>${name}</b>(${args ? args.trim() : "(no args)"})</div>`);
  // 5) Discord-style inline formatting (bold/italic/underline/strike/code)
  s = applyInlineMarkdown(s);
  // 6) Restore code blocks (rendered with their own highlighter)
  s = s.replace(/\u200BCBLK(\d+)\u200B/g, (_, i) => renderCodeBlock(blocks[Number(i)]));
  // 7) Build the unified Thinking pill at the top.
  //    - While streaming: always shown (logo spins, label cycles or shows think summary).
  //    - When done streaming AND thinkText present: pill stays, label = "Show thinking",
  //      body holds the full think trace, click to expand.
  //    - When done AND no thinkText: pill is omitted (nothing to show).
  let pill = "";
  if (streaming || thinkText) {
    const liveCls = streaming ? " thinking-live" : "";
    const ambientCls = thinkText ? "" : " thinking-ambient";
    const initialLabel = streaming ? (thinkSummary || "Thinking\u2026") : "Show thinking";
    const escSummary = escapeHtml(thinkSummary);
    const escBody = escapeHtml(thinkText);
    pill = `<details class="thinking thinking-pill${liveCls}${ambientCls}" open><summary class="think-summary">` +
      `<img class="think-logo" src="/api/think-logo.gif" alt="">` +
      `<span class="think-text" data-summary="${escSummary}">${escapeHtml(initialLabel)}</span>` +
      `<span class="think-chev">\u25BE</span></summary>` +
      `<div class="think-body">${escBody}</div></details>`;
  }
  return pill + s;
}
function stripOrphanThinkCloses(s) {
  const OPENS = ["&lt;|think|&gt;", "&lt;think&gt;"];
  const CLOSES = ["&lt;|/think|&gt;", "&lt;/think&gt;"];
  let out = "", i = 0, depth = 0;
  while (i < s.length) {
    let matched = false;
    for (const t of OPENS) { if (s.startsWith(t, i)) { depth++; out += t; i += t.length; matched = true; break; } }
    if (matched) continue;
    for (const t of CLOSES) {
      if (s.startsWith(t, i)) {
        if (depth > 0) { depth--; out += t; }
        i += t.length; matched = true; break;
      }
    }
    if (matched) continue;
    out += s[i]; i++;
  }
  return out;
}
function scrollToBottom() { const m = $("messages"); m.scrollTop = m.scrollHeight; }

// ---- Approval modal --------------------------------------------------------
function showApprovalModal(p) {
  if (S.pendingApproval && S.pendingApproval.timer) clearInterval(S.pendingApproval.timer);
  $("approval-name").textContent = p.name;
  $("approval-args").textContent = JSON.stringify(p.args, null, 2);
  $("approval-reason").textContent = p.reason || "";
  const total = p.timeout_s || 120;
  $("approval-countdown").textContent = total;
  $("approval-modal").hidden = false;
  toast("Tool approval requested");
  let left = total;
  const tick = setInterval(() => {
    left--; const cd = $("approval-countdown");
    if (cd) cd.textContent = Math.max(0, left);
    if (left <= 0) { clearInterval(tick); hideApprovalModal(); }
  }, 1000);
  S.pendingApproval = { approval_id: p.approval_id, name: p.name, timer: tick };
}
function hideApprovalModal() {
  $("approval-modal").hidden = true;
  if (S.pendingApproval && S.pendingApproval.timer) clearInterval(S.pendingApproval.timer);
  S.pendingApproval = null;
}
async function submitApproval(decision) {
  if (!S.pendingApproval) return;
  try { await apiJSON("/api/approve", { approval_id: S.pendingApproval.approval_id, decision }); }
  catch (e) { toast(`Approval submit failed: ${e.message}`); }
  hideApprovalModal();
}

function showTerminationOverlay(seconds, dopamine) {
  const old = document.getElementById("term-overlay"); if (old) old.remove();
  const overlay = document.createElement("div");
  overlay.id = "term-overlay";
  overlay.innerHTML = `<div class="term-card">
    <div class="term-title">Session terminated</div>
    <div class="term-sub">The bot has left.</div>
    <div class="term-meta">final dopamine: <b>${dopamine}</b></div>
    <div class="term-meta">reload in <span id="term-countdown">${seconds}</span>s</div>
  </div>`;
  document.body.appendChild(overlay);
  let s = seconds;
  const tick = setInterval(() => {
    s--; const cd = document.getElementById("term-countdown");
    if (cd) cd.textContent = Math.max(0, s);
    if (s <= 0) { clearInterval(tick); try { window.close(); } catch (_) {} window.location.reload(); }
  }, 1000);
}

function renderToolCall(p) {
  // Compact notes indicator instead of the full tool card.
  if (p.name === "personality_note" && !p.denied) {
    const act = (p.args && p.args.action) || "read";
    const r = p.result || {};
    let label;
    if (act === "append")   label = "📝 noted something";
    else if (act === "replace") label = "📝 rewrote notes";
    else if (act === "delete")  label = `🗑 removed ${r.removed ?? "?"} line(s) from notes`;
    else if (act === "clear")   label = "🧹 cleared notes";
    else                        label = "📖 read notes";
    const w = document.createElement("div");
    w.className = "msg notes-indicator";
    w.innerHTML = `<div></div><div class="bubble notes-pill">${label}</div>`;
    $("messages").appendChild(w);
    scrollToBottom();
    return;
  }
  const wrap = document.createElement("div");
  wrap.className = "msg tool-call-msg";
  const denied = !!p.denied;
  const hasError = !denied && p.result && p.result.error;
  const status = denied ? "DENIED" : (hasError ? "ERROR" : "OK");
  const cls = denied || hasError ? "tool-result-denied" : "tool-result-ok";
  const r = p.result || {};
  const argStr = JSON.stringify(p.args);
  const argText = argStr.length > 200 ? argStr.slice(0, 200) + "…" : argStr;
  let body = "";
  if (denied) {
    body = `<div class="tool-reason">${escapeHtml(r.denied_reason || r.error || "denied")}</div>`;
    if (r.hint) body += `<div class="tool-hint">${escapeHtml(r.hint)}</div>`;
  } else if (hasError) {
    body = `<div class="tool-reason">${escapeHtml(r.error)}</div>`;
  } else if (p.name === "run_command") {
    const out = (r.output || "").replace(/\s+$/, "");
    body = `<div class="tool-meta">exit ${r.exit_code}${r.cwd ? " • cwd " + r.cwd : ""}</div>
            <pre class="tool-output">${escapeHtml(out || "(no output)")}</pre>
            ${r.truncated ? `<div class="tool-meta">…truncated</div>` : ""}`;
  } else if (r.url) {
    const meta = `${r.backend || "image"} • ${r.width || "?"}×${r.height || "?"} • ${r.steps || "?"} steps${r.seed != null ? ` • seed ${r.seed}` : ""}`;
    body = `<img class="tool-image" src="${escapeAttr(r.url)}" alt="" />
            <div class="tool-meta">${escapeHtml(meta)}</div>`;
  } else {
    let dump; try { dump = JSON.stringify(r, null, 2); } catch (_) { dump = String(r); }
    if (dump.length > 4000) dump = dump.slice(0, 4000) + "\n…[truncated]";
    body = `<pre class="tool-output">${escapeHtml(dump)}</pre>`;
  }
  wrap.innerHTML = `<div></div><div class="bubble ${cls}">
    <div class="tool-head"><span class="tool-name">🔧 ${escapeHtml(p.name)}</span><span class="tool-status">${status}</span></div>
    <div class="tool-args">${escapeHtml(argText)}</div>${body}</div>`;
  $("messages").appendChild(wrap);
  scrollToBottom();
}

// ---- Image upload ----------------------------------------------------------
S.pendingImagePreview = "";  // data URL for thumb in next user bubble
async function uploadImage(file) {
  const fd = new FormData(); fd.append("image", file);
  // Build a data-URL preview so we can show the thumbnail in the user bubble
  // when they send. Independent from the server-side upload.
  try {
    S.pendingImagePreview = await new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = () => res(r.result);
      r.onerror = rej;
      r.readAsDataURL(file);
    });
  } catch (e) { S.pendingImagePreview = ""; }
  const r = await fetch("/api/upload_image", { method: "POST", body: fd });
  const j = await r.json();
  if (j.error) { toast(j.error); S.pendingImagePreview = ""; return; }
  $("img-status").hidden = false;
  $("img-name").textContent = j.attached;
}
async function clearImage() {
  await fetch("/api/clear_image", { method: "POST" });
  $("img-status").hidden = true;
  S.pendingImagePreview = "";
}

// ---- Send + stream ---------------------------------------------------------
async function send() {
  const text = $("input").value.trim();
  if (!text) return;
  if (!S.modelLoaded) { openSettings("models"); toast("Pick a model in settings"); return; }
  if (!S.selectedPersonality) { openSettings("personality"); toast("Pick a character"); return; }
  if (!S.sessionActive) {
    await newChat();
    if (!S.sessionActive) return;
  }
  if (S.generating) { toast("Already generating"); return; }

  $("input").value = "";
  const userBubble = appendMessage("user", text);
  // If an image was attached this turn, show its thumbnail in the user bubble
  // so the sender sees what was sent. Clear the preview after rendering.
  if (S.pendingImagePreview) {
    const thumb = document.createElement("img");
    thumb.src = S.pendingImagePreview;
    thumb.className = "attached-thumb";
    userBubble.prepend(thumb);
    S.pendingImagePreview = "";
    $("img-status").hidden = true;
  }
  scrollToBottom();
  const bubble = appendMessage("assistant", "");
  bubble.parentElement.classList.add("streaming");
  // Render an initial empty pill so the logo is visible before any tokens arrive.
  bubble.innerHTML = formatContent("", true);
  scrollToBottom();
  let raw = "";

  S.generating = true;
  const sendBtn = $("send-btn");
  sendBtn.textContent = "Stop";
  sendBtn.classList.add("stopping");
  sendBtn.disabled = false;
  S.abortCtl = new AbortController();
  $("diag-turn").textContent = "turn: streaming…";
  startThinkCycle();

  let resp;
  try {
    resp = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, session_id: S.sessionId }),
      signal: S.abortCtl.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") {
      bubble.textContent += "\n[stopped]";
    } else {
      bubble.textContent = `[network error: ${e.message}]`;
    }
    S.generating = false;
    sendBtn.textContent = "Send";
    sendBtn.classList.remove("stopping");
    sendBtn.disabled = false;
    bubble.parentElement.classList.remove("streaming");
    stopThinkCycle();
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let leftover = "", inFarewell = false, farewellBubble = null, farewellRaw = "";
  let followupBubble = null, followupRaw = "";

  function flush(b, txt, streaming = true) { b.innerHTML = formatContent(txt, streaming); scrollToBottom(); }

  let aborted = false;
  while (true) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (e) {
      if (e.name === "AbortError") { aborted = true; break; }
      throw e;
    }
    const { done, value } = chunk;
    if (done) break;
    leftover += decoder.decode(value, { stream: true });
    const blocks = leftover.split("\n\n");
    leftover = blocks.pop();
    for (const block of blocks) {
      if (!block.trim()) continue;
      let ev = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) ev = line.slice(7).trim();
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;
      let p; try { p = JSON.parse(data); } catch { continue; }

      if (ev === "mood") { updateMood(p.dopamine); if (p.emotions) renderEmotionPanel(p.emotions); }
      else if (ev === "slot") {
        $("diag-turn").textContent = `turn: slot ${p.slot}/${p.max}`;
      }
      else if (ev === "gen_start") attachProgressBar(bubble.parentElement, p);
      else if (ev === "progress") updateProgressBar(bubble.parentElement, p);
      else if (ev === "followup_start_meta") { if (followupBubble) attachProgressBar(followupBubble.parentElement, p); }
      else if (ev === "followup_progress") { if (followupBubble) updateProgressBar(followupBubble.parentElement, p); }
      else if (ev === "token") { raw += p.t; flush(bubble, raw, true); }
      else if (ev === "tool_approval_request") showApprovalModal(p);
      else if (ev === "tool_call") renderToolCall(p);
      else if (ev === "followup_start") {
        bubble.parentElement.classList.remove("streaming");
        flush(bubble, raw, false);
        followupBubble = appendMessage("assistant", "");
        followupBubble.parentElement.classList.add("streaming", "followup");
        followupRaw = "";
      } else if (ev === "followup_token") {
        followupRaw += p.t;
        if (followupBubble) flush(followupBubble, followupRaw, true);
      } else if (ev === "followup_done") {
        if (followupBubble) {
          followupBubble.parentElement.classList.remove("streaming");
          flush(followupBubble, followupRaw, false);
          if (p.stats_line) finalizeProgressBar(followupBubble.parentElement, p.stats_line);
        }
      } else if (ev === "dopamine_penalty") {
        updateMood(p.dopamine);
        toast(`Tool denial penalty: ${p.penalty}`);
      } else if (ev === "done") {
        bubble.parentElement.classList.remove("streaming");
        flush(bubble, raw, false);
        if (p.stats_line) finalizeProgressBar(bubble.parentElement, p.stats_line);
        updateMood(p.dopamine);
        $("diag-turn").textContent = `turn: ${p.tokens} tok • ${p.elapsed_s}s • ${p.tok_per_s} tok/s`;
        clearImage();
        // TTS if enabled
        if (S.settings.tts_enabled) speakText(raw);
        // refresh history (may have auto-renamed)
        setTimeout(loadHistory, 1500);
        if (p.terminated) {
          inFarewell = true;
          farewellBubble = appendMessage("assistant", "");
          farewellBubble.parentElement.style.opacity = "0.7";
        }
      } else if (ev === "farewell_token") {
        farewellRaw += p.t;
        if (farewellBubble) flush(farewellBubble, farewellRaw);
      } else if (ev === "farewell_done") {
        S.sessionActive = false;
        loadHistory();
        showTerminationOverlay(p.shutdown_in_seconds || 4, p.dopamine);
      } else if (ev === "error") {
        bubble.textContent += `\n[error: ${p.error}]`;
        toast(`Error: ${p.error}`);
      }
    }
  }
  S.generating = false;
  S.abortCtl = null;
  const sb = $("send-btn");
  sb.textContent = "Send";
  sb.classList.remove("stopping");
  sb.disabled = false;
  if (aborted) {
    raw += "\n[stopped]";
    flush(bubble, raw, false);
  }
  bubble.parentElement.classList.remove("streaming");
  stopThinkCycle();
}

// ---- Thinking-text cycler --------------------------------------------------
// Rotates the visible `<summary>` of any live <details.thinking-live> while
// the model is generating, so the user sees: thinking → still thinking →
// random text → random text → loops. Pure cosmetic.
// Gemini-style progress phrases for the in-progress "Thinking" pill.
const THINK_PHRASES = [
  "Thinking…",
  "Assessing the Task",
  "Analyzing User Data",
  "Considering the Angles",
  "Drafting a Response",
  "Reviewing Context",
  "Choosing the Right Words",
  "Almost There",
];
let _thinkInterval = null;
let _thinkIdx = 0;
function startThinkCycle() {
  stopThinkCycle();
  _thinkIdx = 0;
  const tick = () => {
    const phrase = THINK_PHRASES[_thinkIdx % THINK_PHRASES.length];
    _thinkIdx++;
    for (const t of document.querySelectorAll(".thinking-live .think-text")) {
      // If the pill has a live <think> summary, show that instead of the
      // generic cycler phrases. flush() refreshes data-summary on each
      // streamed chunk so this stays in sync with the model's thoughts.
      const sum = t.getAttribute("data-summary") || "";
      t.textContent = sum ? sum : phrase;
    }
  };
  tick();
  _thinkInterval = setInterval(tick, 2200);
}
function stopThinkCycle() {
  if (_thinkInterval) clearInterval(_thinkInterval);
  _thinkInterval = null;
  // Freeze: stop the logo spin (drop .thinking-live). Ambient pills (no
  // real <think> body) get removed entirely. Real <think>-content pills
  // collapse to "Show thinking" label and stay clickable.
  for (const d of document.querySelectorAll(".thinking-live")) {
    d.classList.remove("thinking-live");
    if (d.classList.contains("thinking-ambient")) {
      d.remove();
      continue;
    }
    const t = d.querySelector(".think-text");
    if (t) { t.removeAttribute("data-summary"); t.textContent = "Show thinking"; }
    d.removeAttribute("open");
  }
}

// ---- TTS helpers -----------------------------------------------------------
async function speakText(text) {
  // Strip thinking blocks for TTS
  const clean = text
    .replace(/<\|?\/?think\|?>/g, " ")
    .replace(/<tool>[\s\S]*?<\/args>/g, " ")
    .replace(/\s+/g, " ").trim();
  if (!clean) return;
  try {
    const r = await fetch("/api/voice/speak", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: clean }),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      toast(`TTS: ${j.error || r.statusText}`);
      return;
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url); audio.play();
  } catch (e) { toast(`TTS failed: ${e.message}`); }
}

// ---- Settings modal --------------------------------------------------------
function openSettings(tab) {
  $("settings-modal").hidden = false;
  if (tab) switchTab(tab);
}
function closeSettings() { $("settings-modal").hidden = true; }
function switchTab(name) {
  document.querySelectorAll(".settings-tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".pane").forEach(p => p.hidden = p.dataset.pane !== name);
  // Refetch GPU list whenever the GPU tab is opened — page-load fetch can lose
  // races vs. user opening settings, and torch state may change after model load.
  if (name === "gpu") { loadGPU().catch(e => console.warn("loadGPU:", e)); }
}

async function saveSettingsPatch(patch) {
  S.settings = await apiJSON("/api/settings", patch);
  // Side-effects from saved values
  applyTheme(S.settings.theme || "dark");
  $("user-strip-name").textContent = S.settings.user_name || "You";
}

async function saveUserProfile() {
  await saveSettingsPatch({
    user_name: $("user-name-input").value,
    user_description: $("user-desc-input").value,
  });
  const f = $("user-pfp-input").files[0];
  if (f) {
    const fd = new FormData(); fd.append("image", f);
    const r = await fetch("/api/user_pfp", { method: "POST", body: fd });
    const j = await r.json();
    if (j.error) toast(j.error);
    else {
      const img = $("user-pfp-img"); img.src = `/api/user_pfp?t=${Date.now()}`; img.hidden = false;
      $("user-pfp-preview").src = img.src; $("user-pfp-preview").hidden = false;
    }
  }
  toast("Profile saved");
}

async function testTTS() {
  const text = $("tts-test-text").value;
  const voice = $("tts-voice").value;
  const r = await fetch("/api/voice/speak", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice }),
  });
  if (!r.ok) { const j = await r.json().catch(() => ({})); toast(`TTS: ${j.error || r.statusText}`); return; }
  const blob = await r.blob();
  const audio = $("tts-audio");
  audio.src = URL.createObjectURL(blob); audio.hidden = false; audio.play();
}

function setRvcProgress(pct, indeterminate = false) {
  const wrap = $("rvc-progress");
  const fill = $("rvc-progress-fill");
  const label = $("rvc-progress-label");
  wrap.hidden = false;
  if (indeterminate) {
    fill.classList.add("indeterminate");
    label.textContent = "…";
  } else {
    fill.classList.remove("indeterminate");
    const p = Math.max(0, Math.min(100, pct));
    fill.style.width = p + "%";
    label.textContent = Math.round(p) + "%";
  }
}

function hideRvcProgress() {
  const wrap = $("rvc-progress");
  const fill = $("rvc-progress-fill");
  fill.classList.remove("indeterminate");
  fill.style.width = "0%";
  wrap.hidden = true;
}

async function convertAudio() {
  const f = $("rvc-audio-in").files[0];
  if (!f) { toast("pick an audio file first"); return; }
  const fd = new FormData();
  fd.append("audio", f);
  fd.append("rvc_pth", $("rvc-pth").value);
  fd.append("rvc_index", $("rvc-index").value);
  fd.append("pitch", $("rvc-pitch").value);
  fd.append("index_rate", $("rvc-index-rate").value);
  fd.append("extractor", $("rvc-extractor").value);

  const btn = $("rvc-convert");
  btn.disabled = true;
  $("rvc-output").hidden = true;
  setRvcProgress(0);

  // Phase 1: upload (real progress, 0–25%).
  // Phase 2: server processing (time-estimated curve, 25–95%).
  // Phase 3: download complete → 100%.
  let processingTimer = null;
  let phase = "upload";

  const blob = await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/voice/convert");
    xhr.responseType = "blob";

    xhr.upload.onprogress = (e) => {
      if (phase !== "upload") return;
      if (e.lengthComputable) {
        const p = (e.loaded / e.total) * 25;
        setRvcProgress(p);
      }
    };
    xhr.upload.onload = () => {
      phase = "processing";
      // Smooth asymptotic climb 25% → 95% over ~estimated time.
      // RVC inference scales with audio length; rough estimate 8s per MB.
      const estMs = Math.max(4000, f.size / 1024 / 1024 * 8000);
      const t0 = Date.now();
      processingTimer = setInterval(() => {
        const elapsed = Date.now() - t0;
        const ratio = 1 - Math.exp(-elapsed / estMs);
        setRvcProgress(25 + ratio * 70);
      }, 200);
    };

    xhr.onerror = () => reject(new Error("network error"));
    xhr.onload = () => {
      if (processingTimer) { clearInterval(processingTimer); processingTimer = null; }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response);
      } else {
        let msg = xhr.statusText || `HTTP ${xhr.status}`;
        try {
          const reader = new FileReader();
          reader.onload = () => {
            try { msg = JSON.parse(reader.result).error || msg; } catch {}
            reject(new Error(msg));
          };
          reader.readAsText(xhr.response);
          return;
        } catch {}
        reject(new Error(msg));
      }
    };
    xhr.send(fd);
  }).catch((err) => {
    if (processingTimer) clearInterval(processingTimer);
    toast(`RVC: ${err.message}`);
    hideRvcProgress();
    return null;
  }).finally(() => {
    btn.disabled = false;
  });

  if (!blob) return;

  setRvcProgress(100);
  const url = URL.createObjectURL(blob);
  const audio = $("rvc-audio-out");
  audio.src = url;
  $("rvc-audio-download").href = url;
  $("rvc-output").hidden = false;
  audio.play();
  // Auto-hide bar after a moment so the output gets focus.
  setTimeout(hideRvcProgress, 800);
}

// ---- Event wiring ----------------------------------------------------------
$("send-btn").onclick = () => {
  if (S.generating && S.abortCtl) { S.abortCtl.abort(); return; }
  send();
};
$("new-chat").onclick = newChat;
$("reset").onclick = resetSession;
$("emo-toggle").onclick = () => toggleEmotionPanel();
$("img-btn").onclick = () => $("img-input").click();
$("img-input").onchange = e => { const f = e.target.files[0]; if (f) uploadImage(f); e.target.value = ""; };
$("img-clear").onclick = clearImage;
$("input").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });

$("approval-allow").onclick = () => submitApproval("allow");
$("approval-deny").onclick  = () => submitApproval("deny");

$("toggle-thinking").onclick = () => {
  S.thinkingHidden = !S.thinkingHidden;
  document.body.classList.toggle("hide-thinking", S.thinkingHidden);
  $("toggle-thinking").classList.toggle("active", S.thinkingHidden);
};

$("open-settings").onclick = () => openSettings("general");
$("open-settings-2").onclick = () => openSettings("general");
$("settings-close").onclick = closeSettings;
document.querySelectorAll(".settings-tab").forEach(t => t.onclick = () => switchTab(t.dataset.tab));
document.querySelectorAll(".theme-card").forEach(c => c.onclick = () => saveSettingsPatch({ theme: c.dataset.theme }));

$("auto-rename").onchange = () => saveSettingsPatch({ auto_rename_chats: $("auto-rename").checked });
$("user-save").onclick = saveUserProfile;
$("model-refresh").onclick = loadModels;
$("model-load").onclick = loadModel;
$("tts-refresh").onclick = loadVoiceLists;
$("rvc-refresh").onclick = loadVoiceLists;
$("tts-enabled").onchange = () => saveSettingsPatch({ tts_enabled: $("tts-enabled").checked });
$("rvc-enabled").onchange = () => saveSettingsPatch({ rvc_enabled: $("rvc-enabled").checked });
$("tts-voice").onchange = () => saveSettingsPatch({ tts_voice: $("tts-voice").value });
$("rvc-pth").addEventListener("change", () => saveSettingsPatch({ rvc_pth: $("rvc-pth").value, rvc_index: $("rvc-index").value }));
$("rvc-index").onchange = () => saveSettingsPatch({ rvc_index: $("rvc-index").value });
$("rvc-pitch").onchange = () => saveSettingsPatch({ rvc_pitch: Number($("rvc-pitch").value) });
$("rvc-index-rate").onchange = () => saveSettingsPatch({ rvc_index_rate: Number($("rvc-index-rate").value) });
$("rvc-extractor").onchange = () => saveSettingsPatch({ rvc_pitch_extractor: $("rvc-extractor").value });
$("tts-test").onclick = testTTS;
$("rvc-convert").onclick = convertAudio;
$("gpu-apply").onclick = applyGPU;
$("gpu-reinstall-torch").onclick = async () => {
  if (!confirm("Reinstall torch with CUDA wheel matched to nvidia-smi? ~2GB. Restart ./run-web.sh after it finishes.")) return;
  toast("Reinstalling torch — this can take several minutes…");
  try {
    const r = await apiJSON("/api/gpu/reinstall_torch", {});
    toast(`tag=${r.used_tag || "FAIL"} • verify=${r.verify}. Restart now.`);
    $("gpu-status").textContent =
      `used_tag=${r.used_tag}, tried=[${(r.tried||[]).join(",")}], verify=${r.verify}\n\n${r.log_tail}`;
  } catch (e) { toast(`failed: ${e.message}`); }
};
$("imggen-refresh").onclick = loadImgGen;
$("imggen-loras-refresh").onclick = loadImgGen;
$("imggen-save").onclick = saveImgGen;
$("imggen-test-btn").onclick = testImgGen;
document.querySelectorAll("#imggen-mode-tabs .gpu-bt-tab").forEach(t =>
  t.onclick = () => setImgGenMode(t.dataset.mode));
document.querySelectorAll(".imggen-kind-tabs .gpu-bt-tab").forEach(t =>
  t.onclick = () => setImgGenKind(t.dataset.kind));
$("imggen-dl-btn").onclick = async () => {
  const repo = $("imggen-dl-repo").value.trim();
  const file = $("imggen-dl-file").value.trim();
  if (!repo) { toast("repo_id required"); return; }
  toast("Downloading… (this may take a while)");
  try {
    const r = await apiJSON("/api/imggen/download", { repo_id: repo, filename: file });
    toast(`Saved to ${r.saved}`);
    await loadImgGen();
  } catch (e) { toast(`download failed: ${e.message}`); }
};

// ---- Drag & drop file upload ----------------------------------------------
// Images go to /api/upload_image (vision). Anything else goes to
// /api/upload_document (text extraction → injected as next turn's context).
const IMAGE_EXTS = new Set(["png","jpg","jpeg","gif","webp","bmp"]);
let _dragDepth = 0;

function showDropOverlay() { const el = $("drop-overlay"); if (el) el.hidden = false; }
function hideDropOverlay() { const el = $("drop-overlay"); if (el) el.hidden = true; }

window.addEventListener("dragenter", (e) => {
  if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes("Files")) return;
  _dragDepth++; showDropOverlay();
});
window.addEventListener("dragover", (e) => {
  if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes("Files")) return;
  e.preventDefault(); e.dataTransfer.dropEffect = "copy";
});
window.addEventListener("dragleave", (e) => {
  _dragDepth = Math.max(0, _dragDepth - 1);
  if (_dragDepth === 0) hideDropOverlay();
});
window.addEventListener("drop", async (e) => {
  if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
  e.preventDefault();
  _dragDepth = 0; hideDropOverlay();
  for (const f of e.dataTransfer.files) {
    await uploadDropped(f);
  }
});

async function uploadDropped(file) {
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  // Images go through uploadImage() so the preview thumbnail is built.
  if (IMAGE_EXTS.has(ext)) {
    toast(`Uploading ${file.name}…`);
    await uploadImage(file);
    return;
  }
  const fd = new FormData();
  fd.append("document", file);
  toast(`Uploading ${file.name}…`);
  try {
    const r = await fetch("/api/upload_document", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) { toast(`Upload failed: ${j.error || r.status}`); return; }
    toast(`Document attached: ${j.attached} (${j.chars} chars)`);
  } catch (e) {
    toast(`Upload error: ${e}`);
  }
}

// Delegated handler for code-block copy/download buttons.
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".codeblock-btn");
  if (!btn) return;
  const box = btn.closest(".codeblock");
  if (!box) return;
  const src = box.querySelector(".codeblock-src");
  const text = src ? src.value : "";
  const act = btn.dataset.act;
  if (act === "copy") {
    navigator.clipboard.writeText(text).then(
      () => toast("Copied"),
      () => toast("Copy failed"));
  } else if (act === "download") {
    const lang = box.dataset.lang || "code";
    const extMap = { lua: "lua", python: "py", javascript: "js", typescript: "ts",
                     bash: "sh", sh: "sh", c: "c", cpp: "cpp", java: "java",
                     go: "go", rust: "rs", html: "html", css: "css", json: "json" };
    const ext = extMap[lang.toLowerCase()] || "txt";
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `snippet.${ext}`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 100);
  }
});

// Restore current model on page refresh. Chat session is intentionally NOT
// auto-restored — the user picks one from the history sidebar so the wrong
// session doesn't pop up when they switch characters.
async function restoreOnRefresh() {
  try {
    const st = await api("/api/status");
    if (st.model_loaded) {
      S.modelLoaded = true;
      // Prefer the discovered model name (matches the dropdown values);
      // fall back to backend label if name wasn't reported by the server.
      S.selectedModel = st.model_name || st.backend || S.selectedModel;
      const ml = $("model-select"); if (ml && S.selectedModel) ml.value = S.selectedModel;
    }
  } catch (e) { /* server may not be ready */ }
}

// Boot
(async () => {
  await loadSettings();
  await loadPersonalities();
  await loadModels();
  await loadHistory();
  await loadVoiceLists();
  await loadGPU();
  await loadImgGen();
  await restoreOnRefresh();
  // If no model/personality, pop settings on first visit
  if (!S.models.length) toast("Drop a model into models/ folder");
})();
