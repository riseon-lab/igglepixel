// ─── Forge — frontend ─────────────────────────────────────────────────────
//
// ES module entry point. Talks to FastAPI at the same origin. Imports the
// crypto + IDB helpers used for browser-side asset encryption (Phase 3).
// ──────────────────────────────────────────────────────────────────────────

import { deriveKey, encryptBlob, hexToBytes, verifyCanary, PBKDF2_DEFAULT_ITERATIONS } from './crypto.js';
import { storeKey, loadKey, clearKey } from './key_store.js';

function readStorage(key, fallback = null) {
  try {
    const value = localStorage.getItem(key);
    return value == null ? fallback : value;
  } catch {
    return fallback;
  }
}

function readJSONStorage(key, fallback = {}) {
  try {
    return JSON.parse(readStorage(key, JSON.stringify(fallback))) || fallback;
  } catch {
    try { localStorage.removeItem(key); } catch {}
    return fallback;
  }
}

function writeJSONStorage(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
}

// ── State ────────────────────────────────────────────────────────────────
const state = {
  models:        [],
  gpu:           null,
  filter:        'all',
  assetFilter:   'all',
  running:       {},
  loras:         [],
  assets:        [],
  civResults:    [],
  selected:      null,        // model currently in drawer
  params:        {},          // working copy of params (workspace + drawer)
  settings:      readJSONStorage('forge_settings', {}),

  // Per-model lifecycle (downloaded? loading? running? generating?)
  modelStates:   {},

  // Workspace
  workspace:     null,        // model_id of currently-open workspace, or null
  workspaceParams: {},        // model_id -> persisted prompt/settings while app is open
  promptModes:   readJSONStorage('forge_prompt_modes', {}),   // model_id -> 'builder' | 'raw'
  promptBuilders: readJSONStorage('forge_prompt_builders', {}),// model_id -> guided prompt fields
  recents:       {},          // model_id -> [asset, …] (this session only)
  imgInputs:     {},          // { model_id -> { ref:{img,enabled}, pose:{...}, ... } }
  loraStates:    readJSONStorage('forge_lora_states', {}), // persisted per-model LoRA toggles + strengths
  components:    [],          // installed split-file components: [{target, filename, size_mb, path}]
  runtimes:      {},          // { model_id -> {state: 'missing'|'installing'|'ready'|'error', last_line?, error?} }
  // HF browser + active download queue. The browser is a flat list of
  // weight files (safetensors/ckpt/bin/gguf/pth) for a given repo. The
  // queue tracks server-side download jobs so the user can see progress,
  // cancel, retry, and dismiss without leaving the page.
  hfBrowse:      { repo: '', revision: 'main', files: [], selected: new Set(), filter: '', loading: false, error: null },
  downloadJobs:  [],          // mirror of /api/hf/jobs response
  downloadStats: {},          // { job_id -> {prevBytes, prevAt, mbps} } for rolling MB/s
  chats:         {},          // { model_id -> [{role, text, thinking?, streaming?}] }
  chatSessions:  {},          // { model_id -> [{id,title,messages,compact_summary,…}] }
  activeChat:    {},          // { model_id -> session_id }
  llmDrawerOpen: true,
  llmBusy:       false,
  queue:         [],          // [{ id, model_id, params, loras, status }]
  queueRunning:  false,
  lastSeed:      {},          // { model_id -> int } — actual seed used by last generation

  // Auth — the bearer token now lives in an HttpOnly cookie (set by the
  // backend at login/setup) so JS can't read it. That defangs XSS-driven
  // token theft. We only keep the username in localStorage for greeting
  // the user on the unlock screen — non-sensitive.
  auth: {
    token:    null,                              // unused; cookie carries it
    username: readStorage('forge_user', null),
    mode:     'unknown',                         // 'unknown' | 'setup' | 'login' | 'unlock' | 'in'
  },

  // Preview mode — toggled by ?preview in URL. Skips auth, marks every
  // model as ready, fakes generations with a placeholder. Lets us walk
  // through the UI without a backend or any downloaded weights.
  preview: new URLSearchParams(location.search).has('preview'),

  // Phase 3 — browser-side AES-GCM key. Derived from password + salt via
  // PBKDF2; cached in IndexedDB; posted to the service worker so it can
  // decrypt asset GETs transparently. Null when locked.
  dataKey:        null,
  swRegistration: null,
};

// ── Universal param superset ─────────────────────────────────────────────
// Every option the UI knows how to render. Each model declares which
// `param_groups` (and/or `params` overrides) it actually uses; the rest stay
// hidden. This means we can iterate on a model's UI just by editing the
// registry — never the JS.
const PARAM_GROUPS = {
  prompt: {
    title: 'Prompt',
    fields: [
      { key: 'prompt',          label: 'Prompt',          type: 'textarea',
        placeholder: 'Describe what to generate…' },
      { key: 'negative_prompt', label: 'Negative prompt', type: 'textarea',
        placeholder: 'What to avoid (optional)' },
    ],
  },
  generation: {
    title: 'Generation',
    fields: [
      { key: 'seed',      label: 'Seed',      type: 'number', default: -1,   hint: '-1 = random' },
      { key: 'steps',     label: 'Steps',     type: 'slider', min: 1,  max: 100, step: 1,    default: 25 },
      { key: 'cfg',       label: 'CFG',       type: 'slider', min: 0,  max: 20,  step: 0.1,  default: 7 },
      { key: 'cfg_low',   label: 'CFG (low-noise)', type: 'slider', min: 0, max: 20, step: 0.1, default: 5 },
      { key: 'sampler',   label: 'Sampler',   type: 'select',
        options: ['euler', 'euler_a', 'dpmpp_2m', 'dpmpp_3m_sde', 'ddim', 'unipc', 'lcm'],
        default: 'dpmpp_2m' },
      { key: 'scheduler', label: 'Scheduler', type: 'select',
        options: ['normal', 'karras', 'simple', 'sgm_uniform', 'exponential'],
        default: 'karras' },
    ],
  },
  video: {
    title: 'Video',
    fields: [
      { key: 'duration',   label: 'Seconds', type: 'slider', min: 1, max: 5, step: 0.1, default: 3.4 },
      { key: 'fps',        label: 'FPS',     type: 'slider', min: 8, max: 60, step: 1,   default: 24 },
      { key: 'num_frames', label: 'Frames',  type: 'slider', min: 16, max: 121, step: 1, default: 81, hint: 'Advanced' },
    ],
    layout: 'grid-2',
  },
  dimensions: {
    title: 'Dimensions',
    fields: [
      { key: 'width',      label: 'Width',      type: 'number', default: 1024, step: 64, min: 64, max: 4096 },
      { key: 'height',     label: 'Height',     type: 'number', default: 1024, step: 64, min: 64, max: 4096 },
      { key: 'batch_size', label: 'Batch size', type: 'number', default: 1,    step: 1,  min: 1,  max: 16  },
    ],
    layout: 'grid-2',
  },
  reference: {
    title: 'Reference image',
    fields: [
      { key: 'ref_image',    label: 'Reference image', type: 'asset' },
      { key: 'ref_strength', label: 'Strength',        type: 'slider',
        min: 0, max: 1, step: 0.01, default: 0.65 },
    ],
  },
  precision: {
    title: 'Precision & quantization',
    fields: [
      { key: 'precision',    label: 'Precision', type: 'select',
        options: ['fp32', 'fp16', 'bf16', 'fp8'], default: 'bf16' },
      { key: 'quantization', label: 'Quantization', type: 'select',
        options: ['none', 'int8', 'int4', 'awq', 'gptq', 'gguf'], default: 'none' },
      { key: 'offload',      label: 'CPU offload', type: 'bool', default: false },
    ],
  },
  llm: {
    title: 'Inference',
    fields: [
      { key: 'thinking',       label: 'Thinking',        type: 'bool',   default: true },
      { key: 'max_new_tokens', label: 'Output length',   type: 'slider', min: 64, max: 2048, step: 64, default: 512 },
      { key: 'temperature',    label: 'Temperature',     type: 'slider', min: 0,  max: 2,    step: 0.05, default: 0.7 },
      { key: 'top_p',          label: 'Top P',           type: 'slider', min: 0,  max: 1,    step: 0.01, default: 0.9 },
    ],
  },
  audio: {
    title: 'Audio',
    fields: [
      { key: 'compute_type', label: 'Compute type', type: 'select',
        options: ['float16', 'int8_float16', 'int8', 'float32'], default: 'float16' },
      { key: 'language',     label: 'Language',     type: 'select',
        options: ['en','es','fr','de','it','pt','pl','tr','ru','nl','cs','ar','zh'], default: 'en' },
    ],
  },
  server: {
    title: 'Server',
    fields: [
      { key: 'port', label: 'Port', type: 'number', default: 7860, min: 1024, max: 65535 },
    ],
  },
};

// Default group set per category — registry can override per-model.
const DEFAULT_GROUPS_BY_CATEGORY = {
  image: ['prompt', 'generation', 'dimensions', 'reference', 'precision', 'server'],
  video: ['prompt', 'generation', 'video', 'reference', 'precision', 'server'],
  llm:   ['llm'],
  audio: ['audio', 'server'],
};

const PROMPT_BUILDER_TEXT_FIELDS = [
  { key: 'idea',       label: 'Core idea',    placeholder: 'person, product, place, creature...' },
  { key: 'hairStyle',  label: 'Hair style',   placeholder: 'long wavy hair, short bob...' },
  { key: 'hairColor',  label: 'Hair colour',  placeholder: 'black, auburn, platinum blonde...' },
  { key: 'bodyType',   label: 'Body type',    placeholder: 'athletic, slim, broad build...' },
  { key: 'outfit',     label: 'Outfit',       placeholder: 'oversized coat, formal suit...' },
  { key: 'scene',      label: 'Scene',        placeholder: 'neon street, studio backdrop...' },
  { key: 'props',      label: 'Props',        placeholder: 'umbrella, headphones, camera...' },
  { key: 'action',     label: 'Action',       placeholder: 'walking, holding a drink...' },
  { key: 'style',      label: 'Style / light',placeholder: 'natural light, cinematic, editorial...' },
];

const PROMPT_BUILDER_CHIP_GROUPS = [
  { key: 'pose', label: 'Pose', options: ['standing', 'sitting', 'walking', 'kneeling', 'leaning', 'lying down'] },
  { key: 'distance', label: 'Subject distance', options: ['close-up', 'half body', 'full body', 'wide shot'] },
  { key: 'camera', label: 'Camera position', options: ['eye-level', 'low angle', 'high angle', 'overhead', 'front view', '3/4 view', 'side profile', 'back view'] },
  { key: 'head', label: 'Head direction', options: ['looking at camera', 'looking left', 'looking right', 'looking up', 'looking down'] },
  { key: 'bodyAxis', label: 'Body facing', options: ['facing camera', 'turned left', 'turned right', 'back-facing'] },
  { key: 'orientation', label: 'Orientation', options: ['upright subject', 'natural gravity', 'centred subject'] },
];

const DEFAULT_NEGATIVE_PROMPT =
  'extra fingers, extra arms, duplicate limbs, malformed hands, distorted face, bad anatomy, blurry, low quality, watermark, text artifacts, upside down subject';

const loraId = (l) => l?.rel_path || l?.filename || '';

// ── Helpers ──────────────────────────────────────────────────────────────
const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// HTML-escape any string before interpolating into a template literal that
// gets assigned to innerHTML. Use this for ALL strings sourced outside our
// own code: CivitAI fields, user prompts, uploaded filenames, LoRA names,
// HF responses. Numbers + booleans don't need it.
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));

const fmtNum = (n) => {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n ?? 0);
};

function categoryIcon(cat) {
  return ({ image: 'i-image', video: 'i-video', llm: 'i-llm', audio: 'i-audio' })[cat] || 'i-models';
}

function toast(msg, kind = 'info') {
  const c = $('#toasts');
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  const close = kind === 'error'
    ? '<button class="toast-close" type="button" title="Dismiss">×</button>'
    : '';
  el.innerHTML = `<div class="toast-message">${esc(msg)}</div>${close}`;
  c.appendChild(el);
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add('show')));
  const dismiss = () => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 240);
  };
  el.querySelector('.toast-close')?.addEventListener('click', dismiss);
  if (kind !== 'error') setTimeout(dismiss, 2800);
}

// Resolve which groups a model uses, in order
function groupsForModel(m) {
  if (Array.isArray(m.param_groups) && m.param_groups.length) return m.param_groups;
  return DEFAULT_GROUPS_BY_CATEGORY[m.category] || ['server'];
}

// Resolve fields for a model: walk groups, apply per-model overrides
function resolveFields(m) {
  const overrides = m.param_overrides || {};
  const enabledKeys = m.param_keys ? new Set(m.param_keys) : null; // optional whitelist
  const groups = groupsForModel(m);
  const out = [];
  for (const g of groups) {
    const grp = PARAM_GROUPS[g];
    if (!grp) continue;
    const fields = grp.fields
      .filter(f => !enabledKeys || enabledKeys.has(f.key))
      .map(f => ({ ...f, ...(overrides[f.key] || {}) }));
    if (fields.length) out.push({ id: g, title: grp.title, layout: grp.layout, fields });
  }
  // Legacy: if registry still uses flat `params`, append unknown keys as a custom group.
  if (Array.isArray(m.params) && m.params.length) {
    const known = new Set(out.flatMap(s => s.fields.map(f => f.key)));
    const extras = m.params.filter(p => !known.has(p.key));
    if (extras.length) out.push({ id: 'custom', title: 'Model-specific', fields: extras });
  }
  return out;
}

// ── Auth-aware API wrapper ───────────────────────────────────────────────
// The bearer token is in an HttpOnly cookie now; the browser sends it
// automatically with same-origin requests. We just need `credentials:
// "same-origin"` (the default for `fetch` since 2018, but explicit here so
// it's obvious why no Authorization header).
async function apiCall(url, opts = {}) {
  const fetchOpts = { ...opts };
  const timeoutMs = fetchOpts.timeoutMs;
  delete fetchOpts.timeoutMs;
  let timeout = null;
  if (timeoutMs && !fetchOpts.signal) {
    const controller = new AbortController();
    fetchOpts.signal = controller.signal;
    timeout = setTimeout(() => controller.abort(), timeoutMs);
  }
  let res;
  try {
    res = await fetch(url, { credentials: 'same-origin', cache: 'no-store', ...fetchOpts });
  } finally {
    if (timeout) clearTimeout(timeout);
  }
  if (res.status === 401) {
    clearStoredAuth();
    openAuthGate('login', 'Your pod session expired. Sign in again.');
    throw new Error('Unauthorized');
  }
  if (res.status === 423) {
    openAuthGate('unlock', 'The pod restarted. Re-enter your password to unlock encrypted assets.');
    throw new Error('Locked');
  }
  if (!res.ok) {
    // Extract FastAPI's detail message if available, otherwise fall back to status.
    let msg = `Server error ${res.status}`;
    if (res.status === 524) {
      msg = 'The public proxy timed out before the server replied. Long generations now use background jobs; please retry after refreshing the app.';
    }
    try {
      const ct = res.headers.get('content-type') || '';
      if (ct.includes('application/json')) {
        const body = await res.json();
        if (body.detail) msg = String(body.detail);
      }
    } catch {}
    throw new Error(msg);
  }
  return res;
}
const json = async (p) => {
  const r = await p;
  const ct = r.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    throw new Error(`Server error ${r.status}`);
  }
  return r.json();
};
const jsonBody = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

const api = {
  // Auth
  authStatus:    () => json(apiCall('/api/auth/status', { timeoutMs: 3500 })),
  authSetup:     (b) => json(apiCall('/api/auth/setup',  jsonBody(b))),
  authLogin:     (b) => json(apiCall('/api/auth/login',  jsonBody(b))),
  authUnlock:    (b) => json(apiCall('/api/auth/unlock', jsonBody(b))),
  authLogout:    () => json(apiCall('/api/auth/logout', { method: 'POST' })),
  // Core
  models:        () => json(apiCall('/api/models')),
  gpu:           () => json(apiCall('/api/gpu')),
  status:        () => json(apiCall('/api/status')),
  loras:         () => json(apiCall('/api/loras')),
  assets:        () => json(apiCall('/api/assets', { timeoutMs: 8000 })),
  // Lifecycle
  launch:        (b) => json(apiCall('/api/launch', jsonBody(b))),
  stop:          (id) => json(apiCall(`/api/stop/${id}`, { method: 'POST' })),
  runnerHealth:  (id) => json(apiCall(`/api/runner/${id}/healthz`)),
  weightStatus:  (id, variant) => {
    const qs = variant ? `?variant=${encodeURIComponent(variant)}` : '';
    return json(apiCall(`/api/models/${id}/weight-status${qs}`));
  },
  downloadWeights: (id, body) => json(apiCall(`/api/models/${id}/download`, jsonBody(body || {}))),
  // Inference
  generate:      (b) => json(apiCall('/api/generate', jsonBody(b))),
  startGenerateJob: (b) => json(apiCall('/api/generate-jobs', jsonBody(b))),
  generationJob: (id) => json(apiCall(`/api/generate-jobs/${encodeURIComponent(id)}`)),
  enhancePrompt: (b) => json(apiCall('/api/prompt/enhance', jsonBody(b))),
  cancel:        (id) => json(apiCall(`/api/cancel/${id}`, { method: 'POST' })),
  // Mutations
  deleteModel:   (id)   => json(apiCall(`/api/models/${id}`, { method: 'DELETE' })),
  deleteLora:    (fn)   => json(apiCall(`/api/loras/${encodeURIComponent(fn)}`, { method: 'DELETE' })),
  patchLora:     (fn, body) => json(apiCall(`/api/loras/${encodeURIComponent(fn)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })),
  installLoras:  (b) => json(apiCall('/api/loras/install', jsonBody(b))),
  components:           () => json(apiCall('/api/components')),
  installComponents:    (b)  => json(apiCall('/api/components/install', jsonBody(b))),
  componentJobStatus:   (id) => json(apiCall(`/api/components/install-status/${encodeURIComponent(id)}`)),
  runtimeStatus:        (mid)      => json(apiCall(`/api/runtime/${encodeURIComponent(mid)}/status`)),
  installRuntime:       (mid)      => json(apiCall(`/api/runtime/${encodeURIComponent(mid)}/install`, { method: 'POST' })),
  runtimeJobStatus:     (mid, jid) => json(apiCall(`/api/runtime/${encodeURIComponent(mid)}/install-status/${encodeURIComponent(jid)}`)),
  // HF browser + downloads. Replaces the old huggingface-cli subprocess
  // path which deadlocked on a blocked stdout pipe.
  hfFiles:     (repo, revision, hfToken) =>
    json(apiCall(`/api/hf/files?repo=${encodeURIComponent(repo)}&revision=${encodeURIComponent(revision || 'main')}${hfToken ? '&hf_token=' + encodeURIComponent(hfToken) : ''}`)),
  hfDownload:  (b)  => json(apiCall('/api/hf/download', jsonBody(b))),
  hfJobs:      ()   => json(apiCall('/api/hf/jobs')),
  hfJobCancel: (id) => json(apiCall(`/api/hf/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' })),
  hfJobRetry:  (id) => json(apiCall(`/api/hf/jobs/${encodeURIComponent(id)}/retry`, { method: 'POST' })),
  hfLoraImportDone: (b) => json(apiCall('/api/hf/import-loras/done', jsonBody(b))),
  deleteAsset:   (path) => json(apiCall(`/api/assets?path=${encodeURIComponent(path)}`, { method: 'DELETE' })),
  // Phase 3: encrypt the file in the browser BEFORE upload when we have the
  // data key. Backend stores the bytes as-is (sees ciphertext, period).
  // Falls back to plaintext upload only when the browser is locked — the
  // backend will then either encrypt server-side (Phase 2 fallback) or
  // store plaintext (dev mode).
  uploadAsset:   async (file) => {
    const fd = new FormData();
    const headers = {};
    if (state.dataKey) {
      const ctBlob = await encryptBlob(state.dataKey, file);
      // Preserve the original filename (with original extension) so the
      // backend's `<name>.enc` naming + MIME-by-extension still works.
      fd.append('file', ctBlob, file.name);
      headers['X-Forge-Encrypted'] = '1';
    } else {
      fd.append('file', file);
    }
    return json(apiCall('/api/assets/upload', {
      method: 'POST',
      body: fd,
      headers,
    }));
  },
  civSearch:     (q, key) => {
    const p = new URLSearchParams({ query: q, types: 'LORA', limit: 24 });
    if (key) p.append('api_key', key);
    return json(apiCall(`/api/civitai/search?${p}`));
  },
  civModel:      (id, key) => {
    const p = new URLSearchParams();
    if (key) p.append('api_key', key);
    const qs = p.toString() ? `?${p}` : '';
    return json(apiCall(`/api/civitai/model/${id}${qs}`));
  },
  civDownload:   (b) => json(apiCall('/api/civitai/download', jsonBody(b))),
};

// ── Boot ─────────────────────────────────────────────────────────────────
async function init() {
  bindShell();
  bindAuth();
  applySettingsToInputs();
  if (state.preview) {
    // Skip auth entirely; jump straight to the main UI.
    state.auth.token    = 'preview';
    state.auth.username = 'preview';
    state.auth.mode     = 'in';
    await bootApp();
    toast('Preview mode — generation is faked', 'info');
    return;
  }
  const authFallback = setTimeout(() => {
    if (state.auth.mode === 'unknown') {
      openAuthGate('login', 'Connecting to the pod…');
    }
  }, 1500);
  try {
    await checkAuth();
  } finally {
    clearTimeout(authFallback);
  }
}

// Called after a successful sign-in / setup — loads the actual app data.
async function bootApp() {
  hideAuth();
  // Reveal the app shell. Hidden by default so it doesn't flash behind the
  // auth screen during the brief moment between HTML render and the auth
  // check completing.
  const appEl = $('.app');
  if (appEl) {
    appEl.style.display = '';
    appEl.classList.add('ready');
  }
  await ensureServiceWorker();   // register SW so <img> requests get decrypted
  await primeBrowserKey();       // hand the IDB-cached CryptoKey to the SW
  await Promise.all([loadModels(), loadLoras(), loadAssets(), loadComponents()]);
  const initialView = window.location.hash.replace('#', '');
  const canRestoreView = initialView
    && initialView !== 'workspace'
    && $(`.view[data-view="${CSS.escape(initialView)}"]`);
  if (canRestoreView) switchView(initialView, false);
  pollRunning();
}

// ── Browser-side AES key lifecycle ──────────────────────────────────────
//
// `state.dataKey` holds the non-extractable CryptoKey used for both upload
// encryption and the SW's view-time decryption. Source of truth is
// IndexedDB (survives page refresh + tab close). We post the key into the
// service worker so it can transparently decrypt /api/assets/file/* GETs.

async function ensureServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  try {
    const reg = await navigator.serviceWorker.register('/sw.js', { scope: '/', updateViaCache: 'none' });
    try { await reg.update(); } catch {}
    // Wait for it to be controlling the page so the first asset GET goes
    // through. Without this, the very first <img> on a fresh tab can race
    // and bypass decryption.
    if (!navigator.serviceWorker.controller) {
      await new Promise((resolve) => {
        const onCtrl = () => {
          navigator.serviceWorker.removeEventListener('controllerchange', onCtrl);
          resolve();
        };
        navigator.serviceWorker.addEventListener('controllerchange', onCtrl);
        // 2s safety so we don't hang forever if SW install fails
        setTimeout(resolve, 2000);
      });
    }
    state.swRegistration = reg;
  } catch (e) {
    // SW unavailable (file:// or older browser) — encrypted asset views
    // will 503 from the SW path, but everything else still works.
    console.warn('Service worker registration failed:', e);
  }
}

async function setBrowserKey(key) {
  state.dataKey = key;
  try { await storeKey(key); } catch {}
  await postKeyToServiceWorker(key);
}

async function clearBrowserKey() {
  state.dataKey = null;
  try { await clearKey(); } catch {}
  await postKeyToServiceWorker(null);
}

async function primeBrowserKey() {
  // On boot, restore the key from IDB if we have one, and post it to the
  // SW. If IDB is empty, leave state.dataKey null — the unlock prompt path
  // will derive + store a new one.
  try {
    const saved = await loadKey();
    if (saved) {
      state.dataKey = saved;
      await postKeyToServiceWorker(saved);
    }
  } catch {}
}

async function postKeyToServiceWorker(key) {
  const ctl = navigator.serviceWorker?.controller;
  if (!ctl) return;
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => resolve();
    if (key) ctl.postMessage({ type: 'set-key',   key }, [channel.port2]);
    else     ctl.postMessage({ type: 'clear-key' },     [channel.port2]);
    setTimeout(resolve, 1000);   // fail-open after a beat
  });
}

// ── Auth flow ────────────────────────────────────────────────────────────
function authModeForStatus(status, browserKey = null) {
  if (!status) return 'login';
  if (status.needs_setup) return 'setup';
  if (status.authenticated && (status.locked || !browserKey)) return 'unlock';
  return 'login';
}

function openAuthGate(mode, message = '') {
  const shell = $('#authShell');
  const alreadyOpen = shell?.classList.contains('open') && state.auth.mode === mode;
  state.auth.mode = mode;
  if (!alreadyOpen) showAuth(mode);
  if (message) $('#authError').textContent = message;
}

async function checkAuth() {
  // The HttpOnly cookie carries the token. JS can't read it, so we ask the
  // backend whether *this* request is authenticated (the cookie is sent
  // automatically) plus whether setup/unlock screens are needed.
  let status = null;
  try { status = await api.authStatus(); } catch {}

  // Even with a valid cookie + non-locked backend, we ALSO need the
  // browser-side AES key in IDB (Phase 3). If it's missing we have to
  // prompt for the password to re-derive it, even though the backend
  // happily lets us through.
  let browserKey = null;
  try { browserKey = await loadKey(); } catch {}

  // Authenticated + unlocked + we have the key → boot the app.
  if (status?.authenticated && !status.locked && browserKey) {
    state.dataKey = browserKey;
    state.auth.mode = 'in';
    state.auth.username = status.username || state.auth.username;
    await bootApp();
    return;
  }

  // Pick the right screen. If we have a session but no browser key, we're
  // in "browser unlock" — same UI as backend unlock (just a password).
  if (!status) {
    openAuthGate('login', 'Could not reach the pod. Refresh once it is ready.');
    return;
  }
  const mode = authModeForStatus(status, browserKey);
  state.auth.username = status?.username || state.auth.username;
  openAuthGate(mode);
}

async function enforceLiveAuthGate() {
  if (state.preview || state.auth.mode !== 'in') return false;
  let status = null;
  try { status = await api.authStatus(); } catch { return false; }

  state.auth.username = status?.username || state.auth.username;
  if (status.needs_setup) {
    await clearBrowserKey();
    openAuthGate('setup', 'The pod auth store changed. Set up access again.');
    return true;
  }
  if (!status.authenticated) {
    await clearBrowserKey();
    openAuthGate('login', 'Your pod session expired. Sign in again.');
    return true;
  }
  if (status.locked) {
    await clearBrowserKey();
    openAuthGate('unlock', 'The pod restarted. Re-enter your password to unlock encrypted assets.');
    return true;
  }
  return false;
}

function clearStoredAuth() {
  // Cookie is HttpOnly — JS can't delete it. Best we can do is ask the
  // backend (fire-and-forget). Also dump the browser-side key from IDB +
  // SW so subsequent asset views fail closed instead of leaking through
  // a still-resident CryptoKey.
  state.auth.username = null;
  localStorage.removeItem('forge_user');
  try { api.authLogout().catch(() => {}); } catch {}
  try { clearBrowserKey().catch(() => {}); } catch {}
}

function showAuth(mode = state.auth.mode) {
  // Three variants:
  //   setup  — first-time user creates credentials.
  //   login  — credentials exist on disk; user signs in normally.
  //   unlock — session token still valid, but the encryption key isn't in
  //            RAM (post-restart). Just need the password to re-derive it.
  const isSetup  = mode === 'setup';
  const isUnlock = mode === 'unlock';
  const titles = {
    setup:  ['Set up access',                  'Create credentials for this pod. Encrypts your assets at rest.'],
    login:  ['Sign in',                        'Enter the credentials you set when this pod started.'],
    unlock: [`Welcome back${state.auth.username ? ', ' + state.auth.username : ''}`,
             'Re-enter your password to unlock your encrypted assets in this browser.'],
  };
  const [title, sub] = titles[mode] || titles.login;
  $('#authTitle').textContent = title;
  $('#authSub').textContent   = sub;

  // Username field: hidden in unlock mode (we know who they are), shown otherwise.
  $('#authUsername').style.display = isUnlock ? 'none' : '';
  if (isUnlock && state.auth.username) $('#authUsername').value = state.auth.username;
  $('#authConfirm').style.display = isSetup ? '' : 'none';
  $('#authSubmit').textContent =
    isSetup  ? 'Continue' :
    isUnlock ? 'Unlock' :
               'Sign in';
  $('#authError').textContent = '';
  $('#authShell').classList.add('open');
  $('#authShell').setAttribute('aria-hidden', 'false');
  setTimeout(() => (isUnlock ? $('#authPassword') : $('#authUsername')).focus(), 50);
}

function hideAuth() {
  $('#authShell').classList.remove('open');
  $('#authShell').setAttribute('aria-hidden', 'true');
}

function bindAuth() {
  $('#authForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await submitAuth();
  });
}

async function submitAuth() {
  const mode     = state.auth.mode;
  const isUnlock = mode === 'unlock';
  const username = isUnlock
    ? state.auth.username
    : $('#authUsername').value.trim();
  const password = $('#authPassword').value;
  const confirm  = $('#authConfirm').value;
  const errEl    = $('#authError');
  errEl.textContent = '';

  if (!password || (!isUnlock && !username)) {
    errEl.textContent = isUnlock
      ? 'Password is required.'
      : 'Username and password are required.';
    return;
  }
  if (mode === 'setup' && password !== confirm) {
    errEl.textContent = 'Passwords don\'t match.';
    return;
  }
  if (mode === 'setup' && password.length < 4) {
    errEl.textContent = 'Pick a password with at least 4 characters.';
    return;
  }

  const btn = $('#authSubmit');
  btn.disabled = true;
  const labelOf = (m) => m === 'setup' ? 'Continue' : m === 'unlock' ? 'Unlock' : 'Sign in';
  btn.innerHTML = '<span class="spinner"></span><span>Working…</span>';

  try {
    // The auth response carries `salt`, `canary_ct`, and `iterations` so
    // the browser can derive the SAME AES key the backend would. Both the
    // setup/login (`api.authSetup` / `api.authLogin`) and the unlock path
    // return them via the new _key_material() helper.
    let res;
    if (isUnlock) {
      res = await api.authUnlock({ password });
    } else {
      const fn = mode === 'setup' ? api.authSetup : api.authLogin;
      res = await fn({ username, password });
      if (res.detail || res.message) throw new Error(res.detail || res.message);
      state.auth.username = username;
      localStorage.setItem('forge_user', username);
    }
    if (res.username) {
      state.auth.username = res.username;
      localStorage.setItem('forge_user', res.username);
    }

    // Derive the browser-side key from password + salt and verify it
    // against the stored canary (guards against backend-supplied bad salt).
    if (res.salt) {
      const salt = hexToBytes(res.salt);
      const iterations = res.iterations || PBKDF2_DEFAULT_ITERATIONS;
      const key = await deriveKey(password, salt, iterations);
      const canaryOk = await verifyCanary(key, res.canary_ct);
      if (!canaryOk) throw new Error('Canary verification failed — wrong password?');
      await ensureServiceWorker();
      await setBrowserKey(key);
    }

    state.auth.mode = 'in';
    await bootApp();
  } catch (e) {
    errEl.textContent = e.message || 'Auth failed';
  } finally {
    btn.disabled = false;
    btn.textContent = labelOf(state.auth.mode === 'in' ? mode : state.auth.mode);
  }
}

// ── App shell wiring ─────────────────────────────────────────────────────
function bindShell() {
  // Bind to nav items + topbar icon buttons only — NOT the view sections
  // themselves, which also carry data-view as a marker. Clicking inside a
  // section was bubbling up and re-switching to the same view (which broke
  // the workspace back button: closeWorkspace switched to 'models', then
  // the bubble up the section reset it to 'workspace').
  $$('[data-view]:not(.view)').forEach(el => el.addEventListener('click', () => switchView(el.dataset.view)));

  $$('.chip[data-filter]').forEach(c => c.addEventListener('click', () => {
    state.filter = c.dataset.filter;
    $$('.chip[data-filter]').forEach(x => x.classList.toggle('active', x === c));
    renderModels();
  }));
  $('#modelSearch').addEventListener('input', renderModels);

  $$('.chip[data-asset-filter]').forEach(c => c.addEventListener('click', () => {
    state.assetFilter = c.dataset.assetFilter;
    $$('.chip[data-asset-filter]').forEach(x => x.classList.toggle('active', x === c));
    renderAssets();
  }));

  $('#drawerScrim').addEventListener('click', closeDrawer);
  $('#drawerClose').addEventListener('click', closeDrawer);
  $('#drawerPrimary').addEventListener('click', onDrawerPrimary);

  // Workspace controls
  $('#wsBack').addEventListener('click', closeWorkspace);
  $('#wsStop').addEventListener('click', async () => {
    const id = state.workspace;
    if (!id) return;
    if (!await confirmModal('Stop Runner', 'Stop the runner? Weights will stay cached for next time.', 'Stop', true)) return;
    toast('Stopping…', 'info');
    try {
      await api.stop(id);
      markModelStopped(id);
      toast('Runner stopped', 'success');
    } catch (e) {
      toast(`Stop failed: ${e.message || 'unknown'}`, 'error');
    }
  });
  $('#wsGenerate').addEventListener('click', onGenerate);
  $('#wsCancel').addEventListener('click', cancelGenerate);
  $('#wsEnhancePrompt')?.addEventListener('click', enhanceWorkspacePrompt);
  $('#wsPromptModeBuilder')?.addEventListener('click', () => setPromptMode('builder'));
  $('#wsPromptModeRaw')?.addEventListener('click', () => setPromptMode('raw'));
  $('#wsLoraAdd').addEventListener('click', () => {
    if (state.selected) openLoraPicker(state.selected);
  });
  $('#llmSend').addEventListener('click', sendLLMMessage);
  $('#llmPrompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendLLMMessage();
    }
  });
  $('#llmSessionsToggle').addEventListener('click', () => {
    state.llmDrawerOpen = !state.llmDrawerOpen;
    if (state.selected?.category === 'llm') renderLLMSessions(state.selected);
  });
  $('#llmSessionsClose').addEventListener('click', () => {
    state.llmDrawerOpen = false;
    if (state.selected?.category === 'llm') renderLLMSessions(state.selected);
  });
  $('#llmNewSession').addEventListener('click', () => {
    if (!state.selected || state.selected.category !== 'llm') return;
    createChatSession(state.selected.id, true);
    renderLLMWorkspace(state.selected, resolveFields(state.selected));
  });

  // Lightbox
  $('#lightboxClose').addEventListener('click', closeLightbox);
  $('#lightbox').addEventListener('click', (e) => { if (e.target.id === 'lightbox') closeLightbox(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if ($('#lightbox').classList.contains('open')) closeLightbox();
      if ($('#assetPop').classList.contains('open')) closeAssetMenu();
    }
    // Arrow keys cycle through the lightbox list when it's open.
    if ($('#lightbox').classList.contains('open')) {
      if (e.key === 'ArrowRight') { e.preventDefault(); _lightboxStep(1);  }
      if (e.key === 'ArrowLeft')  { e.preventDefault(); _lightboxStep(-1); }
    }
  });
  // Close the asset 3-dot menu when clicking anywhere outside it.
  document.addEventListener('click', (e) => {
    const pop = $('#assetPop');
    if (!pop.classList.contains('open')) return;
    if (pop.contains(e.target) || e.target.closest('[data-asset-menu]')) return;
    closeAssetMenu();
  });

  $('#modalScrim').addEventListener('click', (e) => { if (e.target.id === 'modalScrim') closeModal(); });
  $('#modalClose').addEventListener('click', closeModal);

  $('#civSearchBtn').addEventListener('click', civSearch);
  $('#civSearch').addEventListener('keydown', (e) => { if (e.key === 'Enter') civSearch(); });
  $('#civUrlGo').addEventListener('click', civOpenByUrl);
  $('#civUrl').addEventListener('keydown', (e) => { if (e.key === 'Enter') civOpenByUrl(); });

  $('#hfDownloadBtn')?.addEventListener('click', startHFDownload);
  $('#hfBrowseBtn')?.addEventListener('click', () => openHFBrowser());
  $('#hfRepo')?.addEventListener('keydown', (e) => { if (e.key === 'Enter') openHFBrowser(); });
  // HF Browser modal wiring. Each handler is null-safe so the modal works
  // even before any tab navigation has touched the markup.
  $('#hfBrowserClose')?.addEventListener('click', closeHFBrowser);
  $('#hfBrowserScrim')?.addEventListener('click', (e) => { if (e.target === e.currentTarget) closeHFBrowser(); });
  $('#hfBrowserGo')?.addEventListener('click', loadHFFiles);
  $('#hfBrowserRepo')?.addEventListener('keydown', (e) => { if (e.key === 'Enter') loadHFFiles(); });
  $('#hfBrowserFilter')?.addEventListener('input', (e) => {
    state.hfBrowse.filter = e.target.value;
    renderHFBrowserBody();
  });
  $('#hfBrowserSelectAll')?.addEventListener('click', () => {
    const filt = (state.hfBrowse.filter || '').toLowerCase();
    state.hfBrowse.files
      .filter(f => !filt || f.rel_path.toLowerCase().includes(filt))
      .forEach(f => state.hfBrowse.selected.add(f.rel_path));
    renderHFBrowserBody();
    updateHFBrowserSummary();
  });
  $('#hfBrowserClear')?.addEventListener('click', () => {
    state.hfBrowse.selected = new Set();
    renderHFBrowserBody();
    updateHFBrowserSummary();
  });
  $('#hfBrowserDownload')?.addEventListener('click', submitHFBrowserDownload);
  $('#dlQueueClear')?.addEventListener('click', clearCompletedDownloads);
  ['#settingsHFToken', '#wsHFToken', '#hfTokenDl'].forEach(sel => {
    $(sel)?.addEventListener('change', e => saveHFToken(e.target.value));
  });

  $$('[data-save]').forEach(b => b.addEventListener('click', () => {
    const key = b.dataset.save;
    const val = $('#' + b.dataset.source).value;
    state.settings[key] = val;
    localStorage.setItem('forge_settings', JSON.stringify(state.settings));
    if (key === 'hf_token') syncHFTokenInputs();
    toast('Saved', 'success');
  }));

  $('#uploadBtn').addEventListener('click', () => $('#uploadInput').click());
  $('#uploadZone').addEventListener('click', () => $('#uploadInput').click());
  $('#uploadInput').addEventListener('change', (e) => uploadFiles([...e.target.files]));
  const dz = $('#uploadZone');
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('drag');
    uploadFiles([...e.dataTransfer.files]);
  });
}

function applySettingsToInputs() {
  if (state.settings.hf_token)    $('#settingsHFToken').value = state.settings.hf_token;
  if (state.settings.civitai_key) $('#settingsCivKey').value  = state.settings.civitai_key;
  syncHFTokenInputs();
}

function saveHFToken(value) {
  const token = (value || '').trim();
  state.settings.hf_token = token;
  localStorage.setItem('forge_settings', JSON.stringify(state.settings));
  syncHFTokenInputs();
  return token;
}

function currentHFToken(selector) {
  const input = selector ? $(selector) : null;
  const token = input?.value?.trim() || state.settings.hf_token || '';
  if (input?.value?.trim()) saveHFToken(input.value);
  return token;
}

function syncHFTokenInputs() {
  const token = state.settings.hf_token || '';
  ['#settingsHFToken', '#drawerHFToken', '#wsHFToken', '#hfTokenDl'].forEach(sel => {
    const el = $(sel);
    if (el && el.value !== token) el.value = token;
  });
}

// ── Views ────────────────────────────────────────────────────────────────
function switchView(name, push = true) {
  if (name !== 'workspace') {
    $('.main')?.classList.remove('llm-open');
    $('[data-view="workspace"]')?.classList.remove('llm-mode');
    const toggle = $('#llmSessionsToggle');
    if (toggle) toggle.style.display = 'none';
  }
  if (push) {
    const url = new URL(window.location);
    url.hash = name === 'models' ? '' : name;
    history.pushState({ view: name, workspace: state.workspace }, '', url);
  }
  $$('.view').forEach(v => v.classList.toggle('active', v.dataset.view === name));
  $$('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === name));
  state.view = name;
  if (name === 'loras')   loadLoras();
  if (name === 'assets')  loadAssets();
  if (name === 'running') renderRunning();
  if (name === 'downloads') {
    // Make the queue panel visible (or render the empty state) and wake the
    // poller so freshly-queued jobs appear right away when the user lands
    // back on the Downloads view.
    pokeDownloadQueuePoll();
  } else {
    // Repaint so any active jobs that should be hidden on other views are
    // taken down — the panel is still scoped to the Downloads view today.
    renderDownloadQueue();
  }
}

window.addEventListener('popstate', (e) => {
  if (e.state && e.state.view) {
    if (e.state.workspace) {
      // Re-open workspace if we are going back to it
      openWorkspace(e.state.workspace, false);
    } else {
      state.workspace = null;
      switchView(e.state.view, false);
    }
  } else {
    state.workspace = null;
    const hash = window.location.hash.replace('#', '');
    switchView(hash || 'models', false);
  }
});

// ── Models ───────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const data = await api.models();
    state.models    = data.models;
    state.upscalers = data.upscalers || [];
    state.gpu       = data.gpu;
    renderGPU(data.gpu);
    renderModels();
    $('#modelCount').textContent = state.models.length;
    $('#gpuDebug').textContent   = JSON.stringify(data.gpu, null, 2);
  } catch (e) {
    if (state.preview) {
      await loadPreviewRegistry();
    } else {
      state.models    = mockModels();
      state.upscalers = [];
      renderModels();
      $('#modelCount').textContent = state.models.length;
      $('#gpuPill').innerHTML = '<div class="dot"></div><span>Dev mode (no backend)</span>';
    }
  }
  if (state.preview) {
    // Pretend everything is downloaded + running so cards click through to
    // the workspace and the drawer would say "Open workspace" if you visit it.
    for (const m of state.models) {
      state.modelStates[m.id] = { downloaded: true, ready: true };
    }
    renderModels();
  }
}

async function loadPreviewRegistry() {
  try {
    const res = await fetch('/preview_registry.static.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`preview registry ${res.status}`);
    const data = await res.json();
    state.models    = data.models || [];
    state.upscalers = data.upscalers || [];
    state.gpu       = { type: 'nvidia', name: 'Preview GPU', vram_gb: 80 };
    renderGPU(state.gpu);
    $('#modelCount').textContent = state.models.length;
    $('#gpuDebug').textContent   = JSON.stringify(state.gpu, null, 2);
    renderModels();
  } catch (err) {
    state.models    = mockModels();
    state.upscalers = [];
    state.gpu       = { type: 'unknown', name: 'Preview mode', vram_gb: 0 };
    $('#modelCount').textContent = state.models.length;
    $('#gpuPill').innerHTML = '<div class="dot"></div><span>Preview fallback models</span>';
    renderModels();
    console.warn('Preview registry unavailable, using mockModels()', err);
  }
}

function renderGPU(gpu) {
  const pill = $('#gpuPill');
  pill.classList.toggle('online', gpu.type !== 'cpu' && gpu.type !== 'unknown');
  const name = gpu.name || gpu.type || 'Unknown';
  const vram = gpu.vram_gb ? `${gpu.vram_gb} GB` : '';
  pill.innerHTML = `<div class="dot"></div><span>${name}${vram ? ' · ' + vram : ''}</span>`;
}

function modelVramValues(m) {
  const values = [];
  if (Number.isFinite(Number(m.min_vram_gb))) values.push(Number(m.min_vram_gb));
  if (Number.isFinite(Number(m.recommended_vram_gb))) values.push(Number(m.recommended_vram_gb));
  for (const q of (m.quants || [])) {
    if (Number.isFinite(Number(q.vram_gb))) values.push(Number(q.vram_gb));
  }
  for (const v of (m.variants || [])) {
    for (const q of (v.quants || [])) {
      if (Number.isFinite(Number(q.vram_gb))) values.push(Number(q.vram_gb));
    }
  }
  return values;
}

function modelVramProfile(m) {
  const values = modelVramValues(m);
  const min = Number(m.min_vram_gb) || (values.length ? Math.min(...values) : 0);
  const max = values.length ? Math.max(...values) : (Number(m.recommended_vram_gb) || min);
  return {
    min,
    max,
    label: max && max !== min ? `${min}–${max} GB` : `${min}+ GB`,
  };
}

function renderModels() {
  const grid = $('#modelGrid');
  const search = $('#modelSearch').value.toLowerCase();
  const filtered = state.models.filter(m => {
    const cat = state.filter === 'all' || m.category === state.filter;
    const q   = !search || m.name.toLowerCase().includes(search) || (m.description || '').toLowerCase().includes(search);
    return cat && q;
  });
  if (!filtered.length) {
    grid.innerHTML = `<div class="empty" style="grid-column:1/-1">No models match.</div>`;
    return;
  }
  const gpu = state.gpu;
  grid.innerHTML = filtered.map(m => {
    const vram = gpu?.vram_gb || 0;
    const vramProfile = modelVramProfile(m);
    const phase = drawerPhase(m.id);
    let btnText = 'Configure';
    if (phase === 'ready') btnText = 'Open';
    else if (phase === 'downloaded') btnText = 'Start';
    const pct = gpu && vramProfile.max ? Math.min(100, (vram / Math.max(vramProfile.max, 1)) * 100) : 50;
    const vramClass = !gpu ? '' : vram >= vramProfile.max ? '' : vram >= vramProfile.min ? 'warn' : 'bad';
    const vramTag = !gpu ? '' : vram >= vramProfile.max
      ? `<span class="tag solid">✓ ${vramProfile.max} GB</span>`
      : vram >= vramProfile.min
        ? `<span class="tag warn">${vramProfile.min} GB min</span>`
        : `<span class="tag bad">need ${vramProfile.min} GB</span>`;
    const gpuOk = !gpu || gpu.type === 'cpu' || (m.gpu_support || []).includes(gpu.type);
    const dim = gpu && !gpuOk;
    return `
      <article class="model-card${dim ? ' dim' : ''}" data-id="${m.id}">
        <div class="mc-head">
          <div class="mc-icon ${m.category}"><svg><use href="#${categoryIcon(m.category)}"/></svg></div>
          <div class="mc-meta">
            <div class="mc-name">${m.name}</div>
            <div class="mc-desc">${m.description || ''}</div>
          </div>
          <button class="mc-menu" data-action="menu" data-id="${m.id}" title="Actions"><svg style="width:14px;height:14px"><use href="#i-dots"/></svg></button>
        </div>
        <div class="mc-tags">
          ${vramTag}
          ${m.supports_lora ? '<span class="tag">LoRA</span>' : ''}
          ${dim ? '<span class="tag bad">GPU mismatch</span>' : ''}
        </div>
        <div class="vram-bar"><div class="vram-fill ${vramClass}" style="width:${Math.min(pct,100)}%"></div></div>
        <div class="mc-foot">
          <span class="mc-vram">${vramProfile.label}</span>
          <button class="btn sm" data-action="configure" data-id="${m.id}" ${dim ? 'disabled' : ''}>${btnText}</button>
        </div>
      </article>`;
  }).join('');

  const onModelOpen = (id) => {
    // If the runner is already up, skip the drawer and jump straight to the workspace.
    if (modelStateOf(id).ready) openWorkspace(id);
    else                        openDrawer(id);
  };
  grid.querySelectorAll('[data-action="configure"]').forEach(b =>
    b.addEventListener('click', (e) => { e.stopPropagation(); onModelOpen(b.dataset.id); }));
  grid.querySelectorAll('.model-card').forEach(c =>
    c.addEventListener('click', (e) => {
      if (e.target.closest('[data-action]')) return;
      onModelOpen(c.dataset.id);
    }));
  grid.querySelectorAll('[data-action="menu"]').forEach(b =>
    b.addEventListener('click', (e) => { e.stopPropagation(); openModelMenu(b.dataset.id); }));
}

function openModelMenu(id) {
  const m = state.models.find(x => x.id === id);
  if (!m) return;
  $('#modalTitle').textContent = m.name;
  $('#modalMeta').textContent  = m.hf_repo || '';
  $('#modalBody').innerHTML = `
    <div style="font-size:13px;color:var(--t-2);line-height:1.55">
      Manage downloaded weights for this model. Deleting frees disk on the pod —
      it will be re-downloaded next launch.
    </div>`;
  $('#modalFoot').innerHTML = `
    <button class="btn ghost" id="modalCancel">Close</button>
    <button class="btn danger" id="modalDelete"><svg><use href="#i-trash"/></svg>Delete weights</button>`;
  $('#modalCancel').onclick = closeModal;
  $('#modalDelete').onclick = async () => {
    const mName = m.name;
    const mId = m.id;
    closeModal();
    if (!await confirmModal('Delete Weights', `Delete cached weights for ${mName}? This cannot be undone.`, 'Delete', true)) return;
    try {
      if (state.running[mId] || modelStateOf(mId).ready || modelStateOf(mId).starting) {
        try { await api.stop(mId); } catch {}
        markModelStopped(mId);
      }
      const res = await api.deleteModel(mId);
      if (res.status === 'deleted' || res.ok) {
        resetModelWeightsState(mId);
        // Show freed disk so the user knows the volume actually shrank.
        // freed_mb of 0 is suspicious — flag it instead of pretending success.
        const freed = res.freed_mb ?? 0;
        toast(freed > 0 ? `Weights removed · ${freed.toLocaleString()} MB freed` : 'Weights removed (nothing to delete)', 'success');
        closeModal();
      } else if (res.status === 'partial') {
        const freed = res.freed_mb ?? 0;
        toast(`Partial delete · ${freed.toLocaleString()} MB freed · ${res.errors?.[0] || 'see logs'}`, 'error');
        resetModelWeightsState(mId);
        closeModal();
      } else {
        toast(res.message || 'Delete failed', 'error');
      }
    } catch (e) { toast(`Delete failed: ${e.message || 'unknown'}`, 'error'); }
  };
  openModal();
}

// ── Drawer (configure & launch) ──────────────────────────────────────────
// ── Drawer (info + state-driven primary button) ──────────────────────────
//
// State machine per model:
//   not_downloaded   → primary: "Download weights"
//   downloading      → primary: spinner "Downloading…"
//   downloaded       → primary: "Start runner"
//   starting         → primary: spinner "Starting…"
//   ready            → primary: "Open workspace"
//   error            → primary: retry of whatever stage failed
function modelStateOf(id) {
  return state.modelStates[id] || { downloaded: false, loading: false, ready: false };
}

function setModelState(id, patch) {
  state.modelStates[id] = { ...modelStateOf(id), ...patch };
}

function markModelStopped(id) {
  delete state.running[id];
  clearWorkspaceMemory(id);
  setModelState(id, { ready: false, starting: false, generating: false });
  if (state.workspace === id) updateWsStatus();
  if (state.selected?.id === id) renderDrawerBody();
  renderQueue();
  renderModels();
  if ($('.view.active')?.dataset.view === 'running') renderRunning();
}

function resetModelWeightsState(id) {
  delete state.running[id];
  clearWorkspaceMemory(id);
  setModelState(id, {
    downloaded: false,
    downloading: false,
    ready: false,
    starting: false,
    generating: false,
    error: null,
    progress: 0,
  });
  if (state.workspace === id) updateWsStatus();
  if (state.selected?.id === id) renderDrawerBody();
  renderQueue();
  renderModels();
  if ($('.view.active')?.dataset.view === 'running') renderRunning();
}

function clearWorkspaceMemory(id) {
  delete state.workspaceParams[id];
  delete state.imgInputs[id];
  delete state.loraStates[id];
  delete state.recents[id];
  delete state.lastSeed[id];
  persistLoraStates();
  state.queue = state.queue.filter(j => j.model_id !== id);
  if (state.workspace === id) {
    const m = state.models.find(x => x.id === id);
    if (m) state.params = state.workspaceParams[id] = defaultParamsForModel(m);
  }
}

function persistLoraStates() {
  const out = {};
  for (const [modelId, entries] of Object.entries(state.loraStates || {})) {
    out[modelId] = {};
    for (const [key, value] of Object.entries(entries || {})) {
      out[modelId][key] = {
        on: !!value.on,
        strength: Number(value.strength ?? 1.0),
        strength_high: Number(value.strength_high ?? value.strength ?? 1.0),
        strength_low: Number(value.strength_low ?? value.strength ?? 1.0),
      };
    }
  }
  writeJSONStorage('forge_lora_states', out);
}

// Pick the highest-quality variant the GPU can run. Variants are model
// sizes (14B vs 5B for Wan etc.); within each variant there's a quant
// list. Returns the variant object or null if the model isn't variant-based.
function pickAutoVariant(m, vramGb) {
  const variants = m.variants || [];
  if (!variants.length) return null;
  for (const v of variants) {
    if (vramGb >= (v.auto_min_vram_gb ?? 0)) return v;
  }
  return [...variants].sort((a, b) => (a.auto_min_vram_gb ?? 0) - (b.auto_min_vram_gb ?? 0))[0];
}

// Resolve the current variant — explicit user choice from drawer state, or
// the auto-pick. Returns null for non-variant models.
function selectedVariant(m) {
  const variants = m.variants || [];
  if (!variants.length) return null;
  const saved = modelStateOf(m.id).variant || 'auto';
  if (saved !== 'auto') {
    const found = variants.find(v => v.id === saved);
    if (found) return found;
  }
  return pickAutoVariant(m, state.gpu?.vram_gb || 0);
}

function selectedVariantId(m) {
  return selectedVariant(m)?.id;
}

// Pick the highest-quality quant the detected GPU can comfortably run.
// For variant-based models, the quant list comes from the resolved variant.
// Registry lists quants best-first; first match wins. Falls back to the
// smallest if nothing else fits.
function pickAutoQuant(m, vramGb) {
  const variant = pickAutoVariant(m, vramGb);
  const quants = (variant?.quants) || m.quants || [];
  if (!quants.length) return null;
  for (const q of quants) {
    if (vramGb >= (q.auto_min_vram_gb ?? 0)) return q;
  }
  return quants[quants.length - 1];
}

function selectedQuantId(m) {
  const saved = modelStateOf(m.id).quant || 'auto';
  if (saved !== 'auto') return saved;
  return pickAutoQuant(m, state.gpu?.vram_gb || 0)?.id || null;
}

// Quants list to render in the drawer dropdown — comes from the currently
// selected variant if the model is variant-based, otherwise the model's
// flat quants list.
function quantsFor(m) {
  const variants = m.variants || [];
  if (variants.length) return selectedVariant(m)?.quants || [];
  return m.quants || [];
}

function resolveContextConfig(m) {
  const vram = state.gpu?.vram_gb || 0;
  const quant = selectedQuantId(m);
  let cfg = {
    tokens: Number(m.context_window || m.max_context_tokens || m.max_model_len || 2048),
    mode: 'workspace',
    auto_compact_at: 0.82,
    keep_recent_messages: 8,
    label: 'Default',
  };

  if (Array.isArray(m.context_by_vram)) {
    const tier = m.context_by_vram.find(t => {
      const minGb = t.min_gb ?? 0;
      const quantOk = !t.quant || !quant || t.quant === quant;
      return vram >= minGb && quantOk;
    });
    if (tier) cfg = { ...cfg, ...tier, label: tier.label || `${tier.tokens?.toLocaleString?.() || tier.tokens} tokens` };
  }

  if (m.context_policy) cfg = { ...cfg, ...m.context_policy };
  cfg.tokens = Number(cfg.tokens || 2048);
  return cfg;
}

function drawerPhase(id) {
  const s = modelStateOf(id);
  if (s.error)             return 'error';
  if (s.ready)             return 'ready';
  if (s.starting)          return 'starting';
  if (s.downloading)       return 'downloading';
  if (s.downloaded)        return 'downloaded';
  return 'not_downloaded';
}

async function refreshWeightState(m) {
  try {
    const ws = await api.weightStatus(m.id, selectedVariantId(m));
    setModelState(m.id, {
      downloaded: !!ws.downloaded,
      downloading: !!ws.downloading,
      progress: ws.progress || 0,
      ...(ws.error ? { error: ws.error } : {}),
    });
  } catch {}
}

async function openDrawer(id) {
  const m = state.models.find(x => x.id === id);
  if (!m) return;
  state.selected = m;

  // Title links to the source HuggingFace repo if known. innerHTML rather
  // than textContent here because we're injecting a link; m.name comes from
  // our own registry so it's safe.
  $('#drawerTitle').innerHTML = m.hf_repo
    ? `<a class="title-link" href="https://huggingface.co/${esc(m.hf_repo)}" target="_blank" rel="noopener">${esc(m.name)} ↗</a>`
    : esc(m.name);
  $('#drawerSub').textContent = m.hf_repo || '';
  const dIcon = $('#drawerIcon');
  dIcon.className = `mc-icon ${m.category}`;
  dIcon.innerHTML = `<svg><use href="#${categoryIcon(m.category)}"/></svg>`;

  $('#drawerScrim').classList.add('open');
  $('#drawer').classList.add('open');

  // Reconcile weight + runner state so reopening the drawer never shows stale "starting".
  await refreshWeightState(m);
  // Runtime profile status is informational here; Start will prepare/repair
  // the profile automatically before launching.
  if (m.runtime) await refreshRuntimeStatus(m.id);
  try {
    const h = await api.runnerHealth(m.id);
    if (h.ready)           setModelState(m.id, { ready: true,  starting: false, error: null });
    else if (h.load_error) setModelState(m.id, { ready: false, starting: false, error: h.load_error });
    else if (!h.loading)   setModelState(m.id, { ready: false, starting: false });
  } catch {}

  renderDrawerBody();
}

function closeDrawer() {
  $('#drawerScrim').classList.remove('open');
  $('#drawer').classList.remove('open');
}

function renderDrawerBody() {
  const m = state.selected;
  if (!m) return;
  const phase = drawerPhase(m.id);
  const sizeStr = modelVramProfile(m).label;
  const rt = m.runtime ? state.runtimes[m.id] : null;
  const startingTitle = rt?.state === 'installing' ? 'Preparing runtime…' : 'Starting runner…';
  const startingSub = rt?.state === 'installing'
    ? `${m.runtime?.label || 'Runtime'} dependencies are being prepared.`
    : 'Loading the model onto the GPU.';
  const startingLog = rt?.state === 'installing'
    ? (rt.last_line || 'Preparing runtime profile…')
    : 'Waiting for runner…';

  // State block — short status with subtitle. Icon + colour shift per phase.
  const stateBlocks = {
    not_downloaded: { cls:'',       icon:'i-down',    title:'Not downloaded',    sub:'Pull model weights from HuggingFace.' },
    downloading:    { cls:'busy',   icon:'i-spinner', title:'Downloading weights', sub:'Large repos can take a while.' },
    downloaded:     { cls:'ready',  icon:'i-check',   title:'Weights cached',    sub:'Ready to start the runner.' },
    starting:       { cls:'busy',   icon:'i-spinner', title:startingTitle,       sub:startingSub },
    ready:          { cls:'ready',  icon:'i-check',   title:'Runner ready',      sub:'Open the workspace to generate.' },
    error:          { cls:'error',  icon:'i-x',       title:'Something went wrong', sub: modelStateOf(m.id).error || '' },
  }[phase];

  $('#drawerBody').innerHTML = `
    <p style="font-size:13px;color:var(--t-2);line-height:1.55;margin-bottom:12px">
      ${m.description || ''}
    </p>

    <div class="drawer-stats">
      <div class="drawer-stat"><div class="label">VRAM</div><div class="value">${sizeStr}</div></div>
      <div class="drawer-stat"><div class="label">LoRA</div><div class="value">${m.supports_lora ? 'yes' : 'no'}</div></div>
      ${m.category === 'llm' ? `
        <div class="drawer-stat" style="grid-column:1/-1">
          <div class="label">Context</div>
          <div class="value">${contextStatText(m)}</div>
        </div>` : ''}
      <div class="drawer-stat" style="grid-column:1/-1"><div class="label">HF repo</div><div class="value">${m.hf_repo
        ? `<a class="ext-link" href="https://huggingface.co/${esc(m.hf_repo)}" target="_blank" rel="noopener">${esc(m.hf_repo)} ↗</a>`
        : '—'}</div></div>
      ${m.license ? `
        <div class="drawer-stat" style="grid-column:1/-1">
          <div class="label">License</div>
          <div class="value">
            <a class="license-pill ${esc(m.license.type || 'custom')}" href="${esc(m.license.url || '#')}" target="_blank" rel="noopener" title="${esc(m.license.label || '')}">${esc(m.license.label || 'Unknown')} ↗</a>
          </div>
        </div>` : ''}
    </div>

    <div class="drawer-state ${stateBlocks.cls}">
      <div class="state-icon"><svg><use href="#${stateBlocks.icon}"/></svg></div>
      <div class="state-text">
        <div class="state-title">${stateBlocks.title}</div>
        <div class="state-sub">${stateBlocks.sub}</div>
      </div>
    </div>

    ${phase === 'downloading' ? `
      <div class="drawer-progress"><div class="fill" id="drawerProgress" style="width:${(modelStateOf(m.id).progress || 0.05) * 100}%"></div></div>
    ` : ''}
    ${phase === 'starting' ? `
      <div class="drawer-log-stream" id="drawerLogStream"><div class="log-line" style="color:rgba(255,255,255,0.35)">${esc(startingLog)}</div></div>
    ` : ''}

    ${(() => {
      // Variant dropdown — only for models with multiple sizes (Wan 14B / 5B etc).
      // "Auto" picks the largest variant the GPU can comfortably run.
      if (!m.variants?.length) return '';
      const vram = state.gpu?.vram_gb || 0;
      const auto = pickAutoVariant(m, vram);
      const cur  = modelStateOf(m.id).variant || 'auto';
      const autoHint = auto ? `auto picks ${auto.label}` : 'auto';
      const opts = m.variants.map(v => {
        const minVram = v.auto_min_vram_gb ?? 0;
        const tooBig = vram > 0 && vram < minVram - 4;
        return `<option value="${esc(v.id)}" ${cur === v.id ? 'selected' : ''} ${tooBig ? 'disabled' : ''}>${esc(v.label)}</option>`;
      }).join('');
      return `
        <div class="field" style="margin-top:14px">
          <label class="field-label">Size <span class="hint">${autoHint}</span></label>
          <select class="select" id="drawerVariant">
            <option value="auto" ${cur === 'auto' ? 'selected' : ''}>Auto</option>
            ${opts}
          </select>
        </div>`;
    })()}

    ${(() => {
      // Quantisation dropdown — for variant-based models the quants come from
      // the resolved variant's list; otherwise the flat m.quants list.
      // "Auto" resolves to the highest-quality option the GPU can run.
      const quants = quantsFor(m);
      if (!quants.length) return '';
      const vram = state.gpu?.vram_gb || 0;
      const auto = pickAutoQuant(m, vram);
      const cur  = modelStateOf(m.id).quant || 'auto';
      const autoHint = auto ? `auto picks ${auto.label} on this GPU` : 'auto';
      const opts = quants.map(q => {
        const tooBig = vram > 0 && vram < (q.vram_gb - 4);
        const note   = `${q.vram_gb} GB · ${q.description}`;
        return `<option value="${esc(q.id)}" ${cur === q.id ? 'selected' : ''} ${tooBig ? 'disabled' : ''}>${esc(q.label)} — ${esc(note)}</option>`;
      }).join('');
      return `
        <div class="field" style="margin-top:14px">
          <label class="field-label">Quantisation <span class="hint">${autoHint}</span></label>
          <select class="select" id="drawerQuant">
            <option value="auto" ${cur === 'auto' ? 'selected' : ''}>Auto</option>
            ${opts}
          </select>
        </div>`;
    })()}

    <div class="field" style="margin-top:14px">
      <label class="field-label">HF token <span class="hint">gated repos</span></label>
      <input class="text-input" id="drawerHFToken" type="password" placeholder="hf_…" value="${state.settings.hf_token || ''}">
    </div>

    ${(() => {
      // Components (split-file transformer / VAE / text-encoder swaps).
      // Only renders when the model declares m.components.options.
      // Each target gets a "Default" option (load from base HF repo) plus
      // the curated options. Installed options can be selected; uninstalled
      // ones show an Install button. Selection persists in modelState.components.
      const comp = m.components;
      if (!comp?.options?.length) return '';
      const sel = modelStateOf(m.id).components || {};
      const byTarget = {};
      for (const o of comp.options) (byTarget[o.target] = byTarget[o.target] || []).push(o);
      const targetLabel = { transformer: 'Transformer', vae: 'VAE', text_encoder: 'Text encoder' };
      const sections = Object.keys(byTarget).map(target => {
        const opts = byTarget[target];
        const cur  = sel[target] || '';
        const optHtml = opts.map(o => {
          const installed = isComponentInstalled(o);
          const basename  = String(o.filename).split('/').pop();
          const label     = `${o.label}${installed ? '' : ' · not installed'}`;
          return `<option value="${esc(basename)}" ${cur === basename ? 'selected' : ''} ${installed ? '' : 'disabled'}>${esc(label)}</option>`;
        }).join('');
        return `
          <div class="field" style="margin-top:10px">
            <label class="field-label">${esc(targetLabel[target] || target)} <span class="hint">component</span></label>
            <select class="select" data-component-target="${esc(target)}">
              <option value="" ${cur === '' ? 'selected' : ''}>Default (from base repo)</option>
              ${optHtml}
            </select>
            ${opts.filter(o => !isComponentInstalled(o)).map(o =>
              `<button class="btn sm" style="margin-top:6px" data-component-install="${esc(o.id)}"><svg><use href="#i-down"/></svg>Install · ${esc(o.label)}</button>`
            ).join('')}
          </div>`;
      }).join('');
      return `
        <div class="field" style="margin-top:18px">
          <label class="field-label">Components <span class="hint">${esc(comp.description || 'split-file weight swaps')}</span></label>
          ${sections}
        </div>`;
    })()}
  `;

  // Persist the quant choice immediately on change so reopening the drawer
  // remembers what the user picked.
  $('#drawerQuant')?.addEventListener('change', (e) => {
    setModelState(m.id, { quant: e.target.value });
  });
  // Variant change resets quant to 'auto' (since quant options differ per
  // variant) and re-renders the drawer so the quant dropdown shows the new
  // variant's quants.
  $('#drawerVariant')?.addEventListener('change', (e) => {
    setModelState(m.id, {
      variant: e.target.value,
      quant: 'auto',
      downloaded: false,
      downloading: false,
      progress: 0,
      error: null,
    });
    renderDrawerBody();
    refreshWeightState(m).then(() => renderDrawerBody());
  });
  $('#drawerHFToken')?.addEventListener('change', e => saveHFToken(e.target.value));

  // Component picker: persist selected filename per target on change.
  $$('[data-component-target]', $('#drawer')).forEach(sel => {
    sel.addEventListener('change', () => {
      const target = sel.dataset.componentTarget;
      const cur = modelStateOf(m.id).components || {};
      cur[target] = sel.value || '';
      setModelState(m.id, { components: { ...cur } });
    });
  });
  // Install button — kicks off a background download job (component files
  // are 20–40 GB, so a sync request gets killed by RunPod's public-proxy
  // timeout). We start the job, poll status every few seconds, and only
  // update the UI to "installed" when the backend confirms the file is on
  // disk. Survives reload because we re-poll active jobs from
  // state.componentJobs on next render.
  $$('[data-component-install]', $('#drawer')).forEach(btn => btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    const optId = btn.dataset.componentInstall;
    const opt = (m.components.options || []).find(o => o.id === optId);
    if (!opt) return;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span><span>Starting…</span>';
    try {
      const { job_id } = await api.installComponents({
        files: [{ target: opt.target, hf_repo: opt.hf_repo, filename: opt.filename }],
        hf_token: state.settings.hf_token || null,
      });
      if (!job_id) throw new Error('No job id returned');
      pollComponentInstall(job_id, m, opt, btn);
    } catch (err) {
      toast(`Install failed: ${err.message || 'unknown'}`, 'error');
      btn.disabled = false;
      btn.innerHTML = `<svg><use href="#i-down"/></svg>Install · ${esc(opt.label)}`;
    }
  }));

  renderDrawerPrimary();
}

function contextStatText(m) {
  const cfg = resolveContextConfig(m);
  const bits = [`${cfg.tokens.toLocaleString()} tokens`];
  if (cfg.label) bits.push(cfg.label);
  if (cfg.mode) bits.push(cfg.mode);
  return esc(bits.join(' · '));
}

function renderDrawerPrimary() {
  const m = state.selected;
  if (!m) return;
  const btn = $('#drawerPrimary');

  const phase = drawerPhase(m.id);
  const rt = m.runtime ? state.runtimes[m.id] : null;
  const startingHtml = rt?.state === 'installing'
    ? '<span class="spinner"></span><span>Preparing runtime…</span>'
    : '<span class="spinner"></span><span>Starting…</span>';
  const labels = {
    not_downloaded: { html: '<svg><use href="#i-down"/></svg>Download weights',  busy: false, action: 'download' },
    downloading:    { html: '<span class="spinner"></span><span>Downloading…</span>', busy: true, action: null },
    downloaded:     { html: '<svg><use href="#i-bolt"/></svg>Start runner',      busy: false, action: 'start' },
    starting:       { html: startingHtml, busy: true, action: null },
    ready:          { html: '<svg><use href="#i-link"/></svg>Open workspace',    busy: false, action: 'open' },
    error:          {
      html: '<svg><use href="#i-x"/></svg>Retry',
      busy: false,
      action: modelStateOf(m.id).downloaded ? 'retry-start' : 'retry',
    },
  }[phase];
  btn.innerHTML  = labels.html;
  btn.disabled   = labels.busy;
  btn.dataset.action = labels.action || '';
}

function renderField(f) {
  const v = state.params[f.key] ?? f.default ?? '';
  switch (f.type) {
    case 'textarea':
      return `<div class="field">
        <label class="field-label">${f.label}${f.hint ? `<span class="hint">${f.hint}</span>` : ''}</label>
        <textarea class="textarea" data-key="${f.key}" placeholder="${f.placeholder || ''}">${v}</textarea>
      </div>`;
    case 'select':
      return `<div class="field">
        <label class="field-label">${f.label}${f.hint ? `<span class="hint">${f.hint}</span>` : ''}</label>
        <select class="select" data-key="${f.key}">
          ${(f.options || []).map(o => `<option value="${o}" ${o == v ? 'selected' : ''}>${o}</option>`).join('')}
        </select>
      </div>`;
    case 'bool':
      return `<div class="toggle-row">
        <span>${f.label}</span>
        <button class="toggle ${v ? 'on' : ''}" data-key="${f.key}" data-toggle></button>
      </div>`;
    case 'slider': {
      const min = f.min ?? 0, max = f.max ?? 100, step = f.step ?? 1;
      return `<div class="field slider-row" data-control-key="${esc(f.key)}">
        <div class="slider-head">
          <span>${f.label}</span><span class="val" id="val_${f.key}" data-slider-value="${esc(f.key)}">${v}</span>
        </div>
        <div class="slider-stepper">
          <button class="slider-step" type="button" data-key="${f.key}" data-stepper="-1" aria-label="Decrease ${esc(f.label)}">
            <svg><use href="#i-minus"/></svg>
          </button>
          <input class="text-input slider-number" type="number" inputmode="decimal" min="${min}" max="${max}" step="${step}" value="${v}" data-default="${v}" data-key="${f.key}" data-slider-number aria-label="${esc(f.label)} value">
          <button class="slider-step" type="button" data-key="${f.key}" data-stepper="1" aria-label="Increase ${esc(f.label)}">
            <svg><use href="#i-plus"/></svg>
          </button>
        </div>
      </div>`;
    }
    case 'asset':
      return `<div class="field">
        <label class="field-label">${f.label}${f.hint ? `<span class="hint">${f.hint}</span>` : ''}</label>
        <div style="display:flex;gap:8px">
          <input class="text-input" data-key="${f.key}" placeholder="Pick from Assets, or paste path…" value="${v}">
          <button class="btn" type="button" data-pick-asset data-target="${f.key}">Pick</button>
        </div>
      </div>`;
    case 'number':
    default:
      return `<div class="field" data-control-key="${esc(f.key)}">
        <label class="field-label">${f.label}${f.hint ? `<span class="hint">${f.hint}</span>` : ''}</label>
        <input class="text-input" type="${f.type === 'number' ? 'number' : 'text'}"
               ${f.min !== undefined ? `min="${f.min}"` : ''}
               ${f.max !== undefined ? `max="${f.max}"` : ''}
               ${f.step !== undefined ? `step="${f.step}"` : ''}
               value="${v}" data-key="${f.key}">
      </div>`;
  }
}

function numericStepDecimals(step) {
  const raw = String(step || 1).toLowerCase();
  if (raw.includes('e-')) return Number(raw.split('e-')[1]) || 0;
  const dot = raw.indexOf('.');
  return dot >= 0 ? raw.length - dot - 1 : 0;
}

function formatControlNumber(value, step) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '';
  const decimals = numericStepDecimals(step);
  const rounded = Number(n.toFixed(decimals));
  if (!decimals) return String(Math.round(rounded));
  return rounded.toFixed(decimals).replace(/\.?0+$/, '');
}

function sliderSpec(field, key) {
  const escaped = CSS.escape(key);
  const el = field?.querySelector(`[data-key="${escaped}"][data-slider-number]`)
          || field?.querySelector(`[data-key="${escaped}"][data-slider]`);
  const min = el?.min !== undefined && el.min !== '' ? Number(el.min) : -Infinity;
  const max = el?.max !== undefined && el.max !== '' ? Number(el.max) : Infinity;
  const step = el?.step && el.step !== 'any' ? Number(el.step) : 1;
  const fallback = el?.dataset.default !== undefined && el.dataset.default !== ''
    ? Number(el.dataset.default)
    : (Number.isFinite(min) ? min : 0);
  return {
    min,
    max,
    step: Number.isFinite(step) && step > 0 ? step : 1,
    fallback: Number.isFinite(fallback) ? fallback : 0,
  };
}

function clampNumber(value, min, max) {
  let n = Number(value);
  if (!Number.isFinite(n)) return n;
  if (Number.isFinite(min)) n = Math.max(min, n);
  if (Number.isFinite(max)) n = Math.min(max, n);
  return n;
}

function snapNumber(value, spec) {
  let n = clampNumber(value, spec.min, spec.max);
  if (!Number.isFinite(n)) return n;
  const base = Number.isFinite(spec.min) ? spec.min : 0;
  n = base + Math.round((n - base) / spec.step) * spec.step;
  return clampNumber(n, spec.min, spec.max);
}

function setSliderParam(rootEl, key, value, sourceEl = null, opts = {}) {
  const field = sourceEl?.closest?.('[data-control-key]')
             || rootEl.querySelector(`[data-control-key="${CSS.escape(key)}"]`);
  const spec = sliderSpec(field, key);
  const raw = value === undefined ? sourceEl?.value : value;

  if (raw === '') {
    state.params[key] = '';
    field?.querySelectorAll(`[data-slider-value="${CSS.escape(key)}"]`).forEach(el => { el.textContent = '—'; });
    return;
  }

  let n = Number(raw);
  if (!Number.isFinite(n)) return;
  n = opts.snap ? snapNumber(n, spec) : clampNumber(n, spec.min, spec.max);
  const display = formatControlNumber(n, spec.step);
  state.params[key] = Number(display);

  field?.querySelectorAll(`[data-slider-value="${CSS.escape(key)}"]`).forEach(el => { el.textContent = display; });
  field?.querySelectorAll(`[data-key="${CSS.escape(key)}"]`).forEach(peer => {
    if (!peer.hasAttribute('data-slider') && !peer.hasAttribute('data-slider-number')) return;
    if (peer === sourceEl && opts.preserveTyping) return;
    peer.value = display;
  });

  if ((key === 'width' || key === 'height') && state.selected) {
    renderAspectChips(state.selected, true);
  }
}

// Generic param input binding — works for both drawer and workspace.
function bindParamInputs(rootEl) {
  rootEl.querySelectorAll('input[data-key],select[data-key],textarea[data-key]').forEach(el => {
    el.addEventListener('input', () => {
      const key = el.dataset.key;
      if (el.hasAttribute('data-slider') || el.hasAttribute('data-slider-number')) {
        setSliderParam(rootEl, key, el.value, el, {
          snap: el.hasAttribute('data-slider'),
          preserveTyping: el.hasAttribute('data-slider-number'),
        });
        return;
      }
      let v = el.value;
      if (el.type === 'number') v = el.value === '' ? '' : Number(el.value);
      state.params[key] = v;
      const valEl = rootEl.querySelector(`#val_${key}`);
      if (valEl) valEl.textContent = v;
      const field = el.closest('[data-control-key]');
      if (field) {
        field.querySelectorAll(`[data-key="${CSS.escape(key)}"]`).forEach(peer => {
          if (peer !== el && (peer.hasAttribute('data-slider') || peer.hasAttribute('data-slider-number'))) peer.value = el.value;
        });
      }
      // Keep aspect chips in sync when user manually edits width or height.
      if ((key === 'width' || key === 'height') && state.selected) {
        renderAspectChips(state.selected, true);
      }
    });
    if (el.hasAttribute('data-slider-number')) {
      el.addEventListener('change', () => {
        const key = el.dataset.key;
        const field = el.closest('[data-control-key]');
        const spec = sliderSpec(field, key);
        const next = el.value === '' ? spec.fallback : el.value;
        setSliderParam(rootEl, key, next, el, { snap: true });
      });
    }
  });
  rootEl.querySelectorAll('[data-stepper]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const key = btn.dataset.key;
      const field = btn.closest('[data-control-key]');
      const spec = sliderSpec(field, key);
      const raw = state.params[key] !== '' && state.params[key] !== undefined
        ? state.params[key]
        : field?.querySelector(`[data-key="${CSS.escape(key)}"][data-slider-number]`)?.value;
      const current = Number(raw);
      const multiplier = e.shiftKey ? 10 : 1;
      const direction = Number(btn.dataset.stepper || 0);
      const base = Number.isFinite(current) ? current : spec.fallback;
      setSliderParam(rootEl, key, base + direction * spec.step * multiplier, btn, { snap: true });
    });
  });
  rootEl.querySelectorAll('[data-toggle]').forEach(el => {
    el.addEventListener('click', () => {
      el.classList.toggle('on');
      state.params[el.dataset.key] = el.classList.contains('on');
    });
  });
  rootEl.querySelectorAll('[data-pick-asset]').forEach(b => {
    b.addEventListener('click', () => pickAssetInto(b.dataset.target, rootEl));
  });
}

function pickAssetInto(targetKey, rootEl) {
  $('#modalTitle').textContent = 'Pick reference asset';
  $('#modalMeta').textContent  = 'Click an asset to use it as the reference';
  const imgs = state.assets.filter(a => a.kind === 'image');
  $('#modalBody').innerHTML = imgs.length
    ? `<div class="asset-grid" style="max-height:50vh;overflow:auto">
        ${imgs.map(a => `
          <div class="asset" data-pick="${esc(a.path)}">
            <img src="${esc(a.url)}" loading="lazy">
            <div class="asset-meta">${esc(a.name)}</div>
          </div>`).join('')}
      </div>`
    : `<div class="empty">No image assets yet — upload some on the Assets tab.</div>`;
  $('#modalFoot').innerHTML = `<button class="btn ghost" id="modalCancel">Cancel</button>`;
  $('#modalCancel').onclick = closeModal;
  $$('.asset[data-pick]', $('#modalBody')).forEach(a => a.addEventListener('click', () => {
    const p = a.dataset.pick;
    state.params[targetKey] = p;
    const input = rootEl.querySelector(`[data-key="${targetKey}"]`);
    if (input) input.value = p;
    closeModal();
  }));
  openModal();
}

// ── Drawer primary button: dispatches by current phase ───────────────────
async function onDrawerPrimary() {
  const m = state.selected;
  if (!m) return;
  const action = $('#drawerPrimary').dataset.action;
  if (action === 'download' || action === 'retry') return downloadWeights(m);
  if (action === 'retry-start')                    return startRunner(m);
  if (action === 'start')                          return startRunner(m);
  if (action === 'open')                           return openWorkspace(m.id);
}

// ── Runtime profiles ────────────────────────────────────────────────────
// Some model families need an isolated dependency profile. Start prepares
// that profile automatically, then launches the runner once it is ready.

async function refreshRuntimeStatus(modelId) {
  try {
    const s = await api.runtimeStatus(modelId);
    if (!s.required) {
      // No runtime block on this model — clear any stale entry. Keeps
      // the drawer phase machine off the install path.
      delete state.runtimes[modelId];
      return;
    }
    state.runtimes[modelId] = {
      ...(state.runtimes[modelId] || {}),
      state:       s.state || 'missing',
      python_path: s.python_path,
    };
  } catch {
    // If status endpoint is unreachable, leave whatever we already had.
  }
}

async function waitForRuntimeInstall(m, jobId) {
  while (true) {
    let job;
    try {
      job = await api.runtimeJobStatus(m.id, jobId);
    } catch (e) {
      await new Promise(r => setTimeout(r, 6000));
      continue;
    }
    if (job.status === 'queued' || job.status === 'running') {
      setRuntimeState(m.id, { state: 'installing', last_line: job.last_line || '' });
      if (state.selected?.id === m.id) renderDrawerBody();
      await new Promise(r => setTimeout(r, 4000));
      continue;
    }
    if (job.status === 'done') {
      setRuntimeState(m.id, { state: 'ready', python_path: job.python_path });
      return true;
    }
    throw new Error(job.error || 'install failed');
  }
}

async function ensureRuntimeReady(m) {
  if (!m.runtime) return true;
  let status = await api.runtimeStatus(m.id);
  if (!status.required) return true;
  setRuntimeState(m.id, {
    state: status.state || 'missing',
    python_path: status.python_path,
    last_line: status.note || '',
  });
  if (status.state === 'ready') return true;

  setRuntimeState(m.id, {
    state: 'installing',
    last_line: status.state === 'stale'
      ? 'Runtime profile changed; repairing dependencies'
      : 'Preparing dependencies',
  });
  if (state.selected?.id === m.id) renderDrawerBody();

  const job = await api.installRuntime(m.id);
  if (job.status === 'already_installed') {
    setRuntimeState(m.id, { state: 'ready' });
    return true;
  }
  if (!job.job_id) throw new Error('Runtime install did not return a job id');
  return waitForRuntimeInstall(m, job.job_id);
}

function setRuntimeState(modelId, patch) {
  state.runtimes[modelId] = { ...(state.runtimes[modelId] || {}), ...patch };
}

async function downloadWeights(m) {
  // Read token before any re-render that would wipe the typed value.
  const hfToken = currentHFToken('#drawerHFToken');
  const variant = selectedVariantId(m);

  // Check upfront — snapshot_download returns instantly for cached repos, which
  // looks like a broken 1-second download. Detect it here and skip the POST.
  try {
    const ws = await api.weightStatus(m.id, variant);
    if (ws.downloaded) {
      setModelState(m.id, { downloading: false, downloaded: true });
      toast(`${m.name} already on disk`, 'info');
      renderDrawerBody();
      return;
    }
  } catch { /* if weight-status errors, fall through to the normal download path */ }

  setModelState(m.id, { downloading: true, downloaded: false, error: null });
  renderDrawerBody();

  try {
    await api.downloadWeights(m.id, { hf_token: hfToken, variant });
    // Poll weight-status until done. Endpoint reports {downloading, downloaded, progress, error}.
    const ok = await waitForDownload(m.id, 60 * 60, variant);   // up to 1 hour for first 40GB pull
    if (!ok) throw new Error('Download did not complete (timeout)');
    setModelState(m.id, { downloading: false, downloaded: true });
    toast(`${m.name} weights ready`, 'success');
  } catch (e) {
    setModelState(m.id, { downloading: false, downloaded: false, error: e.message || 'Download failed' });
  }
  renderDrawerBody();
}

async function waitForDownload(modelId, maxSeconds, variant) {
  const deadline = Date.now() + maxSeconds * 1000;
  while (Date.now() < deadline) {
    try {
      const ws = await api.weightStatus(modelId, variant);
      if (ws.error)       throw new Error(ws.error);
      if (ws.downloaded)  return true;
      if (typeof ws.progress === 'number') {
        setModelState(modelId, { progress: ws.progress });
        const fill = $('#drawerProgress');
        if (fill) fill.style.width = `${Math.min(100, ws.progress * 100)}%`;
      }
    } catch { /* tolerate transient errors */ }
    await new Promise(r => setTimeout(r, 2000));
  }
  return false;
}

async function startRunner(m) {
  const hfToken = currentHFToken('#drawerHFToken');
  // Resolve variant first (size: 14B / 5B for Wan etc.), then quant within
  // that variant. Both default to 'auto'; auto resolves at launch time.
  let variant;
  if (m.variants?.length) {
    const variantSel = $('#drawerVariant')?.value || modelStateOf(m.id).variant || 'auto';
    if (variantSel === 'auto') {
      variant = pickAutoVariant(m, state.gpu?.vram_gb || 0)?.id || m.variants[0].id;
    } else {
      variant = variantSel;
    }
  }
  const quantSel = $('#drawerQuant')?.value || modelStateOf(m.id).quant || 'auto';
  let quant = quantSel;
  if (quantSel === 'auto' && (m.quants?.length || m.variants?.length)) {
    const auto = pickAutoQuant(m, state.gpu?.vram_gb || 0);
    quant = auto?.id || 'bf16';
  }
  // Strip empty/default ('' = use base repo) entries so the backend only
  // gets explicit overrides. Split-component swaps are only supported on
  // BF16 in the runner; launching them with int8/nf4 silently falls back to
  // base weights and makes any 2511 LoRA selection look broken.
  const compSel = modelStateOf(m.id).components || {};
  const components = Object.fromEntries(Object.entries(compSel).filter(([, v]) => v));
  if (Object.keys(components).length && quant !== 'bf16') {
    setModelState(m.id, {
      error: 'Split components require BF16. Choose BF16 or clear the component override.',
    });
    renderDrawerBody();
    toast('Split components require BF16', 'error');
    return;
  }
  setModelState(m.id, { starting: true, error: null });
  renderDrawerBody();
  let logEs = null;
  try {
    await ensureRuntimeReady(m);
    renderDrawerBody();
    const launched = await api.launch({ model_id: m.id, loras: [], hf_token: hfToken, quant, variant, components });
    if (launched.status !== 'launched' && launched.status !== 'already_running') {
      throw new Error(launched.message || 'Launch failed');
    }
    // Pipe runner stdout/stderr into the drawer log box while we wait.
    logEs = new EventSource(`/api/logs/${m.id}`);
    logEs.onmessage = (e) => {
      const logEl = document.getElementById('drawerLogStream');
      if (!logEl) return;
      if (logEl.querySelector('.log-line[style]')) logEl.innerHTML = ''; // clear placeholder
      const line = document.createElement('div');
      line.className = 'log-line' + (e.data.toLowerCase().includes('error') ? ' err' : '');
      line.textContent = e.data;
      logEl.appendChild(line);
      logEl.scrollTop = logEl.scrollHeight;
    };
    const ok = await waitForRunner(m.id, 600);
    if (!ok) throw new Error('Runner did not become ready');
    setModelState(m.id, { starting: false, ready: true });
    toast(`${m.name} ready`, 'success');
  } catch (e) {
    setModelState(m.id, { starting: false, ready: false, downloaded: true, error: e.message });
  } finally {
    if (logEs) logEs.close();
  }
  renderDrawerBody();
}

// ── Workspace view ───────────────────────────────────────────────────────
function openWorkspace(modelId, push = true) {
  const m = state.models.find(x => x.id === modelId);
  if (!m) return;
  state.workspace = modelId;
  state.selected  = m;
  const sections = resolveFields(m);
  state.params = state.workspaceParams[modelId] || (state.workspaceParams[modelId] = defaultParamsForModel(m, sections));
  if (m.category === 'llm') {
    const cfg = resolveContextConfig(m);
    state.params.max_model_len ??= cfg.tokens;
    state.params.context_policy ??= {
      mode: cfg.mode || 'workspace',
      auto_compact_at: cfg.auto_compact_at ?? 0.82,
      keep_recent_messages: cfg.keep_recent_messages ?? 8,
    };
  }
  closeDrawer();
  renderWorkspace(m, sections);
  switchView('workspace', push);
}

function defaultParamsForModel(m, sections = resolveFields(m)) {
  const params = {};
  // Seed defaults from resolved fields.
  for (const s of sections) for (const f of s.fields) {
    if (f.default !== undefined) params[f.key] = f.default;
  }
  // VRAM-aware size override — registry can declare default_size_by_vram so a
  // high-VRAM card lands on native size instead of the 1024 fallback.
  if (Array.isArray(m.default_size_by_vram)) {
    const vram = state.gpu?.vram_gb || 0;
    const tier = m.default_size_by_vram.find(t => vram >= (t.min_gb || 0));
    if (tier) {
      params.width  = tier.w;
      params.height = tier.h;
    }
  }
  const variantDefaults = selectedVariant(m)?.default_params;
  if (variantDefaults && typeof variantDefaults === 'object') {
    Object.assign(params, variantDefaults);
  }
  return params;
}

function closeWorkspace() {
  state.workspace = null;
  switchView('models');
}

function renderWorkspace(m, sections) {
  const isLLM = m.category === 'llm';

  // Header
  $('#wsTitle').textContent = m.name;
  $('#wsSub').textContent   = `${m.hf_repo || ''} · ${m.runner_module || ''}`;
  const ic = $('#wsIcon');
  ic.className = `mc-icon ${m.category}`;
  ic.innerHTML = `<svg><use href="#${categoryIcon(m.category)}"/></svg>`;

  // Status pill — derived from modelStates + running poll
  updateWsStatus();

  const workspaceView = $('[data-view="workspace"]');
  workspaceView?.classList.toggle('llm-mode', isLLM);
  $('.main')?.classList.toggle('llm-open', isLLM);
  $('#llmSessionsToggle').style.display = isLLM ? '' : 'none';
  $('#llmWorkspace').style.display = isLLM ? '' : 'none';
  $('#wsGrid').style.display = isLLM ? 'none' : '';
  $('#wsRecentSection').style.display = isLLM ? 'none' : '';
  if (isLLM) {
    $('#wsImgInputs').style.display = 'none';
    renderLLMWorkspace(m, sections);
    return;
  }

  // Image-input strip (Ref/Pose/Style/+ Generated) — only if the model declares it.
  const inputs = m.image_inputs || [];
  $('#wsImgInputs').style.display = inputs.length ? '' : 'none';
  if (inputs.length) renderImgStrip(m);

  // Prompt fields — only show negative_prompt if the model uses it.
  const promptKeys = sections.flatMap(s => s.fields.map(f => f.key));
  $('#wsNegSection').style.display = promptKeys.includes('negative_prompt') ? '' : 'none';

  // Pre-fill prompt textareas with current state.
  $('#wsPrompt').value    = state.params.prompt || '';
  $('#wsNegPrompt').value = state.params.negative_prompt || '';
  $('#wsNegPrompt').placeholder = DEFAULT_NEGATIVE_PROMPT;
  renderPromptMode(m);
  renderPromptBuilder(m);

  // Aspect chips (only when the model exposes width/height).
  const hasDims = promptKeys.includes('width') && promptKeys.includes('height');
  renderAspectChips(m, hasDims);

  // Settings grid — render every non-prompt, non-dim field as an inline control.
  // Width/height are rendered manually below the chips so they can be hidden cleanly.
  const skipKeys = new Set(['prompt', 'negative_prompt', 'ref_image', 'width', 'height', 'seed']);
  const settingsFields = sections.flatMap(s => s.fields).filter(f => !skipKeys.has(f.key));
  let settingsHtml = settingsFields.map(f => renderField(f)).join('');
  // Seed gets a custom random/fixed UI.
  if (promptKeys.includes('seed')) settingsHtml = renderSeedField() + settingsHtml;
  // Manual width/height beneath the chips.
  if (hasDims) {
    const wField = sections.flatMap(s => s.fields).find(f => f.key === 'width');
    const hField = sections.flatMap(s => s.fields).find(f => f.key === 'height');
    if (wField && hField) {
      settingsHtml += `<div class="field" data-control-key="width"><label class="field-label">Width</label>${miniNumber(wField)}</div>`;
      settingsHtml += `<div class="field" data-control-key="height"><label class="field-label">Height</label>${miniNumber(hField)}</div>`;
    }
  }
  // Upscaler dropdown — only when at least one upscaler is compatible with this model's category.
  // Spans full width of the settings grid. Empty value = no upscale.
  const upscalers = (state.upscalers || []).filter(u =>
    !u.compatible_with || u.compatible_with.includes(m.category)
  );
  if (upscalers.length) {
    const cur = (state.params.upscale && (state.params.upscale.id || state.params.upscale)) || '';
    const opts = upscalers.map(u =>
      `<option value="${esc(u.id)}" ${cur === u.id ? 'selected' : ''}>${esc(u.label)}</option>`
    ).join('');
    settingsHtml += `
      <div class="field" style="grid-column:1/-1">
        <label class="field-label">Upscale <span class="hint">lazy-loads on first use</span></label>
        <select class="select" id="wsUpscale" data-key="upscale">
          <option value="" ${!cur ? 'selected' : ''}>None</option>
          ${opts}
        </select>
      </div>`;
  }
  const settingsEl = $('#wsSettings');
  settingsEl.classList.toggle('video-controls', promptKeys.includes('num_frames') || promptKeys.includes('fps') || m.category === 'video');
  settingsEl.innerHTML = settingsHtml;

  // LoRA panel (with strength sliders + toggles)
  renderWorkspaceLoras(m);

  // HF token
  $('#wsHFToken').value = state.settings.hf_token || '';
  $('#wsHFToken').onchange = (e) => saveHFToken(e.target.value);

  // Recents + queue
  renderRecents(m.id);
  renderQueue();

  // Wire dynamic inputs (param fields, seed toggle)
  bindWorkspaceInputs();
}

function renderLLMWorkspace(m, sections) {
  ensureChatSession(m.id);

  const controls = sections.flatMap(s => s.fields)
    .filter(f => f.key !== 'prompt' && f.key !== 'messages')
    .map(f => renderField(f))
    .join('');
  $('#llmSettings').innerHTML = controls;
  bindParamInputs($('#llmSettings'));
  renderLLMSessions(m);
  renderLLMThread(m);
  renderLLMContext(m);
  updateLLMSendButton();
}

function renderLLMThread(modelOrId) {
  const m = typeof modelOrId === 'string' ? state.models.find(x => x.id === modelOrId) : modelOrId;
  const modelId = m?.id || modelOrId;
  const thread = $('#llmThread');
  const session = activeChatSession(modelId);
  const messages = session?.messages || [];
  if (!messages.length) {
    thread.innerHTML = `
      <div class="llm-empty">
        <div class="llm-orbit" aria-hidden="true"><span></span><span></span><span></span></div>
        <div class="llm-empty-title">Start a conversation</div>
        <div class="llm-empty-sub">Ask for drafts, plans, code ideas, or a quick second brain check.</div>
      </div>`;
    return;
  }

  // Pre-compute which assistant messages have a saved counterpart so we can
  // mark them visually and swap "Save" → "Saved" on the action row.
  const savedKeys = new Set(
    (session.saved_context || []).map(b => `${b.user}\u001f${b.assistant}`)
  );
  thread.innerHTML = messages.map((msg, idx) => {
    const busy = msg.thinking || msg.streaming;
    const canEdit = msg.role === 'user' && !state.llmBusy;
    const canCopy = msg.text && !msg.streaming;
    const prev = messages[idx - 1];
    const canSave = msg.role === 'assistant' && msg.text && !msg.streaming
                  && prev && prev.role === 'user';
    const isSaved = canSave && savedKeys.has(`${prev.text}\u001f${msg.text}`);
    return `
      <div class="llm-message ${esc(msg.role)}${isSaved ? ' saved' : ''}" data-msg-index="${idx}">
        <div class="llm-bubble">
          ${busy ? thinkingIndicator(msg.thinking) : ''}
          ${msg.text ? `<div class="llm-text">${formatChatText(msg.text)}</div>` : ''}
          ${canEdit || canCopy || canSave ? `
            <div class="llm-msg-actions">
              ${canCopy ? `<button class="llm-msg-action" data-copy-msg="${idx}">Copy</button>` : ''}
              ${canEdit ? `<button class="llm-msg-action" data-edit-msg="${idx}">Edit</button>` : ''}
              ${canSave ? (isSaved
                ? `<span class="llm-msg-action saved-tag">Saved</span>`
                : `<button class="llm-msg-action" data-save-msg="${idx}">Save</button>`) : ''}
            </div>` : ''}
        </div>
      </div>`;
  }).join('');
  bindLLMMessageActions(m, session);
  thread.scrollTop = thread.scrollHeight;
  if (m) renderLLMContext(m);
}

function bindLLMMessageActions(m, session) {
  $('#llmThread').querySelectorAll('[data-copy-msg]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const msg = session.messages[Number(btn.dataset.copyMsg)];
      if (!msg?.text) return;
      try {
        await navigator.clipboard.writeText(msg.text);
        toast('Copied', 'success');
      } catch {
        toast('Copy failed', 'error');
      }
    });
  });
  $('#llmThread').querySelectorAll('[data-edit-msg]').forEach(btn => {
    btn.addEventListener('click', () => editLLMMessage(m, Number(btn.dataset.editMsg)));
  });
  $('#llmThread').querySelectorAll('[data-save-msg]').forEach(btn => {
    btn.addEventListener('click', () => saveContextFromMessage(m, session, Number(btn.dataset.saveMsg)));
  });
}

// Save a user→assistant turn as persistent context for this session.
// Stays verbatim across compaction; visible in transcript with a marker;
// manageable from the Saved Context modal.
function saveContextFromMessage(m, session, assistantIdx) {
  const assistant = session.messages[assistantIdx];
  const user = session.messages[assistantIdx - 1];
  if (!assistant || assistant.role !== 'assistant' || !user || user.role !== 'user') return;
  session.saved_context ||= [];
  // Don't double-save the same exact pair.
  if (session.saved_context.some(b => b.user === user.text && b.assistant === assistant.text)) return;
  session.saved_context.push({
    id: `ctx_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    user: user.text,
    assistant: assistant.text,
    label: user.text.slice(0, 40).trim() || 'Saved exchange',
    tokens: estimateTokens(`${user.text} ${assistant.text}`),
  });
  session.updated_at = Date.now();
  toast('Saved to context', 'success');
  renderLLMThread(m);
  renderLLMContext(m);
}

function openSavedContextModal(m) {
  const session = activeChatSession(m.id);
  if (!session) return;
  const blocks = session.saved_context || [];
  const totalTokens = blocks.reduce((sum, b) => sum + (b.tokens || 0), 0);

  $('#modalTitle').textContent = 'Saved context';
  $('#modalMeta').textContent  = blocks.length
    ? `${blocks.length} block${blocks.length === 1 ? '' : 's'} · ~${totalTokens.toLocaleString()} tokens`
    : 'Nothing saved yet — use the Save button under any assistant reply.';

  const renderRows = () => blocks.map(b => `
    <div class="ctx-row" data-ctx-id="${esc(b.id)}">
      <div class="ctx-row-head">
        <input class="ctx-label-input" type="text" value="${esc(b.label || '')}" data-ctx-label="${esc(b.id)}" placeholder="Label">
        <span class="ctx-tokens">~${(b.tokens || 0).toLocaleString()} tokens</span>
        <button class="icon-btn" data-ctx-delete="${esc(b.id)}" title="Delete"><svg><use href="#i-trash"/></svg></button>
      </div>
      <div class="ctx-snippet"><span class="ctx-role">user</span> ${esc(b.user.slice(0, 200))}${b.user.length > 200 ? '…' : ''}</div>
      <div class="ctx-snippet"><span class="ctx-role">assistant</span> ${esc(b.assistant.slice(0, 200))}${b.assistant.length > 200 ? '…' : ''}</div>
    </div>`).join('');

  $('#modalBody').innerHTML = blocks.length
    ? `<div class="ctx-list">${renderRows()}</div>`
    : `<div class="empty" style="padding:24px;text-align:center"><span style="font-size:11.5px;color:var(--t-3)">Nothing saved yet</span></div>`;
  $('#modalFoot').innerHTML = `<button class="btn ghost" id="modalCancel">Close</button>`;
  $('#modalCancel').onclick = closeModal;

  // Inline label edits update the session immediately.
  $$('[data-ctx-label]', $('#modalBody')).forEach(input => {
    input.addEventListener('input', () => {
      const id = input.dataset.ctxLabel;
      const block = (session.saved_context || []).find(b => b.id === id);
      if (block) {
        block.label = input.value;
        session.updated_at = Date.now();
      }
    });
  });
  // Delete drops the block and re-renders the modal so the token count updates live.
  $$('[data-ctx-delete]', $('#modalBody')).forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.ctxDelete;
      session.saved_context = (session.saved_context || []).filter(b => b.id !== id);
      session.updated_at = Date.now();
      renderLLMThread(m);
      renderLLMContext(m);
      openSavedContextModal(m);   // reopen so it re-renders with updated list/total
    });
  });

  openModal();
}

function editLLMMessage(m, index) {
  const session = activeChatSession(m.id);
  const msg = session?.messages?.[index];
  if (!session || !msg || msg.role !== 'user' || state.llmBusy) return;
  const node = $(`.llm-message[data-msg-index="${index}"] .llm-bubble`, $('#llmThread'));
  if (!node) return;
  node.classList.add('editing');
  node.innerHTML = `
    <textarea class="llm-edit-input">${esc(msg.text)}</textarea>
    <div class="llm-edit-actions">
      <button class="btn sm ghost" data-edit-cancel>Cancel</button>
      <button class="btn sm primary" data-edit-save>Save & rerun</button>
    </div>`;
  const textarea = $('.llm-edit-input', node);
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  $('[data-edit-cancel]', node).addEventListener('click', () => renderLLMThread(m));
  $('[data-edit-save]', node).addEventListener('click', () => {
    const next = textarea.value.trim();
    if (!next) return;
    rerunFromEditedMessage(m, session, index, next);
  });
}

function thinkingIndicator(showLabel = true) {
  return `
    <div class="llm-thinking">
      <span class="think-core"></span>
      ${showLabel ? '<span class="think-label">Thinking</span>' : ''}
      <span class="think-dots"><i></i><i></i><i></i></span>
    </div>`;
}

function formatChatText(text) {
  return esc(text).replace(/\n/g, '<br>');
}

function ensureChatSession(modelId) {
  state.chatSessions[modelId] ||= [];
  if (!state.chatSessions[modelId].length) createChatSession(modelId, true);
  const activeId = state.activeChat[modelId];
  const active = state.chatSessions[modelId].find(s => s.id === activeId);
  if (!active) state.activeChat[modelId] = state.chatSessions[modelId][0].id;
  return activeChatSession(modelId);
}

function createChatSession(modelId, activate = false) {
  state.chatSessions[modelId] ||= [];
  const session = {
    id: `chat_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    title: 'New chat',
    messages: [],
    compact_summary: '',
    compacted_count: 0,
    saved_context: [],   // [{id, user, assistant, label, tokens}] — user-pinned exchanges
    created_at: Date.now(),
    updated_at: Date.now(),
  };
  state.chatSessions[modelId].unshift(session);
  if (activate) state.activeChat[modelId] = session.id;
  return session;
}

function activeChatSession(modelId) {
  const sessions = state.chatSessions[modelId] || [];
  return sessions.find(s => s.id === state.activeChat[modelId]) || sessions[0] || null;
}

function renderLLMSessions(m) {
  const drawer = $('#llmSessions');
  drawer.classList.toggle('open', state.llmDrawerOpen);
  $('.llm-shell')?.classList.toggle('drawer-open', state.llmDrawerOpen);
  $('#llmSessionsToggle').classList.toggle('active', state.llmDrawerOpen);
  $('#llmSessionsMeta').textContent = `${(state.chatSessions[m.id] || []).length} session${(state.chatSessions[m.id] || []).length === 1 ? '' : 's'}`;

  const sessions = state.chatSessions[m.id] || [];
  $('#llmSessionList').innerHTML = sessions.map(session => {
    const active = session.id === state.activeChat[m.id];
    const tokens = estimateSessionTokens(session);
    const last = session.messages.at(-1)?.text || 'No messages yet';
    return `
      <div class="llm-session ${active ? 'active' : ''}" data-session="${esc(session.id)}">
        <button class="llm-session-main" data-open-session="${esc(session.id)}">
          <span class="llm-session-name">${esc(session.title)}</span>
          <span class="llm-session-last">${esc(last)}</span>
          <span class="llm-session-meta">${tokens} tokens${session.compacted_count ? ` · ${session.compacted_count} compacted` : ''}</span>
        </button>
        <button class="icon-btn" data-rename-session="${esc(session.id)}" title="Rename"><svg><use href="#i-dots"/></svg></button>
        <button class="icon-btn danger" data-delete-session="${esc(session.id)}" title="Delete"><svg><use href="#i-trash"/></svg></button>
      </div>`;
  }).join('');

  $('#llmSessionList').querySelectorAll('[data-open-session]').forEach(btn => {
    btn.addEventListener('click', () => {
      state.activeChat[m.id] = btn.dataset.openSession;
      renderLLMWorkspace(m, resolveFields(m));
    });
  });
  $('#llmSessionList').querySelectorAll('[data-rename-session]').forEach(btn => {
    btn.addEventListener('click', () => renameChatSession(m.id, btn.dataset.renameSession));
  });
  $('#llmSessionList').querySelectorAll('[data-delete-session]').forEach(btn => {
    btn.addEventListener('click', () => deleteChatSession(m.id, btn.dataset.deleteSession));
  });
}

function renameChatSession(modelId, sessionId) {
  const session = (state.chatSessions[modelId] || []).find(s => s.id === sessionId);
  if (!session) return;
  const next = prompt('Rename chat', session.title);
  if (!next) return;
  session.title = next.trim().slice(0, 80) || session.title;
  session.updated_at = Date.now();
  const m = state.models.find(x => x.id === modelId);
  if (m) renderLLMSessions(m);
}

async function deleteChatSession(modelId, sessionId) {
  const sessions = state.chatSessions[modelId] || [];
  const session = sessions.find(s => s.id === sessionId);
  if (!session) return;
  if (!await confirmModal('Delete Chat', `Delete "${esc(session.title)}" from this session? This removes the conversation from the current app memory.`, 'Delete', true)) return;
  state.chatSessions[modelId] = sessions.filter(s => s.id !== sessionId);
  if (!state.chatSessions[modelId].length) createChatSession(modelId, true);
  if (state.activeChat[modelId] === sessionId) state.activeChat[modelId] = state.chatSessions[modelId][0].id;
  const m = state.models.find(x => x.id === modelId);
  if (m) renderLLMWorkspace(m, resolveFields(m));
}

function autoNameSession(session, promptText) {
  if (!session || session.title !== 'New chat') return;
  const words = promptText.replace(/\s+/g, ' ').trim().split(' ').slice(0, 7).join(' ');
  session.title = words.length > 36 ? `${words.slice(0, 34)}…` : (words || 'New chat');
}

function renderLLMContext(m) {
  const session = activeChatSession(m.id);
  const used = estimateSessionTokens(session);
  const limit = contextLimitFor(m);
  const pct = Math.min(100, Math.round((used / limit) * 100));
  const left = Math.max(0, 100 - pct);
  const remaining = Math.max(0, limit - used);
  const compacted = session?.compacted_count || 0;
  const compacting = (session?.compacting_until || 0) > Date.now();
  const savedCount = (session?.saved_context || []).length;
  $('#llmContext').innerHTML = `
    <button class="llm-context-meter ${compacting ? 'compacting' : ''}" type="button" aria-label="Context window ${pct}% full">
      <span class="context-ring" aria-hidden="true">
        <svg viewBox="0 0 36 36">
          <circle class="context-ring-bg" cx="18" cy="18" r="14"></circle>
          <circle class="context-ring-fill" cx="18" cy="18" r="14" pathLength="100" stroke-dasharray="${pct} 100"></circle>
        </svg>
      </span>
      <span class="context-mini">${left}% left</span>
      <span class="llm-context-card" role="tooltip">
        <span class="ctx-muted">Context window</span>
        <strong>${pct}% full</strong>
        <span>${used.toLocaleString()} / ${limit.toLocaleString()} tokens used</span>
        <span>${remaining.toLocaleString()} tokens left</span>
        <span>${savedCount} saved block${savedCount === 1 ? '' : 's'}</span>
        <span>${compacting ? 'Compacting older turns now' : compacted ? `${compacted} messages compacted` : 'Auto compacts near 82% full'}</span>
      </span>
    </button>
    <button class="llm-context-manage" type="button" id="llmContextManage" aria-label="Manage saved context">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v10Z"/></svg>
      <span>Saved${savedCount ? ` · ${savedCount}` : ''}</span>
    </button>`;
  $('#llmContextManage')?.addEventListener('click', () => openSavedContextModal(m));
}

function contextLimitFor(m) {
  return Number(state.params.max_model_len || resolveContextConfig(m).tokens || 2048);
}

function estimateSessionTokens(session) {
  if (!session) return 0;
  const body = [
    session.compact_summary || '',
    ...(session.saved_context || []).map(b => `${b.user}\n${b.assistant}`),
    ...(session.messages || []).map(msg => `${msg.role}: ${msg.text || ''}`),
  ].join('\n');
  return estimateTokens(body);
}

function estimateTokens(text) {
  return Math.max(1, Math.ceil(String(text || '').length / 4));
}

function compactSessionIfNeeded(session, m) {
  if (!session) return false;
  const limit = contextLimitFor(m);
  const policy = state.params.context_policy || resolveContextConfig(m);
  const compactAt = Number(policy.auto_compact_at ?? 0.82);
  const keepCount = Number(policy.keep_recent_messages ?? 8);
  if (estimateSessionTokens(session) < limit * compactAt) return false;
  const keep = session.messages.slice(-keepCount);
  const old = session.messages.slice(0, -keepCount);
  if (!old.length) return false;
  const digest = old.map(msg => `${msg.role}: ${String(msg.text || '').replace(/\s+/g, ' ').slice(0, 220)}`).join('\n');
  session.compact_summary = [
    session.compact_summary,
    `Earlier conversation summary:\n${digest}`,
  ].filter(Boolean).join('\n\n').slice(-Math.floor(limit * 2.2));
  session.compacted_count += old.length;
  session.messages = keep;
  session.compacting_until = Date.now() + 1800;
  setTimeout(() => {
    if (state.selected?.id === m.id && activeChatSession(m.id)?.id === session.id) {
      renderLLMContext(m);
    }
  }, 1850);
  return true;
}

function messagesForLLMPayload(session, assistant) {
  const messages = [];
  if (session?.compact_summary) {
    messages.push({ role: 'system', content: session.compact_summary });
  }
  // Saved context (user-pinned exchanges). Framed so the model treats them
  // as established background, not as recent conversation to reference.
  // The "do not paraphrase / do not say 'as we discussed'" line is the bit
  // most chat UIs forget — it's what stops the performative-recall problem.
  if (session?.saved_context?.length) {
    const blocks = session.saved_context
      .map(b => `User: ${b.user}\nAssistant: ${b.assistant}`)
      .join('\n\n');
    messages.push({
      role: 'system',
      content:
        'The user has saved the following past exchanges as relevant context. ' +
        'Treat them as established background — preferences, facts, or reference ' +
        'material that already shaped this conversation. Apply them silently. ' +
        'Do NOT paraphrase them back to the user. Do NOT say "as you mentioned earlier" ' +
        'or "as we discussed". Use the information only when it directly bears on the ' +
        'current question.\n\n=== Saved context ===\n' + blocks + '\n=== End saved context ===',
    });
  }
  messages.push(...(session?.messages || [])
    .filter(msg => msg !== assistant)
    .slice(-16)
    .map(msg => ({ role: msg.role, content: msg.text })));
  return messages;
}

function miniNumber(f) {
  const v = state.params[f.key] ?? f.default ?? '';
  return `<input class="text-input" type="number"
       ${f.min !== undefined ? `min="${f.min}"` : ''}
       ${f.max !== undefined ? `max="${f.max}"` : ''}
       ${f.step !== undefined ? `step="${f.step}"` : ''}
       value="${v}" data-key="${f.key}">`;
}

function updateWsStatus() {
  const m = state.selected;
  if (!m) return;
  const ms = modelStateOf(m.id);
  const isRunning = state.running[m.id]?.status === 'running';
  const el = $('#wsStatus');
  if (ms.generating || state.queueRunning) { el.className = 'ws-status busy';    el.lastChild.textContent = 'generating'; }
  else if (ms.ready && (isRunning || state.preview)) { el.className = 'ws-status ready';   el.lastChild.textContent = 'ready'; }
  else if (ms.starting)                    { el.className = 'ws-status loading'; el.lastChild.textContent = 'starting'; }
  else                                     { el.className = 'ws-status';         el.lastChild.textContent = 'idle'; }
}

// ── Image-input strip ────────────────────────────────────────────────────
function renderImgStrip(m) {
  const slots = state.imgInputs[m.id] || (state.imgInputs[m.id] = {});
  const cells = [];
  let hasMedia = false;
  for (const inp of m.image_inputs) {
    // Migrate legacy {image} shape to carousel shape on first access.
    let slot = slots[inp.key];
    if (!slot) {
      slot = slots[inp.key] = { enabled: !!inp.required, images: [], idx: 0 };
    } else if (slot.image !== undefined) {
      slot = slots[inp.key] = { enabled: slot.enabled, images: slot.image ? [slot.image] : [], idx: 0 };
    }
    if ((slot.images || [])[slot.idx ?? 0]) hasMedia = true;
    cells.push(imgCellHtml(inp.key, inp.label, slot, inp.required, inp.hint));
  }
  // The Generated cell shows the latest result (or empty).
  const recents = state.recents[m.id] || [];
  const latest  = recents[recents.length - 1] || null;
  hasMedia = hasMedia || !!latest;
  cells.push(generatedCellHtml(latest));

  const strip = $('#wsImgStrip');
  strip.classList.toggle('has-media', hasMedia);
  strip.innerHTML = cells.join('');
  updateImageCellRatios(strip);
  bindImgStrip(m);
}

function imgCellHtml(key, label, slot, required, hint) {
  const images = slot.images || [];
  const idx    = slot.idx ?? 0;
  const active = images[idx] || null;
  const off    = !slot.enabled ? ' off' : '';
  const empty  = !active ? ' empty' : '';
  const req    = required ? ' required' : '';
  const nav    = images.length > 1
    ? `<button class="icon-btn" data-prev-input="${key}" title="Previous">&#8249;</button>
       <span class="carousel-pos">${idx + 1}/${images.length}</span>
       <button class="icon-btn" data-next-input="${key}" title="Next">&#8250;</button>`
    : '';
  const media = active ? ' has-media' : '';
  return `<div class="img-cell${media}${off}${empty}${req}" data-input="${key}">
    <div class="img-cell-head">
      <div class="label">${label}${required ? ' *' : ''}</div>
      <button class="toggle ${slot.enabled ? 'on' : ''}" data-toggle-input="${key}" title="Use this input"></button>
    </div>
    <div class="img-cell-content" data-add-input="${key}">
      ${active
        ? `<img src="${esc(active.url)}" alt="${esc(active.name || '')}">`
        : `<div class="img-cell-empty">
             <svg class="icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/></svg>
             ${esc(hint || 'Click to add')}
           </div>`}
    </div>
    ${active ? `
      <div class="img-cell-foot">
        ${nav}
        <div class="spacer"></div>
        <button class="icon-btn" data-add-input="${key}" title="Add image"><svg width="13" height="13"><use href="#i-upload"/></svg></button>
        <button class="icon-btn" data-clear-input="${key}" title="Remove"><svg width="13" height="13"><use href="#i-trash"/></svg></button>
      </div>` : ''}
  </div>`;
}

function generatedCellHtml(asset) {
  const u = asset ? esc(asset.url) : '';
  const n = asset ? esc(asset.name) : '';
  return `<div class="img-cell generated${asset ? ' has-media' : ' empty'}" data-input="generated">
    <div class="img-cell-head">
      <div class="label">Generated</div>
    </div>
    ${asset ? `
      <button class="asset-menu-btn img-cell-menu" data-generated-menu title="Actions">
        <svg><use href="#i-dots"/></svg>
      </button>` : ''}
    <div class="img-cell-content" ${asset ? 'data-zoom="1"' : ''}>
      ${asset
        ? (asset.kind === 'video'
            ? `<video src="${u}" muted playsinline preload="metadata"></video>`
            : `<img src="${u}" alt="${n}">`)
        : `<div class="img-cell-empty">
             <svg class="icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
             Output appears here
           </div>`}
    </div>
  </div>`;
}

function updateImageCellRatios(root = document) {
  root.querySelectorAll('.img-cell.has-media img, .img-cell.has-media video').forEach(media => {
    const apply = () => {
      const w = media.videoWidth || media.naturalWidth;
      const h = media.videoHeight || media.naturalHeight;
      if (!w || !h) return;
      const cell = media.closest('.img-cell');
      if (!cell) return;
      cell.style.setProperty('--media-ratio', `${w} / ${h}`);
      cell.classList.toggle('portrait-media', h > w * 1.15);
      cell.classList.toggle('landscape-media', w > h * 1.35);
    };
    apply();
    media.addEventListener(media.tagName === 'VIDEO' ? 'loadedmetadata' : 'load', apply, { once: true });
  });
}

function bindImgStrip(m) {
  const slots = state.imgInputs[m.id];
  const latest = (state.recents[m.id] || []).at(-1);
  $$('[data-generated-menu]', $('#wsImgStrip')).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    if (latest) showAssetMenu(latest, b);
  }));
  // Toggle on/off per cell
  $$('[data-toggle-input]', $('#wsImgStrip')).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    slots[b.dataset.toggleInput].enabled = !slots[b.dataset.toggleInput].enabled;
    renderImgStrip(m);
  }));
  // Add image (opens picker, pushes to images[])
  $$('[data-add-input]', $('#wsImgStrip')).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    pickImageForInput(m, b.dataset.addInput);
  }));
  // Carousel prev / next
  $$('[data-prev-input]', $('#wsImgStrip')).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    const slot = slots[b.dataset.prevInput];
    slot.idx = (slot.idx - 1 + slot.images.length) % slot.images.length;
    renderImgStrip(m);
  }));
  $$('[data-next-input]', $('#wsImgStrip')).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    const slot = slots[b.dataset.nextInput];
    slot.idx = (slot.idx + 1) % slot.images.length;
    renderImgStrip(m);
  }));
  // Remove current image from carousel
  $$('[data-clear-input]', $('#wsImgStrip')).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    const slot = slots[b.dataset.clearInput];
    slot.images.splice(slot.idx, 1);
    slot.idx = Math.max(0, Math.min(slot.idx, slot.images.length - 1));
    renderImgStrip(m);
  }));
  // Zoom on the generated cell
  $$('[data-zoom]', $('#wsImgStrip')).forEach(c => c.addEventListener('click', () => {
    const recents = state.recents[m.id] || [];
    if (recents.length) openLightbox(recents[recents.length - 1], { list: recents, index: recents.length - 1 });
  }));
}

function pickImageForInput(m, key) {
  // Two-step: show asset library; if empty, fall back to file upload.
  $('#modalTitle').textContent = `Pick ${key} image`;
  $('#modalMeta').textContent  = 'From assets, or upload from disk.';
  const imgs = state.assets.filter(a => a.kind === 'image');
  $('#modalBody').innerHTML = `
    <div style="margin-bottom:10px"><button class="btn" id="modalUpload"><svg><use href="#i-upload"/></svg>Upload from disk</button></div>
    ${imgs.length
      ? `<div class="asset-grid" style="max-height:50vh;overflow:auto">
          ${imgs.map(a => `
            <div class="asset" data-pick="${esc(a.path)}">
              <img src="${esc(a.url)}" loading="lazy">
              <div class="asset-meta">${esc(a.name)}</div>
            </div>`).join('')}
        </div>`
      : `<div class="empty">No image assets yet — use Upload above, or drop files into the Assets tab.</div>`}`;
  $('#modalFoot').innerHTML = `<button class="btn ghost" id="modalCancel">Cancel</button>`;
  $('#modalCancel').onclick = closeModal;
  const apply = (asset) => {
    const slot = state.imgInputs[m.id][key];
    slot.enabled = true;
    slot.images.push(asset);
    slot.idx = slot.images.length - 1;   // jump to newly added
    closeModal();
    renderImgStrip(m);
  };
  $$('.asset[data-pick]', $('#modalBody')).forEach(a => a.addEventListener('click', () => {
    const path = a.dataset.pick;
    const asset = state.assets.find(x => x.path === path);
    if (asset) apply(asset);
  }));
  $('#modalUpload').onclick = () => {
    closeModal();
    const inp = $('#imgCellInput');
    inp.value = '';
    inp.onchange = async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      if (state.preview) {
        // No backend in preview mode — keep the asset in-memory only,
        // but mirror it into state.assets so the Library actually shows it.
        const url = URL.createObjectURL(f);
        const asset = { name: f.name, url, path: f.name, kind: 'image', source: 'upload' };
        state.assets.unshift(asset);
        $('#assetNavCount').textContent = state.assets.length;
        renderAssets();
        apply(asset);
      } else {
        try {
          const r = await api.uploadAsset(f);
          await loadAssets();
          apply({ name: r.name, url: r.url, path: r.path, kind: 'image', source: 'upload' });
        } catch { toast('Upload failed', 'error'); }
      }
    };
    inp.click();
  };
  openModal();
}

// ── Aspect-ratio chips ───────────────────────────────────────────────────
function renderAspectChips(m, hasDims) {
  const row = $('#wsAspectRow');
  if (!hasDims) { row.innerHTML = ''; return; }
  const presets = m.aspect_presets || [
    { label: 'Square',          w: 1024, h: 1024 },
    { label: 'Landscape 16:9',  w: 1664, h: 928  },
    { label: 'Portrait 9:16',   w: 928,  h: 1664 },
    { label: '3:4',             w: 1152, h: 1536 },
    { label: '4:3',             w: 1536, h: 1152 },
  ];
  const w = state.params.width, h = state.params.height;
  row.innerHTML = presets.map(p => {
    const active = w === p.w && h === p.h ? ' active' : '';
    const glyph = aspectGlyph(p.w / p.h);
    return `<button class="aspect-chip${active}" data-w="${p.w}" data-h="${p.h}" type="button">
      ${glyph}<span>${p.label}</span><span style="font-family:var(--font-mono);font-size:10px;opacity:.7">${p.w}×${p.h}</span>
    </button>`;
  }).join('');
  $$('.aspect-chip', row).forEach(c => c.addEventListener('click', () => {
    state.params.width  = Number(c.dataset.w);
    state.params.height = Number(c.dataset.h);
    renderAspectChips(m, true);
    // Reflect in the manual fields if they're rendered.
    const wIn = $('#wsSettings [data-key="width"]');
    const hIn = $('#wsSettings [data-key="height"]');
    if (wIn) wIn.value = state.params.width;
    if (hIn) hIn.value = state.params.height;
  }));
}

function aspectGlyph(ratio) {
  // Tiny rectangle inline-block sized to the ratio.
  const max = 12, w = Math.round(max * Math.min(1, ratio)), h = Math.round(max / Math.max(1, ratio));
  return `<span class="ratio-glyph" style="width:${w}px;height:${h}px"></span>`;
}

// ── Seed: Random / Fixed segmented + input ───────────────────────────────
//
// Random mode: params.seed is -1 (signals "pick one"). The actual seed used
// by the last run is shown in the input as read-only — clicking it copies
// to clipboard so the user can paste into Fixed.
//
// Fixed mode: input becomes editable; clicking Fixed pre-fills with the
// last-used seed so iterating ("re-roll a tweak of that one") is one click.
function lastSeedFor(modelId) {
  return state.lastSeed?.[modelId];
}

function renderSeedField() {
  const m = state.selected;
  const last = m ? lastSeedFor(m.id) : null;
  const isFixed = (state.params.seed !== undefined && state.params.seed !== -1);
  // What appears in the input depends on mode + history.
  const display = isFixed ? state.params.seed : (last ?? '');
  const showCopyHint = !isFixed && last != null;
  return `<div class="field" data-seed-field data-control-key="seed">
    <label class="field-label">Seed${showCopyHint ? ' <span class="hint">click value to copy</span>' : ''}</label>
    <div class="seed-row">
      <div class="segmented" data-seed-segmented>
        <div class="seg ${isFixed ? '' : 'active'}" data-seed-mode="random">Random</div>
        <div class="seg ${isFixed ? 'active' : ''}" data-seed-mode="fixed">Fixed</div>
      </div>
      <input class="text-input ${isFixed ? '' : 'readonly-seed'}" type="number" data-seed-input
             value="${display}" placeholder="auto"
             ${isFixed ? '' : 'readonly'}>
    </div>
  </div>`;
}

function refreshSeedField() {
  const wrapper = $('[data-seed-field]');
  if (!wrapper) return;
  wrapper.outerHTML = renderSeedField();
  bindSeedField();
}

function bindSeedField() {
  const segs = $$('[data-seed-mode]');
  const input = $('[data-seed-input]');
  segs.forEach(s => s.addEventListener('click', () => {
    const mode = s.dataset.seedMode;
    segs.forEach(x => x.classList.toggle('active', x === s));
    const last = lastSeedFor(state.selected?.id);
    if (mode === 'random') {
      state.params.seed = -1;
      input.readOnly = true;
      input.classList.add('readonly-seed');
      input.value = last ?? '';
    } else {
      input.readOnly = false;
      input.classList.remove('readonly-seed');
      // Carry the last-used seed forward so the user can iterate from it.
      const cur = Number(input.value);
      const seedToUse = last ?? (Number.isFinite(cur) && cur >= 0 ? cur : 0);
      state.params.seed = seedToUse;
      input.value = seedToUse;
      input.focus();
      input.select();
    }
  }));
  if (!input) return;
  input.addEventListener('input', () => {
    if (input.readOnly) return;
    const n = Number(input.value);
    if (Number.isFinite(n) && n >= 0) state.params.seed = n;
  });
  // Click-to-copy when in random mode (read-only display).
  input.addEventListener('click', async () => {
    if (!input.readOnly) return;
    const v = input.value;
    if (!v) return;
    try {
      await navigator.clipboard.writeText(String(v));
      toast(`Copied seed ${v}`, 'success');
    } catch { /* clipboard unavailable */ }
  });
}

function componentRequirementStatus(modelId, entry) {
  const req = entry?.requires_component;
  if (!req || typeof req !== 'object') return { ok: true, label: '' };
  const selected = modelStateOf(modelId).components || {};
  const targetLabel = { transformer: 'Transformer', vae: 'VAE', text_encoder: 'Text encoder' };
  for (const [target, required] of Object.entries(req)) {
    const allowed = (Array.isArray(required) ? required : [required])
      .filter(Boolean)
      .map(v => String(v).split('/').pop());
    const cur = selected[target] || '';
    if (!cur || !allowed.includes(cur)) {
      const pretty = allowed.map(v => v.replace(/\.safetensors$/i, '')).join(' or ');
      return {
        ok: false,
        label: `${targetLabel[target] || target}: ${pretty}`,
      };
    }
  }
  return { ok: true, label: '' };
}

function renderWorkspaceLoras(m) {
  if (!m.supports_lora) {
    $('#wsLoraPane').style.display = 'none';
    return;
  }
  $('#wsLoraPane').style.display = '';

  // Filter bundled LoRAs by applies_to vs the currently resolved variant.
  // Wan I2V Lightning only applies to the 14B variant, so it's hidden when
  // the user has the 5B variant selected (or auto-resolves to 5B).
  const currentVariant = m.variants?.length
    ? (selectedVariant(m)?.id || null)
    : null;
  const allDefaults = m.default_loras || [];
  const defaults = allDefaults.filter(l =>
    !l.applies_to || !currentVariant || l.applies_to.includes(currentVariant)
  );
  const assigned = state.loras.filter(l => l.model_id === m.id);

  // Stable key per logical LoRA. Multi-file LoRAs use `id` since they don't
  // have a single filename; everything else falls back to filename.
  const keyOf = (l) => l.id || loraId(l);

  // Seed per-model lora state. Dual-expert models (Wan etc.) carry two
  // strength values per LoRA — one for each expert — so high-noise and
  // low-noise can be tuned independently.
  const lstate = state.loraStates[m.id] || (state.loraStates[m.id] = {});
  const defaultHigh = (l) => l.strength_high ?? l.strength ?? 1.0;
  const defaultLow  = (l) => l.strength_low  ?? l.strength ?? 1.0;
  for (const l of defaults) {
    const k = keyOf(l);
    const saved = lstate[k] || {};
    lstate[k] = {
      ...saved,
      on: saved.on ?? !!l.default_on,
      strength: saved.strength ?? l.strength ?? 1.0,
      strength_high: saved.strength_high ?? defaultHigh(l),
      strength_low:  saved.strength_low  ?? defaultLow(l),
      bundled: true,
      label: l.label || l.filename || k,
      entry: l,                 // keep the registry entry so generate() can pick up files/hf_repo
    };
  }
  for (const l of assigned) {
    const k = keyOf(l);
    const saved = lstate[k] || {};
    const highDefault = m.dual_expert ? 0.7 : 1.0;
    const lowDefault = m.dual_expert ? 0.9 : 1.0;
    lstate[k] = {
      ...saved,
      on: saved.on ?? false,
      strength: saved.strength ?? 1.0,
      strength_high: saved.strength_high ?? highDefault,
      strength_low: saved.strength_low ?? lowDefault,
      bundled: false,
      label: l.rel_path || l.filename,
      entry: l,
    };
  }

  // A LoRA is "installed" when every file it needs is on disk.
  // state.loras lists what's actually present (basename + rel_path).
  const installedNames = new Set((state.loras || []).flatMap(l => [l.filename, l.rel_path].filter(Boolean)));
  const isInstalled = (entry) => {
    const files = entry.files?.length ? entry.files.map(f => f.filename) : [entry.filename].filter(Boolean);
    if (!files.length) return true;
    return files.every(fn => installedNames.has(fn) || installedNames.has(fn.split('/').pop()));
  };

  const defaultsEl = $('#wsLoraDefaults');
  if (defaults.length) {
    defaultsEl.style.display = '';
    defaultsEl.innerHTML = `<div class="lora-group-head">Bundled with model</div>` +
      defaults.map(l => loraDetailedHtml(m.id, keyOf(l), { installed: isInstalled(l) })).join('');
  } else {
    defaultsEl.style.display = 'none';
  }

  const assignedEl = $('#wsLoraAssigned');
  assignedEl.innerHTML = `<div class="lora-group-head">Assigned (${assigned.length})</div>` +
    (assigned.length
      ? assigned.map(l => loraDetailedHtml(m.id, keyOf(l), { installed: true })).join('')
      : `<div style="font-size:11.5px;color:var(--t-3);padding:6px 4px">No LoRAs assigned. Tag any LoRA with <span style="font-family:var(--font-mono)">${m.id}</span> to attach it here.</div>`);

  $('#wsLoraCount').textContent = `${defaults.length + assigned.length}`;

  bindLoraCards(m);
}

function loraDetailedHtml(modelId, key, opts = {}) {
  const lstate = state.loraStates[modelId][key];
  if (!lstate) return '';
  const installed = opts.installed !== false;
  const compReq = componentRequirementStatus(modelId, lstate.entry);
  const cls = lstate.bundled ? 'lora-card detailed bundled' : 'lora-card detailed';
  const m = state.models.find(x => x.id === modelId);
  const dual = !!m?.dual_expert;
  const eKey = esc(key);
  const eLbl = esc(lstate.label);
  // Sub-line shows the file (single) or "N files" (paired). Truncated.
  const filesNote = lstate.entry?.files?.length
    ? `${lstate.entry.files.length} files · ${esc(lstate.entry.hf_repo || 'lightx2v')}`
    : esc(lstate.entry?.filename || '');
  // Usage hint surfaces the "set steps=4 / cfg=1.0" requirement that most
  // Lightning LoRAs have. Without this guidance users see "messes up the
  // gen even at 0.1" because the LoRA fights the default sampler config.
  const hintRow = lstate.entry?.usage_hint
    ? `<div class="lora-hint">${esc(lstate.entry.usage_hint)}</div>`
    : '';
  // When uninstalled, the toggle is disabled and we surface an Install
  // button instead. Strength sliders are also disabled until installed.
  const disabledByMissing = !installed;
  const disabledByRequirement = !compReq.ok;
  const disabled = disabledByMissing || disabledByRequirement;
  const effectiveOn = lstate.on && !disabled;
  const enabledForSlider = effectiveOn;
  const targetSet = new Set();
  if (Array.isArray(lstate.entry?.files) && lstate.entry.files.length) {
    lstate.entry.files.forEach(f => targetSet.add((f.target || 'high').toLowerCase()));
  } else if (lstate.entry?.target || lstate.entry?.target_guess) {
    targetSet.add((lstate.entry.target || lstate.entry.target_guess).toLowerCase());
  }
  const targets = [...targetSet].filter(t => t === 'high' || t === 'low');
  if (!targets.length) targets.push('high');
  const sliderRow = dual
    ? targets.map(target => {
        const isLow = target === 'low';
        const attr = isLow ? 'data-lora-strength-low' : 'data-lora-strength-high';
        const valAttr = isLow ? 'data-lora-strength-low-val' : 'data-lora-strength-high-val';
        const value = isLow ? lstate.strength_low : lstate.strength_high;
        const label = isLow ? 'Low' : 'High';
        return `<div class="row">
          <span class="lora-tier">${label}</span>
          <input class="slider" type="range" min="0" max="2" step="0.05" value="${value}" ${attr}="${eKey}" ${enabledForSlider ? '' : 'disabled'}>
          <input class="text-input lora-strength-number" type="number" min="0" max="2" step="0.05" value="${value}" ${attr}="${eKey}" ${enabledForSlider ? '' : 'disabled'}>
          <span class="strength" ${valAttr}="${eKey}">×${Number(value).toFixed(2)}</span>
        </div>`;
      }).join('')
    : `
      <div class="row">
        <input class="slider" type="range" min="0" max="2" step="0.05" value="${lstate.strength}" data-lora-strength="${eKey}" ${enabledForSlider ? '' : 'disabled'}>
        <input class="text-input lora-strength-number" type="number" min="0" max="2" step="0.05" value="${lstate.strength}" data-lora-strength="${eKey}" ${enabledForSlider ? '' : 'disabled'}>
        <span class="strength" data-lora-strength-val="${eKey}">×${lstate.strength.toFixed(2)}</span>
      </div>`;
  const installRow = !installed
    ? `<div class="row"><button class="btn sm" data-lora-install="${eKey}"><svg><use href="#i-down"/></svg>Install</button>
        <span style="font-size:11px;color:var(--t-3)">Files not on disk</span></div>`
    : '';
  const reqRow = disabledByRequirement
    ? `<div class="lora-hint warn">Requires ${esc(compReq.label)} selected in Components.</div>`
    : '';
  return `<div class="${cls} ${effectiveOn ? 'on' : ''}${disabled ? ' missing' : ''}" data-lora="${eKey}">
    <div class="row">
      <span class="label" title="${eLbl}">${eLbl}</span>
      <button class="toggle ${effectiveOn ? 'on' : ''}" data-lora-toggle="${eKey}" ${disabled ? `disabled title="${disabledByMissing ? 'Install before enabling' : 'Select required component before enabling'}"` : ''}></button>
    </div>
    <div class="filename" title="${esc(filesNote)}">${filesNote}</div>
    ${hintRow}
    ${reqRow}
    ${installRow}
    ${sliderRow}
  </div>`;
}

function bindLoraCards(m) {
  const lstate = state.loraStates[m.id];
  $$('[data-lora-toggle]', $('#wsLoraPane')).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    if (b.disabled) return;
    const fn = b.dataset.loraToggle;
    lstate[fn].on = !lstate[fn].on;
    persistLoraStates();
    renderWorkspaceLoras(m);
  }));
  // Install missing files for a bundled LoRA. Pulls each {hf_repo, filename}
  // into LORAS_DIR via the install endpoint, then refreshes state.loras so
  // the card flips to installed and the toggle/sliders enable.
  $$('[data-lora-install]', $('#wsLoraPane')).forEach(b => b.addEventListener('click', async (e) => {
    e.stopPropagation();
    const k = b.dataset.loraInstall;
    const entry = lstate[k]?.entry;
    if (!entry) return;
    const hfRepo = entry.hf_repo;
    if (!hfRepo) { toast('No source repo declared for this LoRA', 'error'); return; }
    const files = (entry.files?.length ? entry.files : [{ filename: entry.filename }])
      .filter(f => f.filename)
      .map(f => ({ hf_repo: hfRepo, filename: f.filename }));
    b.disabled = true;
    b.innerHTML = '<span class="spinner"></span><span>Installing…</span>';
    try {
      const res = await api.installLoras({ files, hf_token: state.settings.hf_token || null });
      const errors = (res.results || []).filter(r => r.status === 'error');
      if (errors.length) {
        toast(`Install failed: ${errors[0].error || 'unknown'}`, 'error');
      } else {
        toast(`${entry.label || 'LoRA'} installed`, 'success');
      }
      await loadLoras();
      renderWorkspaceLoras(m);
    } catch (err) {
      toast(`Install failed: ${err.message || 'unknown'}`, 'error');
      b.disabled = false;
      b.innerHTML = '<svg><use href="#i-down"/></svg>Install';
    }
  }));
  $$('[data-lora-strength]', $('#wsLoraPane')).forEach(s => {
    s.addEventListener('input', () => {
      const fn = s.dataset.loraStrength;
      lstate[fn].strength = Number(s.value);
      $$(`[data-lora-strength="${CSS.escape(fn)}"]`, $('#wsLoraPane')).forEach(peer => {
        if (peer !== s) peer.value = s.value;
      });
      const out = $(`[data-lora-strength-val="${CSS.escape(fn)}"]`);
      if (out) out.textContent = `×${Number(s.value).toFixed(2)}`;
      persistLoraStates();
    });
  });
  // Dual-expert (high-noise / low-noise) sliders for MoE pipelines (Wan etc.)
  $$('[data-lora-strength-high]', $('#wsLoraPane')).forEach(s => {
    s.addEventListener('input', () => {
      const fn = s.dataset.loraStrengthHigh;
      lstate[fn].strength_high = Number(s.value);
      $$(`[data-lora-strength-high="${CSS.escape(fn)}"]`, $('#wsLoraPane')).forEach(peer => {
        if (peer !== s) peer.value = s.value;
      });
      const out = $(`[data-lora-strength-high-val="${CSS.escape(fn)}"]`);
      if (out) out.textContent = `×${Number(s.value).toFixed(2)}`;
      persistLoraStates();
    });
  });
  $$('[data-lora-strength-low]', $('#wsLoraPane')).forEach(s => {
    s.addEventListener('input', () => {
      const fn = s.dataset.loraStrengthLow;
      lstate[fn].strength_low = Number(s.value);
      $$(`[data-lora-strength-low="${CSS.escape(fn)}"]`, $('#wsLoraPane')).forEach(peer => {
        if (peer !== s) peer.value = s.value;
      });
      const out = $(`[data-lora-strength-low-val="${CSS.escape(fn)}"]`);
      if (out) out.textContent = `×${Number(s.value).toFixed(2)}`;
      persistLoraStates();
    });
  });
}

function renderRecents(modelId) {
  const items   = state.recents[modelId] || [];
  // Pending + running jobs for *this* model become loading cards prepended
  // to the recents grid. They share the same footprint and 3-dot menu shape
  // as completed assets, but with a Cancel-only menu instead of Download/Delete.
  const queued = state.queue.filter(j => j.model_id === modelId && j.status !== 'done');
  const grid = $('#wsRecent');

  if (!items.length && !queued.length) {
    grid.innerHTML = `<div class="empty" style="grid-column:1/-1;padding:18px"><span style="font-size:11.5px">Nothing generated yet.</span></div>`;
    return;
  }

  const queuedHtml = queueCardsHtml(queued);

  const recentHtml = items.slice().reverse().map((_, i) => {
    const idx = items.length - 1 - i;
    const a = items[idx];
    const u = esc(a.url);
    return `<div class="asset" data-recent-idx="${idx}">
      ${a.kind === 'video' ? `<video src="${u}" muted playsinline preload="metadata"></video>` : `<img src="${u}" loading="lazy">`}
      <button class="asset-menu-btn" data-asset-menu="${idx}" data-asset-source="recent" title="Actions">
        <svg><use href="#i-dots"/></svg>
      </button>
      <div class="asset-meta">${esc(a.name)}</div>
    </div>`;
  }).join('');

  grid.innerHTML = queuedHtml + recentHtml;

  // Recent tile click → lightbox; 3-dot → asset menu (Download / Delete).
  $$('.asset[data-recent-idx]', grid).forEach(el => el.addEventListener('click', (e) => {
    if (e.target.closest('[data-asset-menu]')) return;
    const idx = Number(el.dataset.recentIdx);
    openLightbox(items[idx], { list: items, index: idx });
  }));
  $$('[data-asset-menu]', grid).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    const idx = Number(b.dataset.assetMenu);
    showAssetMenu(items[idx], b);
  }));
  bindQueueCardMenus(grid);
}

function queueCardsHtml(jobs, opts = {}) {
  const showModel = !!opts.showModel;
  return jobs.map(j => {
    const stateLabel = j.status === 'running'    ? 'Generating'
                     : j.status === 'cancelling' ? 'Cancelling'
                     :                             'Queued';
    const pct = Math.max(0, Math.min(100, Math.round((j.progress || 0) * 100)));
    const hasProgress = pct > 0 && j.status !== 'pending';
    const eta = j.etaMs != null && pct < 100 ? ` · ${formatETA(j.etaMs)}` : '';
    const metaText = j.status === 'running'
      ? (hasProgress ? `${pct}%${eta}` : 'Estimating')
      : '';
    const promptText = esc((j.params.prompt || '(no prompt)').slice(0, 80));
    const modelText = showModel ? `<div class="q-model">${esc(j.model_name || j.model_id)}</div>` : '';
    return `<div class="asset queued${j.status === 'running' ? ' active' : ''}${hasProgress ? ' has-progress' : ''}" data-queue-job="${esc(j.id)}">
      <div class="q-state">${stateLabel}</div>
      ${metaText ? `<div class="q-meta">${esc(metaText)}</div>` : ''}
      ${modelText}
      <div class="qspinner"></div>
      <button class="asset-menu-btn" data-queue-menu="${esc(j.id)}" title="Actions">
        <svg><use href="#i-dots"/></svg>
      </button>
      ${hasProgress ? `<div class="q-progress"><span style="width:${pct}%"></span></div>` : ''}
      <div class="q-prompt">${promptText}</div>
    </div>`;
  }).join('');
}

function bindQueueCardMenus(root) {
  $$('[data-queue-menu]', root).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    showQueueMenu(b.dataset.queueMenu, b);
  }));
}

function showQueueMenu(jobId, anchorEl) {
  const pop = $('#assetPop');
  const rect = anchorEl.getBoundingClientRect();
  const W = 168;
  const left = Math.max(8, Math.min(window.innerWidth - W - 8, rect.right - W));
  const top  = Math.min(window.innerHeight - 60, rect.bottom + 4);
  pop.style.left = `${left}px`;
  pop.style.top  = `${top}px`;
  pop.innerHTML = `
    <div class="asset-pop-item danger" data-act="cancel">
      <svg><use href="#i-x"/></svg>Cancel job
    </div>`;
  pop.classList.add('open');
  pop.setAttribute('aria-hidden', 'false');
  pop.querySelector('[data-act="cancel"]').onclick = async () => {
    closeAssetMenu();
    await cancelQueueJob(jobId);
  };
}

// ── LoRA picker modal (workspace "+ Add LoRA" button) ────────────────────
//
// Two states:
//   - empty library  → "no LoRAs downloaded" + a Browse CivitAI button that
//                      jumps to the Loras tab so the user can search/download.
//   - has LoRAs      → checkbox list of every downloaded LoRA, ticked if
//                      already assigned to this model. Save patches each
//                      lora's meta.json with the model_id.
function openLoraPicker(model) {
  $('#modalTitle').textContent = `LoRAs · ${model.name}`;
  $('#modalMeta').textContent  = 'Pick which downloaded LoRAs are available in this workspace.';

  const all = state.loras || [];
  if (!all.length) {
    $('#modalBody').innerHTML = `
      <div class="empty" style="padding:18px 8px">
        <div class="icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 7h16M4 12h16M4 17h10"/></svg></div>
        <p style="margin-bottom:14px">No LoRAs downloaded yet.</p>
        <button class="btn primary" id="loraBrowseBtn"><svg><use href="#i-down"/></svg>Browse &amp; download</button>
      </div>`;
    $('#modalFoot').innerHTML = `<button class="btn ghost" id="modalCancel">Close</button>`;
    $('#modalCancel').onclick = closeModal;
    $('#loraBrowseBtn').onclick = () => { closeModal(); switchView('loras'); };
    openModal();
    return;
  }

  $('#modalBody').innerHTML = `
    <div class="field-stack" style="max-height:50vh;overflow-y:auto">
      ${all.map(l => {
        const checked = l.model_id === model.id ? 'checked' : '';
        const id = loraId(l);
        return `<label class="check-row ${checked ? 'on' : ''}">
          <input type="checkbox" data-lora-pick="${esc(id)}" ${checked}>
          <span class="name" title="${esc(l.rel_path || l.filename)}">${esc(l.rel_path || l.filename)}</span>
          <span class="meta">${l.size_mb || '—'}MB${l.model_id && l.model_id !== model.id ? ` · also on ${esc(l.model_id)}` : ''}</span>
        </label>`;
      }).join('')}
    </div>`;
  $('#modalFoot').innerHTML = `
    <button class="btn ghost" id="modalCancel">Cancel</button>
    <button class="btn primary" id="loraSaveBtn">Save</button>`;
  $('#modalCancel').onclick = closeModal;

  // Visual on/off when toggling each row.
  $$('[data-lora-pick]', $('#modalBody')).forEach(cb => {
    cb.addEventListener('change', () => cb.closest('.check-row').classList.toggle('on', cb.checked));
  });

  $('#loraSaveBtn').onclick = async () => {
    const btn = $('#loraSaveBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span><span>Saving…</span>';
    const checks = $$('[data-lora-pick]', $('#modalBody'));
    let changed = 0;
    let failed = 0;
    for (const c of checks) {
      const fn          = c.dataset.loraPick;
      const lora        = state.loras.find(l => loraId(l) === fn);
      const wasAssigned = lora?.model_id === model.id;
      const nowAssigned = c.checked;
      if (wasAssigned === nowAssigned) continue;
      try {
        if (state.preview) {
          // No backend — mutate in-memory so the workspace updates.
          if (lora) lora.model_id = nowAssigned ? model.id : '';
        } else {
          await api.patchLora(fn, {
            tags: lora?.tags || [],
            model_id: nowAssigned ? model.id : '',
          });
        }
        changed++;
      } catch {
        failed++;
      }
    }
    if (!state.preview) await loadLoras();
    if (state.workspace === model.id) renderWorkspaceLoras(model);
    closeModal();
    if (changed) toast(`Updated ${changed} LoRA${changed === 1 ? '' : 's'}`, 'success');
    if (failed) toast(`${failed} LoRA${failed === 1 ? '' : 's'} could not be updated`, 'error');
  };
  openModal();
}

// ── Asset 3-dot menu (shared between recents + assets grid) ──────────────
function showAssetMenu(asset, anchorEl) {
  const pop = $('#assetPop');
  // Position the popup just under the anchor's bottom-right corner.
  const rect = anchorEl.getBoundingClientRect();
  const W = 168;
  const left = Math.max(8, Math.min(window.innerWidth - W - 8, rect.right - W));
  const top  = Math.min(window.innerHeight - 90, rect.bottom + 4);
  pop.style.left = `${left}px`;
  pop.style.top  = `${top}px`;
  pop.innerHTML = `
    <div class="asset-pop-item" data-act="download">
      <svg><use href="#i-down"/></svg>Download
    </div>
    <div class="asset-pop-item danger" data-act="delete">
      <svg><use href="#i-trash"/></svg>Delete
    </div>`;
  pop.classList.add('open');
  pop.setAttribute('aria-hidden', 'false');

  pop.querySelector('[data-act="download"]').onclick = () => {
    downloadAsset(asset);
    closeAssetMenu();
  };
  pop.querySelector('[data-act="delete"]').onclick = async () => {
    closeAssetMenu();
    if (!await confirmModal('Delete Asset', 'Delete this asset?', 'Delete', true)) return;
    await deleteAssetAnywhere(asset);
  };
}

function closeAssetMenu() {
  const pop = $('#assetPop');
  pop.classList.remove('open');
  pop.setAttribute('aria-hidden', 'true');
}

async function downloadAsset(asset) {
  if (!asset?.url) return;
  try {
    toast(`Preparing ${asset.name || 'download'}…`, 'info');
    const res = await fetch(asset.url, { credentials: 'same-origin' });
    if (!res.ok) throw new Error(`Download failed (${res.status})`);
    const blob = await res.blob();
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = objUrl;
    a.download = asset.name || asset.path?.split('/').pop() || 'download';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(objUrl), 30_000);
    toast(`Saved ${asset.name || 'asset'}`, 'success');
  } catch (e) {
    toast(`Download failed: ${e.message || 'unknown error'}`, 'error');
  }
}

async function deleteAssetAnywhere(asset) {
  // 1. Sweep every consumer that might still hold a reference to this asset.
  //    Without this, deleted assets linger as broken refs in workspace input
  //    slots and as ghosts in the recents strip. Order matters: prune the
  //    in-memory holders BEFORE the fetch so a re-render mid-delete doesn't
  //    snapshot stale state.
  pruneFromRecents(asset);
  pruneFromImgInputs(asset);

  // 2. Remove the underlying file (or fake-delete in preview mode).
  if (asset.path && !state.preview) {
    state.assets = state.assets.filter(a => a.path !== asset.path);
    renderAssets();
    try {
      await api.deleteAsset(asset.path);
      refreshAssetsSoon();
    } catch (e) {
      toast(`Delete failed: ${e.message || 'unknown'}`, 'error');
      refreshAssetsSoon();
    }
  } else {
    state.assets = state.assets.filter(a => a.path !== asset.path);
    renderAssets();
  }

  // 3. Re-paint the open workspace's recents + Generated cell if it was the latest.
  if (state.workspace) {
    renderRecents(state.workspace);
    const m = state.models.find(x => x.id === state.workspace);
    if (m?.image_inputs?.length) renderImgStrip(m);
  }
  toast('Deleted', 'info');
}

function pruneFromRecents(asset) {
  if (!asset?.path) return;
  for (const id of Object.keys(state.recents)) {
    state.recents[id] = state.recents[id].filter(a => a.path !== asset.path);
  }
}

// Clear the deleted asset out of every model's input slots (ref / pose / style /
// any image_input the model declares). Without this, a deleted asset stays
// "selected" in the workspace and renders broken — and submitting a generation
// would either 404 in the runner or fail signature checks.
function pruneFromImgInputs(asset) {
  if (!asset?.path) return;
  for (const modelId of Object.keys(state.imgInputs)) {
    const slots = state.imgInputs[modelId];
    if (!slots) continue;
    for (const key of Object.keys(slots)) {
      const slot = slots[key];
      if (slot?.img?.path === asset.path) {
        slots[key] = { ...slot, img: null, enabled: false };
      }
    }
  }
}

function bindWorkspaceInputs() {
  bindParamInputs($('[data-view="workspace"]'));
  // Prompt textareas live outside the param grid; wire them by id.
  $('#wsPrompt').oninput     = (e) => { state.params.prompt = e.target.value; };
  $('#wsNegPrompt').oninput  = (e) => { state.params.negative_prompt = e.target.value; };
  bindSeedField();
}

function promptModeForModel(m = state.selected) {
  if (!m) return 'builder';
  return state.promptModes[m.id] || 'builder';
}

function promptBuilderForModel(m = state.selected) {
  if (!m) return {};
  state.promptBuilders[m.id] ||= {};
  return state.promptBuilders[m.id];
}

function savePromptBuilderState() {
  writeJSONStorage('forge_prompt_builders', state.promptBuilders);
}

function savePromptModes() {
  writeJSONStorage('forge_prompt_modes', state.promptModes);
}

function setPromptMode(mode) {
  const m = state.selected;
  if (!m || !['builder', 'raw'].includes(mode)) return;
  state.promptModes[m.id] = mode;
  savePromptModes();
  if (mode === 'raw') {
    const compiled = compilePromptBuilder(m);
    const input = $('#wsPrompt');
    if (compiled && input && !input.value.trim()) {
      input.value = compiled;
      state.params.prompt = compiled;
    }
  }
  renderPromptMode(m);
}

function renderPromptMode(m) {
  const mode = promptModeForModel(m);
  const builder = $('#wsPromptBuilder');
  const raw = $('#wsPromptRaw');
  if (builder) builder.style.display = mode === 'builder' ? '' : 'none';
  if (raw) raw.style.display = mode === 'raw' ? '' : 'none';
  $('#wsPromptModeBuilder')?.classList.toggle('active', mode === 'builder');
  $('#wsPromptModeRaw')?.classList.toggle('active', mode === 'raw');
}

function renderPromptBuilder(m) {
  const root = $('#wsPromptBuilder');
  if (!root || !m || m.category === 'llm') return;
  const values = promptBuilderForModel(m);
  const textFields = PROMPT_BUILDER_TEXT_FIELDS.map(f => `
    <label class="pb-field">
      <span>${esc(f.label)}</span>
      <input class="text-input" data-pb-text="${esc(f.key)}" value="${esc(values[f.key] || '')}" placeholder="${esc(f.placeholder)}">
    </label>
  `).join('');
  const chipGroups = PROMPT_BUILDER_CHIP_GROUPS.map(g => `
    <div class="pb-chip-group">
      <div class="pb-chip-label">${esc(g.label)}</div>
      <div class="pb-chips">
        ${g.options.map(opt => {
          const active = values[g.key] === opt;
          return `<button type="button" class="pb-chip ${active ? 'active' : ''}" data-pb-chip="${esc(g.key)}" data-pb-value="${esc(opt)}">${esc(opt)}</button>`;
        }).join('')}
      </div>
    </div>
  `).join('');
  root.innerHTML = `
    <div class="pb-grid">${textFields}</div>
    <div class="pb-control-grid">${chipGroups}</div>
    <div class="pb-preview">
      <div class="pb-preview-label">Compiled prompt</div>
      <div class="pb-preview-text" id="wsPromptBuilderPreview">${esc(compilePromptBuilder(m) || 'Fill in a few fields to build the generation prompt.')}</div>
    </div>
  `;
  $$('[data-pb-text]', root).forEach(input => input.addEventListener('input', () => {
    values[input.dataset.pbText] = input.value;
    savePromptBuilderState();
    updatePromptBuilderPreview(m);
  }));
  $$('[data-pb-chip]', root).forEach(btn => btn.addEventListener('click', () => {
    const key = btn.dataset.pbChip;
    const value = btn.dataset.pbValue;
    values[key] = values[key] === value ? '' : value;
    savePromptBuilderState();
    renderPromptBuilder(m);
  }));
}

function updatePromptBuilderPreview(m) {
  const el = $('#wsPromptBuilderPreview');
  if (!el) return;
  el.textContent = compilePromptBuilder(m) || 'Fill in a few fields to build the generation prompt.';
}

function compilePromptBuilder(m = state.selected) {
  const values = promptBuilderForModel(m);
  const parts = [];
  for (const f of PROMPT_BUILDER_TEXT_FIELDS) {
    const value = String(values[f.key] || '').trim();
    if (value) parts.push(value);
  }
  for (const g of PROMPT_BUILDER_CHIP_GROUPS) {
    const value = String(values[g.key] || '').trim();
    if (value) parts.push(value);
  }
  return parts.join(', ');
}

function currentWorkspacePrompt() {
  const m = state.selected;
  if (!m || m.category === 'llm') return state.params.prompt || '';
  if (promptModeForModel(m) === 'builder') return compilePromptBuilder(m) || state.params.prompt || '';
  return $('#wsPrompt')?.value || state.params.prompt || '';
}

function previewEnhancePrompt(prompt, m) {
  const cleaned = String(prompt || '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return '';
  const categoryCue = m?.category === 'video'
    ? 'clear subject, natural motion, consistent lighting, stable camera, coherent temporal detail'
    : 'clear subject, composition, lighting, material detail, coherent style';
  return `${cleaned}, ${categoryCue}`;
}

async function enhanceWorkspacePrompt() {
  const m = state.selected || state.models.find(x => x.id === state.workspace);
  if (!m || m.category === 'llm') return;
  const input = $('#wsPrompt');
  const original = currentWorkspacePrompt().trim();
  if (!original) return toast('Write a prompt first.', 'error');
  const btn = $('#wsEnhancePrompt');
  const oldText = btn?.textContent || 'Enhance';
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Enhancing…';
  }
  try {
    let next;
    if (state.preview) {
      await delay(350);
      next = previewEnhancePrompt(original, m);
    } else {
      const res = await api.enhancePrompt({
        prompt: original,
        model_id: m.id,
        category: m.category,
        hf_token: currentHFToken('#wsHFToken') || null,
      });
      next = (res.prompt || '').trim();
    }
    if (next && next !== original) {
      input.value = next;
      state.params.prompt = next;
      if (promptModeForModel(m) === 'builder') {
        state.promptModes[m.id] = 'raw';
        savePromptModes();
        renderPromptMode(m);
      }
      toast('Prompt enhanced', 'success');
    } else {
      toast('Prompt already looks usable.', 'info');
    }
  } catch (e) {
    toast(e.message || 'Prompt enhancer failed', 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = oldText;
    }
  }
}

async function sendLLMMessage() {
  const m = state.selected;
  if (!m || m.category !== 'llm' || state.llmBusy) return;
  const input = $('#llmPrompt');
  const prompt = (input.value || '').trim();
  if (!prompt) return;

  const session = ensureChatSession(m.id);
  input.value = '';
  await generateLLMReply(m, session, prompt);
}

async function generateLLMReply(m, session, prompt, options = {}) {
  const messages = session.messages;
  if (options.replaceFromIndex != null) {
    messages.splice(options.replaceFromIndex);
  }
  messages.push({ role: 'user', text: prompt });
  autoNameSession(session, prompt);
  const assistant = {
    role: 'assistant',
    text: '',
    thinking: state.params.thinking !== false,
    streaming: true,
  };
  messages.push(assistant);
  state.llmBusy = true;
  setModelState(m.id, { generating: true });
  updateWsStatus();
  updateLLMSendButton();
  renderLLMSessions(m);
  renderLLMThread(m);

  try {
    let text = '';
    const compactedNow = compactSessionIfNeeded(session, m);
    if (compactedNow) {
      renderLLMThread(m);
      await delay(450);
    }
    if (state.preview) {
      await delay(650);
      text = previewLLMText(prompt, state.params.thinking !== false);
    } else {
      const history = messagesForLLMPayload(session, assistant);
      const res = await generateWithBackendJob({
        model_id: m.id,
        params: {
          ...state.params,
          prompt,
          messages: history,
        },
        loras: [],
        hf_token: currentHFToken('#wsHFToken') || null,
      });
      text = res.meta?.text || res.text || '';
    }
    assistant.thinking = false;
    await streamLLMText(m, assistant, text || 'I did not get any text back from the runner.');
    session.updated_at = Date.now();
    renderLLMSessions(m);
  } catch (e) {
    assistant.thinking = false;
    assistant.streaming = false;
    assistant.text = `Generation failed: ${e.message || 'unknown error'}`;
    renderLLMThread(m);
    toast(`Text generation failed: ${e.message || 'unknown'}`, 'error');
  } finally {
    state.llmBusy = false;
    setModelState(m.id, { generating: false });
    updateWsStatus();
    updateLLMSendButton();
  }
}

function rerunFromEditedMessage(m, session, index, text) {
  if (state.llmBusy) return;
  generateLLMReply(m, session, text, { replaceFromIndex: index });
}

async function streamLLMText(modelOrId, message, text) {
  const m = typeof modelOrId === 'string' ? state.models.find(x => x.id === modelOrId) : modelOrId;
  const modelId = m?.id || modelOrId;
  message.text = '';
  message.streaming = true;
  renderLLMThread(m || modelId);
  const source = String(text || '');
  for (let i = 0; i < source.length; i += chunkSizeFor(source, i)) {
    message.text = source.slice(0, Math.min(source.length, i + chunkSizeFor(source, i)));
    renderLLMThread(m || modelId);
    await delay(source.length > 800 ? 10 : 18);
  }
  message.text = source;
  message.streaming = false;
  renderLLMThread(m || modelId);
}

function chunkSizeFor(text, index) {
  const remaining = text.length - index;
  if (remaining > 1200) return 18;
  if (remaining > 500) return 10;
  return 4;
}

function previewLLMText(prompt, thinking) {
  const trimmed = prompt.replace(/\s+/g, ' ').slice(0, 180);
  const preface = thinking
    ? 'I would start by narrowing the goal, then turn it into the smallest useful next step.'
    : 'Here is a simple first pass.';
  return `${preface}\n\nFor "${trimmed}", I would keep the interface quiet: one clear conversation column, a compact composer, and controls tucked into a collapsible panel so they are available without taking over the workspace.\n\nThe useful default is to make the model feel alive while it works: show a thinking state immediately, then reveal the answer in small chunks so the user can see progress instead of waiting on a blank screen.`;
}

function updateLLMSendButton() {
  const btn = $('#llmSend');
  const input = $('#llmPrompt');
  if (!btn || !input) return;
  btn.disabled = state.llmBusy;
  btn.innerHTML = state.llmBusy
    ? '<span class="spinner"></span><span>Thinking</span>'
    : '<svg><use href="#i-bolt"/></svg><span>Send</span>';
}

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function formatETA(ms) {
  if (!Number.isFinite(ms) || ms <= 0) return 'done';
  const totalSeconds = Math.max(1, Math.round(ms / 1000));
  if (totalSeconds < 60) return `${totalSeconds}s left`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return seconds ? `${minutes}m ${seconds}s left` : `${minutes}m left`;
}

function updateJobProgress(job, step, total) {
  if (!job || !total) return;
  const safeStep = Math.max(0, Math.min(Number(step) || 0, Number(total) || 0));
  const safeTotal = Math.max(1, Number(total) || 1);
  job.progressStep = safeStep;
  job.progressTotal = safeTotal;
  job.progress = Math.max(job.progress || 0, safeStep / safeTotal);
  if (!job.startedAt) job.startedAt = Date.now();
  if (safeStep > 0 && safeStep < safeTotal) {
    const elapsed = Date.now() - job.startedAt;
    job.etaMs = (elapsed / safeStep) * (safeTotal - safeStep);
  } else {
    job.etaMs = safeStep >= safeTotal ? 0 : null;
  }
  renderQueue({ progressOnly: true });
}

function startJobProgress(job) {
  if (state.preview || typeof EventSource === 'undefined') return null;
  const es = new EventSource(`/api/logs/${encodeURIComponent(job.model_id)}?tail=1`);
  es.onmessage = (e) => {
    const match = String(e.data || '').match(/\[gen\]\s+step\s+(\d+)\s*\/\s*(\d+)/i);
    if (!match) return;
    updateJobProgress(job, Number(match[1]), Number(match[2]));
  };
  es.onerror = () => {};
  return es;
}

async function generateWithBackendJob(payload) {
  const started = await api.startGenerateJob(payload);
  if (!started.job_id) throw new Error('Generation job did not start');
  for (;;) {
    await delay(1500);
    const job = await api.generationJob(started.job_id);
    if (job.status === 'done') return job.result || {};
    if (job.status === 'error') throw new Error(job.error || 'Generation failed');
  }
}

// ── Queue ────────────────────────────────────────────────────────────────
function snapshotJob() {
  // Snapshot current workspace state into a queueable job. Runs the same
  // payload through `generate` later — including any selected LoRAs and
  // any image inputs that are toggled on.
  const m = state.selected;
  if (!m) return null;
  const slots = state.imgInputs[m.id] || {};
  const params = { ...state.params };
  if (m.category !== 'llm') {
    params.prompt = currentWorkspacePrompt();
  }
  for (const inp of (m.image_inputs || [])) {
    const slot = slots[inp.key];
    const active = slot?.images?.[slot.idx ?? 0];
    if (slot?.enabled && active) params[inp.key === 'ref' ? 'ref_image' : inp.key] = active.path;
  }
  const lstate = state.loraStates[m.id] || {};
  // Build the LoRA payload from the per-key state. Multi-file logical LoRAs
  // (Wan Lightning ships as paired high+low files) forward their `files`
  // array verbatim so the runner loads each piece as its own adapter.
  const loras = Object.entries(lstate)
    .filter(([_, v]) => v.on && componentRequirementStatus(m.id, v.entry).ok)
    .map(([key, v]) => {
      const entry = v.entry || {};
      const hasFiles = Array.isArray(entry.files) && entry.files.length;
      if (hasFiles) {
        return {
          id: key,
          files: entry.files,
          strength_high: v.strength_high ?? v.strength ?? 1.0,
          strength_low:  v.strength_low  ?? v.strength ?? 1.0,
        };
      }
      const filename = entry.rel_path || entry.filename || key;
      // For dual-expert models, forward the per-LoRA `target` (the user's
      // explicit high/low tag, or the filename auto-detect) so the runner
      // applies the file to the matching expert. Falls back to "high" when
      // unknown — single-transformer pipes ignore it anyway.
      if (m.dual_expert) {
        const target = entry.target || entry.target_guess || undefined;
        return {
          filename,
          target,
          strength_high: v.strength_high ?? v.strength ?? 1.0,
          strength_low:  v.strength_low  ?? v.strength ?? 1.0,
        };
      }
      return { filename, strength: v.strength ?? 1.0 };
    });
  return {
    id: `q_${Date.now()}_${Math.random().toString(36).slice(2,7)}`,
    model_id: m.id,
    model_name: m.name,
    params,
    loras,
    status: 'pending',
  };
}

function enqueueCurrent() {
  const job = snapshotJob();
  if (!job) return;
  state.queue.push(job);
  toast(`Queued (${state.queue.length})`, 'info');
  renderQueue();
  if (!state.queueRunning) runQueue();
}

async function runQueue() {
  if (state.queueRunning) return;
  state.queueRunning = true;
  updateGenerateBtn();
  $('#wsCancel').disabled = false;
  while (state.queue.find(j => j.status === 'pending')) {
    const job = state.queue.find(j => j.status === 'pending');
    job.status = 'running';
    setModelState(job.model_id, { generating: true });
    renderQueue();
    updateWsStatus();
    await runJob(job);
    setModelState(job.model_id, { generating: false });
    job.status = 'done';
    renderQueue();
  }
  // Drain finished + cancelled items so the next batch starts clean.
  state.queue = state.queue.filter(j => j.status === 'pending' || j.status === 'running');
  state.queueRunning = false;
  $('#wsCancel').disabled = true;
  updateGenerateBtn();
  updateWsStatus();
  renderQueue();
  if (!state.preview) refreshAssetsSoon();
}

async function runJob(job) {
  // Poll /api/runner/{id}/preview and update the Generated cell while inference runs.
  let previewInterval = null;
  let progressStream = null;
  let previewProgressInterval = null;
  job.startedAt = Date.now();
  job.progress = 0;
  job.progressStep = 0;
  job.progressTotal = 0;
  job.etaMs = null;
  progressStream = startJobProgress(job);
  if (!state.preview) {
    previewInterval = setInterval(async () => {
      if (state.workspace !== job.model_id) return;
      try {
        const r = await fetch(`/api/runner/${job.model_id}/preview?t=${Date.now()}`, { credentials: 'same-origin' });
        if (!r.ok) return;
        const blob = await r.blob();
        const objUrl = URL.createObjectURL(blob);
        const genCell = document.querySelector('.img-cell.generated');
        if (!genCell) return;
        const content = genCell.querySelector('.img-cell-content');
        if (!content) return;
        genCell.classList.remove('empty');
        let img = content.querySelector('img');
        if (!img) {
          content.innerHTML = '';
          img = document.createElement('img');
          content.appendChild(img);
        }
        if (img._previewUrl) URL.revokeObjectURL(img._previewUrl);
        img._previewUrl = objUrl;
        img.src = objUrl;
        genCell.classList.add('has-media');
        genCell.closest('.img-strip')?.classList.add('has-media');
        updateImageCellRatios(genCell);
      } catch {}
    }, 1200);
  }

  try {
    let res;
    if (state.preview) {
      const previewTotal = 100;
      previewProgressInterval = setInterval(() => {
        const elapsed = Date.now() - job.startedAt;
        updateJobProgress(job, Math.min(94, Math.round((elapsed / 1500) * previewTotal)), previewTotal);
      }, 180);
      await new Promise(r => setTimeout(r, 1500));
      updateJobProgress(job, previewTotal, previewTotal);
      const m = state.models.find(x => x.id === job.model_id);
      const usedSeed = job.params.seed >= 0
        ? Number(job.params.seed)
        : Math.floor(Math.random() * 2147483647);
      res = { assets: [fakeAsset(m, job.params, usedSeed)], meta: { preview: true, seed: usedSeed } };
    } else {
      res = await generateWithBackendJob({
        model_id: job.model_id,
        params:   job.params,
        loras:    (job.loras || []).map(l => {
          // Multi-file logical LoRA (Wan Lightning paired high+low etc.)
          if (Array.isArray(l.files) && l.files.length) {
            return {
              files: l.files,
              strength_high: l.strength_high ?? 1.0,
              strength_low:  l.strength_low  ?? 1.0,
            };
          }
          // Dual-expert (single file with per-expert strengths). The user's
          // explicit `target` (high/low from CivitAI Wan LoRA tagging) is
          // forwarded so the runner loads the file onto the matching expert.
          if (l.strength_high !== undefined || l.strength_low !== undefined) {
            const out = {
              filename: l.filename,
              strength_high: l.strength_high ?? 1.0,
              strength_low:  l.strength_low  ?? 1.0,
            };
            if (l.target) out.target = l.target;
            return out;
          }
          // Standard single-strength
          return { filename: l.filename, strength: l.strength ?? 1.0 };
        }),
        hf_token: currentHFToken('#wsHFToken') || null,
      });
    }
    // Moderation: backend signals a flagged output by returning empty assets
    // with meta.flagged = true. Show a neutral toast and skip the rest of the
    // success path so nothing lands in recents.
    if (res.meta?.flagged) {
      toast("This output can't be generated. Please change your request and try again.", 'error');
      return;
    }
    if (res.meta?.seed != null) {
      state.lastSeed[job.model_id] = Number(res.meta.seed);
    }
    const newAssets = res.assets || [];
    state.recents[job.model_id] = [...(state.recents[job.model_id] || []), ...newAssets];
    if (state.preview) {
      state.assets = [...newAssets, ...state.assets];
      renderAssets();
      const countEl = $('#assetNavCount');
      if (countEl) countEl.textContent = state.assets.length;
    }
    if (state.workspace === job.model_id) {
      renderRecents(job.model_id);
      refreshSeedField();
      // Refresh the Generated cell on edit-style models.
      const m = state.models.find(x => x.id === job.model_id);
      if (m?.image_inputs?.length) renderImgStrip(m);
    }
  } catch (e) {
    job.error = e.message;
    toast(`Job failed: ${e.message || 'unknown'}`, 'error');
  } finally {
    if (previewInterval) clearInterval(previewInterval);
    if (previewProgressInterval) clearInterval(previewProgressInterval);
    if (progressStream) progressStream.close();
  }
}

function queueStateLabel(job) {
  return job.status === 'running'    ? 'Generating'
       : job.status === 'cancelling' ? 'Cancelling'
       :                                'Queued';
}

function queueCardBits(job) {
  const pct = Math.max(0, Math.min(100, Math.round((job.progress || 0) * 100)));
  const hasProgress = pct > 0 && job.status !== 'pending';
  const eta = job.etaMs != null && pct < 100 ? ` · ${formatETA(job.etaMs)}` : '';
  return {
    pct,
    hasProgress,
    stateLabel: queueStateLabel(job),
    metaText: job.status === 'running'
      ? (hasProgress ? `${pct}%${eta}` : 'Estimating')
      : '',
  };
}

function updateQueueCardsInPlace(root) {
  if (!root) return true;
  let complete = true;
  const openJobs = state.queue.filter(j => j.status !== 'done');
  for (const job of openJobs) {
    const card = root.querySelector(`[data-queue-job="${CSS.escape(job.id)}"]`);
    if (!card) {
      complete = false;
      continue;
    }
    const bits = queueCardBits(job);
    card.classList.toggle('active', job.status === 'running');
    card.classList.toggle('has-progress', bits.hasProgress);
    const stateEl = card.querySelector('.q-state');
    if (stateEl) stateEl.textContent = bits.stateLabel;

    let metaEl = card.querySelector('.q-meta');
    if (bits.metaText) {
      if (!metaEl) {
        metaEl = document.createElement('div');
        metaEl.className = 'q-meta';
        stateEl?.after(metaEl);
      }
      metaEl.textContent = bits.metaText;
    } else if (metaEl) {
      metaEl.remove();
    }

    let progressEl = card.querySelector('.q-progress');
    if (bits.hasProgress) {
      if (!progressEl) {
        progressEl = document.createElement('div');
        progressEl.className = 'q-progress';
        progressEl.innerHTML = '<span></span>';
        card.querySelector('.q-prompt')?.before(progressEl);
      }
      const fill = progressEl.querySelector('span');
      if (fill) fill.style.width = `${bits.pct}%`;
    } else if (progressEl) {
      progressEl.remove();
    }
  }
  return complete;
}

// Status line under Generate. Format: "Running 2 of 5 · 3 queued".
function renderQueue(opts = {}) {
  const open = state.queue.filter(j => j.status !== 'done');
  const status = $('#wsQueueStatus');
  if (!open.length) {
    status.style.display = 'none';
    status.innerHTML = '';
  } else {
    status.style.display = '';
    const total   = state.queue.length;                  // includes done
    const doneN   = state.queue.filter(j => j.status === 'done').length;
    const running = state.queue.find(j => j.status === 'running');
    const pos     = doneN + 1;                           // 1-based "now on N of M"
    const pieces = [];
    if (running) {
      const pct = Math.round((running.progress || 0) * 100);
      const eta = running.etaMs != null && pct > 0 && pct < 100 ? ` · ${formatETA(running.etaMs)}` : '';
      const progress = pct > 0 ? ` · ${pct}%${eta}` : '';
      pieces.push(`<span class="pulse-dot"></span><span>Running ${pos} of ${total}${progress}</span>`);
    }
    const pending = open.filter(j => j.status === 'pending').length;
    if (pending) pieces.push(`<span class="sep">·</span><span>${pending} queued</span>`);
    status.innerHTML = pieces.join('');
  }
  // Queue cards live inside the recents grid (prepended). During progress
  // updates, mutate the queue card in place so existing recent media elements
  // are not destroyed/recreated every step, which makes video thumbnails flicker.
  if (opts.progressOnly) {
    let recentsOk = true;
    let assetsOk = true;
    if (state.workspace) recentsOk = updateQueueCardsInPlace($('#wsRecent'));
    if ($('.view.active')?.dataset.view === 'assets') assetsOk = updateQueueCardsInPlace($('#assetGrid'));
    if (recentsOk && assetsOk) return;
  }
  if (state.workspace) renderRecents(state.workspace);
  if ($('.view.active')?.dataset.view === 'assets') renderAssets();
}

async function cancelQueueJob(jobId) {
  const job = state.queue.find(j => j.id === jobId);
  if (!job) return;
  if (job.status === 'running') {
    // Ask the runner to interrupt; runQueue will mark it done after.
    try { await api.cancel(job.model_id); } catch {}
    job.status = 'cancelling';
  } else {
    // Pending — just drop it from the queue.
    state.queue = state.queue.filter(j => j.id !== jobId);
  }
  renderQueue();
}

// ── Lightbox ────────────────────────────────────────────────────────────
// State for the lightbox's prev/next nav. When openLightbox is called with
// a list + index, ◀ ▶ buttons cycle through that list; otherwise it's a
// single-image view with the buttons hidden.
let _lightboxList = null;
let _lightboxIdx = 0;

function openLightbox(asset, opts = {}) {
  if (!asset) return;
  // Backward-compat: callers can still pass just an asset. New callers can
  // pass {list, index} so the lightbox knows what to scroll through.
  if (Array.isArray(opts.list) && opts.list.length) {
    _lightboxList = opts.list;
    _lightboxIdx  = Math.max(0, Math.min(opts.list.length - 1, opts.index ?? opts.list.indexOf(asset)));
  } else {
    _lightboxList = null;
    _lightboxIdx  = 0;
  }
  _renderLightbox();
  const lb = $('#lightbox');
  lb.classList.add('open');
  lb.setAttribute('aria-hidden', 'false');
}

function _renderLightbox() {
  const asset = _lightboxList ? _lightboxList[_lightboxIdx] : _lightboxCurrentAsset;
  if (!asset) return;
  _lightboxCurrentAsset = asset;
  $('#lightboxMeta').textContent = _lightboxList
    ? `${_lightboxIdx + 1} / ${_lightboxList.length} · ${asset.name || asset.path || ''}`
    : (asset.name || asset.path || '');
  const stage = $('#lightboxStage');
  const u = esc(asset.url);
  const navArrows = (_lightboxList && _lightboxList.length > 1) ? `
    <button class="lightbox-nav prev" id="lightboxPrev" aria-label="Previous">‹</button>
    <button class="lightbox-nav next" id="lightboxNext" aria-label="Next">›</button>
  ` : '';
  stage.innerHTML = (asset.kind === 'video'
    ? `<video src="${u}" controls autoplay loop playsinline preload="metadata"></video>`
    : `<img src="${u}" alt="${esc(asset.name || '')}">`) + navArrows;
  $('#lightboxFoot').textContent = asset.path || '';
  $('#lightboxDownload').onclick = () => {
    downloadAsset(asset);
  };
  $('#lightboxPrev')?.addEventListener('click', (e) => { e.stopPropagation(); _lightboxStep(-1); });
  $('#lightboxNext')?.addEventListener('click', (e) => { e.stopPropagation(); _lightboxStep(1);  });
}

let _lightboxCurrentAsset = null;

function _lightboxStep(delta) {
  if (!_lightboxList || !_lightboxList.length) return;
  _lightboxIdx = (_lightboxIdx + delta + _lightboxList.length) % _lightboxList.length;
  _renderLightbox();
}
function closeLightbox() {
  const lb = $('#lightbox');
  lb.classList.remove('open');
  lb.setAttribute('aria-hidden', 'true');
  $('#lightboxStage').innerHTML = '';
}

// Single-button flow: clicking Generate always pushes a job onto the queue.
// If nothing's running, runQueue() picks it up immediately. If something IS
// running, the job goes behind it and the button label flips to "Queue" so
// the user can see clicks land. The queue runner refreshes the label each
// time it ticks.
function onGenerate() {
  const m = state.selected;
  if (!m) return;
  // Required-input gate for edit-style models.
  for (const inp of (m.image_inputs || [])) {
    if (inp.required) {
      const slot = (state.imgInputs[m.id] || {})[inp.key];
      if (!slot?.enabled || !slot.images?.length) {
        toast(`Add a ${inp.label} image first`, 'error');
        return;
      }
    }
  }
  enqueueCurrent();
  $('#wsCancel').disabled = !state.queueRunning;
}

function updateGenerateBtn() {
  const label = $('#wsGenerateLabel');
  const btn   = $('#wsGenerate');
  if (!label || !btn) return;
  if (state.queueRunning) {
    label.textContent = 'Queue';
    btn.firstElementChild.outerHTML = '<svg><use href="#i-plus"/></svg>';
  } else {
    label.textContent = 'Generate';
    btn.firstElementChild.outerHTML = '<svg><use href="#i-bolt"/></svg>';
  }
}

// Build a placeholder "result" image so the Recent strip has something to show.
// Pure SVG data URL — text says the prompt, dimensions match params.
function fakeAsset(m, params) {
  const w = Math.min(1024, params.width  || 1024);
  const h = Math.min(1024, params.height || 1024);
  const prompt = (params.prompt || 'preview').slice(0, 80);
  const ts  = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#1a1a20"/>
        <stop offset="100%" stop-color="#0a0a0c"/>
      </linearGradient>
    </defs>
    <rect width="${w}" height="${h}" fill="url(#g)"/>
    <text x="50%" y="48%" fill="rgba(255,255,255,0.85)" font-family="-apple-system,Outfit,sans-serif" font-size="${Math.round(w/22)}" text-anchor="middle">${prompt.replace(/[<>&"]/g, '')}</text>
    <text x="50%" y="56%" fill="rgba(255,255,255,0.35)" font-family="ui-monospace,monospace" font-size="${Math.round(w/40)}" text-anchor="middle">${m.name} · ${w}×${h} · preview</text>
  </svg>`;
  const url = 'data:image/svg+xml;utf8,' + encodeURIComponent(svg);
  return {
    name: `preview_${ts}.svg`,
    path: `assets/generated/preview_${ts}.svg`,
    url,
    kind: 'image',
    source: 'generated',
  };
}

async function cancelGenerate() {
  const m = state.selected;
  if (!m) return;
  try { await api.cancel(m.id); toast('Cancelling…', 'info'); } catch {}
}

async function waitForRunner(modelId, maxSeconds) {
  const deadline = Date.now() + maxSeconds * 1000;
  while (Date.now() < deadline) {
    let h;
    try {
      h = await api.runnerHealth(modelId);
    } catch {
      // Network error — runner not yet listening, keep trying.
      await new Promise(r => setTimeout(r, 1500));
      continue;
    }
    if (h.ready) return true;
    // load_error is a definitive failure — stop immediately instead of waiting the full timeout.
    if (h.load_error) throw new Error(h.load_error);
    await new Promise(r => setTimeout(r, 1500));
  }
  return false;
}

// ── Running ──────────────────────────────────────────────────────────────
async function pollRunning() {
  try {
    const gated = await enforceLiveAuthGate();
    if (!gated) {
      state.running = await api.status();
      const live = Object.values(state.running).filter(r => r.status === 'running').length;
      const liveIds = new Set(Object.entries(state.running)
        .filter(([, info]) => info.status === 'running')
        .map(([id]) => id));
      $('#runningDot').style.display = live > 0 ? 'block' : 'none';
      // Sync modelStates so a runner that became ready while the drawer was closed
      // shows as ready immediately on reopen, without needing a manual interaction.
      for (const [id, info] of Object.entries(state.running)) {
        if (info.status === 'running') {
          const s = modelStateOf(id);
          if (!s.ready) setModelState(id, { ready: true, starting: false, downloaded: true, error: null });
        }
      }
      for (const [id, s] of Object.entries(state.modelStates)) {
        if (!liveIds.has(id) && (s.ready || s.starting || s.generating)) {
          setModelState(id, { ready: false, starting: false, generating: false });
        }
      }
      if (state.workspace) updateWsStatus();
      if ($('.view.active')?.dataset.view === 'models') renderModels();
      if ($('.view.active')?.dataset.view === 'running') renderRunning();
    }
  } catch {}
  setTimeout(pollRunning, 3000);
}

function renderRunning() {
  const list = $('#runList');
  const entries = Object.values(state.running || {});
  if (!entries.length) {
    list.innerHTML = `<div class="empty">Nothing running. Launch one from Models.</div>`;
    return;
  }
  list.innerHTML = entries.map(r => `
    <div class="run-card" data-id="${r.model_id}">
      <div class="run-head">
        <div class="run-status ${r.status !== 'running' ? 'stopped' : ''}"></div>
        <div>
          <div class="run-name">${r.name}</div>
          <div class="run-pid">PID ${r.pid} · 127.0.0.1:${r.port}</div>
        </div>
        <div class="run-actions">
          <button class="btn sm danger" data-act="stop" data-id="${r.model_id}"><svg><use href="#i-stop"/></svg>Stop</button>
        </div>
      </div>
      <div class="log" id="log_${r.model_id}"></div>
    </div>`).join('');
  list.querySelectorAll('.run-head').forEach(h => h.addEventListener('click', () => {
    const card = h.closest('.run-card');
    const log = card.querySelector('.log');
    log.classList.toggle('open');
    if (log.classList.contains('open') && !log.dataset.streaming) {
      log.dataset.streaming = '1';
      streamLogs(card.dataset.id, log);
    }
  }));
  list.querySelectorAll('[data-act="stop"]').forEach(b => b.addEventListener('click', async (e) => {
    e.stopPropagation();
    toast('Stopping…', 'info');
    try {
      await api.stop(b.dataset.id);
      markModelStopped(b.dataset.id);
      toast('Runner stopped', 'success');
    } catch (err) {
      toast(`Stop failed: ${err.message || 'unknown'}`, 'error');
    }
  }));
}

function streamLogs(id, el) {
  const es = new EventSource(`/api/logs/${id}`);
  es.onmessage = (e) => {
    const line = document.createElement('div');
    line.className = 'log-line' + (e.data.toLowerCase().includes('error') ? ' err' : '');
    line.textContent = e.data;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
  };
  es.onerror = () => es.close();
}

// ── LoRAs ────────────────────────────────────────────────────────────────
async function loadLoras() {
  try {
    const data = await api.loras();
    state.loras = data.loras || [];
  } catch { state.loras = []; }
  $('#loraNavCount').textContent = state.loras.length;
  renderLoraPanel();
}

// ── Components (transformer / VAE / text-encoder split-file swaps) ──────
async function loadComponents() {
  try {
    const data = await api.components();
    state.components = data.components || [];
  } catch { state.components = []; }
}

function isComponentInstalled(opt) {
  if (!opt?.filename || !opt?.target) return false;
  const basename = String(opt.filename).split('/').pop();
  return (state.components || []).some(c => c.target === opt.target && c.filename === basename);
}

// Poll a component-install background job until it finishes. Updates the
// install button's label with downloaded MB, then on success refreshes
// state.components, auto-selects the freshly-installed component, and
// re-renders the drawer. Polling continues even if the user closes/opens
// the drawer because we look up the button by data-component-install on
// each tick.
const _componentJobTimers = {};
function pollComponentInstall(jobId, m, opt, initialBtn) {
  if (_componentJobTimers[jobId]) return; // already polling
  const optId = opt.id;
  const tick = async () => {
    let job;
    try {
      job = await api.componentJobStatus(jobId);
    } catch (err) {
      toast(`Install lost: ${err.message || 'status unreachable'}`, 'error');
      delete _componentJobTimers[jobId];
      return;
    }
    const btn = document.querySelector(`[data-component-install="${CSS.escape(optId)}"]`) || initialBtn;
    if (job.status === 'running' || job.status === 'queued') {
      if (btn) {
        const mb = job.downloaded_mb;
        const label = job.status === 'queued'
          ? 'Queued…'
          : (mb ? `Downloading… ${mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB'}` : 'Downloading…');
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner"></span><span>${esc(label)}</span>`;
      }
      _componentJobTimers[jobId] = setTimeout(tick, 4000);
      return;
    }
    delete _componentJobTimers[jobId];
    if (job.status === 'done') {
      const errors = (job.results || []).filter(r => r.status === 'error');
      if (errors.length) {
        toast(`Install failed: ${errors[0].error || 'unknown'}`, 'error');
      } else {
        toast(`${opt.label} installed`, 'success');
        const cur = modelStateOf(m.id).components || {};
        cur[opt.target] = String(opt.filename).split('/').pop();
        setModelState(m.id, { components: { ...cur } });
      }
    } else if (job.status === 'error') {
      toast(`Install failed: ${job.error || 'unknown'}`, 'error');
    }
    await loadComponents();
    if (state.selected?.id === m.id) renderDrawerBody();
  };
  tick();
}

function renderLoraPanel() {
  const list = $('#loraList');
  $('#loraCount').textContent = `${state.loras.length} file${state.loras.length === 1 ? '' : 's'}`;
  if (!state.loras.length) {
    list.innerHTML = `<div class="empty" style="padding:18px"><span style="font-size:11.5px">No LoRAs yet</span></div>`;
    return;
  }
  list.innerHTML = state.loras.map(l => {
    const id = loraId(l);
    const fn = esc(id);
    const label = esc(l.rel_path || l.filename);
    // Dual-expert target: explicit user setting wins; otherwise show the
    // filename-based guess so the user can confirm or override. Wan
    // CivitAI LoRAs ship as paired _high.safetensors / _low.safetensors,
    // so the auto-detect handles the common case for free.
    const cur = l.target || '';
    const guess = l.target_guess || '';
    const opt = (val, label) =>
      `<option value="${val}" ${cur === val ? 'selected' : ''}>${label}${!cur && guess === val ? ' · auto' : ''}</option>`;
    return `<div class="lora-row">
      <span class="name" title="${label}">${label}</span>
      <span class="size">${l.size_mb}MB</span>
      <select class="lora-target-select" data-target-for="${fn}" title="Dual-expert target (Wan etc.)">
        <option value="" ${!cur ? 'selected' : ''}>—${guess ? ` (auto: ${guess})` : ''}</option>
        ${opt('high', 'high')}
        ${opt('low',  'low')}
      </select>
      <button class="del" data-del="${fn}" title="Delete"><svg style="width:13px;height:13px"><use href="#i-trash"/></svg></button>
    </div>`;
  }).join('');
  list.querySelectorAll('[data-del]').forEach(b => b.addEventListener('click', async () => {
    if (!await confirmModal('Delete LoRA', `Delete ${b.dataset.del}?`, 'Delete', true)) return;
    await api.deleteLora(b.dataset.del);
    await loadLoras();
    toast('LoRA deleted');
  }));
  // Persist target choice via PATCH so the runner sees it on the next launch.
  list.querySelectorAll('[data-target-for]').forEach(s => s.addEventListener('change', async () => {
    const fn  = s.dataset.targetFor;
    const lora = state.loras.find(x => loraId(x) === fn);
    try {
      await api.patchLora(fn, {
        tags: lora?.tags || [],
        target: s.value,                  // empty string clears
      });
      // Refresh local state so the workspace immediately knows.
      if (lora) lora.target = s.value || undefined;
      toast(s.value ? `Set ${fn} → ${s.value}-noise` : 'Cleared target', 'success');
    } catch (e) {
      toast(`Failed to update target: ${e.message || 'unknown'}`, 'error');
    }
  }));
}

// ── CivitAI ──────────────────────────────────────────────────────────────
async function civSearch() {
  const q = $('#civSearch').value;
  const key = state.settings.civitai_key || '';
  const out = $('#civResults');
  out.innerHTML = `<div class="empty" style="grid-column:1/-1"><div class="spinner"></div></div>`;
  try {
    const data = await api.civSearch(q, key);
    state.civResults = data.items || [];
    renderCivResults();
  } catch {
    out.innerHTML = `<div class="empty" style="grid-column:1/-1">Failed to reach CivitAI.</div>`;
  }
}

function renderCivResults() {
  const out = $('#civResults');
  if (!state.civResults.length) {
    out.innerHTML = `<div class="empty" style="grid-column:1/-1">No results.</div>`;
    return;
  }
  out.innerHTML = state.civResults.map((m, i) => {
    const img = m.modelVersions?.[0]?.images?.[0]?.url || '';
    const dl = m.stats?.downloadCount || 0;
    const r = m.stats?.rating?.toFixed(1) || '—';
    return `<div class="civ-card" data-idx="${i}">
      <div class="civ-thumb">${img ? `<img src="${esc(img)}" loading="lazy" alt="${esc(m.name)}">` : ''}</div>
      <div class="civ-info">
        <div class="civ-name">${esc(m.name)}</div>
        <div class="civ-stats"><span>↓ ${fmtNum(dl)}</span><span>★ ${esc(r)}</span></div>
      </div>
    </div>`;
  }).join('');
  out.querySelectorAll('.civ-card').forEach(c =>
    c.addEventListener('click', () => openCivDetail(Number(c.dataset.idx))));
}

// ── CivitAI: paste URL → version picker modal ────────────────────────────
//
// Accepts any URL of the shape  https://civitai.com/models/<id>[/slug][?modelVersionId=<vid>]
// Pulls the model record from /api/civitai/model/<id> and opens the picker.
function parseCivitaiUrl(input) {
  if (!input) return null;
  const trimmed = input.trim();
  // Bare id (e.g. "12345") is fine too.
  if (/^\d+$/.test(trimmed)) return { modelId: Number(trimmed), versionId: null };
  let u;
  try { u = new URL(trimmed); } catch { return null; }
  const m = u.pathname.match(/\/models\/(\d+)/);
  if (!m) return null;
  const versionId = u.searchParams.get('modelVersionId');
  return {
    modelId:   Number(m[1]),
    versionId: versionId ? Number(versionId) : null,
  };
}

function selectCivitaiFile(files = []) {
  const safeFiles = files.filter(f => f?.name?.toLowerCase().endsWith('.safetensors'));
  const pool = safeFiles.length ? safeFiles : files.filter(f => f?.name);
  return [...pool].sort((a, b) => {
    const primary = Number(!!b.primary) - Number(!!a.primary);
    if (primary) return primary;
    return Number(b.sizeKB || 0) - Number(a.sizeKB || 0);
  })[0];
}

async function civOpenByUrl() {
  const raw = $('#civUrl').value;
  const parsed = parseCivitaiUrl(raw);
  if (!parsed) return toast('Paste a civitai.com/models/… URL or numeric id', 'error');

  $('#modalTitle').textContent = 'Loading…';
  $('#modalMeta').textContent  = `civitai.com/models/${parsed.modelId}`;
  $('#modalBody').innerHTML    = `<div class="empty"><div class="spinner"></div></div>`;
  $('#modalFoot').innerHTML    = `<button class="btn ghost" id="modalCancel">Close</button>`;
  $('#modalCancel').onclick = closeModal;
  openModal();

  try {
    const m = await api.civModel(parsed.modelId, state.settings.civitai_key || '');
    openCivVersionPicker(m, parsed.versionId);
  } catch (e) {
    $('#modalBody').innerHTML = `<div class="empty">Failed to load model. ${esc(e.message || '')}</div>`;
  }
}

function openCivVersionPicker(model, preselectVersionId) {
  $('#modalTitle').textContent = model.name || 'CivitAI model';
  $('#modalMeta').textContent  = `by ${model.creator?.username || 'unknown'} · ${model.type || ''}`;
  const versions = model.modelVersions || [];

  if (!versions.length) {
    $('#modalBody').innerHTML = `<div class="empty">This model has no published versions.</div>`;
    $('#modalFoot').innerHTML = `<button class="btn ghost" id="modalCancel">Close</button>`;
    $('#modalCancel').onclick = closeModal;
    return;
  }

  // Preselect either the version id from the URL or the newest one.
  const initialIdx = preselectVersionId
    ? Math.max(0, versions.findIndex(v => v.id === preselectVersionId))
    : 0;

  // Strip HTML tags from the description and then escape what remains.
  // The regex strip alone isn't enough (`<<script>script>` gets through);
  // textContent-style escaping after the strip closes that gap.
  const desc = (model.description || '').replace(/<[^>]+>/g, '').slice(0, 220);
  const descMore = (model.description || '').length > 220 ? '…' : '';

  $('#modalBody').innerHTML = `
    <div style="font-size:12.5px;color:var(--t-3);line-height:1.55;margin-bottom:12px">
      ${esc(desc)}${descMore}
    </div>

    <div class="field-stack" id="civVersionList" style="max-height:54vh;overflow:auto">
      ${versions.map((v, i) => {
        const thumb = v.images?.[0]?.url || '';
        const file  = selectCivitaiFile(v.files || []);
        const fname = file?.name || '—';
        const fsize = file?.sizeKB ? `${(file.sizeKB / 1024).toFixed(1)} MB` : '';
        const checked = i === initialIdx ? 'checked' : '';
        return `<label class="check-row ${checked ? 'on' : ''}" data-vrow="${i}">
          <input type="radio" name="civVersion" value="${i}" ${checked}>
          ${thumb
            ? `<img src="${esc(thumb)}" alt="" style="width:46px;height:46px;border-radius:6px;object-fit:cover;flex-shrink:0">`
            : `<div style="width:46px;height:46px;border-radius:6px;background:var(--glass-2);flex-shrink:0"></div>`}
          <div style="flex:1;min-width:0">
            <div style="font-size:12.5px;font-weight:500;color:var(--t-1)">${esc(v.name)}${i === 0 ? ' · latest' : ''}</div>
            <div style="font-family:var(--font-mono);font-size:10.5px;color:var(--t-3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(fname)}${fsize ? ' · ' + fsize : ''}</div>
          </div>
          <div style="font-family:var(--font-mono);font-size:10px;color:var(--t-4)">${esc(v.baseModel || '')}</div>
        </label>`;
      }).join('')}
    </div>

    <div class="field" style="margin-top:14px">
      <label class="field-label">Tags <span class="hint">used to auto-suggest with models</span></label>
      <input class="text-input" id="civTags" placeholder="comma,separated,tags" value="${esc((model.tags || []).join(', '))}">
    </div>`;

  $('#modalFoot').innerHTML = `
    <button class="btn ghost" id="modalCancel">Cancel</button>
    <button class="btn primary" id="civDownloadBtn"><svg><use href="#i-down"/></svg>Download</button>`;

  // Light visual sync between the radio tick and `.check-row.on` highlight.
  $$('input[name="civVersion"]', $('#modalBody')).forEach(r => r.addEventListener('change', () => {
    $$('.check-row[data-vrow]', $('#modalBody')).forEach(el => el.classList.toggle('on', el.contains(r) && r.checked));
  }));

  $('#modalCancel').onclick = closeModal;
  $('#civDownloadBtn').onclick = async () => {
    const sel = document.querySelector('input[name="civVersion"]:checked');
    const vIdx = sel ? Number(sel.value) : 0;
    const v = versions[vIdx];
    const tags = $('#civTags').value.split(',').map(t => t.trim()).filter(Boolean);
    const file = selectCivitaiFile(v.files || []);
    if (!file) return toast('No downloadable file in this version', 'error');
    try {
      await api.civDownload({
        model_version_id: v.id,
        filename:         file.name,
        model_id:         (model.type || 'lora').toLowerCase(),
        tags,
        api_key:          state.settings.civitai_key || null,
      });
      toast(`Downloading ${file.name}…`);
      closeModal();
      // Civitai downloads land in /workspace/loras — same poll behaviour as HF.
      watchForNewLoras(model.name || 'CivitAI');
    } catch (e) {
      toast(`Download failed: ${e.message || ''}`, 'error');
    }
  };
}

function openCivDetail(idx) {
  const m = state.civResults[idx];
  if (!m) return;
  $('#modalTitle').textContent = m.name;
  $('#modalMeta').textContent  = `by ${m.creator?.username || 'unknown'} · ${m.type}`;
  const versions = m.modelVersions || [];
  const desc = (m.description || '').replace(/<[^>]+>/g,'').slice(0, 220);
  const descMore = (m.description || '').length > 220 ? '…' : '';
  $('#modalBody').innerHTML = `
    <p style="font-size:12.5px;color:var(--t-3);line-height:1.55;margin-bottom:12px">
      ${esc(desc)}${descMore}
    </p>
    <div class="field-stack">
      <div class="field">
        <label class="field-label">Version</label>
        <select class="select" id="civVersionSelect">
          ${versions.map((v, i) => `<option value="${i}">${esc(v.name)} · ${esc(v.baseModel || '')}</option>`).join('')}
        </select>
      </div>
      <div class="field">
        <label class="field-label">Tags <span class="hint">used to auto-suggest with models</span></label>
        <input class="text-input" id="civTags" placeholder="comma,separated,tags" value="${esc((m.tags||[]).join(', '))}">
      </div>
    </div>`;
  $('#modalFoot').innerHTML = `
    <button class="btn ghost" id="modalCancel">Cancel</button>
    <button class="btn primary" id="modalDownload"><svg><use href="#i-down"/></svg>Download</button>`;
  $('#modalCancel').onclick = closeModal;
  $('#modalDownload').onclick = async () => {
    const vIdx = Number($('#civVersionSelect').value);
    const v = versions[vIdx];
    const tags = $('#civTags').value.split(',').map(t => t.trim()).filter(Boolean);
    const file = selectCivitaiFile(v.files || []);
    if (!file) return toast('No downloadable file found', 'error');
    try {
      await api.civDownload({
        model_version_id: v.id,
        filename: file.name,
        model_id: m.type.toLowerCase(),
        tags,
        api_key: state.settings.civitai_key || null,
      });
      toast(`Downloading ${file.name}…`);
      closeModal();
      setTimeout(loadLoras, 1500);
    } catch { toast('Download failed', 'error'); }
  };
  openModal();
}

// ── Assets ───────────────────────────────────────────────────────────────
async function loadAssets() {
  if (state.preview) {
    // Preview mode keeps assets entirely in-memory (uploads + fake gens).
    // Don't reach for a backend that isn't there — that would just wipe
    // everything the user added when they switch to the Assets tab.
    $('#assetNavCount').textContent = state.assets.length;
    renderAssets();
    return;
  }
  try {
    const data = await api.assets();
    state.assets = data.assets || [];
  } catch {
    // Keep the current grid instead of blanking it if the proxy/API stalls.
    return;
  }
  $('#assetNavCount').textContent = state.assets.length;
  renderAssets();
}

function refreshAssetsSoon() {
  setTimeout(() => {
    loadAssets().catch(() => {});
  }, 250);
}

function renderAssets() {
  const grid = $('#assetGrid');
  const f = state.assetFilter;
  const queued = state.queue.filter(j => j.status !== 'done');
  const items = state.assets.filter(a => {
    if (f === 'all') return true;
    if (f === 'image' || f === 'video') return a.kind === f;
    return a.source === f;
  });
  if (!items.length && !queued.length) {
    grid.innerHTML = `<div class="empty" style="grid-column:1/-1">Nothing here yet.</div>`;
    return;
  }
  const queueHtml = queueCardsHtml(queued, { showModel: true });
  const assetHtml = items.map((a, i) => {
    const u = esc(a.url), n = esc(a.name);
    return `<div class="asset" data-asset-idx="${i}">
      ${a.kind === 'video'
        ? `<video src="${u}" muted playsinline preload="metadata"></video>`
        : `<img src="${u}" loading="lazy" alt="${n}">`}
      <span class="asset-kind">${esc(a.source)}</span>
      <button class="asset-menu-btn" data-asset-menu="${i}" data-asset-source="library" title="Actions">
        <svg><use href="#i-dots"/></svg>
      </button>
      <div class="asset-meta">${n}</div>
    </div>`;
  }).join('');
  grid.innerHTML = queueHtml + assetHtml;
  bindQueueCardMenus(grid);
  // Click an asset → lightbox; click the 3-dot → menu
  $$('.asset[data-asset-idx]', grid).forEach(el => el.addEventListener('click', (e) => {
    if (e.target.closest('[data-asset-menu]')) return;
    const idx = Number(el.dataset.assetIdx);
    openLightbox(items[idx], { list: items, index: idx });
  }));
  $$('[data-asset-menu]', grid).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    const idx = Number(b.dataset.assetMenu);
    showAssetMenu(items[idx], b);
  }));
}

async function uploadFiles(files) {
  if (!files?.length) return;
  toast(`Uploading ${files.length}…`);
  for (const f of files) {
    if (state.preview) {
      const url = URL.createObjectURL(f);
      state.assets.unshift({ name: f.name, url, path: f.name, kind: f.type.startsWith('video') ? 'video' : 'image', source: 'upload' });
    } else {
      try { await api.uploadAsset(f); } catch {}
    }
  }
  if (!state.preview) await loadAssets();
  else {
    renderAssets();
    const countEl = $('#assetNavCount');
    if (countEl) countEl.textContent = state.assets.length;
  }
  toast('Upload complete', 'success');
}

// ── HF download ──────────────────────────────────────────────────────────
// Advanced "type the exact file path" path. Used when the user already
// knows the rel_path inside the repo and doesn't want to open the browser.
// The flow now goes through the new job-backed endpoint, so progress shows
// up in the queue panel just like browser-driven downloads.
async function startHFDownload() {
  const repo = $('#hfRepo').value.trim();
  if (!repo) return toast('Enter a repo ID', 'error');
  const file = $('#hfFile').value.trim();
  if (!file) return toast('Enter a file path inside the repo, or use Browse', 'error');
  const target = $('#hfTargetDir')?.value || 'models';
  const token  = currentHFToken('#hfTokenDl');
  try {
    const cleaned = parseHFRepoInput(repo);
    await api.hfDownload({
      repo_id: cleaned.repo,
      revision: cleaned.revision,
      files:    [{ rel_path: file, target_dir: target }],
      hf_token: token || null,
    });
    toast(`Queued ${file}`, 'success');
    pokeDownloadQueuePoll();
  } catch (e) {
    toast(`Failed to queue: ${e.message || 'unknown'}`, 'error');
  }
}

// Accepts username/repo, full https URLs, and tree/branch URLs.
//   "Lightricks/LTX-2.3"
//   "https://huggingface.co/Lightricks/LTX-2.3"
//   "https://huggingface.co/Lightricks/LTX-2.3/tree/main"
function parseHFRepoInput(raw) {
  const value = String(raw || '').trim();
  let repo = value, revision = 'main';
  const m = value.match(/huggingface\.co\/([^/]+\/[^/?#]+)(?:\/(?:tree|blob)\/([^/?#]+))?/i);
  if (m) {
    repo = m[1];
    if (m[2]) revision = m[2];
  }
  return { repo, revision };
}

// ── HF Browser modal ────────────────────────────────────────────────────
function openHFBrowser(initialRepo) {
  const scrim = $('#hfBrowserScrim');
  if (!scrim) return;
  scrim.classList.add('open');
  scrim.setAttribute('aria-hidden', 'false');
  // Pre-fill from the Downloads page's repo input if the user is using the
  // top-level Browse button. Empty otherwise.
  const inp = $('#hfBrowserRepo');
  if (inp) {
    inp.value = initialRepo || $('#hfRepo')?.value || '';
    inp.focus();
  }
  // Reset selection and filter when reopening — keeps state clean across
  // independent browse sessions.
  state.hfBrowse.selected = new Set();
  state.hfBrowse.filter = '';
  $('#hfBrowserFilter').value = '';
  if (initialRepo || inp.value) loadHFFiles();
  else renderHFBrowserBody();
  updateHFBrowserSummary();
}

function closeHFBrowser() {
  const scrim = $('#hfBrowserScrim');
  if (!scrim) return;
  scrim.classList.remove('open');
  scrim.setAttribute('aria-hidden', 'true');
}

async function loadHFFiles() {
  const raw = $('#hfBrowserRepo').value.trim();
  if (!raw) return;
  const { repo, revision } = parseHFRepoInput(raw);
  const token = currentHFToken('#hfTokenDl');
  state.hfBrowse = { ...state.hfBrowse, repo, revision, files: [], selected: new Set(), loading: true, error: null };
  renderHFBrowserBody();
  try {
    const data = await api.hfFiles(repo, revision, token);
    state.hfBrowse.files   = data.files || [];
    state.hfBrowse.loading = false;
  } catch (e) {
    state.hfBrowse.loading = false;
    state.hfBrowse.error   = e.message || 'Failed to fetch files';
  }
  renderHFBrowserBody();
  updateHFBrowserSummary();
}

function renderHFBrowserBody() {
  const body = $('#hfBrowserBody');
  if (!body) return;
  const s = state.hfBrowse;
  if (s.loading) {
    body.innerHTML = `<div class="empty" style="padding:18px"><span class="spinner"></span> Loading files…</div>`;
    return;
  }
  if (s.error) {
    body.innerHTML = `<div class="empty hf-error" style="padding:18px"><svg><use href="#i-x"/></svg> ${esc(s.error)}</div>`;
    return;
  }
  if (!s.files.length && !s.repo) {
    body.innerHTML = `<div class="empty" style="padding:18px">Enter a repo and click Go.</div>`;
    return;
  }
  if (!s.files.length) {
    body.innerHTML = `<div class="empty" style="padding:18px">No safetensors / ckpt / bin / gguf files found in <span class="mono">${esc(s.repo)}</span>.</div>`;
    return;
  }
  const filt = (s.filter || '').toLowerCase();
  const visible = filt ? s.files.filter(f => f.rel_path.toLowerCase().includes(filt)) : s.files;
  if (!visible.length) {
    body.innerHTML = `<div class="empty" style="padding:18px">No matches for "${esc(s.filter)}".</div>`;
    return;
  }
  body.innerHTML = visible.map(f => {
    const checked = s.selected.has(f.rel_path) ? 'checked' : '';
    const sizeStr = f.size ? formatBytes(f.size) : '?';
    const lfs = f.lfs ? '<span class="lfs-pill">LFS</span>' : '';
    return `<label class="hf-file-row ${checked ? 'selected' : ''}">
      <input type="checkbox" data-hf-file="${esc(f.rel_path)}" ${checked}>
      <span class="hf-file-name" title="${esc(f.rel_path)}">${esc(f.rel_path)}</span>
      <span class="hf-file-size">${sizeStr}</span>
      ${lfs}
    </label>`;
  }).join('');
  $$('[data-hf-file]', body).forEach(cb => cb.addEventListener('change', () => {
    const path = cb.dataset.hfFile;
    if (cb.checked) state.hfBrowse.selected.add(path);
    else            state.hfBrowse.selected.delete(path);
    cb.closest('.hf-file-row')?.classList.toggle('selected', cb.checked);
    updateHFBrowserSummary();
  }));
}

function updateHFBrowserSummary() {
  const s = state.hfBrowse;
  const sel = s.selected;
  const totalBytes = s.files.filter(f => sel.has(f.rel_path)).reduce((acc, f) => acc + (f.size || 0), 0);
  const summary = $('#hfBrowserSummary');
  if (summary) {
    summary.textContent = sel.size
      ? `${sel.size} selected · ${formatBytes(totalBytes)}`
      : 'No files selected';
  }
  const btn = $('#hfBrowserDownload');
  if (btn) {
    btn.disabled = sel.size === 0;
    btn.textContent = sel.size ? `Download ${sel.size} selected` : 'Download 0 selected';
  }
}

async function submitHFBrowserDownload() {
  const s = state.hfBrowse;
  if (!s.selected.size) return;
  const dest = $('#hfBrowserDest')?.value || 'models';
  const token = currentHFToken('#hfTokenDl');
  const files = [...s.selected].map(rel_path => ({ rel_path, target_dir: dest }));
  try {
    await api.hfDownload({
      repo_id:  s.repo,
      revision: s.revision || 'main',
      files,
      hf_token: token || null,
    });
    toast(`Queued ${files.length} file${files.length === 1 ? '' : 's'} from ${s.repo}`, 'success');
    closeHFBrowser();
    pokeDownloadQueuePoll();
  } catch (e) {
    toast(`Failed to queue: ${e.message || 'unknown'}`, 'error');
  }
}

// ── Download queue panel ────────────────────────────────────────────────
// Polls /api/hf/jobs while there are active jobs OR while the user is on
// the Downloads view. Stops when everything is settled to keep idle pods
// quiet. `pokeDownloadQueuePoll` kicks the poller awake after a new job
// is queued so we don't wait up to 4s for the first render tick.

let _dlPollTimer = null;
let _dlCompletedSeen = new Set();
let _dlDismissed = new Set();

function pokeDownloadQueuePoll() {
  if (_dlPollTimer) { clearTimeout(_dlPollTimer); _dlPollTimer = null; }
  pollDownloadQueue();
}

async function pollDownloadQueue() {
  let data;
  try {
    data = await api.hfJobs();
  } catch {
    // Backoff on network errors — try again in 5s.
    _dlPollTimer = setTimeout(pollDownloadQueue, 5000);
    return;
  }
  const now = Date.now();
  // Compute rolling MB/s per job using the previous poll's bytes.
  for (const j of data.jobs || []) {
    const prev = state.downloadStats[j.id];
    if (prev && j.status === 'running' && j.downloaded_bytes >= prev.prevBytes) {
      const dt = (now - prev.prevAt) / 1000;
      const db = j.downloaded_bytes - prev.prevBytes;
      if (dt > 0) j.mbps = (db / 1024 / 1024) / dt;
    }
    state.downloadStats[j.id] = { prevBytes: j.downloaded_bytes || 0, prevAt: now, mbps: j.mbps };
  }
  state.downloadJobs = (data.jobs || []).filter(j => !_dlDismissed.has(j.id));
  await refreshCompletedDownloadTargets(state.downloadJobs);
  renderDownloadQueue();
  // Keep polling while any job is queued/running, or every 4s while the
  // Downloads view is the active route (so newly queued items show up
  // even when the previous batch finished).
  const hasActive = state.downloadJobs.some(j => j.status === 'queued' || j.status === 'running');
  const onDownloads = state.view === 'downloads';
  const interval = hasActive ? 1500 : (onDownloads ? 4000 : 0);
  if (interval) {
    _dlPollTimer = setTimeout(pollDownloadQueue, interval);
  } else {
    _dlPollTimer = null;
  }
}

async function refreshCompletedDownloadTargets(jobs) {
  const completed = (jobs || []).filter(j => j.status === 'done' && !_dlCompletedSeen.has(j.id));
  if (!completed.length) return;
  completed.forEach(j => _dlCompletedSeen.add(j.id));
  const targets = new Set(completed.map(j => j.target_dir));
  if (targets.has('loras')) {
    const byRepo = new Map();
    for (const j of completed.filter(j => j.target_dir === 'loras')) {
      const key = `${j.repo}@@${j.revision || 'main'}`;
      if (!byRepo.has(key)) byRepo.set(key, { repo_id: j.repo, revision: j.revision || 'main', rel_paths: [] });
      byRepo.get(key).rel_paths.push(j.rel_path);
    }
    for (const req of byRepo.values()) {
      try {
        await api.hfLoraImportDone({ ...req, hf_token: currentHFToken('#hfTokenDl') || null });
      } catch {}
    }
    await loadLoras();
  }
  if (targets.has('components')) await loadComponents();
  if (targets.has('models') || targets.has('checkpoints')) {
    await Promise.all((state.models || []).map(m => refreshWeightState(m)));
    renderModels();
    if (state.selected) renderDrawerBody();
  }
}

function renderDownloadQueue() {
  const wrap = $('#dlQueueWrap');
  const list = $('#dlQueueList');
  if (!wrap || !list) return;
  // Show the panel only on the Downloads page or when there's actually
  // something happening — keeps other views unchanged.
  const onDownloads = state.view === 'downloads';
  const visible = onDownloads || state.downloadJobs.some(j => j.status === 'queued' || j.status === 'running');
  if (!visible || !state.downloadJobs.length) {
    wrap.style.display = 'none';
    return;
  }
  wrap.style.display = '';
  list.innerHTML = state.downloadJobs.map(renderDLJob).join('');
  // Wire row buttons.
  $$('[data-dl-cancel]', list).forEach(b => b.addEventListener('click', () => cancelDownloadJob(b.dataset.dlCancel)));
  $$('[data-dl-retry]',  list).forEach(b => b.addEventListener('click', () => retryDownloadJob(b.dataset.dlRetry)));
  $$('[data-dl-dismiss]', list).forEach(b => b.addEventListener('click', () => dismissDownloadJob(b.dataset.dlDismiss)));
}

function renderDLJob(j) {
  const total  = j.total_bytes || 0;
  const got    = j.downloaded_bytes || 0;
  const pct    = total > 0 ? Math.min(100, (got / total) * 100) : (j.status === 'done' ? 100 : 0);
  const sizes  = total > 0 ? `${formatBytes(got)} / ${formatBytes(total)}` : (got ? formatBytes(got) : '');
  const mbps   = j.mbps && j.status === 'running' ? ` · ${j.mbps.toFixed(1)} MB/s` : '';
  const stateLabel = {
    queued:    'Queued',
    running:   'Downloading…',
    done:      'Done',
    error:     'Failed',
    cancelled: 'Cancelled',
  }[j.status] || j.status;
  const action = j.status === 'queued' || j.status === 'running'
    ? `<button class="btn xs" data-dl-cancel="${esc(j.id)}" title="Cancel"><svg><use href="#i-x"/></svg></button>`
    : j.status === 'error' || j.status === 'cancelled'
    ? `<button class="btn xs" data-dl-retry="${esc(j.id)}">Retry</button>
       <button class="btn xs" data-dl-dismiss="${esc(j.id)}" title="Dismiss"><svg><use href="#i-x"/></svg></button>`
    : `<button class="btn xs" data-dl-dismiss="${esc(j.id)}" title="Dismiss"><svg><use href="#i-x"/></svg></button>`;
  const errRow = j.status === 'error' && j.error
    ? `<div class="dl-job-message">${esc(j.error)}</div>`
    : '';
  return `<div class="dl-job dl-job-${j.status}">
    <div class="dl-job-meta">
      <div class="dl-job-name" title="${esc(j.repo)} · ${esc(j.rel_path)}">${esc(j.rel_path)}</div>
      <div class="dl-job-sub">${esc(j.repo)} · ${esc(j.target_dir)} · ${esc(stateLabel)}${mbps}</div>
    </div>
    <div class="dl-job-bar"><div class="dl-job-fill" style="width:${pct.toFixed(1)}%"></div></div>
    <div class="dl-job-side">
      <div class="dl-job-sizes">${sizes}</div>
      <div class="dl-job-actions">${action}</div>
    </div>
    ${errRow}
  </div>`;
}

async function cancelDownloadJob(id) {
  try {
    await api.hfJobCancel(id);
    pokeDownloadQueuePoll();
  } catch (e) {
    toast(`Cancel failed: ${e.message || 'unknown'}`, 'error');
  }
}

async function retryDownloadJob(id) {
  try {
    await api.hfJobRetry(id);
    toast('Retrying…', 'info');
    pokeDownloadQueuePoll();
  } catch (e) {
    toast(`Retry failed: ${e.message || 'unknown'}`, 'error');
  }
}

function dismissDownloadJob(id) {
  _dlDismissed.add(id);
  state.downloadJobs = state.downloadJobs.filter(j => j.id !== id);
  delete state.downloadStats[id];
  renderDownloadQueue();
  api.hfJobCancel(id).catch(() => {});
}

async function clearCompletedDownloads() {
  const done = state.downloadJobs.filter(j => j.status !== 'queued' && j.status !== 'running');
  done.forEach(j => {
    _dlDismissed.add(j.id);
    delete state.downloadStats[j.id];
    api.hfJobCancel(j.id).catch(() => {});
  });
  state.downloadJobs = state.downloadJobs.filter(j => j.status === 'queued' || j.status === 'running');
  renderDownloadQueue();
}

function formatBytes(n) {
  if (!n || n < 1024) return `${n || 0} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

async function watchForNewLoras(label) {
  const startCount = state.loras.length;
  const startNames = new Set(state.loras.map(loraId));
  const deadline = Date.now() + 3 * 60 * 1000;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 5000));
    try { await loadLoras(); } catch { continue; }
    const newOnes = state.loras.filter(l => !startNames.has(loraId(l)));
    if (newOnes.length) {
      toast(`${label}: ${newOnes.length} new LoRA${newOnes.length === 1 ? '' : 's'} ready`, 'success');
      return;
    }
    if (state.loras.length > startCount) return;  // count drifted (e.g., delete) — stop
  }
}

// ── Modal helpers ────────────────────────────────────────────────────────
let modalResolve = null;

function openModal()  { $('#modalScrim').classList.add('open'); }
function closeModal() {
  $('#modalScrim').classList.remove('open');
  if (modalResolve) { modalResolve(false); modalResolve = null; }
}

function confirmModal(title, text, confirmLabel = 'OK', danger = false) {
  return new Promise(resolve => {
    modalResolve = resolve;
    $('#modalTitle').textContent = title;
    $('#modalMeta').textContent  = '';
    $('#modalBody').innerHTML = `<div style="font-size:13px;color:var(--t-2);line-height:1.55">${text}</div>`;
    $('#modalFoot').innerHTML = `
      <button class="btn ghost" id="modalCancel">Cancel</button>
      <button class="btn ${danger ? 'danger' : 'primary'}" id="modalConfirm">${confirmLabel}</button>`;
    
    $('#modalCancel').onclick = () => { modalResolve = null; closeModal(); resolve(false); };
    $('#modalConfirm').onclick = () => { modalResolve = null; closeModal(); resolve(true); };
    openModal();
  });
}

// ── Mock data (for previewing without backend) ───────────────────────────
// Mirrors backend/model_registry.json so the dev preview renders the same
// cards as production. Append entries here when adding new runners.
function mockModels() {
  return [
    {
      id: 'qwen-image', name: 'Qwen-Image', category: 'image',
      description: "Alibaba's Qwen-Image text-to-image. Strong prompt adherence, native 1328×1328. ~40 GB on first download.",
      runner_module: 'backend.runners.qwen_image',
      hf_repo: 'Qwen/Qwen-Image',
      min_vram_gb: 14, recommended_vram_gb: 47,
      supports_lora: true, gpu_support: ['nvidia'],
      default_size_by_vram: [
        { min_gb: 47, w: 1328, h: 1328 },
        { min_gb: 36, w: 1024, h: 1024 },
        { min_gb: 0,  w: 768,  h: 768  },
      ],
      quants: [
        { id: 'bf16', label: 'BF16', description: 'Best quality',        vram_gb: 47, auto_min_vram_gb: 47 },
        { id: 'int8', label: 'INT8', description: 'Faster, near-bf16',   vram_gb: 24, auto_min_vram_gb: 36 },
        { id: 'nf4',  label: 'NF4',  description: 'Fits any modern card',vram_gb: 14, auto_min_vram_gb: 0  },
      ],
      param_groups: ['prompt', 'generation', 'dimensions'],
      param_keys:   ['prompt', 'negative_prompt', 'seed', 'steps', 'cfg', 'width', 'height'],
      param_overrides: {
        steps:  { default: 30 },
        cfg:    { default: 4.0, min: 0, max: 10, step: 0.1 },
        width:  { default: 1024, step: 16, min: 256, max: 2048 },
        height: { default: 1024, step: 16, min: 256, max: 2048 },
      },
      aspect_presets: [
        { label: 'Square',          w: 1328, h: 1328 },
        { label: 'Landscape 16:9',  w: 1664, h: 928  },
        { label: 'Portrait 9:16',   w: 928,  h: 1664 },
        { label: '3:4',             w: 1152, h: 1536 },
        { label: '4:3',             w: 1536, h: 1152 },
      ],
      default_loras: [
        { filename: 'Qwen-Image-Lightning-4steps-V2.0.safetensors', label: 'Lightning · 4 steps', strength: 1.0, default_on: false },
        { filename: 'Qwen-Image-Lightning-8steps-V2.0.safetensors', label: 'Lightning · 8 steps', strength: 1.0, default_on: false },
      ],
    },
    {
      id: 'qwen-image-edit', name: 'Qwen-Image-Edit', category: 'image',
      description: "Alibaba's original Qwen-Image-Edit — instruction-based image editing with one reference image. Use the 2511 card for the newer consistency model.",
      runner_module: 'backend.runners.qwen_image_edit',
      hf_repo: 'Qwen/Qwen-Image-Edit',
      min_vram_gb: 14, recommended_vram_gb: 47,
      supports_lora: true, gpu_support: ['nvidia'],
      default_size_by_vram: [
        { min_gb: 47, w: 1328, h: 1328 },
        { min_gb: 36, w: 1024, h: 1024 },
        { min_gb: 0,  w: 768,  h: 768  },
      ],
      quants: [
        { id: 'bf16', label: 'BF16', description: 'Best quality',        vram_gb: 47, auto_min_vram_gb: 47 },
        { id: 'int8', label: 'INT8', description: 'Faster, near-bf16',   vram_gb: 24, auto_min_vram_gb: 36 },
        { id: 'nf4',  label: 'NF4',  description: 'Fits any modern card',vram_gb: 14, auto_min_vram_gb: 0  },
      ],
      image_inputs: [
        { key: 'ref', label: 'Reference', required: true, hint: 'What to keep from the original' },
      ],
      param_groups: ['prompt', 'generation', 'dimensions'],
      param_keys:   ['prompt', 'negative_prompt', 'seed', 'steps', 'cfg', 'width', 'height'],
      param_overrides: {
        steps:  { default: 30 },
        cfg:    { default: 4.0, min: 0, max: 10, step: 0.1 },
        width:  { default: 1024, step: 16, min: 256, max: 2048 },
        height: { default: 1024, step: 16, min: 256, max: 2048 },
      },
      aspect_presets: [
        { label: 'Square',          w: 1328, h: 1328 },
        { label: 'Landscape 16:9',  w: 1664, h: 928  },
        { label: 'Portrait 9:16',   w: 928,  h: 1664 },
        { label: '3:4',             w: 1152, h: 1536 },
        { label: '4:3',             w: 1536, h: 1152 },
      ],
      default_loras: [],
    },
    {
      id: 'qwen-image-edit-2511', name: 'Qwen-Image-Edit 2511', category: 'image',
      description: 'Official 2511 refresh of Qwen-Image-Edit. Better identity consistency, multi-person edits, reduced drift, and built-in support for popular edit behaviours.',
      runner_module: 'backend.runners.qwen_image_edit_2511',
      hf_repo: 'Qwen/Qwen-Image-Edit-2511',
      min_vram_gb: 14, recommended_vram_gb: 47,
      supports_lora: true, gpu_support: ['nvidia'],
      default_size_by_vram: [
        { min_gb: 47, w: 1328, h: 1328 },
        { min_gb: 36, w: 1024, h: 1024 },
        { min_gb: 0,  w: 768,  h: 768  },
      ],
      quants: [
        { id: 'bf16', label: 'BF16', description: 'Best quality',        vram_gb: 47, auto_min_vram_gb: 47 },
        { id: 'int8', label: 'INT8', description: 'Faster, near-bf16',   vram_gb: 24, auto_min_vram_gb: 36 },
        { id: 'nf4',  label: 'NF4',  description: 'Fits any modern card',vram_gb: 14, auto_min_vram_gb: 0  },
      ],
      image_inputs: [
        { key: 'ref', label: 'Reference', required: true, hint: 'Subject, scene or product to preserve' },
      ],
      param_groups: ['prompt', 'generation', 'dimensions'],
      param_keys:   ['prompt', 'negative_prompt', 'seed', 'steps', 'cfg', 'width', 'height'],
      param_overrides: {
        steps:  { default: 40 },
        cfg:    { default: 4.0, min: 0, max: 10, step: 0.1, label: 'Guidance' },
        width:  { default: 1024, step: 16, min: 256, max: 2048 },
        height: { default: 1024, step: 16, min: 256, max: 2048 },
      },
      aspect_presets: [
        { label: 'Square',          w: 1328, h: 1328 },
        { label: 'Landscape 16:9',  w: 1664, h: 928  },
        { label: 'Portrait 9:16',   w: 928,  h: 1664 },
        { label: '3:4',             w: 1152, h: 1536 },
        { label: '4:3',             w: 1536, h: 1152 },
      ],
      default_loras: [
        {
          filename: 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors',
          label: 'Lightning · 4 steps',
          usage_hint: 'Set Steps to 4 and CFG to 1.0',
          strength: 1.0, default_on: false,
        },
        {
          filename: 'Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors',
          label: 'Lightning · 8 steps',
          usage_hint: 'Set Steps to 8 and CFG to 1.0',
          strength: 1.0, default_on: false,
        },
        {
          filename: 'qwen-image-edit-2511-multiple-angles-lora.safetensors',
          label: 'Multiple angles',
          strength: 0.8, default_on: false,
        },
      ],
    },
  ];
}

init().catch((err) => {
  console.error('App failed to start:', err);
  openAuthGate('login', 'The app failed to start. Refresh once the pod is ready.');
});
