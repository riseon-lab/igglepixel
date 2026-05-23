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
  gpuStatus:     'loading',
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
  activePreviews: {},         // model_id -> focused asset in the hero preview
  imgInputs:     {},          // { model_id -> { ref:{img,enabled}, pose:{...}, ... } }
  loraStates:    readJSONStorage('forge_lora_states', {}), // persisted per-model LoRA toggles + strengths
  trainers:      [],
  trainJobs:     [],
  trainPoll:     null,
  trainValidation: null,
  datasets:      [],
  activeDataset: null,
  datasetItems:  [],
  datasetFilter: 'all',
  datasetSelected: -1,
  datasetBusy:   false,
  datasetVisionPoll: null,
  previewDatasetStore: {},
  components:    [],          // installed split-file components: [{target, filename, size_mb, path}]
  runtimes:      {},          // { model_id -> {state: 'missing'|'installing'|'ready'|'error', last_line?, error?} }
  runtimeDepsPoll: null,
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
  weights: {
    title: 'Weights',
    fields: [
      { key: 'ltx_weight_repo', label: 'HF repo', type: 'text',
        placeholder: 'Lightricks/LTX-2.3-fp8' },
      { key: 'ltx_weight_name', label: 'Weight filename', type: 'hf_file',
        placeholder: 'ltx-2.3-22b-distilled-fp8.safetensors' },
      { key: 'ltx_weight_pipeline', label: 'Pipeline', type: 'select',
        options: ['distilled', 'two_stage'], default: 'distilled' },
      { key: 'ltx_weight_prequantized_fp8', label: 'Pre-quantized FP8', type: 'select',
        options: ['true', 'false'], default: 'true' },
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

const COLOUR_OPTIONS = [
  'black', 'white', 'ivory', 'cream', 'charcoal', 'silver', 'grey', 'navy', 'cobalt blue',
  'sky blue', 'emerald green', 'forest green', 'olive', 'sage', 'red', 'burgundy', 'rose',
  'pink', 'lavender', 'purple', 'mustard yellow', 'gold', 'copper', 'orange', 'rust',
  'tan', 'camel', 'brown', 'oxblood', 'gunmetal', 'pearl', 'mint green', 'electric blue',
  'transparent', 'iridescent', 'holographic', 'neon', 'pastel', 'monochrome',
  'Aerochrome false-colour infrared'
];

const MATERIAL_OPTIONS = [
  'cotton', 'linen', 'silk', 'satin', 'velvet', 'chiffon', 'lace', 'wool', 'cashmere',
  'denim', 'leather', 'suede', 'latex', 'vinyl', 'mesh', 'organza', 'tulle', 'knit',
  'ribbed fabric', 'technical nylon', 'waterproof fabric', 'metallic fabric', 'sequins',
  'embroidered fabric', 'sheer fabric', 'matte fabric', 'glossy fabric', 'distressed fabric',
  'jersey fabric', 'fleece', 'canvas', 'corduroy', 'tweed', 'brocade', 'lame metallic fabric', 'neoprene',
  'patent leather', 'faux fur', 'feather trim', 'rubber', 'crochet', 'chainmail'
];

const PROMPT_BUILDER_SECTIONS = [
  {
    id: 'subject',
    title: 'Subject',
    fields: [
      { key: 'idea', label: 'Core idea', type: 'textarea', wide: true, placeholder: 'person, product, place, creature, character, object...' },
      { key: 'subjectCount', label: 'Subjects', type: 'select', options: ['single subject', 'two subjects', 'small group', 'crowd', 'subject with pet', 'subject with vehicle'] },
      { key: 'presentation', label: 'Presentation', type: 'select', options: ['feminine presentation', 'masculine presentation', 'androgynous presentation', 'gender-neutral styling', 'fashion model look', 'everyday person look', 'fantasy character look'] },
      { key: 'age', label: 'Age', type: 'select', options: ['adult 18-24', 'adult 25-34', 'adult 35-44', 'adult 45-54', 'adult 55-64', 'senior adult', 'elderly adult', 'ageless fantasy character'] },
      { key: 'bodyType', label: 'Body type', type: 'select', options: ['slim build', 'lean build', 'athletic build', 'toned build', 'muscular build', 'soft build', 'curvy build', 'plus-size build', 'broad-shouldered build', 'petite build', 'tall build', 'average build', 'willowy build', 'stocky build', 'lithe build', 'statuesque build'] },
      { key: 'silhouette', label: 'Silhouette', type: 'select', options: ['clean simple silhouette', 'hourglass silhouette', 'boxy silhouette', 'long vertical silhouette', 'wide-shoulder silhouette', 'narrow-waist silhouette', 'layered silhouette', 'oversized silhouette', 'tailored silhouette', 'flowing silhouette'] },
      { key: 'posture', label: 'Posture', type: 'select', options: ['relaxed posture', 'confident posture', 'upright posture', 'casual slouch', 'elegant posture', 'dynamic posture', 'balanced stance', 'contrapposto stance', 'guarded posture', 'playful posture', 'commanding posture', 'resting posture'] },
      { key: 'subjectRole', label: 'Role / archetype', type: 'select', options: ['fashion editorial subject', 'street style subject', 'athlete', 'performer', 'artist', 'musician', 'detective', 'explorer', 'scientist', 'royal figure', 'warrior', 'pilot', 'mechanic', 'chef', 'student', 'office worker', 'traveller', 'witch', 'spacefarer'] },
      { key: 'subjectDetail', label: 'Extra subject detail', type: 'text', wide: true, placeholder: 'distinctive silhouette, era, occupation, character notes...' },
    ],
  },
  {
    id: 'face-hair',
    title: 'Face, Hair, Details',
    fields: [
      { key: 'expression', label: 'Expression', type: 'select', options: ['neutral expression', 'soft smile', 'wide smile', 'serious expression', 'confident expression', 'pensive expression', 'curious expression', 'surprised expression', 'dreamy expression', 'intense gaze', 'laughing', 'subtle smirk', 'calm expression', 'melancholic expression', 'closed-mouth smile', 'open-mouth smile', 'raised eyebrow', 'focused expression', 'playful expression', 'defiant expression', 'tired expression', 'serene expression'] },
      { key: 'head', label: 'Head direction', type: 'select', options: ['looking at camera', 'looking left', 'looking right', 'looking up', 'looking down', 'looking over shoulder', 'eyes closed', 'gaze off-frame', 'chin lifted', 'chin lowered', 'looking past camera', 'looking into mirror', 'head tilted left', 'head tilted right', 'profile gaze', 'downcast eyes'] },
      { key: 'hairColor', label: 'Hair colour', type: 'select', options: ['black hair', 'dark brown hair', 'brown hair', 'chestnut hair', 'auburn hair', 'red hair', 'ginger hair', 'copper hair', 'strawberry blonde hair', 'dark blonde hair', 'blonde hair', 'platinum blonde hair', 'silver hair', 'grey hair', 'white hair', 'pastel pink hair', 'blue hair', 'green hair', 'purple hair', 'two-tone hair', 'dark roots', 'balayage hair', 'ombre hair', 'highlighted hair', 'shaved head', 'bald head'] },
      { key: 'hairStyle', label: 'Hair style', type: 'select', options: ['long straight hair', 'long wavy hair', 'long curly hair', 'shoulder-length hair', 'short bob', 'lob haircut', 'pixie cut', 'buzz cut', 'undercut', 'slicked-back hair', 'middle part hair', 'side part hair', 'messy hair', 'braided hair', 'loose braid', 'box braids', 'cornrows', 'high ponytail', 'low ponytail', 'bun', 'messy bun', 'space buns', 'afro', 'locs', 'wet-look hair', 'windblown hair', 'fringe bangs', 'curtain bangs'] },
      { key: 'makeup', label: 'Make-up', type: 'select', options: ['no visible make-up', 'natural make-up', 'soft glam make-up', 'editorial make-up', 'smoky eye make-up', 'winged eyeliner', 'graphic eyeliner', 'glossy lips', 'matte lips', 'bold red lip', 'nude lip', 'metallic eye shadow', 'glitter make-up', 'goth make-up', 'avant-garde face paint', 'dewy skin finish', 'matte skin finish', 'sun-kissed blush', 'highlighter glow'] },
      { key: 'piercings', label: 'Piercings', type: 'select', options: ['no piercings visible', 'ear piercings', 'multiple ear piercings', 'nose ring', 'nose stud', 'septum piercing', 'lip piercing', 'eyebrow piercing', 'industrial piercing', 'helix piercings', 'facial piercing set', 'minimal jewellery piercings'] },
      { key: 'tattoos', label: 'Tattoos / marks', type: 'select', options: ['no visible tattoos', 'small tattoos', 'sleeve tattoo', 'neck tattoo', 'hand tattoos', 'delicate line tattoos', 'geometric tattoos', 'floral tattoos', 'script tattoo', 'freckles', 'beauty mark', 'scars', 'birthmark', 'paint marks', 'glitter on skin'] },
      { key: 'faceDetail', label: 'Extra face detail', type: 'text', wide: true, placeholder: 'eye colour, freckles, facial hair, skin finish, jewellery detail...' },
    ],
  },
  {
    id: 'wardrobe',
    title: 'Wardrobe',
    fields: [
      { key: 'outfit', label: 'Full outfit', type: 'select', options: ['tailored suit', 'evening gown', 'cocktail dress', 'sundress', 'slip dress', 'linen resort outfit', 'streetwear outfit', 'casual outfit', 'oversized sweater and jeans', 'athleisure outfit', 'workwear outfit', 'bohemian outfit', 'punk outfit', 'goth outfit', 'minimal monochrome outfit', 'high-fashion runway look', 'cyberpunk outfit', 'sci-fi armour', 'fantasy robe', 'formal uniform', 'raincoat outfit', 'winter outfit', 'summer outfit', 'swimwear outfit', 'festival outfit'] },
      { key: 'fullColor', label: 'Full colour', type: 'select', options: COLOUR_OPTIONS },
      { key: 'fullMaterial', label: 'Full material', type: 'select', options: MATERIAL_OPTIONS },
      { key: 'outfitFit', label: 'Fit / silhouette', type: 'select', options: ['tailored fit', 'relaxed fit', 'oversized fit', 'body-skimming fit', 'structured silhouette', 'flowing silhouette', 'layered styling', 'minimal clean styling', 'cinched waist fit', 'boxy fit', 'draped fit', 'asymmetrical styling'] },
      { key: 'pattern', label: 'Pattern', type: 'select', options: ['solid colour', 'floral print', 'stripes', 'pinstripes', 'plaid', 'houndstooth', 'polka dots', 'geometric pattern', 'animal print', 'abstract print', 'embroidered detail', 'lace detail', 'logo-free garment', 'subtle texture pattern'] },
      { key: 'topGarment', label: 'Top', type: 'select', options: ['t-shirt', 'tank top', 'crop top', 'button-up shirt', 'oversized shirt', 'blouse', 'hoodie', 'sweater', 'cardigan', 'blazer', 'bomber jacket', 'leather jacket', 'denim jacket', 'trench coat', 'corset top', 'bodysuit', 'turtleneck', 'kimono jacket', 'sports jersey', 'utility vest', 'puffer jacket', 'halter top'] },
      { key: 'topColor', label: 'Top colour', type: 'select', options: COLOUR_OPTIONS },
      { key: 'topMaterial', label: 'Top material', type: 'select', options: MATERIAL_OPTIONS },
      { key: 'bottomGarment', label: 'Bottom', type: 'select', options: ['jeans', 'straight-leg jeans', 'wide-leg trousers', 'tailored trousers', 'cargo pants', 'parachute pants', 'shorts', 'tailored shorts', 'mini skirt', 'midi skirt', 'maxi skirt', 'pleated skirt', 'pencil skirt', 'slip skirt', 'leggings', 'joggers', 'leather pants', 'utility skirt', 'flowing fabric pants'] },
      { key: 'bottomColor', label: 'Bottom colour', type: 'select', options: COLOUR_OPTIONS },
      { key: 'bottomMaterial', label: 'Bottom material', type: 'select', options: MATERIAL_OPTIONS },
      { key: 'accessories', label: 'Accessories', type: 'select', options: ['no accessories', 'minimal jewellery', 'statement earrings', 'layered necklaces', 'rings', 'bracelets', 'watch', 'belt', 'gloves', 'scarf', 'silk scarf', 'hat', 'wide-brim hat', 'beanie', 'baseball cap', 'sunglasses', 'eyeglasses', 'handbag', 'backpack', 'crossbody bag', 'clutch bag', 'umbrella', 'headphones', 'hair clips', 'choker necklace'] },
      { key: 'shoes', label: 'Shoes', type: 'select', options: ['barefoot', 'white sneakers', 'black boots', 'ankle boots', 'knee-high boots', 'combat boots', 'cowboy boots', 'loafers', 'dress shoes', 'heels', 'platform heels', 'kitten heels', 'sandals', 'strappy sandals', 'running shoes', 'hiking boots', 'futuristic shoes', 'delicate flats', 'Mary Jane shoes'] },
      { key: 'shoeColor', label: 'Shoe colour', type: 'select', options: COLOUR_OPTIONS },
      { key: 'clothingDetail', label: 'Extra clothing detail', type: 'text', wide: true, placeholder: 'fit, layering, patterns, logos to avoid, wear level, styling notes...' },
    ],
  },
  {
    id: 'scene',
    title: 'Scene',
    fields: [
      { key: 'scene', label: 'Scene', type: 'select', options: ['studio backdrop', 'minimal interior', 'luxury apartment', 'living room', 'bedroom', 'bathroom mirror', 'kitchen', 'office', 'workshop', 'art studio', 'gallery', 'museum', 'library', 'cafe', 'restaurant', 'vintage diner', 'record store', 'hotel lobby', 'rooftop', 'city street', 'neon alley', 'subway station', 'train platform', 'parking garage', 'gym', 'nightclub', 'garden patio', 'greenhouse', 'park path', 'forest', 'beach', 'poolside', 'lake shore', 'desert', 'mountain path', 'rainy street', 'snowy landscape', 'futuristic city', 'medieval village', 'spaceship interior', 'fantasy castle', 'concert stage'] },
      { key: 'timeOfDay', label: 'Time', type: 'select', options: ['early morning', 'golden hour', 'midday', 'late afternoon', 'sunset', 'blue hour', 'night', 'midnight', 'overcast day', 'rainy afternoon', 'late-night interior', 'dawn light', 'twilight'] },
      { key: 'weather', label: 'Weather', type: 'select', options: ['clear weather', 'light rain', 'heavy rain', 'mist', 'fog', 'snow', 'wind', 'storm clouds', 'humid haze', 'dust in the air', 'drizzle', 'wet pavement after rain', 'falling leaves', 'sunlit haze', 'sea breeze'] },
      { key: 'background', label: 'Background', type: 'select', options: ['clean background', 'soft blurred background', 'busy detailed background', 'architectural background', 'natural landscape background', 'industrial background', 'luxury background', 'cluttered lived-in background', 'symmetrical background', 'deep background layers', 'shallow background layers', 'bokeh city lights', 'plain paper backdrop', 'painted canvas backdrop', 'mirror reflection background'] },
      { key: 'sceneMood', label: 'Atmosphere', type: 'select', options: ['minimal editorial set', 'cozy lived-in atmosphere', 'busy urban atmosphere', 'romantic atmosphere', 'moody cinematic atmosphere', 'clean commercial set', 'dreamlike surreal atmosphere', 'quiet intimate atmosphere', 'high-energy nightlife atmosphere', 'nostalgic atmosphere', 'luxury boutique atmosphere'] },
      { key: 'props', label: 'Props', type: 'select', options: ['no props', 'flowers', 'book', 'phone', 'camera', 'laptop', 'microphone', 'musical instrument', 'coffee cup', 'drink glass', 'shopping bags', 'vehicle', 'chair', 'stool', 'sofa', 'mirror', 'neon sign', 'smoke machine', 'floating objects', 'vinyl record', 'paintbrush', 'skateboard', 'bicycle', 'suitcase', 'flowers in vase', 'sword', 'tool kit', 'sports equipment'] },
      { key: 'sceneDetail', label: 'Extra scene detail', type: 'text', wide: true, placeholder: 'set dressing, background objects, era, location specifics...' },
    ],
  },
  {
    id: 'pose-action',
    title: 'Pose & Action',
    fields: [
      { key: 'pose', label: 'Pose', type: 'select', options: ['standing', 'sitting', 'walking', 'running', 'kneeling', 'crouching', 'leaning', 'leaning against wall', 'lying down', 'reclining pose', 'seated on floor', 'perched on stool', 'jumping', 'dancing', 'turning around', 'over-the-shoulder pose', 'looking back over shoulder', 'crossed arms', 'arms at sides', 'hands in pockets', 'hands on hips', 'one hand on hip', 'arms raised', 'reaching forward', 'resting on chair', 'mid-stride', 'walking across frame', 'balanced action pose', 'fashion editorial pose', 'lookbook stance'] },
      { key: 'bodyAxis', label: 'Body facing', type: 'select', options: ['facing camera', '3/4 view', 'side profile', 'back-facing', 'turned left', 'turned right', 'twisted torso', 'diagonal body line', 'centred subject', 'off-centre subject', 'weight shifted to one leg', 'shoulders angled toward camera', 'hips angled away from camera'] },
      { key: 'handPose', label: 'Hands', type: 'select', options: ['relaxed hands', 'hands visible', 'hands clasped', 'holding object', 'touching face', 'fingertips near face', 'touching necklace', 'adjusting glasses', 'adjusting hair', 'adjusting jacket', 'holding clothing hem', 'hand on railing', 'pointing', 'reaching', 'waving', 'one hand raised', 'hands behind back', 'hands in pockets'] },
      { key: 'action', label: 'Action', type: 'select', options: ['standing still', 'walking toward camera', 'walking away', 'walking across frame', 'turning to camera', 'talking', 'laughing', 'reading', 'writing', 'typing', 'taking a photo', 'holding a drink', 'playing music', 'dancing', 'stretching', 'looking through a window', 'opening a door', 'climbing stairs', 'riding a bike', 'driving', 'checking phone', 'putting on sunglasses', 'adjusting outfit', 'casting a spell', 'fighting stance', 'floating', 'running through rain'] },
      { key: 'motion', label: 'Motion feel', type: 'select', options: ['frozen moment', 'subtle movement', 'natural motion blur', 'dynamic motion', 'wind in clothing', 'hair moving in wind', 'fabric flowing', 'splashing water', 'floating particles', 'cinematic stillness', 'gentle body movement', 'sharp decisive gesture', 'graceful movement', 'candid in-between moment'] },
      { key: 'actionDetail', label: 'Extra action detail', type: 'text', wide: true, placeholder: 'gesture, interaction, object being used, exact body language...' },
    ],
  },
  {
    id: 'camera',
    title: 'Camera',
    fields: [
      { key: 'distance', label: 'Subject distance', type: 'select', options: ['extreme close-up', 'close-up', 'headshot', 'bust shot', 'waist-up shot', 'half body', 'three-quarter body', 'full body', 'wide shot', 'extreme wide shot'] },
      { key: 'camera', label: 'Camera angle', type: 'select', options: ['eye-level angle', 'low angle', 'slightly low angle', 'high angle', 'slightly high angle', 'waist-level angle', 'ground-level angle', 'overhead angle', 'bird\'s-eye view', 'worm-eye view', 'Dutch angle', 'front view', '3/4 camera view', 'side profile view', 'back view', 'mirror reflection angle', 'over-the-shoulder view', 'point-of-view shot'] },
      { key: 'lens', label: 'Lens', type: 'select', options: ['24mm wide-angle lens', '28mm environmental portrait lens', '35mm documentary lens', '50mm natural lens', '70mm portrait lens', '85mm portrait lens', '100mm macro lens', '135mm telephoto lens', '200mm compression lens', 'anamorphic lens', 'fisheye lens', 'tilt-shift lens', 'macro close-up lens'] },
      { key: 'framing', label: 'Framing', type: 'select', options: ['centred composition', 'rule of thirds', 'symmetrical framing', 'negative space', 'tight crop', 'loose crop', 'full-body lookbook framing', 'centered full-body with headroom', 'three-quarter portrait crop', 'close portrait crop', 'environmental portrait framing', 'foreground frame', 'layered foreground', 'leading lines', 'diagonal composition', 'low horizon line', 'high horizon line'] },
      { key: 'depthOfField', label: 'Depth of field', type: 'select', options: ['deep focus', 'shallow depth of field', 'creamy bokeh', 'background separation', 'foreground blur', 'rack focus feel', 'macro depth of field', 'soft background falloff', 'everything sharp', 'cinematic focus rolloff'] },
      { key: 'cameraMove', label: 'Video camera move', type: 'select', categories: ['video'], options: ['static camera', 'locked-off tripod shot', 'slow push-in', 'slow pull-back', 'dolly in', 'dolly out', 'tracking shot', 'gimbal follow shot', 'sideways truck left', 'sideways truck right', 'orbit shot', 'crane up', 'crane down', 'pan left', 'pan right', 'tilt up', 'tilt down', 'handheld camera'] },
    ],
  },
  {
    id: 'video-direction',
    title: 'Video Direction',
    fields: [
      { key: 'temporalAction', label: 'Temporal action', type: 'select', categories: ['video'], options: ['single continuous action', 'starts still then moves', 'approaches camera', 'moves across frame', 'turns and reveals subject', 'object enters frame', 'object exits frame', 'repeating loopable action', 'before-and-after transformation', 'cause-and-effect action'] },
      { key: 'actionPacing', label: 'Pacing', type: 'select', categories: ['video'], options: ['very slow pace', 'slow cinematic pace', 'natural real-time pace', 'quick energetic pace', 'gradual build-up', 'pause then action', 'smooth loop pacing'] },
      { key: 'shotContinuity', label: 'Shot continuity', type: 'select', categories: ['video'], options: ['single unbroken shot', 'continuous motion with no cuts', 'match start and end for seamless loop', 'same scene throughout', 'same lighting throughout', 'no sudden scene changes', 'no jump cuts'] },
      { key: 'subjectConsistency', label: 'Subject consistency', type: 'select', categories: ['video'], options: ['keep subject identity consistent', 'keep outfit consistent', 'keep face consistent', 'keep body proportions consistent', 'keep object shape consistent', 'keep character on-model', 'no morphing or identity drift'] },
      { key: 'startFrame', label: 'Start framing', type: 'select', categories: ['video'], options: ['starts on close-up', 'starts on medium shot', 'starts on wide shot', 'starts with subject centred', 'starts with subject off-frame', 'starts with empty scene', 'starts from behind subject', 'starts on detail insert'] },
      { key: 'endFrame', label: 'End framing', type: 'select', categories: ['video'], options: ['ends on close-up', 'ends on medium shot', 'ends on wide shot', 'ends with subject centred', 'ends with subject exiting frame', 'ends on held final pose', 'ends matching first frame', 'ends on detail insert'] },
      { key: 'motionConstraints', label: 'Motion constraints', type: 'select', categories: ['video'], options: ['minimal subject movement', 'controlled natural movement', 'no camera shake', 'no fast motion', 'no warping or morphing', 'no extra limbs appearing', 'no sudden pose changes', 'physically plausible motion', 'stable background geometry'] },
      { key: 'videoDirectionDetail', label: 'Extra video direction', type: 'text', wide: true, categories: ['video'], placeholder: 'timing, beats, loop notes, motion limits, continuity requirements...' },
    ],
  },
  {
    id: 'lighting-style',
    title: 'Lighting & Style',
    fields: [
      { key: 'lighting', label: 'Lighting', type: 'select', options: ['soft natural light', 'hard sunlight', 'golden hour light', 'blue hour light', 'overcast soft daylight', 'dappled sunlight', 'sunbeam through window', 'studio softbox lighting', 'ring light', 'window light', 'neon lighting', 'colour gel lighting', 'cinematic side lighting', 'Rembrandt lighting', 'butterfly lighting', 'practical lamp light', 'candlelight', 'moonlight', 'backlit silhouette', 'rim lighting', 'volumetric light beams', 'flash photography lighting'] },
      { key: 'exposure', label: 'Exposure', type: 'select', options: ['balanced exposure', 'underexposed shadows', 'overexposed highlights', 'high-key exposure', 'low-key exposure', 'silhouette exposure', 'HDR tonal range', 'film-like contrast', 'soft shadow detail', 'deep shadow contrast', 'bright airy exposure'] },
      { key: 'colourGrade', label: 'Colour grade', type: 'select', options: ['natural colour grade', 'cinematic teal and amber', 'warm colour grade', 'cool colour grade', 'muted colours', 'high saturation', 'pastel palette', 'monochrome black and white', 'sepia tone', 'bleach bypass', 'Aerochrome false-colour infrared', 'neon cyberpunk palette', 'earth-tone palette', 'soft mint and gold palette', 'cool moonlit palette', 'warm nostalgic film palette'] },
      { key: 'style', label: 'Style', type: 'select', options: ['photorealistic', 'editorial fashion photography', 'lookbook photography', 'lifestyle photography', 'high-end e-commerce photography', 'minimalist studio photography', 'cinematic film still', 'analog film photography', 'Polaroid snapshot', 'documentary photography', 'street photography', 'luxury campaign', 'beauty campaign', 'commercial product style', 'fine art portrait', 'Y2K fashion editorial', 'dark fantasy', 'high fantasy', 'sci-fi concept art', 'anime style', 'graphic novel style', 'oil painting', 'watercolour illustration', '3D render', 'clay render'] },
      { key: 'mood', label: 'Mood', type: 'select', options: ['calm intimate mood', 'playful mood', 'confident editorial mood', 'nostalgic mood', 'mysterious mood', 'surreal mood', 'romantic mood', 'quiet luxury mood', 'high-energy mood', 'melancholic mood', 'dreamy mood'] },
      { key: 'finish', label: 'Finish', type: 'select', options: ['ultra-detailed', 'clean realistic detail', 'soft film grain', 'sharp crisp detail', 'dreamy soft focus', 'gritty texture', 'polished commercial finish', 'matte finish', 'glossy finish', 'subtle haze', 'high contrast detail', 'natural skin texture', 'fine fabric texture', 'cinematic grain'] },
      { key: 'styleDetail', label: 'Extra style detail', type: 'text', wide: true, placeholder: 'artist-free aesthetic notes, texture, mood, production design...' },
    ],
  },
];

const DEFAULT_NEGATIVE_PROMPT =
  'extra fingers, extra arms, duplicate limbs, malformed hands, distorted face, bad anatomy, blurry, low quality, watermark, text artifacts, upside down subject';

const loraId = (l) => l?.rel_path || l?.filename || '';
const modelSupportsLora = (m) => {
  if (!m) return false;
  if (m.supports_lora === true) return true;
  const category = String(m.category || '').toLowerCase();
  if (['image', 'video'].includes(category)) return true;
  if (m.supports_lora === false) return false;
  const id = String(m.id || '').toLowerCase();
  const runner = String(m.runner_module || '').toLowerCase();
  return /\b(qwen|flux|z-image|wan|hunyuan|ltx|longcat)/.test(id) ||
    /\b(qwen|flux|z_image|wan|hunyuan|ltx|longcat)/.test(runner);
};

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
  if (!c) return;
  const duplicate = [...c.querySelectorAll('.toast-message')].find(el => el.textContent === String(msg));
  if (duplicate) {
    const parent = duplicate.closest('.toast');
    parent?.classList.add('show');
    return;
  }
  [...c.querySelectorAll('.toast')].slice(0, Math.max(0, c.children.length - 2)).forEach(el => el.remove());
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
  const base = Array.isArray(m.param_groups) && m.param_groups.length
    ? m.param_groups
    : (DEFAULT_GROUPS_BY_CATEGORY[m.category] || ['server']);
  const variantGroups = selectedVariant(m)?.param_groups;
  if (!Array.isArray(variantGroups) || !variantGroups.length) return base;
  return [...base, ...variantGroups.filter(g => !base.includes(g))];
}

// Resolve fields for a model: walk groups, apply per-model overrides
function resolveFields(m) {
  const overrides = {
    ...(m.param_overrides || {}),
    ...(selectedVariant(m)?.param_overrides || {}),
  };
  const modelKeys = Array.isArray(m.param_keys) ? m.param_keys : null;
  const variantKeys = selectedVariant(m)?.param_keys;
  const enabledKeys = (modelKeys || Array.isArray(variantKeys))
    ? new Set([...(modelKeys || []), ...(Array.isArray(variantKeys) ? variantKeys : [])])
    : null; // optional whitelist
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
    let moderation = null;
    if (res.status === 524) {
      msg = 'The public proxy timed out before the server replied. Long generations now use background jobs; please retry after refreshing the app.';
    }
    try {
      const ct = res.headers.get('content-type') || '';
      if (ct.includes('application/json')) {
        const body = await res.json();
        // Structured moderation rejection: detail is a dict with `moderated: true`.
        // Attach to the Error so callers can render the right input-side hint
        // (prompt vs ref) instead of collapsing to "[object Object]".
        if (body.detail && typeof body.detail === 'object' && body.detail.moderated) {
          moderation = body.detail;
          msg = body.detail.message || 'Flagged by moderation';
        } else if (body.detail) {
          msg = String(body.detail);
        }
      }
    } catch {}
    const err = new Error(msg);
    if (moderation) err.moderation = moderation;
    throw err;
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
  trainers:      () => json(apiCall('/api/trainers')),
  trainerDatasets: () => json(apiCall('/api/trainers/datasets')),
  createTrainerDataset: (b) => json(apiCall('/api/trainers/dataset/create', jsonBody(b))),
  deleteTrainerDataset: (datasetPath) => {
    const qs = new URLSearchParams({ dataset_path: datasetPath || '' });
    return json(apiCall(`/api/trainers/dataset?${qs}`, { method: 'DELETE' }));
  },
  downloadTrainerDataset: (b) => json(apiCall('/api/trainers/datasets/download', jsonBody(b))),
  uploadTrainerDataset:   (files, targetName) => {
    const fd = new FormData();
    files.forEach(file => fd.append('files', file, file.name));
    const qs = new URLSearchParams({ target_name: targetName || 'upload' });
    return json(apiCall(`/api/trainers/dataset/upload?${qs}`, {
      method: 'POST',
      body: fd,
      timeoutMs: 120000,
    }));
  },
  uploadDatasetFiles:   (files, datasetPath) => {
    const fd = new FormData();
    files.forEach(file => fd.append('files', file, file.name));
    const qs = new URLSearchParams({ dataset_path: datasetPath || '' });
    return json(apiCall(`/api/trainers/dataset/upload?${qs}`, {
      method: 'POST',
      body: fd,
      timeoutMs: 120000,
    }));
  },
  validateTrainerDataset: (b) => json(apiCall('/api/trainers/validate', jsonBody(b))),
  listTrainerDataset:     (b) => json(apiCall('/api/trainers/dataset/list', jsonBody(b))),
  saveTrainerCaption:     (b) => json(apiCall('/api/trainers/dataset/caption', jsonBody(b))),
  toggleTrainerExclude:   (b) => json(apiCall('/api/trainers/dataset/exclude', jsonBody(b))),
  datasetVisionProxy:     (b) => json(apiCall('/api/trainers/dataset/vision-proxy', jsonBody(b))),
  datasetVisionRuntimeStatus: (b) => {
    const qs = new URLSearchParams(b || {});
    return json(apiCall(`/api/trainers/dataset/vision-runtime/status?${qs}`));
  },
  datasetVisionRuntimeStart:  (b) => json(apiCall('/api/trainers/dataset/vision-runtime/start', jsonBody(b))),
  datasetVisionRuntimeStop:   ()  => json(apiCall('/api/trainers/dataset/vision-runtime/stop', { method: 'POST' })),
  trainJobs:     () => json(apiCall('/api/train-jobs')),
  startTrainJob: (b) => json(apiCall('/api/train-jobs', jsonBody(b))),
  trainJob:      (id) => json(apiCall(`/api/train-jobs/${encodeURIComponent(id)}`)),
  cancelTrainJob:(id) => json(apiCall(`/api/train-jobs/${encodeURIComponent(id)}`, { method: 'DELETE' })),
  trainJobSamples:(id) => json(apiCall(`/api/train-jobs/${encodeURIComponent(id)}/samples`)),
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
  uploadLoraHF:  (b)    => json(apiCall('/api/loras/upload-hf', jsonBody(b))),
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
  civSearch:     (q, opts = {}) => {
    const p = new URLSearchParams({
      query: q || '',
      types: 'LORA',
      limit: String(opts.limit ?? 48),
      page:  String(opts.page  ?? 1),
      sort:  opts.sort || 'Most Downloaded',
    });
    if (opts.baseModel) p.append('base_model', opts.baseModel);
    if (opts.key)       p.append('api_key',   opts.key);
    return json(apiCall(`/api/civitai/search?${p}`));
  },
  civModel:      (id, key) => {
    const p = new URLSearchParams();
    if (key) p.append('api_key', key);
    const qs = p.toString() ? `?${p}` : '';
    return json(apiCall(`/api/civitai/model/${id}${qs}`));
  },
  civDownload:   (b) => json(apiCall('/api/civitai/download', jsonBody(b))),
  modStatus:     ()      => json(apiCall('/api/moderation/status')),
  modOverride:   (token) => json(apiCall('/api/moderation/override', jsonBody({ token }))),
  modClear:      ()      => json(apiCall('/api/moderation/override', { method: 'DELETE' })),
  repoStatus:    ()      => json(apiCall('/api/app/repo-status')),
  repoRefresh:   ()      => json(apiCall('/api/app/refresh-repo', { method: 'POST', timeoutMs: 240000 })),
  runtimeDepsStatus: ()      => json(apiCall('/api/app/runtime-deps/status')),
  runtimeDepsInstall: ()     => json(apiCall('/api/app/runtime-deps/install', { method: 'POST', timeoutMs: 10000 })),
  runtimeDepsJob: (jobId)    => json(apiCall(`/api/app/runtime-deps/install-status/${encodeURIComponent(jobId)}`)),
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
    applySidebarPreference();
    appEl.classList.add('ready');
  }
  await ensureServiceWorker();   // register SW so <img> requests get decrypted
  await primeBrowserKey();       // hand the IDB-cached CryptoKey to the SW
  await Promise.all([loadModels(), loadLoras(), loadAssets(), loadComponents(), loadTrainers()]);
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

async function ensureAssetDecryptorReady() {
  if (state.preview) return true;
  await ensureServiceWorker();
  if (!state.dataKey) {
    try { state.dataKey = await loadKey(); } catch {}
  }
  if (state.dataKey) {
    await postKeyToServiceWorker(state.dataKey);
    return true;
  }
  openAuthGate('unlock', 'Unlock encrypted assets to view this file.');
  return false;
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
  $('#sidebarToggle')?.addEventListener('click', toggleSidebar);
  window.addEventListener('resize', () => requestAnimationFrame(updateSidebarGlow));

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
  $('#wsViewportLightbox')?.addEventListener('click', async () => {
    const asset = activePreviewAsset(state.workspace);
    if (!asset) return;
    await openWorkspacePreviewAsset(state.workspace, asset);
  });
  $('#wsViewportDownload')?.addEventListener('click', () => {
    const asset = activePreviewAsset(state.workspace);
    if (asset) downloadAsset(asset);
  });
  $('#wsEnhancePrompt')?.addEventListener('click', enhanceWorkspacePrompt);
  $('#wsPromptModeBuilder')?.addEventListener('click', () => setPromptMode('builder'));
  $('#wsPromptModeRaw')?.addEventListener('click', () => setPromptMode('raw'));
  bindWorkspaceLoraAddButtons();
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

  $('#civSearchBtn').addEventListener('click', () => civSearch());
  $('#civSearch').addEventListener('keydown', (e) => { if (e.key === 'Enter') civSearch(); });
  $('#civSort')?.addEventListener('change', () => civSearch());
  $('#civBaseModel')?.addEventListener('change', () => civSearch());
  $('#civUrlGo').addEventListener('click', civOpenByUrl);
  $('#civUrl').addEventListener('keydown', (e) => { if (e.key === 'Enter') civOpenByUrl(); });

  $('#hfDownloadBtn')?.addEventListener('click', startHFDownload);
  $('#hfBrowseBtn')?.addEventListener('click', () => openHFBrowser());
  $('#hfRepo')?.addEventListener('keydown', (e) => { if (e.key === 'Enter') openHFBrowser(); });
  bindTrainerWizard();
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
  ['#settingsHFToken', '#wsHFToken', '#hfTokenDl', '#trainerWizardHFToken'].forEach(sel => {
    $(sel)?.addEventListener('change', e => saveHFToken(e.target.value));
  });
  bindDatasetStudio();

  $$('[data-save]').forEach(b => b.addEventListener('click', () => {
    const key = b.dataset.save;
    const val = $('#' + b.dataset.source).value;
    state.settings[key] = val;
    localStorage.setItem('forge_settings', JSON.stringify(state.settings));
    if (key === 'hf_token') syncHFTokenInputs();
    toast('Saved', 'success');
  }));

  $('#moderationDisableBtn')?.addEventListener('click', disableModeration);
  $('#moderationEnableBtn')?.addEventListener('click', enableModeration);
  $('#repoRefreshBtn')?.addEventListener('click', refreshRepoFromSettings);
  $('#runtimeDepsInstallBtn')?.addEventListener('click', installRuntimeDepsFromSettings);

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

function applySidebarPreference() {
  setSidebarCollapsed(!!state.settings.sidebar_collapsed, { persist: false });
}

function toggleSidebar() {
  const app = $('.app');
  const next = !app?.classList.contains('sidebar-collapsed');
  setSidebarCollapsed(next);
}

function setSidebarCollapsed(collapsed, opts = {}) {
  const app = $('.app');
  const btn = $('#sidebarToggle');
  if (!app) return;
  app.classList.toggle('sidebar-collapsed', collapsed);
  if (btn) {
    btn.classList.toggle('active', collapsed);
    btn.setAttribute('aria-pressed', collapsed ? 'true' : 'false');
    btn.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
  }
  if (opts.persist !== false) {
    state.settings.sidebar_collapsed = collapsed;
    localStorage.setItem('forge_settings', JSON.stringify(state.settings));
  }
  requestAnimationFrame(updateSidebarGlow);
}

function updateSidebarGlow() {
  const glow = $('#sidebarGlow');
  const sidebar = $('.sidebar-scroll');
  const active = $('.sidebar .nav-item.active');
  const sidebarShell = $('.sidebar');
  if (!glow || !sidebar || !active || !sidebarShell || getComputedStyle(sidebarShell).display === 'none') return;
  const sidebarRect = sidebar.getBoundingClientRect();
  const activeRect = active.getBoundingClientRect();
  glow.style.opacity = '1';
  glow.style.height = `${Math.max(24, Math.round(activeRect.height))}px`;
  glow.style.transform = `translateY(${Math.round(activeRect.top - sidebarRect.top)}px)`;
}

function applySettingsToInputs() {
  if (state.settings.hf_token)    $('#settingsHFToken').value = state.settings.hf_token;
  if (state.settings.civitai_key) $('#settingsCivKey').value  = state.settings.civitai_key;
  syncHFTokenInputs();
}

// ── Moderation card ──────────────────────────────────────────────────────
async function renderModerationCard() {
  const pill = $('#moderationPill');
  const sub  = $('#moderationSource');
  const off  = $('#moderationDisableBtn');
  const on   = $('#moderationEnableBtn');
  if (!pill || !off || !on) return;
  try {
    const s = await api.modStatus();
    pill.dataset.state = s.enabled ? 'on' : 'off';
    pill.textContent   = s.enabled ? 'Moderation: ON' : 'Moderation: OFF';
    // 'source' explains where the current state came from. The fallback
    // branch is when an env var or marker held a wrong-token value — we
    // surface that explicitly so the operator knows their value didn't
    // take effect.
    const sourceLabel = {
      'default':          '',
      'env':              'Disabled via fork-operator env var',
      'runtime_override': 'Disabled via Settings override on this pod',
      'default_fallback': 'A disable token was provided but did not match — moderation remains on',
    }[s.source] || '';
    sub.textContent = sourceLabel;
    // Hide the runtime "re-enable" button when the env var is what's
    // disabling things — clearing the marker won't override env config.
    if (s.enabled) {
      off.style.display = '';
      on.style.display  = 'none';
    } else if (s.source === 'runtime_override') {
      off.style.display = 'none';
      on.style.display  = '';
    } else {
      off.style.display = 'none';
      on.style.display  = 'none';
    }
  } catch (e) {
    pill.dataset.state = 'unknown';
    pill.textContent   = 'Moderation: status unavailable';
    sub.textContent    = e.message || '';
  }
}

async function disableModeration() {
  // Friction-by-design: the operator must paste the acknowledgement
  // token from the source. Plain prompt() is fine here — the awkwardness
  // of pasting a long literal IS the deliberate-act surface. A fancy
  // dialog would lower the friction we're trying to preserve.
  const token = window.prompt(
    'Disabling moderation requires the acknowledgement token from backend/moderator.py (DISABLE_TOKEN).\n\n' +
    'Paste the token to confirm. The token reads as a written acknowledgement of liability for outputs from this pod.'
  );
  if (token == null) return;        // cancel
  try {
    const s = await api.modOverride(token.trim());
    toast(s.enabled ? 'Token did not take effect — moderation still on' : 'Moderation disabled', s.enabled ? 'error' : 'success');
    renderModerationCard();
  } catch (e) {
    toast(`Disable failed: ${e.message || 'unknown'}`, 'error');
  }
}

async function enableModeration() {
  if (!window.confirm('Re-enable moderation now?')) return;
  try {
    await api.modClear();
    toast('Moderation re-enabled', 'success');
    renderModerationCard();
  } catch (e) {
    toast(`Re-enable failed: ${e.message || 'unknown'}`, 'error');
  }
}

function renderRepoUpdateResult(res) {
  const pill = $('#repoStatusPill');
  const line = $('#repoRefreshStatus');
  const log = $('#repoRefreshLog');
  if (!pill || !line || !log) return;

  const branch = res.branch && res.branch !== 'HEAD' ? res.branch : 'repo';
  const commit = res.after || res.commit || res.before || '';
  const shortCommit = commit ? ` @ ${commit}` : '';
  const status = res.status || 'unknown';
  const state = status === 'updated' || status === 'up_to_date' || status === 'clean' ? 'on'
    : status === 'dirty' || status === 'error' ? 'off'
      : 'unknown';

  pill.dataset.state = state;
  pill.textContent = `Repo: ${branch}${shortCommit}`;
  line.textContent = res.message || '';

  const chunks = [];
  if (res.before || res.after) chunks.push(`commit: ${res.before || '?'} -> ${res.after || '?'}`);
  if (res.dirty_lines?.length) chunks.push(`local changes:\n${res.dirty_lines.join('\n')}`);
  if (res.stdout) chunks.push(res.stdout.trim());
  if (res.stderr) chunks.push(res.stderr.trim());
  log.textContent = chunks.filter(Boolean).join('\n\n') || (res.message || 'No repo output yet.');
  log.style.display = chunks.length ? '' : 'none';
}

async function renderRepoUpdateCard() {
  const pill = $('#repoStatusPill');
  const line = $('#repoRefreshStatus');
  if (!pill || !line) return;
  pill.dataset.state = 'loading';
  pill.textContent = 'Repo: checking…';
  line.textContent = '';
  try {
    renderRepoUpdateResult(await api.repoStatus());
  } catch (e) {
    pill.dataset.state = 'unknown';
    pill.textContent = 'Repo: status unavailable';
    line.textContent = e.message || '';
  }
}

async function refreshRepoFromSettings() {
  const btn = $('#repoRefreshBtn');
  const line = $('#repoRefreshStatus');
  if (btn) btn.disabled = true;
  if (line) line.textContent = 'Pulling latest code…';
  try {
    const res = await api.repoRefresh();
    renderRepoUpdateResult(res);
    if (res.status === 'updated') {
      toast('Pulled latest code. Refresh the browser to pick up UI changes.', 'success');
    } else if (res.status === 'up_to_date') {
      toast('Repo already up to date', 'info');
    } else if (res.status === 'dirty') {
      toast('Repo has local changes; pull skipped', 'error');
    } else {
      toast(res.message || 'Repo refresh did not complete', 'error');
    }
  } catch (e) {
    if (line) line.textContent = e.message || 'Repo refresh failed';
    toast(`Repo refresh failed: ${e.message || 'unknown'}`, 'error');
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderRuntimeDepsResult(payload = {}) {
  const pill = $('#runtimeDepsPill');
  const line = $('#runtimeDepsStatus');
  const log = $('#runtimeDepsLog');
  const btn = $('#runtimeDepsInstallBtn');
  if (!pill || !line || !log) return;

  const job = payload.latest_job || payload;
  const status = job?.status || (payload.available ? 'idle' : 'missing');
  const running = status === 'queued' || status === 'running';
  const done = status === 'done';
  const error = status === 'error' || status === 'missing';
  const torchaoReady = payload.torchao_available ?? job?.torchao_available;
  const vllmReady = payload.vllm_available ?? job?.vllm_available;
  const depsReady = payload.deps_ready ?? job?.deps_ready ?? (torchaoReady && vllmReady);

  pill.dataset.state = done || depsReady ? 'on' : error ? 'off' : running ? 'loading' : 'unknown';
  if (running) {
    pill.textContent = status === 'queued' ? 'Deps: queued' : 'Deps: installing';
  } else if (done) {
    pill.textContent = 'Deps: installed';
  } else if (error) {
    pill.textContent = 'Deps: attention needed';
  } else {
    pill.textContent = depsReady ? 'Deps: ready' : 'Deps: available';
  }

  if (!payload.available && !job?.requirements_path) {
    line.textContent = 'requirements-runtime.txt was not found on this pod.';
  } else if (running) {
    line.textContent = job.last_line || 'Installing packages...';
  } else if (job?.error) {
    line.textContent = job.error;
  } else if (done) {
    line.textContent = 'Install complete. Stop and relaunch affected models before testing.';
  } else if (depsReady) {
    line.textContent = 'TorchAO and vLLM are importable in this Python environment.';
  } else if (torchaoReady && !vllmReady) {
    line.textContent = 'TorchAO is ready. vLLM is still missing for the Dataset Studio pod captioner.';
  } else if (!torchaoReady && vllmReady) {
    line.textContent = 'vLLM is ready. TorchAO is still missing for Qwen INT8.';
  } else {
    line.textContent = 'Ready to install TorchAO, vLLM, and other packages added by the latest app update.';
  }

  const chunks = [];
  if (job?.command) chunks.push(`command: ${job.command}`);
  if (job?.python_path || payload.python_path) chunks.push(`python: ${job?.python_path || payload.python_path}`);
  if (job?.requirements_path || payload.requirements_path) chunks.push(`requirements: ${job?.requirements_path || payload.requirements_path}`);
  if (job?.log?.length) chunks.push(job.log.slice(-120).join('\n'));
  log.textContent = chunks.join('\n\n') || 'No install output yet.';
  log.style.display = chunks.length ? '' : 'none';
  if (btn) btn.disabled = running || payload.available === false;
}

async function pollRuntimeDepsInstall(jobId) {
  if (!jobId) return;
  if (state.runtimeDepsPoll) clearTimeout(state.runtimeDepsPoll);
  try {
    const job = await api.runtimeDepsJob(jobId);
    renderRuntimeDepsResult(job);
    if (job.status === 'queued' || job.status === 'running') {
      state.runtimeDepsPoll = setTimeout(() => pollRuntimeDepsInstall(jobId), 1500);
    } else {
      state.runtimeDepsPoll = null;
      toast(job.status === 'done' ? 'Runtime deps installed' : (job.error || 'Runtime deps install failed'), job.status === 'done' ? 'success' : 'error');
    }
  } catch (e) {
    state.runtimeDepsPoll = null;
    const line = $('#runtimeDepsStatus');
    if (line) line.textContent = e.message || 'Runtime deps status failed';
  }
}

async function renderRuntimeDepsCard() {
  const pill = $('#runtimeDepsPill');
  const line = $('#runtimeDepsStatus');
  if (!pill || !line) return;
  pill.dataset.state = 'loading';
  pill.textContent = 'Deps: checking...';
  line.textContent = '';
  try {
    const res = await api.runtimeDepsStatus();
    renderRuntimeDepsResult(res);
    const job = res.latest_job;
    if (job?.id && (job.status === 'queued' || job.status === 'running')) {
      pollRuntimeDepsInstall(job.id);
    }
  } catch (e) {
    pill.dataset.state = 'unknown';
    pill.textContent = 'Deps: status unavailable';
    line.textContent = e.message || '';
  }
}

async function installRuntimeDepsFromSettings() {
  const btn = $('#runtimeDepsInstallBtn');
  const line = $('#runtimeDepsStatus');
  if (btn) btn.disabled = true;
  if (line) line.textContent = 'Starting install...';
  try {
    const res = await api.runtimeDepsInstall();
    renderRuntimeDepsResult(res);
    pollRuntimeDepsInstall(res.job_id);
  } catch (e) {
    if (line) line.textContent = e.message || 'Runtime deps install failed';
    toast(`Runtime deps install failed: ${e.message || 'unknown'}`, 'error');
    if (btn) btn.disabled = false;
  }
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
  ['#settingsHFToken', '#drawerHFToken', '#wsHFToken', '#hfTokenDl', '#trainerWizardHFToken'].forEach(sel => {
    const el = $(sel);
    if (el && el.value !== token) el.value = token;
  });
}

// ── Views ────────────────────────────────────────────────────────────────
function switchView(name, push = true) {
  // Stop the trainer-running polling whenever we leave that view.
  // openTrainerRunningView starts its own poll, so this only fires on
  // navigation AWAY from the running monitor.
  if (name !== 'trainer-running' && typeof stopTrainerRunningPoll === 'function') {
    stopTrainerRunningPoll();
  }
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
  const navName = name === 'trainer-running' ? 'trainers' : name;
  $$('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === navName));
  requestAnimationFrame(updateSidebarGlow);
  state.view = name;
  if (name === 'loras')   loadLoras();
  if (name === 'assets')  loadAssets();
  if (name === 'running') renderRunning();
  if (name === 'datasets') loadDatasetLibrary();
  if (name === 'trainers') {
    loadTrainers();
    pokeTrainerPoll();
  }
  if (name === 'settings') {
    renderModerationCard();
    renderRepoUpdateCard();
    renderRuntimeDepsCard();
  }
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
  state.gpuStatus = 'loading';
  renderGPU(state.gpu, 'loading');
  if (state.preview) {
    await loadPreviewRegistry();
    for (const m of state.models) {
      state.modelStates[m.id] = { downloaded: true, ready: true };
    }
    renderModels();
    return;
  }
  try {
    const data = await api.models();
    state.models    = data.models;
    state.upscalers = data.upscalers || [];
    state.gpu       = data.gpu;
    state.gpuStatus = 'ready';
    renderGPU(data.gpu);
    renderModels();
    $('#modelCount').textContent = state.models.length;
    $('#gpuDebug').textContent   = JSON.stringify(data.gpu, null, 2);
  } catch (e) {
    state.models    = mockModels();
    state.upscalers = [];
    renderModels();
    $('#modelCount').textContent = state.models.length;
    state.gpuStatus = 'error';
    renderGPU(state.gpu, 'error');
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
    state.gpuStatus = 'ready';
    renderGPU(state.gpu);
    $('#modelCount').textContent = state.models.length;
    $('#gpuDebug').textContent   = JSON.stringify(state.gpu, null, 2);
    renderModels();
  } catch (err) {
    state.models    = mockModels();
    state.upscalers = [];
    state.gpu       = { type: 'unknown', name: 'Preview mode', vram_gb: 0 };
    state.gpuStatus = 'disconnected';
    $('#modelCount').textContent = state.models.length;
    renderGPU(state.gpu, 'disconnected');
    renderModels();
    console.warn('Preview registry unavailable, using mockModels()', err);
  }
}

let _gpuLoadHistory = Array(20).fill(15);

function updateGPUSparkline(load) {
  _gpuLoadHistory.push(load);
  if (_gpuLoadHistory.length > 20) _gpuLoadHistory.shift();

  const path = $('#gpuPill .sparkline-path');
  if (!path) return;

  const width = 60;
  const height = 20;
  const points = _gpuLoadHistory.map((val, idx) => {
    const x = (idx / (_gpuLoadHistory.length - 1)) * width;
    const y = height - 2 - (val / 100) * (height - 4);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  path.setAttribute('d', `M ${points.join(' L ')}`);
}

function gpuVramGb(gpu) {
  const candidates = [gpu?.vram_gb, gpu?.total_gb, gpu?.total_vram_gb, gpu?.memory_total_gb];
  const found = candidates.find(value => Number.isFinite(Number(value)) && Number(value) > 0);
  return found == null ? 0 : Number(found);
}

function gpuDisplayState(gpu, requested = state.gpuStatus) {
  if (requested === 'error') return 'error';
  if (requested === 'loading' || !gpu) return 'loading';
  if (requested === 'disconnected' || gpu.type === 'unknown') return 'disconnected';
  return 'ready';
}

function gpuDisplayLabel(gpu, displayState = gpuDisplayState(gpu)) {
  if (displayState === 'loading') return 'GPU · checking...';
  if (displayState === 'error') return 'GPU · unavailable';
  if (displayState === 'disconnected') return gpu?.name ? `GPU · ${gpu.name}` : 'GPU · disconnected';
  const name = gpu?.name || gpu?.type || 'GPU';
  const vram = gpuVramGb(gpu);
  return `${name}${vram ? ` · ${Math.round(vram)} GB` : ''}`;
}

function renderGPU(gpu, requestedState = state.gpuStatus) {
  const pill = $('#gpuPill');
  if (!pill) return;
  const displayState = gpuDisplayState(gpu, requestedState);
  state.gpuStatus = displayState;
  pill.dataset.state = displayState;
  pill.classList.toggle('online', displayState === 'ready');
  pill.classList.toggle('loading', displayState === 'loading');
  pill.classList.toggle('offline', displayState === 'disconnected');
  pill.classList.toggle('error', displayState === 'error');
  const label = gpuDisplayLabel(gpu, displayState);
  pill.innerHTML = `
    <div class="dot"></div>
    <span>${label}</span>
    <svg class="gpu-sparkline" viewBox="0 0 60 20">
      <path class="sparkline-path" d="" fill="none" stroke="var(--cat-image)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path>
    </svg>
  `;
  updateGPUSparkline(_gpuLoadHistory[_gpuLoadHistory.length - 1]);
  renderTrainerGpuChip();
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
  if (m.category === 'audio' && max === 0) {
    return { min: 0, max: 0, label: 'CPU' };
  }
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
    const vram = gpuVramGb(gpu);
    const vramProfile = modelVramProfile(m);
    const phase = drawerPhase(m.id);
    let btnText = 'Configure';
    if (phase === 'ready') btnText = 'Open';
    else if (phase === 'downloaded') btnText = 'Start';
    const cpuProfile = vramProfile.label === 'CPU';
    const pct = cpuProfile ? 100 : gpu && vramProfile.max ? Math.min(100, (vram / Math.max(vramProfile.max, 1)) * 100) : 50;
    const vramClass = cpuProfile || !gpu ? '' : vram >= vramProfile.max ? '' : vram >= vramProfile.min ? 'warn' : 'bad';
    const vramTag = cpuProfile ? '<span class="tag solid">CPU</span>' : !gpu ? '' : vram >= vramProfile.max
      ? `<span class="tag solid">✓ ${vramProfile.max} GB</span>`
      : vram >= vramProfile.min
        ? `<span class="tag warn">${vramProfile.min} GB min</span>`
        : `<span class="tag bad">need ${vramProfile.min} GB</span>`;
    const gpuOk = !gpu || gpu.type === 'cpu' || (m.gpu_support || []).includes(gpu.type);
    const dim = gpu && !gpuOk;
    const access = downloadAccessFor(m);
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
          ${modelSupportsLora(m) ? '<span class="tag">LoRA</span>' : ''}
          ${access.gated ? '<span class="tag warn">HF terms</span>' : ''}
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

function downloadAccessFor(m) {
  const variant = selectedVariant(m);
  const out = {
    gated: false,
    requiresHFToken: false,
    repos: [],
    note: '',
  };
  const seen = new Set();

  const addRepo = (entry) => {
    if (!entry) return;
    const repo = typeof entry === 'string' ? entry : entry.repo;
    if (!repo || seen.has(repo)) return;
    seen.add(repo);
    out.repos.push({
      repo,
      label: typeof entry === 'object' && entry.label ? entry.label : repo,
      url: typeof entry === 'object' && entry.url ? entry.url : `https://huggingface.co/${repo}`,
    });
  };

  const merge = (access) => {
    if (!access) return;
    out.gated = !!(out.gated || access.gated);
    out.requiresHFToken = !!(out.requiresHFToken || access.requires_hf_token || access.requiresHFToken);
    if (access.note) out.note = access.note;
    (access.repos || access.gated_repos || []).forEach(addRepo);
  };

  merge(m.access);
  merge(variant?.access);
  if (m.gated || m.requires_hf_token) {
    out.gated = m.gated !== false;
    out.requiresHFToken = m.requires_hf_token !== false;
    addRepo(m.hf_repo);
  }
  return out;
}

function accessAckKey(m, access, variant) {
  const repos = access.repos.map(r => r.repo).join('|') || m.hf_repo || m.id;
  return `forge_hf_access_ack:${m.id}:${variant || 'base'}:${repos}`;
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
  const rt = state.runtimes[id];
  if (s.error)             return 'error';
  if (s.ready)             return 'ready';
  if (rt?.state === 'installing') return 'starting';
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
      error: ws.error || null,
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
  const access = downloadAccessFor(m);
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
    not_downloaded: { cls:'',       icon:'i-down',    brand:false, title:'Not downloaded',    sub:'Pull model weights from HuggingFace.' },
    downloading:    { cls:'busy',   icon:'i-spinner', brand:true,  title:'Downloading weights', sub:'Large repos can take a while.' },
    downloaded:     { cls:'ready',  icon:'i-check',   brand:false, title:'Weights cached',    sub:'Ready to start the runner.' },
    starting:       { cls:'busy',   icon:'i-spinner', brand:true,  title:startingTitle,       sub:startingSub },
    ready:          { cls:'ready',  icon:'i-check',   brand:false, title:'Runner ready',      sub:'Open the workspace to generate.' },
    error:          { cls:'error',  icon:'i-x',       brand:false, title:'Something went wrong', sub: modelStateOf(m.id).error || '' },
  }[phase];

  $('#drawerBody').innerHTML = `
    <p style="font-size:13px;color:var(--t-2);line-height:1.55;margin-bottom:12px">
      ${m.description || ''}
    </p>

    <div class="drawer-stats">
      <div class="drawer-stat"><div class="label">VRAM</div><div class="value">${sizeStr}</div></div>
      <div class="drawer-stat"><div class="label">LoRA</div><div class="value">${modelSupportsLora(m) ? 'yes' : 'no'}</div></div>
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

    ${access.gated ? `
      <div class="drawer-access">
        <div class="access-icon"><svg><use href="#i-lock"/></svg></div>
        <div class="access-copy">
          <div class="access-title">Hugging Face access required</div>
          <div class="access-text">
            Accept the repo terms on Hugging Face, then paste a read token before downloading.
            ${access.note ? `<span>${esc(access.note)}</span>` : ''}
          </div>
          <div class="access-links">
            ${access.repos.map(r => `<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.label || r.repo)} ↗</a>`).join('')}
          </div>
        </div>
      </div>
    ` : ''}

    <div class="drawer-state ${stateBlocks.cls}">
      <div class="state-icon ${stateBlocks.brand ? 'brand-loading' : ''}">
        ${stateBlocks.brand ? '<span class="brand-loader" aria-hidden="true"></span>' : `<svg><use href="#${stateBlocks.icon}"/></svg>`}
      </div>
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
      <input class="text-input mask-text" id="drawerHFToken" type="text" placeholder="hf_…" value="${esc(state.settings.hf_token || '')}" autocomplete="off" autocapitalize="none" autocorrect="off" spellcheck="false" data-1p-ignore="true" data-lpignore="true" data-bwignore="true" data-form-type="other">
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
  const access = downloadAccessFor(m);
  const rt = m.runtime ? state.runtimes[m.id] : null;
  const startingHtml = rt?.state === 'installing'
    ? '<span class="spinner"></span><span>Preparing runtime…</span>'
    : '<span class="spinner"></span><span>Starting…</span>';
  const labels = {
    not_downloaded: {
      html: access.gated
        ? '<svg><use href="#i-lock"/></svg>Download with HF token'
        : '<svg><use href="#i-down"/></svg>Download weights',
      busy: false,
      action: 'download',
    },
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
    case 'hf_file':
      return `<div class="field">
        <label class="field-label">${f.label}${f.hint ? `<span class="hint">${f.hint}</span>` : ''}</label>
        <div style="display:flex;gap:8px">
          <input class="text-input" data-key="${f.key}" placeholder="${esc(f.placeholder || '')}" value="${esc(v)}">
          <button class="btn" type="button" data-pick-hf-weight data-target="${esc(f.key)}">Browse</button>
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
               ${f.placeholder ? `placeholder="${esc(f.placeholder)}"` : ''}
               value="${esc(v)}" data-key="${f.key}">
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
  rootEl.querySelectorAll('[data-pick-hf-weight]').forEach(b => {
    b.addEventListener('click', () => pickHFWeightInto(b.dataset.target, rootEl));
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

async function pickHFWeightInto(targetKey, rootEl) {
  const repoInput = rootEl.querySelector('[data-key="ltx_weight_repo"]');
  const rawRepo = repoInput?.value || state.params.ltx_weight_repo || '';
  const { repo, revision } = parseHFRepoInput(rawRepo);
  if (!repo) {
    toast('Enter a Hugging Face repo first', 'error');
    return;
  }
  $('#modalTitle').textContent = 'Select LTX weight';
  $('#modalMeta').textContent  = repo;
  $('#modalBody').innerHTML = `<div class="empty" style="padding:18px"><span class="spinner"></span> Loading files…</div>`;
  $('#modalFoot').innerHTML = `<button class="btn ghost" id="modalCancel">Cancel</button>`;
  $('#modalCancel').onclick = closeModal;
  openModal();
  try {
    const token = currentHFToken('#drawerHFToken') || currentHFToken('#hfTokenDl');
    const data = await api.hfFiles(repo, revision, token || null);
    const files = (data.files || []).filter(f => String(f.rel_path || '').toLowerCase().endsWith('.safetensors'));
    if (!files.length) {
      $('#modalBody').innerHTML = `<div class="empty" style="padding:18px">No .safetensors files found in <span class="mono">${esc(repo)}</span>.</div>`;
      return;
    }
    $('#modalBody').innerHTML = `<div class="hf-file-list" style="max-height:52vh;overflow:auto">
      ${files.map(f => {
        const sizeStr = f.size ? formatBytes(f.size) : '?';
        const lfs = f.lfs ? '<span class="lfs-pill">LFS</span>' : '';
        return `<button class="hf-file-row" type="button" data-hf-weight="${esc(f.rel_path)}">
          <span></span>
          <span class="hf-file-name" title="${esc(f.rel_path)}">${esc(f.rel_path)}</span>
          <span class="hf-file-size">${sizeStr}</span>
          ${lfs}
        </button>`;
      }).join('')}
    </div>`;
    $$('[data-hf-weight]', $('#modalBody')).forEach(btn => btn.addEventListener('click', () => {
      const filename = btn.dataset.hfWeight;
      state.params.ltx_weight_repo = repo;
      state.params[targetKey] = filename;
      if (repoInput) repoInput.value = repo;
      const input = rootEl.querySelector(`[data-key="${CSS.escape(targetKey)}"]`);
      if (input) input.value = filename;
      closeModal();
    }));
  } catch (e) {
    $('#modalBody').innerHTML = `<div class="empty hf-error" style="padding:18px"><svg><use href="#i-x"/></svg> ${esc(e.message || 'Failed to fetch files')}</div>`;
  }
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
      job_id:      s.job_id,
      last_line:   s.last_line || s.note || '',
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
  const access = downloadAccessFor(m);

  if (access.gated && !hfToken) {
    const names = access.repos.map(r => r.repo).join(', ') || m.hf_repo || m.name;
    toast(`HF access required for ${names}. Accept the terms on Hugging Face, then paste a read token.`, 'error');
    $('#drawerHFToken')?.focus();
    return;
  }

  if (access.gated) {
    const key = accessAckKey(m, access, variant);
    if (localStorage.getItem(key) !== '1') {
      const names = access.repos.map(r => r.repo).join(', ') || m.hf_repo || m.name;
      const ok = await confirmModal(
        'HF Access Required',
        `Before downloading ${m.name}, open the Hugging Face link in this card and accept the access terms for ${names}. Your token must have read access to those repos.`,
        'I accepted terms'
      );
      if (!ok) {
        $('#drawerHFToken')?.focus();
        return;
      }
      localStorage.setItem(key, '1');
    }
  }

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
    toast(e.message || 'Download failed', 'error');
  }
  renderDrawerBody();
}

async function waitForDownload(modelId, maxSeconds, variant) {
  const deadline = Date.now() + maxSeconds * 1000;
  while (Date.now() < deadline) {
    let ws = null;
    try {
      ws = await api.weightStatus(modelId, variant);
    } catch {
      // Tolerate transient status-poll failures. Stored backend download
      // errors are handled below so gated/private repos don't look stuck.
    }
    if (ws?.error)       throw new Error(ws.error);
    if (ws?.downloaded)  return true;
    if (typeof ws?.progress === 'number') {
      setModelState(modelId, { progress: ws.progress });
      const fill = $('#drawerProgress');
      if (fill) fill.style.width = `${Math.min(100, ws.progress * 100)}%`;
    }
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
  clampParamsToFields(state.params, sections);
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

function clampParamsToFields(params, sections) {
  for (const s of sections) for (const f of s.fields) {
    if (!['slider', 'number'].includes(f.type)) continue;
    if (params[f.key] === undefined || params[f.key] === '') continue;
    const current = Number(params[f.key]);
    if (!Number.isFinite(current)) continue;
    const min = f.min !== undefined ? Number(f.min) : -Infinity;
    const max = f.max !== undefined ? Number(f.max) : Infinity;
    const step = f.step !== undefined ? Number(f.step) : 1;
    let next = clampNumber(current, min, max);
    if (Number.isFinite(step) && step > 0) {
      next = snapNumber(next, {
        min,
        max,
        step,
        fallback: f.default !== undefined ? Number(f.default) : next,
      });
    }
    params[f.key] = Number(formatControlNumber(next, step));
  }
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
  renderHeroPreview(m.id);
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

  const strip = $('#wsImgStrip');
  strip.classList.toggle('has-media', hasMedia);
  strip.innerHTML = `${generatedCellHtml(latest)}
    <details class="img-ref-panel"${hasMedia ? '' : ' open'}>
      <summary>
        <span>References</span>
        <em>${cells.length} optional input${cells.length === 1 ? '' : 's'}</em>
        <svg aria-hidden="true"><use href="#i-down"/></svg>
      </summary>
      <div class="img-ref-grid">${cells.join('')}</div>
    </details>`;
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
        ? (asset.kind === 'audio'
            ? `<div class="asset-audio generated-audio"><svg aria-hidden="true"><use href="#i-audio"/></svg><div class="asset-audio-label">Audio</div><audio src="${u}" controls preload="metadata"></audio></div>`
            : asset.kind === 'video'
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
  $$('[data-zoom]', $('#wsImgStrip')).forEach(c => c.addEventListener('click', async () => {
    const recents = state.recents[m.id] || [];
    if (recents.length) await openAssetLightbox(recents[recents.length - 1], { list: recents, index: recents.length - 1 });
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
function withStandardResolutionPresets(presets = []) {
  const landscape1080 = { label: '16:9 1080p', w: 1920, h: 1080 };
  const portrait1080 = { label: '9:16 1080p', w: 1080, h: 1920 };
  const seen = new Set(presets.map(p => `${p.w}x${p.h}`));
  const out = [...presets];

  const insertMissing = (preset, matcher) => {
    const key = `${preset.w}x${preset.h}`;
    if (seen.has(key)) return;
    let insertAt = -1;
    out.forEach((p, i) => {
      if (matcher(p)) insertAt = i;
    });
    if (insertAt === -1) {
      out.push(preset);
    } else {
      out.splice(insertAt + 1, 0, preset);
    }
    seen.add(key);
  };

  insertMissing(landscape1080, p => p.w > p.h && Math.abs((p.w / p.h) - (16 / 9)) < 0.08);
  insertMissing(portrait1080, p => p.h > p.w && Math.abs((p.w / p.h) - (9 / 16)) < 0.08);
  return out;
}

function renderAspectChips(m, hasDims) {
  const row = $('#wsAspectRow');
  if (!hasDims) { row.innerHTML = ''; return; }
  const basePresets = m.aspect_presets || [
    { label: 'Square',          w: 1024, h: 1024 },
    { label: 'Landscape 16:9',  w: 1664, h: 928  },
    { label: 'Portrait 9:16',   w: 928,  h: 1664 },
    { label: '3:4',             w: 1152, h: 1536 },
    { label: '4:3',             w: 1536, h: 1152 },
  ];
  const presets = withStandardResolutionPresets(basePresets);
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
  const pane = ensureWorkspaceLoraPane();
  const supported = modelSupportsLora(m);
  if (!pane) return;
  pane.dataset.model = m?.id || '';
  pane.dataset.supported = supported ? 'true' : 'false';
  if (!supported) {
    pane.style.display = 'none';
    return;
  }
  pane.hidden = false;
  pane.style.setProperty('display', 'block', 'important');

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
      : `<div class="lora-empty-note">No LoRAs assigned yet. Use Add to attach a downloaded LoRA to this model.</div>`);

  $('#wsLoraCount').textContent = `${defaults.length + assigned.length}`;
  bindWorkspaceLoraAddButtons(m);

  bindLoraCards(m);
}

function ensureWorkspaceLoraPane() {
  let pane = $('#wsLoraPane');
  if (!pane) {
    pane = document.createElement('div');
    pane.className = 'ws-section lora-pane';
    pane.id = 'wsLoraPane';
    pane.innerHTML = `
      <div class="ws-section-title lora-title">
        <span>LoRAs <span class="meta" id="wsLoraCount">0</span></span>
        <button class="lora-title-add" id="wsLoraAddTop" type="button">+ Add</button>
      </div>
      <div class="lora-group" id="wsLoraDefaults" style="display:none">
        <div class="lora-group-head">Bundled with model</div>
      </div>
      <div class="lora-group" id="wsLoraAssigned">
        <div class="lora-group-head">Assigned to this model</div>
      </div>
      <div class="lora-add" id="wsLoraAdd">+ Add LoRA from library</div>`;
  }

  const authSection = $('#wsHFToken')?.closest('.ws-section');
  if (authSection && pane.nextElementSibling !== authSection) {
    authSection.parentNode.insertBefore(pane, authSection);
  }
  return pane;
}

function bindWorkspaceLoraAddButtons(model = state.selected) {
  ['#wsLoraAdd', '#wsLoraAddTop'].forEach(sel => {
    const btn = $(sel);
    if (!btn) return;
    btn.onclick = () => {
      const m = model || state.selected;
      if (m) openLoraPicker(m);
    };
  });
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

function assetKey(asset) {
  return asset ? (asset.path || asset.url || asset.name || '') : '';
}

function sameAsset(a, b) {
  if (!a || !b) return false;
  if (a.path && b.path) return a.path === b.path;
  return !!a.name && a.name === b.name && (!a.kind || !b.kind || a.kind === b.kind);
}

function mergeFreshAsset(asset, fresh) {
  if (!asset || !fresh) return asset || fresh || null;
  Object.assign(asset, fresh);
  for (const list of Object.values(state.recents)) {
    if (!Array.isArray(list)) continue;
    list.forEach(item => {
      if (sameAsset(item, fresh)) Object.assign(item, fresh);
    });
  }
  for (const [modelId, active] of Object.entries(state.activePreviews)) {
    if (sameAsset(active, fresh)) state.activePreviews[modelId] = { ...active, ...fresh };
  }
  return asset;
}

async function refreshAssetReference(asset) {
  if (state.preview || !asset?.path) return asset;
  if (!await ensureAssetDecryptorReady()) return asset;
  try {
    const data = await api.assets();
    const freshAssets = data.assets || [];
    state.assets = freshAssets;
    const countEl = $('#assetNavCount');
    if (countEl) countEl.textContent = state.assets.length;
    const fresh = freshAssets.find(a => sameAsset(a, asset));
    if (fresh) return mergeFreshAsset(asset, fresh);
  } catch {}
  return asset;
}

function activePreviewAsset(modelId) {
  if (!modelId) return null;
  const items = state.recents[modelId] || [];
  const active = state.activePreviews[modelId];
  const activeKey = assetKey(active);
  if (activeKey) {
    const fresh = items.find(a => assetKey(a) === activeKey);
    if (fresh) return fresh;
    if (active?.url) return active;
  }
  return items[items.length - 1] || null;
}

function setActivePreview(modelId, asset, opts = {}) {
  if (!modelId || !asset) return;
  state.activePreviews[modelId] = asset;
  renderHeroPreview(modelId, opts);
  renderRecents(modelId);
}

async function openWorkspacePreviewAsset(modelId, asset) {
  if (!modelId || !asset) return;
  const items = state.recents[modelId] || [asset];
  const idx = Math.max(0, items.findIndex(item => sameAsset(item, asset)));
  await openAssetLightbox(asset, { list: items.length ? items : [asset], index: idx });
}

function viewportIdleHtml() {
  return `<div class="viewport-idle">
    <div class="idle-core" aria-hidden="true">
      <span></span><span></span><span></span>
    </div>
    <div class="idle-copy">
      <strong>Inference field online</strong>
      <span>Compose a prompt and generate when ready.</span>
    </div>
  </div>`;
}

function heroAssetHtml(asset) {
  const u = esc(asset?.url || '');
  const n = esc(asset?.name || '');
  if (asset?.kind === 'video') {
    return `<video class="viewport-media" src="${u}" controls playsinline preload="metadata"></video>`;
  }
  if (asset?.kind === 'audio') {
    return `<div class="viewport-audio">
      <svg aria-hidden="true"><use href="#i-audio"/></svg>
      <div class="asset-audio-label">${esc(asset.name || 'Audio generation')}</div>
      <audio src="${u}" controls preload="metadata"></audio>
    </div>`;
  }
  return `<img class="viewport-media" src="${u}" alt="${n}">`;
}

function bindHeroMediaRecovery(modelId, asset) {
  const media = $('#wsViewportStage')?.querySelector('.viewport-media');
  if (!media || !asset?.path) return;
  media.addEventListener('error', async () => {
    if (media.dataset.retried === '1') return;
    media.dataset.retried = '1';
    const before = asset.url;
    await refreshAssetReference(asset);
    if (asset.url && asset.url !== before) {
      renderHeroPreview(modelId, { asset });
    }
  }, { once: true });
}

function bindHeroPreviewClick(modelId, asset) {
  const wrap = $('#wsViewportStage')?.querySelector('.viewport-asset-wrap');
  if (!wrap || !asset) return;
  wrap.addEventListener('click', async (e) => {
    if (e.target.closest('audio, button, a')) return;
    await openWorkspacePreviewAsset(modelId, asset);
  });
}

function heroMetaText(asset) {
  if (!asset) return 'Dimensions: - · Seed: -';
  const bits = [];
  if (asset.width && asset.height) bits.push(`${asset.width}x${asset.height}`);
  if (asset.seed != null) bits.push(`seed ${asset.seed}`);
  bits.push(asset.name || asset.path || 'generation');
  return bits.join(' · ');
}

function generationStatusText(job) {
  if (!job) return 'Preparing render field';
  const bits = queueCardBits(job);
  if (bits.hasProgress) return `${bits.pct}%${job.etaMs != null && bits.pct < 100 ? ` · ${formatETA(job.etaMs)}` : ''}`;
  return job.status === 'pending' ? 'Queued for generation' : 'Estimating render path';
}

function generationHeroHtml(job) {
  const pct = Math.max(0, Math.min(100, Math.round((job?.progress || 0) * 100)));
  const prompt = esc((job?.params?.prompt || currentWorkspacePrompt() || 'Rendering generation').replace(/\s+/g, ' ').slice(0, 120));
  return `<div class="viewport-generating">
    <div class="generating-aura" aria-hidden="true">
      <span></span><span></span><span></span>
    </div>
    <div class="generating-copy">
      <strong>Rendering latent field</strong>
      <span class="generating-prompt">${prompt}</span>
    </div>
    <div class="generating-progress" aria-hidden="true"><span style="width:${pct}%"></span></div>
    <div class="generating-status">${esc(generationStatusText(job))}</div>
  </div>`;
}

function updateGenerationHero(stage, job) {
  const root = stage?.querySelector('.viewport-generating');
  if (!root) {
    stage.innerHTML = generationHeroHtml(job);
    return;
  }
  const pct = Math.max(0, Math.min(100, Math.round((job?.progress || 0) * 100)));
  const prompt = (job?.params?.prompt || currentWorkspacePrompt() || 'Rendering generation').replace(/\s+/g, ' ').slice(0, 120);
  const promptEl = root.querySelector('.generating-prompt');
  const bar = root.querySelector('.generating-progress span');
  const status = root.querySelector('.generating-status');
  if (promptEl) promptEl.textContent = prompt;
  if (bar) bar.style.width = `${pct}%`;
  if (status) status.textContent = generationStatusText(job);
}

function renderHeroPreview(modelId, opts = {}) {
  const canvas = $('#wsViewportCanvas');
  const stage = $('#wsViewportStage');
  const hud = $('#wsViewportHud');
  const meta = $('#wsViewportMeta');
  if (!canvas || !stage || !modelId) return;
  const runningJob = opts.job || state.queue.find(j => j.model_id === modelId && j.status === 'running');
  const asset = opts.asset || activePreviewAsset(modelId);

  canvas.classList.toggle('is-generating', !!runningJob);
  canvas.classList.toggle('is-ready', !runningJob && !!asset);
  canvas.classList.toggle('is-idle', !runningJob && !asset);

  if (runningJob) {
    updateGenerationHero(stage, runningJob);
    if (hud) hud.style.setProperty('display', 'none', 'important');
    return;
  }

  if (asset) {
    stage.innerHTML = `<div class="viewport-asset-wrap">${heroAssetHtml(asset)}</div>`;
    bindHeroMediaRecovery(modelId, asset);
    bindHeroPreviewClick(modelId, asset);
    if (meta) meta.textContent = heroMetaText(asset);
    if (hud) hud.style.removeProperty('display');
    return;
  }

  stage.innerHTML = viewportIdleHtml();
  if (hud) hud.style.setProperty('display', 'none', 'important');
}

function updateHeroProgress(job) {
  if (!job || state.workspace !== job.model_id) return;
  renderHeroPreview(job.model_id, { job });
}

function renderRecents(modelId) {
  const items   = state.recents[modelId] || [];
  // Pending + running jobs for *this* model become loading cards prepended
  // to the recents grid. They share the same footprint and 3-dot menu shape
  // as completed assets, but with a Cancel-only menu instead of Download/Delete.
  const queued = state.queue.filter(j => j.model_id === modelId && j.status !== 'done');
  const grid = $('#wsRecent');

  if (!items.length && !queued.length) {
    grid.innerHTML = `<div class="recent-empty"><span>No generations yet</span><small>Finished renders will pin here as selectable previews.</small></div>`;
    return;
  }

  const queuedHtml = queueCardsHtml(queued);
  const activeKey = assetKey(activePreviewAsset(modelId));

  const recentHtml = items.slice().reverse().map((_, i) => {
    const idx = items.length - 1 - i;
    const a = items[idx];
    const active = activeKey && assetKey(a) === activeKey ? ' active' : '';
    return `<div class="asset${active}" data-recent-idx="${idx}">
      ${assetMediaHtml(a, { lazy: true })}
      <button class="asset-menu-btn" data-asset-menu="${idx}" data-asset-source="recent" title="Actions">
        <svg><use href="#i-dots"/></svg>
      </button>
      <div class="asset-meta">${esc(a.name)}</div>
    </div>`;
  }).join('');

  grid.innerHTML = queuedHtml + recentHtml;

  // Recent tile click → focus the hero preview; 3-dot → asset menu.
  $$('.asset[data-recent-idx]', grid).forEach(el => el.addEventListener('click', async (e) => {
    if (e.target.closest('[data-asset-menu]')) return;
    if (e.target.closest('audio')) return;
    const idx = Number(el.dataset.recentIdx);
    const asset = await refreshAssetReference(items[idx]);
    setActivePreview(modelId, asset);
    await openWorkspacePreviewAsset(modelId, asset);
  }));
  $$('[data-asset-menu]', grid).forEach(b => b.addEventListener('click', (e) => {
    e.stopPropagation();
    const idx = Number(b.dataset.assetMenu);
    showAssetMenu(items[idx], b);
  }));
  bindQueueCardMenus(grid);
}

function assetMediaHtml(asset, opts = {}) {
  const u = esc(asset?.url || '');
  const n = esc(asset?.name || '');
  if (asset?.kind === 'video') {
    return `<video src="${u}" muted playsinline preload="metadata"></video>`;
  }
  if (asset?.kind === 'audio') {
    return `<div class="asset-audio">
      <svg aria-hidden="true"><use href="#i-audio"/></svg>
      <div class="asset-audio-label">Audio</div>
      <audio src="${u}" controls preload="metadata"></audio>
    </div>`;
  }
  return `<img src="${u}" ${opts.lazy ? 'loading="lazy"' : ''} alt="${n}">`;
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
    if (!state.preview && asset.path && !await ensureAssetDecryptorReady()) return;
    toast(`Preparing ${asset.name || 'download'}…`, 'info');
    await refreshAssetReference(asset);
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
    renderHeroPreview(state.workspace);
    const m = state.models.find(x => x.id === state.workspace);
    if (m?.image_inputs?.length) renderImgStrip(m);
  }
  toast('Deleted', 'info');
}

function pruneFromRecents(asset) {
  if (!asset?.path) return;
  const removedKey = assetKey(asset);
  for (const id of Object.keys(state.recents)) {
    state.recents[id] = state.recents[id].filter(a => a.path !== asset.path);
    if (assetKey(state.activePreviews[id]) === removedKey) delete state.activePreviews[id];
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
  if (!['image', 'video'].includes(m.category)) return 'raw';
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
  const supportsBuilder = ['image', 'video'].includes(m?.category);
  const builder = $('#wsPromptBuilder');
  const raw = $('#wsPromptRaw');
  if (builder) builder.style.display = supportsBuilder && mode === 'builder' ? '' : 'none';
  if (raw) raw.style.display = mode === 'raw' ? '' : 'none';
  $('.prompt-mode-toggle')?.style.setProperty('display', supportsBuilder ? '' : 'none');
  $('#wsEnhancePrompt')?.style.setProperty('display', supportsBuilder ? '' : 'none');
  $('#wsPromptModeBuilder')?.classList.toggle('active', mode === 'builder');
  $('#wsPromptModeRaw')?.classList.toggle('active', mode === 'raw');
}

function renderPromptBuilder(m) {
  const root = $('#wsPromptBuilder');
  if (!root || !m || !['image', 'video'].includes(m.category)) {
    if (root) root.innerHTML = '';
    return;
  }
  const values = promptBuilderForModel(m);
  const sections = PROMPT_BUILDER_SECTIONS.map(section => {
    const fields = section.fields.filter(f => !f.categories || f.categories.includes(m.category));
    if (!fields.length) return '';
    return `
    <section class="pb-section">
      <div class="pb-section-title">${esc(section.title)}</div>
      <div class="pb-section-grid">
        ${fields.map(f => renderPromptBuilderControl(f, values)).join('')}
      </div>
    </section>
  `;
  }).join('');
  root.innerHTML = `
    <div class="pb-stack">${sections}</div>
    <div class="pb-preview">
      <div class="pb-preview-label">Compiled prompt</div>
      <div class="pb-preview-text" id="wsPromptBuilderPreview">${esc(compilePromptBuilder(m) || 'Fill in a few fields to build the generation prompt.')}</div>
    </div>
  `;
  $$('[data-pb-key]', root).forEach(input => input.addEventListener(input.tagName === 'SELECT' ? 'change' : 'input', () => {
    values[input.dataset.pbKey] = input.value;
    savePromptBuilderState();
    updatePromptBuilderPreview(m);
  }));
}

function renderPromptBuilderControl(f, values) {
  const value = String(values[f.key] || '');
  const cls = `pb-field ${f.wide ? 'wide' : ''}`;
  if (f.type === 'textarea') {
    return `<label class="${cls}">
      <span>${esc(f.label)}</span>
      <textarea class="textarea" data-pb-key="${esc(f.key)}" placeholder="${esc(f.placeholder || '')}">${esc(value)}</textarea>
    </label>`;
  }
  if (f.type === 'select') {
    const options = Array.isArray(f.options) ? f.options : [];
    const hasStoredCustom = value && !options.includes(value);
    return `<label class="${cls}">
      <span>${esc(f.label)}</span>
      <select class="select" data-pb-key="${esc(f.key)}">
        <option value="">Any / unset</option>
        ${hasStoredCustom ? `<option value="${esc(value)}" selected>${esc(value)}</option>` : ''}
        ${options.map(opt => `<option value="${esc(opt)}" ${opt === value ? 'selected' : ''}>${esc(opt)}</option>`).join('')}
      </select>
    </label>`;
  }
  return `<label class="${cls}">
    <span>${esc(f.label)}</span>
    <input class="text-input" data-pb-key="${esc(f.key)}" value="${esc(value)}" placeholder="${esc(f.placeholder || '')}">
  </label>`;
}

function updatePromptBuilderPreview(m) {
  const el = $('#wsPromptBuilderPreview');
  if (!el) return;
  el.textContent = compilePromptBuilder(m) || 'Fill in a few fields to build the generation prompt.';
}

function compilePromptBuilder(m = state.selected) {
  const values = promptBuilderForModel(m);
  const clean = (key) => String(values[key] || '').replace(/\s+/g, ' ').trim();
  const add = (parts, label, key) => {
    const value = clean(key);
    if (value) parts.push(`${label}: ${value}`);
  };
  const addCombined = (parts, label, keys) => {
    const value = keys.map(clean).filter(Boolean).join(' ');
    if (value) parts.push(`${label}: ${value}`);
  };
  const sectionText = (label, parts) => parts.length ? `${label}: ${parts.join('; ')}` : '';

  const subject = [];
  add(subject, 'main subject', 'idea');
  add(subject, 'subject count', 'subjectCount');
  add(subject, 'appearance', 'presentation');
  add(subject, 'age', 'age');
  add(subject, 'body type', 'bodyType');
  add(subject, 'silhouette', 'silhouette');
  add(subject, 'posture', 'posture');
  add(subject, 'role or archetype', 'subjectRole');
  add(subject, 'extra detail', 'subjectDetail');

  const faceHair = [];
  add(faceHair, 'expression', 'expression');
  add(faceHair, 'head direction', 'head');
  add(faceHair, 'hair colour', 'hairColor');
  add(faceHair, 'hair style', 'hairStyle');
  add(faceHair, 'make-up', 'makeup');
  add(faceHair, 'piercings', 'piercings');
  add(faceHair, 'tattoos or marks', 'tattoos');
  add(faceHair, 'extra face detail', 'faceDetail');

  const wardrobe = [];
  addCombined(wardrobe, 'outfit', ['fullColor', 'fullMaterial', 'outfit']);
  add(wardrobe, 'fit and silhouette', 'outfitFit');
  add(wardrobe, 'pattern', 'pattern');
  addCombined(wardrobe, 'top', ['topColor', 'topMaterial', 'topGarment']);
  addCombined(wardrobe, 'bottom', ['bottomColor', 'bottomMaterial', 'bottomGarment']);
  addCombined(wardrobe, 'shoes', ['shoeColor', 'shoes']);
  add(wardrobe, 'accessories', 'accessories');
  add(wardrobe, 'extra clothing detail', 'clothingDetail');

  const scene = [];
  add(scene, 'setting', 'scene');
  add(scene, 'time of day', 'timeOfDay');
  add(scene, 'weather', 'weather');
  add(scene, 'background', 'background');
  add(scene, 'atmosphere', 'sceneMood');
  add(scene, 'props', 'props');
  add(scene, 'extra scene detail', 'sceneDetail');

  const poseAction = [];
  add(poseAction, 'pose', 'pose');
  add(poseAction, 'body facing', 'bodyAxis');
  add(poseAction, 'hands', 'handPose');
  add(poseAction, 'action', 'action');
  add(poseAction, 'motion feel', 'motion');
  add(poseAction, 'extra action detail', 'actionDetail');

  const videoDirection = [];
  if (m?.category === 'video') {
    add(videoDirection, 'temporal action', 'temporalAction');
    add(videoDirection, 'pacing', 'actionPacing');
    add(videoDirection, 'shot continuity', 'shotContinuity');
    add(videoDirection, 'subject consistency', 'subjectConsistency');
    add(videoDirection, 'start framing', 'startFrame');
    add(videoDirection, 'end framing', 'endFrame');
    add(videoDirection, 'motion constraints', 'motionConstraints');
    add(videoDirection, 'extra video direction', 'videoDirectionDetail');
  }

  const camera = [];
  add(camera, 'shot size', 'distance');
  add(camera, 'viewpoint', 'camera');
  add(camera, 'lens', 'lens');
  add(camera, 'composition', 'framing');
  add(camera, 'depth of field', 'depthOfField');
  if (m?.category === 'video') add(camera, 'camera move', 'cameraMove');

  const lightingStyle = [];
  add(lightingStyle, 'lighting', 'lighting');
  add(lightingStyle, 'exposure', 'exposure');
  add(lightingStyle, 'colour grade', 'colourGrade');
  add(lightingStyle, 'visual style', 'style');
  add(lightingStyle, 'mood', 'mood');
  add(lightingStyle, 'image finish', 'finish');
  add(lightingStyle, 'extra style detail', 'styleDetail');

  return [
    sectionText('Subject', subject),
    sectionText('Face and hair', faceHair),
    sectionText('Wardrobe', wardrobe),
    sectionText('Scene', scene),
    sectionText('Pose and action', poseAction),
    sectionText('Video direction', videoDirection),
    sectionText('Camera', camera),
    sectionText('Lighting and style', lightingStyle),
  ].filter(Boolean).join('. ');
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
  updateHeroProgress(job);
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
    renderHeroPreview(job.model_id, { job });
    await runJob(job);
    setModelState(job.model_id, { generating: false });
    job.status = 'done';
    renderQueue();
    renderHeroPreview(job.model_id);
  }
  // Drain finished + cancelled items so the next batch starts clean.
  state.queue = state.queue.filter(j => j.status === 'pending' || j.status === 'running');
  state.queueRunning = false;
  $('#wsCancel').disabled = true;
  updateGenerateBtn();
  updateWsStatus();
  renderQueue();
  if (state.workspace) renderHeroPreview(state.workspace);
  if (!state.preview) refreshAssetsSoon();
}

async function runJob(job) {
  // Poll /api/runner/{id}/preview and update the Generated cell while inference runs.
  let previewInterval = null;
  let progressStream = null;
  let previewProgressInterval = null;
  const m = state.models.find(x => x.id === job.model_id);
  job.startedAt = Date.now();
  job.progress = 0;
  job.progressStep = 0;
  job.progressTotal = 0;
  job.etaMs = null;
  progressStream = startJobProgress(job);
  const canPollPreview = !state.preview && m?.category === 'image' && m.supports_preview !== false;
  if (canPollPreview) {
    previewInterval = setInterval(async () => {
      if (state.workspace !== job.model_id) return;
      try {
        const r = await fetch(`/api/runner/${job.model_id}/preview?t=${Date.now()}`, { credentials: 'same-origin' });
        if (r.status === 204 || r.status === 404) return;
        if (!r.ok) return;
        const blob = await r.blob();
        if (!blob.size) return;
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
    if (newAssets.length) {
      state.activePreviews[job.model_id] = newAssets[newAssets.length - 1];
    }
    if (state.preview) {
      state.assets = [...newAssets, ...state.assets];
      renderAssets();
      const countEl = $('#assetNavCount');
      if (countEl) countEl.textContent = state.assets.length;
    }
    if (state.workspace === job.model_id) {
      renderRecents(job.model_id);
      renderHeroPreview(job.model_id);
      refreshSeedField();
      // Refresh the Generated cell on edit-style models.
      const m = state.models.find(x => x.id === job.model_id);
      if (m?.image_inputs?.length) renderImgStrip(m);
    }
  } catch (e) {
    job.error = e.message;
    if (e.moderation) {
      // Friendlier copy than a generic "Job failed" — points the user at
      // the input that needs editing (prompt vs ref image).
      const stage = e.moderation.stage === 'ref_image' ? 'Reference image' : 'Prompt';
      toast(`${stage} flagged: ${e.moderation.label || 'NSFW'}. Edit and try again.`, 'error');
    } else {
      toast(`Job failed: ${e.message || 'unknown'}`, 'error');
    }
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

async function openAssetLightbox(asset, opts = {}) {
  if (!asset) return;
  if (!state.preview && asset.path && !await ensureAssetDecryptorReady()) return;
  const fresh = await refreshAssetReference(asset);
  openLightbox(fresh || asset, opts);
}

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

function lightboxNavHtml() {
  return (_lightboxList && _lightboxList.length > 1) ? `
    <button class="lightbox-nav prev" id="lightboxPrev" aria-label="Previous">‹</button>
    <button class="lightbox-nav next" id="lightboxNext" aria-label="Next">›</button>
  ` : '';
}

function bindLightboxNav() {
  $('#lightboxPrev')?.addEventListener('click', (e) => { e.stopPropagation(); _lightboxStep(-1); });
  $('#lightboxNext')?.addEventListener('click', (e) => { e.stopPropagation(); _lightboxStep(1);  });
}

function _renderLightbox() {
  const asset = _lightboxList ? _lightboxList[_lightboxIdx] : _lightboxCurrentAsset;
  if (!asset) return;
  _lightboxCurrentAsset = asset;
  $('#lightboxMeta').textContent = _lightboxList
    ? `${_lightboxIdx + 1} / ${_lightboxList.length} · ${asset.name || asset.path || ''}`
    : (asset.name || asset.path || '');
  const stage = $('#lightboxStage');
  stage.classList.remove('portrait-media', 'landscape-media');
  stage.style.removeProperty('--asset-ratio');
  const u = esc(asset.url);
  const navArrows = lightboxNavHtml();
  stage.innerHTML = (asset.kind === 'video'
    ? `<video src="${u}" controls autoplay loop playsinline preload="metadata"></video>`
    : asset.kind === 'audio'
      ? `<div class="lightbox-audio">
           <svg aria-hidden="true"><use href="#i-audio"/></svg>
           <div class="lightbox-audio-title">${esc(asset.name || 'Audio')}</div>
           <audio src="${u}" controls autoplay preload="metadata"></audio>
         </div>`
      : `<img src="${u}" alt="${esc(asset.name || '')}">`) + navArrows;
  const foot = $('#lightboxFoot');
  foot.textContent = asset.path || '';
  foot.title = asset.path || '';
  const media = stage.querySelector('img, video');
  if (media) {
    const applyRatio = () => {
      const w = media.videoWidth || media.naturalWidth;
      const h = media.videoHeight || media.naturalHeight;
      if (!w || !h) return;
      stage.style.setProperty('--asset-ratio', `${w} / ${h}`);
      stage.classList.toggle('portrait-media', h > w * 1.15);
      stage.classList.toggle('landscape-media', w > h * 1.35);
      foot.textContent = asset.path ? `${asset.path} · ${w}×${h}` : `${w}×${h}`;
      foot.title = foot.textContent;
    };
    applyRatio();
    media.addEventListener(media.tagName === 'VIDEO' ? 'loadedmetadata' : 'load', applyRatio, { once: true });
    media.addEventListener('error', async () => {
      if (media.dataset.retried !== '1') {
        media.dataset.retried = '1';
        const before = asset.url;
        await refreshAssetReference(asset);
        if (asset.url && asset.url !== before) {
          _renderLightbox();
          return;
        }
      }
      showLightboxLoadError(asset);
    }, { once: true });
  }
  $('#lightboxDownload').onclick = () => {
    downloadAsset(asset);
  };
  bindLightboxNav();
}

function showLightboxLoadError(asset) {
  const stage = $('#lightboxStage');
  stage.classList.remove('portrait-media', 'landscape-media');
  stage.innerHTML = `
    <div class="lightbox-error">
      <svg aria-hidden="true"><use href="#i-image"/></svg>
      <strong>Could not load this asset</strong>
      <span>The signed view link may have expired, or the encrypted asset key needs unlocking.</span>
      <button class="btn" id="lightboxRetry" type="button">Retry</button>
    </div>
    ${lightboxNavHtml()}`;
  $('#lightboxRetry')?.addEventListener('click', async (e) => {
    e.stopPropagation();
    await refreshAssetReference(asset);
    _renderLightbox();
  });
  bindLightboxNav();
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
  if (state.preview) {
    updateGPUSparkline(12 + Math.random() * 4);
    setTimeout(pollRunning, 3000);
    return;
  }
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
          if (s.starting && state.runtimes[id]?.state === 'installing') continue;
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

// ── Dataset Studio ───────────────────────────────────────────────────────
const DATASET_REVIEW_PROMPT = `You are curating images for a single-character FLUX LoRA training dataset.
Return JSON only in this exact shape: {"keep":true,"reason":"short reason"}.
Keep only if the image has exactly one clear human subject, a readable unoccluded face, a single frame, and the subject is large enough for identity details.
Reject object-only images, multiple people, composites, watermarks/text overlays, extreme long shots, severe blur, or heavy face occlusion.`;

const DATASET_CAPTION_FALLBACK = `You are a computer vision labeling engine for FLUX LoRA character training. Analyze the image and generate one concise descriptive prose paragraph containing only visible elements.

Describe subject framing, camera angle, pose, action, expression, clothing, layers, environment, background, and lighting.

Do not use the subject's name, trigger tags, gender nouns, guessed age, ethnicity, identity, or introductory phrases. Output only the raw descriptive caption.`;

let datasetPipelineCancel = false;

function bindDatasetStudio() {
  $('#btnCreateDataset')?.addEventListener('click', openCreateDatasetModal);
  $('#datasetSearchInput')?.addEventListener('input', renderDatasetLibrary);
  $('#btnBackToDatasets')?.addEventListener('click', showDatasetList);
  $('#btnToggleCaptionDrawer')?.addEventListener('click', () => toggleDatasetDrawer(true));
  $('#btnCloseCaptionDrawer')?.addEventListener('click', () => toggleDatasetDrawer(false));
  $('#btnCheckCaptionRuntime')?.addEventListener('click', () => checkDatasetVisionRuntime({ quiet: false }));
  $('#btnStartCaptionRuntime')?.addEventListener('click', startDatasetVisionRuntime);
  $('#btnStopCaptionRuntime')?.addEventListener('click', stopDatasetVisionRuntime);
  $('#btnRunCaptioning')?.addEventListener('click', runDatasetPipeline);
  $('#btnCancelCaptioning')?.addEventListener('click', () => {
    datasetPipelineCancel = true;
    datasetLog('Stop requested. Finishing the current image...', 'warn');
  });
  $('#btnSendToTrainer')?.addEventListener('click', sendDatasetToTrainer);
  $('#btnDeleteDataset')?.addEventListener('click', () => {
    if (state.activeDataset?.dataset_path) deleteDatasetByPath(state.activeDataset.dataset_path);
  });
  $('#btnSaveCaption')?.addEventListener('click', saveDatasetInspectorCaption);
  $('#btnToggleExclude')?.addEventListener('click', toggleDatasetInspectorExclude);
  $('#inspectorCaptionText')?.addEventListener('input', (e) => {
    const len = $('#captionLength');
    if (len) len.textContent = `${e.target.value.length} chars`;
  });

  $$('[data-dataset-filter]').forEach(btn => btn.addEventListener('click', () => {
    state.datasetFilter = btn.dataset.datasetFilter || 'all';
    $$('[data-dataset-filter]').forEach(x => x.classList.toggle('active', x === btn));
    renderDatasetGrid();
  }));

  const drop = $('#datasetDropZone');
  const input = $('#datasetFileInput');
  drop?.addEventListener('click', () => input?.click());
  input?.addEventListener('change', async (e) => {
    await uploadFilesToActiveDataset([...e.target.files]);
    e.target.value = '';
  });
  drop?.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('drag');
  });
  drop?.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop?.addEventListener('drop', async (e) => {
    e.preventDefault();
    drop.classList.remove('drag');
    await uploadFilesToActiveDataset([...e.dataTransfer.files]);
  });
}

function datasetNameFromPath(path) {
  const parts = String(path || '').split('/').filter(Boolean);
  return parts.at(-1) || 'dataset';
}

function datasetStatCounts() {
  const all = state.datasetItems || [];
  const included = all.filter(x => !x.excluded);
  const captioned = included.filter(x => (x.caption || '').trim());
  return {
    all: all.length,
    included: included.length,
    excluded: all.length - included.length,
    captioned: captioned.length,
    missing: included.length - captioned.length,
    ready: included.length >= 8 && included.length === captioned.length,
  };
}

async function loadDatasetLibrary() {
  const grid = $('#datasetFolderGrid');
  if (!grid) return;
  grid.innerHTML = `<div class="empty"><span class="spinner"></span><span>Scanning datasets...</span></div>`;
  try {
    if (state.preview) {
      state.datasets = Object.values(state.previewDatasetStore);
    } else {
      const res = await api.trainerDatasets();
      state.datasets = res.datasets || [];
    }
    renderDatasetLibrary();
  } catch (e) {
    grid.innerHTML = `<div class="empty"><span>Dataset scan failed: ${esc(e.message || 'unknown error')}</span></div>`;
  }
}

function renderDatasetLibrary() {
  const grid = $('#datasetFolderGrid');
  if (!grid) return;
  const q = ($('#datasetSearchInput')?.value || '').trim().toLowerCase();
  const rows = (state.datasets || []).filter(ds => {
    const hay = `${ds.name || ''} ${ds.dataset_path || ''}`.toLowerCase();
    return !q || hay.includes(q);
  });
  const navCount = $('#datasetNavCount');
  if (navCount) navCount.textContent = String((state.datasets || []).length);
  if (!rows.length) {
    grid.innerHTML = `
      <div class="empty dataset-library-empty">
        <strong>No datasets yet.</strong>
        <span>Create a dataset, upload images, then run auto-review and pod captions.</span>
      </div>`;
    return;
  }
  grid.innerHTML = rows.map(ds => {
    const count = Number(ds.image_count || ds.pairs || 0);
    const caps = Number(ds.caption_count || 0);
    const pct = count ? Math.round((caps / count) * 100) : 0;
    return `
      <div class="dataset-card dataset-folder-card" data-dataset-open="${esc(ds.dataset_path)}" role="button" tabindex="0">
        <div class="dataset-card-head">
          <div>
            <strong>${esc(ds.name || datasetNameFromPath(ds.dataset_path))}</strong>
            <span>${esc(ds.dataset_path || '')}</span>
          </div>
          <div class="dataset-card-actions">
            <em>${count} img</em>
            <button class="dataset-delete-btn" data-dataset-delete="${esc(ds.dataset_path)}" type="button" title="Delete dataset" aria-label="Delete ${esc(ds.name || datasetNameFromPath(ds.dataset_path))}">Delete</button>
          </div>
        </div>
        <div class="dataset-card-progress"><span style="width:${Math.min(100, pct)}%"></span></div>
        <div class="dataset-card-foot">
          <span>${caps} captions</span>
          <span>${ds.valid === false ? esc(ds.error || 'needs attention') : (pct === 100 && count ? 'ready' : 'needs prep')}</span>
        </div>
      </div>`;
  }).join('');
  $$('[data-dataset-open]', grid).forEach(card => {
    card.addEventListener('click', () => openDatasetByPath(card.dataset.datasetOpen));
    card.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      e.preventDefault();
      openDatasetByPath(card.dataset.datasetOpen);
    });
  });
  $$('[data-dataset-delete]', grid).forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      deleteDatasetByPath(btn.dataset.datasetDelete);
    });
  });
}

function openCreateDatasetModal() {
  $('#modalTitle').textContent = 'Create Dataset';
  $('#modalMeta').textContent = 'A clean folder will be created under /workspace/datasets.';
  $('#modalBody').innerHTML = `
    <div class="field">
      <label>Dataset name</label>
      <input class="input" id="newDatasetName" placeholder="melissa_flux" autocomplete="off" spellcheck="false">
    </div>
    <div class="hint">Use a simple folder name. You can add images immediately after creation.</div>`;
  $('#modalFoot').innerHTML = `
    <button class="btn ghost" id="modalCancel">Cancel</button>
    <button class="btn primary" id="modalCreateDataset">Create</button>`;
  $('#modalCancel').onclick = closeModal;
  $('#modalCreateDataset').onclick = async () => {
    const name = ($('#newDatasetName')?.value || '').trim();
    if (!name) return toast('Name the dataset first.', 'error');
    $('#modalCreateDataset').disabled = true;
    try {
      let ds;
      if (state.preview) {
        const safe = name.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'dataset';
        ds = { name: safe, dataset_path: `datasets/${safe}`, image_count: 0, caption_count: 0, pairs: 0, valid: true, items: [] };
        state.previewDatasetStore[ds.dataset_path] = ds;
      } else {
        try {
          ds = await api.createTrainerDataset({ name });
        } catch (e) {
          if (!/method not allowed|server error 405/i.test(e.message || '')) throw e;
          const safe = name.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'dataset';
          ds = {
            name: safe,
            dataset_path: `datasets/${safe}`,
            image_count: 0,
            caption_count: 0,
            pairs: 0,
            valid: true,
            virtual: true,
          };
          state.datasets = [ds, ...(state.datasets || []).filter(x => x.dataset_path !== ds.dataset_path)];
          toast('Backend create route is not live yet; dataset folder will be created on first upload.', 'info');
        }
      }
      closeModal();
      if (!ds.virtual) await loadDatasetLibrary();
      await openDatasetByPath(ds.dataset_path);
      toast('Dataset created', 'success');
    } catch (e) {
      toast(`Create failed: ${e.message || 'unknown error'}`, 'error');
      $('#modalCreateDataset').disabled = false;
    }
  };
  openModal();
  setTimeout(() => $('#newDatasetName')?.focus(), 50);
}

function showDatasetList() {
  $('#datasetWorkspaceScreen').style.display = 'none';
  $('#datasetListScreen').style.display = '';
  state.activeDataset = null;
  state.datasetItems = [];
  state.datasetSelected = -1;
  loadDatasetLibrary();
}

async function openDatasetByPath(datasetPath) {
  const ds = (state.datasets || []).find(x => x.dataset_path === datasetPath)
    || { name: datasetNameFromPath(datasetPath), dataset_path: datasetPath };
  state.activeDataset = ds;
  state.datasetSelected = -1;
  $('#datasetListScreen').style.display = 'none';
  const workspace = $('#datasetWorkspaceScreen');
  workspace.style.display = 'flex';
  $('#activeDatasetTitle').textContent = ds.name || datasetNameFromPath(ds.dataset_path);
  $('#activeDatasetSub').textContent = ds.dataset_path || '';
  resetDatasetConsole();
  await loadActiveDatasetItems();
  toggleDatasetDrawer(!state.datasetItems.length);
}

async function deleteDatasetByPath(datasetPath) {
  if (!datasetPath) return;
  const name = datasetNameFromPath(datasetPath);
  const ok = await confirmModal(
    'Delete Dataset',
    `Delete <b>${esc(name)}</b>? This removes the dataset folder, images, captions, and curation metadata from <span style="font-family:var(--font-mono)">${esc(datasetPath)}</span>.`,
    'Delete dataset',
    true
  );
  if (!ok) return;
  try {
    if (state.preview) {
      delete state.previewDatasetStore[datasetPath];
      state.datasets = (state.datasets || []).filter(ds => ds.dataset_path !== datasetPath);
    } else {
      await api.deleteTrainerDataset(datasetPath);
      state.datasets = (state.datasets || []).filter(ds => ds.dataset_path !== datasetPath);
    }
    if (state.activeDataset?.dataset_path === datasetPath) showDatasetList();
    else renderDatasetLibrary();
    toast(`Deleted ${name}`, 'success');
  } catch (e) {
    toast(`Delete failed: ${e.message || 'unknown error'}`, 'error');
  }
}

async function loadActiveDatasetItems() {
  if (!state.activeDataset?.dataset_path) return;
  if (state.activeDataset.virtual) {
    state.datasetItems = [];
    renderDatasetWorkspace();
    return;
  }
  try {
    if (state.preview) {
      state.datasetItems = state.previewDatasetStore[state.activeDataset.dataset_path]?.items || [];
    } else {
      const res = await api.listTrainerDataset({ dataset_path: state.activeDataset.dataset_path });
      state.datasetItems = (res.pairs || []).map((p, i) => ({ ...p, _idx: i }));
    }
    renderDatasetWorkspace();
  } catch (e) {
    state.datasetItems = [];
    renderDatasetWorkspace();
    toast(`Dataset load failed: ${e.message || 'unknown error'}`, 'error');
  }
}

function toggleDatasetDrawer(open) {
  const drawer = $('#datasetCaptionDrawer');
  if (drawer) drawer.style.display = open ? 'flex' : 'none';
  if (open) checkDatasetVisionRuntime({ quiet: true });
}

function resetDatasetConsole() {
  const el = $('#captionConsole');
  if (el) el.textContent = 'Ready.';
}

function datasetLog(message, kind = 'info') {
  const el = $('#captionConsole');
  if (!el) return;
  const prefix = kind === 'error' ? 'ERR' : kind === 'warn' ? 'WARN' : 'OK';
  el.textContent = `${(el.textContent || '').trim()}\n[${prefix}] ${message}`.trim();
  el.scrollTop = el.scrollHeight;
}

function renderDatasetWorkspace() {
  renderDatasetStats();
  renderDatasetProcess();
  renderDatasetGrid();
  renderDatasetInspector();
}

function renderDatasetStats() {
  const host = $('#datasetStats');
  if (!host) return;
  const c = datasetStatCounts();
  host.innerHTML = [
    ['Images', c.all],
    ['Included', c.included],
    ['Captioned', c.captioned],
    ['Needs caption', c.missing],
    ['Excluded', c.excluded],
  ].map(([label, value]) => `<div class="dataset-stat"><strong>${value}</strong><span>${label}</span></div>`).join('');
}

function renderDatasetProcess() {
  const c = datasetStatCounts();
  const stepState = {
    upload: c.all > 0 ? 'done' : 'active',
    review: c.all > 0 && c.excluded > 0 ? 'done' : c.all > 0 ? 'active' : '',
    caption: c.missing === 0 && c.included > 0 ? 'done' : c.included > 0 ? 'active' : '',
    curate: c.captioned > 0 ? 'active' : '',
    train: c.ready ? 'done' : '',
  };
  $$('.dataset-process-step').forEach(step => {
    const s = stepState[step.dataset.dsStep] || '';
    step.classList.toggle('done', s === 'done');
    step.classList.toggle('active', s === 'active');
  });
}

function filteredDatasetItems() {
  const rows = state.datasetItems || [];
  if (state.datasetFilter === 'included') return rows.filter(x => !x.excluded);
  if (state.datasetFilter === 'missing') return rows.filter(x => !x.excluded && !(x.caption || '').trim());
  if (state.datasetFilter === 'excluded') return rows.filter(x => x.excluded);
  return rows;
}

function renderDatasetGrid() {
  const grid = $('#datasetWorkspaceGrid');
  if (!grid) return;
  const rows = filteredDatasetItems();
  if (!rows.length) {
    const emptyText = state.datasetItems.length ? 'No images match this filter.' : 'Upload images to start building this dataset.';
    grid.innerHTML = `<div class="empty dataset-grid-empty"><span>${emptyText}</span></div>`;
    return;
  }
  grid.innerHTML = rows.map((item) => {
    const idx = state.datasetItems.indexOf(item);
    const cap = (item.caption || '').trim();
    const flags = item.flags || [];
    const status = item.excluded ? 'Excluded' : cap ? 'Captioned' : 'Needs caption';
    return `
      <button class="dataset-grid-item ${idx === state.datasetSelected ? 'active' : ''} ${item.excluded ? 'excluded' : ''} ${cap ? 'captioned' : 'needs-caption'}" data-dataset-index="${idx}" type="button">
        <img src="${esc(item.url || item.object_url || '')}" alt="${esc(item.image || 'dataset image')}" loading="lazy">
        <span class="dataset-tile-badge">${esc(status)}</span>
        ${flags.length ? `<span class="dataset-tile-flag">${esc(flags[0].label || 'flag')}</span>` : ''}
      </button>`;
  }).join('');
  $$('[data-dataset-index]', grid).forEach(tile => {
    tile.addEventListener('click', () => {
      state.datasetSelected = Number(tile.dataset.datasetIndex);
      renderDatasetGrid();
      renderDatasetInspector();
    });
  });
}

function selectedDatasetItem() {
  return state.datasetItems[state.datasetSelected] || null;
}

function renderDatasetInspector() {
  const item = selectedDatasetItem();
  const empty = $('#inspectorEmptyState');
  const content = $('#inspectorContent');
  if (!empty || !content) return;
  if (!item) {
    empty.style.display = 'flex';
    content.style.display = 'none';
    return;
  }
  empty.style.display = 'none';
  content.style.display = 'flex';
  $('#inspectorImage').src = item.url || item.object_url || '';
  $('#inspectorFilename').textContent = item.image || item.filename || 'image';
  const sizeKb = item.size_bytes ? `${Math.max(1, Math.round(item.size_bytes / 1024))} KB` : 'local file';
  $('#inspectorDetails').textContent = `${item.excluded ? 'Excluded' : 'Included'} · ${sizeKb}`;
  $('#inspectorCaptionText').value = item.caption || '';
  $('#captionLength').textContent = `${(item.caption || '').length} chars`;
  const warning = $('#inspectorWarning');
  const flag = (item.flags || [])[0];
  if (warning && flag) {
    warning.style.display = 'block';
    warning.textContent = `${flag.label}: ${flag.detail || 'needs attention'}`;
  } else if (warning) {
    warning.style.display = 'none';
  }
  $('#btnToggleExclude').textContent = item.excluded ? 'Include' : 'Exclude';
}

function setDatasetBusy(busy) {
  state.datasetBusy = busy;
  const run = $('#btnRunCaptioning');
  const stop = $('#btnCancelCaptioning');
  if (run) run.disabled = busy;
  if (stop) stop.style.display = busy ? '' : 'none';
  $('#datasetDropZone')?.classList.toggle('busy', busy);
}

async function uploadFilesToActiveDataset(files) {
  const images = (files || []).filter(f => /^image\//.test(f.type));
  if (!state.activeDataset?.dataset_path) return toast('Create or open a dataset first.', 'error');
  if (!images.length) return toast('Choose image files to upload.', 'error');
  setDatasetBusy(true);
  datasetLog(`Uploading ${images.length} image${images.length === 1 ? '' : 's'}...`);
  try {
    if (state.preview) {
      const store = state.previewDatasetStore[state.activeDataset.dataset_path] || state.activeDataset;
      store.items = store.items || [];
      for (const file of images) {
        store.items.push({
          image: file.name,
          filename: file.name,
          caption: '',
          excluded: false,
          flags: [{ label: 'NO CAP', tone: 'warn', detail: 'no caption yet' }],
          size_bytes: file.size,
          object_url: URL.createObjectURL(file),
          image_base64: await fileToDataUrl(file),
        });
      }
      store.image_count = store.items.length;
      store.caption_count = store.items.filter(x => x.caption).length;
      state.previewDatasetStore[state.activeDataset.dataset_path] = store;
      state.activeDataset = store;
    } else {
      const res = await api.uploadDatasetFiles(images, state.activeDataset.dataset_path);
      if (res?.dataset_path && res.dataset_path !== state.activeDataset.dataset_path) {
        state.activeDataset = {
          ...(state.activeDataset || {}),
          name: datasetNameFromPath(res.dataset_path),
          dataset_path: res.dataset_path,
          virtual: false,
        };
        $('#activeDatasetTitle') && ($('#activeDatasetTitle').textContent = state.activeDataset.name);
        $('#activeDatasetSub') && ($('#activeDatasetSub').textContent = state.activeDataset.dataset_path);
      } else {
        state.activeDataset.virtual = false;
      }
    }
    await loadActiveDatasetItems();
    datasetLog('Upload complete.');
  } catch (e) {
    toast(`Upload failed: ${e.message || 'unknown error'}`, 'error');
    datasetLog(e.message || 'Upload failed', 'error');
  } finally {
    setDatasetBusy(false);
  }
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error || new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

function datasetVisionSettings() {
  return {
    provider: $('#captionProvider')?.value || 'openai',
    endpoint: ($('#captionEndpoint')?.value || '').trim(),
    model: ($('#captionModel')?.value || '').trim(),
    temperature: Number($('#captionTemp')?.value || 0.1),
  };
}

function renderDatasetVisionRuntime(payload = {}) {
  const pill = $('#captionRuntimePill');
  const status = $('#captionRuntimeStatus');
  const log = $('#captionRuntimeLog');
  const start = $('#btnStartCaptionRuntime');
  const stop = $('#btnStopCaptionRuntime');
  if (!pill || !status) return;
  const stateName = payload.state || (payload.ready ? 'ready' : payload.running ? 'starting' : 'stopped');
  const loading = stateName === 'starting' || payload.status === 'starting';
  pill.dataset.state = payload.ready ? 'on' : loading ? 'loading' : stateName === 'exited' ? 'off' : 'unknown';
  pill.textContent = payload.ready
    ? (stateName === 'external' ? 'Vision: external ready' : 'Vision: ready')
    : loading ? 'Vision: starting'
      : stateName === 'exited' ? 'Vision: exited'
        : 'Vision: stopped';
  if (payload.ready) {
    status.textContent = `${payload.model || 'vision model'} is reachable at ${payload.endpoint || 'endpoint'}.`;
  } else if (loading) {
    status.textContent = 'Starting the pod vision server. First load can take a few minutes.';
  } else if (payload.probe?.error) {
    status.textContent = payload.probe.error;
  } else if (payload.returncode != null) {
    status.textContent = `Managed server exited with code ${payload.returncode}.`;
  } else {
    status.textContent = 'No vision server is reachable yet.';
  }
  if (log) {
    const lines = payload.log || [];
    log.textContent = lines.join('\n');
    log.style.display = lines.length ? '' : 'none';
  }
  if (start) start.disabled = loading || payload.ready;
  if (stop) stop.disabled = !payload.running;
}

async function checkDatasetVisionRuntime({ quiet = false } = {}) {
  const settings = datasetVisionSettings();
  if (!settings.endpoint || !settings.model) return null;
  try {
    const payload = state.preview
      ? { ready: true, state: 'external', model: settings.model, endpoint: settings.endpoint, log: ['preview mode: vision endpoint mocked'] }
      : await api.datasetVisionRuntimeStatus(settings);
    renderDatasetVisionRuntime(payload);
    if (!quiet) toast(payload.ready ? 'Vision model is reachable' : 'Vision model is not running yet', payload.ready ? 'success' : 'info');
    return payload;
  } catch (e) {
    renderDatasetVisionRuntime({ state: 'stopped', probe: { error: e.message || 'status unavailable' } });
    if (!quiet) toast(`Vision check failed: ${e.message || 'unknown error'}`, 'error');
    return null;
  }
}

async function startDatasetVisionRuntime() {
  const settings = datasetVisionSettings();
  if (!settings.endpoint || !settings.model) return toast('Set the vision endpoint and model first.', 'error');
  const btn = $('#btnStartCaptionRuntime');
  if (btn) btn.disabled = true;
  renderDatasetVisionRuntime({ state: 'starting', model: settings.model, endpoint: settings.endpoint });
  try {
    const payload = state.preview
      ? { ready: true, state: 'external', model: settings.model, endpoint: settings.endpoint, log: ['preview mode: vision endpoint mocked'] }
      : await api.datasetVisionRuntimeStart(settings);
    renderDatasetVisionRuntime(payload);
    datasetLog(`Vision runtime ${payload.status || payload.state || 'starting'}.`);
    if (!payload.ready) pollDatasetVisionRuntime();
    toast(payload.ready ? 'Vision model is already reachable' : 'Starting vision model', payload.ready ? 'success' : 'info');
  } catch (e) {
    renderDatasetVisionRuntime({ state: 'stopped', probe: { error: e.message || 'start failed' } });
    toast(`Vision start failed: ${e.message || 'unknown error'}`, 'error');
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function stopDatasetVisionRuntime() {
  const btn = $('#btnStopCaptionRuntime');
  if (btn) btn.disabled = true;
  try {
    const payload = state.preview ? { state: 'stopped', ready: false } : await api.datasetVisionRuntimeStop();
    renderDatasetVisionRuntime(payload);
    datasetLog('Vision runtime stopped.');
    toast('Vision model stopped', 'success');
  } catch (e) {
    toast(`Vision stop failed: ${e.message || 'unknown error'}`, 'error');
  } finally {
    if (btn) btn.disabled = false;
  }
}

function pollDatasetVisionRuntime() {
  clearTimeout(state.datasetVisionPoll);
  state.datasetVisionPoll = setTimeout(async () => {
    const payload = await checkDatasetVisionRuntime({ quiet: true });
    if (payload && !payload.ready && (payload.running || payload.state === 'starting')) {
      pollDatasetVisionRuntime();
    }
  }, 2500);
}

async function runDatasetPipeline() {
  if (!state.activeDataset?.dataset_path) return toast('Open a dataset first.', 'error');
  const rows = state.datasetItems || [];
  if (!rows.length) return toast('Upload images before running the pipeline.', 'error');
  const settings = datasetVisionSettings();
  if (!settings.endpoint || !settings.model) return toast('Set the vision endpoint and model first.', 'error');
  const runtime = await checkDatasetVisionRuntime({ quiet: true });
  if (!runtime?.ready) {
    toggleDatasetDrawer(true);
    return toast('Start the pod vision model before running the dataset pipeline.', 'error');
  }

  datasetPipelineCancel = false;
  setDatasetBusy(true);
  toggleDatasetDrawer(true);
  resetDatasetConsole();
  datasetLog(`Starting auto-review + caption with ${settings.model}`);
  try {
    for (let i = 0; i < rows.length; i++) {
      if (datasetPipelineCancel) break;
      const item = rows[i];
      if (item.excluded) continue;
      datasetLog(`Review ${i + 1}/${rows.length}: ${item.image}`);
      const reviewText = await requestDatasetVision(item, DATASET_REVIEW_PROMPT, settings);
      const review = parseDatasetReview(reviewText);
      item.review = review;
      if (review && review.keep === false) {
        item.excluded = true;
        item.flags = [{ label: 'REJECTED', tone: 'warn', detail: review.reason || 'auto-review rejected this image' }, ...(item.flags || [])];
        await persistDatasetExclude(item, true);
        datasetLog(`Excluded ${item.image}: ${review.reason || 'review rejected'}`, 'warn');
        renderDatasetWorkspace();
        continue;
      }

      if (!(item.caption || '').trim()) {
        if (datasetPipelineCancel) break;
        datasetLog(`Caption ${i + 1}/${rows.length}: ${item.image}`);
        const prompt = ($('#captionPrompt')?.value || DATASET_CAPTION_FALLBACK).trim();
        const raw = await requestDatasetVision(item, prompt, settings);
        const caption = cleanDatasetCaption(raw);
        if (!caption) throw new Error(`Captioner returned empty text for ${item.image}`);
        item.caption = caption;
        item.caption_path = item.caption_path || item.image.replace(/\.[^.]+$/, '.txt');
        item.flags = (item.flags || []).filter(f => !['NO CAP', 'EMPTY CAP', 'SHORT CAP'].includes(f.label));
        await persistDatasetCaption(item, caption);
        datasetLog(`Saved caption for ${item.image}`);
        renderDatasetWorkspace();
      }
    }
    datasetLog(datasetPipelineCancel ? 'Pipeline stopped by user.' : 'Pipeline complete.');
    toast(datasetPipelineCancel ? 'Dataset pipeline stopped' : 'Dataset pipeline complete', datasetPipelineCancel ? 'info' : 'success');
  } catch (e) {
    datasetLog(e.message || 'Pipeline failed', 'error');
    toast(`Pipeline failed: ${e.message || 'unknown error'}`, 'error');
  } finally {
    datasetPipelineCancel = false;
    setDatasetBusy(false);
    await loadActiveDatasetItems();
  }
}

async function requestDatasetVision(item, prompt, settings) {
  if (state.preview) {
    await new Promise(r => setTimeout(r, 160));
    if (prompt === DATASET_REVIEW_PROMPT) return '{"keep":true,"reason":"single clear subject"}';
    return 'A close portrait with clear facial details, relaxed posture, visible clothing layers, soft ambient lighting, and a simple background.';
  }
  const imagePath = `${state.activeDataset.dataset_path}/${item.image}`;
  const res = await api.datasetVisionProxy({
    provider: settings.provider,
    endpoint: settings.endpoint,
    model: settings.model,
    temperature: settings.temperature,
    prompt,
    image_path: imagePath,
  });
  return res.response || '';
}

function parseDatasetReview(text) {
  const raw = String(text || '').trim();
  const jsonText = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
  try {
    const parsed = JSON.parse(jsonText);
    return { keep: parsed.keep !== false, reason: String(parsed.reason || '').slice(0, 160) };
  } catch {
    return { keep: !/\breject\b|\bmultiple\b|\bobject\b|\bwatermark\b/i.test(raw), reason: raw.slice(0, 160) };
  }
}

function cleanDatasetCaption(text) {
  let out = String(text || '')
    .replace(/^["'`\s]+|["'`\s]+$/g, '')
    .replace(/^(in this image|we see|this is a shot of|the image shows|there is)\s*:?\s*/i, '')
    .replace(/\b(photorealistic|masterpiece|best quality|high quality|ultra detailed|8k|ai generated|generated image|stunning photography|hyperrealistic)\b/gi, '')
    .replace(/[“”]{2,}/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  out = out.replace(/\s+([,.])/g, '$1').replace(/,\s*,+/g, ',');
  return out.slice(0, 900);
}

async function persistDatasetCaption(item, caption) {
  if (state.preview) return;
  await api.saveTrainerCaption({
    dataset_path: state.activeDataset.dataset_path,
    image_rel: item.image,
    caption,
  });
}

async function persistDatasetExclude(item, excluded) {
  if (state.preview) return;
  await api.toggleTrainerExclude({
    dataset_path: state.activeDataset.dataset_path,
    image_rel: item.image,
    excluded,
  });
}

async function saveDatasetInspectorCaption() {
  const item = selectedDatasetItem();
  if (!item) return;
  const caption = cleanDatasetCaption($('#inspectorCaptionText')?.value || '');
  try {
    await persistDatasetCaption(item, caption);
    item.caption = caption;
    item.caption_path = item.caption_path || item.image.replace(/\.[^.]+$/, '.txt');
    item.flags = (item.flags || []).filter(f => !['NO CAP', 'EMPTY CAP', 'SHORT CAP'].includes(f.label));
    renderDatasetWorkspace();
    toast('Caption saved', 'success');
  } catch (e) {
    toast(`Save failed: ${e.message || 'unknown error'}`, 'error');
  }
}

async function toggleDatasetInspectorExclude() {
  const item = selectedDatasetItem();
  if (!item) return;
  const next = !item.excluded;
  try {
    await persistDatasetExclude(item, next);
    item.excluded = next;
    renderDatasetWorkspace();
    toast(next ? 'Image excluded' : 'Image included', 'success');
  } catch (e) {
    toast(`Exclude failed: ${e.message || 'unknown error'}`, 'error');
  }
}

async function sendDatasetToTrainer() {
  if (!state.activeDataset?.dataset_path) return toast('Open a dataset first.', 'error');
  const c = datasetStatCounts();
  if (c.included < 8) return toast('Keep at least 8 included images before training.', 'error');
  if (c.missing > 0 && !await confirmModal('Missing captions', `${c.missing} included images still need captions. Send to trainer anyway?`, 'Send anyway')) return;
  setTrainerWizardSource('folder');
  setTrainerDatasetPath(state.activeDataset.dataset_path);
  const input = $('#trainerWizardFolder');
  if (input) input.value = state.activeDataset.dataset_path;
  switchView('trainers');
  setTrainerWizardStep(1);
  validateTrainerDataset().catch(() => {});
  toast('Dataset loaded into trainer', 'success');
}

// ── Trainers ────────────────────────────────────────────────────────────
async function loadTrainers() {
  try {
    const data = state.preview ? mockTrainers() : await api.trainers();
    state.trainers = data.trainers || [];
    state.trainJobs = data.jobs || state.trainJobs || [];
  } catch {
    state.trainers = [];
    state.trainJobs = [];
  }
  renderTrainers();
}

function activeTrainer() {
  const wanted = trainerWizardCfg?.trainerId || 'qwen-character-lora';
  return state.trainers.find(t => t.id === wanted) || state.trainers.find(t => t.id === 'qwen-character-lora') || state.trainers[0] || null;
}

// ── Trainer wizard (Phase 1) ────────────────────────────────────────────
// State lives on the module — small enough not to warrant the global
// state object. Steps 2-5 are stubbed (Phase 2 work); Step 1 is the
// fully-functional recipe picker + run name. Next/Back navigate
// strictly; clicking a completed stepper chip jumps back to that step.
const RECIPES = {
  character: { steps: 3000, rank: 64,  lr: 0.0002, label: 'Character / person' },
  style:     { steps: 4000, rank: 32,  lr: 0.0001, label: 'Art style' },
  concept:   { steps: 2500, rank: 32,  lr: 0.0002, label: 'Concept / object' },
  custom:    { steps: 2000, rank: 32,  lr: 0.0002, label: 'From scratch' },
};
let trainerWizardStep = 0;
let trainerWizardRecipe = 'character';
let trainerWizardSource = 'hf';            // 'hf' | 'upload' | 'folder'
let trainerWizardUploadFiles = [];          // File[] queued for Step 2 upload
let trainerWizardDatasetPath = 'datasets/kerry';
let trainerValidationPath = '';             // path that state.trainValidation belongs to

// ── Step 3 state ──
let trainerCurateDataset = null;            // { dataset_path, pair_count, pairs, health }
let trainerCurateSelected = null;           // image rel_path
let trainerCurateFilter = 'all';            // 'all' | 'flagged' | 'missing' | 'included'
let trainerCuratePendingSaves = 0;           // caption/exclude writes still in flight

// ── Step 4 state ──
// trainerWizardCfg is the wizard's source of truth for the launch payload.
// Advanced knobs are written into the backend job manifest and trainer wrapper
// environment so the generated command and real training run stay aligned.
const trainerWizardCfg = {
  trainerId: 'qwen-character-lora',
  trigger:   'A woman named Kerry',
  base:      'Qwen/Qwen-Image-2512',
  steps:     3000,
  rank:      64,
  lr:        0.0002,
  res:       1024,
  batch:     1,
  repeats:   1,
  opt:       'adamw8bit',
  sched:     'cosine',
  alpha:     64,
  saveEvery: 500,
  precision: 'bf16',
  gradCkpt:  'on',
  hourlyRate: 3.6,
};
const trainerWizardSamples = [
  'A woman named Kerry, portrait, natural skin texture, studio lighting',
  'A woman named Kerry, smiling, golden hour, soft daylight',
];
let trainerWizardAdvanced = false;
let trainerAlphaTouched = false;

// Step 5 launch options. Mirrored into the trainer payload at launch time
// (Phase 3 will wire 'notify' + 'autoShutdown' to backend hooks; right
// now they're UI-only acknowledgements).
const trainerLaunchOpts = {
  samples:      true,
  autoPublish:  true,
  notify:       false,
  autoShutdown: false,
};

function bindTrainerWizard() {
  // Recipe selection — update the wizard config that feeds the launch payload.
  $$('.recipe-card').forEach(card => {
    card.addEventListener('click', () => {
      trainerWizardRecipe = card.dataset.recipe;
      $$('.recipe-card').forEach(c => c.classList.toggle('selected', c === card));
      applyRecipeToWizard();
    });
  });
  // Live-derive filename hint from the name input.
  $('#trainerRunName')?.addEventListener('input', renderTrainerFilenameHint);
  renderTrainerFilenameHint();
  // Step navigation.
  $('#trainerWizardNext')?.addEventListener('click', onTrainerWizardNext);
  $('#trainerWizardBack')?.addEventListener('click', () => setTrainerWizardStep(trainerWizardStep - 1));
  $('#trainerWizardCancel')?.addEventListener('click', () => {
    setTrainerWizardStep(0);
    toast('Trainer wizard reset', 'info');
  });
  $$('#trainerStepper .px-step').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.step);
      // Only allow jumping back to completed steps (left-of-current).
      if (idx < trainerWizardStep) setTrainerWizardStep(idx);
    });
  });
  // GPU chip — pull current VRAM tier from state.gpu if available.
  renderTrainerGpuChip();

  // ── Step 2: source picker ──
  $$('.source-card').forEach(card => {
    card.addEventListener('click', () => setTrainerWizardSource(card.dataset.source));
  });

  // HF inputs — the wizard owns the dataset source and launch payload.
  const hfRepo   = $('#trainerWizardHFRepo');
  const hfType   = $('#trainerWizardHFType');
  const hfTarget = $('#trainerWizardHFTarget');
  hfRepo  ?.addEventListener('input',  () => clearTrainerDatasetState());
  hfType  ?.addEventListener('change', () => clearTrainerDatasetState());
  hfTarget?.addEventListener('input',  () => {
    const path = $('#trainerWizardHFCachePath');
    if (path) path.textContent = `datasets/${(hfTarget.value || 'kerry').trim()}`;
    setTrainerDatasetPath(`datasets/${(hfTarget.value || 'kerry').trim()}`);
    clearTrainerDatasetState();
  });
  $('#trainerWizardHFPull')?.addEventListener('click', async () => {
    await downloadTrainerDataset();
  });

  // Upload — collect files, count them, queue for upload on Next.
  const drop  = $('#trainerWizardDrop');
  const file  = $('#trainerWizardFileInput');
  const browse = $('#trainerWizardBrowseBtn');
  browse?.addEventListener('click', (e) => { e.stopPropagation(); file?.click(); });
  drop?.addEventListener('click',  () => file?.click());
  file?.addEventListener('change', () => addTrainerUploadFiles([...(file.files || [])]));
  drop?.addEventListener('dragover',  (e) => { e.preventDefault(); drop.classList.add('dragging'); });
  drop?.addEventListener('dragleave', () => drop.classList.remove('dragging'));
  drop?.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('dragging');
    addTrainerUploadFiles([...(e.dataTransfer?.files || [])]);
  });

  // Folder — path input + validate.
  $('#trainerWizardFolder')?.addEventListener('input', (e) => {
    setTrainerDatasetPath(e.target.value);
    clearTrainerDatasetState();
  });
  $('#trainerWizardFolderValidate')?.addEventListener('click', async () => {
    await validateTrainerDataset();
    renderTrainerWizardStats(state.trainValidation);
    const status = $('#trainerWizardFolderStatus');
    if (status) {
      if (state.trainValidation?.valid) {
        status.innerHTML = `<span class="ok">✓ Dataset readable</span>`;
      } else if (isTrainerValidationUsable()) {
        status.innerHTML = `<span class="warn">✓ Readable · fix captions in Curate</span>`;
      } else {
        status.innerHTML = `<span class="err">✗ ${esc(state.trainValidation?.error || 'Validation failed')}</span>`;
      }
    }
  });

  // Re-render the stat strip when validation changes.
  document.addEventListener('forge:trainer-validated', () => renderTrainerWizardStats(state.trainValidation));

  // ── Step 4: trigger / base / knobs / samples / summary ──
  bindTrainerConfigure();
  // ── Step 5: launch ──
  bindTrainerLaunch();

  // ── Step 3: filter pills + bulk actions + inspector ──
  $$('#trainerCurateFilters .filter-pill').forEach(p => {
    p.addEventListener('click', () => {
      trainerCurateFilter = p.dataset.filter;
      $$('#trainerCurateFilters .filter-pill').forEach(x => x.classList.toggle('active', x === p));
      renderTrainerCurateGrid();
    });
  });
  $('#trainerCurateReload')?.addEventListener('click', () => loadTrainerCurate(true));
  $('#trainerInspectorSave')?.addEventListener('click', saveTrainerInspectorCaption);
  $('#trainerInspectorCaption')?.addEventListener('input', () => {
    const text = $('#trainerInspectorCaption').value;
    const meta = $('#trainerInspectorCaptionMeta');
    if (meta) meta.textContent = `${text.length} chars · ${text.split(/\s+/).filter(Boolean).length} tokens`;
  });
}

function currentTrainerDatasetPath() {
  return (trainerWizardDatasetPath || '').trim();
}

function setTrainerDatasetPath(value) {
  trainerWizardDatasetPath = (value || '').trim();
}

function resetTrainerCurateState() {
  trainerCurateDataset = null;
  trainerCurateSelected = null;
  trainerCuratePendingSaves = 0;
}

function clearTrainerDatasetState() {
  state.trainValidation = null;
  trainerValidationPath = '';
  resetTrainerCurateState();
  renderTrainerWizardStats(null);
  renderTrainerDatasetSummary(null);
  renderTrainerConfigureSummary();
  updateTrainerWizardNextGate();
}

function setTrainerValidation(summary, datasetPath) {
  state.trainValidation = summary || null;
  trainerValidationPath = (datasetPath || summary?.requested_path || currentTrainerDatasetPath()).trim();
  resetTrainerCurateState();
  renderTrainerDatasetSummary(state.trainValidation);
  renderTrainerWizardStats(state.trainValidation);
  document.dispatchEvent(new CustomEvent('forge:trainer-validated'));
}

function isTrainerValidationUsable() {
  const summary = state.trainValidation;
  if (!summary) return false;
  if (trainerValidationPath && trainerValidationPath !== currentTrainerDatasetPath()) return false;
  return Number(summary.image_count || 0) > 0;
}

function setTrainerWizardSource(id) {
  if (trainerWizardSource === id) return;
  trainerWizardSource = id;
  $$('.source-card').forEach(c => c.classList.toggle('selected', c.dataset.source === id));
  $$('[data-source-panel]').forEach(p => { p.hidden = p.dataset.sourcePanel !== id; });
  if (id === 'hf') {
    const target = ($('#trainerWizardHFTarget')?.value || 'kerry').trim() || 'kerry';
    setTrainerDatasetPath(`datasets/${target}`);
  } else if (id === 'folder') {
    setTrainerDatasetPath($('#trainerWizardFolder')?.value || '');
  }
  clearTrainerDatasetState();
}

function addTrainerUploadFiles(files) {
  if (!files.length) return;
  trainerWizardUploadFiles.push(...files);
  clearTrainerDatasetState();
  const count = $('#trainerWizardUploadCount');
  if (count) {
    const imgs = trainerWizardUploadFiles.filter(f => /^image\//.test(f.type)).length;
    const txts = trainerWizardUploadFiles.length - imgs;
    count.innerHTML = `<b style="color:var(--px-text)">${imgs} image${imgs === 1 ? '' : 's'}</b> queued · ${txts} caption file${txts === 1 ? '' : 's'}`;
  }
}

async function uploadTrainerDatasetFromWizard() {
  if (!trainerWizardUploadFiles.length) return null;
  const runName = ($('#trainerRunName')?.value || 'dataset').trim();
  const count = $('#trainerWizardUploadCount');
  if (count) count.innerHTML = `<span class="spinner"></span><span>Uploading ${trainerWizardUploadFiles.length} files…</span>`;
  try {
    let res;
    if (state.preview) {
      await delay(450);
      const target = runName.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'dataset';
      res = {
        dataset_path: `datasets/uploads/${target}_preview`,
        scan: { valid: true, image_count: 80, caption_count: 80, pair_count: 80 },
      };
    } else {
      res = await api.uploadTrainerDataset(trainerWizardUploadFiles, `${runName}_dataset`);
    }
    setTrainerDatasetPath(res.dataset_path || '');
    setTrainerValidation(res.scan || null, res.dataset_path || currentTrainerDatasetPath());
    trainerWizardUploadFiles = [];
    if (count) count.innerHTML = `<b style="color:var(--px-text)">${res.scan?.image_count || 0} images</b> uploaded · ${res.dataset_path || 'dataset cached'}`;
    toast('Dataset uploaded', 'success');
    return res;
  } catch (e) {
    clearTrainerDatasetState();
    if (count) count.innerHTML = `<b style="color:var(--px-text)">${trainerWizardUploadFiles.length} files</b> queued`;
    toast(e.message || 'Dataset upload failed', 'error');
    return null;
  }
}

function renderTrainerWizardStats(summary) {
  const strip = $('#trainerWizardStats');
  if (!strip) return;
  const cards = $$('.stat-card', strip);
  if (!summary) {
    cards.forEach(c => {
      c.classList.add('empty');
      const num = $('.stat-num', c);
      if (num) num.textContent = '—';
    });
    updateTrainerWizardNextGate();
    return;
  }
  const images   = Number(summary.image_count   || 0);
  const captions = Number(summary.caption_count || 0);
  const pairs    = Number(summary.pair_count    || 0);
  const missing  = (summary.missing_captions || []).length;
  const orphan   = (summary.orphan_captions  || []).length;
  const empty    = (summary.empty_captions   || []).length;
  const issues   = missing + orphan + empty;

  const setStat = (key, value, subText) => {
    const card = strip.querySelector(`[data-stat="${key}"]`)?.closest('.stat-card');
    if (!card) return;
    card.classList.toggle('empty', value === '—' || value === 0);
    const num = $('.stat-num', card);
    const sub = card.querySelector(`[data-stat-sub="${key}"]`);
    if (num) num.textContent = String(value);
    if (sub && subText) sub.textContent = subText;
  };
  setStat('images',   images);
  setStat('captions', captions, missing ? `${missing} missing` : 'all matched');
  setStat('pairs',    pairs);
  setStat('issues',   issues,   issues ? 'auto-flagged' : 'none');
  // Update Next-gating based on validity.
  updateTrainerWizardNextGate();
}

function updateTrainerWizardNextGate() {
  const next = $('#trainerWizardNext');
  if (!next) return;
  if (trainerWizardStep === 1) {
    // Step 2: curation can fix captions, so require a readable image set,
    // not a fully train-ready caption set.
    if (trainerWizardSource === 'upload') {
      next.disabled = !(trainerWizardUploadFiles.length > 0 || isTrainerValidationUsable());
    } else {
      next.disabled = !isTrainerValidationUsable();
    }
    return;
  }
  if (trainerWizardStep === 2) {
    // Step 3: Need at least 8 included pairs and the dataset to be "ready".
    const included = trainerCurateDataset?.pairs.filter(p => !p.excluded).length || 0;
    const ready = trainerCurateDataset?.health?.ready;
    next.disabled = trainerCuratePendingSaves > 0 || !(included >= 8 && ready);
    return;
  }
  next.disabled = false;
}

// Next behavior is source-aware on Step 2: HF/folder need a validation,
// upload needs files queued. Other steps just advance.
async function onTrainerWizardNext() {
  if (trainerWizardStep === 1) {
    if (trainerWizardSource === 'upload' && trainerWizardUploadFiles.length > 0 && !isTrainerValidationUsable()) {
      const uploaded = await uploadTrainerDatasetFromWizard();
      if (!uploaded) return;
    }
    if (!isTrainerValidationUsable()) {
      if (trainerWizardSource === 'upload') {
        toast('Add images to upload before continuing.', 'error');
        return;
      }
      // Auto-run validate for HF/folder paths the user typed but didn't click.
      await validateTrainerDataset();
      if (!isTrainerValidationUsable()) {
        toast('Dataset has no readable training images yet.', 'error');
        return;
      }
    }
  }
  setTrainerWizardStep(trainerWizardStep + 1);
}

function applyRecipeToWizard() {
  const r = RECIPES[trainerWizardRecipe];
  if (!r) return;
  trainerWizardCfg.steps = r.steps;
  trainerWizardCfg.rank = r.rank;
  trainerWizardCfg.lr = r.lr;
  if (!trainerAlphaTouched) {
    trainerWizardCfg.alpha = r.rank;
    setTrainerOptionRowValue('alpha', r.rank);
  }
  setTrainerSliderValue('steps', r.steps);
  setTrainerOptionRowValue('rank', r.rank);
  setTrainerOptionRowValue('lr', r.lr);
  renderTrainerConfigureSummary();
}

function setTrainerSliderValue(key, value) {
  const slider = $(`input.px-slider[data-cfg="${key}"]`);
  if (slider) slider.value = String(value);
  const chip = document.querySelector(`[data-value="${key}"]`);
  if (chip) chip.textContent = String(value);
}

function setTrainerOptionRowValue(key, value) {
  const row = $(`.option-row[data-cfg="${key}"]`);
  if (!row) return;
  const wanted = String(value);
  $$('button[data-value]', row).forEach(btn => {
    btn.classList.toggle('selected', btn.dataset.value === wanted);
  });
}

function renderTrainerFilenameHint() {
  const name = ($('#trainerRunName')?.value || '').trim() || 'kerry_qwen_lora';
  const safe = name.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'kerry_qwen_lora';
  const hint = $('#trainerFilenameHint');
  if (hint) hint.textContent = `→ ${safe}.safetensors`;
}

function renderTrainerGpuChip() {
  const chip = $('#trainerGpuChip');
  if (!chip) return;
  const displayState = gpuDisplayState(state.gpu);
  const vram = gpuVramGb(state.gpu);
  chip.dataset.state = displayState;
  chip.classList.toggle('ok', displayState === 'ready');
  chip.classList.toggle('warn', displayState === 'loading' || displayState === 'disconnected');
  chip.classList.toggle('err', displayState === 'error');
  if (displayState === 'loading') {
    chip.textContent = 'GPU · checking...';
  } else if (displayState === 'error') {
    chip.textContent = 'GPU · unavailable';
  } else if (displayState === 'disconnected') {
    chip.textContent = 'GPU · disconnected';
  } else {
    chip.textContent = `GPU · ${vram ? `${Math.round(vram)} GB` : state.gpu?.name || 'ready'} · ready`;
  }
}

function setTrainerWizardStep(idx) {
  const clamped = Math.max(0, Math.min(4, idx));
  trainerWizardStep = clamped;
  $$('[data-wizard-step]').forEach(el => {
    el.hidden = Number(el.dataset.wizardStep) !== clamped;
  });
  $$('#trainerStepper .px-step').forEach((btn) => {
    const stepIdx = Number(btn.dataset.step);
    btn.classList.toggle('active', stepIdx === clamped);
    btn.classList.toggle('done',   stepIdx <  clamped);
  });
  const back = $('#trainerWizardBack');
  const next = $('#trainerWizardNext');
  if (back) back.disabled = (clamped === 0);
  if (next) {
    next.textContent = 'Next →';
    next.style.display = (clamped === 4) ? 'none' : '';
  }
  const label = $('#trainerStepLabel');
  if (label) label.textContent = `Step ${clamped + 1} of 5`;
  // Step 5 entry — refresh the generated command + final-check numerics.
  if (clamped === 4) {
    renderTrainerLaunchPreview();
  }
  // Step 3 entry — pull the dataset listing for the current path. Explicit
  // Reload refetches even when the path hasn't changed.
  if (clamped === 2 && (!trainerCurateDataset || trainerCurateDataset.requested_path !== currentTrainerDatasetPath())) {
    loadTrainerCurate(false);
  }
  // Re-evaluate Next-gate based on what the new step requires.
  updateTrainerWizardNextGate();
}

// ── Step 3 (Curate) data + rendering ────────────────────────────────────
function previewTrainerImageUrl(index) {
  const hue = (index * 37) % 360;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="640" viewBox="0 0 512 640">
    <rect width="512" height="640" fill="hsl(${hue},35%,18%)"/>
    <circle cx="256" cy="220" r="92" fill="hsl(${hue},45%,72%)"/>
    <rect x="124" y="330" width="264" height="240" rx="64" fill="hsl(${(hue + 90) % 360},45%,48%)"/>
    <text x="256" y="610" text-anchor="middle" font-family="monospace" font-size="34" fill="white">${String(index + 1).padStart(2, '0')}</text>
  </svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function previewTrainerCurateDataset(datasetPath) {
  const pairs = Array.from({ length: 18 }, (_, i) => ({
    image: `${String(i + 1).padStart(2, '0')}.png`,
    caption_path: `${String(i + 1).padStart(2, '0')}.txt`,
    caption: i % 7 === 0 ? '' : `A woman named Kerry, training reference image ${i + 1}, varied outfit and lighting.`,
    size_bytes: 420000 + i * 13000,
    flags: i % 7 === 0 ? [{ label: 'EMPTY CAP', tone: 'warn', detail: 'caption file exists but is empty' }] : [],
    excluded: i === 3,
    url: previewTrainerImageUrl(i),
  }));
  return {
    dataset_path: datasetPath,
    requested_path: datasetPath,
    pair_count: pairs.length,
    pairs,
    health: computeTrainerCurateHealth(pairs),
  };
}

async function loadTrainerCurate(force) {
  const datasetPath = currentTrainerDatasetPath();
  if (!datasetPath) {
    toast('Validate a dataset first in Step 2.', 'error');
    return;
  }
  if (trainerCurateDataset && !force && trainerCurateDataset.requested_path === datasetPath) return;
  if (!trainerCurateDataset || trainerCurateDataset.requested_path !== datasetPath) {
    resetTrainerCurateState();
  }
  const grid = $('#trainerCurateGrid');
  const empty = $('#trainerCurateEmpty');
  if (grid) grid.innerHTML = `<div class="px-mute" style="grid-column:1/-1;padding:24px;text-align:center">Loading…</div>`;
  if (empty) empty.hidden = true;
  try {
    if (state.preview) {
      await delay(250);
      trainerCurateDataset = previewTrainerCurateDataset(datasetPath);
    } else {
      trainerCurateDataset = await api.listTrainerDataset({ dataset_path: datasetPath });
    }
    trainerCurateDataset.requested_path = datasetPath;
    recomputeTrainerCurateHealth();
    if (!trainerCurateDataset?.pairs?.length) {
      if (grid) grid.innerHTML = '';
      if (empty) {
        empty.hidden = false;
        empty.textContent = 'No images found in this dataset folder.';
      }
      return;
    }
    // Auto-select the first image if nothing's selected yet.
    if (!trainerCurateSelected || !trainerCurateDataset.pairs.find(p => p.image === trainerCurateSelected)) {
      trainerCurateSelected = trainerCurateDataset.pairs[0].image;
    }
    renderTrainerCurateAll();
  } catch (e) {
    if (grid) grid.innerHTML = `<div class="px-mute" style="grid-column:1/-1;padding:24px;text-align:center;color:var(--px-err)">Failed to load: ${esc(e.message || 'unknown')}</div>`;
    updateTrainerWizardNextGate();
  }
}

function computeTrainerCurateHealth(pairs) {
  const included = (pairs || []).filter(p => !p.excluded);
  const total = included.length;
  const denom = total || 1;
  const covered = included.filter(p => p.caption_path).length;
  const hasText = included.filter(p => (p.caption || '').trim()).length;
  const avgLen = included.reduce((sum, p) => sum + (p.caption || '').trim().length, 0) / denom;
  const buckets = new Set(included.map(p => Math.floor(Number(p.size_bytes || 0) / (200 * 1024))));
  return {
    coverage: total ? Math.round(100 * covered / denom) : 0,
    captions: total ? Math.round(100 * hasText / denom) : 0,
    balance: avgLen ? Math.round(Math.max(0, Math.min(100, 100 * (1 - Math.abs(avgLen - 90) / 90)))) : 0,
    variety: total ? Math.min(100, Math.round(100 * buckets.size / Math.max(8, Math.floor(total / 6)))) : 0,
    ready: total >= 8 && covered === total && hasText === total,
  };
}

function recomputeTrainerCurateHealth() {
  if (!trainerCurateDataset) return;
  trainerCurateDataset.health = computeTrainerCurateHealth(trainerCurateDataset.pairs);
  trainerCurateDataset.included_count = trainerCurateDataset.pairs.filter(p => !p.excluded).length;
  trainerCurateDataset.excluded_count = trainerCurateDataset.pairs.length - trainerCurateDataset.included_count;
}

function renderTrainerCurateAll() {
  recomputeTrainerCurateHealth();
  renderTrainerCurateGrid();
  renderTrainerCurateInspector();
  renderTrainerCurateHealth();
  updateTrainerWizardNextGate();
}

// Filter + render the grid. Visibility is purely client-side — we hold the
// full list in trainerCurateDataset and never refetch on filter change.
function renderTrainerCurateGrid() {
  const grid = $('#trainerCurateGrid');
  const empty = $('#trainerCurateEmpty');
  if (!grid) return;
  if (!trainerCurateDataset) return;
  const all = trainerCurateDataset.pairs;
  const visible = all.filter(p => {
    if (trainerCurateFilter === 'flagged')  return p.flags.length > 0;
    if (trainerCurateFilter === 'missing')  return !p.caption;
    if (trainerCurateFilter === 'included') return !p.excluded;
    return true;
  });
  const counts = {
    all:      all.length,
    flagged:  all.filter(p => p.flags.length > 0).length,
    missing:  all.filter(p => !p.caption).length,
    included: all.filter(p => !p.excluded).length,
  };
  $$('#trainerCurateFilters [data-count]').forEach(el => {
    el.textContent = counts[el.dataset.count] ?? 0;
  });
  if (!visible.length) {
    grid.innerHTML = `<div class="px-mute" style="grid-column:1/-1;padding:24px;text-align:center">Nothing matches this filter.</div>`;
    if (empty) empty.hidden = true;
    return;
  }
  if (empty) empty.hidden = true;
  grid.innerHTML = visible.map(p => trainerCurateTileHtml(p)).join('');
  // Wire interactions per-tile. Doing it inline rather than via delegation
  // keeps the handler logic simple and the check vs body click distinct.
  $$('.curate-tile', grid).forEach(tile => {
    const rel = tile.dataset.rel;
    tile.addEventListener('click', () => {
      trainerCurateSelected = rel;
      renderTrainerCurateGrid();
      renderTrainerCurateInspector();
    });
    const check = $('.curate-check', tile);
    check?.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleTrainerCurateExclude(rel);
    });
  });
}

function trainerCurateTileHtml(p) {
  const selectedCls = (p.image === trainerCurateSelected) ? ' selected' : '';
  const excludedCls = p.excluded ? ' excluded' : '';
  const checkCls    = p.excluded ? '' : 'on';
  const chips = (p.flags || []).slice(0, 2)
    .map(f => `<span class="px-chip ${f.tone || 'warn'}">${esc(f.label)}</span>`)
    .join('');
  const nocap = !p.caption
    ? `<div class="curate-tile-nocap"><span class="px-chip warn">NO CAP</span></div>`
    : '';
  return `
    <div class="curate-tile${selectedCls}${excludedCls}" data-rel="${esc(p.image)}">
      <img src="${esc(p.url)}" alt="${esc(p.image)}" loading="lazy" onerror="this.style.display='none'">
      <button class="curate-check ${checkCls}" type="button" title="${p.excluded ? 'Include in training' : 'Exclude from training'}"></button>
      ${nocap}
      ${chips ? `<div class="curate-tile-chips">${chips}</div>` : ''}
    </div>`;
}

function renderTrainerCurateInspector() {
  const inspector = $('#trainerCurateInspector');
  if (!inspector || !trainerCurateDataset) return;
  const sel = trainerCurateDataset.pairs.find(p => p.image === trainerCurateSelected);
  if (!sel) {
    inspector.hidden = true;
    return;
  }
  inspector.hidden = false;
  const idx = trainerCurateDataset.pairs.indexOf(sel);
  const title = $('#trainerInspectorTitle');
  const sub   = $('#trainerInspectorSub');
  const img   = $('#trainerInspectorImg');
  const cap   = $('#trainerInspectorCaption');
  const meta  = $('#trainerInspectorCaptionMeta');
  const chip  = $('#trainerInspectorStatusChip');
  const flags = $('#trainerInspectorFlags');
  if (title) title.textContent = sel.image;
  if (sub)   sub.textContent   = `Image ${idx + 1} of ${trainerCurateDataset.pair_count}`;
  if (img)   img.src = sel.url;
  if (cap)   cap.value = sel.caption || '';
  if (meta)  meta.textContent = `${(sel.caption || '').length} chars · ${(sel.caption || '').split(/\s+/).filter(Boolean).length} tokens`;
  if (chip) {
    chip.className = 'px-chip ' + (sel.excluded ? 'warn' : (sel.flags.length ? 'warn' : 'ok'));
    chip.textContent = sel.excluded ? 'EXCLUDED' : (sel.flags.length ? 'FLAGGED' : 'INCLUDED');
  }
  if (flags) {
    if (!sel.flags.length) {
      flags.innerHTML = `<div class="flag-row"><span class="px-dot ok"></span><span>No issues detected.</span></div>`;
    } else {
      flags.innerHTML = sel.flags.map(f => `
        <div class="flag-row">
          <span class="px-dot ${f.tone === 'err' ? 'err' : 'warn'}"></span>
          <span><b style="color:var(--px-text)">${esc(f.label)}</b> — ${esc(f.detail || '')}</span>
        </div>`).join('');
    }
  }
}

function renderTrainerCurateHealth() {
  if (!trainerCurateDataset) return;
  const h = trainerCurateDataset.health || {};
  const set = (key, value) => {
    const pct = document.querySelector(`[data-health="${key}"]`);
    const bar = document.querySelector(`[data-health-bar="${key}"]`);
    const v = Math.max(0, Math.min(100, Number(value) || 0));
    if (pct) pct.textContent = `${v}%`;
    if (bar) bar.style.width = `${v}%`;
  };
  set('coverage', h.coverage);
  set('variety',  h.variety);
  set('captions', h.captions);
  set('balance',  h.balance);
  // Counts + ready chip
  const included = trainerCurateDataset.pairs.filter(p => !p.excluded).length;
  const excluded = trainerCurateDataset.pairs.length - included;
  const counts = $('#trainerHealthCounts');
  const ready  = $('#trainerHealthReady');
  if (counts) counts.textContent = `${included} included · ${excluded} excluded`;
  if (ready) {
    const isReady = h.ready && included >= 8;
    ready.className = 'px-chip ' + (isReady && trainerCuratePendingSaves === 0 ? 'ok' : 'warn');
    ready.textContent = trainerCuratePendingSaves > 0
      ? 'SAVING...'
      : (isReady ? '✓ READY TO TRAIN' : 'NOT READY');
  }
}

async function toggleTrainerCurateExclude(rel) {
  const pair = trainerCurateDataset?.pairs.find(p => p.image === rel);
  if (!pair) return;
  const next = !pair.excluded;
  // Optimistic flip so the click feels instant.
  pair.excluded = next;
  if (!state.preview) trainerCuratePendingSaves += 1;
  renderTrainerCurateAll();
  if (state.preview) return;
  try {
    await api.toggleTrainerExclude({
      dataset_path: trainerCurateDataset.dataset_path,
      image_rel:    rel,
      excluded:     next,
    });
  } catch (e) {
    // Revert on failure.
    pair.excluded = !next;
    toast(`Could not update exclude: ${e.message || 'unknown'}`, 'error');
  } finally {
    trainerCuratePendingSaves = Math.max(0, trainerCuratePendingSaves - 1);
    renderTrainerCurateAll();
  }
}

// ── Step 4 (Configure) ──────────────────────────────────────────────────
function bindTrainerConfigure() {
  // Trigger phrase
  $('#cfgTrigger')?.addEventListener('input', (e) => {
    trainerWizardCfg.trigger = e.target.value;
    renderTrainerConfigureSummary();
  });

  // Base model cards
  $$('#cfgBaseGrid .base-card').forEach(card => {
    card.addEventListener('click', () => {
      const id = card.dataset.base;
      trainerWizardCfg.base = id;
      trainerWizardCfg.trainerId = card.dataset.trainer || 'qwen-character-lora';
      $$('#cfgBaseGrid .base-card').forEach(c => c.classList.toggle('selected', c === card));
      renderTrainerConfigureSummary();
    });
  });

  // Sliders — Steps / Batch / Repeats. Update value chip + summary live.
  $$('.knob-grid input.px-slider, #cfgAdvanced input.px-slider').forEach(slider => {
    slider.addEventListener('input', () => {
      const key = slider.dataset.cfg;
      const val = Number(slider.value);
      trainerWizardCfg[key] = val;
      const value = document.querySelector(`[data-value="${key}"]`);
      if (value) value.textContent = String(val);
      renderTrainerConfigureSummary();
    });
  });

  // Option rows — rank / lr / res / advanced pills. Delegate to clicks
  // on inner buttons; first click wins, mark selected, persist.
  $$('.option-row').forEach(row => {
    row.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-value]');
      if (!btn) return;
      const key = row.dataset.cfg;
      const rawVal = btn.dataset.value;
      // Coerce types so summary math doesn't break (rank=64 not "64").
      const num = Number(rawVal);
      const val = Number.isFinite(num) && String(num) === rawVal ? num : rawVal;
      trainerWizardCfg[key] = val;
      if (key === 'alpha') {
        trainerAlphaTouched = true;
      }
      if (key === 'rank' && !trainerAlphaTouched) {
        trainerWizardCfg.alpha = Number(val) || trainerWizardCfg.rank;
        setTrainerOptionRowValue('alpha', trainerWizardCfg.alpha);
      }
      $$('button[data-value]', row).forEach(b => b.classList.toggle('selected', b === btn));
      renderTrainerConfigureSummary();
    });
  });

  // Advanced reveal
  $('#cfgAdvancedToggle')?.addEventListener('click', () => {
    trainerWizardAdvanced = !trainerWizardAdvanced;
    const adv = $('#cfgAdvanced');
    if (adv) adv.hidden = !trainerWizardAdvanced;
    const btn = $('#cfgAdvancedToggle');
    if (btn) btn.textContent = trainerWizardAdvanced ? 'Hide advanced' : 'Show advanced';
  });

  $('#cfgHourlyRate')?.addEventListener('input', (e) => {
    trainerWizardCfg.hourlyRate = Math.max(0, Number(e.target.value) || 0);
    renderTrainerConfigureSummary();
  });

  // Sample prompts list — initial render + add/remove
  renderTrainerSampleList();
  $('#cfgSampleAdd')?.addEventListener('click', () => {
    trainerWizardSamples.push('');
    renderTrainerSampleList();
    // Focus the new row's input so the user can start typing.
    const inputs = $$('#cfgSampleList input');
    inputs[inputs.length - 1]?.focus();
  });

  renderTrainerConfigureSummary();
}

function renderTrainerSampleList() {
  const list = $('#cfgSampleList');
  if (!list) return;
  list.innerHTML = trainerWizardSamples.map((s, i) => `
    <div class="sample-row" data-sample="${i}">
      <span class="sample-idx">${String(i + 1).padStart(2, '0')}</span>
      <input class="px-field" value="${esc(s)}" data-sample-input="${i}">
      <button class="icon-btn" data-sample-remove="${i}" title="Remove">×</button>
    </div>
  `).join('');
  // Wire inputs + remove buttons.
  $$('input[data-sample-input]', list).forEach(inp => {
    inp.addEventListener('input', () => {
      const idx = Number(inp.dataset.sampleInput);
      trainerWizardSamples[idx] = inp.value;
    });
  });
  $$('button[data-sample-remove]', list).forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.sampleRemove);
      trainerWizardSamples.splice(idx, 1);
      renderTrainerSampleList();
    });
  });
}

function renderTrainerConfigureSummary() {
  const cfg = trainerWizardCfg;
  const set = (key, value, color) => {
    const el = document.querySelector(`[data-summary="${key}"]`);
    if (!el) return;
    el.textContent = value;
    if (color) el.style.color = color;
  };
  // Recipe name + tint from the Step 1 selection.
  const recipeMeta = {
    character: { label: 'Character',  color: 'var(--px-pink)' },
    style:     { label: 'Art style',  color: 'var(--px-cyan)' },
    concept:   { label: 'Concept',    color: 'var(--px-lime)' },
    custom:    { label: 'Custom',     color: 'var(--px-violet)' },
  }[trainerWizardRecipe] || { label: trainerWizardRecipe, color: 'var(--px-text)' };
  set('recipe',  recipeMeta.label, recipeMeta.color);
  set('base',    cfg.base.replace(/^.*\//, ''));
  set('trigger', `"${cfg.trigger}"`);
  set('steps',   Number(cfg.steps).toLocaleString());
  set('rank',    String(cfg.rank));
  set('res',     `${cfg.res}px`);

  // VRAM budget — heuristic that scales with rank, batch, res. The
  // numbers are a rough proxy of resident weights + activations on an
  // 80 GB card; not authoritative, just enough to warn before launch.
  const vramGb = Math.min(76,
    cfg.rank * 0.4 + cfg.batch * 8 + cfg.res / 32 + 12 // base model ~12 GB residual
  );
  const vramPct = (vramGb / 80) * 100;
  const bar  = $('#cfgVramBar');
  const text = $('#cfgVramText');
  if (bar)  bar.style.width = `${vramPct}%`;
  if (text) text.textContent = `${Math.round(vramGb)} / 80 GB`;

  // ETA — same crude proxy as the design: ~55 steps/min on H100 for a
  // 20B Qwen-Image LoRA at rank 64. Real numbers come from trainer logs
  // once Phase 3 ships the live metrics stream.
  const etaMin = Math.max(1, Math.round(cfg.steps / 55));
  set('steps', Number(cfg.steps).toLocaleString());
  const etaEl  = $('#cfgEtaNum');
  if (etaEl) etaEl.textContent = String(etaMin);

  // Cost — assume H100 @ $3.60/hr default; configurable later via Settings.
  const ratePerHour = Math.max(0, Number(cfg.hourlyRate || 0));
  const cost = (etaMin / 60) * ratePerHour;
  const costEl = $('#cfgCostNum');
  if (costEl) costEl.textContent = `$${cost.toFixed(2)}`;
  const rateEl = $('#cfgHourlyRate');
  if (rateEl && document.activeElement !== rateEl) rateEl.value = ratePerHour ? ratePerHour.toFixed(2) : '';
  const rateLabel = $('#cfgRateLabel');
  if (rateLabel) {
    rateLabel.textContent = ratePerHour
      ? `Using your all-in instance rate: $${ratePerHour.toFixed(2)}/hr.`
      : 'Enter your RunPod instance rate including storage.';
  }

  // Check row — basic readiness: rank/steps/res in valid ranges and
  // dataset is ready (from Step 3 state).
  const ready = (
    Number(cfg.steps) >= 500 &&
    cfg.rank >= 8 &&
    cfg.res >= 512 &&
    !!trainerCurateDataset?.health?.ready &&
    trainerCuratePendingSaves === 0
  );
  const row = $('#cfgCheckRow');
  if (row) {
    row.innerHTML = ready
      ? `<span class="px-dot ok"></span><span class="px-mute">All checks pass · ready to launch</span>`
      : `<span class="px-dot warn"></span><span class="px-mute">${trainerCuratePendingSaves > 0 ? 'Saving curation changes…' : 'Curate at least 8 images in Step 3 first.'}</span>`;
  }
}

function trainerCaptionPathForImage(imageRel) {
  return String(imageRel || '').replace(/\.[^/.]+$/, '.txt');
}

// ── Step 5 (Launch) ─────────────────────────────────────────────────────
function bindTrainerLaunch() {
  // Confirmation check rows — first two are wired (samples is informational
  // and autoPublish controls whether the trained LoRA gets dropped into
  // /workspace/loras when done). Last two are Phase 3 placeholders.
  $$('[data-launch-flag]').forEach(row => {
    row.addEventListener('click', () => {
      if (row.dataset.disabled === 'true') return;
      const key = row.dataset.launchFlag;
      trainerLaunchOpts[key] = !trainerLaunchOpts[key];
      row.classList.toggle('on', trainerLaunchOpts[key]);
      renderTrainerLaunchPreview();
    });
  });

  // Copy the rendered command to clipboard.
  $('#launchCopyBtn')?.addEventListener('click', async () => {
    const block = $('#launchCmdBlock');
    if (!block) return;
    try {
      await navigator.clipboard.writeText(block.textContent || '');
      toast('Command copied', 'success');
    } catch (e) {
      toast(`Copy failed: ${e.message || 'unknown'}`, 'error');
    }
  });

  // Big launch CTA -> create the backend job from the wizard payload, then
  // land directly in the five-tab run monitor once the job exists.
  $('#launchTrainingBtn')?.addEventListener('click', async () => {
    const btn = $('#launchTrainingBtn');
    if (!btn) return;
    btn.disabled = true;
    const label = btn.innerHTML;
    btn.innerHTML = '<span class="bolt">⏳</span><span>Launching…</span>';
    try {
      const job = await startTrainerJob();
      if (!job?.id) return;
      openTrainerRunningView(job.id);
    } catch (e) {
      toast(`Launch failed: ${e.message || 'unknown'}`, 'error');
    } finally {
      btn.disabled = false;
      btn.innerHTML = label;
    }
  });
}

function renderTrainerLaunchPreview() {
  // Update the run name + summary numerics from current cfg.
  const name = $('#trainerRunName')?.value.trim() || 'kerry_qwen_lora';
  const safeName = name.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'kerry_qwen_lora';
  const cfg = trainerWizardCfg;
  const datasetPath = currentTrainerDatasetPath() || 'datasets/kerry';

  const runEl = $('#launchRunName');
  if (runEl) runEl.textContent = safeName;
  const etaMin = Math.max(1, Math.round(cfg.steps / 55));
  const hourlyRate = Math.max(0, Number(cfg.hourlyRate || 0));
  const cost   = (etaMin / 60) * hourlyRate;
  const setLaunch = (key, value) => {
    const el = document.querySelector(`[data-launch="${key}"]`);
    if (el) el.textContent = value;
  };
  setLaunch('cost',      `$${cost.toFixed(2)}`);
  setLaunch('eta',       `${etaMin} minutes`);
  setLaunch('saveevery', String(cfg.saveEvery));
  setLaunch('rate',      hourlyRate ? `$${hourlyRate.toFixed(2)}/hr` : 'rate not set');

  // Generated command — multi-line monospace with coloured spans for
  // arg names. Mirrors the actual trainer wrapper invocation shape
  // This mirrors the launch payload that backend/main.py sends as trainer
  // environment variables.
  const arg = (k, v) => `  <span class="arg">${esc(k)}</span> <span class="val">${esc(String(v))}</span>`;
  const lines = [
    `<span class="accent">$ iggle train</span> \\`,
    arg('--base-model', cfg.base) + ' \\',
    arg('--dataset',    datasetPath) + ' \\',
    arg('--output',     `${safeName}.safetensors`) + ' \\',
    arg('--trigger',    `"${cfg.trigger}"`) + ' \\',
    arg('--steps',      cfg.steps) + ' ' + arg('--rank', cfg.rank) + ' \\',
    arg('--lr',         cfg.lr) + ' ' + arg('--resolution', cfg.res) + ' \\',
    arg('--batch',      cfg.batch) + ' ' + arg('--repeats', cfg.repeats) + ' \\',
    arg('--optimizer',  cfg.opt) + ' ' + arg('--scheduler', cfg.sched) + ' \\',
    arg('--alpha',      cfg.alpha) + ' ' + arg('--save-every', cfg.saveEvery) + ' \\',
    arg('--precision',  cfg.precision) + ' ' + arg('--gradient-checkpointing', cfg.gradCkpt),
    arg('--instance-usd-per-hour', hourlyRate || 'unset'),
  ];
  if (!trainerLaunchOpts.samples) {
    lines[lines.length - 1] += ' \\';
    lines.push(arg('--samples', 'off'));
  }
  if (!trainerLaunchOpts.autoPublish) {
    lines[lines.length - 1] += ' \\';
    lines.push(arg('--auto-import-lora', 'off'));
  }
  // Sample prompts (one per --sample flag) if the user authored any.
  const samples = trainerWizardSamples.filter(s => s.trim());
  if (trainerLaunchOpts.samples && samples.length) {
    lines[lines.length - 1] += ' \\';
    samples.forEach((s, i) => {
      lines.push(arg('--sample', `"${s.replace(/"/g, '\\"')}"`) + (i < samples.length - 1 ? ' \\' : ''));
    });
  }
  const block = $('#launchCmdBlock');
  if (block) block.innerHTML = lines.join('\n');
  const launchBtn = $('#launchTrainingBtn');
  const ready = !!trainerCurateDataset?.health?.ready && trainerCuratePendingSaves === 0;
  if (launchBtn) launchBtn.disabled = !ready;
}

// ── Phase 3: Running monitor ────────────────────────────────────────────
// State for the active running view. Polling kicks off when the view opens
// and stops on leaving or when the job hits a terminal state.
let runJobId = null;            // job id currently being monitored
let trainerRunJob = null;       // last polled trainer job object
let runTab = 'monitor';
let runPollTimer = null;
let runSamplesCache = null;     // last samples response, indexed by job id
let runSamplesJobId = null;
const RUN_POLL_MS = 1500;
const RUN_SAMPLES_POLL_MS = 6000;   // samples poll runs at a slower cadence
const RUN_TERMINAL_STATUSES = new Set(['done', 'error', 'cancelled']);
let runLastSamplesPoll = 0;

function trainerRunIsTerminal(job = trainerRunJob) {
  return RUN_TERMINAL_STATUSES.has(job?.status);
}

function trainerPromptSlug(prompt, index = 0) {
  return String(prompt || `prompt ${index + 1}`)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 42) || `prompt_${index + 1}`;
}

function previewTrainerMetrics(current, total, lr) {
  const stepsDone = Math.max(1, Number(current || 1));
  const stepsTotal = Math.max(stepsDone, Number(total || stepsDone || 1));
  const count = Math.max(6, Math.min(40, Math.ceil(stepsDone / Math.max(1, stepsTotal / 60))));
  const baseLr = Number(lr || 0.0002);
  const loss = [];
  const lrs = [];
  for (let i = 0; i < count; i += 1) {
    const step = Math.max(1, Math.round((stepsDone * (i + 1)) / count));
    const t = Math.min(1, step / stepsTotal);
    const curve = 0.88 - 0.42 * Math.log1p(t * 6) / Math.log(7);
    const wobble = Math.sin(i * 1.7) * 0.018 + Math.cos(i * 0.61) * 0.009;
    loss.push([step, Math.max(0.19, curve + wobble)]);
    lrs.push([step, Math.max(baseLr * 0.04, baseLr * (1 - t * 0.82))]);
  }
  return { loss, lr: lrs };
}

function previewTrainerCheckpointSteps(job, includeFuture = false) {
  const total = Math.max(1, Number(job?.total_steps || job?.steps || 3000));
  const current = Math.max(0, Number(job?.current_step || 0));
  const every = Math.max(1, Number(job?.save_every || Math.max(250, Math.min(1000, Math.floor(total / 4) || 250))));
  const limit = includeFuture ? total : current;
  const steps = [];
  for (let step = every; step < total; step += every) {
    if (step <= limit) steps.push(step);
  }
  if ((includeFuture || current >= total) && !steps.includes(total)) steps.push(total);
  return steps;
}

function previewTrainerSampleUrl(job, step, slug) {
  const name = String(job?.output_name || 'preview_lora').replace(/[<>&"]/g, '');
  const label = String(slug || 'sample').replace(/[<>&"]/g, '');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 768 768">
    <defs>
      <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#151018"/>
        <stop offset="55%" stop-color="#221126"/>
        <stop offset="100%" stop-color="#07191d"/>
      </linearGradient>
    </defs>
    <rect width="768" height="768" fill="url(#bg)"/>
    <rect x="54" y="54" width="660" height="660" rx="18" fill="rgba(255,255,255,0.045)" stroke="rgba(255,255,255,0.18)"/>
    <circle cx="384" cy="322" r="132" fill="rgba(255,43,214,0.28)" stroke="rgba(0,229,255,0.46)" stroke-width="4"/>
    <path d="M210 620c45-128 302-128 348 0" fill="rgba(0,229,255,0.16)" stroke="rgba(255,255,255,0.22)" stroke-width="3"/>
    <text x="384" y="610" fill="rgba(255,255,255,0.86)" font-family="ui-monospace,monospace" font-size="34" text-anchor="middle">${name}</text>
    <text x="384" y="656" fill="rgba(255,255,255,0.55)" font-family="ui-monospace,monospace" font-size="22" text-anchor="middle">step ${step} / ${label}</text>
  </svg>`;
  return 'data:image/svg+xml;utf8,' + encodeURIComponent(svg);
}

function previewTrainerSamplesForJob(job) {
  if (!job || job.generate_samples === false) {
    return { job_id: job?.id || '', output_dir: job?.output_dir || '', prompts: [], checkpoints: [] };
  }
  const rawPrompts = (Array.isArray(job.sample_prompts) && job.sample_prompts.length
    ? job.sample_prompts
    : trainerWizardSamples).slice(0, 6);
  const prompts = rawPrompts.map((p, i) => trainerPromptSlug(p, i));
  const checkpoints = previewTrainerCheckpointSteps(job).map(step => ({
    step,
    samples: prompts.map(slug => ({
      name: `${String(job.output_name || 'preview')}_${step}_${slug}.png`,
      url: previewTrainerSampleUrl(job, step, slug),
      prompt_slug: slug,
    })),
  }));
  return { job_id: job.id, output_dir: job.output_dir || '', prompts, checkpoints };
}

function previewAdvanceTrainerJob(job, progressStep = 5) {
  if (!job || !['queued', 'running'].includes(job.status)) return job;
  const total = Math.max(1, Number(job.total_steps || job.steps || 3000));
  const previousStep = Math.max(0, Number(job.current_step || 0));
  const previousProgress = Math.max(0, Math.min(100, Number(job.progress || (previousStep / total) * 100 || 0)));
  const progress = Math.min(100, previousProgress + progressStep);
  const current = Math.min(total, Math.round((progress / 100) * total));
  const metrics = previewTrainerMetrics(current, total, job.learning_rate);
  const latestLoss = metrics.loss[metrics.loss.length - 1]?.[1];
  const latestLr = metrics.lr[metrics.lr.length - 1]?.[1];
  const every = Math.max(1, Number(job.save_every || Math.max(250, Math.min(1000, Math.floor(total / 4) || 250))));
  const crossedCheckpoint = Math.floor(previousStep / every) < Math.floor(current / every);
  const logLine = progress >= 100
    ? 'Preview LoRA imported'
    : `step ${current}/${total} loss=${latestLoss?.toFixed(4) || '0.0000'} lr=${latestLr?.toExponential(2) || job.learning_rate}`;
  const logTail = [
    ...(job.log_tail || []),
    crossedCheckpoint ? `Saved checkpoint at step ${Math.floor(current / every) * every}` : null,
    logLine,
  ].filter(Boolean).slice(-80);
  const finished = progress >= 100;
  const loraSteps = previewTrainerCheckpointSteps({ ...job, current_step: total, total_steps: total }, true);
  return {
    ...job,
    status: finished ? 'done' : 'running',
    phase: finished ? 'Done' : 'Training',
    phase_progress: progress,
    current_step: current,
    total_steps: total,
    progress,
    metrics,
    lora_filename: finished ? `${job.output_name}_step${String(total).padStart(4, '0')}.safetensors` : job.lora_filename,
    lora_filenames: finished ? loraSteps.map(step => `${job.output_name}_step${String(step).padStart(4, '0')}.safetensors`) : job.lora_filenames,
    log_tail: logTail,
  };
}

function openTrainerRunningView(jobId) {
  if (!jobId) return;
  runJobId = jobId;
  trainerRunJob = null;
  runTab = 'monitor';
  runSamplesCache = null;
  runSamplesJobId = null;
  runLastSamplesPoll = 0;
  // Switch view + start polling. switchView already wires the section
  // toggle; we trigger our own once-then-interval refresh.
  switchView('trainer-running');
  startTrainerRunningPoll();
  // Bind tab buttons + actions once per view open. They live in a
  // detached <section> so the first time we touch the view we wire
  // everything; idempotent re-wiring is fine (event listeners get
  // re-added but the handlers are pure).
  bindTrainerRunningOnce();
  refreshTrainerRunning();   // immediate
}

let trainerRunningBound = false;
function bindTrainerRunningOnce() {
  if (trainerRunningBound) return;
  trainerRunningBound = true;
  $('#runBack')?.addEventListener('click', closeTrainerRunningView);
  $('#runStopBtn')?.addEventListener('click', () => cancelRunningJob('stop'));
  $('#runAbortBtn')?.addEventListener('click', () => cancelRunningJob('abort'));
  $$('#runTabs .px-tab').forEach(tab => {
    tab.addEventListener('click', () => setTrainerRunningTab(tab.dataset.runTab));
  });
  $('#runSampleReload')?.addEventListener('click', () => {
    runLastSamplesPoll = 0;       // force the next poll to refetch
    refreshTrainerRunning();
  });
}

function closeTrainerRunningView() {
  stopTrainerRunningPoll();
  runJobId = null;
  trainerRunJob = null;
  switchView('trainers');
}

function startTrainerRunningPoll() {
  stopTrainerRunningPoll();
  runPollTimer = setInterval(refreshTrainerRunning, RUN_POLL_MS);
}

function stopTrainerRunningPoll() {
  if (runPollTimer) {
    clearInterval(runPollTimer);
    runPollTimer = null;
  }
}

async function refreshTrainerRunning() {
  if (!runJobId) return;
  if (state.preview) {
    const existing = (state.trainJobs || []).find(j => j.id === runJobId);
    if (!existing) {
      stopTrainerRunningPoll();
      return;
    }
    const next = previewAdvanceTrainerJob(existing, 4.5);
    state.trainJobs = (state.trainJobs || []).map(j => j.id === runJobId ? next : j);
    trainerRunJob = next;
    if (runTab === 'samples') {
      runSamplesCache = previewTrainerSamplesForJob(next);
      runSamplesJobId = runJobId;
    }
    renderTrainerRunning();
    if (trainerRunIsTerminal(next)) stopTrainerRunningPoll();
    return;
  }
  try {
    trainerRunJob = await api.trainJob(runJobId);
  } catch (e) {
    // 404 → job dismissed; stop polling and bail back to trainers.
    console.warn('trainer run poll failed', e);
    stopTrainerRunningPoll();
    return;
  }
  renderTrainerRunning();
  // Stop polling once we reach a terminal state. The render call above
  // still shows the final state; subsequent renders need a manual reload.
  if (trainerRunIsTerminal(trainerRunJob)) {
    stopTrainerRunningPoll();
  }
  // Throttle samples polling to RUN_SAMPLES_POLL_MS — image listings are
  // cheap but enumerating an output dir on every 1.5s poll is wasteful.
  const now = Date.now();
  if (runTab === 'samples' && now - runLastSamplesPoll > RUN_SAMPLES_POLL_MS) {
    runLastSamplesPoll = now;
    pollTrainerSamples();
  }
}

async function pollTrainerSamples() {
  if (!runJobId) return;
  if (state.preview) {
    const job = trainerRunJob || (state.trainJobs || []).find(j => j.id === runJobId);
    runSamplesCache = previewTrainerSamplesForJob(job);
    runSamplesJobId = runJobId;
    if (runTab === 'samples') renderTrainerRunningSamples();
    return;
  }
  try {
    const data = await api.trainJobSamples(runJobId);
    runSamplesCache = data;
    runSamplesJobId = runJobId;
    if (runTab === 'samples') renderTrainerRunningSamples();
  } catch {/* silent — common 401/404 during job lifecycle */}
}

function setTrainerRunningTab(name) {
  runTab = name;
  $$('#runTabs .px-tab').forEach(b => b.classList.toggle('active', b.dataset.runTab === name));
  $$('[data-run-pane]').forEach(p => { p.hidden = p.dataset.runPane !== name; });
  if (name === 'samples') {
    // Force a refresh on tab open if we don't have cached samples yet.
    if (!runSamplesCache || runSamplesJobId !== runJobId) pollTrainerSamples();
    else renderTrainerRunningSamples();
  }
  if (name === 'test') {
    if (!runSamplesCache || runSamplesJobId !== runJobId) pollTrainerSamples();
    renderTrainerRunningTest();
  }
}

function renderTrainerRunning() {
  if (!trainerRunJob) return;
  const j = trainerRunJob;
  // ── Header strip ──
  const setEl = (sel, text) => { const el = $(sel); if (el) el.textContent = text; };
  setEl('#runIdLabel', `Run · ${j.output_name || j.id || '—'}`);
  setEl('#runName',    j.output_name || j.id || 'Run');
  const trigger = j.trigger_phrase ? `"${j.trigger_phrase}"` : 'no trigger';
  const startedAgo = j.started_at ? `${Math.max(0, Math.round((Date.now() / 1000 - j.started_at) / 60))} min ago` : '';
  setEl('#runSub', `started ${startedAgo} · ${trigger}`);
  // Status chip + dot.
  const statusChip = $('#runStatusChip');
  const dot = $('#runStatusDot');
  const status = (j.status || 'unknown').toUpperCase();
  if (statusChip) statusChip.textContent = status;
  if (dot) {
    dot.className = 'px-dot ' + (
      j.status === 'running' ? 'pink blink' :
      j.status === 'done'    ? 'ok'         :
      j.status === 'error'   ? 'err'        :
      j.status === 'cancelled' ? 'warn'     : 'pink'
    );
  }
  setEl('#runRecipeChip', `BASE · ${String(j.base_model || 'qwen').replace(/^.*\//, '').toUpperCase()}`);
  setEl('#runBaseChip',   j.phase ? j.phase.toUpperCase() : 'STARTING');

  const terminal = trainerRunIsTerminal(j);
  const active = ['queued', 'running'].includes(j.status);
  const cancelBtn = $('#runStopBtn');
  const dismissBtn = $('#runAbortBtn');
  if (cancelBtn) {
    cancelBtn.hidden = terminal;
    cancelBtn.disabled = !active;
    cancelBtn.textContent = 'Cancel run';
    cancelBtn.title = active ? 'Cancel this training process' : 'Run is not active';
  }
  if (dismissBtn) {
    dismissBtn.hidden = !terminal;
    dismissBtn.disabled = !terminal;
    dismissBtn.textContent = 'Dismiss';
    dismissBtn.title = 'Remove this job from the list';
  }

  // Progress bar.
  const bar = $('#runProgressBar');
  const progressValue = Math.max(0, Math.min(100, Number(j.progress || 0)));
  if (bar) bar.style.width = `${progressValue}%`;
  $('#runProgressTrack')?.setAttribute('aria-valuenow', String(Math.round(progressValue)));

  // 6-cell stat strip.
  const lossSeries = j.metrics?.loss || [];
  const lrSeries   = j.metrics?.lr   || [];
  const lastLoss   = lossSeries.length ? lossSeries[lossSeries.length - 1][1] : null;
  const firstLoss  = lossSeries.length ? lossSeries[0][1] : null;
  const lastLr     = lrSeries.length ? lrSeries[lrSeries.length - 1][1] : null;
  const stepDisplay = trainerStepDisplay(j);
  setRunStat('step', stepDisplay.step || '—', stepDisplay.total ? `/ ${stepDisplay.total.toLocaleString()}` : '/ —');
  setRunStat('loss', lastLoss != null ? lastLoss.toFixed(3) : '—',
             (lastLoss != null && firstLoss != null) ? `↓ ${(firstLoss - lastLoss).toFixed(2)}` : 'trend');
  setRunStat('lr',   lastLr != null ? lastLr.toExponential(2) : '—', j.scheduler || 'scheduler');
  setRunStat('phase', j.phase || '—', `${Math.round(Number(j.phase_progress || 0))}%`);

  const eta = trainerEtaEstimate(j, stepDisplay);
  setRunStat('eta',  eta.minutes != null ? formatRunMinutes(eta.minutes) : 'Calibrating', eta.label);
  const elapsedMin = j.started_at ? Math.max(0, (Date.now() / 1000 - j.started_at) / 60) : 0;
  const gpuRate = trainerGpuRate(j);
  const cost      = gpuRate.usdPerHour ? (elapsedMin / 60) * gpuRate.usdPerHour : 0;
  const totalCost = (gpuRate.usdPerHour && eta.minutes != null) ? ((elapsedMin + eta.minutes) / 60) * gpuRate.usdPerHour : 0;
  setRunStat('cost', gpuRate.usdPerHour ? `$${cost.toFixed(2)}` : '—', totalCost ? `of $${totalCost.toFixed(2)} · ${gpuRate.label}` : gpuRate.label);

  // ── Tab body — render only the active tab to keep DOM cheap.
  if (runTab === 'monitor')  renderTrainerRunningMonitor();
  if (runTab === 'terminal') renderTrainerRunningTerminal();
  if (runTab === 'config')   renderTrainerRunningConfig();
  if (runTab === 'samples')  renderTrainerRunningSamples();
  if (runTab === 'test')     renderTrainerRunningTest();
}

function setRunStat(key, value, sub) {
  const num = document.querySelector(`[data-run-stat="${key}"]`);
  const subEl = document.querySelector(`[data-run-stat-sub="${key}"]`);
  if (num)  num.textContent  = String(value);
  if (subEl) subEl.textContent = String(sub || '');
}

function trainerStepDisplay(job) {
  const configured = Number(job?.steps || 0);
  const rawTotal = Number(job?.total_steps || 0);
  const plausibleTotal = rawTotal && (!configured || rawTotal <= Math.max(configured + 1000, configured * 1.2));
  const total = plausibleTotal ? rawTotal : configured;
  const rawStep = Number(job?.current_step || 0);
  const phase = String(job?.phase || '').toLowerCase();
  const isTraining = phase === 'training' || phase === 'done' || job?.status === 'done';
  const step = isTraining ? (total ? Math.min(rawStep, total) : rawStep) : 0;
  return { step, total };
}

function trainerEtaEstimate(job, stepDisplay = trainerStepDisplay(job)) {
  const step = Number(stepDisplay.step || 0);
  const total = Number(stepDisplay.total || job?.steps || 0);
  const remaining = total && step ? Math.max(0, total - step) : null;
  if (remaining == null) return { minutes: null, label: 'waiting for real steps' };

  const reportedRate = Number(job?.steps_per_min || 0);
  if (reportedRate > 0) {
    return { minutes: Math.max(0, Math.round(remaining / reportedRate)), label: `${reportedRate.toFixed(1)} steps/min observed` };
  }

  const elapsedMin = job?.started_at ? Math.max(0.1, (Date.now() / 1000 - job.started_at) / 60) : 0;
  if (step >= 10 && elapsedMin > 0) {
    const elapsedRate = step / elapsedMin;
    return { minutes: Math.max(0, Math.round(remaining / elapsedRate)), label: `${elapsedRate.toFixed(1)} steps/min from run` };
  }

  return { minutes: null, label: 'calibrating from step logs' };
}

function formatRunMinutes(minutes) {
  const m = Math.max(0, Math.round(Number(minutes) || 0));
  if (m < 90) return `${m}m`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem ? `${h}h ${rem}m` : `${h}h`;
}

function trainerGpuRate(job) {
  const explicit = Number(job?.gpu_usd_per_hour || 0);
  if (explicit > 0) {
    const source = job?.gpu_rate_source === 'user' ? 'your rate' : (job?.gpu_rate_source === 'env' ? 'exact' : 'est.');
    return { usdPerHour: explicit, label: `${source} $${explicit.toFixed(2)}/hr` };
  }
  const name = String(job?.gpu_name || '').toLowerCase();
  const fallbackRates = [
    [/b200/, 5.98],
    [/h200/, 4.31],
    [/h100.*nvl/, 3.99],
    [/h100/, 2.99],
    [/a100/, 1.64],
    [/rtx pro 6000/, 1.22],
    [/rtx 6000 ada/, 0.89],
    [/l40s|l40/, 1.03],
    [/a40/, 0.79],
    [/5090/, 0.89],
    [/4090/, 0.69],
    [/3090/, 0.43],
    [/\bl4\b/, 0.43],
  ];
  const match = fallbackRates.find(([re]) => re.test(name));
  if (!match) return { usdPerHour: 0, label: 'rate unknown' };
  return { usdPerHour: match[1], label: `est. $${match[1].toFixed(2)}/hr` };
}

// SVG sparkline renderer — no external deps, scales to the viewBox so
// the CSS height controls the on-screen size.
function renderSparklinePath(svg, data, options = {}) {
  if (!svg) return;
  const w = 400;
  const h = svg.id === 'runLossSpark' ? 180 : 90;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  if (!data || data.length < 2) {
    svg.innerHTML = '';
    return;
  }
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const padX = 4;
  const padY = 8;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;
  const xs = data.map((_, i) => padX + (i / (data.length - 1)) * innerW);
  const ys = data.map(v => padY + innerH - ((v - min) / range) * innerH);
  const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ');
  const fillPath = `${path} L${xs[xs.length - 1].toFixed(1)},${(padY + innerH).toFixed(1)} L${xs[0].toFixed(1)},${(padY + innerH).toFixed(1)} Z`;
  const grid = options.grid !== false ? `
    <line x1="0" y1="${(padY).toFixed(1)}"          x2="${w}" y2="${(padY).toFixed(1)}"          stroke="var(--px-border-soft)" stroke-width="1"/>
    <line x1="0" y1="${(padY + innerH / 2).toFixed(1)}" x2="${w}" y2="${(padY + innerH / 2).toFixed(1)}" stroke="var(--px-border-soft)" stroke-width="1"/>
    <line x1="0" y1="${(padY + innerH).toFixed(1)}" x2="${w}" y2="${(padY + innerH).toFixed(1)}" stroke="var(--px-border-soft)" stroke-width="1"/>
  ` : '';
  svg.innerHTML = `
    ${grid}
    <path d="${fillPath}" fill="${options.fill || 'rgba(255,43,214,0.12)'}" stroke="none"/>
    <path d="${path}"     fill="none" stroke="${options.color || 'var(--px-pink)'}" stroke-width="${options.width || 1.6}" shape-rendering="geometricPrecision"/>
  `;
}

function renderTrainerRunningMonitor() {
  const lossSeries = trainerRunJob?.metrics?.loss || [];
  const lrSeries   = trainerRunJob?.metrics?.lr   || [];
  const lossValues = lossSeries.map(([, v]) => v);
  const lrValues   = lrSeries.map(([, v]) => v);
  renderSparklinePath($('#runLossSpark'), lossValues, { color: 'var(--px-pink)', fill: 'rgba(255,43,214,0.12)' });
  renderSparklinePath($('#runLrSpark'),   lrValues,   { color: 'var(--px-cyan)', fill: 'rgba(0,229,255,0.08)' });
  const schedName = $('#runSchedName');
  if (schedName) schedName.textContent = trainerRunJob?.scheduler || 'scheduler';

  const window = $('#runLossWindow');
  if (window) window.textContent = `last ${lossValues.length} steps`;

  const gpuTitle = $('#runGpuTitle');
  if (gpuTitle) {
    const vram = trainerRunJob?.gpu_vram_gb ? ` · ${trainerRunJob.gpu_vram_gb}GB` : '';
    gpuTitle.textContent = trainerRunJob?.gpu_name ? `GPU · ${trainerRunJob.gpu_name}${vram}` : 'GPU · detecting host';
  }
  const gpuBars = $('#runGpuBars');
  if (gpuBars) {
    const progress = Math.max(0, Math.min(100, Number(trainerRunJob?.progress || 0)));
    const { step, total } = trainerStepDisplay(trainerRunJob);
    const eta = trainerEtaEstimate(trainerRunJob, { step, total });
    const speed = Number(trainerRunJob?.steps_per_min || 0);
    const gpuRate = trainerGpuRate(trainerRunJob);
    const elapsed = trainerRunJob?.started_at ? Math.max(0, Math.round((Date.now() / 1000 - trainerRunJob.started_at) / 60)) : null;
    const sampleState = trainerRunJob?.generate_samples === false ? 'off' : 'on';
    gpuBars.innerHTML = `
      <div class="run-gpu-row"><span>Overall progress</span><b>${progress.toFixed(progress % 1 ? 1 : 0)}%</b></div>
      <div class="run-gpu-meter"><i style="width:${progress}%"></i></div>
      <div class="run-gpu-row"><span>Training steps</span><b>${step.toLocaleString()}${total ? ` / ${total.toLocaleString()}` : ''}</b></div>
      <div class="run-gpu-row"><span>Step speed</span><b>${speed > 0 ? `${speed.toFixed(1)} / min` : eta.label}</b></div>
      <div class="run-gpu-row"><span>ETA basis</span><b>${eta.minutes != null ? formatRunMinutes(eta.minutes) : 'calibrating'}</b></div>
      <div class="run-gpu-row"><span>Instance rate</span><b>${gpuRate.usdPerHour ? gpuRate.label : 'unknown'}</b></div>
      <div class="run-gpu-row"><span>Elapsed</span><b>${elapsed != null ? `${elapsed}m` : 'starting'}</b></div>
      <div class="run-gpu-row"><span>Samples</span><b>${sampleState}</b></div>
    `;
  }

  // Footer min/max/step bounds.
  if (lossSeries.length >= 2) {
    const minV = Math.min(...lossValues).toFixed(3);
    const maxV = Math.max(...lossValues).toFixed(3);
    const left  = document.querySelector('[data-loss-range="left"]');
    const right = document.querySelector('[data-loss-range="right"]');
    const mid   = document.querySelector('[data-loss-range="middle"]');
    if (left)  left.textContent  = `step ${Number(lossSeries[0][0]).toLocaleString()}`;
    if (right) right.textContent = `step ${Number(lossSeries[lossSeries.length - 1][0]).toLocaleString()}`;
    if (mid)   mid.textContent   = `min ${minV} · max ${maxV}`;
  }

  // Health signals — derived from the data we have.
  const health = [];
  if (lossValues.length >= 6) {
    const recent = lossValues.slice(-6);
    const trend = recent[recent.length - 1] - recent[0];
    health.push(trend <= 0
      ? { dot: 'ok',   text: 'Loss trending down · OK' }
      : { dot: 'warn', text: `Loss flat or rising · Δ ${trend.toFixed(3)} over last ${recent.length} steps` });
  } else {
    health.push({ dot: 'warn', text: `Collecting metrics… ${lossValues.length}/6 samples` });
  }
  const hasNan = lossValues.some(v => !Number.isFinite(v));
  health.push(hasNan
    ? { dot: 'err', text: 'NaN/Inf detected in loss — abort + retry with lower LR' }
    : { dot: 'ok',  text: 'No NaN / Inf gradients' });
  if (trainerRunJob?.rank_warning) {
    health.push({ dot: 'warn', text: trainerRunJob.rank_warning });
  }
  if (trainerRunJob?.status === 'error' && trainerRunJob?.error) {
    health.push({ dot: 'err', text: trainerRunJob.error });
  }
  const list = $('#runHealthList');
  if (list) {
    list.innerHTML = health.map(h => `
      <div class="flag-row"><span class="px-dot ${h.dot}"></span><span>${esc(h.text)}</span></div>
    `).join('');
  }
}

function renderTrainerRunningTerminal() {
  const term = $('#runTerminal');
  if (!term) return;
  const tail = trainerRunJob?.log_tail || [];
  term.textContent = tail.length ? tail.join('\n') : 'Waiting for the trainer to print…';
  // Auto-scroll to bottom.
  term.scrollTop = term.scrollHeight;
}

function renderTrainerRunningConfig() {
  const rows = $('#runConfigRows');
  if (!rows || !trainerRunJob) return;
  const fields = [
    ['Run name',     trainerRunJob.output_name],
    ['Trainer',      trainerRunJob.trainer_id],
    ['Base model',   trainerRunJob.base_model],
    ['Trigger',      trainerRunJob.trigger_phrase],
    ['Dataset',      trainerRunJob.dataset_path],
    ['Steps',        trainerRunJob.steps],
    ['Rank',         trainerRunJob.rank],
    ['Observed rank',trainerRunJob.observed_rank],
    ['Network alpha',trainerRunJob.network_alpha],
    ['Learning rate',trainerRunJob.learning_rate],
    ['Resolution',   trainerRunJob.resolution],
    ['Batch size',   trainerRunJob.batch_size],
    ['Repeats',      trainerRunJob.repeats],
    ['Optimizer',    trainerRunJob.optimizer],
    ['Scheduler',    trainerRunJob.scheduler],
    ['Precision',    trainerRunJob.precision],
    ['Save every',   trainerRunJob.save_every],
    ['Instance $/hr', trainerRunJob.instance_usd_per_hour || trainerRunJob.gpu_usd_per_hour],
    ['Generate samples', trainerRunJob.generate_samples === false ? 'no' : 'yes'],
    ['Auto-import LoRA', trainerRunJob.auto_import_lora === false ? 'no' : 'yes'],
    ['Output dir',   trainerRunJob.output_dir],
  ].filter(([, v]) => v !== undefined && v !== null && v !== '');
  rows.innerHTML = fields.map(([k, v]) => `
    <div class="row"><span class="k">${esc(k)}</span><span class="v">${esc(String(v))}</span></div>
  `).join('');
}

function renderTrainerRunningSamples() {
  const body = $('#runSampleBody');
  if (!body) return;
  const data = runSamplesCache;
  if (!data || !data.checkpoints?.length) {
    body.innerHTML = `<div class="px-mute" style="padding:24px;text-align:center;font-family:var(--px-font-mono);font-size:11.5px">
      No samples yet. They appear after the first save-every checkpoint completes.
    </div>`;
    return;
  }
  const prompts = data.prompts || [];
  const checkpoints = data.checkpoints || [];
  // Build a step × prompt grid. The first column is prompt-slug labels;
  // each subsequent column is a checkpoint.
  const cols = `200px ${checkpoints.map(() => '140px').join(' ')}`;
  const headerRow = `
    <div></div>
    ${checkpoints.map(c => `<div class="col-head">step ${Number(c.step).toLocaleString()}</div>`).join('')}
  `;
  const rows = prompts.map(slug => {
    const cells = checkpoints.map(c => {
      const hit = (c.samples || []).find(s => s.prompt_slug === slug);
      if (!hit) return `<div class="sample-cell empty">—</div>`;
      return `<div class="sample-cell"><img src="${esc(hit.url)}" alt="${esc(hit.name)}" loading="lazy"></div>`;
    }).join('');
    return `<div class="row-head" title="${esc(slug)}">${esc(slug)}</div>${cells}`;
  }).join('');
  body.innerHTML = `
    <div class="sample-grid" style="grid-template-columns:${cols};min-width:${200 + checkpoints.length * 148}px">
      ${headerRow}
      ${rows}
    </div>
  `;
}

function renderTrainerRunningTest() {
  const prompt = $('#runTestPrompt');
  if (prompt && !prompt.value.trim()) {
    const firstSample = Array.isArray(trainerRunJob?.sample_prompts) ? trainerRunJob.sample_prompts.find(Boolean) : '';
    prompt.value = firstSample || `${trainerRunJob?.trigger_phrase || 'subject'}, portrait, natural light, detailed face`;
  }

  const checkpointSteps = runSamplesCache?.checkpoints?.length
    ? runSamplesCache.checkpoints.map(c => Number(c.step))
    : (state.preview ? previewTrainerCheckpointSteps(trainerRunJob) : []);
  const select = $('#runTestCheckpoint');
  if (select) {
    if (checkpointSteps.length) {
      select.innerHTML = checkpointSteps
        .slice()
        .sort((a, b) => a - b)
        .map(step => `<option value="${step}">step ${Number(step).toLocaleString()}</option>`)
        .join('');
      select.value = String(checkpointSteps[checkpointSteps.length - 1]);
    } else {
      select.innerHTML = '<option value="">waiting for checkpoint</option>';
    }
  }

  const list = $('#runCheckpointList');
  if (list) {
    if (!checkpointSteps.length) {
      list.innerHTML = `<div class="flag-row"><span class="px-dot warn"></span><span>Waiting for first checkpoint...</span></div>`;
    } else {
      list.innerHTML = checkpointSteps.slice().sort((a, b) => b - a).map((step, idx) => `
        <div class="flag-row"><span class="px-dot ${idx === 0 ? 'ok' : 'pink'}"></span><span>step ${Number(step).toLocaleString()}</span></div>
      `).join('');
    }
  }

  const btn = $('#runTestBtn');
  if (btn) {
    btn.disabled = true;
    btn.title = 'Checkpoint inference is not wired to the trainer wrapper yet';
  }
  const hint = $('#runTestHint');
  if (hint) {
    hint.textContent = checkpointSteps.length
      ? 'Latest checkpoint is ready to test once checkpoint inference is connected.'
      : 'The first saved checkpoint will appear here before live testing is available.';
  }
}

async function cancelRunningJob(mode) {
  if (!runJobId) return;
  const terminal = trainerRunIsTerminal(trainerRunJob);
  const verb = terminal ? 'Dismiss' : 'Cancel';
  if (!terminal && !window.confirm('Cancel this training run?')) return;
  if (state.preview) {
    if (terminal) {
      state.trainJobs = (state.trainJobs || []).filter(j => j.id !== runJobId);
      closeTrainerRunningView();
      renderTrainerJobs();
      return;
    }
    const cancelled = {
      ...(trainerRunJob || (state.trainJobs || []).find(j => j.id === runJobId)),
      status: 'cancelled',
      phase: 'Cancelled',
      log_tail: [...((trainerRunJob?.log_tail) || []), 'Preview run cancelled by user'].slice(-80),
    };
    state.trainJobs = (state.trainJobs || []).map(j => j.id === runJobId ? cancelled : j);
    trainerRunJob = cancelled;
    stopTrainerRunningPoll();
    renderTrainerRunning();
    renderTrainerJobs();
    toast('Run cancelled', 'success');
    return;
  }
  try {
    await api.cancelTrainJob(runJobId);
    toast(terminal ? 'Run dismissed' : 'Cancelling run...', 'success');
    if (terminal) {
      closeTrainerRunningView();
      await refreshTrainerJobs();
    } else {
      // Force an immediate refresh so the UI flips status.
      refreshTrainerRunning();
    }
  } catch (e) {
    toast(`${verb} failed: ${e.message || 'unknown'}`, 'error');
  }
}

async function saveTrainerInspectorCaption() {
  const sel = trainerCurateDataset?.pairs.find(p => p.image === trainerCurateSelected);
  if (!sel) return;
  const text = $('#trainerInspectorCaption')?.value || '';
  const btn = $('#trainerInspectorSave');
  if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }
  if (!state.preview) {
    trainerCuratePendingSaves += 1;
    updateTrainerWizardNextGate();
  }
  try {
    if (!state.preview) {
      await api.saveTrainerCaption({
        dataset_path: trainerCurateDataset.dataset_path,
        image_rel:    sel.image,
        caption:      text,
      });
    } else {
      await delay(120);
    }
    // Patch local state — keeps the grid + filter counts accurate without
    // a full refetch. Flags need re-deriving because empty caption was a
    // flag; the backend recomputes on next list call but we mirror its
    // logic here so the chip strip stays in sync.
    sel.caption = text.trim();
    sel.caption_path = trainerCaptionPathForImage(sel.image);
    sel.flags = sel.flags.filter(f => f.label !== 'NO CAP' && f.label !== 'EMPTY CAP' && f.label !== 'SHORT CAP');
    if (!sel.caption) {
      sel.flags.unshift({ label: 'EMPTY CAP', tone: 'warn', detail: 'caption was just cleared' });
    } else if (sel.caption.length < 12) {
      sel.flags.unshift({ label: 'SHORT CAP', tone: 'warn', detail: `caption is only ${sel.caption.length} chars` });
    }
    renderTrainerCurateAll();
    toast('Caption saved', 'success');
  } catch (e) {
    toast(`Save failed: ${e.message || 'unknown'}`, 'error');
  } finally {
    if (!state.preview) {
      trainerCuratePendingSaves = Math.max(0, trainerCuratePendingSaves - 1);
      renderTrainerCurateAll();
    }
    if (btn) { btn.disabled = false; btn.textContent = 'Save caption'; }
  }
}

function trainerPayload() {
  // The pixel wizard is the single UI source of truth for trainer launches.
  const samples = trainerWizardSamples.filter(s => s.trim());
  const name = ($('#trainerRunName')?.value || '').trim();
  const safeName = name.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'kerry_qwen_lora';
  return {
    trainer_id: trainerWizardCfg.trainerId || 'qwen-character-lora',
    dataset_path: currentTrainerDatasetPath(),
    output_name: safeName,
    trigger_phrase: trainerWizardCfg.trigger || 'A woman named Kerry',
    base_model: trainerWizardCfg.base || 'Qwen/Qwen-Image-2512',
    steps: Number(trainerWizardCfg.steps || 3000),
    rank: Number(trainerWizardCfg.rank || 64),
    learning_rate: Number(trainerWizardCfg.lr || 0.0002),
    resolution: Number(trainerWizardCfg.res || 1024),
    batch_size: Number(trainerWizardCfg.batch || 1),
    repeats: Number(trainerWizardCfg.repeats || 1),
    hf_token: currentHFToken('#trainerWizardHFToken') || null,
    optimizer:             trainerWizardCfg.opt,
    scheduler:             trainerWizardCfg.sched,
    network_alpha:         Number(trainerWizardCfg.alpha) || null,
    save_every:            Number(trainerWizardCfg.saveEvery) || null,
    precision:             trainerWizardCfg.precision,
    gradient_checkpointing: trainerWizardCfg.gradCkpt === 'on',
    instance_usd_per_hour: Math.max(0, Number(trainerWizardCfg.hourlyRate || 0)) || null,
    sample_prompts:        samples.length ? samples : null,
    generate_samples:      !!trainerLaunchOpts.samples,
    auto_import_lora:      !!trainerLaunchOpts.autoPublish,
  };
}

async function downloadTrainerDataset() {
  const repo = $('#trainerWizardHFRepo')?.value.trim() || '';
  const target = $('#trainerWizardHFTarget')?.value.trim() || repo.split('/').pop() || 'dataset';
  const repoType = $('#trainerWizardHFType')?.value || 'dataset';
  const btn = $('#trainerWizardHFPull');
  if (!repo) return toast('Enter a Hugging Face dataset repo first.', 'error');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span><span>Pulling…</span>';
  }
  try {
    let res;
    if (state.preview) {
      await delay(450);
      res = {
        dataset_path: `datasets/${target}`,
        scan: { valid: true, image_count: 80, caption_count: 80, pair_count: 80 },
      };
    } else {
      res = await api.downloadTrainerDataset({
        hf_repo: repo,
        repo_type: repoType,
        target_name: target,
        hf_token: currentHFToken('#trainerWizardHFToken') || null,
      });
    }
    setTrainerDatasetPath(res.dataset_path || `datasets/${target}`);
    setTrainerValidation(res.scan || null, res.dataset_path || `datasets/${target}`);
    toast(state.trainValidation?.valid ? 'Dataset pulled and ready' : 'Dataset pulled; check captions', state.trainValidation?.valid ? 'success' : 'info');
  } catch (e) {
    clearTrainerDatasetState();
    toast(e.message || 'Dataset pull failed', 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '↓ Pull dataset';
    }
  }
}

function renderTrainers() {
  const trainer = activeTrainer();
  const count = (state.trainJobs || []).filter(j => ['queued', 'running'].includes(j.status)).length;
  const nav = $('#trainerNavCount');
  if (nav) nav.textContent = count || '';
  const configured = !!trainer?.configured || state.preview;
  const sub = $('#trainerConfigState');
  const tag = $('#trainerReadyTag');
  if (sub) {
    sub.textContent = configured
      ? 'Trainer command configured'
      : `Set ${trainer?.command_env || 'IGGLEPIXEL_QWEN_LORA_TRAIN_CMD'} on the pod`;
  }
  if (tag) {
    tag.textContent = configured ? 'Ready' : 'Needs command';
    tag.classList.toggle('solid', configured);
  }
  renderTrainerDatasetSummary(state.trainValidation);
  renderTrainerModelSupport(trainer);
  renderTrainerJobs();
}

function renderTrainerModelSupport(trainer) {
  const root = $('#trainerFamilyGrid');
  if (!root) return;
  const families = trainer?.model_families || [];
  if (!families.length) {
    root.innerHTML = `<div class="empty trainer-empty">No model-family roadmap reported.</div>`;
    return;
  }
  const toneFor = (status) => status === 'live' ? 'ok' : (status === 'next' ? 'pink' : '');
  root.innerHTML = families.map(f => `
    <div class="trainer-family ${esc(f.status || 'planned')}">
      <div class="trainer-family-head">
        <strong>${esc(f.label || f.id)}</strong>
        <span class="px-chip ${toneFor(f.status)}">${esc(f.status || 'planned')}</span>
      </div>
      <p>${esc(f.description || '')}</p>
    </div>
  `).join('');
}

function renderTrainerDatasetSummary(summary) {
  const root = $('#trainerDatasetSummary');
  if (!root) return;
  if (!summary) {
    root.innerHTML = `<div class="empty trainer-empty">No validation yet.</div>`;
    return;
  }
  const metric = (label, value) => `
    <div class="trainer-metric">
      <div class="label">${esc(label)}</div>
      <div class="value">${esc(value)}</div>
    </div>`;
  const warnings = [];
  if (summary.error) warnings.push(summary.error);
  if ((summary.missing_captions || []).length) warnings.push(`Missing captions: ${summary.missing_captions.slice(0, 4).join(', ')}`);
  if ((summary.orphan_captions || []).length) warnings.push(`Orphan captions: ${summary.orphan_captions.slice(0, 4).join(', ')}`);
  if ((summary.empty_captions || []).length) warnings.push(`Empty captions: ${summary.empty_captions.slice(0, 4).join(', ')}`);
  const imageValue = summary.total_image_count && summary.total_image_count !== summary.image_count
    ? `${summary.image_count} / ${summary.total_image_count}`
    : (summary.image_count ?? 0);
  root.innerHTML = `
    <div class="trainer-summary-grid">
      ${metric('Images', imageValue)}
      ${metric('Captions', summary.caption_count ?? 0)}
      ${metric('Pairs', summary.pair_count ?? 0)}
      ${metric('Excluded', summary.excluded_count ?? 0)}
    </div>
    ${summary.valid ? `<div class="trainer-ok">Dataset is ready.</div>` : `<div class="trainer-warnings">${esc(warnings.join('\n') || 'Dataset is not ready.')}</div>`}
  `;
}

async function validateTrainerDataset() {
  const payload = trainerPayload();
  const btn = $('#trainerWizardFolderValidate');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span><span>Validating…</span>';
  }
  try {
    let summary;
    if (state.preview) {
      await delay(300);
      summary = { valid: true, image_count: 80, caption_count: 80, pair_count: 80 };
    } else {
      summary = await api.validateTrainerDataset({ dataset_path: payload.dataset_path });
    }
    setTrainerValidation(summary, payload.dataset_path);
    toast(summary.valid ? 'Dataset ready' : 'Dataset needs attention', summary.valid ? 'success' : 'error');
  } catch (e) {
    clearTrainerDatasetState();
    toast(e.message || 'Dataset validation failed', 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '✓ Validate';
    }
  }
}

async function startTrainerJob() {
  if (trainerCuratePendingSaves > 0) {
    toast('Still saving curation changes. Try again in a moment.', 'error');
    return false;
  }
  const payload = trainerPayload();
  const btn = null;
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span><span>Starting…</span>';
  }
  try {
    if (state.preview) {
      const startStep = Math.max(1, Math.round(Number(payload.steps || 3000) * 0.04));
      const job = {
        id: 'preview-train-' + Date.now(),
        status: 'running',
        phase: 'Training',
        phase_progress: 4,
        trainer_id: payload.trainer_id,
        output_name: payload.output_name,
        trigger_phrase: payload.trigger_phrase,
        dataset_path: payload.dataset_path,
        output_dir: `outputs/${payload.output_name}`,
        base_model: payload.base_model,
        steps: payload.steps,
        rank: payload.rank,
        learning_rate: payload.learning_rate,
        resolution: payload.resolution,
        save_every: payload.save_every || Math.max(250, Math.min(1000, Math.floor(payload.steps / 4) || 250)),
        optimizer: payload.optimizer,
        scheduler: payload.scheduler,
        network_alpha: payload.network_alpha,
        precision: payload.precision,
        instance_usd_per_hour: payload.instance_usd_per_hour,
        gpu_usd_per_hour: payload.instance_usd_per_hour,
        gpu_rate_source: payload.instance_usd_per_hour ? 'user' : '',
        gpu_rate_label: payload.instance_usd_per_hour ? 'User-entered all-in instance rate' : '',
        batch_size: payload.batch_size,
        repeats: payload.repeats,
        generate_samples: payload.generate_samples,
        sample_prompts: payload.sample_prompts,
        auto_import_lora: payload.auto_import_lora,
        current_step: startStep,
        total_steps: payload.steps,
        progress: 4,
        metrics: previewTrainerMetrics(startStep, payload.steps, payload.learning_rate),
        imported_to_library: payload.auto_import_lora,
        log_tail: [
          'Preview training job started',
          `Requested settings: rank ${payload.rank}, steps ${payload.steps}, lr ${payload.learning_rate}, resolution ${payload.resolution}`,
          `Advanced settings: optimizer ${payload.optimizer}, scheduler ${payload.scheduler}, alpha ${payload.network_alpha}, precision ${payload.precision}, samples ${payload.generate_samples ? 'on' : 'off'}, auto-import ${payload.auto_import_lora ? 'on' : 'off'}`,
        ],
        created_at: Date.now() / 1000,
        started_at: Date.now() / 1000,
      };
      state.trainJobs = [job, ...state.trainJobs];
      renderTrainerJobs();
      pokeTrainerPoll();
      toast('Training queued', 'success');
      return job;
    }
    const res = await api.startTrainJob(payload);
    state.trainJobs = [res.job, ...state.trainJobs.filter(j => j.id !== res.job.id)];
    renderTrainerJobs();
    pokeTrainerPoll();
    toast('Training queued', 'success');
    return res.job;
  } catch (e) {
    toast(e.message || 'Training failed to start', 'error');
    return false;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<svg><use href="#i-bolt"/></svg>Start training';
    }
  }
}

function renderTrainerJobs() {
  const root = $('#trainerJobList');
  if (!root) return;
  const jobs = state.trainJobs || [];
  if (!jobs.length) {
    root.innerHTML = `<div class="empty trainer-empty">No training jobs yet.</div>`;
    return;
  }
  root.innerHTML = jobs.map(j => {
    const progress = Math.max(0, Math.min(100, Number(j.progress || 0)));
    const tail = (j.log_tail || []).slice(-28).join('\n');
    const phase = j.phase || j.status || 'unknown';
    const stepText = j.current_step && j.total_steps
      ? `${j.current_step}/${j.total_steps} steps`
      : (j.steps ? `${j.steps} steps` : '');
    const modelLabel = j.base_model ? String(j.base_model).replace('Qwen/', '') : '';
    const imported = Array.isArray(j.lora_filenames) && j.lora_filenames.length
      ? j.lora_filenames
      : (j.lora_filename ? [j.lora_filename] : []);
    const importedText = imported.length
      ? imported.length === 1
        ? `${j.imported_to_library === false ? 'Output ready' : 'Imported'} ${imported[0]}`
        : `${j.imported_to_library === false ? 'Outputs ready' : 'Imported'} ${imported.length} checkpoints: ${imported.join(', ')}`
      : '';
    const detail = [
      phase,
      stepText,
      j.rank ? `rank ${j.rank}` : '',
      j.observed_rank && j.observed_rank !== j.rank ? `reported rank ${j.observed_rank}` : '',
      j.learning_rate ? `lr ${j.learning_rate}` : '',
      j.resolution ? `${j.resolution}px` : '',
      j.save_every ? `save every ${j.save_every}` : '',
      j.optimizer ? j.optimizer : '',
      j.scheduler ? j.scheduler : '',
      j.network_alpha ? `alpha ${j.network_alpha}` : '',
      j.generate_samples === false ? 'samples off' : '',
      j.auto_import_lora === false ? 'no auto-import' : '',
      modelLabel,
    ].filter(Boolean).join(' · ');
    const progressLabel = j.current_step && j.total_steps
      ? `${progress.toFixed(progress % 1 ? 1 : 0)}% · ${j.current_step}/${j.total_steps}`
      : `${progress.toFixed(progress % 1 ? 1 : 0)}%`;
    const action = ['queued', 'running'].includes(j.status)
      ? `<button class="btn sm danger" data-train-cancel="${esc(j.id)}">Cancel</button>`
      : `<button class="btn sm" data-train-dismiss="${esc(j.id)}">Dismiss</button>`;
    return `<div class="trainer-job" data-train-open="${esc(j.id)}" style="cursor:pointer">
      <div class="trainer-job-head">
        <div class="trainer-job-name" title="${esc(j.output_name || j.id)}">${esc(j.output_name || j.id)}</div>
        <div class="trainer-job-status">${esc(j.status || 'unknown')}</div>
      </div>
      <div class="trainer-job-meta">${esc(detail)}</div>
      ${j.rank_warning ? `<div class="trainer-warnings">${esc(j.rank_warning)}</div>` : ''}
      <div class="trainer-progress-row">
        <span>${esc(phase)}</span>
        <span>${esc(progressLabel)}</span>
      </div>
      <div class="trainer-progress" style="--p:${progress}%"><span></span></div>
      ${j.error ? `<div class="trainer-warnings">${esc(j.error)}</div>` : ''}
      ${importedText ? `<div class="trainer-ok">${esc(importedText)}</div>` : ''}
      ${tail ? `<div class="trainer-log">${esc(tail)}</div>` : ''}
      <div class="trainer-actions">${action}</div>
    </div>`;
  }).join('');
  // Whole card clicks open the running monitor; the inner cancel/dismiss
  // button stops propagation in its own handler below.
  root.querySelectorAll('[data-train-open]').forEach(card => {
    card.addEventListener('click', (e) => {
      if (e.target.closest('[data-train-cancel],[data-train-dismiss]')) return;
      openTrainerRunningView(card.dataset.trainOpen);
    });
  });
  root.querySelectorAll('[data-train-cancel],[data-train-dismiss]').forEach(btn => {
    btn.addEventListener('click', async (ev) => {
      ev.stopPropagation();
      const id = btn.dataset.trainCancel || btn.dataset.trainDismiss;
      if (!id) return;
      if (state.preview) {
        state.trainJobs = state.trainJobs.filter(j => j.id !== id);
        renderTrainerJobs();
        return;
      }
      try {
        await api.cancelTrainJob(id);
        await refreshTrainerJobs();
      } catch (e) {
        toast(e.message || 'Could not update job', 'error');
      }
    });
  });
}

async function refreshTrainerJobs() {
  if (state.preview) {
    state.trainJobs = (state.trainJobs || []).map(j => {
      if (!['queued', 'running'].includes(j.status)) return j;
      return previewAdvanceTrainerJob(j, 12);
    });
    if (runJobId) trainerRunJob = state.trainJobs.find(j => j.id === runJobId) || trainerRunJob;
    renderTrainerJobs();
    return;
  }
  try {
    const data = await api.trainJobs();
    const previousDone = new Set((state.trainJobs || []).filter(j => j.status === 'done').map(j => j.id));
    state.trainJobs = data.jobs || [];
    renderTrainerJobs();
    if (state.trainJobs.some(j => j.status === 'done' && !previousDone.has(j.id))) {
      await loadLoras();
    }
  } catch {}
}

function pokeTrainerPoll() {
  if (state.trainPoll) clearTimeout(state.trainPoll);
  const tick = async () => {
    await refreshTrainerJobs();
    const active = (state.trainJobs || []).some(j => ['queued', 'running'].includes(j.status));
    if (active || state.view === 'trainers') {
      state.trainPoll = setTimeout(tick, active ? 2500 : 6000);
    } else {
      state.trainPoll = null;
    }
  };
  tick();
}

// ── LoRAs ────────────────────────────────────────────────────────────────
async function loadLoras() {
  if (state.preview) {
    state.loras = [];
    $('#loraNavCount').textContent = 0;
    renderLoraPanel();
    return;
  }
  try {
    const data = await api.loras();
    state.loras = data.loras || [];
  } catch { state.loras = []; }
  $('#loraNavCount').textContent = state.loras.length;
  renderLoraPanel();
}

// ── Components (transformer / VAE / text-encoder split-file swaps) ──────
async function loadComponents() {
  if (state.preview) {
    state.components = [];
    return;
  }
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
      <button class="icon-btn" data-hf-upload="${fn}" title="Save to Hugging Face"><svg style="width:13px;height:13px"><use href="#i-upload"/></svg></button>
      <button class="del" data-del="${fn}" title="Delete"><svg style="width:13px;height:13px"><use href="#i-trash"/></svg></button>
    </div>`;
  }).join('');
  list.querySelectorAll('[data-hf-upload]').forEach(b => b.addEventListener('click', () => {
    const lora = state.loras.find(x => loraId(x) === b.dataset.hfUpload);
    if (lora) openLoraHFUpload(lora);
  }));
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

function openLoraHFUpload(lora) {
  const id = loraId(lora);
  const base = String(lora.filename || id || 'lora')
    .replace(/\.safetensors$/i, '')
    .replace(/[^a-z0-9._-]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
  $('#modalTitle').textContent = 'Save LoRA to Hugging Face';
  $('#modalMeta').textContent = lora.rel_path || lora.filename || id;
  $('#modalBody').innerHTML = `
    <div class="trainer-form compact">
      <label class="field wide">
        <span class="field-label">Model repo</span>
        <input class="text-input" id="hfLoraRepo" placeholder="username/${esc(base || 'my-lora')}" autocomplete="off" autocapitalize="none" autocorrect="off" spellcheck="false">
      </label>
      <label class="field wide">
        <span class="field-label">HF write token</span>
        <input class="text-input mask-text" id="hfLoraToken" type="text" value="${esc(state.settings.hf_token || '')}" placeholder="hf_…" autocomplete="off" autocapitalize="none" autocorrect="off" spellcheck="false" data-1p-ignore="true" data-lpignore="true" data-bwignore="true" data-form-type="other">
      </label>
      <label class="field wide inline-check">
        <input type="checkbox" id="hfLoraPrivate" checked>
        <span>Private repo</span>
      </label>
    </div>`;
  $('#modalFoot').innerHTML = `
    <button class="btn ghost" id="modalCancel">Cancel</button>
    <button class="btn primary" id="modalUpload"><svg><use href="#i-upload"/></svg>Save to HF</button>`;
  $('#modalCancel').onclick = closeModal;
  $('#modalUpload').onclick = async () => {
    const repo = $('#hfLoraRepo')?.value.trim() || '';
    if (!repo.includes('/')) return toast('Use a repo id like username/kerry-qwen-lora', 'error');
    const btn = $('#modalUpload');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span><span>Uploading…</span>';
    try {
      if (state.preview) {
        await delay(500);
        closeModal();
        toast(`Saved to ${repo}`, 'success');
        return;
      }
      const res = await api.uploadLoraHF({
        filename: id,
        hf_repo: repo,
        private: !!$('#hfLoraPrivate')?.checked,
        hf_token: currentHFToken('#hfLoraToken') || null,
      });
      closeModal();
      toast(`Saved to ${res.repo}`, 'success');
    } catch (e) {
      toast(e.message || 'HF upload failed', 'error');
      btn.disabled = false;
      btn.innerHTML = '<svg><use href="#i-upload"/></svg>Save to HF';
    }
  };
  openModal();
}

// ── CivitAI ──────────────────────────────────────────────────────────────
// Pagination state lives on the module so Load More keeps appending. Reset
// on every new search (different query OR sort).
let civPage = 1;
let civHasMore = false;
let civHiddenTotal = 0;

async function civSearch(opts = {}) {
  const q = $('#civSearch').value;
  const sort = $('#civSort')?.value || 'Most Downloaded';
  const baseModel = $('#civBaseModel')?.value || '';
  const key = state.settings.civitai_key || '';
  const out = $('#civResults');
  const append = !!opts.append;
  if (!append) {
    civPage = 1;
    civHiddenTotal = 0;
    state.civResults = [];
    out.innerHTML = `<div class="empty" style="grid-column:1/-1"><div class="spinner"></div></div>`;
  }
  try {
    const data = await api.civSearch(q, { sort, baseModel, key, page: civPage });
    const items = data.items || [];
    state.civResults = append ? [...state.civResults, ...items] : items;
    civHasMore = !!data.metadata?.forge_has_more && items.length > 0;
    civHiddenTotal += Number(data.metadata?.forge_hidden || 0);
    renderCivResults();
  } catch {
    out.innerHTML = `<div class="empty" style="grid-column:1/-1">Failed to reach CivitAI.</div>`;
  }
}

function renderCivResults() {
  const out = $('#civResults');
  if (!state.civResults.length) {
    out.innerHTML = `<div class="empty" style="grid-column:1/-1">No results — try a different sort or query.</div>`;
    return;
  }
  const cards = state.civResults.map((m, i) => {
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
  // Hidden-by-moderation note only shows when there's something to report.
  // Load More appears whenever the upstream said there's more to fetch.
  const noteParts = [];
  if (civHiddenTotal > 0) noteParts.push(`<span style="color:var(--t-3);font-size:11.5px">${civHiddenTotal} hidden by moderation</span>`);
  noteParts.push(`<span style="color:var(--t-3);font-size:11.5px">${state.civResults.length} shown</span>`);
  const more = civHasMore
    ? `<button class="btn" id="civLoadMore" style="grid-column:1/-1;margin-top:10px">Load more</button>`
    : '';
  out.innerHTML = cards + `<div style="grid-column:1/-1;display:flex;gap:12px;align-items:center;justify-content:center;margin-top:12px">${noteParts.join('')}</div>` + more;
  out.querySelectorAll('.civ-card').forEach(c =>
    c.addEventListener('click', () => openCivDetail(Number(c.dataset.idx))));
  $('#civLoadMore')?.addEventListener('click', async () => {
    civPage += 1;
    await civSearch({ append: true });
  });
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
    if (f === 'image' || f === 'video' || f === 'audio') return a.kind === f;
    return a.source === f;
  });
  if (!items.length && !queued.length) {
    grid.innerHTML = `<div class="empty" style="grid-column:1/-1">Nothing here yet.</div>`;
    return;
  }
  const queueHtml = queueCardsHtml(queued, { showModel: true });
  const assetHtml = items.map((a, i) => {
    const n = esc(a.name);
    return `<div class="asset" data-asset-idx="${i}">
      ${assetMediaHtml(a, { lazy: true })}
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
  $$('.asset[data-asset-idx]', grid).forEach(el => el.addEventListener('click', async (e) => {
    if (e.target.closest('[data-asset-menu]')) return;
    if (e.target.closest('audio')) return;
    const idx = Number(el.dataset.assetIdx);
    await openAssetLightbox(items[idx], { list: items, index: idx });
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
      const kind = f.type.startsWith('video') ? 'video' : f.type.startsWith('audio') ? 'audio' : 'image';
      state.assets.unshift({ name: f.name, url, path: f.name, kind, source: 'upload' });
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
//   "tencent/HunyuanVideo"
//   "https://huggingface.co/tencent/HunyuanVideo"
//   "https://huggingface.co/tencent/HunyuanVideo/tree/main"
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
  if (targets.has('datasets') && state.view === 'datasets') await loadDatasetLibrary();
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

function mockTrainers() {
  return {
    trainers: [
      {
        id: 'qwen-character-lora',
        name: 'Qwen Character LoRA',
        configured: true,
        command_env: 'IGGLEPIXEL_QWEN_LORA_TRAIN_CMD',
        base_models: [
          { id: 'Qwen/Qwen-Image', label: 'Qwen-Image' },
          { id: 'Qwen/Qwen-Image-2512', label: 'Qwen-Image-2512' },
          { id: 'Qwen/Qwen-Image-Edit', label: 'Qwen-Image-Edit' },
          { id: 'Qwen/Qwen-Image-Edit-2511', label: 'Qwen-Image-Edit 2511' },
        ],
        model_families: [
          { id: 'qwen', label: 'Qwen Image', status: 'live', description: 'Character LoRA training is wired today, including checkpoints and library import.' },
          { id: 'flux', label: 'Flux Klein', status: 'live', description: 'Flux.2 Klein 9B turbo/base LoRA training uses the same guided dataset, config, and monitor workflow.' },
          { id: 'z-image', label: 'Z Image', status: 'planned', description: 'Planned after Flux once the wrapper and validation path are proven.' },
        ],
      },
      {
        id: 'flux-klein-character-lora',
        name: 'Flux Klein Character LoRA',
        configured: true,
        command_env: 'IGGLEPIXEL_FLUX_LORA_TRAIN_CMD',
        base_models: [
          { id: 'black-forest-labs/FLUX.2-klein-9B', label: 'FLUX.2 [klein] 9B Turbo' },
          { id: 'black-forest-labs/FLUX.2-klein-base-9B', label: 'FLUX.2 [klein] 9B Base' },
        ],
        model_families: [
          { id: 'qwen', label: 'Qwen Image', status: 'live', description: 'Character LoRA training is wired today, including checkpoints and library import.' },
          { id: 'flux', label: 'Flux Klein', status: 'live', description: 'Flux.2 Klein 9B turbo/base LoRA training uses the same guided dataset, config, and monitor workflow.' },
          { id: 'z-image', label: 'Z Image', status: 'planned', description: 'Planned after Flux once the wrapper and validation path are proven.' },
        ],
      },
    ],
    jobs: state.trainJobs || [],
  };
}

init().catch((err) => {
  console.error('App failed to start:', err);
  openAuthGate('login', 'The app failed to start. Refresh once the pod is ready.');
});
