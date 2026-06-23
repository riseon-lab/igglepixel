import type { ModelInfo, ResolutionPreset } from "./types";

export const DEFAULT_NEGATIVE_PROMPT =
  "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, " +
  "fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, " +
  "watermark, username, blurry";

export const RESOLUTION_PRESETS: ResolutionPreset[] = [
  { label: "Square", width: 512, height: 512 },
  { label: "Square HD", width: 1024, height: 1024 },
  { label: "Square XL", width: 1380, height: 1380 },
  { label: "Square 2K", width: 2048, height: 2048 },
  { label: "Portrait", width: 1344, height: 1792 },
  { label: "Portrait Tall", width: 1024, height: 1360 },
  { label: "Story", width: 1080, height: 1920 },
  { label: "Landscape", width: 1920, height: 1080 },
];

export const MODELS: ModelInfo[] = [
  {
    id: "qwen-2512",
    name: "Qwen 2512",
    tagline: "Base Generation",
    description:
      "High-fidelity text-to-image generation. Best for original artwork, concepts and renders.",
    kind: "generation",
    vramGb: 57,
  },
  {
    id: "qwen-edit-2511",
    name: "Qwen Edit 2511",
    tagline: "Image Editing",
    description:
      "Reference-guided editing. Supply an image and a prompt to transform, restyle or extend it.",
    kind: "editing",
    vramGb: 57,
  },
];
