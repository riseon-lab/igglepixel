"use client";

import { clsx } from "clsx";
import {
  Check,
  Dice5,
  Download,
  ImagePlus,
  Layers,
  Maximize2,
  Loader2,
  RotateCcw,
  Plus,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { PreviewTile } from "@/components/PreviewTile";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Textarea } from "@/components/ui/Field";
import { Slider } from "@/components/ui/Slider";
import { useToast } from "@/components/ui/Toast";
import { useEncryption } from "@/lib/crypto/provider";
import { uploadAsset } from "@/lib/vault/client";
import { DEFAULT_NEGATIVE_PROMPT, RESOLUTION_PRESETS } from "@/lib/models";
import type { RunnerHealth } from "@/lib/runners/client";
import type {
  Lora,
  LoraSelection,
  ModelInfo,
  QueueJob,
  ResolutionPreset,
} from "@/lib/types";
import { ResolutionPicker } from "./ResolutionPicker";
import { QueuePanel } from "./QueuePanel";

type SeedMode = "random" | "fixed";
const GENERATION_POLL_MS = 500;

function randomSeed() {
  return Math.floor(Math.random() * 1_000_000);
}

function isLoraEnabled(lora: LoraSelection) {
  return lora.enabled !== false;
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

export function GenerationWorkspace({ model }: { model: ModelInfo }) {
  const isEdit = model.kind === "editing";
  const toast = useToast();
  const { client: cryptoClient } = useEncryption();

  // ---- Controls ----
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState(DEFAULT_NEGATIVE_PROMPT);
  const [resolution, setResolution] = useState<ResolutionPreset>(
    RESOLUTION_PRESETS[1],
  );
  const [steps, setSteps] = useState(20);
  const [cfg, setCfg] = useState(4);
  const [seedMode, setSeedMode] = useState<SeedMode>("random");
  const [seed, setSeed] = useState(randomSeed());
  const [lastSeed, setLastSeed] = useState<number | null>(null);
  const [loras, setLoras] = useState<Lora[]>([]);
  const [selectedLoras, setSelectedLoras] = useState<LoraSelection[]>([]);
  const [loraToAdd, setLoraToAdd] = useState("");
  const [loraError, setLoraError] = useState<string | null>(null);
  const [reference, setReference] = useState<{
    name: string;
    dataUrl: string;
  } | null>(null);
  const refInput = useRef<HTMLInputElement>(null);

  // ---- Queue / preview ----
  const [jobs, setJobs] = useState<QueueJob[]>([]);
  const [focused, setFocused] = useState<QueueJob | null>(null);
  const [lightbox, setLightbox] = useState<QueueJob | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/loras", { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error("Could not load LoRAs.");
        return res.json();
      })
      .then(setLoras)
      .catch((err) =>
        setLoraError(err instanceof Error ? err.message : "Could not load LoRAs."),
      );
  }, []);

  useEffect(() => {
    if (!busyId) return;

    async function pollGeneration() {
      try {
        const res = await fetch(`/api/runners/${model.id}`, { cache: "no-store" });
        if (!res.ok) return;
        const health: RunnerHealth = await res.json();
        const generation = health.generation;
        if (!generation) return;
        const progress = Math.max(0, Math.min(100, Math.round(generation.progress)));
        const preview = generation.preview_base64
          ? `data:${generation.preview_mime ?? "image/png"};base64,${generation.preview_base64}`
          : undefined;
        const update = (job: QueueJob): QueueJob =>
          job.id === busyId
            ? { ...job, progress, imageDataUrl: preview ?? job.imageDataUrl }
            : job;
        setJobs((prev) => prev.map(update));
        setFocused((job) => (job?.id === busyId ? update(job) : job));
      } catch {
        // Final /generate error handling owns user-visible failures.
      }
    }

    void pollGeneration();
    const timer = setInterval(() => void pollGeneration(), GENERATION_POLL_MS);
    return () => clearInterval(timer);
  }, [busyId, model.id]);

  function buildJob(status: QueueJob["status"]): QueueJob {
    const useSeed = seedMode === "random" ? randomSeed() : seed;
    if (seedMode === "random") setSeed(useSeed);
    return {
      id: `job-${Date.now()}`,
      model: model.id,
      prompt: prompt.trim() || "(no prompt)",
      width: resolution.width,
      height: resolution.height,
      steps,
      cfg,
      seed: useSeed,
      status,
      progress: 0,
      createdAt: new Date().toISOString(),
      loras: selectedLoras,
    };
  }

  async function generate() {
    const job = buildJob("running");
    setJobs((prev) => [job, ...prev.filter((j) => j.status !== "running")]);
    setFocused(job);
    setBusyId(job.id);
    setLastSeed(job.seed);
    // Keep both the queue and the focused panel in sync as the job evolves.
    let view = job;
    const apply = (patch: Partial<QueueJob>) => {
      view = { ...view, ...patch };
      setJobs((prev) => prev.map((j) => (j.id === view.id ? view : j)));
      setFocused(view);
    };

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: model.id,
          prompt: job.prompt,
          negativePrompt: negative,
          width: job.width,
          height: job.height,
          steps: job.steps,
          cfg: job.cfg,
          seed: job.seed,
          imageBase64: reference?.dataUrl,
          loras: job.loras?.filter(isLoraEnabled),
        }),
      });

      // A non-OK response is a plain error (validation/auth) — read it defensively
      // so an HTML proxy page never crashes JSON.parse.
      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => "");
        let msg = text;
        try {
          msg = JSON.parse(text).error ?? text;
        } catch {
          /* not JSON (e.g. proxy HTML) — keep the raw text */
        }
        throw new Error(msg || "Runner failed.");
      }

      // Stream of NDJSON frames: progress (with live preview) then a final result.
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      type GenResult = {
        width: number;
        height: number;
        seed: number;
        mime: string;
        image_base64: string;
        path?: string | null;
      };
      let result: GenResult | null = null;
      let streamError: string | null = null;

      for (;;) {
        const { done: streamDone, value } = await reader.read();
        if (streamDone) break;
        buffer += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (!line) continue;
          let msg: Record<string, unknown>;
          try {
            msg = JSON.parse(line);
          } catch {
            continue;
          }
          if (msg.type === "progress") {
            apply({
              progress: typeof msg.progress === "number" ? msg.progress : view.progress,
              imageDataUrl: msg.preview_base64
                ? `data:${msg.preview_mime ?? "image/png"};base64,${msg.preview_base64}`
                : view.imageDataUrl,
            });
          } else if (msg.type === "result") {
            result = msg as unknown as GenResult;
          } else if (msg.type === "error") {
            streamError = (msg.error as string) || "Runner failed.";
          }
        }
      }

      if (streamError) throw new Error(streamError);
      if (!result) throw new Error("Runner returned no image.");

      apply({
        width: result.width,
        height: result.height,
        seed: result.seed,
        status: "completed",
        progress: 100,
        imageDataUrl: `data:${result.mime};base64,${result.image_base64}`,
        outputPath: result.path ?? undefined,
      });

      // Persist the result to the encrypted vault so it appears in Assets.
      // Best-effort: a save failure must not turn a good generation into an error.
      if (cryptoClient) {
        try {
          const bytes = base64ToBytes(result.image_base64);
          const ciphertext = await cryptoClient.encrypt(bytes);
          await uploadAsset(ciphertext, {
            name: `${model.id}-${result.seed}.png`,
            kind: "generated",
            mime: result.mime || "image/png",
            width: result.width,
            height: result.height,
            size: bytes.length,
          });
          toast.success("Saved to Assets", "Encrypted in your browser.");
        } catch {
          toast.error(
            "Couldn't save to Assets",
            "The image generated but wasn't stored — download it to keep it.",
          );
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Runner failed.";
      const failed: QueueJob = {
        ...job,
        status: "failed",
        progress: 0,
        error: message,
      };
      setJobs((prev) => prev.map((j) => (j.id === job.id ? failed : j)));
      setFocused(failed);
      toast.error(
        `${model.name} generation failed`,
        /unreachable|fetch failed|ECONNREFUSED|502/.test(message)
          ? "The model runner isn't reachable. Start it on the Running page."
          : message,
      );
    } finally {
      setBusyId(null);
    }
  }

  function removeJob(id: string) {
    setJobs((prev) => prev.filter((j) => j.id !== id));
    setFocused((f) => (f && f.id === id ? null : f));
    toast.success("Removed from queue");
  }

  function reuseSettings(job: QueueJob) {
    setPrompt(job.prompt === "(no prompt)" ? "" : job.prompt);
    const preset =
      RESOLUTION_PRESETS.find(
        (p) => p.width === job.width && p.height === job.height,
      ) ?? RESOLUTION_PRESETS[1];
    setResolution(preset);
    setSteps(job.steps);
    setCfg(job.cfg);
    setSeed(job.seed);
    setSeedMode("fixed");
    setSelectedLoras(job.loras ?? []);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function downloadJob(job: QueueJob) {
    if (!job.imageDataUrl) return toast.error("No image to download yet.");
    const link = document.createElement("a");
    link.href = job.imageDataUrl;
    link.download = `${job.id}.png`;
    link.click();
  }

  function addLora() {
    if (!loraToAdd) return;
    setSelectedLoras((prev) =>
      prev.some((item) => item.path === loraToAdd)
        ? prev
        : [...prev, { path: loraToAdd, strength: 1 }],
    );
    setLoraToAdd("");
  }

  function removeLora(path: string) {
    setSelectedLoras((prev) => prev.filter((item) => item.path !== path));
  }

  function setLoraStrength(path: string, strength: number) {
    setSelectedLoras((prev) =>
      prev.map((item) =>
        item.path === path ? { ...item, strength: Math.max(0.1, strength) } : item,
      ),
    );
  }

  function toggleLora(path: string) {
    setSelectedLoras((prev) =>
      prev.map((item) =>
        item.path === path ? { ...item, enabled: !isLoraEnabled(item) } : item,
      ),
    );
  }

  const generating = busyId !== null;
  const selectedPaths = new Set(selectedLoras.map((item) => item.path));
  const enabledLoraCount = selectedLoras.filter(isLoraEnabled).length;
  const availableLoras = loras.filter((lora) => !selectedPaths.has(lora.path));

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(320px,380px)]">
      {/* ---------- Controls column ---------- */}
      <div className="min-w-0 flex flex-col gap-5">
        {isEdit && (
          <Card className="flex flex-col gap-3">
            <SectionLabel>Reference Image</SectionLabel>
            <input
              ref={refInput}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (!f) return;
                const reader = new FileReader();
                reader.onload = () => {
                  setReference({
                    name: f.name,
                    dataUrl: String(reader.result),
                  });
                };
                reader.readAsDataURL(f);
              }}
            />
            {reference ? (
              <div className="flex items-center gap-4">
                <div className="w-24 overflow-hidden rounded-[12px]">
                  <PreviewTile
                    width={1}
                    height={1}
                    src={reference.dataUrl}
                    showLock={false}
                  />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{reference.name}</p>
                  <p className="text-xs text-text-muted">Ready to edit</p>
                </div>
                <Button variant="ghost" onClick={() => setReference(null)} aria-label="Remove reference">
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <button
                onClick={() => refInput.current?.click()}
                className="flex flex-col items-center justify-center gap-2 rounded-[12px] border-2 border-dashed border-border py-10 text-text-muted transition-colors hover:border-lilac hover:text-white"
              >
                <ImagePlus className="h-7 w-7" />
                <span className="text-sm">Upload a reference image to edit</span>
              </button>
            )}
          </Card>
        )}

        <Card className="flex flex-col gap-3">
          <SectionLabel>Prompt</SectionLabel>
          <Textarea
            rows={5}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the image you want to create…"
          />
        </Card>

        <Card className="flex flex-col gap-3">
          <SectionLabel>Negative Prompt</SectionLabel>
          <Textarea
            rows={3}
            value={negative}
            onChange={(e) => setNegative(e.target.value)}
          />
        </Card>

        <Card className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <SectionLabel>LoRAs</SectionLabel>
            <Badge tone={selectedLoras.length ? "lilac" : "neutral"}>
              {selectedLoras.length
                ? `${enabledLoraCount}/${selectedLoras.length}`
                : 0}
            </Badge>
          </div>
          {loraError && <p className="text-sm text-danger-text">{loraError}</p>}
          {selectedLoras.length > 0 ? (
            <div className="grid gap-3">
              {selectedLoras.map((selection) => {
                const lora = loras.find((item) => item.path === selection.path);
                const enabled = isLoraEnabled(selection);
                return (
                  <div
                    key={selection.path}
                    className={clsx(
                      "rounded-[12px] border border-border bg-background p-3",
                      !enabled && "opacity-60",
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <button
                        type="button"
                        onClick={() => toggleLora(selection.path)}
                        className={clsx(
                          "grid h-8 w-8 shrink-0 place-items-center rounded-md border transition-colors",
                          enabled
                            ? "border-lilac bg-lilac text-white"
                            : "border-border text-text-muted hover:text-white",
                        )}
                        aria-label={`${enabled ? "Disable" : "Enable"} ${
                          lora?.name ?? selection.path
                        }`}
                      >
                        {enabled && <Check className="h-4 w-4" />}
                      </button>
                      <Layers className="h-4 w-4 shrink-0 text-lilac" />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium">
                          {lora?.name ?? selection.path}
                        </p>
                        <p className="truncate font-mono text-xs text-text-muted">
                          {selection.path}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => removeLora(selection.path)}
                        className="rounded-md p-2 text-text-muted transition-colors hover:bg-surface-hover hover:text-white"
                        aria-label={`Remove ${lora?.name ?? selection.path}`}
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                    <Slider
                      className="mt-3"
                      label="Strength"
                      min={0.1}
                      max={2}
                      step={0.1}
                      value={selection.strength}
                      onChange={(value) => setLoraStrength(selection.path, value)}
                      format={(value) => value.toFixed(1)}
                    />
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-text-muted">No LoRAs selected.</p>
          )}
          {loras.length > 0 ? (
            <div className="flex gap-2">
              <select
                value={loraToAdd}
                onChange={(e) => setLoraToAdd(e.target.value)}
                className="h-12 min-w-0 flex-1 rounded-[12px] border border-border bg-background px-3 text-base text-white focus:border-lilac focus:outline-none"
              >
                <option value="">Add installed LoRA...</option>
                {availableLoras.map((lora) => (
                  <option key={lora.id} value={lora.path}>
                    {lora.name}
                  </option>
                ))}
              </select>
              <Button variant="secondary" onClick={addLora} disabled={!loraToAdd}>
                <Plus className="h-4 w-4" /> Add
              </Button>
            </div>
          ) : (
            <Link
              href="/loras"
              className="inline-flex h-10 w-fit items-center justify-center gap-2 rounded-[12px] border border-border bg-surface px-4 text-base font-semibold text-white transition-colors hover:bg-surface-hover"
            >
              <Plus className="h-4 w-4" /> Add LoRA
            </Link>
          )}
        </Card>

        <Card className="flex flex-col gap-3">
          <SectionLabel>Resolution</SectionLabel>
          <ResolutionPicker
            presets={RESOLUTION_PRESETS}
            value={resolution}
            onChange={setResolution}
          />
        </Card>

        <Card className="flex flex-col gap-5">
          <SectionLabel>Generation Controls</SectionLabel>
          <Slider
            label="Steps"
            min={1}
            max={200}
            value={steps}
            onChange={setSteps}
          />
          <Slider
            label="CFG Scale"
            min={0}
            max={100}
            step={0.5}
            value={cfg}
            onChange={setCfg}
            format={(v) => v.toFixed(1)}
          />

          {/* Seed */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-base font-medium text-text-secondary">Seed</span>
              <div className="flex gap-1 rounded-[10px] bg-background p-1">
                <SeedTab active={seedMode === "random"} onClick={() => setSeedMode("random")}>
                  Random
                </SeedTab>
                <SeedTab active={seedMode === "fixed"} onClick={() => setSeedMode("fixed")}>
                  Fixed
                </SeedTab>
              </div>
            </div>
            <div className="flex gap-2">
              <input
                type="number"
                value={seed}
                disabled={seedMode === "random"}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="h-12 min-w-0 flex-1 rounded-[12px] border border-border bg-background px-4 text-base tabular-nums disabled:opacity-50 focus:border-lilac focus:outline-none"
              />
              <Button
                variant="secondary"
                onClick={() => {
                  setSeed(randomSeed());
                  setSeedMode("fixed");
                }}
                aria-label="Roll new seed"
              >
                <Dice5 className="h-4 w-4" />
              </Button>
              <Button
                variant="secondary"
                onClick={() => {
                  if (lastSeed !== null) {
                    setSeed(lastSeed);
                    setSeedMode("fixed");
                  }
                }}
                disabled={lastSeed === null}
              >
                Reuse
              </Button>
            </div>
          </div>
        </Card>

        {/* Action bar */}
        <div className="sticky bottom-24 z-10 flex flex-wrap gap-3 rounded-[16px] border border-border bg-surface/90 p-3 backdrop-blur md:bottom-4">
          <Button
            className="min-w-[180px] flex-1"
            onClick={generate}
            disabled={generating || (isEdit && !reference)}
          >
            <Sparkles className="h-4 w-4" /> Generate
          </Button>
        </div>
      </div>

      {/* ---------- Preview + Queue column ---------- */}
      <div className="min-w-0 flex flex-col gap-5 xl:sticky xl:top-20 xl:self-start">
        <Card className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <SectionLabel>Preview</SectionLabel>
            {focused?.status === "running" && (
              <Badge tone="lilac">{focused.progress}%</Badge>
            )}
          </div>

          {focused ? (
            <>
              {focused.imageDataUrl ? (
                <div
                  className="relative w-full overflow-hidden rounded-[12px] bg-background"
                  style={{ aspectRatio: `${focused.width} / ${focused.height}` }}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={focused.imageDataUrl}
                    alt=""
                    className="h-full w-full object-contain"
                  />
                  {focused.status === "running" && (
                    <span className="absolute left-2 top-2 rounded-md bg-black/55 px-2 py-0.5 text-xs text-white/90 backdrop-blur">
                      preview
                    </span>
                  )}
                </div>
              ) : (
                <div className="grid aspect-square place-items-center rounded-[12px] border border-border bg-background text-text-muted">
                  <div className="flex flex-col items-center gap-2 text-center">
                    {focused.status === "running" ? (
                      <>
                        <Loader2 className="h-8 w-8 animate-spin text-lilac" />
                        <p className="text-sm">Starting generation…</p>
                      </>
                    ) : (
                      <>
                        <Trash2 className="h-8 w-8 text-danger-text" />
                        <p className="text-sm">{focused.error ?? "Generation failed."}</p>
                      </>
                    )}
                  </div>
                </div>
              )}
              {focused.status === "running" && (
                <div className="flex flex-col gap-1">
                  <div className="h-1.5 overflow-hidden rounded-full bg-border">
                    <div
                      className="h-full rounded-full bg-lilac transition-all"
                      style={{ width: `${focused.progress}%` }}
                    />
                  </div>
                  <p className="text-center text-xs text-text-muted tabular-nums">
                    Step {Math.round((focused.progress / 100) * focused.steps)} /{" "}
                    {focused.steps}
                  </p>
                </div>
              )}
              <p className="text-xs text-text-muted">
                {focused.width}×{focused.height} · seed {focused.seed} · {focused.steps}{" "}
                steps · CFG {focused.cfg}
              </p>
              {focused.status === "completed" && focused.imageDataUrl && (
                <div className="grid grid-cols-2 gap-2">
                  <Button size="sm" variant="secondary" onClick={() => setLightbox(focused)}>
                    <Maximize2 className="h-4 w-4" /> Full Size
                  </Button>
                  <Button size="sm" variant="secondary" onClick={() => downloadJob(focused)}>
                    <Download className="h-4 w-4" /> Download
                  </Button>
                  <Button size="sm" variant="secondary" onClick={() => reuseSettings(focused)}>
                    <RotateCcw className="h-4 w-4" /> Reuse
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => removeJob(focused.id)}>
                    <Trash2 className="h-4 w-4" /> Delete
                  </Button>
                </div>
              )}
              {focused.status === "failed" && (
                <Button size="sm" variant="ghost" onClick={() => removeJob(focused.id)}>
                  <Trash2 className="h-4 w-4" /> Delete
                </Button>
              )}
            </>
          ) : (
            <div className="grid aspect-square place-items-center rounded-[12px] border border-dashed border-border text-text-muted">
              <div className="flex flex-col items-center gap-2">
                <Sparkles className="h-8 w-8" />
                <p className="text-sm">Your generated image will appear here</p>
              </div>
            </div>
          )}
        </Card>

        <Card className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <SectionLabel>Queue</SectionLabel>
            <span className="text-xs text-text-muted">{jobs.length} jobs</span>
          </div>
          <QueuePanel
            jobs={jobs}
            onRemove={removeJob}
            onView={setLightbox}
            onDownload={downloadJob}
          />
        </Card>
      </div>

      {/* Lightbox — shows the whole image at its real aspect ratio, no cropping */}
      {lightbox?.imageDataUrl && (
        <div
          className="fixed inset-0 z-50 grid place-items-center bg-black/85 p-4 backdrop-blur-sm animate-fade-in"
          onClick={() => setLightbox(null)}
        >
          {/* Close sits at the screen corner, clear of the image itself. */}
          <button
            onClick={() => setLightbox(null)}
            className="fixed right-4 top-4 z-10 grid h-10 w-10 place-items-center rounded-full bg-white/10 text-white hover:bg-white/20"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={lightbox.imageDataUrl}
            alt=""
            width={lightbox.width}
            height={lightbox.height}
            onClick={(e) => e.stopPropagation()}
            className="block max-h-[90dvh] max-w-[92vw] w-auto h-auto rounded-[12px] object-contain"
          />
        </div>
      )}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <h3 className="text-[20px] font-semibold">{children}</h3>;
}

function SeedTab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        "rounded-[8px] px-3 py-1.5 text-sm font-medium transition-colors",
        active ? "bg-lilac text-white" : "text-text-muted hover:text-white",
      )}
    >
      {children}
    </button>
  );
}
