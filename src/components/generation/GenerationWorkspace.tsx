"use client";

import { clsx } from "clsx";
import {
  Dice5,
  Download,
  ImagePlus,
  Layers,
  ListPlus,
  Maximize2,
  RotateCcw,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { PreviewTile } from "@/components/PreviewTile";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Textarea } from "@/components/ui/Field";
import { Slider } from "@/components/ui/Slider";
import { DEFAULT_NEGATIVE_PROMPT, QUEUE, RESOLUTION_PRESETS } from "@/lib/mock";
import type { Lora, ModelInfo, QueueJob, ResolutionPreset } from "@/lib/types";
import { ResolutionPicker } from "./ResolutionPicker";
import { QueuePanel } from "./QueuePanel";

type SeedMode = "random" | "fixed";

function randomSeed() {
  return Math.floor(Math.random() * 1_000_000);
}

function svgForJob(job: QueueJob): string {
  const h2 = (job.hue + 40) % 360;
  const prompt = job.prompt.replace(/[<>&"]/g, (c) =>
    ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" })[c]!,
  );
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${job.width}" height="${job.height}" viewBox="0 0 ${job.width} ${job.height}">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="hsl(${job.hue} 45% 28%)"/><stop offset="1" stop-color="hsl(${h2} 55% 18%)"/></linearGradient></defs>
<rect width="100%" height="100%" fill="url(#g)"/>
<circle cx="${job.width * 0.3}" cy="${job.height * 0.22}" r="${Math.min(job.width, job.height) * 0.22}" fill="rgba(255,255,255,0.12)"/>
<text x="50%" y="50%" text-anchor="middle" fill="white" font-family="system-ui, sans-serif" font-size="${Math.max(24, Math.round(job.width / 22))}" font-weight="700">Citivia Preview</text>
<text x="50%" y="57%" text-anchor="middle" fill="rgba(255,255,255,0.72)" font-family="system-ui, sans-serif" font-size="${Math.max(14, Math.round(job.width / 48))}">seed ${job.seed} · ${prompt}</text>
</svg>`;
}

export function GenerationWorkspace({ model }: { model: ModelInfo }) {
  const isEdit = model.kind === "editing";

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
  const [selectedLoras, setSelectedLoras] = useState<string[]>([]);
  const [loraError, setLoraError] = useState<string | null>(null);
  const [reference, setReference] = useState<{
    hue: number;
    name: string;
    dataUrl: string;
  } | null>(null);
  const refInput = useRef<HTMLInputElement>(null);

  // ---- Queue / preview ----
  const [jobs, setJobs] = useState<QueueJob[]>(() =>
    QUEUE.filter((j) => j.model === model.id),
  );
  const [focused, setFocused] = useState<QueueJob | null>(
    () => QUEUE.find((j) => j.model === model.id && j.status === "completed") ?? null,
  );
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
      hue: (useSeed % 360),
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
          loras: job.loras,
        }),
      });
      if (!res.ok) throw new Error((await res.json()).error ?? "Runner failed.");
      const result = await res.json();
      const done: QueueJob = {
        ...job,
        width: result.width,
        height: result.height,
        seed: result.seed,
        status: "completed",
        progress: 100,
        imageDataUrl: `data:${result.mime};base64,${result.image_base64}`,
        outputPath: result.path,
      };
      setJobs((prev) => prev.map((j) => (j.id === job.id ? done : j)));
      setFocused(done);
    } catch (err) {
      const failed: QueueJob = {
        ...job,
        status: "failed",
        progress: 0,
        error: err instanceof Error ? err.message : "Runner failed.",
      };
      setJobs((prev) => prev.map((j) => (j.id === job.id ? failed : j)));
      setFocused(failed);
    } finally {
      setBusyId(null);
    }
  }

  function addToQueue() {
    const job = buildJob("pending");
    setJobs((prev) => [...prev, job]);
  }

  function removeJob(id: string) {
    setJobs((prev) => prev.filter((j) => j.id !== id));
    setFocused((f) => (f && f.id === id ? null : f));
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
    if (job.imageDataUrl) {
      const link = document.createElement("a");
      link.href = job.imageDataUrl;
      link.download = `${job.id}.png`;
      link.click();
      return;
    }
    const url = URL.createObjectURL(
      new Blob([svgForJob(job)], { type: "image/svg+xml" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `${job.id}.svg`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function toggleLora(loraPath: string) {
    setSelectedLoras((prev) =>
      prev.includes(loraPath)
        ? prev.filter((item) => item !== loraPath)
        : [...prev, loraPath],
    );
  }

  const generating = busyId !== null;

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
                    hue: (f.size % 360) || 200,
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
                    hue={reference.hue}
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
              {selectedLoras.length}
            </Badge>
          </div>
          {loraError && <p className="text-sm text-danger">{loraError}</p>}
          {loras.length > 0 ? (
            <div className="grid gap-2">
              {loras.map((lora) => (
                <label
                  key={lora.id}
                  className="flex cursor-pointer items-center gap-3 rounded-[12px] border border-border bg-background px-3 py-3 transition-colors hover:border-lilac"
                >
                  <input
                    type="checkbox"
                    checked={selectedLoras.includes(lora.path)}
                    onChange={() => toggleLora(lora.path)}
                    className="h-4 w-4 accent-lilac"
                  />
                  <Layers className="h-4 w-4 shrink-0 text-lilac" />
                  <span className="min-w-0 flex-1 truncate text-sm font-medium">
                    {lora.name}
                  </span>
                  <span className="hidden max-w-[180px] truncate font-mono text-xs text-text-muted sm:block">
                    {lora.path}
                  </span>
                </label>
              ))}
            </div>
          ) : (
            <p className="text-sm text-text-muted">No installed LoRAs.</p>
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
          <Button className="min-w-[150px] flex-1" variant="secondary" onClick={addToQueue}>
            <ListPlus className="h-4 w-4" /> Add to Queue
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
              <div className="relative w-full overflow-hidden rounded-[12px] bg-background">
                <PreviewTile
                  hue={focused.hue}
                  width={focused.width}
                  height={focused.height}
                  src={focused.imageDataUrl}
                  showLock={focused.status !== "running"}
                />
                {focused.status === "running" && (
                  <div className="absolute inset-x-0 bottom-0 p-3">
                    <div className="h-1.5 overflow-hidden rounded-full bg-black/40">
                      <div
                        className="h-full rounded-full bg-lilac transition-all"
                        style={{ width: `${focused.progress}%` }}
                      />
                    </div>
                    <p className="mt-1 text-center text-xs text-white/80">
                      Step {Math.round((focused.progress / 100) * focused.steps)} /{" "}
                      {focused.steps}
                    </p>
                  </div>
                )}
              </div>
              <p className="text-xs text-text-muted">
                {focused.width}×{focused.height} · seed {focused.seed} · {focused.steps}{" "}
                steps · CFG {focused.cfg}
              </p>
              {focused.status === "completed" && (
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

      {/* Lightbox */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 grid place-items-center bg-black/80 p-4 backdrop-blur-sm animate-fade-in"
          onClick={() => setLightbox(null)}
        >
          <div
            className="relative max-h-[88dvh] w-full max-w-3xl overflow-hidden rounded-[16px]"
            onClick={(e) => e.stopPropagation()}
          >
            <PreviewTile
              hue={lightbox.hue}
              width={lightbox.width}
              height={lightbox.height}
              src={lightbox.imageDataUrl}
              showLock={false}
            />
            <button
              onClick={() => setLightbox(null)}
              className="absolute right-3 top-3 grid h-9 w-9 place-items-center rounded-full bg-black/50 text-white hover:bg-black/70"
              aria-label="Close"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
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
