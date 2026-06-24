"use client";

import { clsx } from "clsx";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Clock,
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
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Textarea } from "@/components/ui/Field";
import { Slider } from "@/components/ui/Slider";
import { useToast } from "@/components/ui/Toast";
import { useEncryption } from "@/lib/crypto/provider";
import { useModelWorkspace } from "@/lib/generation/workspace-store";
import { deleteAsset, uploadAsset } from "@/lib/vault/client";
import { DEFAULT_NEGATIVE_PROMPT, RESOLUTION_PRESETS } from "@/lib/models";
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

interface JobParams {
  negativePrompt: string;
  imageBase64?: string;
  matchRef: boolean;
}

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

interface GenResult {
  width: number;
  height: number;
  seed: number;
  mime: string;
  image_base64: string;
  path?: string | null;
}

export function GenerationWorkspace({ model }: { model: ModelInfo }) {
  const isEdit = model.kind === "editing";
  const toast = useToast();
  const { client: cryptoClient } = useEncryption();
  const { jobs, references, refIndex, setJobs, setReferences, setRefIndex } =
    useModelWorkspace(model.id);

  // ---- Controls (local; not persisted) ----
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState(DEFAULT_NEGATIVE_PROMPT);
  const [resolution, setResolution] = useState<ResolutionPreset>(
    RESOLUTION_PRESETS[1],
  );
  const [matchRef, setMatchRef] = useState(true); // edit only
  const [steps, setSteps] = useState(20);
  const [cfg, setCfg] = useState(4);
  const [seedMode, setSeedMode] = useState<SeedMode>("random");
  const [seed, setSeed] = useState(randomSeed());
  const [lastSeed, setLastSeed] = useState<number | null>(null);
  const [loras, setLoras] = useState<Lora[]>([]);
  const [selectedLoras, setSelectedLoras] = useState<LoraSelection[]>([]);
  const [loraToAdd, setLoraToAdd] = useState("");
  const [loraError, setLoraError] = useState<string | null>(null);
  const refInput = useRef<HTMLInputElement>(null);

  // ---- Queue / preview ----
  const [focused, setFocused] = useState<QueueJob | null>(null);
  const [lightbox, setLightbox] = useState<QueueJob | null>(null);
  const [runningId, setRunningId] = useState<string | null>(null);

  // Mirror jobs into a ref so the async queue drainer reads the latest list, and
  // keep per-job generation params (negative/ref/matchRef) that aren't on QueueJob.
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  const drainingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const paramsRef = useRef<Record<string, JobParams>>({});

  const activeRef = references[refIndex];
  const generating = runningId !== null;
  const pendingCount = jobs.filter((j) => j.status === "pending").length;

  // Write through both the ref (immediate, for the drainer) and the store.
  function setJobsBoth(update: (prev: QueueJob[]) => QueueJob[]) {
    const next = update(jobsRef.current);
    jobsRef.current = next;
    setJobs(() => next);
  }

  function patchJob(id: string, patch: Partial<QueueJob>) {
    setJobsBoth((prev) => prev.map((j) => (j.id === id ? { ...j, ...patch } : j)));
    setFocused((f) => (f && f.id === id ? { ...f, ...patch } : f));
  }

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

  // On (re)mount, any job still "running"/"pending" is a leftover from a previous
  // visit whose in-flight request and params are gone — mark it interrupted so the
  // queue doesn't show a stuck spinner. Completed/failed jobs persist as history.
  useEffect(() => {
    setJobsBoth((prev) =>
      prev.map((j) =>
        j.status === "running" || j.status === "pending"
          ? {
              ...j,
              status: "failed",
              progress: 0,
              error: "Interrupted — generation didn't finish.",
            }
          : j,
      ),
    );
    setFocused((f) => f ?? jobsRef.current[jobsRef.current.length - 1] ?? null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function enqueue() {
    if (isEdit && !activeRef) return;
    const useSeed = seedMode === "random" ? randomSeed() : seed;
    if (seedMode === "random") setSeed(useSeed);
    const job: QueueJob = {
      id: `job-${Date.now()}-${Math.floor(Math.random() * 10000)}`,
      model: model.id,
      prompt: prompt.trim() || "(no prompt)",
      width: resolution.width,
      height: resolution.height,
      steps,
      cfg,
      seed: useSeed,
      status: "pending",
      progress: 0,
      createdAt: new Date().toISOString(),
      loras: selectedLoras,
    };
    paramsRef.current[job.id] = {
      negativePrompt: negative,
      imageBase64: isEdit ? activeRef?.dataUrl : undefined,
      matchRef: isEdit ? matchRef : false,
    };
    setJobsBoth((prev) => [...prev, job]);
    setLastSeed(useSeed);
    if (!generating) setFocused(job);
    void drain();
  }

  async function drain() {
    if (drainingRef.current) return;
    drainingRef.current = true;
    try {
      for (;;) {
        const next = jobsRef.current.find((j) => j.status === "pending");
        if (!next) break;
        await runJob(next);
      }
    } finally {
      drainingRef.current = false;
    }
  }

  async function runJob(job: QueueJob) {
    const extra =
      paramsRef.current[job.id] ?? { negativePrompt: " ", matchRef: false };
    setRunningId(job.id);
    patchJob(job.id, { status: "running", progress: 0, error: undefined });
    setFocused(jobsRef.current.find((j) => j.id === job.id) ?? null);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model: model.id,
          prompt: job.prompt,
          negativePrompt: extra.negativePrompt,
          width: job.width,
          height: job.height,
          steps: job.steps,
          cfg: job.cfg,
          seed: job.seed,
          imageBase64: extra.imageBase64,
          matchRef: extra.matchRef,
          loras: job.loras?.filter(isLoraEnabled),
        }),
      });

      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => "");
        let msg = text;
        try {
          msg = JSON.parse(text).error ?? text;
        } catch {
          /* not JSON (e.g. proxy HTML) — keep raw text */
        }
        throw new Error(msg || "Runner failed.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let result: GenResult | null = null;
      let streamError: string | null = null;

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
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
            patchJob(job.id, {
              progress:
                typeof msg.progress === "number" ? msg.progress : undefined,
              imageDataUrl: msg.preview_base64
                ? `data:${msg.preview_mime ?? "image/png"};base64,${msg.preview_base64}`
                : undefined,
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

      patchJob(job.id, {
        width: result.width,
        height: result.height,
        seed: result.seed,
        status: "completed",
        progress: 100,
        imageDataUrl: `data:${result.mime};base64,${result.image_base64}`,
        outputPath: result.path ?? undefined,
      });

      // Persist to the encrypted vault so it shows in Assets (best-effort).
      if (cryptoClient) {
        try {
          const bytes = base64ToBytes(result.image_base64);
          const ciphertext = await cryptoClient.encrypt(bytes);
          const meta = await uploadAsset(ciphertext, {
            name: `${model.id}-${result.seed}.png`,
            kind: "generated",
            mime: result.mime || "image/png",
            width: result.width,
            height: result.height,
            size: bytes.length,
          });
          patchJob(job.id, { vaultId: meta.id });
        } catch {
          toast.error(
            "Couldn't save to Assets",
            "The image generated but wasn't stored — download it to keep it.",
          );
        }
      }
    } catch (err) {
      const aborted =
        controller.signal.aborted ||
        (err instanceof DOMException && err.name === "AbortError");
      if (aborted) {
        patchJob(job.id, { status: "failed", progress: 0, error: "Cancelled" });
      } else {
        const message = err instanceof Error ? err.message : "Runner failed.";
        patchJob(job.id, { status: "failed", progress: 0, error: message });
        toast.error(
          `${model.name} generation failed`,
          /unreachable|fetch failed|ECONNREFUSED|502/.test(message)
            ? "The model runner isn't reachable. Start it on the Models page."
            : message,
        );
      }
    } finally {
      abortRef.current = null;
      setRunningId((cur) => (cur === job.id ? null : cur));
      delete paramsRef.current[job.id];
    }
  }

  function cancelCurrent() {
    abortRef.current?.abort();
    void fetch(`/api/runners/${model.id}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ action: "cancel" }),
    }).catch(() => {});
    toast.success("Cancelling…");
  }

  function removeJob(id: string) {
    const job = jobsRef.current.find((j) => j.id === id);
    if (job?.status === "running") cancelCurrent();
    setJobsBoth((prev) => prev.filter((j) => j.id !== id));
    setFocused((f) => (f && f.id === id ? null : f));
    delete paramsRef.current[id];
    if (job?.vaultId) {
      void deleteAsset(job.vaultId).catch(() => {});
      toast.success("Deleted", "Removed from queue and Assets.");
    } else {
      toast.success("Removed from queue");
    }
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

  // ---- Reference images ----
  function addReferenceFiles(files: FileList | null) {
    if (!files?.length) return;
    const startIndex = references.length;
    Array.from(files).forEach((f) => {
      const reader = new FileReader();
      reader.onload = () =>
        setReferences((prev) => [
          ...prev,
          { name: f.name, dataUrl: String(reader.result) },
        ]);
      reader.readAsDataURL(f);
    });
    setRefIndex(startIndex);
    if (refInput.current) refInput.current.value = "";
  }

  function cycleRef(dir: number) {
    if (references.length < 2) return;
    setRefIndex((refIndex + dir + references.length) % references.length);
  }

  function removeActiveRef() {
    setReferences((prev) => prev.filter((_, i) => i !== refIndex));
  }

  // ---- LoRAs ----
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

  const selectedPaths = new Set(selectedLoras.map((item) => item.path));
  const enabledLoraCount = selectedLoras.filter(isLoraEnabled).length;
  const availableLoras = loras.filter((lora) => !selectedPaths.has(lora.path));

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(320px,380px)]">
      {/* ---------- Controls column ---------- */}
      <div className="order-2 min-w-0 flex flex-col gap-5 xl:order-1">
        {isEdit && (
          <Card className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <SectionLabel>Reference Images</SectionLabel>
              {references.length > 0 && (
                <Badge tone="lilac">
                  {refIndex + 1}/{references.length}
                </Badge>
              )}
            </div>
            <input
              ref={refInput}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(e) => addReferenceFiles(e.target.files)}
            />
            {activeRef ? (
              <div className="flex flex-col gap-3">
                <div className="relative">
                  <div className="overflow-hidden rounded-[12px] bg-background">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={activeRef.dataUrl}
                      alt={activeRef.name}
                      className="max-h-72 w-full object-contain"
                    />
                  </div>
                  {references.length > 1 && (
                    <>
                      <button
                        onClick={() => cycleRef(-1)}
                        className="absolute left-2 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-full bg-black/60 text-white hover:bg-black/80"
                        aria-label="Previous reference"
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => cycleRef(1)}
                        className="absolute right-2 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-full bg-black/60 text-white hover:bg-black/80"
                        aria-label="Next reference"
                      >
                        <ChevronRight className="h-4 w-4" />
                      </button>
                    </>
                  )}
                  <button
                    onClick={removeActiveRef}
                    className="absolute right-2 top-2 grid h-7 w-7 place-items-center rounded-full bg-black/60 text-white hover:bg-black/80"
                    aria-label="Remove this reference"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
                <p className="truncate text-sm font-medium" title={activeRef.name}>
                  {activeRef.name}
                </p>
                {references.length > 1 && (
                  <div className="flex gap-2 overflow-x-auto pb-1">
                    {references.map((r, i) => (
                      <button
                        key={`${r.name}-${i}`}
                        onClick={() => setRefIndex(i)}
                        className={clsx(
                          "h-12 w-12 shrink-0 overflow-hidden rounded-md border",
                          i === refIndex
                            ? "border-lilac"
                            : "border-border opacity-70 hover:opacity-100",
                        )}
                        aria-label={`Reference ${i + 1}`}
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={r.dataUrl}
                          alt=""
                          className="h-full w-full object-cover"
                        />
                      </button>
                    ))}
                  </div>
                )}
                <Button
                  variant="secondary"
                  onClick={() => refInput.current?.click()}
                >
                  <ImagePlus className="h-4 w-4" /> Add more
                </Button>
              </div>
            ) : (
              <button
                onClick={() => refInput.current?.click()}
                className="flex flex-col items-center justify-center gap-2 rounded-[12px] border-2 border-dashed border-border py-10 text-text-muted transition-colors hover:border-lilac hover:text-white"
              >
                <ImagePlus className="h-7 w-7" />
                <span className="text-sm">Upload reference image(s) to edit</span>
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
          {isEdit && (
            <div className="flex w-fit gap-1 rounded-[10px] bg-background p-1">
              <SeedTab active={matchRef} onClick={() => setMatchRef(true)}>
                Match reference
              </SeedTab>
              <SeedTab active={!matchRef} onClick={() => setMatchRef(false)}>
                Custom size
              </SeedTab>
            </div>
          )}
          {isEdit && matchRef ? (
            <p className="text-sm text-text-muted">
              Output keeps each reference image&apos;s own dimensions.
            </p>
          ) : (
            <ResolutionPicker
              presets={RESOLUTION_PRESETS}
              value={resolution}
              onChange={setResolution}
            />
          )}
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
            className="min-w-[160px] flex-1"
            onClick={enqueue}
            disabled={isEdit && !activeRef}
          >
            <Sparkles className="h-4 w-4" />
            {generating ? "Queue another" : "Generate"}
          </Button>
          {generating && (
            <Button variant="secondary" onClick={cancelCurrent}>
              <X className="h-4 w-4" /> Cancel
            </Button>
          )}
        </div>
      </div>

      {/* ---------- Preview + Queue column ---------- */}
      <div className="order-1 min-w-0 flex flex-col gap-5 xl:order-2 xl:sticky xl:top-20 xl:self-start">
        <Card className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <SectionLabel>Preview</SectionLabel>
            {focused?.status === "running" ? (
              <Badge tone="lilac">{focused.progress}%</Badge>
            ) : focused?.status === "pending" ? (
              <Badge tone="neutral">Queued</Badge>
            ) : null}
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
                    ) : focused.status === "pending" ? (
                      <>
                        <Clock className="h-8 w-8" />
                        <p className="text-sm">Queued — waiting to run</p>
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
              {(focused.status === "failed" || focused.status === "pending") && (
                <Button size="sm" variant="ghost" onClick={() => removeJob(focused.id)}>
                  <Trash2 className="h-4 w-4" /> Remove
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
            <span className="text-xs text-text-muted">
              {jobs.length} job{jobs.length === 1 ? "" : "s"}
              {pendingCount > 0 ? ` · ${pendingCount} queued` : ""}
            </span>
          </div>
          <QueuePanel
            jobs={[...jobs].reverse()}
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
