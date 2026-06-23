"use client";

import { clsx } from "clsx";
import {
  AlertTriangle,
  ArrowRight,
  ImageIcon,
  Loader2,
  Play,
  Square,
  Terminal,
  Trash2,
  Wand2,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { useToast } from "@/components/ui/Toast";
import { MODELS } from "@/lib/models";
import type { RunnerHealth } from "@/lib/runners/client";
import type { ModelId } from "@/lib/types";

type LiveStatus = "running" | "starting" | "stopping" | "stopped" | "offline";

interface HealthEntry {
  model: ModelId;
  ok: boolean;
  detail?: RunnerHealth;
  error?: string;
}

const POLL_MS = 4000;

const STATUS_META: Record<
  LiveStatus,
  { label: string; tone: "success" | "warning" | "neutral" | "danger"; dot: string }
> = {
  running: { label: "Running", tone: "success", dot: "bg-success" },
  starting: { label: "Starting…", tone: "warning", dot: "bg-warning" },
  stopping: { label: "Stopping…", tone: "warning", dot: "bg-warning" },
  stopped: { label: "Stopped", tone: "neutral", dot: "bg-text-muted" },
  offline: { label: "Runner offline", tone: "danger", dot: "bg-danger" },
};

function deriveStatus(
  entry: HealthEntry | undefined,
  pending: "starting" | "stopping" | undefined,
): LiveStatus {
  if (pending) return pending;
  if (!entry || !entry.ok || !entry.detail) return "offline";
  if (entry.detail.loaded) return "running";
  if (entry.detail.loading) return "starting";
  return "stopped";
}

export default function ModelsPage() {
  const toast = useToast();
  const [health, setHealth] = useState<Record<string, HealthEntry>>({});
  const [pending, setPending] = useState<
    Record<string, "starting" | "stopping" | undefined>
  >({});
  const [deleting, setDeleting] = useState<Record<string, boolean>>({});
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/runners/health", { cache: "no-store" });
      if (!res.ok) throw new Error();
      const data: HealthEntry[] = await res.json();
      const map: Record<string, HealthEntry> = {};
      for (const e of data) map[e.model] = e;
      setHealth(map);
      setPending((prev) => {
        const next = { ...prev };
        for (const e of data) {
          const p = next[e.model];
          if (p === "starting" && e.detail?.loaded) delete next[e.model];
          if (p === "stopping" && e.detail && !e.detail.loaded) delete next[e.model];
        }
        return next;
      });
    } catch {
      setHealth({});
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    const initial = setTimeout(() => void refresh(), 0);
    const t = setInterval(() => void refresh(), POLL_MS);
    return () => {
      clearTimeout(initial);
      clearInterval(t);
    };
  }, [refresh]);

  async function control(model: ModelId, action: "start" | "stop", name: string) {
    setPending((p) => ({ ...p, [model]: action === "start" ? "starting" : "stopping" }));
    try {
      const res = await fetch(`/api/runners/${model}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.error || `Could not ${action} ${name}.`);
      if (action === "start")
        toast.info(`Starting ${name}`, "First load downloads weights and can take a few minutes.");
      else toast.success(`${name} stopped`, "Its VRAM has been freed.");
      refresh();
    } catch (err) {
      setPending((p) => ({ ...p, [model]: undefined }));
      toast.error(
        `Could not ${action} ${name}`,
        err instanceof Error ? err.message : "The runner is unreachable.",
      );
    }
  }

  async function deleteWeights(model: ModelId, name: string) {
    if (
      !confirm(
        `Delete the downloaded weights for ${name}? It will re-download (tens of GB) the next time you start it.`,
      )
    )
      return;
    setDeleting((d) => ({ ...d, [model]: true }));
    try {
      const res = await fetch(`/api/runners/${model}`, { method: "DELETE" });
      const body = await res.json().catch(() => ({}));
      if (!res.ok || body.ok === false)
        throw new Error(body.error || "Could not delete weights.");
      toast.success(`${name} weights deleted`, "Freed disk on /workspace.");
      refresh();
    } catch (err) {
      toast.error(
        `Could not delete ${name} weights`,
        err instanceof Error ? err.message : "The runner is unreachable.",
      );
    } finally {
      setDeleting((d) => ({ ...d, [model]: false }));
    }
  }

  const entries = Object.values(health);
  const anyReachable = entries.some((e) => e.ok);
  const loadErrors = entries.filter((e) => e.detail?.load_error);
  const terminals = MODELS.map((m) => ({
    model: m,
    status: deriveStatus(health[m.id], pending[m.id]),
    logs: health[m.id]?.detail?.logs ?? [],
    error: health[m.id]?.detail?.load_error,
  })).filter((item) => item.status === "starting" || item.error);

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Models"
        description="Start a model, watch its runner, manage weights, then open the workspace."
      />

      {loaded && !anyReachable && (
        <Card className="flex items-start gap-3 border-warning/40">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-warning" />
          <div>
            <p className="font-medium">No runners reachable</p>
            <p className="mt-1 text-sm text-text-secondary">
              Runner control becomes available once the inference services are online.
              On RunPod they start with the container; locally, run the Docker image.
            </p>
          </div>
        </Card>
      )}

      {loadErrors.map((e) => (
        <Card key={e.model} className="flex items-start gap-3 border-danger/50">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-danger-text" />
          <div>
            <p className="font-medium">Load failed: {e.model}</p>
            <p className="mt-1 text-sm text-text-secondary">{e.detail?.load_error}</p>
          </div>
        </Card>
      ))}

      {terminals.map(({ model, logs }) => (
        <Card key={model.id} className="flex flex-col gap-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-text-secondary">
            <Terminal className="h-4 w-4" />
            {model.name} terminal
          </div>
          <div className="max-h-56 overflow-auto rounded-[12px] bg-black p-4 font-mono text-xs leading-6 text-[#9cffb4]">
            {logs.length ? (
              logs.map((line, index) => <p key={`${line}-${index}`}>{line}</p>)
            ) : (
              <p>Starting runner… waiting for first log line.</p>
            )}
          </div>
        </Card>
      ))}

      <section className="grid gap-4">
        {MODELS.map((m) => {
          const Icon = m.kind === "editing" ? ImageIcon : Wand2;
          const status = deriveStatus(health[m.id], pending[m.id]);
          const meta = STATUS_META[status];
          const busy = status === "starting" || status === "stopping";
          const isRunning = status === "running";
          const isDeleting = !!deleting[m.id];
          return (
            <Card key={m.id} className="flex flex-col gap-4">
              <div className="flex items-start justify-between gap-4">
                <div className="grid h-12 w-12 shrink-0 place-items-center rounded-[12px] bg-lilac/15 text-lilac">
                  <Icon className="h-6 w-6" />
                </div>
                <Badge tone={meta.tone}>
                  {busy ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <span className={clsx("h-2 w-2 rounded-full", meta.dot)} />
                  )}
                  {meta.label}
                </Badge>
              </div>

              <div className="max-w-3xl">
                <h3 className="text-xl font-semibold">{m.name}</h3>
                <p className="text-sm font-medium text-lilac">{m.tagline}</p>
                <p className="mt-2 text-sm text-text-secondary">{m.description}</p>
                <p className="mt-1 text-xs text-text-muted">~{m.vramGb} GB VRAM</p>
              </div>

              <div className="flex flex-wrap gap-3">
                <Link
                  href={`/generate/${m.id}`}
                  className="inline-flex h-10 items-center justify-center gap-2 rounded-[12px] border border-border bg-surface px-4 text-base font-semibold text-white transition-colors hover:bg-surface-hover"
                >
                  Open workspace <ArrowRight className="h-4 w-4" />
                </Link>
                {isRunning ? (
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={busy}
                    onClick={() => control(m.id, "stop", m.name)}
                  >
                    <Square className="h-4 w-4" /> Stop
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    disabled={busy || isDeleting || status === "offline"}
                    onClick={() => control(m.id, "start", m.name)}
                  >
                    {busy ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                    Start
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-danger-text hover:text-danger-text"
                  disabled={busy || isDeleting || status === "offline"}
                  onClick={() => deleteWeights(m.id, m.name)}
                >
                  {isDeleting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                  Delete weights
                </Button>
              </div>
            </Card>
          );
        })}
      </section>
    </div>
  );
}
