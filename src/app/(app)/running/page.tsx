"use client";

import { clsx } from "clsx";
import {
  AlertTriangle,
  Cpu,
  Loader2,
  Play,
  Power,
  Square,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { useToast } from "@/components/ui/Toast";
import { MODELS } from "@/lib/mock";
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

function Meter({
  icon: Icon,
  label,
  value,
  detail,
  pct,
}: {
  icon: typeof Cpu;
  label: string;
  value: string;
  detail?: string;
  pct?: number;
}) {
  return (
    <Card>
      <div className="flex items-center justify-between text-text-secondary">
        <span className="flex items-center gap-2 text-sm">
          <Icon className="h-4 w-4" /> {label}
        </span>
        {detail && <span className="truncate text-sm text-text-muted">{detail}</span>}
      </div>
      <p className="mt-2 text-2xl font-semibold tabular-nums">{value}</p>
      {pct !== undefined && (
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-border">
          <div
            className="h-full rounded-full bg-lilac transition-all"
            style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
          />
        </div>
      )}
    </Card>
  );
}

export default function RunningPage() {
  const toast = useToast();
  const [health, setHealth] = useState<Record<string, HealthEntry>>({});
  const [pending, setPending] = useState<
    Record<string, "starting" | "stopping" | undefined>
  >({});
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/runners/health", { cache: "no-store" });
      if (!res.ok) throw new Error();
      const data: HealthEntry[] = await res.json();
      const map: Record<string, HealthEntry> = {};
      for (const e of data) map[e.model] = e;
      setHealth(map);
      // Clear optimistic state once the runner confirms the target state.
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
    fetch("/api/runners/health", { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json() as Promise<HealthEntry[]>;
      })
      .then((data) => {
        const map: Record<string, HealthEntry> = {};
        for (const e of data) map[e.model] = e;
        setHealth(map);
      })
      .catch(() => setHealth({}))
      .finally(() => setLoaded(true));
    const t = setInterval(refresh, POLL_MS);
    return () => clearInterval(t);
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
        toast.info(`Starting ${name}`, "Loading the model — first run downloads weights and can take a few minutes.");
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

  // --- Resource telemetry derived from live runner health ---
  const entries = Object.values(health);
  const anyReachable = entries.some((e) => e.ok);
  const gpu = entries.find((e) => e.detail?.vram_total_gb)?.detail;
  const loadedCount = entries.filter((e) => e.detail?.loaded).length;
  const loadErrors = entries.filter((e) => e.detail?.load_error);

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Running"
        description="Start, stop and monitor the model runners."
      />

      {loaded && !anyReachable ? (
        <Card className="flex items-start gap-3 border-warning/40">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-warning" />
          <div>
            <p className="font-medium">No runners reachable</p>
            <p className="mt-1 text-sm text-text-secondary">
              GPU telemetry and model control become available once the inference
              runners are online (port 8011 / 8012). On RunPod they start with the
              container; locally, run the Docker image.
            </p>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-3">
          <Meter
            icon={Zap}
            label="VRAM"
            value={
              gpu?.vram_total_gb
                ? `${gpu.vram_used_gb ?? 0} / ${gpu.vram_total_gb} GB`
                : "—"
            }
            detail={gpu?.device ?? undefined}
            pct={
              gpu?.vram_total_gb
                ? ((gpu.vram_used_gb ?? 0) / gpu.vram_total_gb) * 100
                : undefined
            }
          />
          <Meter
            icon={Power}
            label="Models loaded"
            value={`${loadedCount} / ${MODELS.length}`}
          />
          <Meter
            icon={Cpu}
            label="Compute"
            value={gpu?.cuda || entries.some((e) => e.detail?.cuda) ? "CUDA" : "CPU"}
            detail={anyReachable ? "Runners online" : undefined}
          />
        </div>
      )}

      {loadErrors.map((e) => (
        <Card key={e.model} className="flex items-start gap-3 border-danger/50">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-[#ff8a80]" />
          <div>
            <p className="font-medium">Load failed: {e.model}</p>
            <p className="mt-1 text-sm text-text-secondary">{e.detail?.load_error}</p>
          </div>
        </Card>
      ))}

      <section className="flex flex-col gap-4">
        <h2 className="text-[20px] font-semibold">Models</h2>
        <div className="grid gap-4">
          {MODELS.map((m) => {
            const entry = health[m.id];
            const status = deriveStatus(entry, pending[m.id]);
            const meta = STATUS_META[status];
            const busy = status === "starting" || status === "stopping";
            const isRunning = status === "running";
            return (
              <Card
                key={m.id}
                className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="flex items-start gap-4">
                  <div className="grid h-12 w-12 shrink-0 place-items-center rounded-[12px] bg-lilac/15 text-lilac">
                    <Zap className="h-6 w-6" />
                  </div>
                  <div>
                    <div className="flex flex-wrap items-center gap-3">
                      <h3 className="text-lg font-semibold">{m.name}</h3>
                      <Badge tone={meta.tone}>
                        {busy ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          <span className={clsx("h-2 w-2 rounded-full", meta.dot)} />
                        )}
                        {meta.label}
                      </Badge>
                    </div>
                    <p className="mt-1 max-w-prose text-sm text-text-secondary">
                      {m.description}
                    </p>
                    <p className="mt-1 text-xs text-text-muted">
                      {m.tagline} · ~{m.vramGb} GB VRAM
                    </p>
                  </div>
                </div>
                <div className="shrink-0">
                  {isRunning ? (
                    <Button
                      variant="secondary"
                      disabled={busy}
                      onClick={() => control(m.id, "stop", m.name)}
                    >
                      <Square className="h-4 w-4" /> Stop
                    </Button>
                  ) : (
                    <Button
                      disabled={busy || status === "offline"}
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
                </div>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
