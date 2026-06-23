"use client";

import { Cpu, MemoryStick, Play, Square, Thermometer, Zap } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { MODELS, RESOURCE_USAGE } from "@/lib/mock";
import type { ModelStatus } from "@/lib/types";

function statusBadge(status: ModelStatus) {
  switch (status) {
    case "running":
      return (
        <Badge tone="success">
          <span className="h-2 w-2 rounded-full bg-success" /> Running
        </Badge>
      );
    case "starting":
      return <Badge tone="warning">Starting…</Badge>;
    case "stopping":
      return <Badge tone="warning">Stopping…</Badge>;
    default:
      return (
        <Badge tone="neutral">
          <span className="h-2 w-2 rounded-full bg-text-muted" /> Stopped
        </Badge>
      );
  }
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
  pct: number;
}) {
  return (
    <Card>
      <div className="flex items-center justify-between text-text-secondary">
        <span className="flex items-center gap-2 text-sm">
          <Icon className="h-4 w-4" /> {label}
        </span>
        {detail && <span className="text-sm text-text-muted">{detail}</span>}
      </div>
      <p className="mt-2 text-2xl font-semibold tabular-nums">{value}</p>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-border">
        <div
          className="h-full rounded-full bg-lilac transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
    </Card>
  );
}

export default function RunningPage() {
  const [models, setModels] = useState(MODELS);

  function toggle(id: string) {
    setModels((prev) =>
      prev.map((m) =>
        m.id === id
          ? { ...m, status: m.status === "running" ? "stopped" : "running" }
          : m,
      ),
    );
  }

  const r = RESOURCE_USAGE;

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Running"
        description="Monitor active models and live resource usage."
      />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Meter
          icon={Zap}
          label="GPU"
          value={`${r.gpuUtil}%`}
          detail={r.gpuName}
          pct={r.gpuUtil}
        />
        <Meter
          icon={Cpu}
          label="VRAM"
          value={`${r.vramUsedGb} / ${r.vramTotalGb} GB`}
          pct={(r.vramUsedGb / r.vramTotalGb) * 100}
        />
        <Meter
          icon={MemoryStick}
          label="System RAM"
          value={`${r.ramUsedGb} / ${r.ramTotalGb} GB`}
          pct={(r.ramUsedGb / r.ramTotalGb) * 100}
        />
        <Meter
          icon={Thermometer}
          label="Temp"
          value={`${r.temperatureC}°C`}
          pct={(r.temperatureC / 100) * 100}
        />
      </div>

      <section className="flex flex-col gap-4">
        <h2 className="text-[20px] font-semibold">Models</h2>
        <div className="grid gap-4">
          {models.map((m) => {
            const running = m.status === "running";
            return (
              <Card key={m.id} className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-start gap-4">
                  <div className="grid h-12 w-12 shrink-0 place-items-center rounded-[12px] bg-lilac/15 text-lilac">
                    <Zap className="h-6 w-6" />
                  </div>
                  <div>
                    <div className="flex flex-wrap items-center gap-3">
                      <h3 className="text-lg font-semibold">{m.name}</h3>
                      {statusBadge(m.status)}
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
                  {running ? (
                    <Button variant="secondary" onClick={() => toggle(m.id)}>
                      <Square className="h-4 w-4" /> Stop
                    </Button>
                  ) : (
                    <Button onClick={() => toggle(m.id)}>
                      <Play className="h-4 w-4" /> Start
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
