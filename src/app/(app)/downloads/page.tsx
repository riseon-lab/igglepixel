"use client";

import { clsx } from "clsx";
import {
  CheckCircle2,
  Loader2,
  Pause,
  Play,
  RotateCw,
  XCircle,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { formatBytes } from "@/lib/format";
import { DOWNLOADS } from "@/lib/mock";
import type { DownloadItem, DownloadStatus } from "@/lib/types";

function StatusPill({ status }: { status: DownloadStatus }) {
  switch (status) {
    case "downloading":
      return (
        <Badge tone="lilac">
          <Loader2 className="h-3 w-3 animate-spin" /> Downloading
        </Badge>
      );
    case "queued":
      return <Badge tone="neutral">Queued</Badge>;
    case "completed":
      return (
        <Badge tone="success">
          <CheckCircle2 className="h-3 w-3" /> Completed
        </Badge>
      );
    case "failed":
      return (
        <Badge tone="danger">
          <XCircle className="h-3 w-3" /> Failed
        </Badge>
      );
    case "paused":
      return <Badge tone="warning">Paused</Badge>;
  }
}

export default function DownloadsPage() {
  const [items, setItems] = useState<DownloadItem[]>(DOWNLOADS);

  // Mock progress ticking so the preview feels alive.
  useEffect(() => {
    const t = setInterval(() => {
      setItems((prev) =>
        prev.map((d) => {
          if (d.status !== "downloading") return d;
          const next = Math.min(100, d.progress + Math.round(2 + (d.id.length % 4)));
          return next >= 100
            ? { ...d, progress: 100, status: "completed" }
            : { ...d, progress: next };
        }),
      );
    }, 1200);
    return () => clearInterval(t);
  }, []);

  function setStatus(id: string, status: DownloadStatus) {
    setItems((prev) => prev.map((d) => (d.id === id ? { ...d, status } : d)));
  }

  const active = items.filter((d) =>
    ["downloading", "queued", "paused"].includes(d.status),
  );
  const history = items.filter((d) =>
    ["completed", "failed"].includes(d.status),
  );

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Downloads"
        description="Track model and LoRA downloads and their history."
      />

      <Section title="Active" count={active.length} empty="No active downloads.">
        {active.map((d) => (
          <Card key={d.id} className="flex flex-col gap-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="font-semibold">{d.name}</h3>
                  <Badge tone="neutral">{d.kind}</Badge>
                </div>
                <p className="mt-0.5 text-sm text-text-muted">
                  {formatBytes(d.sizeBytes)} · {d.source}
                </p>
              </div>
              <StatusPill status={d.status} />
            </div>
            <div className="flex items-center gap-3">
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-border">
                <div
                  className="h-full rounded-full bg-lilac transition-all duration-500"
                  style={{ width: `${d.progress}%` }}
                />
              </div>
              <span className="w-10 text-right text-sm tabular-nums text-text-secondary">
                {d.progress}%
              </span>
              {d.status === "downloading" ? (
                <Button size="sm" variant="ghost" onClick={() => setStatus(d.id, "paused")}>
                  <Pause className="h-4 w-4" />
                </Button>
              ) : d.status === "paused" ? (
                <Button size="sm" variant="ghost" onClick={() => setStatus(d.id, "downloading")}>
                  <Play className="h-4 w-4" />
                </Button>
              ) : null}
            </div>
          </Card>
        ))}
      </Section>

      <Section title="History" count={history.length} empty="No download history.">
        {history.map((d) => (
          <Card key={d.id} className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="truncate font-semibold">{d.name}</h3>
                <Badge tone="neutral">{d.kind}</Badge>
              </div>
              <p className="mt-0.5 text-sm text-text-muted">
                {formatBytes(d.sizeBytes)} · {d.source}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <StatusPill status={d.status} />
              {d.status === "failed" && (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => setStatus(d.id, "downloading")}
                >
                  <RotateCw className="h-4 w-4" /> Retry
                </Button>
              )}
            </div>
          </Card>
        ))}
      </Section>
    </div>
  );
}

function Section({
  title,
  count,
  empty,
  children,
}: {
  title: string;
  count: number;
  empty: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-[20px] font-semibold">{title}</h2>
        <span className="text-sm text-text-muted">{count}</span>
      </div>
      {count === 0 ? (
        <Card className="py-10 text-center text-text-muted">{empty}</Card>
      ) : (
        <div className={clsx("grid gap-3")}>{children}</div>
      )}
    </section>
  );
}
