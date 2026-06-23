"use client";

import { clsx } from "clsx";
import { Download, Loader2, Maximize2, Trash2 } from "lucide-react";
import { PreviewTile } from "@/components/PreviewTile";
import { Badge } from "@/components/ui/Badge";
import type { JobStatus, QueueJob } from "@/lib/types";

function statusTone(s: JobStatus) {
  return s === "running"
    ? "lilac"
    : s === "completed"
      ? "success"
      : s === "failed"
        ? "danger"
        : "neutral";
}

export function QueuePanel({
  jobs,
  onRemove,
  onView,
  onDownload,
}: {
  jobs: QueueJob[];
  onRemove: (id: string) => void;
  onView: (job: QueueJob) => void;
  onDownload: (job: QueueJob) => void;
}) {
  if (jobs.length === 0) {
    return (
      <p className="rounded-[12px] bg-background p-6 text-center text-sm text-text-muted">
        Queue is empty. Generate or add jobs to see them here.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {jobs.map((job) => (
        <div
          key={job.id}
          className="flex gap-3 rounded-[12px] border border-border bg-background p-3"
        >
          <button
            onClick={() => onView(job)}
            className="relative grid w-16 shrink-0 place-items-center overflow-hidden rounded-[8px] bg-surface"
            aria-label="View full size"
          >
            {job.imageDataUrl ? (
              <PreviewTile
                hue={job.hue}
                width={job.width}
                height={job.height}
                src={job.imageDataUrl}
                showLock={false}
                className="rounded-[8px]"
              />
            ) : job.status === "running" ? (
              <Loader2 className="h-5 w-5 animate-spin text-lilac" />
            ) : (
              <span className="h-2 w-2 rounded-full bg-text-muted" />
            )}
            {job.status === "running" && job.imageDataUrl && (
              <span className="absolute inset-0 grid place-items-center bg-black/40">
                <Loader2 className="h-5 w-5 animate-spin text-white" />
              </span>
            )}
          </button>

          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <div className="flex items-center justify-between gap-2">
              <Badge tone={statusTone(job.status)}>{job.status}</Badge>
              <span className="text-xs text-text-muted tabular-nums">
                {job.width}×{job.height}
              </span>
            </div>
            <p className="truncate text-sm text-text-secondary" title={job.prompt}>
              {job.prompt}
            </p>
            {job.status === "failed" && job.error ? (
              <p className="truncate text-xs text-danger-text" title={job.error}>
                {job.error}
              </p>
            ) : null}
            {job.status === "running" ? (
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-border">
                <div
                  className="h-full rounded-full bg-lilac transition-all"
                  style={{ width: `${job.progress}%` }}
                />
              </div>
            ) : (
              <div className="mt-0.5 flex items-center gap-1">
                {job.status === "completed" && (
                  <>
                    <IconBtn label="View full size" onClick={() => onView(job)}>
                      <Maximize2 className="h-3.5 w-3.5" />
                    </IconBtn>
                    <IconBtn label="Download" onClick={() => onDownload(job)}>
                      <Download className="h-3.5 w-3.5" />
                    </IconBtn>
                  </>
                )}
                <IconBtn label="Delete" onClick={() => onRemove(job.id)}>
                  <Trash2 className="h-3.5 w-3.5" />
                </IconBtn>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function IconBtn({
  children,
  label,
  onClick,
}: {
  children: React.ReactNode;
  label: string;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      className={clsx(
        "grid h-7 w-7 place-items-center rounded-md text-text-muted",
        "transition-colors hover:bg-surface-hover hover:text-white",
      )}
    >
      {children}
    </button>
  );
}
