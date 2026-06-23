import { clsx } from "clsx";
import type { ReactNode } from "react";

type Tone = "neutral" | "lilac" | "success" | "danger" | "warning";

const tones: Record<Tone, string> = {
  neutral: "bg-surface-hover text-text-secondary",
  lilac: "bg-lilac/15 text-lilac",
  success: "bg-success/15 text-success",
  danger: "bg-danger/20 text-danger-text",
  warning: "bg-warning/15 text-warning",
};

export function Badge({
  tone = "neutral",
  children,
  className,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium whitespace-nowrap",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
