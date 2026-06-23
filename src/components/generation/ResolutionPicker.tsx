"use client";

import { clsx } from "clsx";
import type { ResolutionPreset } from "@/lib/types";

// Visual, selectable cards showing the actual aspect ratio (plan.md).
export function ResolutionPicker({
  presets,
  value,
  onChange,
}: {
  presets: ResolutionPreset[];
  value: ResolutionPreset;
  onChange: (p: ResolutionPreset) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {presets.map((p) => {
        const selected = p.width === value.width && p.height === value.height;
        return (
          <button
            key={p.label}
            type="button"
            onClick={() => onChange(p)}
            className={clsx(
              "min-w-0 flex flex-col items-center gap-2 rounded-[12px] border p-3 transition-colors",
              selected
                ? "border-lilac bg-lilac/10"
                : "border-border bg-background hover:bg-surface-hover",
            )}
            title={`${p.width} × ${p.height}`}
          >
            <span className="grid h-12 w-12 place-items-center">
              <span
                className={clsx(
                  "rounded-[4px] border-2",
                  selected ? "border-lilac bg-lilac/20" : "border-text-muted",
                )}
                style={{
                  width: p.width >= p.height ? 40 : (40 * p.width) / p.height,
                  height: p.height >= p.width ? 40 : (40 * p.height) / p.width,
                }}
              />
            </span>
            <span className="text-center text-[11px] font-medium leading-tight">
              {p.width}×{p.height}
            </span>
          </button>
        );
      })}
    </div>
  );
}
