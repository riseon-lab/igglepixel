"use client";

import { clsx } from "clsx";

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  /** Optional formatter for the value chip. */
  format?: (value: number) => string;
  className?: string;
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
  className,
}: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className={clsx("flex flex-col gap-2", className)}>
      <div className="flex items-center justify-between">
        <label className="text-base font-medium text-text-secondary">
          {label}
        </label>
        <span className="rounded-lg bg-surface-hover px-2.5 py-0.5 text-base font-semibold tabular-nums">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="citivia-range"
        style={{
          background: `linear-gradient(to right, var(--color-lilac) ${pct}%, var(--color-border) ${pct}%)`,
        }}
      />
    </div>
  );
}
