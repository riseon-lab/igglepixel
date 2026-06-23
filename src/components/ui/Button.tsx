import { clsx } from "clsx";
import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "danger" | "ghost";
type Size = "md" | "sm";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  children: ReactNode;
}

const base =
  "inline-flex items-center justify-center gap-2 rounded-[12px] font-semibold " +
  "transition-colors transition-shadow duration-200 select-none disabled:opacity-40 " +
  "disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 " +
  "focus-visible:ring-lilac/70 focus-visible:ring-offset-2 focus-visible:ring-offset-background";

const variants: Record<Variant, string> = {
  primary:
    "bg-lilac text-white shadow-md shadow-lilac/20 hover:bg-lilac-dark active:bg-lilac-dark",
  secondary:
    "bg-surface text-white border border-border hover:bg-surface-hover",
  danger: "bg-danger text-white hover:brightness-110",
  ghost: "text-text-secondary hover:text-white hover:bg-surface-hover",
};

const sizes: Record<Size, string> = {
  md: "h-12 px-5 text-base",
  sm: "h-10 px-4 text-base",
};

export function Button({
  variant = "primary",
  size = "md",
  className,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={clsx(base, variants[variant], sizes[size], className)}
      {...props}
    >
      {children}
    </button>
  );
}
