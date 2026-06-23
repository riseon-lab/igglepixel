import { clsx } from "clsx";
import type { HTMLAttributes, ReactNode } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  /** Adds hover elevation + cursor for clickable cards. */
  interactive?: boolean;
  /** Lilac border to indicate selection. */
  selected?: boolean;
  padded?: boolean;
}

export function Card({
  children,
  interactive,
  selected,
  padded = true,
  className,
  ...props
}: CardProps) {
  return (
    <div
      className={clsx(
        "rounded-[16px] bg-surface border transition-all duration-200",
        padded && "p-6",
        selected ? "border-lilac" : "border-border",
        interactive &&
          "cursor-pointer hover:bg-surface-hover hover:border-border hover:-translate-y-0.5 hover:shadow-lg hover:shadow-black/30",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
