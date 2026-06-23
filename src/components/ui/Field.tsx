import { clsx } from "clsx";
import type {
  InputHTMLAttributes,
  TextareaHTMLAttributes,
  ReactNode,
} from "react";

const fieldBase =
  "w-full rounded-[12px] bg-background border border-border px-4 py-3 text-base " +
  "text-white placeholder:text-text-muted transition-colors " +
  "focus:border-lilac focus:outline-none focus:ring-2 focus:ring-lilac/30";

export function Label({ children }: { children: ReactNode }) {
  return (
    <label className="mb-2 block text-base font-medium text-text-secondary">
      {children}
    </label>
  );
}

export function Input({
  className,
  ...props
}: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={clsx(fieldBase, "h-12", className)} {...props} />;
}

export function Textarea({
  className,
  ...props
}: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea className={clsx(fieldBase, "resize-y", className)} {...props} />
  );
}
