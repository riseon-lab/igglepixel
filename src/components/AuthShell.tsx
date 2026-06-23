import { Wand2 } from "lucide-react";
import type { ReactNode } from "react";

// Shared layout for the setup/login screens. Lives in components/ (not in a
// page file) because Next.js only allows specific exports from app/ pages.
export function AuthShell({
  eyebrow,
  title,
  subtitle,
  children,
}: {
  eyebrow: string;
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <div className="grid min-h-dvh place-items-center bg-background px-4 py-10">
      <div className="w-full max-w-md animate-fade-in">
        <div className="mb-8 flex flex-col items-center text-center">
          <div className="mb-4 grid h-14 w-14 place-items-center rounded-[16px] bg-lilac text-white shadow-lg shadow-lilac/25">
            <Wand2 className="h-7 w-7" />
          </div>
          <p className="text-sm font-medium tracking-wide text-lilac uppercase">
            {eyebrow}
          </p>
          <h1 className="mt-1 text-[28px] font-bold">{title}</h1>
          <p className="mt-2 text-text-secondary">{subtitle}</p>
        </div>
        <div className="rounded-[24px] border border-border bg-surface p-6 sm:p-8">
          {children}
        </div>
      </div>
    </div>
  );
}
