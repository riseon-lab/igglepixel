import { clsx } from "clsx";
import { Lock } from "lucide-react";
import { previewGradient } from "@/lib/format";

// Stands in for a real (encrypted, client-decrypted) image preview. In the live
// build this becomes an in-memory object URL produced by the decryption worker;
// nothing is ever written to disk (see plan.md "Encryption").
export function PreviewTile({
  hue,
  width,
  height,
  className,
  showLock = true,
  label,
}: {
  hue: number;
  width: number;
  height: number;
  className?: string;
  showLock?: boolean;
  label?: string;
}) {
  return (
    <div
      className={clsx(
        "relative overflow-hidden",
        className,
      )}
      style={{ aspectRatio: `${width} / ${height}`, background: previewGradient(hue) }}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(255,255,255,0.12),transparent_60%)]" />
      {label && (
        <span className="absolute bottom-2 left-2 rounded-md bg-black/40 px-2 py-0.5 text-xs text-white/90 backdrop-blur">
          {label}
        </span>
      )}
      {showLock && (
        <span
          className="absolute right-2 top-2 grid h-6 w-6 place-items-center rounded-full bg-black/40 text-white/80 backdrop-blur"
          title="Encrypted at rest"
        >
          <Lock className="h-3 w-3" />
        </span>
      )}
    </div>
  );
}
