import { clsx } from "clsx";
import { Lock } from "lucide-react";

export function PreviewTile({
  width,
  height,
  className,
  showLock = true,
  label,
  src,
}: {
  width: number;
  height: number;
  className?: string;
  showLock?: boolean;
  label?: string;
  src?: string;
}) {
  return (
    <div
      className={clsx(
        "relative overflow-hidden bg-background",
        className,
      )}
      style={{ aspectRatio: `${width} / ${height}` }}
    >
      {src && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt="" className="h-full w-full object-cover" />
      )}
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
