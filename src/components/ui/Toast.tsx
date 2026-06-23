"use client";

// Lightweight global toast system. Mounted once near the root; any client
// component calls `useToast()` to surface success/error/info/warning feedback.
// Matches the design system (dark surface, lilac/success/danger/warning tones)
// and sits clear of the mobile bottom nav.

import { clsx } from "clsx";
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

export type ToastTone = "success" | "error" | "warning" | "info";

interface Toast {
  id: number;
  tone: ToastTone;
  title: string;
  description?: string;
}

interface ToastApi {
  show: (tone: ToastTone, title: string, description?: string) => void;
  success: (title: string, description?: string) => void;
  error: (title: string, description?: string) => void;
  warning: (title: string, description?: string) => void;
  info: (title: string, description?: string) => void;
  dismiss: (id: number) => void;
}

const ToastCtx = createContext<ToastApi | null>(null);

const TONES: Record<
  ToastTone,
  { icon: typeof Info; ring: string; iconColor: string }
> = {
  success: { icon: CheckCircle2, ring: "border-success/40", iconColor: "text-success" },
  error: { icon: XCircle, ring: "border-danger/50", iconColor: "text-[#ff8a80]" },
  warning: { icon: AlertTriangle, ring: "border-warning/40", iconColor: "text-warning" },
  info: { icon: Info, ring: "border-lilac/40", iconColor: "text-lilac" },
};

// Errors linger longer so they can be read; others auto-dismiss quickly.
const DURATION: Record<ToastTone, number> = {
  success: 4000,
  info: 4000,
  warning: 6000,
  error: 8000,
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const idRef = useRef(0);
  const timers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const show = useCallback(
    (tone: ToastTone, title: string, description?: string) => {
      const id = ++idRef.current;
      setToasts((prev) => [...prev, { id, tone, title, description }]);
      const timer = setTimeout(() => dismiss(id), DURATION[tone]);
      timers.current.set(id, timer);
    },
    [dismiss],
  );

  // Clean up any pending timers on unmount.
  useEffect(() => {
    const map = timers.current;
    return () => map.forEach((t) => clearTimeout(t));
  }, []);

  const api: ToastApi = {
    show,
    success: (t, d) => show("success", t, d),
    error: (t, d) => show("error", t, d),
    warning: (t, d) => show("warning", t, d),
    info: (t, d) => show("info", t, d),
    dismiss,
  };

  return (
    <ToastCtx.Provider value={api}>
      {children}
      <Toaster toasts={toasts} onDismiss={dismiss} />
    </ToastCtx.Provider>
  );
}

function Toaster({
  toasts,
  onDismiss,
}: {
  toasts: Toast[];
  onDismiss: (id: number) => void;
}) {
  return (
    <div
      // Below the topbar; offset above the mobile bottom nav. Non-interactive
      // wrapper so it never blocks clicks behind it.
      className="pointer-events-none fixed inset-x-0 top-20 z-[60] flex flex-col items-center gap-3 px-4 sm:inset-x-auto sm:right-6 sm:items-end"
      role="region"
      aria-live="polite"
      aria-label="Notifications"
    >
      {toasts.map((t) => {
        const { icon: Icon, ring, iconColor } = TONES[t.tone];
        return (
          <div
            key={t.id}
            className={clsx(
              "animate-toast-in pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-[12px] border bg-surface/95 p-4 shadow-lg shadow-black/40 backdrop-blur",
              ring,
            )}
            role={t.tone === "error" ? "alert" : "status"}
          >
            <Icon className={clsx("mt-0.5 h-5 w-5 shrink-0", iconColor)} />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold leading-tight">{t.title}</p>
              {t.description && (
                <p className="mt-1 break-words text-sm text-text-secondary">
                  {t.description}
                </p>
              )}
            </div>
            <button
              onClick={() => onDismiss(t.id)}
              aria-label="Dismiss"
              className="-m-1 shrink-0 rounded-md p-1 text-text-muted transition-colors hover:bg-surface-hover hover:text-white"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        );
      })}
    </div>
  );
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastCtx);
  if (!ctx) throw new Error("useToast must be used within a ToastProvider");
  return ctx;
}
