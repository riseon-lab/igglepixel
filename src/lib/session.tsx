"use client";

// Client session, backed by the real server auth API.
//
// The server owns the account and the single active session token (httpOnly
// cookie). This context mirrors auth status and — because asset decryption is
// client-side — holds the password in sessionStorage so the encryption key can be
// re-derived after a reload. The password is POSTed only to the auth endpoints;
// it is never persisted server-side in the clear, and deliberately never written
// to durable browser storage. A fresh browser must re-enter it to unlock — that
// is the cost of real client-side encryption, not a bug to "fix" by stashing the
// key somewhere persistent.
//
// Restart-hardening: until we have actually reached the server once, we never
// guess the auth state. A pod restart (especially with build-on-boot) leaves the
// server unreachable for a while; treating that as "no account" is what used to
// dump people into setup and then fail with "an account already exists".

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

interface SessionState {
  ready: boolean;
  hasAccount: boolean;
  authenticated: boolean;
  username: string | null;
  /** Present only when this tab has the password available for key derivation. */
  password: string | null;
  createAccount: (username: string, password: string) => Promise<string | null>;
  /** Password-only: single-account deployment, so a username is never required. */
  login: (password: string) => Promise<string | null>;
  logout: () => Promise<void>;
}

const PW_KEY = "citivia.pw";

const Ctx = createContext<SessionState | null>(null);

interface Status {
  hasAccount: boolean;
  authenticated: boolean;
  username: string | null;
  corrupt: boolean;
}

const EMPTY: Status = {
  hasAccount: false,
  authenticated: false,
  username: null,
  corrupt: false,
};

type Probe = { reachable: true; status: Status } | { reachable: false };

async function probe(): Promise<Probe> {
  try {
    const res = await fetch("/api/auth/session", { cache: "no-store" });
    if (!res.ok) return { reachable: false }; // booting / proxy 502 / 5xx
    const body = await res.json();
    return {
      reachable: true,
      status: {
        hasAccount: !!body.hasAccount,
        authenticated: !!body.authenticated,
        username: body.username ?? null,
        corrupt: !!body.corrupt,
      },
    };
  } catch {
    return { reachable: false }; // network down — pod is likely mid-restart
  }
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>(EMPTY);
  const [password, setPassword] = useState<string | null>(null);
  const [reachable, setReachable] = useState(true);
  // "Have we reached the server at least once this mount?" Until we have, the auth
  // state is unknown and we must not route anywhere.
  const [contacted, setContacted] = useState(false);

  // Poll the session endpoint: back off while unreachable, then settle into a slow
  // heartbeat so a later restart re-surfaces the reconnecting screen.
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    let loadedKey = false;

    const tick = async (delay: number) => {
      const p = await probe();
      if (!active) return;
      if (!loadedKey) {
        // Read the in-tab key material once, after mount — sessionStorage is
        // client-only, and doing it past the await keeps it out of the synchronous
        // effect body. The UI is gated on `contacted` until the first probe, so
        // the password is always in place before any authed route renders.
        loadedKey = true;
        setPassword(sessionStorage.getItem(PW_KEY));
      }
      if (p.reachable) {
        setStatus(p.status);
        setReachable(true);
        setContacted(true);
        timer = setTimeout(() => tick(15000), 15000);
      } else {
        setReachable(false);
        const next = Math.min(delay * 1.5, 5000);
        timer = setTimeout(() => tick(next), next);
      }
    };
    tick(1000);

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);

  const createAccount = useCallback(async (username: string, pw: string) => {
    const res = await fetch("/api/auth/setup", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ username, password: pw }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      if (res.status === 409) {
        // An account already exists (e.g. a restart race won by the server).
        // Reflect reality so the UI routes to login instead of looping on setup.
        setStatus((s) => ({ ...s, hasAccount: true, corrupt: false }));
      }
      return data.error ?? "Setup failed.";
    }
    const data = await res.json();
    sessionStorage.setItem(PW_KEY, pw);
    setStatus({
      hasAccount: true,
      authenticated: true,
      username: data.username,
      corrupt: false,
    });
    setPassword(pw);
    return null;
  }, []);

  const login = useCallback(async (pw: string) => {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ password: pw }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      return data.error ?? "Login failed.";
    }
    const data = await res.json();
    sessionStorage.setItem(PW_KEY, pw);
    setStatus({
      hasAccount: true,
      authenticated: true,
      username: data.username,
      corrupt: false,
    });
    setPassword(pw);
    return null;
  }, []);

  const logout = useCallback(async () => {
    await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
    sessionStorage.removeItem(PW_KEY);
    setStatus((s) => ({ ...s, authenticated: false, username: null }));
    setPassword(null);
  }, []);

  // First contact not yet made (initial load, or page reopened during a pod
  // restart / cold build): show a connecting screen, never a guess.
  if (!contacted) return <BootScreen reachable={reachable} />;
  // Present-but-unreadable account: offer recovery, not the setup dead-end.
  if (status.corrupt) return <CorruptScreen />;

  return (
    <Ctx.Provider
      value={{
        ready: true,
        hasAccount: status.hasAccount,
        authenticated: status.authenticated,
        username: status.username,
        password,
        createAccount,
        login,
        logout,
      }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useSession(): SessionState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useSession must be used within SessionProvider");
  return ctx;
}

function BootScreen({ reachable }: { reachable: boolean }) {
  return (
    <div className="grid min-h-dvh place-items-center bg-background px-6 text-center text-text-muted">
      <div className="flex max-w-sm flex-col items-center gap-3">
        <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-lilac" />
        {reachable ? (
          <p>Loading studio…</p>
        ) : (
          <>
            <p className="font-medium text-text-secondary">
              Reconnecting to your studio…
            </p>
            <p className="text-sm">
              Your pod may still be starting up — this can take a minute after a
              restart. We&apos;ll continue automatically.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

function CorruptScreen() {
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function reset() {
    setWorking(true);
    setError(null);
    const res = await fetch("/api/auth/reset", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ confirm: "RESET" }),
    }).catch(() => null);
    if (!res || !res.ok) {
      setError("Couldn't reset the account. Try again.");
      setWorking(false);
      return;
    }
    sessionStorage.removeItem(PW_KEY);
    window.location.assign("/setup");
  }

  return (
    <div className="grid min-h-dvh place-items-center bg-background px-6">
      <div className="w-full max-w-md rounded-[24px] border border-border bg-surface p-6 text-center sm:p-8">
        <h1 className="text-xl font-bold text-danger-text">
          Account data is damaged
        </h1>
        <p className="mt-2 text-text-secondary">
          This studio&apos;s account file exists but can&apos;t be read — usually a
          pod that stopped mid-write. The login can&apos;t be recovered, and any
          encrypted assets tied to it won&apos;t decrypt. Resetting clears it so you
          can set up a fresh account.
        </p>
        {error && <p className="mt-3 text-sm text-danger-text">{error}</p>}
        <button
          type="button"
          onClick={reset}
          disabled={working}
          className="mt-5 w-full rounded-[10px] bg-lilac px-4 py-3 font-semibold text-white disabled:opacity-60"
        >
          {working ? "Resetting…" : "Reset & start over"}
        </button>
      </div>
    </div>
  );
}
