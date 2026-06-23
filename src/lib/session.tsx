"use client";

// Client session, backed by the real server auth API.
//
// The server owns the account and the single active session token (httpOnly
// cookie). This context just mirrors auth status and — because asset decryption
// is client-side — holds the password in sessionStorage so the encryption key
// can be re-derived after a reload. The password is sent only to the auth
// endpoints over POST; it's never persisted server-side in the clear.

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
  login: (username: string, password: string) => Promise<string | null>;
  logout: () => Promise<void>;
}

const PW_KEY = "citivia.pw";

const Ctx = createContext<SessionState | null>(null);

interface Status {
  hasAccount: boolean;
  authenticated: boolean;
  username: string | null;
}

async function fetchStatus(): Promise<Status> {
  try {
    const res = await fetch("/api/auth/session", { cache: "no-store" });
    if (!res.ok) throw new Error();
    return await res.json();
  } catch {
    return { hasAccount: false, authenticated: false, username: null };
  }
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState<Status>({
    hasAccount: false,
    authenticated: false,
    username: null,
  });
  const [password, setPassword] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    (async () => {
      const pw = sessionStorage.getItem(PW_KEY);
      const s = await fetchStatus();
      if (!active) return;
      setStatus(s);
      setPassword(pw);
      setReady(true);
    })();
    return () => {
      active = false;
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
      return data.error ?? "Setup failed.";
    }
    const data = await res.json();
    sessionStorage.setItem(PW_KEY, pw);
    setStatus({ hasAccount: true, authenticated: true, username: data.username });
    setPassword(pw);
    return null;
  }, []);

  const login = useCallback(async (username: string, pw: string) => {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ username, password: pw }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      return data.error ?? "Login failed.";
    }
    const data = await res.json();
    sessionStorage.setItem(PW_KEY, pw);
    setStatus({ hasAccount: true, authenticated: true, username: data.username });
    setPassword(pw);
    return null;
  }, []);

  const logout = useCallback(async () => {
    await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
    sessionStorage.removeItem(PW_KEY);
    setStatus((s) => ({ ...s, authenticated: false, username: null }));
    setPassword(null);
  }, []);

  return (
    <Ctx.Provider
      value={{
        ready,
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
