"use client";

// Routes the user to setup (no account), or login (logged out, or this tab lacks
// the password needed to derive the decryption key). The server enforces the
// actual session on every API call; this is just navigation.

import { useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";
import { useSession } from "@/lib/session";

export function AuthGate({ children }: { children: ReactNode }) {
  const { ready, hasAccount, authenticated, password } = useSession();
  const router = useRouter();

  const needsKey = authenticated && !password;
  const blocked = !hasAccount || !authenticated || !password;

  useEffect(() => {
    if (!ready) return;
    if (!hasAccount) router.replace("/setup");
    else if (!authenticated || needsKey) router.replace("/login");
  }, [ready, hasAccount, authenticated, needsKey, router]);

  if (!ready || blocked) {
    return (
      <div className="grid min-h-dvh place-items-center bg-background text-text-muted">
        <div className="flex items-center gap-3">
          <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-lilac" />
          Loading studio…
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
