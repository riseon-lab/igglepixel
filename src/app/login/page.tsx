"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AuthShell } from "@/components/AuthShell";
import { Button } from "@/components/ui/Button";
import { Input, Label } from "@/components/ui/Field";
import { useSession } from "@/lib/session";

export default function LoginPage() {
  const { ready, hasAccount, authenticated, password: hasKey, login } =
    useSession();
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    if (!ready) return;
    if (!hasAccount) router.replace("/setup");
    // Only leave once fully ready: authenticated AND this tab has the key material.
    else if (authenticated && hasKey) router.replace("/running");
  }, [ready, hasAccount, authenticated, hasKey, router]);

  // Already logged in (valid cookie) but this browser lacks the key → "unlock",
  // not "log in". This is the common case after a restart on a fresh browser.
  const unlocking = authenticated && !hasKey;

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    const err = await login(password);
    if (err) {
      setError(err);
      setSubmitting(false);
      return;
    }
    router.replace("/running");
  }

  async function resetAccount() {
    if (
      !confirm(
        "Reset this local account? Encrypted files stay on disk, but may not decrypt unless you recreate the account with the old password.",
      )
    )
      return;
    setError(null);
    setResetting(true);
    const res = await fetch("/api/auth/reset", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ confirm: "RESET" }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      setError(data.error ?? "Reset failed.");
      setResetting(false);
      return;
    }
    sessionStorage.removeItem("citivia.pw");
    window.location.assign("/setup");
  }

  return (
    <AuthShell
      eyebrow="Citivia Studio"
      title={unlocking ? "Unlock your studio" : "Welcome back"}
      subtitle={
        unlocking
          ? "Enter your password to unlock your encrypted studio on this device."
          : "Enter your password to log in. Logging in here ends any other active session."
      }
    >
      <form onSubmit={onSubmit} className="flex flex-col gap-4">
        <div>
          <Label>Password</Label>
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Your password"
            autoComplete="current-password"
          />
        </div>
        {error && <p className="text-sm text-danger-text">{error}</p>}
        <Button type="submit" className="mt-2 w-full" disabled={submitting}>
          {submitting
            ? unlocking
              ? "Unlocking…"
              : "Logging in…"
            : unlocking
              ? "Unlock"
              : "Log in"}
        </Button>
        <Button
          type="button"
          variant="ghost"
          className="w-full text-danger-text hover:text-danger-text"
          disabled={submitting || resetting}
          onClick={resetAccount}
        >
          {resetting ? "Resetting…" : "Reset local account"}
        </Button>
      </form>
    </AuthShell>
  );
}
