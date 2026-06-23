"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Input, Label } from "@/components/ui/Field";
import { useSession } from "@/lib/session";
import { AuthShell } from "../setup/page";

export default function LoginPage() {
  const { ready, hasAccount, authenticated, password: hasKey, login } =
    useSession();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!ready) return;
    if (!hasAccount) router.replace("/setup");
    // Only leave if fully ready: authenticated AND this tab has the key material.
    else if (authenticated && hasKey) router.replace("/running");
  }, [ready, hasAccount, authenticated, hasKey, router]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    const err = await login(username, password);
    if (err) {
      setError(err);
      setSubmitting(false);
      return;
    }
    router.replace("/running");
  }

  return (
    <AuthShell
      eyebrow="Citivia Studio"
      title="Welcome back"
      subtitle="Log in to your studio. Logging in here will end any other active session."
    >
      <form onSubmit={onSubmit} className="flex flex-col gap-4">
        <div>
          <Label>Username</Label>
          <Input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="username"
            autoComplete="username"
          />
        </div>
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
          {submitting ? "Logging in…" : "Log in"}
        </Button>
      </form>
    </AuthShell>
  );
}
