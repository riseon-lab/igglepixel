"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AuthShell } from "@/components/AuthShell";
import { Button } from "@/components/ui/Button";
import { Input, Label } from "@/components/ui/Field";
import { useSession } from "@/lib/session";

export default function SetupPage() {
  const { ready, hasAccount, createAccount } = useSession();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // If an account already exists, setup is done — go to login.
  useEffect(() => {
    if (ready && hasAccount) router.replace("/login");
  }, [ready, hasAccount, router]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    setSubmitting(true);
    const err = await createAccount(username, password);
    if (err) {
      setError(err);
      setSubmitting(false);
      return;
    }
    router.replace("/running");
  }

  return (
    <AuthShell
      eyebrow="Welcome"
      title="Set up Citivia Studio"
      subtitle="Create the account that will secure this deployment. You can change these later in Settings."
    >
      <form onSubmit={onSubmit} className="flex flex-col gap-4">
        <div>
          <Label>Username</Label>
          <Input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="studio-admin"
            autoComplete="username"
          />
        </div>
        <div>
          <Label>Password</Label>
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="At least 8 characters"
            autoComplete="new-password"
          />
        </div>
        <div>
          <Label>Confirm password</Label>
          <Input
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            placeholder="Re-enter password"
            autoComplete="new-password"
          />
        </div>
        {error && <p className="text-sm text-danger-text">{error}</p>}
        <Button type="submit" className="mt-2 w-full" disabled={submitting}>
          {submitting ? "Creating…" : "Create account & enter studio"}
        </Button>
      </form>
    </AuthShell>
  );
}
