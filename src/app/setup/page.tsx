"use client";

import { Wand2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
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

export function AuthShell({
  eyebrow,
  title,
  subtitle,
  children,
}: {
  eyebrow: string;
  title: string;
  subtitle: string;
  children: React.ReactNode;
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
