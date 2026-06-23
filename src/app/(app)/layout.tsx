import type { ReactNode } from "react";
import { AppShell } from "@/components/AppShell";
import { AuthGate } from "@/components/AuthGate";
import { Topbar } from "@/components/Topbar";
import { EncryptionProvider } from "@/lib/crypto/provider";

export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <AuthGate>
      <EncryptionProvider>
        <AppShell topbar={<Topbar />}>{children}</AppShell>
      </EncryptionProvider>
    </AuthGate>
  );
}
