import type { ReactNode } from "react";
import { AppShell } from "@/components/AppShell";
import { AuthGate } from "@/components/AuthGate";
import { Topbar } from "@/components/Topbar";
import { EncryptionProvider } from "@/lib/crypto/provider";
import { WorkspaceStoreProvider } from "@/lib/generation/workspace-store";

export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <AuthGate>
      <EncryptionProvider>
        <WorkspaceStoreProvider>
          <AppShell topbar={<Topbar />}>{children}</AppShell>
        </WorkspaceStoreProvider>
      </EncryptionProvider>
    </AuthGate>
  );
}
