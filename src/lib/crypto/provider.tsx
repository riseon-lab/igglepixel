"use client";

// Owns the live, Worker-backed encryption key for the signed-in session.
//
// On login it fetches the account's PBKDF2 salt from the server, derives the
// AES-256-GCM key from the password, and hands it to a single EncryptionClient
// shared across the app. The key lives only in memory (and inside the Worker).

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { useSession } from "@/lib/session";
import { fetchSalt } from "@/lib/vault/client";
import { deriveKey, exportRawKey } from "./aesgcm";
import { EncryptionClient } from "./client";

interface EncryptionState {
  client: EncryptionClient | null;
  ready: boolean;
  error: string | null;
}

const Ctx = createContext<EncryptionState>({
  client: null,
  ready: false,
  error: null,
});

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function EncryptionProvider({ children }: { children: ReactNode }) {
  const { authenticated, password } = useSession();
  const [state, setState] = useState<EncryptionState>({
    client: null,
    ready: false,
    error: null,
  });

  useEffect(() => {
    // Needs both a valid session (for the salt endpoint) and the password (held
    // in this tab) to derive the key. Unmounts on logout, disposing the client.
    if (!authenticated || !password) return;

    let cancelled = false;
    let client: EncryptionClient | null = null;

    (async () => {
      try {
        const salt = base64ToBytes(await fetchSalt());
        const key = await deriveKey(password, salt);
        const raw = await exportRawKey(key);
        client = new EncryptionClient();
        await client.init(raw);
        if (cancelled) {
          client.dispose();
          return;
        }
        setState({ client, ready: true, error: null });
      } catch (err) {
        if (!cancelled)
          setState({
            client: null,
            ready: false,
            error: err instanceof Error ? err.message : "Encryption setup failed.",
          });
      }
    })();

    return () => {
      cancelled = true;
      client?.dispose();
    };
    // Re-derive when auth status or the available password changes.
  }, [authenticated, password]);

  return <Ctx.Provider value={state}>{children}</Ctx.Provider>;
}

export function useEncryption(): EncryptionState {
  return useContext(Ctx);
}
