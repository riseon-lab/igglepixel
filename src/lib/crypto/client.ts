// Main-thread client for the encryption Worker.
//
// Spawns the Worker, derives/holds the session key, and exposes a small
// promise-based encrypt/decrypt API. If Workers are unavailable (e.g. during
// SSR or in a constrained runtime) it transparently falls back to running the
// same crypto on the calling thread, so callers never have to branch.

import {
  decrypt as inlineDecrypt,
  encrypt as inlineEncrypt,
  exportRawKey,
  importRawKey,
} from "./aesgcm";
import type { WorkerRequest, WorkerResponse } from "./worker";

type Pending = {
  resolve: (data: ArrayBuffer | undefined) => void;
  reject: (err: Error) => void;
};

export class EncryptionClient {
  private worker: Worker | null = null;
  private pending = new Map<number, Pending>();
  private nextId = 1;
  private inlineKey: CryptoKey | null = null; // used only in the fallback path

  /** True when a real Worker is backing this client (vs. the inline fallback). */
  get usingWorker(): boolean {
    return this.worker !== null;
  }

  /** Initialise with a raw 256-bit key. Safe to call again to rotate the key. */
  async init(keyRaw: Uint8Array): Promise<void> {
    this.dispose();

    if (typeof Worker !== "undefined") {
      try {
        this.worker = new Worker(new URL("./worker.ts", import.meta.url), {
          type: "module",
        });
        this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
          const res = e.data;
          const p = this.pending.get(res.id);
          if (!p) return;
          this.pending.delete(res.id);
          if (res.ok) p.resolve(res.data);
          else p.reject(new Error(res.error));
        };
        this.worker.onerror = (e) => {
          // Surface worker-load failures to every in-flight request.
          for (const [, p] of this.pending) {
            p.reject(new Error(e.message || "Encryption worker error."));
          }
          this.pending.clear();
        };
        // Send the key once; copy the buffer so the caller's bytes aren't detached.
        await this.send("init", keyRaw.slice().buffer);
        return;
      } catch {
        // Fall through to the inline path below.
        this.worker = null;
      }
    }

    this.inlineKey = await importRawKey(keyRaw);
  }

  async encrypt(plaintext: Uint8Array): Promise<Uint8Array> {
    if (this.worker) {
      const out = await this.send("encrypt", plaintext.slice().buffer);
      return new Uint8Array(out!);
    }
    if (!this.inlineKey) throw new Error("EncryptionClient not initialised.");
    return inlineEncrypt(this.inlineKey, plaintext);
  }

  async decrypt(envelope: Uint8Array): Promise<Uint8Array> {
    if (this.worker) {
      const out = await this.send("decrypt", envelope.slice().buffer);
      return new Uint8Array(out!);
    }
    if (!this.inlineKey) throw new Error("EncryptionClient not initialised.");
    return inlineDecrypt(this.inlineKey, envelope);
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    this.inlineKey = null;
    for (const [, p] of this.pending) p.reject(new Error("Client disposed."));
    this.pending.clear();
  }

  private send(
    op: WorkerRequest["op"],
    buffer: ArrayBuffer,
  ): Promise<ArrayBuffer | undefined> {
    const id = this.nextId++;
    const worker = this.worker!;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const msg = { id, op, ...(op === "init" ? { keyRaw: buffer } : { data: buffer }) };
      worker.postMessage(msg, [buffer]);
    });
  }
}

/** Convenience: a one-off client seeded with a random ephemeral key (for self-tests). */
export async function createEphemeralClient(): Promise<EncryptionClient> {
  const { generateKey } = await import("./aesgcm");
  const key = await generateKey();
  const raw = await exportRawKey(key);
  const client = new EncryptionClient();
  await client.init(raw);
  return client;
}
