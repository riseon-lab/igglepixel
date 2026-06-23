// Dedicated Web Worker that performs all AES-256-GCM work off the main thread,
// so encrypting/decrypting large images never blocks the UI. Decrypted bytes
// exist only here and in the page's memory — nothing is written to disk
// (plan.md "Encryption": "Assets decrypted only in memory").

import { decrypt, encrypt, importRawKey } from "./aesgcm";

// Minimal worker-scope typing that avoids pulling in the full "webworker" lib
// (which conflicts with the "dom" lib used elsewhere in the app).
interface WorkerCtx {
  onmessage: ((e: MessageEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
}
const ctx = self as unknown as WorkerCtx;

export type WorkerRequest =
  | { id: number; op: "init"; keyRaw: ArrayBuffer }
  | { id: number; op: "encrypt"; data: ArrayBuffer }
  | { id: number; op: "decrypt"; data: ArrayBuffer };

export type WorkerResponse =
  | { id: number; ok: true; data?: ArrayBuffer }
  | { id: number; ok: false; error: string };

// The key is sent once via `init` and never leaves the worker afterwards.
let key: CryptoKey | null = null;

ctx.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.op === "init") {
      key = await importRawKey(new Uint8Array(msg.keyRaw));
      ctx.postMessage({ id: msg.id, ok: true } satisfies WorkerResponse);
      return;
    }

    if (!key) throw new Error("Worker not initialised — send `init` first.");

    const input = new Uint8Array(msg.data);
    const result =
      msg.op === "encrypt"
        ? await encrypt(key, input)
        : await decrypt(key, input);

    // Freshly allocated, non-shared buffer — safe to treat as a transferable ArrayBuffer.
    const buffer = result.buffer as ArrayBuffer;
    // Transfer the result buffer (zero-copy) back to the main thread.
    ctx.postMessage(
      { id: msg.id, ok: true, data: buffer } satisfies WorkerResponse,
      [buffer],
    );
  } catch (err) {
    ctx.postMessage({
      id: msg.id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    } satisfies WorkerResponse);
  }
};
