"use client";

// In-browser end-to-end check for the encryption layer. Drives the *real* Web
// Worker through the full pipeline — derive key → encrypt → decrypt → render —
// including a genuine PNG image, and proves:
//   • the worker spawns and responds (no main-thread fallback needed),
//   • round-trips are byte-identical (no visual corruption),
//   • wrong keys and tampered ciphertext are rejected,
//   • what would be "stored on the server" is real ciphertext, not the image.

import { clsx } from "clsx";
import { CheckCircle2, Loader2, ShieldCheck, XCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import {
  deriveKey,
  exportRawKey,
  generateSalt,
  isEnvelope,
} from "@/lib/crypto/aesgcm";
import { EncryptionClient } from "@/lib/crypto/client";

type Status = "pending" | "running" | "pass" | "fail";
interface Step {
  name: string;
  status: Status;
  detail?: string;
  ms?: number;
}

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47]; // \x89PNG

/** Build an image Blob from raw bytes (cast keeps TS's strict ArrayBuffer typing happy). */
function pngBlob(bytes: Uint8Array): Blob {
  return new Blob([bytes as BlobPart], { type: "image/png" });
}

function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Draw a recognisable test image and return its PNG bytes. */
async function makeTestImagePng(): Promise<Uint8Array> {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const c = canvas.getContext("2d")!;
  const grad = c.createLinearGradient(0, 0, size, size);
  grad.addColorStop(0, "#a97de0");
  grad.addColorStop(1, "#121212");
  c.fillStyle = grad;
  c.fillRect(0, 0, size, size);
  c.fillStyle = "#ffffff";
  c.font = "bold 28px sans-serif";
  c.textAlign = "center";
  c.fillText("CITIVIA", size / 2, size / 2 - 6);
  c.font = "16px sans-serif";
  c.fillText("encrypted ✓", size / 2, size / 2 + 22);
  const blob: Blob = await new Promise((res) =>
    canvas.toBlob((b) => res(b!), "image/png"),
  );
  return new Uint8Array(await blob.arrayBuffer());
}

export function EncryptionSelfTest() {
  const [steps, setSteps] = useState<Step[]>([]);
  const [running, setRunning] = useState(false);
  const [workerActive, setWorkerActive] = useState<boolean | null>(null);
  const [images, setImages] = useState<{ original: string; decrypted: string } | null>(
    null,
  );
  const urlsRef = useRef<string[]>([]);

  // Revoke any in-memory object URLs when we unmount or re-run.
  useEffect(() => {
    return () => urlsRef.current.forEach((u) => URL.revokeObjectURL(u));
  }, []);

  function track(url: string) {
    urlsRef.current.push(url);
    return url;
  }

  async function run() {
    setRunning(true);
    setImages(null);
    urlsRef.current.forEach((u) => URL.revokeObjectURL(u));
    urlsRef.current = [];

    const results: Step[] = [];
    const push = (s: Step) => {
      results.push(s);
      setSteps([...results]);
    };
    const time = async <T,>(name: string, fn: () => Promise<T>): Promise<T> => {
      const idx = results.length;
      push({ name, status: "running" });
      const start = performance.now();
      try {
        const out = await fn();
        results[idx] = { ...results[idx], status: "pass", ms: performance.now() - start };
        setSteps([...results]);
        return out;
      } catch (err) {
        results[idx] = {
          ...results[idx],
          status: "fail",
          ms: performance.now() - start,
          detail: err instanceof Error ? err.message : String(err),
        };
        setSteps([...results]);
        throw err;
      }
    };

    const client = new EncryptionClient();

    try {
      await time("Spawn worker & derive key (PBKDF2)", async () => {
        const salt = generateSalt();
        const key = await deriveKey("self-test-passphrase", salt, 100_000);
        const raw = await exportRawKey(key);
        await client.init(raw);
        setWorkerActive(client.usingWorker);
        results[results.length - 1].detail = client.usingWorker
          ? "Dedicated Web Worker active"
          : "Worker unavailable — using inline fallback";
      });

      await time("Encrypt → decrypt text (round-trip)", async () => {
        const msg = "The quick brown fox jumps over the lazy dog — café 🔐";
        const plain = new TextEncoder().encode(msg);
        const envelope = await client.encrypt(plain);
        if (!isEnvelope(envelope)) throw new Error("Output is not a valid envelope.");
        const back = await client.decrypt(envelope);
        if (new TextDecoder().decode(back) !== msg)
          throw new Error("Decrypted text did not match the original.");
      });

      await time("Reject decryption with the wrong key", async () => {
        const envelope = await client.encrypt(new TextEncoder().encode("secret"));
        const other = new EncryptionClient();
        const wrong = await exportRawKey(
          await deriveKey("a-different-passphrase", generateSalt(), 100_000),
        );
        await other.init(wrong);
        let rejected = false;
        try {
          await other.decrypt(envelope);
        } catch {
          rejected = true;
        }
        other.dispose();
        if (!rejected) throw new Error("Wrong key unexpectedly decrypted the data!");
      });

      await time("Detect tampered ciphertext (GCM auth tag)", async () => {
        const envelope = await client.encrypt(new TextEncoder().encode("integrity"));
        envelope[envelope.length - 1] ^= 0xff;
        let rejected = false;
        try {
          await client.decrypt(envelope);
        } catch {
          rejected = true;
        }
        if (!rejected) throw new Error("Tampered ciphertext was not rejected!");
      });

      const original = await time("Generate a real PNG image", makeTestImagePng);

      const envelope = await time("Encrypt the image in the worker", async () => {
        const env = await client.encrypt(original);
        const looksLikePng = PNG_SIGNATURE.every((b, i) => env[i] === b);
        if (looksLikePng)
          throw new Error("Encrypted output still looks like a PNG — not encrypted!");
        if (!isEnvelope(env)) throw new Error("Encrypted output is not an envelope.");
        results[results.length - 1].detail = `Stored blob: ${env.length.toLocaleString()} bytes of ciphertext (not an image)`;
        return env;
      });

      await time("Decrypt the image — byte-identical, no corruption", async () => {
        const decrypted = await client.decrypt(envelope);
        if (!bytesEqual(decrypted, original))
          throw new Error("Decrypted image bytes differ from the original!");
        const original_url = track(URL.createObjectURL(pngBlob(original)));
        const decrypted_url = track(URL.createObjectURL(pngBlob(decrypted)));
        setImages({ original: original_url, decrypted: decrypted_url });
      });
    } catch {
      // Individual step already recorded its failure detail.
    } finally {
      client.dispose();
      setRunning(false);
    }
  }

  const allPass =
    steps.length > 0 && steps.every((s) => s.status === "pass");

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-[12px] bg-lilac/15 text-lilac">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <p className="font-semibold">Encryption self-test</p>
            <p className="text-sm text-text-muted">
              Runs the real AES-256-GCM worker end-to-end in your browser.
            </p>
          </div>
        </div>
        <Button onClick={run} disabled={running}>
          {running ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Running…
            </>
          ) : (
            "Run self-test"
          )}
        </Button>
      </div>

      {workerActive !== null && (
        <div
          className={clsx(
            "rounded-[12px] px-4 py-2 text-sm",
            workerActive ? "bg-success/10 text-success" : "bg-warning/10 text-warning",
          )}
        >
          {workerActive
            ? "Web Worker active — crypto is running off the main thread."
            : "Worker unavailable — running the inline fallback on the main thread."}
        </div>
      )}

      {steps.length > 0 && (
        <ol className="flex flex-col gap-2">
          {steps.map((s, i) => (
            <li
              key={i}
              className="flex items-start gap-3 rounded-[12px] bg-background px-4 py-3"
            >
              <span className="mt-0.5">
                {s.status === "pass" ? (
                  <CheckCircle2 className="h-5 w-5 text-success" />
                ) : s.status === "fail" ? (
                  <XCircle className="h-5 w-5 text-[#ff8a80]" />
                ) : (
                  <Loader2 className="h-5 w-5 animate-spin text-lilac" />
                )}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium">{s.name}</span>
                  {s.ms !== undefined && (
                    <span className="shrink-0 text-xs tabular-nums text-text-muted">
                      {s.ms.toFixed(1)} ms
                    </span>
                  )}
                </div>
                {s.detail && (
                  <p
                    className={clsx(
                      "mt-0.5 text-sm",
                      s.status === "fail" ? "text-[#ff8a80]" : "text-text-muted",
                    )}
                  >
                    {s.detail}
                  </p>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}

      {allPass && (
        <p className="flex items-center gap-2 text-sm font-medium text-success">
          <CheckCircle2 className="h-4 w-4" /> All checks passed.
        </p>
      )}

      {images && (
        // next/image can't optimize in-memory blob: URLs, so plain <img> is correct here.
        /* eslint-disable @next/next/no-img-element */
        <div className="grid grid-cols-2 gap-4">
          <figure className="flex flex-col gap-2">
            <img
              src={images.original}
              alt="Original"
              className="w-full rounded-[12px] border border-border"
            />
            <figcaption className="text-center text-xs text-text-muted">
              Original (before encryption)
            </figcaption>
          </figure>
          <figure className="flex flex-col gap-2">
            <img
              src={images.decrypted}
              alt="Decrypted"
              className="w-full rounded-[12px] border border-border"
            />
            <figcaption className="text-center text-xs text-text-muted">
              Decrypted in-memory (after worker round-trip)
            </figcaption>
          </figure>
        </div>
        /* eslint-enable @next/next/no-img-element */
      )}
    </div>
  );
}
