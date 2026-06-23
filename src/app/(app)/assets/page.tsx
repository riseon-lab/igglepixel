"use client";

import { clsx } from "clsx";
import { Download, Loader2, Lock, Trash2, Upload } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { EncryptedImage } from "@/components/assets/EncryptedImage";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { useEncryption } from "@/lib/crypto/provider";
import { formatBytes, timeAgo } from "@/lib/format";
import {
  deleteAsset,
  fetchCiphertext,
  listAssets,
  uploadAsset,
} from "@/lib/vault/client";
import type { AssetMeta } from "@/lib/vault/types";
import type { AssetKind } from "@/lib/types";

const FILTERS: { label: string; value: AssetKind | "all" }[] = [
  { label: "All", value: "all" },
  { label: "Uploads", value: "upload" },
  { label: "References", value: "reference" },
  { label: "Generated", value: "generated" },
];

const KIND_TONE = {
  upload: "lilac",
  reference: "warning",
  generated: "success",
} as const;

async function imageDimensions(file: File): Promise<{ w: number; h: number }> {
  try {
    const bmp = await createImageBitmap(file);
    const d = { w: bmp.width, h: bmp.height };
    bmp.close();
    return d;
  } catch {
    return { w: 0, h: 0 };
  }
}

export default function AssetsPage() {
  const { client, ready, error } = useEncryption();
  const [assets, setAssets] = useState<AssetMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [filter, setFilter] = useState<AssetKind | "all">("all");
  const [nowMs, setNowMs] = useState(0); // captured on load — avoids Date.now() during render
  const fileInput = useRef<HTMLInputElement>(null);

  // Manual resync (used by event handlers, where synchronous setState is fine).
  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setAssets(await listAssets());
      setNowMs(Date.now());
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load once the encryption key is ready. setState happens only after
  // the await, so it never triggers a synchronous cascade.
  useEffect(() => {
    if (!ready) return;
    let active = true;
    (async () => {
      const data = await listAssets().catch(() => [] as AssetMeta[]);
      if (!active) return;
      setAssets(data);
      setNowMs(Date.now());
      setLoading(false);
    })();
    return () => {
      active = false;
    };
  }, [ready]);

  const visible = assets.filter((a) => filter === "all" || a.kind === filter);

  async function onFiles(files: FileList | null) {
    if (!files?.length || !client) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        const { w, h } = await imageDimensions(file);
        const plain = new Uint8Array(await file.arrayBuffer());
        // Encrypt in the Worker, then upload only ciphertext.
        const ciphertext = await client.encrypt(plain);
        const meta = await uploadAsset(ciphertext, {
          name: file.name,
          kind: "upload",
          mime: file.type || "application/octet-stream",
          width: w,
          height: h,
          size: plain.length,
        });
        setAssets((prev) => [meta, ...prev]);
      }
    } catch (e) {
      console.error("Upload failed", e);
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function remove(id: string) {
    setAssets((prev) => prev.filter((a) => a.id !== id));
    try {
      await deleteAsset(id);
    } catch {
      refresh(); // resync if the delete didn't take
    }
  }

  async function download(a: AssetMeta) {
    if (!client) return;
    const ciphertext = await fetchCiphertext(a.id);
    const plain = await client.decrypt(ciphertext);
    const url = URL.createObjectURL(new Blob([plain as BlobPart], { type: a.mime }));
    const link = document.createElement("a");
    link.href = url;
    link.download = a.name;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Assets"
        description="Uploads, references and generated images — encrypted at rest, decrypted only in your browser."
        actions={
          <Button
            onClick={() => fileInput.current?.click()}
            disabled={!ready || uploading}
          >
            {uploading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Encrypting…
              </>
            ) : (
              <>
                <Upload className="h-4 w-4" /> Upload
              </>
            )}
          </Button>
        }
      />
      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => onFiles(e.target.files)}
      />

      <div className="flex items-center gap-2 text-sm text-text-muted">
        <Lock className="h-3.5 w-3.5 text-lilac" />
        AES-256-GCM · the server only ever stores ciphertext
      </div>

      <div className="flex flex-wrap gap-2">
        {FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => setFilter(f.value)}
            className={clsx(
              "rounded-full px-4 py-2 text-sm font-medium transition-colors",
              filter === f.value
                ? "bg-lilac text-white"
                : "bg-surface text-text-secondary hover:bg-surface-hover",
            )}
          >
            {f.label}
          </button>
        ))}
      </div>

      {error ? (
        <Card className="py-16 text-center text-[#ff8a80]">
          Could not initialise encryption: {error}
        </Card>
      ) : !ready || loading ? (
        <Card className="grid place-items-center py-16 text-text-muted">
          <Loader2 className="h-6 w-6 animate-spin" />
        </Card>
      ) : visible.length === 0 ? (
        <Card className="grid place-items-center py-16 text-center text-text-muted">
          <Upload className="mb-3 h-8 w-8" />
          {assets.length === 0
            ? "No assets yet. Upload an image — it'll be encrypted before it leaves your browser."
            : "No assets match this filter."}
        </Card>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
          {visible.map((a) => (
            <Card key={a.id} padded={false} className="group overflow-hidden">
              <div
                className="relative w-full overflow-hidden bg-background"
                style={{
                  aspectRatio:
                    a.width && a.height ? `${a.width} / ${a.height}` : "1 / 1",
                }}
              >
                {client && (
                  <EncryptedImage
                    id={a.id}
                    mime={a.mime}
                    client={client}
                    alt={a.name}
                    className="h-full w-full object-cover"
                  />
                )}
                <span
                  className="absolute right-2 top-2 grid h-6 w-6 place-items-center rounded-full bg-black/40 text-white/80 backdrop-blur"
                  title="Encrypted at rest"
                >
                  <Lock className="h-3 w-3" />
                </span>
              </div>
              <div className="flex flex-col gap-2 p-3">
                <div className="flex items-center justify-between gap-2">
                  <p className="truncate text-sm font-medium" title={a.name}>
                    {a.name}
                  </p>
                  <Badge tone={KIND_TONE[a.kind]}>{a.kind}</Badge>
                </div>
                <p className="text-xs text-text-muted">
                  {a.width && a.height ? `${a.width}×${a.height} · ` : ""}
                  {formatBytes(a.size)} · {timeAgo(a.createdAt, nowMs)}
                </p>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    className="flex-1"
                    onClick={() => download(a)}
                  >
                    <Download className="h-4 w-4" /> Download
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => remove(a.id)}
                    aria-label="Delete asset"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
