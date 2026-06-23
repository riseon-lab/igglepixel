"use client";

import { Layers, Link2, Trash2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Input } from "@/components/ui/Field";
import { PageHeader } from "@/components/ui/PageHeader";
import { useToast } from "@/components/ui/Toast";
import { formatBytes, timeAgo } from "@/lib/format";
import type { Lora, LoraSource } from "@/lib/types";

const SOURCE_LABEL: Record<LoraSource, string> = {
  civitai: "Civitai",
  huggingface: "Hugging Face",
  upload: "Uploaded",
};

export default function LorasPage() {
  const toast = useToast();
  const fileInput = useRef<HTMLInputElement>(null);
  const [loras, setLoras] = useState<Lora[]>([]);
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    const res = await fetch("/api/loras", { cache: "no-store" });
    if (!res.ok) throw new Error("Could not load LoRAs.");
    setLoras(await res.json());
  }

  useEffect(() => {
    fetch("/api/loras", { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error("Could not load LoRAs.");
        return res.json();
      })
      .then(setLoras)
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load LoRAs."))
      .finally(() => setLoading(false));
  }, []);

  async function addFromUrl() {
    if (!url.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/loras", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ url: url.trim() }),
      });
      if (!res.ok) throw new Error((await res.json()).error ?? "LoRA install failed.");
      setUrl("");
      await refresh();
      toast.success("LoRA installed", "Saved to the shared runner LoRA folder.");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "LoRA install failed.";
      setError(msg);
      toast.error("LoRA install failed", msg);
    } finally {
      setBusy(false);
    }
  }

  async function upload(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const body = new FormData();
      body.set("file", file);
      const res = await fetch("/api/loras", { method: "POST", body });
      if (!res.ok) throw new Error((await res.json()).error ?? "LoRA upload failed.");
      await refresh();
      toast.success("LoRA uploaded", file.name);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "LoRA upload failed.";
      setError(msg);
      toast.error("LoRA upload failed", msg);
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function remove(id: string) {
    setError(null);
    const res = await fetch(`/api/loras/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      setError("Could not delete LoRA.");
      toast.error("Could not delete LoRA", "Please try again.");
      return;
    }
    setLoras((prev) => prev.filter((l) => l.id !== id));
    toast.success("LoRA deleted");
  }

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="LoRAs"
        description="Install and manage LoRAs from Civitai, Hugging Face, or direct upload."
        actions={
          <Button variant="secondary" onClick={() => fileInput.current?.click()} disabled={busy}>
            <Upload className="h-4 w-4" /> Upload LoRA
          </Button>
        }
      />
      <input
        ref={fileInput}
        type="file"
        accept=".safetensors,.pt,.ckpt,.bin"
        hidden
        onChange={(e) => upload(e.target.files?.[0])}
      />

      <Card className="flex flex-col gap-3">
        <label className="text-base font-medium">
          Download from Civitai or Hugging Face
        </label>
        <div className="flex flex-col gap-3 sm:flex-row">
          <div className="relative flex-1">
            <Link2 className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://civitai.com/models/…  or  https://huggingface.co/…"
              className="pl-10"
              onKeyDown={(e) => e.key === "Enter" && addFromUrl()}
            />
          </div>
          <Button onClick={addFromUrl} disabled={busy || !url.trim()}>
            {busy ? "Installing..." : "Install"}
          </Button>
        </div>
        <p className="text-xs text-text-muted">
          API keys are configured in Settings. Files are saved to the shared runner LoRA folder.
        </p>
        {error && <p className="text-sm text-danger">{error}</p>}
      </Card>

      <section className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h2 className="text-[20px] font-semibold">Installed</h2>
          <span className="text-sm text-text-muted">
            {loading ? "Loading..." : `${loras.length} LoRAs`}
          </span>
        </div>
        <div className="grid gap-4">
          {!loading && loras.length === 0 && (
            <Card className="text-sm text-text-muted">No installed LoRAs.</Card>
          )}
          {loras.map((l) => (
            <Card key={l.id} className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-start gap-4">
                <div className="grid h-11 w-11 shrink-0 place-items-center rounded-[12px] bg-lilac/15 text-lilac">
                  <Layers className="h-5 w-5" />
                </div>
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="font-semibold">{l.name}</h3>
                    <Badge tone="neutral">{SOURCE_LABEL[l.source]}</Badge>
                  </div>
                  <p className="mt-0.5 text-sm text-text-muted">
                    {l.baseModel} · {formatBytes(l.sizeBytes)} · added{" "}
                    {timeAgo(l.installedAt)}
                  </p>
                  <p className="mt-1 font-mono text-xs text-text-muted">{l.path}</p>
                  {l.triggerWords.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {l.triggerWords.map((t) => (
                        <span
                          key={t}
                          className="rounded-md bg-surface-hover px-2 py-0.5 text-xs text-text-secondary"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <Button
                variant="ghost"
                onClick={() => remove(l.id)}
                aria-label="Delete LoRA"
                className="self-start sm:self-auto"
              >
                <Trash2 className="h-4 w-4" /> Delete
              </Button>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
