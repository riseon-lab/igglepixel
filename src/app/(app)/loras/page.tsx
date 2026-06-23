"use client";

import { clsx } from "clsx";
import { Check, Layers, Link2, Loader2, Trash2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Input } from "@/components/ui/Field";
import { Modal } from "@/components/ui/Modal";
import { PageHeader } from "@/components/ui/PageHeader";
import { useToast } from "@/components/ui/Toast";
import { formatBytes, timeAgo } from "@/lib/format";
import type { Lora, LoraCandidate, LoraResolution, LoraSource } from "@/lib/types";

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
  const [resolving, setResolving] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Picker modal for URLs that resolve to more than one file.
  const [picker, setPicker] = useState<LoraResolution | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());

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

  // Step 1: resolve the URL into candidate files. One file installs directly;
  // multiple files open the picker.
  async function resolveUrl() {
    if (!url.trim() || resolving || installing) return;
    setResolving(true);
    setError(null);
    try {
      const res = await fetch("/api/loras/resolve", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ url: url.trim() }),
      });
      const data = (await res.json()) as LoraResolution & { error?: string };
      if (!res.ok) throw new Error(data.error ?? "Could not resolve that URL.");

      const candidates = data.candidates ?? [];
      if (candidates.length === 0) {
        toast.error(
          "No LoRA files found",
          "That link didn't contain any installable .safetensors files.",
        );
        return;
      }
      if (candidates.length === 1) {
        await install(candidates);
        return;
      }
      const recommended = candidates.filter((c) => c.recommended).map((c) => c.id);
      setSelected(new Set(recommended.length ? recommended : [candidates[0].id]));
      setPicker(data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Could not resolve that URL.";
      setError(msg);
      toast.error("Couldn’t read that URL", msg);
    } finally {
      setResolving(false);
    }
  }

  // Step 2: download the chosen files.
  async function install(items: LoraCandidate[]) {
    if (!items.length) return;
    setInstalling(true);
    setError(null);
    try {
      const res = await fetch("/api/loras", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ urls: items.map((i) => i.downloadUrl) }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "LoRA install failed.");
      setUrl("");
      setPicker(null);
      await refresh();
      toast.success(
        items.length === 1 ? "LoRA installed" : `${items.length} LoRAs installed`,
        "Saved to the shared runner LoRA folder.",
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : "LoRA install failed.";
      setError(msg);
      toast.error("LoRA install failed", msg);
    } finally {
      setInstalling(false);
    }
  }

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function upload(file: File | undefined) {
    if (!file) return;
    setUploading(true);
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
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function remove(id: string) {
    setError(null);
    const res = await fetch(`/api/loras/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      toast.error("Could not delete LoRA", "Please try again.");
      return;
    }
    setLoras((prev) => prev.filter((l) => l.id !== id));
    toast.success("LoRA deleted");
  }

  const selectedCandidates =
    picker?.candidates.filter((c) => selected.has(c.id)) ?? [];

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="LoRAs"
        description="Install and manage LoRAs from Civitai, Hugging Face, or direct upload."
        actions={
          <Button
            variant="secondary"
            onClick={() => fileInput.current?.click()}
            disabled={uploading}
          >
            {uploading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Uploading…
              </>
            ) : (
              <>
                <Upload className="h-4 w-4" /> Upload LoRA
              </>
            )}
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
              onKeyDown={(e) => e.key === "Enter" && resolveUrl()}
            />
          </div>
          <Button onClick={resolveUrl} disabled={resolving || installing || !url.trim()}>
            {resolving ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Reading…
              </>
            ) : installing ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Installing…
              </>
            ) : (
              "Install"
            )}
          </Button>
        </div>
        <p className="text-xs text-text-muted">
          Paste a model page or a direct file link. If it has more than one file
          you’ll choose which to install. Gated downloads use the API keys from
          Settings; files are saved to the shared runner LoRA folder.
        </p>
        {error && <p className="text-sm text-danger-text">{error}</p>}
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

      <Modal
        open={!!picker}
        onClose={() => !installing && setPicker(null)}
        title="Choose files to install"
        description={
          picker?.modelName
            ? `${SOURCE_LABEL[picker.source]} · ${picker.modelName}`
            : "Select the LoRA files you want to install."
        }
        footer={
          <>
            <Button
              variant="secondary"
              onClick={() => setPicker(null)}
              disabled={installing}
            >
              Cancel
            </Button>
            <Button
              onClick={() => install(selectedCandidates)}
              disabled={installing || selected.size === 0}
            >
              {installing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Installing…
                </>
              ) : (
                `Install ${selected.size}`
              )}
            </Button>
          </>
        }
      >
        <div className="flex flex-col gap-2">
          {picker?.candidates.map((c) => {
            const checked = selected.has(c.id);
            const meta = [
              c.versionName,
              c.baseModel,
              c.sizeBytes ? formatBytes(c.sizeBytes) : null,
            ]
              .filter(Boolean)
              .join(" · ");
            return (
              <button
                key={c.id}
                type="button"
                onClick={() => toggle(c.id)}
                className={clsx(
                  "flex items-start gap-3 rounded-[12px] border p-3 text-left transition-colors",
                  checked
                    ? "border-lilac bg-lilac/10"
                    : "border-border hover:bg-surface-hover",
                )}
              >
                <span
                  className={clsx(
                    "mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-md border transition-colors",
                    checked ? "border-lilac bg-lilac text-white" : "border-border",
                  )}
                >
                  {checked && <Check className="h-3.5 w-3.5" />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2">
                    <span className="truncate font-medium">{c.fileName}</span>
                    {c.recommended && <Badge tone="lilac">Recommended</Badge>}
                  </span>
                  {meta && (
                    <span className="mt-0.5 block truncate text-xs text-text-muted">
                      {meta}
                    </span>
                  )}
                </span>
              </button>
            );
          })}
        </div>
      </Modal>
    </div>
  );
}
