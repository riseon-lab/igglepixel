"use client";

import { Layers, Link2, Trash2, Upload } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Input } from "@/components/ui/Field";
import { PageHeader } from "@/components/ui/PageHeader";
import { formatBytes, timeAgo } from "@/lib/format";
import { LORAS } from "@/lib/mock";
import type { Lora, LoraSource } from "@/lib/types";

const SOURCE_LABEL: Record<LoraSource, string> = {
  civitai: "Civitai",
  huggingface: "Hugging Face",
  upload: "Uploaded",
};

export default function LorasPage() {
  const [loras, setLoras] = useState<Lora[]>(LORAS);
  const [url, setUrl] = useState("");

  function addFromUrl() {
    if (!url.trim()) return;
    const source: LoraSource = url.includes("huggingface")
      ? "huggingface"
      : "civitai";
    const name =
      decodeURIComponent(url.split("/").filter(Boolean).pop() ?? "New LoRA")
        .replace(/[-_]/g, " ")
        .slice(0, 40) || "New LoRA";
    setLoras((prev) => [
      {
        id: `new-${Date.now()}`,
        name,
        source,
        baseModel: "Qwen 2512",
        sizeBytes: 190_000_000,
        triggerWords: [],
        installedAt: new Date().toISOString(),
      },
      ...prev,
    ]);
    setUrl("");
  }

  function remove(id: string) {
    setLoras((prev) => prev.filter((l) => l.id !== id));
  }

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="LoRAs"
        description="Install and manage LoRAs from Civitai, Hugging Face, or direct upload."
        actions={
          <Button variant="secondary">
            <Upload className="h-4 w-4" /> Upload LoRA
          </Button>
        }
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
          <Button onClick={addFromUrl}>Download</Button>
        </div>
        <p className="text-xs text-text-muted">
          API keys are configured in Settings. Downloads appear on the Downloads page.
        </p>
      </Card>

      <section className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h2 className="text-[20px] font-semibold">Installed</h2>
          <span className="text-sm text-text-muted">{loras.length} LoRAs</span>
        </div>
        <div className="grid gap-4">
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
