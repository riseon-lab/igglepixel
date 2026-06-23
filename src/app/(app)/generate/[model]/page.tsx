import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { notFound } from "next/navigation";
import { GenerationWorkspace } from "@/components/generation/GenerationWorkspace";
import { Badge } from "@/components/ui/Badge";
import { MODELS } from "@/lib/mock";
import type { ModelId } from "@/lib/types";

export function generateStaticParams() {
  return MODELS.map((m) => ({ model: m.id }));
}

export default async function GeneratePage({
  params,
}: {
  params: Promise<{ model: string }>;
}) {
  const { model: modelParam } = await params;
  const model = MODELS.find((m) => m.id === (modelParam as ModelId));
  if (!model) notFound();

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <Link
          href="/models"
          className="inline-flex w-fit items-center gap-1.5 text-sm text-text-muted transition-colors hover:text-white"
        >
          <ArrowLeft className="h-4 w-4" /> Models
        </Link>
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-[28px] font-bold leading-tight">{model.name}</h1>
          <Badge tone="lilac">{model.tagline}</Badge>
        </div>
      </div>

      <GenerationWorkspace model={model} />
    </div>
  );
}
