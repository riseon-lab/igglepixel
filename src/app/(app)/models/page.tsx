import { ArrowRight, ImageIcon, Wand2 } from "lucide-react";
import Link from "next/link";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { MODELS } from "@/lib/mock";

export default function ModelsPage() {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Models"
        description="Choose a model to open its generation workspace."
      />

      <div className="grid gap-5 sm:grid-cols-2">
        {MODELS.map((m) => {
          const Icon = m.kind === "editing" ? ImageIcon : Wand2;
          return (
            <Link key={m.id} href={`/generate/${m.id}`} className="group">
              <Card interactive className="flex h-full flex-col gap-4">
                <div className="flex items-start justify-between">
                  <div className="grid h-12 w-12 place-items-center rounded-[12px] bg-lilac/15 text-lilac">
                    <Icon className="h-6 w-6" />
                  </div>
                  <Badge tone={m.status === "running" ? "success" : "neutral"}>
                    {m.status === "running" ? "Running" : "Stopped"}
                  </Badge>
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold">{m.name}</h3>
                  <p className="text-sm font-medium text-lilac">{m.tagline}</p>
                  <p className="mt-2 text-sm text-text-secondary">
                    {m.description}
                  </p>
                </div>
                <div className="flex items-center gap-2 text-sm font-semibold text-text-secondary transition-colors group-hover:text-white">
                  Open workspace
                  <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                </div>
              </Card>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
