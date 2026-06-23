import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";

export default function DownloadsPage() {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Downloads"
        description="Track model and LoRA downloads and their history."
      />

      <Section title="Active" empty="No active downloads." />
      <Section title="History" empty="No download history." />
    </div>
  );
}

function Section({
  title,
  empty,
}: {
  title: string;
  empty: string;
}) {
  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-[20px] font-semibold">{title}</h2>
        <span className="text-sm text-text-muted">0</span>
      </div>
      <Card className="py-10 text-center text-text-muted">{empty}</Card>
    </section>
  );
}
