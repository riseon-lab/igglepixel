import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import { removeLora } from "@/lib/loras/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!(await requireSession(req))) return unauthorized();
  const { id } = await params;
  const removed = await removeLora(decodeURIComponent(id));
  return Response.json({ ok: removed });
}
