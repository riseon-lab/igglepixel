import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import * as vault from "@/lib/vault/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Returns the raw AES-256-GCM envelope. The client decrypts it in a Worker;
// the server has no key and only ever streams ciphertext.
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!(await requireSession(req))) return unauthorized();
  const { id } = await params;
  const bytes = await vault.getCiphertext(id);
  if (!bytes) return new Response("Not found", { status: 404 });

  return new Response(Buffer.from(bytes), {
    headers: {
      "Content-Type": "application/octet-stream",
      "Cache-Control": "no-store",
    },
  });
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!(await requireSession(req))) return unauthorized();
  const { id } = await params;
  const removed = await vault.remove(id);
  return new Response(null, { status: removed ? 204 : 404 });
}
