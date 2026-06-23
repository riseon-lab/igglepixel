import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import * as vault from "@/lib/vault/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The account's PBKDF2 salt (not secret, but session-gated). Served so an
// authenticated device derives the same encryption key from the password and can
// decrypt previously stored assets.
export async function GET(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();
  return Response.json({ salt: await vault.getOrCreateSalt() });
}
