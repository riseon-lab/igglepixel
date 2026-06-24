import type { NextRequest } from "next/server";
import { accountStatus, requireSession } from "@/lib/auth/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Lets the client learn whether an account exists, whether the stored account is
// readable, and whether the current cookie is a valid session — without ever
// exposing the token to page JS. `corrupt` is kept distinct from "no account" so
// the client never mistakes a damaged file (a pod killed mid-write) for a fresh
// install and routes you into setup.
export async function GET(req: NextRequest) {
  const status = await accountStatus();
  const session = await requireSession(req);
  return Response.json({
    hasAccount: status === "ok",
    corrupt: status === "corrupt",
    authenticated: !!session,
    username: session?.username ?? null,
  });
}
