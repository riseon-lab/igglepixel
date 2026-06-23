import type { NextRequest } from "next/server";
import { getAccount, requireSession } from "@/lib/auth/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Lets the client learn whether an account exists and whether the current cookie
// is a valid session — without ever exposing the token to page JS.
export async function GET(req: NextRequest) {
  const account = await getAccount();
  const session = await requireSession(req);
  return Response.json({
    hasAccount: !!account,
    authenticated: !!session,
    username: session?.username ?? null,
  });
}
