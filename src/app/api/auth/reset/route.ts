import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { clearSessionCookie, resetAccount } from "@/lib/auth/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  let body: { confirm?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid request." }, { status: 400 });
  }

  if (body.confirm !== "RESET")
    return NextResponse.json({ error: "Reset was not confirmed." }, { status: 400 });

  await resetAccount();
  const res = NextResponse.json({ ok: true });
  clearSessionCookie(res);
  return res;
}
