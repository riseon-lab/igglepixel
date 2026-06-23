import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import {
  clearSessionCookie,
  logout,
  requireSession,
  unauthorized,
} from "@/lib/auth/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();
  await logout();
  const res = NextResponse.json({ ok: true });
  clearSessionCookie(res);
  return res;
}
