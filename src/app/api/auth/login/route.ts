import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { login, setSessionCookie } from "@/lib/auth/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  let body: { username?: unknown; password?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid request." }, { status: 400 });
  }
  const username = String(body.username ?? "").trim();
  const password = String(body.password ?? "");

  const result = await login(username, password);
  if ("error" in result)
    return NextResponse.json({ error: result.error }, { status: 401 });

  const res = NextResponse.json({ username }, { status: 200 });
  setSessionCookie(res, result.token); // rotating the token ends any other session
  return res;
}
