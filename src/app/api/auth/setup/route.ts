import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { createAccount, setSessionCookie } from "@/lib/auth/server";

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

  if (username.length < 3)
    return NextResponse.json(
      { error: "Username must be at least 3 characters." },
      { status: 400 },
    );
  if (password.length < 8)
    return NextResponse.json(
      { error: "Password must be at least 8 characters." },
      { status: 400 },
    );

  const result = await createAccount(username, password);
  if ("error" in result)
    return NextResponse.json({ error: result.error }, { status: 409 });

  const res = NextResponse.json({ username }, { status: 201 });
  setSessionCookie(res, result.token);
  return res;
}
