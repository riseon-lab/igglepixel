import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { clearSessionCookie, resetAccount } from "@/lib/auth/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return new Response(
    `<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reset local account</title>
<body style="margin:0;background:#111;color:#f5f5f5;font:18px system-ui;display:grid;min-height:100vh;place-items:center">
  <form method="post" style="max-width:460px;padding:32px;border:1px solid #333;border-radius:18px;background:#1a1a1a">
    <h1 style="margin-top:0">Reset local account</h1>
    <p>This removes only the local account record. Assets, models, LoRAs, and downloads stay on disk.</p>
    <p>Type RESET to continue.</p>
    <input name="confirm" autocomplete="off" style="box-sizing:border-box;width:100%;padding:14px;border-radius:10px;border:1px solid #444;background:#111;color:#fff;font:inherit">
    <button style="margin-top:16px;width:100%;padding:14px;border:0;border-radius:10px;background:#a77bea;color:#fff;font:inherit;font-weight:700">Reset account</button>
  </form>
</body>`,
    { headers: { "content-type": "text/html; charset=utf-8" } },
  );
}

/**
 * Reject anything that isn't a same-origin request. Reset is intentionally
 * usable without a session (it is the "forgot password" recovery on the login
 * screen), so this no-auth, account-destroying POST would otherwise be a prime
 * CSRF target — a cross-site auto-submitting <form> could wipe the account.
 */
function sameOrigin(req: NextRequest): boolean {
  // Sec-Fetch-Site is sent by every current browser; trust it when present.
  const site = req.headers.get("sec-fetch-site");
  if (site) return site === "same-origin" || site === "none";
  // Older clients: fall back to comparing Origin against the request host.
  const origin = req.headers.get("origin");
  if (!origin) return true; // non-browser caller (no Origin) — not a CSRF vector
  try {
    return new URL(origin).host === req.headers.get("host");
  } catch {
    return false;
  }
}

async function readConfirm(req: NextRequest): Promise<unknown> {
  const type = req.headers.get("content-type") ?? "";
  if (type.includes("application/json")) {
    const body = (await req.json()) as { confirm?: unknown };
    return body.confirm;
  }
  const body = await req.formData();
  return body.get("confirm");
}

export async function POST(req: NextRequest) {
  if (!sameOrigin(req))
    return NextResponse.json({ error: "Cross-site request blocked." }, { status: 403 });

  let confirm: unknown;
  try {
    confirm = await readConfirm(req);
  } catch {
    return NextResponse.json({ error: "Invalid request." }, { status: 400 });
  }

  if (confirm !== "RESET")
    return NextResponse.json({ error: "Reset was not confirmed." }, { status: 400 });

  await resetAccount();
  const res = NextResponse.redirect(new URL("/setup", req.url));
  clearSessionCookie(res);
  return res;
}
