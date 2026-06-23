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
