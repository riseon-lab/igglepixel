import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import { downloadLora, listLoras, saveUploadedLora } from "@/lib/loras/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();
  return Response.json(await listLoras());
}

export async function POST(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();

  try {
    const contentType = req.headers.get("content-type") || "";
    if (contentType.includes("multipart/form-data")) {
      const file = (await req.formData()).get("file");
      if (!(file instanceof File)) {
        return Response.json({ error: "LoRA file is required." }, { status: 400 });
      }
      return Response.json(await saveUploadedLora(file), { status: 201 });
    }

    const body = (await req.json().catch(() => null)) as { url?: unknown } | null;
    if (!body || typeof body.url !== "string" || !body.url.trim()) {
      return Response.json({ error: "LoRA URL is required." }, { status: 400 });
    }
    return Response.json(await downloadLora(body.url.trim()), { status: 201 });
  } catch (err) {
    return Response.json(
      { error: err instanceof Error ? err.message : "LoRA install failed." },
      { status: 500 },
    );
  }
}
