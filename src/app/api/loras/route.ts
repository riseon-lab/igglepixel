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

    const body = (await req.json().catch(() => null)) as {
      url?: unknown;
      urls?: unknown;
    } | null;

    // Preferred path: install one or more chosen files (from the picker).
    if (body && Array.isArray(body.urls)) {
      const urls = body.urls.filter(
        (u): u is string => typeof u === "string" && !!u.trim(),
      );
      if (!urls.length) {
        return Response.json({ error: "No files selected." }, { status: 400 });
      }
      const installed = [];
      for (const u of urls) installed.push(await downloadLora(u.trim()));
      return Response.json(installed, { status: 201 });
    }

    // Backward-compatible single URL.
    if (body && typeof body.url === "string" && body.url.trim()) {
      return Response.json([await downloadLora(body.url.trim())], { status: 201 });
    }

    return Response.json({ error: "LoRA URL is required." }, { status: 400 });
  } catch (err) {
    return Response.json(
      { error: err instanceof Error ? err.message : "LoRA install failed." },
      { status: 500 },
    );
  }
}
