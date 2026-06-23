import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import { isEnvelope } from "@/lib/crypto/aesgcm";
import * as vault from "@/lib/vault/storage";
import type { NewAssetMeta } from "@/lib/vault/types";

// Node runtime (needs fs); never cache — the vault is mutable per request.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Guarded by the server session: metadata (filenames/sizes) and the ability to
// upload/delete are gated, even though stored bytes are always ciphertext.

export async function GET(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();
  return Response.json(await vault.list());
}

export async function POST(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();
  const form = await req.formData();
  const blob = form.get("blob");
  const metaRaw = form.get("meta");

  if (!(blob instanceof Blob) || typeof metaRaw !== "string") {
    return new Response("Expected `blob` (ciphertext) and `meta` fields.", {
      status: 400,
    });
  }

  let meta: NewAssetMeta;
  try {
    meta = JSON.parse(metaRaw);
  } catch {
    return new Response("Invalid `meta` JSON.", { status: 400 });
  }

  const ciphertext = new Uint8Array(await blob.arrayBuffer());
  if (!isEnvelope(ciphertext)) {
    return new Response("Expected a Citivia encrypted envelope.", { status: 400 });
  }
  const saved = await vault.put(ciphertext, {
    name: String(meta.name ?? "untitled"),
    kind: meta.kind,
    mime: String(meta.mime ?? "application/octet-stream"),
    width: Number(meta.width) || 0,
    height: Number(meta.height) || 0,
    size: Number(meta.size) || ciphertext.length,
  });

  return Response.json(saved, { status: 201 });
}
