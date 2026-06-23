// Browser-side wrapper around the vault API. Deals only in ciphertext + metadata;
// encryption/decryption happens via the EncryptionClient before/after these calls.

import type { AssetMeta, NewAssetMeta } from "./types";

export async function listAssets(): Promise<AssetMeta[]> {
  const res = await fetch("/api/vault", { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load assets (${res.status}).`);
  return res.json();
}

export async function uploadAsset(
  ciphertext: Uint8Array,
  meta: NewAssetMeta,
): Promise<AssetMeta> {
  const form = new FormData();
  form.append("meta", JSON.stringify(meta));
  form.append(
    "blob",
    new Blob([ciphertext as BlobPart], { type: "application/octet-stream" }),
  );
  const res = await fetch("/api/vault", { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed (${res.status}).`);
  return res.json();
}

export async function fetchCiphertext(id: string): Promise<Uint8Array> {
  const res = await fetch(`/api/vault/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch asset (${res.status}).`);
  return new Uint8Array(await res.arrayBuffer());
}

export async function deleteAsset(id: string): Promise<void> {
  const res = await fetch(`/api/vault/${id}`, { method: "DELETE" });
  if (!res.ok && res.status !== 404)
    throw new Error(`Delete failed (${res.status}).`);
}

export async function fetchSalt(): Promise<string> {
  const res = await fetch("/api/keys/salt", { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch key salt (${res.status}).`);
  return (await res.json()).salt as string;
}
