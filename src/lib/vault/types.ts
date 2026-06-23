import type { AssetKind } from "../types";

// Metadata the server keeps alongside each encrypted blob. This is intentionally
// minimal and non-sensitive — the image bytes themselves are stored encrypted.
// (In a hardened build the filename could be encrypted too; kept plaintext here
// for a usable preview.)
export interface AssetMeta {
  id: string;
  name: string;
  kind: AssetKind;
  mime: string;
  width: number;
  height: number;
  /** Plaintext size in bytes (pre-encryption). */
  size: number;
  createdAt: string;
}

export type NewAssetMeta = Omit<AssetMeta, "id" | "createdAt">;
