"use client";

// Fetches a stored ciphertext, decrypts it via the Worker, and renders the
// result from an in-memory object URL. The URL is revoked on unmount/asset
// change so no decrypted bytes linger (plan.md: "decrypted only in memory").

import { clsx } from "clsx";
import { ImageOff } from "lucide-react";
import { useEffect, useState } from "react";
import type { EncryptionClient } from "@/lib/crypto/client";
import { fetchCiphertext } from "@/lib/vault/client";

export function EncryptedImage({
  id,
  mime,
  client,
  alt = "",
  className,
}: {
  id: string;
  mime: string;
  client: EncryptionClient;
  alt?: string;
  className?: string;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    // State is reset via async results below; the parent keys each tile by asset
    // id so this component never gets reused across different assets.
    let active = true;
    let objectUrl: string | null = null;

    (async () => {
      try {
        const ciphertext = await fetchCiphertext(id);
        const plain = await client.decrypt(ciphertext);
        if (!active) return;
        objectUrl = URL.createObjectURL(
          new Blob([plain as BlobPart], { type: mime }),
        );
        setUrl(objectUrl);
      } catch {
        if (active) setFailed(true);
      }
    })();

    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [id, mime, client]);

  if (failed) {
    return (
      <div
        className={clsx(
          "grid place-items-center bg-background text-text-muted",
          className,
        )}
      >
        <ImageOff className="h-6 w-6" />
      </div>
    );
  }

  if (!url) {
    return <div className={clsx("animate-pulse bg-surface-hover", className)} />;
  }

  // Decrypted in-memory blob URL — next/image can't optimize these.
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={url} alt={alt} className={className} />;
}
