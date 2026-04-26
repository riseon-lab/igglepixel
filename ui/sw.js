// ─── Forge service worker — decrypts /api/assets/file/* in the browser ──────
//
// The main thread postMessages a non-extractable CryptoKey to us after the
// user signs in. We hold it in module scope and use it to AES-GCM-decrypt
// any asset response that the backend marks with `X-Forge-Encrypted: 1`.
// `<img src="…">` / `<video src="…">` work unchanged — the SW intercepts
// transparently.
//
// On logout the main thread sends `{type: "clear-key"}` and we drop the
// reference; the browser GC's the underlying material.
// ────────────────────────────────────────────────────────────────────────────

const NONCE_LEN = 12;
let dataKey = null;

self.addEventListener("install",  () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("message", (e) => {
  const msg = e.data || {};
  if (msg.type === "set-key" && msg.key) {
    dataKey = msg.key;
    if (e.ports?.[0]) e.ports[0].postMessage({ ok: true });
  } else if (msg.type === "clear-key") {
    dataKey = null;
    if (e.ports?.[0]) e.ports[0].postMessage({ ok: true });
  } else if (msg.type === "ping") {
    if (e.ports?.[0]) e.ports[0].postMessage({ hasKey: !!dataKey });
  }
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (event.request.method !== "GET") return;
  if (!url.pathname.startsWith("/api/assets/file/")) return;
  event.respondWith(handle(event.request));
});

async function handle(req) {
  const upstream = await fetch(req);
  if (!upstream.ok) return upstream;

  // Backend marks .enc-served files with this header. Plaintext / legacy
  // files are passed through unchanged.
  const isEncrypted = upstream.headers.get("X-Forge-Encrypted") === "1";
  if (!isEncrypted) return upstream;

  if (!dataKey) {
    // Key hasn't been shared yet — return a 503 so the <img> shows broken
    // and the main thread knows to prompt for unlock.
    return new Response("Locked: key not in browser", {
      status: 503,
      headers: { "content-type": "text/plain" },
    });
  }

  let buf;
  try {
    buf = await upstream.arrayBuffer();
  } catch {
    return upstream;
  }
  const data = new Uint8Array(buf);
  if (data.byteLength < NONCE_LEN + 16) {
    return new Response("Malformed ciphertext", { status: 502 });
  }
  const iv = data.subarray(0, NONCE_LEN);
  const ct = data.subarray(NONCE_LEN);

  let pt;
  try {
    pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, dataKey, ct);
  } catch {
    return new Response("Decryption failed (wrong key?)", { status: 502 });
  }

  // Best-effort wipe of the ciphertext buffer we held. Doesn't help against
  // a determined memory-inspecting attacker (browsers may have already
  // copied during GC, OS may have paged) but reduces incidental retention.
  try { data.fill(0); } catch {}

  // The visible filename's extension drives the MIME. Backend sets
  // content-type for us, but we override with what the URL's filename says
  // because FileResponse hands back octet-stream for .enc paths.
  const mime = upstream.headers.get("content-type")
    || guessMime(req.url)
    || "application/octet-stream";
  return new Response(pt, {
    status:  200,
    headers: { "content-type": mime, "x-forge-decrypted": "1" },
  });
}

function guessMime(url) {
  const ext = url.split('?')[0].split('.').pop()?.toLowerCase();
  return {
    png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg",
    webp: "image/webp", gif: "image/gif", bmp: "image/bmp",
    mp4: "video/mp4", webm: "video/webm", mov: "video/quicktime",
    m4v: "video/x-m4v", mkv: "video/x-matroska",
  }[ext];
}
