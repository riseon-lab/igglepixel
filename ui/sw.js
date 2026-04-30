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

// Keep these in sync with ui/key_store.js — the SW reads the same IDB record
// the page wrote, so we don't need a separate handshake on every wake-up.
const KEY_DB    = "forge-keys";
const KEY_STORE = "keys";
const KEY_ID    = "data";

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

// Browsers terminate idle service workers (~30s on Chromium). When that
// happens, the next fetch re-evaluates this script with `dataKey = null`,
// so any encrypted asset request would 503 even though the user is still
// unlocked. Read from the same IDB record the page maintains so we recover
// transparently. The CryptoKey is non-extractable; structured-clone in IDB
// keeps it that way.
async function loadKeyFromIDB() {
  try {
    return await new Promise((resolve, reject) => {
      const req = indexedDB.open(KEY_DB, 1);
      // If the DB doesn't exist yet, create the store so the open() doesn't
      // error out — page-side will write into it on next unlock.
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(KEY_STORE)) db.createObjectStore(KEY_STORE);
      };
      req.onsuccess = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(KEY_STORE)) { db.close(); resolve(null); return; }
        const tx = db.transaction(KEY_STORE, "readonly");
        const g  = tx.objectStore(KEY_STORE).get(KEY_ID);
        g.onsuccess = () => { db.close(); resolve(g.result || null); };
        g.onerror   = () => { db.close(); reject(g.error); };
      };
      req.onerror = () => reject(req.error);
    });
  } catch {
    return null;
  }
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (event.request.method !== "GET") return;
  if (!url.pathname.startsWith("/api/assets/file/")) return;
  event.respondWith(handle(event.request));
});

async function handle(req) {
  // Video elements commonly request byte ranges. AES-GCM ciphertext cannot be
  // decrypted from an arbitrary partial range, so for encrypted assets we
  // intentionally fetch the complete blob from the backend, decrypt it, and
  // then apply any requested range to the plaintext.
  const upstreamReq = req.headers.has("range") ? withoutRange(req) : req;
  const upstream = await fetch(upstreamReq);
  if (!upstream.ok) return upstream;

  // Backend marks .enc-served files with this header. Plaintext / legacy
  // files are passed through unchanged.
  const isEncrypted = upstream.headers.get("X-Forge-Encrypted") === "1";
  if (!isEncrypted) return upstream;

  if (!dataKey) {
    // Try to recover from IDB before giving up — most "key missing" cases
    // are SW eviction, not a real lock state. The page-side writes the key
    // to IDB on every unlock and clears it on logout, so reading here is
    // equivalent to "is this user still unlocked".
    dataKey = await loadKeyFromIDB();
  }
  if (!dataKey) {
    // Key truly absent — return 503 so the <img> shows broken and the
    // main thread can prompt for unlock.
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
  const mime = guessMime(req.url)
    || upstream.headers.get("content-type")
    || "application/octet-stream";
  // Video elements need byte ranges for reliable metadata/duration probing.
  // We cannot decrypt arbitrary ciphertext ranges, so we fetch/decrypt the full
  // encrypted blob above, then satisfy the original Range request against the
  // plaintext bytes here.
  const range = req.headers.get("range");
  if (range) {
    const parsed = parseRange(range, pt.byteLength);
    if (!parsed) {
      return new Response("", {
        status: 416,
        headers: {
          "content-range": `bytes */${pt.byteLength}`,
          "accept-ranges": "bytes",
          "cache-control": "no-store",
        },
      });
    }
    const chunk = pt.slice(parsed.start, parsed.end + 1);
    return new Response(chunk, {
      status: 206,
      headers: {
        "content-type": mime,
        "content-length": String(chunk.byteLength),
        "content-range": `bytes ${parsed.start}-${parsed.end}/${pt.byteLength}`,
        "accept-ranges": "bytes",
        "cache-control": "no-store",
        "x-forge-decrypted": "1",
      },
    });
  }

  // Full response path for images and non-range media requests.
  return new Response(pt, {
    status:  200,
    headers: {
      "content-type":   mime,
      "content-length": String(pt.byteLength),
      "accept-ranges":  "bytes",
      "cache-control":  "no-store",
      "x-forge-decrypted": "1",
    },
  });
}

function parseRange(header, size) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header || "");
  if (!match || size <= 0) return null;
  let start;
  let end;
  if (match[1] === "" && match[2] === "") return null;
  if (match[1] === "") {
    const suffix = Number(match[2]);
    if (!Number.isFinite(suffix) || suffix <= 0) return null;
    start = Math.max(size - suffix, 0);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] === "" ? size - 1 : Number(match[2]);
    if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
    if (start > end || start >= size) return null;
    end = Math.min(end, size - 1);
  }
  return { start, end };
}

function withoutRange(req) {
  const headers = new Headers(req.headers);
  headers.delete("range");
  headers.delete("if-range");
  return new Request(req.url, {
    method: req.method,
    headers,
    mode: req.mode,
    credentials: req.credentials,
    cache: req.cache,
    redirect: req.redirect,
    referrer: req.referrer,
    referrerPolicy: req.referrerPolicy,
    integrity: req.integrity,
    keepalive: req.keepalive,
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
