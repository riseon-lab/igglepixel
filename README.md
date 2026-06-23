# Citivia Studio

A self-hosted AI image generation platform (Qwen 2512 + Qwen Edit 2511), designed
for RunPod deployment. Built from [plan.md](plan.md).

> **Current status: Preview Build.** This is the fully-functional UI prototype the
> spec calls for *before* real model integration — it runs entirely on mock data
> (no inference). The goal is to validate UX, navigation, the queue/upload
> workflows, layout and mobile responsiveness. See [plan.md](plan.md) → "Preview Build".

## Running locally

```bash
npm install
npm run dev      # http://localhost:3000
```

On first launch you'll see the **Setup** screen — create a username/password
(stored in `localStorage` for the preview). After that you'll land on the studio.

```bash
npm run build && npm start   # standalone production server, also on port 3000
npm run lint
npm run test:crypto          # AES-256-GCM unit tests (node --test, no extra deps)
```

## Docker / RunPod

The image listens on port 3000 and stores data under `/workspace`.

```bash
docker build -t citivia-studio .
docker run --rm -p 3000:3000 -v citivia-workspace:/workspace citivia-studio
```

On startup the container pulls the latest UI from
`https://github.com/riseon-lab/igglepixel.git` into `/workspace/igglepixel`,
builds it, and starts the standalone server. Set `CITIVIA_AUTO_PULL=0` to run the
baked image only. If you terminate TLS before the app, set
`CITIVIA_SECURE_COOKIES=1`; plain port-3000 deployments leave it unset so login
works over HTTP.

## Encryption layer

Implements the spec's "encrypted at rest" requirement with **AES-256-GCM** via the
Web Crypto API. The same code runs in the browser and under Node, so it's covered
by fast unit tests *and* exercised in a real browser Worker.

- [src/lib/crypto/aesgcm.ts](src/lib/crypto/aesgcm.ts) — core: PBKDF2 key
  derivation, per-file random IV, a self-describing envelope
  (`CITV` magic + version + IV + GCM ciphertext/tag), tamper-detecting decrypt.
- [src/lib/crypto/worker.ts](src/lib/crypto/worker.ts) — dedicated Web Worker; the
  key is sent once and never leaves it. Decrypted bytes exist only in memory —
  nothing is written to disk.
- [src/lib/crypto/client.ts](src/lib/crypto/client.ts) — main-thread client with a
  promise API and a transparent inline fallback when Workers are unavailable.
- [src/lib/crypto/aesgcm.test.ts](src/lib/crypto/aesgcm.test.ts) — 12 tests:
  round-trips (incl. 2 MiB binary + empty), wrong-key rejection, tamper
  detection, password derivation, raw-key export/import (the Worker hand-off).

### Encrypted vault (server storage)

The Assets flow is wired through the encryption layer end-to-end:

- **Upload** — the browser reads the image, the Worker encrypts it, and only the
  ciphertext envelope is POSTed to the server.
- **Storage** — [src/lib/vault/storage.ts](src/lib/vault/storage.ts) writes each
  asset as `<id>.bin` (ciphertext) + `<id>.json` (non-sensitive metadata). The
  server never sees plaintext or holds a key.
- **Preview / download** — the ciphertext is fetched and decrypted in the Worker
  into an in-memory object URL (revoked on unmount); nothing decrypted is written
  to disk.
- **Key** — [src/lib/crypto/provider.tsx](src/lib/crypto/provider.tsx) derives the
  AES key from the account password + a server-stored PBKDF2 salt
  (`GET /api/keys/salt`), so any device with the password can decrypt.

Routes: `GET/POST /api/vault`, `GET/DELETE /api/vault/[id]`, `GET /api/keys/salt`.

**Storage location** (plan.md: "save to /workspace only"): set
`CITIVIA_DATA_DIR=/workspace` on RunPod; locally it defaults to a gitignored
`.vault/` directory.

**Test it locally three ways:**

1. `npm test` — Node suite for both the AES-GCM core (12) and the vault storage
   layer (4): round-trips, wrong-key/tamper rejection, path-traversal guard.
2. **Settings → Encryption self-test → Run self-test** — drives the *real* Worker
   end-to-end (derive key → encrypt → decrypt → render a PNG), proving the worker
   is active, tampering/wrong keys are rejected, and the round-trip is
   pixel-identical.
3. **Assets → Upload** — upload an image; it's encrypted before leaving the
   browser, stored as ciphertext, then fetched and decrypted in-memory to display.
   Confirm with `xxd .vault/<id>.bin | head -1` that the stored bytes begin with
   `CITV` (the envelope magic) — not a PNG/JPEG header.

## What's implemented (preview)

- **Design system** — all tokens from the spec's Design Language (lilac brand,
  `#121212` canvas, 12/16/24px radii, 16px-minimum type, Geist) live in
  [src/app/globals.css](src/app/globals.css).
- **Responsive app shell** — collapsible left sidebar on desktop, fixed bottom
  nav on mobile ([src/components/AppShell.tsx](src/components/AppShell.tsx)).
- **Auth / session** — first-run setup, login, single-active-session semantics,
  logout (mock, [src/lib/session.tsx](src/lib/session.tsx)).
- **Pages** — Running (live resource meters, start/stop), Models (card select),
  Assets (upload/filter/delete, aspect-preserving previews), LoRAs (Civitai / HF
  / upload), Downloads (progress + history), Settings (API keys, git pull,
  session).
- **Generation workspace** — shared by both models, with Qwen Edit adding the
  reference-image step above the prompt. Includes resolution preset cards,
  steps/CFG/seed controls, live (simulated) progress preview, queue, and a
  full-size lightbox ([src/components/generation/](src/components/generation/)).

## Mock vs. real

Everything under [src/lib/mock.ts](src/lib/mock.ts) and the simulated generation
ticking is placeholder. Integration points for the real build:

| Concern | Preview | Real build |
| --- | --- | --- |
| Auth/session | `localStorage` | Secure server-side session tokens |
| Assets | **encrypted at rest (AES-256-GCM), worker decryption, server stores only ciphertext** ✅ | + auth on the vault API, encrypted filenames |
| Generation | `setInterval` progress | Isolated per-model runners (independent Python envs) |
| Downloads | mock list | Civitai / Hugging Face APIs |
| Storage | `.vault/` locally | `/workspace` via `CITIVIA_DATA_DIR` |

## Project layout

```
src/
  app/
    (app)/            # authenticated routes (shared shell)
      running/ models/ assets/ loras/ downloads/ settings/
      generate/[model]/
    setup/ login/     # public auth routes
  components/
    ui/               # Button, Card, Slider, Toggle, Badge, Field, PageHeader
    generation/       # GenerationWorkspace, ResolutionPicker, QueuePanel
    AppShell, Topbar, AuthGate, PreviewTile
  lib/                # types, mock data, nav config, formatting, session
```
