// ─── IndexedDB persistence for the AES-GCM CryptoKey ───────────────────────
//
// IDB can store CryptoKey objects directly via structured-clone, including
// non-extractable ones. Reload the page → fetch from IDB → use it again,
// without the raw key material ever appearing in JS variables.
//
// On logout we delete the record AND post a "clear-key" message to the
// service worker, so the key drops out of every context that held it.
// ─────────────────────────────────────────────────────────────────────────

const DB_NAME = "forge-keys";
const STORE   = "keys";
const KEY_ID  = "data";

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

export async function storeKey(key) {
  const db = await openDB();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(key, KEY_ID);
    tx.oncomplete = resolve;
    tx.onerror    = () => reject(tx.error);
  });
  db.close();
}

export async function loadKey() {
  const db = await openDB();
  const key = await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).get(KEY_ID);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror   = () => reject(req.error);
  });
  db.close();
  return key;
}

export async function clearKey() {
  try {
    const db = await openDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).delete(KEY_ID);
      tx.oncomplete = resolve;
      tx.onerror    = () => reject(tx.error);
    });
    db.close();
  } catch { /* IDB unavailable — fine, nothing to clear */ }
}
