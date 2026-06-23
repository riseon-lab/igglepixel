import { promises as fs } from "node:fs";
import path from "node:path";
import { vaultDir } from "@/lib/vault/storage";

export interface ApiKeys {
  civitai?: string;
  huggingface?: string;
}

const keyFile = () =>
  path.join(/* turbopackIgnore: true */ vaultDir(), "api-keys.json");

export async function getApiKeyStatus() {
  const keys = await readApiKeys();
  return {
    civitai: !!keys.civitai,
    huggingface: !!keys.huggingface,
  };
}

export async function readApiKeys(): Promise<ApiKeys> {
  try {
    return JSON.parse(await fs.readFile(keyFile(), "utf8"));
  } catch {
    return {};
  }
}

export async function saveApiKeys(next: ApiKeys): Promise<void> {
  const current = await readApiKeys();
  const keys = {
    ...current,
    ...Object.fromEntries(
      Object.entries(next).filter(([, value]) => value && value.trim()),
    ),
  };
  await fs.mkdir(vaultDir(), { recursive: true });
  await fs.writeFile(keyFile(), JSON.stringify(keys), { mode: 0o600 });
}
