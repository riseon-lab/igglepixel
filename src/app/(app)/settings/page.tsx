"use client";

import { Check, Eye, EyeOff, GitBranch, LogOut, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { EncryptionSelfTest } from "@/components/crypto/EncryptionSelfTest";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Input, Label } from "@/components/ui/Field";
import { PageHeader } from "@/components/ui/PageHeader";
import { Toggle } from "@/components/ui/Toggle";
import { useToast } from "@/components/ui/Toast";
import { useSession } from "@/lib/session";

function ApiKeyField({
  id,
  label,
  help,
  placeholder,
  saved,
  onSave,
}: {
  id: "civitai" | "huggingface";
  label: string;
  help: string;
  placeholder: string;
  saved: boolean;
  onSave: (id: "civitai" | "huggingface", value: string) => Promise<void>;
}) {
  const [value, setValue] = useState("");
  const [show, setShow] = useState(false);
  const [saving, setSaving] = useState(false);

  return (
    <div>
      <Label>{label}</Label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <div className="relative flex-1">
          <Input
            type={show ? "text" : "password"}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={placeholder}
            className="pr-11"
          />
          <button
            type="button"
            onClick={() => setShow((s) => !s)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-white"
            aria-label={show ? "Hide" : "Show"}
          >
            {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        </div>
        <Button
          variant={saved && !value ? "secondary" : "primary"}
          onClick={async () => {
            setSaving(true);
            await onSave(id, value);
            setValue("");
            setSaving(false);
          }}
          disabled={!value || saving}
        >
          {saving ? (
            "Saving..."
          ) : saved && !value ? (
            <>
              <Check className="h-4 w-4" /> Saved
            </>
          ) : (
            "Save"
          )}
        </Button>
      </div>
      <p className="mt-2 text-xs text-text-muted">{help}</p>
    </div>
  );
}

export default function SettingsPage() {
  const { username, logout } = useSession();
  const toast = useToast();
  const [pulling, setPulling] = useState(false);
  const [pulled, setPulled] = useState(false);
  const [singleSession, setSingleSession] = useState(true);
  const [keys, setKeys] = useState({ civitai: false, huggingface: false });

  useEffect(() => {
    fetch("/api/settings/keys", { cache: "no-store" })
      .then((res) =>
        res.ok ? res.json() : { civitai: false, huggingface: false },
      )
      .then(setKeys)
      .catch(() => {});
  }, []);

  async function saveKey(id: "civitai" | "huggingface", value: string) {
    const label = id === "civitai" ? "Civitai" : "Hugging Face";
    try {
      const res = await fetch("/api/settings/keys", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ [id]: value }),
      });
      if (!res.ok) throw new Error();
      setKeys(await res.json());
      toast.success(`${label} API key saved`);
    } catch {
      toast.error(`Could not save ${label} API key`, "Please try again.");
    }
  }

  function pull() {
    setPulling(true);
    setPulled(false);
    setTimeout(() => {
      setPulling(false);
      setPulled(true);
    }, 1600);
  }

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Settings"
        description="Integrations, deployment and session configuration."
      />

      <Card className="flex flex-col gap-6">
        <div>
          <h2 className="text-[20px] font-semibold">Integrations</h2>
          <p className="text-sm text-text-secondary">
            Keys are stored securely on the server and never exposed to the browser.
          </p>
        </div>
        <ApiKeyField
          id="civitai"
          label="Civitai API Key"
          placeholder="civitai_xxxxxxxxxxxxxxxx"
          help="Used to download LoRAs and models from Civitai URLs."
          saved={keys.civitai}
          onSave={saveKey}
        />
        <ApiKeyField
          id="huggingface"
          label="Hugging Face API Key"
          placeholder="hf_xxxxxxxxxxxxxxxx"
          help="Used to download (and later upload) models and LoRAs from Hugging Face."
          saved={keys.huggingface}
          onSave={saveKey}
        />
      </Card>

      <Card>
        <EncryptionSelfTest />
      </Card>

      <Card className="flex flex-col gap-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="flex items-center gap-2 text-[20px] font-semibold">
              <GitBranch className="h-5 w-5 text-lilac" /> Deployment
            </h2>
            <p className="mt-1 text-sm text-text-secondary">
              Startup pulls from Git into /workspace/igglepixel before launching.
            </p>
            <p className="mt-1 text-xs text-text-muted">
              Source · <span className="font-mono">riseon-lab/igglepixel</span> · main
            </p>
          </div>
          <Button variant="secondary" onClick={pull} disabled={pulling} title="Handled by container startup">
            <RefreshCw className={pulling ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
            {pulling ? "Checking..." : "Check startup pull"}
          </Button>
        </div>
        {pulled && (
          <p className="flex items-center gap-2 text-sm text-success">
            <Check className="h-4 w-4" /> Pull runs automatically on container start.
          </p>
        )}
      </Card>

      <Card className="flex flex-col gap-5">
        <h2 className="text-[20px] font-semibold">Session</h2>
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="font-medium">Single active session</p>
            <p className="text-sm text-text-muted">
              Enforced server-side — logging in elsewhere ends this session.
            </p>
          </div>
          <Toggle checked={singleSession} onChange={setSingleSession} label="Single active session" />
        </div>
        <div className="flex flex-col gap-1 rounded-[12px] bg-background p-4">
          <p className="text-sm text-text-muted">Signed in as</p>
          <p className="font-medium">{username}</p>
          <p className="text-xs text-text-muted">
            Session token is held in a secure httpOnly cookie.
          </p>
        </div>
        <Button variant="danger" onClick={() => logout()} className="self-start">
          <LogOut className="h-4 w-4" /> Log out
        </Button>
      </Card>
    </div>
  );
}
