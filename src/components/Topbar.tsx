"use client";

import { LogOut, Wand2 } from "lucide-react";
import { useSession } from "@/lib/session";

export function Topbar() {
  const { username, logout } = useSession();

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-background/80 px-4 backdrop-blur sm:px-6 md:px-8">
      <div className="flex items-center gap-2 md:hidden">
        <div className="grid h-8 w-8 place-items-center rounded-[10px] bg-lilac text-white">
          <Wand2 className="h-4 w-4" />
        </div>
        <span className="font-bold">Citivia Studio</span>
      </div>

      <div className="hidden md:block" />

      <div className="flex items-center gap-3">
        {username && (
          <div className="hidden text-right sm:block">
            <p className="text-sm font-medium leading-tight">{username}</p>
            <p className="text-xs text-text-muted">Active session</p>
          </div>
        )}
        <div className="grid h-9 w-9 place-items-center rounded-full bg-surface-hover text-sm font-semibold uppercase">
          {username?.[0] ?? "?"}
        </div>
        <button
          onClick={() => logout()}
          title="Log out"
          className="grid h-9 w-9 place-items-center rounded-full text-text-muted transition-colors hover:bg-surface-hover hover:text-white"
        >
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
}
