"use client";

import { clsx } from "clsx";
import { Menu, PanelLeftClose, Sparkles, Wand2 } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";
import { NAV_ITEMS } from "@/lib/nav";

function isActive(pathname: string, href: string) {
  return pathname === href || pathname.startsWith(href + "/");
}

function NavLink({
  href,
  label,
  icon: Icon,
  future,
  collapsed,
  active,
}: {
  href: string;
  label: string;
  icon: typeof Sparkles;
  future?: boolean;
  collapsed: boolean;
  active: boolean;
}) {
  const content = (
    <>
      <Icon className="h-5 w-5 shrink-0" strokeWidth={active ? 2.4 : 2} />
      {!collapsed && (
        <span className="flex-1 truncate text-base">{label}</span>
      )}
      {!collapsed && future && (
        <span className="rounded-md bg-surface-hover px-1.5 py-0.5 text-xs text-text-muted">
          Soon
        </span>
      )}
    </>
  );

  const cls = clsx(
    "flex items-center gap-3 rounded-[12px] px-3 py-3 transition-colors",
    collapsed && "justify-center",
    active
      ? "bg-lilac text-white font-semibold"
      : "text-text-secondary hover:bg-surface-hover hover:text-white",
    future && "pointer-events-none opacity-50",
  );

  if (future) {
    return (
      <div className={cls} aria-disabled title={`${label} — coming soon`}>
        {content}
      </div>
    );
  }
  return (
    <Link href={href} className={cls} title={collapsed ? label : undefined}>
      {content}
    </Link>
  );
}

export function AppShell({
  children,
  topbar,
}: {
  children: ReactNode;
  topbar?: ReactNode;
}) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex min-h-dvh bg-background">
      {/* Desktop sidebar */}
      <aside
        className={clsx(
          "sticky top-0 hidden h-dvh shrink-0 flex-col border-r border-border bg-surface/40 p-3 md:flex transition-[width] duration-200",
          collapsed ? "w-[76px]" : "w-64",
        )}
      >
        <div
          className={clsx(
            "flex items-center gap-2 px-2 py-3",
            collapsed && "justify-center",
          )}
        >
          <div className="grid h-9 w-9 place-items-center rounded-[12px] bg-lilac text-white">
            <Wand2 className="h-5 w-5" />
          </div>
          {!collapsed && (
            <div className="leading-tight">
              <p className="font-bold">Citivia</p>
              <p className="text-xs text-text-muted">Studio</p>
            </div>
          )}
        </div>

        <nav className="mt-2 flex flex-1 flex-col gap-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.href}
              href={item.href}
              label={item.label}
              icon={item.icon}
              future={item.future}
              collapsed={collapsed}
              active={isActive(pathname, item.href)}
            />
          ))}
        </nav>

        <button
          onClick={() => setCollapsed((c) => !c)}
          className="mt-2 flex items-center gap-3 rounded-[12px] px-3 py-3 text-text-muted transition-colors hover:bg-surface-hover hover:text-white"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <Menu className="h-5 w-5" />
          ) : (
            <>
              <PanelLeftClose className="h-5 w-5" />
              <span className="text-base">Collapse</span>
            </>
          )}
        </button>
      </aside>

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {topbar}
        <main className="flex-1 px-4 pb-28 pt-6 sm:px-6 md:px-8 md:pb-10">
          <div className="mx-auto w-full max-w-6xl animate-fade-in">
            {children}
          </div>
        </main>
      </div>

      {/* Mobile bottom navigation */}
      <nav className="fixed inset-x-0 bottom-0 z-40 overflow-x-auto border-t border-border bg-surface/95 backdrop-blur md:hidden">
        <div className="mx-auto flex w-full max-w-lg items-stretch justify-around">
          {NAV_ITEMS.filter((i) => i.primary && !i.future).map((item) => {
            const active = isActive(pathname, item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className="flex min-w-16 flex-1 flex-col items-center gap-1 py-2.5"
              >
                <span
                  className={clsx(
                    "grid h-9 w-12 place-items-center rounded-full transition-colors",
                    active ? "bg-lilac text-white" : "text-text-muted",
                  )}
                >
                  <Icon className="h-5 w-5" strokeWidth={active ? 2.4 : 2} />
                </span>
                <span
                  className={clsx(
                    "text-[11px]",
                    active ? "text-white" : "text-text-muted",
                  )}
                >
                  {item.label}
                </span>
              </Link>
            );
          })}
        </div>
        <div className="h-[env(safe-area-inset-bottom)]" />
      </nav>
    </div>
  );
}
