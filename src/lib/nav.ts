import {
  Activity,
  Images,
  Layers,
  DownloadCloud,
  Database,
  GraduationCap,
  Settings,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  label: string;
  href: string;
  icon: LucideIcon;
  /** Future items are visible but disabled, per plan.md. */
  future?: boolean;
  /** Show in the mobile bottom bar (space is limited). */
  primary?: boolean;
}

export const NAV_ITEMS: NavItem[] = [
  { label: "Running", href: "/running", icon: Activity, primary: true },
  { label: "Assets", href: "/assets", icon: Images, primary: true },
  { label: "LoRAs", href: "/loras", icon: Layers, primary: true },
  { label: "Downloads", href: "/downloads", icon: DownloadCloud, primary: true },
  { label: "Data Studio", href: "/data-studio", icon: Database, future: true },
  { label: "Training", href: "/training", icon: GraduationCap, future: true },
  { label: "Settings", href: "/settings", icon: Settings, primary: true },
];
