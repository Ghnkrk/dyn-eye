"use client";

import { Wifi, WifiOff } from "lucide-react";
import { usePathname } from "next/navigation";

import { Badge } from "@/components/ui/badge";
import { NAV_ITEMS } from "@/components/layout/nav-items";
import { MobileNav } from "@/components/layout/mobile-nav";
import { ThemeToggle } from "@/components/layout/theme-toggle";
import { useLiveStore } from "@/store/useLiveStore";

function statusVariant(status: string | undefined): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "running":
      return "default";
    case "failed":
      return "destructive";
    case "complete":
      return "secondary";
    default:
      return "outline";
  }
}

function usePageTitle(): string {
  const pathname = usePathname();
  const match = NAV_ITEMS.find((item) => (item.href === "/" ? pathname === "/" : pathname.startsWith(item.href)));
  return match?.label ?? "DYN-EYE";
}

export function Topbar() {
  const connected = useLiveStore((s) => s.connected);
  const status = useLiveStore((s) => s.pipelineStatus);
  const title = usePageTitle();

  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b px-4">
      <div className="flex min-w-0 items-center gap-3">
        <MobileNav />
        <span className="truncate text-sm font-semibold md:hidden">{title}</span>
        <div className="hidden items-center gap-2 text-sm text-muted-foreground md:flex">
          {connected ? <Wifi className="h-4 w-4 text-emerald-500" /> : <WifiOff className="h-4 w-4 text-destructive" />}
          <span>{connected ? "Live" : "Reconnecting..."}</span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <div className="hidden items-center gap-2 sm:flex">
          <Badge variant={statusVariant(status?.discovery.status)}>Discovery: {status?.discovery.status ?? "idle"}</Badge>
          <Badge variant={statusVariant(status?.retraining.status)}>Retraining: {status?.retraining.status ?? "idle"}</Badge>
          <Badge variant={status?.orchestrator.status === "running" ? "default" : "outline"}>
            Orchestrator: {status?.orchestrator.status ?? "idle"}
          </Badge>
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
}
