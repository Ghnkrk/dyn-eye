"use client";

import { Wifi, WifiOff } from "lucide-react";

import { Badge } from "@/components/ui/badge";
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

export function Topbar() {
  const connected = useLiveStore((s) => s.connected);
  const status = useLiveStore((s) => s.pipelineStatus);

  return (
    <header className="flex h-14 items-center justify-between border-b px-4">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        {connected ? <Wifi className="h-4 w-4 text-emerald-500" /> : <WifiOff className="h-4 w-4 text-destructive" />}
        <span>{connected ? "Live" : "Reconnecting..."}</span>
      </div>
      <div className="flex items-center gap-2">
        <Badge variant={statusVariant(status?.discovery.status)}>Discovery: {status?.discovery.status ?? "idle"}</Badge>
        <Badge variant={statusVariant(status?.retraining.status)}>Retraining: {status?.retraining.status ?? "idle"}</Badge>
        <Badge variant={status?.orchestrator.status === "running" ? "default" : "outline"}>
          Orchestrator: {status?.orchestrator.status ?? "idle"}
        </Badge>
      </div>
    </header>
  );
}
