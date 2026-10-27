"use client";

import { useEffect, useRef } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { useLiveStore } from "@/store/useLiveStore";

const LEVEL_COLOR: Record<string, string> = {
  error: "text-destructive",
  warn: "text-amber-500",
  step: "text-primary font-medium",
  success: "text-emerald-500",
  info: "text-muted-foreground",
  progress: "text-muted-foreground",
};

export function LogPanel() {
  const logs = useLiveStore((s) => s.logs);
  const clearLogs = useLiveStore((s) => s.clearLogs);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [logs.length]);

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">Live pipeline log</CardTitle>
        <Button variant="ghost" size="sm" onClick={clearLogs}>
          Clear
        </Button>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden pt-0">
        <ScrollArea className="h-[420px] rounded-md border bg-muted/30 p-3">
          <div className="font-mono text-xs leading-relaxed">
            {logs.length === 0 && <p className="text-muted-foreground">No log events yet.</p>}
            {logs.map((evt, i) => (
              <div key={`${evt.ts}-${i}`} className="flex gap-2">
                <span className="shrink-0 text-muted-foreground">{new Date(evt.ts).toLocaleTimeString()}</span>
                <span className="shrink-0 text-muted-foreground">[{evt.source}]</span>
                <span className={cn(LEVEL_COLOR[evt.level] ?? "text-foreground")}>{evt.message}</span>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
