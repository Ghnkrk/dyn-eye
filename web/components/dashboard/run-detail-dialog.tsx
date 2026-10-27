"use client";

import { useQuery } from "@tanstack/react-query";

import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { pipelineApi } from "@/lib/api";
import type { StepMetric } from "@/lib/types";

interface RunDetailDialogProps {
  runId: string | null;
  onOpenChange: (open: boolean) => void;
}

function statusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  if (status === "running") return "default";
  if (status === "failed") return "destructive";
  if (status === "success" || status === "complete") return "secondary";
  return "outline";
}

function StepRow({ step }: { step: StepMetric }) {
  const hasMetadata = step.metadata_json && Object.keys(step.metadata_json).length > 0;

  return (
    <div className="flex flex-col gap-2 border-b py-3 last:border-b-0">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-medium">{step.step_name}</span>
        <Badge variant={statusVariant(step.status)} className="text-[10px]">
          {step.status}
        </Badge>
        <span className="text-xs text-muted-foreground">{step.latency_seconds.toFixed(2)}s</span>
        <span className="text-xs text-muted-foreground">
          {step.items_processed} processed{step.items_failed > 0 && ` · ${step.items_failed} failed`}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-4 text-xs text-muted-foreground">
        <span>Start: {step.start_time ? new Date(step.start_time).toLocaleString() : "—"}</span>
        <span>End: {step.end_time ? new Date(step.end_time).toLocaleString() : "—"}</span>
      </div>

      {step.error && (
        <pre className="whitespace-pre-wrap rounded-md border border-destructive/30 bg-destructive/10 p-2 text-xs text-destructive">
          {step.error}
        </pre>
      )}

      {hasMetadata && (
        <pre className="whitespace-pre-wrap rounded-md border bg-muted/30 p-2 text-xs">
          {JSON.stringify(step.metadata_json, null, 2)}
        </pre>
      )}
    </div>
  );
}

export function RunDetailDialog({ runId, onOpenChange }: RunDetailDialogProps) {
  const { data, isLoading } = useQuery({
    queryKey: ["pipeline", "run", runId],
    queryFn: () => pipelineApi.run(runId!),
    enabled: !!runId,
  });

  return (
    <Dialog open={!!runId} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <div className="flex flex-wrap items-center gap-2">
            <DialogTitle className="font-mono">{runId}</DialogTitle>
            {data && (
              <>
                <Badge variant="outline" className="text-[10px]">
                  {data.kind}
                </Badge>
                <Badge variant={statusVariant(data.status)} className="text-[10px]">
                  {data.status}
                </Badge>
              </>
            )}
          </div>
        </DialogHeader>

        {isLoading && <p className="text-sm text-muted-foreground">Loading run detail...</p>}

        {data?.error && (
          <pre className="whitespace-pre-wrap rounded-md border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
            {data.error}
          </pre>
        )}

        {data && (
          <ScrollArea className="max-h-[60vh]">
            <div className="pr-3">
              {Object.values(data.steps).length === 0 && (
                <p className="text-sm text-muted-foreground">No step metrics recorded for this run.</p>
              )}
              {Object.values(data.steps).map((step) => (
                <StepRow key={step.step_name} step={step} />
              ))}
            </div>
          </ScrollArea>
        )}
      </DialogContent>
    </Dialog>
  );
}
