"use client";

import { useQuery } from "@tanstack/react-query";
import { ChevronRight } from "lucide-react";
import { Fragment, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { pipelineApi } from "@/lib/api";
import type { PipelineGraph as Graph } from "@/lib/types";
import { useLiveStore } from "@/store/useLiveStore";

type NodeState = "success" | "running" | "failed" | "queued" | "skipped" | "none";

// Airflow's task-state colours
const STATE_STYLE: Record<NodeState, { label: string; color: string }> = {
  success: { label: "success", color: "#16a34a" },
  running: { label: "running", color: "#65a30d" },
  failed: { label: "failed", color: "#dc2626" },
  queued: { label: "queued", color: "#78716c" },
  skipped: { label: "skipped", color: "#db2777" },
  none: { label: "no status", color: "#a8a29e" },
};

function formatDuration(seconds: number) {
  if (!seconds) return "";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  return `${m}m ${Math.round(seconds - m * 60)}s`;
}

function nodeStates(graph: Graph): { states: NodeState[]; stoppedAfter: number | null } {
  const running = graph.run_status === "running";
  let lastSeen = -1;
  graph.nodes.forEach((n, i) => {
    if (graph.steps[n.id]) lastSeen = i;
  });
  const failed = graph.nodes.some((n) => graph.steps[n.id]?.status === "failed");
  const states = graph.nodes.map((n, i): NodeState => {
    const step = graph.steps[n.id];
    if (step) {
      if (step.status === "success") return "success";
      if (step.status === "failed") return "failed";
      // A step left "running" after the run ended was interrupted
      return running ? "running" : "failed";
    }
    if (!graph.run_id) return "none";
    if (running) return "queued";
    return i > lastSeen ? (failed ? "none" : "skipped") : "none";
  });
  // A finished run that never reached the last node stopped at a gate
  const stoppedAfter =
    !running && !failed && lastSeen >= 0 && lastSeen < graph.nodes.length - 1 ? lastSeen : null;
  return { states, stoppedAfter };
}

function runLabel(runId: string | null) {
  if (!runId) return "";
  const m = runId.replace("retrain_", "").match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]} UTC` : runId;
}

export function PipelineGraph() {
  const [kind, setKind] = useState<"discovery" | "retraining">("discovery");
  const live = useLiveStore((s) => s.pipelineStatus);
  const liveRunning = live?.[kind]?.status === "running";

  const { data: graph } = useQuery({
    queryKey: ["pipeline", "graph", kind, live?.[kind]?.status],
    queryFn: () => pipelineApi.graph(kind),
    refetchInterval: (q) => (q.state.data?.run_status === "running" || liveRunning ? 1500 : 15000),
  });

  const { states, stoppedAfter } = graph ? nodeStates(graph) : { states: [], stoppedAfter: null };
  const done = states.filter((s) => s === "success").length;
  const total = graph?.nodes.length ?? 0;
  const totalSeconds = graph ? Object.values(graph.steps).reduce((t, s) => t + (s.latency_seconds || 0), 0) : 0;
  const maxSeconds = graph ? Math.max(1, ...Object.values(graph.steps).map((s) => s.latency_seconds || 0)) : 1;

  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <CardTitle>Pipeline graph</CardTitle>
          <CardDescription>
            {graph?.run_id ? (
              <>
                Latest run {runLabel(graph.run_id)} · {done}/{total} steps done
                {totalSeconds > 0 && <> · {formatDuration(totalSeconds)}</>}
              </>
            ) : (
              "No runs yet."
            )}
          </CardDescription>
        </div>
        <div className="flex items-center gap-2">
          {graph && (
            <Badge variant={graph.run_status === "failed" ? "destructive" : "outline"}>
              {graph.run_status === "running" && (
                <span className="mr-1.5 inline-block h-2 w-2 animate-pulse rounded-full bg-lime-500" />
              )}
              {graph.run_status}
            </Badge>
          )}
          <div className="flex rounded-md border p-0.5 text-sm" role="tablist">
            {(["discovery", "retraining"] as const).map((k) => (
              <button
                key={k}
                type="button"
                role="tab"
                aria-selected={kind === k}
                onClick={() => setKind(k)}
                className={`rounded px-2.5 py-1 ${kind === k ? "bg-muted font-medium" : "text-muted-foreground"}`}
              >
                {k === "discovery" ? "Discovery" : "FineTuning"}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {/* progress bar */}
        <div className="flex h-1.5 gap-0.5 overflow-hidden rounded-full">
          {states.map((s, i) => (
            <div
              key={i}
              className={`flex-1 ${s === "running" ? "animate-pulse" : ""}`}
              style={{ backgroundColor: s === "none" ? "var(--muted)" : STATE_STYLE[s].color, opacity: s === "queued" ? 0.35 : 1 }}
            />
          ))}
        </div>

        {/* DAG */}
        <div className="overflow-x-auto pb-2">
          <div className="flex min-w-max items-center">
            {graph?.nodes.map((n, i) => {
              const state = states[i];
              const step = graph.steps[n.id];
              const style = STATE_STYLE[state];
              const cached = step?.metadata?.cached === true;
              return (
                <Fragment key={n.id}>
                  {i > 0 && (
                    <div className="flex flex-col items-center px-1">
                      <ChevronRight
                        className="h-4 w-4"
                        style={{ color: states[i - 1] === "success" && state !== "skipped" ? STATE_STYLE.success.color : "var(--muted-foreground)" }}
                      />
                    </div>
                  )}
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <div
                          className={`flex w-36 flex-col gap-1 rounded-md border bg-card px-2.5 py-2 text-left ${state === "skipped" || state === "none" ? "border-dashed opacity-70" : ""}`}
                          style={{ borderLeft: `4px solid ${style.color}` }}
                        />
                      }
                    >
                      <span className="truncate text-xs font-medium">{n.label}</span>
                      <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                        <span
                          className={`h-2 w-2 shrink-0 rounded-full ${state === "running" ? "animate-pulse" : ""}`}
                          style={{ backgroundColor: style.color }}
                        />
                        {style.label}
                        {cached && " · cached"}
                      </span>
                      <span className="text-[11px] tabular-nums text-muted-foreground">
                        {step && step.status === "success"
                          ? `${formatDuration(step.latency_seconds) || "<0.1s"} · ${step.items_processed} items`
                          : " "}
                      </span>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <div className="flex flex-col gap-0.5 text-xs">
                        <span className="font-mono">{n.id}</span>
                        <span>State: {style.label}</span>
                        {step?.start_time && <span>Started: {new Date(step.start_time).toLocaleTimeString()}</span>}
                        {step?.latency_seconds ? <span>Duration: {formatDuration(step.latency_seconds)}</span> : null}
                        {step && <span>Items: {step.items_processed}</span>}
                        {step?.error && <span className="text-red-400">Error: {step.error}</span>}
                        {n.gate && <span className="text-muted-foreground">Stops the run if: {n.gate}</span>}
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </Fragment>
              );
            })}
          </div>
        </div>

        {stoppedAfter !== null && graph && (
          <p className="text-xs text-muted-foreground">
            The run ended after <span className="font-medium text-foreground">{graph.nodes[stoppedAfter].label}</span>
            {graph.nodes[stoppedAfter].gate ? ` (${graph.nodes[stoppedAfter].gate})` : ""}; later steps were skipped.
          </p>
        )}
        {graph?.error && graph.run_status === "failed" && <p className="text-xs text-destructive">{graph.error}</p>}

        {/* Gantt-style durations */}
        {graph && totalSeconds > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium">Step durations</span>
            {graph.nodes.map((n, i) => {
              const step = graph.steps[n.id];
              if (!step) return null;
              return (
                <div key={n.id} className="grid grid-cols-[8rem_1fr_4rem] items-center gap-2 text-xs">
                  <span className="truncate text-muted-foreground">{n.label}</span>
                  <div className="h-2 rounded-full bg-muted">
                    <div
                      className={`h-2 rounded-full ${states[i] === "running" ? "animate-pulse" : ""}`}
                      style={{
                        width: `${Math.max(2, ((step.latency_seconds || 0) / maxSeconds) * 100)}%`,
                        backgroundColor: STATE_STYLE[states[i]].color,
                      }}
                    />
                  </div>
                  <span className="text-right tabular-nums text-muted-foreground">
                    {states[i] === "running" ? "…" : formatDuration(step.latency_seconds) || "<0.1s"}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {/* legend */}
        <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
          {(["success", "running", "failed", "queued", "skipped"] as const).map((s) => (
            <span key={s} className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: STATE_STYLE[s].color }} />
              {STATE_STYLE[s].label}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
