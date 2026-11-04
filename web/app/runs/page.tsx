"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { RunDetailDialog } from "@/components/dashboard/run-detail-dialog";
import { TableSkeletonRows } from "@/components/dashboard/table-skeleton-rows";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { pipelineApi } from "@/lib/api";

function statusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  if (status === "running") return "default";
  if (status === "failed") return "destructive";
  if (status === "complete") return "secondary";
  return "outline";
}

export default function RunsPage() {
  const { data, isLoading } = useQuery({ queryKey: ["pipeline", "runs"], queryFn: pipelineApi.runs });
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Pipeline runs</h1>
        <p className="text-muted-foreground">Per-step latency and status for every discovery/retraining run.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Run history</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Run ID</TableHead>
                <TableHead>Kind</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Steps</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading && <TableSkeletonRows columns={4} />}
              {(data ?? []).map((run) => (
                <TableRow
                  key={run.run_id}
                  className="cursor-pointer"
                  onClick={() => setSelectedRunId(run.run_id)}
                >
                  <TableCell>
                    <span className="font-mono text-xs text-primary hover:underline">{run.run_id}</span>
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">{run.kind}</TableCell>
                  <TableCell>
                    <Badge variant={statusVariant(run.status)}>{run.status}</Badge>
                  </TableCell>
                  <TableCell className="flex flex-wrap gap-1">
                    {run.steps.map((step) => (
                      <Badge key={step} variant={statusVariant(run.summary[step] ?? "")} className="text-[10px]">
                        {step}
                      </Badge>
                    ))}
                  </TableCell>
                </TableRow>
              ))}
              {!isLoading && (data ?? []).length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} className="text-center text-muted-foreground">
                    No pipeline runs recorded yet.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <RunDetailDialog runId={selectedRunId} onOpenChange={(open) => !open && setSelectedRunId(null)} />
    </div>
  );
}
