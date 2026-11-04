"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { pipelineApi } from "@/lib/api";

function statusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  if (status === "running") return "default";
  if (status === "failed") return "destructive";
  if (status === "success") return "secondary";
  return "outline";
}

export default function RunDetailPage() {
  const params = useParams<{ runId: string }>();
  const { data, isLoading } = useQuery({
    queryKey: ["pipeline", "run", params.runId],
    queryFn: () => pipelineApi.run(params.runId),
  });

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight font-mono">{params.runId}</h1>
        <p className="text-muted-foreground">{data ? `${data.kind} — ${data.status}` : "Loading..."}</p>
      </div>

      {data?.error && (
        <Card className="border-destructive">
          <CardContent className="py-4 text-sm text-destructive">{data.error}</CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Step metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Step</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Latency</TableHead>
                <TableHead>Processed</TableHead>
                <TableHead>Failed</TableHead>
                <TableHead>Error</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.values(data?.steps ?? {}).map((step) => (
                <TableRow key={step.step_name}>
                  <TableCell className="font-mono text-xs">{step.step_name}</TableCell>
                  <TableCell>
                    <Badge variant={statusVariant(step.status)}>{step.status}</Badge>
                  </TableCell>
                  <TableCell>{step.latency_seconds.toFixed(2)}s</TableCell>
                  <TableCell>{step.items_processed}</TableCell>
                  <TableCell>{step.items_failed}</TableCell>
                  <TableCell className="max-w-xs truncate text-xs text-destructive">{step.error ?? ""}</TableCell>
                </TableRow>
              ))}
              {!isLoading && Object.keys(data?.steps ?? {}).length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-muted-foreground">
                    No step metrics for this run.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
