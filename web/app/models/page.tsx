"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";

import { TableSkeletonRows } from "@/components/dashboard/table-skeleton-rows";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { modelsApi } from "@/lib/api";
import { queryClient } from "@/lib/queryClient";

function statusBadge(status: string) {
  if (status === "deployed") return <Badge>deployed</Badge>;
  if (status === "retired") return <Badge variant="secondary">retired</Badge>;
  return <Badge variant="outline">{status}</Badge>;
}

export default function ModelsPage() {
  const versionsQuery = useQuery({ queryKey: ["models", "versions"], queryFn: modelsApi.list });
  const historyQuery = useQuery({ queryKey: ["models", "history"], queryFn: modelsApi.history });

  function refresh() {
    queryClient.invalidateQueries({ queryKey: ["models"] });
  }

  const deployMutation = useMutation({
    mutationFn: (versionId: string) => modelsApi.deploy(versionId),
    onSuccess: (res) => {
      if (res.success) toast.success("Deployed");
      else toast.error(res.error ?? "Deploy failed");
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const rollbackMutation = useMutation({
    mutationFn: (versionId: string) => modelsApi.rollback(versionId),
    onSuccess: (res) => {
      if (res.success) toast.success("Rolled back");
      else toast.error(res.error ?? "Rollback failed");
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const versions = versionsQuery.data?.versions ?? [];
  const current = versionsQuery.data?.current;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Models</h1>
        <p className="text-muted-foreground">Active version: {current ?? "none"}</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Versions</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Version</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Classes</TableHead>
                <TableHead>mAP50</TableHead>
                <TableHead>Size</TableHead>
                <TableHead>Source</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {versionsQuery.isLoading && <TableSkeletonRows columns={7} />}
              {versions.map((v) => (
                <TableRow key={v.version_id}>
                  <TableCell className="font-mono text-xs">{v.version_id}</TableCell>
                  <TableCell>{statusBadge(v.status)}</TableCell>
                  <TableCell className="max-w-xs truncate text-xs text-muted-foreground">
                    {v.classes.join(", ")}
                  </TableCell>
                  <TableCell>{v.metrics?.map50 != null ? Number(v.metrics.map50).toFixed(3) : "—"}</TableCell>
                  <TableCell>{v.size_mb.toFixed(1)} MB</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{v.source}</TableCell>
                  <TableCell className="text-right">
                    {v.status === "deployed" ? (
                      <span className="text-xs text-muted-foreground">active</span>
                    ) : (
                      <div className="flex justify-end gap-2">
                        <Button size="sm" variant="outline" onClick={() => rollbackMutation.mutate(v.version_id)}>
                          Rollback
                        </Button>
                        <Button size="sm" onClick={() => deployMutation.mutate(v.version_id)}>
                          Deploy
                        </Button>
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              ))}
              {!versionsQuery.isLoading && versions.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} className="text-center text-muted-foreground">
                    No model versions registered yet.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Deployment history</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>When</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Previous</TableHead>
                <TableHead>By</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {historyQuery.isLoading && <TableSkeletonRows columns={5} />}
              {(historyQuery.data?.history ?? []).map((h, i) => (
                <TableRow key={`${h.version_id}-${h.deployed_at}-${i}`}>
                  <TableCell className="text-xs text-muted-foreground">{new Date(h.deployed_at).toLocaleString()}</TableCell>
                  <TableCell>
                    <Badge variant={h.action === "rollback" ? "secondary" : "default"}>{h.action}</Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs">{h.version_id}</TableCell>
                  <TableCell className="font-mono text-xs text-muted-foreground">{h.previous_version ?? "—"}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{h.confirmed_by}</TableCell>
                </TableRow>
              ))}
              {!historyQuery.isLoading && (historyQuery.data?.history ?? []).length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-center text-muted-foreground">
                    No deployment events yet.
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
