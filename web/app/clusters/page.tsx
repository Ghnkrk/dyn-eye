"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";

import { ClusterCard } from "@/components/dashboard/cluster-card";
import { ClusterCardSkeleton } from "@/components/dashboard/cluster-card-skeleton";
import { ClusterDetailDialog } from "@/components/dashboard/cluster-detail-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { clustersApi } from "@/lib/api";
import { queryClient } from "@/lib/queryClient";
import type { Cluster } from "@/lib/types";

export default function ClustersPage() {
  const [pendingNames, setPendingNames] = useState<Record<string, string>>({});
  const [openCluster, setOpenCluster] = useState<Cluster | null>(null);
  const [saving, setSaving] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["clusters"],
    queryFn: clustersApi.list,
    refetchInterval: 10000,
  });

  function refresh() {
    queryClient.invalidateQueries({ queryKey: ["clusters"] });
  }

  async function saveNames() {
    const names = Object.fromEntries(Object.entries(pendingNames).filter(([, v]) => v.trim().length > 0));
    if (Object.keys(names).length === 0) return;
    setSaving(true);
    try {
      const result = await clustersApi.name(names);
      toast.success(result.message);
      setPendingNames({});
      refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to save names");
    } finally {
      setSaving(false);
    }
  }

  const clusters = data?.clusters ?? [];
  const hasPending = Object.keys(pendingNames).length > 0;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Clusters</h1>
          <p className="text-muted-foreground">
            {data?.pipeline_running
              ? "Discovery pipeline is running — clusters hidden until it finishes."
              : `${clusters.length} clusters · mean cohesion ${data?.mean_cohesion?.toFixed(3) ?? "—"}`}
          </p>
        </div>
        <Button disabled={!hasPending || saving} onClick={saveNames}>
          Save names
        </Button>
      </div>

      {!isLoading && clusters.length === 0 && (
        <Card>
          <CardContent className="py-10 text-center text-muted-foreground">
            No clusters yet. Run the discovery pipeline from the Overview page.
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {isLoading && Array.from({ length: 4 }).map((_, i) => <ClusterCardSkeleton key={i} />)}
        {clusters.map((cluster) => (
          <ClusterCard
            key={cluster.name}
            cluster={cluster}
            otherClusters={clusters.filter((c) => c.name !== cluster.name)}
            pendingName={pendingNames[cluster.name]}
            onNameChange={(name, value) => setPendingNames((prev) => ({ ...prev, [name]: value }))}
            onOpen={() => setOpenCluster(cluster)}
            onMerged={refresh}
          />
        ))}
      </div>

      <ClusterDetailDialog
        cluster={openCluster}
        otherClusters={clusters.filter((c) => c.name !== openCluster?.name)}
        onOpenChange={(open) => !open && setOpenCluster(null)}
        onChanged={refresh}
      />
    </div>
  );
}
