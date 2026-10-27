"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { clustersApi } from "@/lib/api";
import type { Cluster } from "@/lib/types";

interface ClusterCardProps {
  cluster: Cluster;
  otherClusters: Cluster[];
  onNameChange: (name: string, value: string) => void;
  pendingName: string | undefined;
  onOpen: () => void;
  onMerged: () => void;
}

export function ClusterCard({ cluster, otherClusters, onNameChange, pendingName, onOpen, onMerged }: ClusterCardProps) {
  const [mergeTarget, setMergeTarget] = useState<string>("");
  const [merging, setMerging] = useState(false);

  const isSpecial = cluster.name === "noise" || cluster.name === "unassigned";
  const currentName = pendingName ?? cluster.defect_name ?? "";

  async function handleMerge() {
    if (!mergeTarget) return;
    setMerging(true);
    try {
      await clustersApi.merge(cluster.name, mergeTarget);
      onMerged();
    } finally {
      setMerging(false);
    }
  }

  return (
    <Card className={isSpecial ? "border-dashed opacity-80" : undefined}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-mono">{cluster.name}</CardTitle>
          <Badge variant="outline">{cluster.image_count} crops</Badge>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <button
          type="button"
          onClick={onOpen}
          className="grid grid-cols-4 gap-1 rounded-md border bg-muted/30 p-1 text-left"
        >
          {cluster.images.slice(0, 8).map((img) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={img}
              src={clustersApi.cropImageUrl(cluster.name, img)}
              alt=""
              className="aspect-square w-full rounded object-cover"
              loading="lazy"
            />
          ))}
        </button>
        {cluster.cohesion != null && (
          <p className="text-xs text-muted-foreground">Cohesion: {cluster.cohesion.toFixed(3)}</p>
        )}
        <Input
          placeholder="Defect name..."
          value={currentName}
          onChange={(e) => onNameChange(cluster.name, e.target.value)}
        />
      </CardContent>
      <CardFooter className="flex items-center gap-2">
        <Select value={mergeTarget} onValueChange={(value) => setMergeTarget(value ?? "")}>
          <SelectTrigger className="h-8 flex-1 text-xs">
            <SelectValue placeholder="Merge into..." />
          </SelectTrigger>
          <SelectContent>
            {otherClusters.map((c) => (
              <SelectItem key={c.name} value={c.name}>
                {c.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button size="sm" variant="outline" disabled={!mergeTarget || merging} onClick={handleMerge}>
          Merge
        </Button>
      </CardFooter>
    </Card>
  );
}
