"use client";

import { ZoomIn } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ImageLightbox } from "@/components/dashboard/image-lightbox";
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
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
  const previewImages = cluster.images.slice(0, 8);

  const isSpecial = cluster.name === "noise" || cluster.name === "unassigned";
  const currentName = pendingName ?? cluster.defect_name ?? "";
  const isDirty = pendingName !== undefined && pendingName !== (cluster.defect_name ?? "");

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
          <div className="flex items-center gap-1.5">
            <CardTitle className="text-sm font-mono">{cluster.name}</CardTitle>
            {isDirty && (
              <span
                title="Unsaved name — click Save names to apply"
                className="h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500"
              />
            )}
          </div>
          <Badge variant="outline">{cluster.image_count} crops</Badge>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <div className="grid grid-cols-4 gap-1 rounded-md border bg-muted/30 p-1">
          {previewImages.map((img, i) => (
            <button
              key={img}
              type="button"
              aria-label={`View ${img} enlarged`}
              onClick={() => setLightboxIndex(i)}
              className="group relative aspect-square overflow-hidden rounded"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={clustersApi.cropImageUrl(cluster.name, img)} alt="" className="h-full w-full object-cover" loading="lazy" />
              <span className="absolute inset-0 flex items-center justify-center bg-black/0 opacity-0 transition group-hover:bg-black/30 group-hover:opacity-100">
                <ZoomIn className="h-4 w-4 text-white" />
              </span>
            </button>
          ))}
        </div>
        <Button type="button" variant="link" size="sm" className="h-auto justify-start p-0 text-xs" onClick={onOpen}>
          View all {cluster.image_count} crops &amp; manage &rarr;
        </Button>
        {cluster.cohesion != null && (
          <p className="text-xs text-muted-foreground">Cohesion: {cluster.cohesion.toFixed(3)}</p>
        )}
        <Input
          placeholder="Defect name..."
          value={currentName}
          onChange={(e) => onNameChange(cluster.name, e.target.value)}
          className={isDirty ? "border-amber-500 focus-visible:ring-amber-500/50" : undefined}
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

      <ImageLightbox
        clusterName={cluster.name}
        images={previewImages}
        index={lightboxIndex}
        onIndexChange={setLightboxIndex}
      />
    </Card>
  );
}
