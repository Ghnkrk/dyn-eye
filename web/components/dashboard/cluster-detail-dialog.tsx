"use client";

import { ZoomIn } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ImageLightbox } from "@/components/dashboard/image-lightbox";
import { clustersApi } from "@/lib/api";
import type { Cluster } from "@/lib/types";

interface ClusterDetailDialogProps {
  cluster: Cluster | null;
  otherClusters: Cluster[];
  onOpenChange: (open: boolean) => void;
  onChanged: () => void;
}

export function ClusterDetailDialog({ cluster, otherClusters, onOpenChange, onChanged }: ClusterDetailDialogProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [target, setTarget] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  if (!cluster) return null;
  const clusterName = cluster.name;

  function toggle(img: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(img)) next.delete(img);
      else next.add(img);
      return next;
    });
  }

  async function runBatch(action: "move" | "drop") {
    if (selected.size === 0) return;
    if (action === "move" && !target) {
      toast.error("Pick a target cluster first");
      return;
    }
    setBusy(true);
    try {
      await clustersApi.batchEdit({
        crop_files: Array.from(selected),
        source_cluster: clusterName,
        target_cluster: action === "move" ? target : null,
        action,
      });
      toast.success(`${action === "move" ? "Moved" : "Dropped"} ${selected.size} crop(s)`);
      setSelected(new Set());
      onChanged();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Batch action failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={!!cluster} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle className="font-mono">{cluster.name}</DialogTitle>
        </DialogHeader>

        <div className="flex items-center gap-2 border-b pb-3">
          <span className="text-sm text-muted-foreground">{selected.size} selected</span>
          <Select value={target} onValueChange={(value) => setTarget(value ?? "")}>
            <SelectTrigger className="h-8 w-48 text-xs">
              <SelectValue placeholder="Move to..." />
            </SelectTrigger>
            <SelectContent>
              {otherClusters.map((c) => (
                <SelectItem key={c.name} value={c.name}>
                  {c.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button size="sm" variant="outline" disabled={selected.size === 0 || busy} onClick={() => runBatch("move")}>
            Move
          </Button>
          <Button size="sm" variant="destructive" disabled={selected.size === 0 || busy} onClick={() => runBatch("drop")}>
            Drop
          </Button>
        </div>

        <div className="grid max-h-[60vh] grid-cols-6 gap-2 overflow-y-auto">
          {cluster.images.map((img, i) => {
            const isSelected = selected.has(img);
            return (
              <div
                key={img}
                className={`group relative aspect-square overflow-hidden rounded border-2 ${
                  isSelected ? "border-primary" : "border-transparent"
                }`}
              >
                <button
                  type="button"
                  aria-pressed={isSelected}
                  aria-label={`${isSelected ? "Deselect" : "Select"} ${img}`}
                  onClick={() => toggle(img)}
                  className="absolute inset-0 h-full w-full"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={clustersApi.cropImageUrl(cluster.name, img)} alt="" className="h-full w-full object-cover" loading="lazy" />
                </button>
                <button
                  type="button"
                  aria-label={`View ${img} enlarged`}
                  onClick={(e) => {
                    e.stopPropagation();
                    setLightboxIndex(i);
                  }}
                  className="absolute right-1 top-1 rounded bg-black/50 p-1 opacity-0 transition group-hover:opacity-100"
                >
                  <ZoomIn className="h-3 w-3 text-white" />
                </button>
              </div>
            );
          })}
        </div>
      </DialogContent>

      <ImageLightbox
        clusterName={cluster.name}
        images={cluster.images}
        index={lightboxIndex}
        onIndexChange={setLightboxIndex}
      />
    </Dialog>
  );
}
