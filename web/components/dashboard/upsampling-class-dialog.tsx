"use client";

import { useMemo, useState } from "react";

import { BoxedImage } from "@/components/dashboard/boxed-image";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { upsamplingApi } from "@/lib/api";
import type { UpsamplingAugmented, UpsamplingGroup } from "@/lib/types";

export function originLabel(origin: string) {
  if (origin.startsWith("external:")) return `External · ${origin.slice("external:".length)}`;
  if (origin === "cluster") return "Cluster";
  if (origin === "replay") return "Replay";
  return "Unknown source";
}

export function transformLabel(a: Pick<UpsamplingAugmented, "geometric" | "photometric">) {
  const geo = a.geometric === "none" ? "" : a.geometric;
  return [geo, ...a.photometric].filter(Boolean).join(" + ") || "no change";
}

interface Props {
  className: string | null;
  groups: UpsamplingGroup[];
  version: number;
  onOpenChange: (open: boolean) => void;
}

export function UpsamplingClassDialog({ className, groups, version, onOpenChange }: Props) {
  const [geometric, setGeometric] = useState("all");
  const [origin, setOrigin] = useState("all");
  const [compare, setCompare] = useState<{ group: UpsamplingGroup; aug: UpsamplingAugmented } | null>(null);

  const forClass = useMemo(
    () => (className ? groups.filter((g) => g.for_classes.includes(className)) : []),
    [groups, className]
  );
  const geometricOptions = useMemo(
    () => Array.from(new Set(forClass.flatMap((g) => g.augmented.map((a) => a.geometric)))).sort(),
    [forClass]
  );
  const originOptions = useMemo(() => Array.from(new Set(forClass.map((g) => g.origin))).sort(), [forClass]);

  if (!className) return null;

  const rows = forClass
    .filter((g) => origin === "all" || g.origin === origin)
    .map((g) => ({
      ...g,
      augmented: g.augmented.filter(
        (a) => a.for_class === className && (geometric === "all" || a.geometric === geometric)
      ),
    }))
    .filter((g) => g.augmented.length > 0);

  return (
    <>
      <Dialog open={!!className} onOpenChange={onOpenChange}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-5xl">
          <DialogHeader>
            <DialogTitle className="font-mono">{className}</DialogTitle>
          </DialogHeader>

          <div className="flex flex-wrap items-center gap-2 border-b pb-3 text-sm">
            <span className="text-muted-foreground">
              {rows.length} originals · {rows.reduce((n, g) => n + g.augmented.length, 0)} augmented
            </span>
            <div className="ml-auto flex flex-wrap gap-2">
              <Select value={geometric} onValueChange={(v) => setGeometric(v ?? "all")}>
                <SelectTrigger className="h-8 w-40 text-xs">
                  <SelectValue>{(v: string | null) => (v && v !== "all" ? v : "All transforms")}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All transforms</SelectItem>
                  {geometricOptions.map((g) => (
                    <SelectItem key={g} value={g}>
                      {g === "none" ? "photometric only" : g}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={origin} onValueChange={(v) => setOrigin(v ?? "all")}>
                <SelectTrigger className="h-8 w-48 text-xs">
                  <SelectValue>{(v: string | null) => (v && v !== "all" ? originLabel(v) : "All sources")}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All sources</SelectItem>
                  {originOptions.map((o) => (
                    <SelectItem key={o} value={o}>
                      {originLabel(o)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <ul className="flex flex-col gap-4">
            {rows.map((g) => (
              <li key={g.source_image} className="rounded-lg border p-3">
                <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-mono">{g.source_image}</span>
                  <Badge variant="outline">{originLabel(g.origin)}</Badge>
                </div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                  <figure className="flex flex-col gap-1">
                    <BoxedImage
                      src={upsamplingApi.imageUrl(g.source_image)}
                      boxes={g.boxes}
                      className="ring-2 ring-primary/60 rounded"
                    />
                    <figcaption className="text-xs font-medium">Original</figcaption>
                  </figure>
                  {g.augmented.map((a) => (
                    <figure key={a.image} className="flex flex-col gap-1">
                      <button
                        type="button"
                        className="text-left"
                        aria-label={`Compare ${a.image} with its original`}
                        onClick={() => setCompare({ group: g, aug: a })}
                      >
                        <BoxedImage src={upsamplingApi.imageUrl(a.image, version)} boxes={a.boxes} />
                      </button>
                      <figcaption className="text-xs text-muted-foreground">{transformLabel(a)}</figcaption>
                    </figure>
                  ))}
                </div>
              </li>
            ))}
            {rows.length === 0 && <li className="text-sm text-muted-foreground">Nothing matches these filters.</li>}
          </ul>
        </DialogContent>
      </Dialog>

      <Dialog open={!!compare} onOpenChange={(open) => !open && setCompare(null)}>
        {compare && (
          <DialogContent className="sm:max-w-5xl">
            <DialogHeader>
              <DialogTitle className="font-mono text-sm">{compare.aug.image}</DialogTitle>
            </DialogHeader>
            <div className="grid gap-4 md:grid-cols-2">
              <figure className="flex flex-col gap-1">
                <BoxedImage src={upsamplingApi.imageUrl(compare.group.source_image)} boxes={compare.group.boxes} showLabels />
                <figcaption className="text-xs font-medium">Original · {originLabel(compare.group.origin)}</figcaption>
              </figure>
              <figure className="flex flex-col gap-1">
                <BoxedImage src={upsamplingApi.imageUrl(compare.aug.image, version)} boxes={compare.aug.boxes} showLabels />
                <figcaption className="text-xs font-medium">Augmented · {transformLabel(compare.aug)}</figcaption>
              </figure>
            </div>
          </DialogContent>
        )}
      </Dialog>
    </>
  );
}
