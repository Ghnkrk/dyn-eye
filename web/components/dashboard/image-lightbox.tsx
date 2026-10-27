"use client";

import { ChevronLeft, ChevronRight, X } from "lucide-react";
import type { KeyboardEvent } from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { clustersApi } from "@/lib/api";

interface ImageLightboxProps {
  clusterName: string;
  images: string[];
  index: number | null;
  onIndexChange: (index: number | null) => void;
}

export function ImageLightbox({ clusterName, images, index, onIndexChange }: ImageLightboxProps) {
  const open = index !== null;
  const current = index !== null ? images[index] : null;

  if (!current || index === null) return null;

  const goPrev = () => onIndexChange(Math.max(index - 1, 0));
  const goNext = () => onIndexChange(Math.min(index + 1, images.length - 1));

  // Bound directly to the dialog content (rather than a window listener) so
  // it fires reliably regardless of how the underlying Dialog primitive
  // handles its own internal key events.
  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key === "ArrowRight") goNext();
    if (e.key === "ArrowLeft") goPrev();
  }

  return (
    <Dialog open={open} onOpenChange={(next) => !next && onIndexChange(null)}>
      <DialogContent
        showCloseButton={false}
        onKeyDown={handleKeyDown}
        className="flex max-w-4xl flex-col items-center gap-3 border-none bg-background/95 p-4 sm:max-w-4xl"
      >
        <div className="flex w-full items-center justify-between">
          <span className="font-mono text-xs text-muted-foreground">
            {clusterName} — {current} ({index + 1}/{images.length})
          </span>
          <Button variant="ghost" size="icon" aria-label="Close image preview" onClick={() => onIndexChange(null)}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="relative flex w-full items-center justify-center">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Previous image"
            className="absolute left-0 z-10"
            disabled={index === 0}
            onClick={goPrev}
          >
            <ChevronLeft className="h-5 w-5" />
          </Button>

          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={clustersApi.cropImageUrl(clusterName, current)}
            alt={current}
            className="max-h-[75vh] max-w-full rounded-md object-contain"
          />

          <Button
            variant="ghost"
            size="icon"
            aria-label="Next image"
            className="absolute right-0 z-10"
            disabled={index === images.length - 1}
            onClick={goNext}
          >
            <ChevronRight className="h-5 w-5" />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
