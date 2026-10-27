"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { inferenceApi } from "@/lib/api";
import type { InferenceResult } from "@/lib/types";

function ResultPanel({ title, result }: { title: string; result: InferenceResult | undefined }) {
  return (
    <Card className="flex-1">
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {result ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`data:image/jpeg;base64,${result.image_b64}`} alt={title} className="w-full rounded-md" />
            <p className="text-xs text-muted-foreground">
              {result.model} — {result.num_detections} detection(s)
            </p>
            <ul className="text-xs">
              {result.detections.map((d, i) => (
                <li key={i}>
                  {d.label} — {(d.confidence * 100).toFixed(1)}%
                </li>
              ))}
            </ul>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Upload an image to compare.</p>
        )}
      </CardContent>
    </Card>
  );
}

export default function InferencePage() {
  const [file, setFile] = useState<File | null>(null);
  const [conf, setConf] = useState(0.1);

  const compareMutation = useMutation({
    mutationFn: () => {
      if (!file) throw new Error("No file selected");
      return inferenceApi.compare(file, conf);
    },
    onError: (e: Error) => toast.error(e.message),
  });

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Inference comparison</h1>
        <p className="text-muted-foreground">Run the baseline and finetuned YOLO models on one image, side by side.</p>
      </div>

      <Card>
        <CardContent className="flex flex-wrap items-center gap-3 pt-6">
          <Input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="max-w-xs" />
          <Input
            type="number"
            step="0.05"
            min={0}
            max={1}
            value={conf}
            onChange={(e) => setConf(Number(e.target.value))}
            className="w-24"
          />
          <Button disabled={!file || compareMutation.isPending} onClick={() => compareMutation.mutate()}>
            {compareMutation.isPending ? "Running..." : "Compare"}
          </Button>
        </CardContent>
      </Card>

      <div className="flex flex-col gap-4 md:flex-row">
        <ResultPanel title="Baseline" result={compareMutation.data?.baseline} />
        <ResultPanel title="Finetuned" result={compareMutation.data?.finetuned} />
      </div>
    </div>
  );
}
