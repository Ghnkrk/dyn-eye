"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";

import { LogPanel } from "@/components/dashboard/log-panel";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { pipelineApi, systemApi } from "@/lib/api";
import { useLiveStore } from "@/store/useLiveStore";

export default function OverviewPage() {
  const liveStatus = useLiveStore((s) => s.pipelineStatus);
  const [useCache, setUseCache] = useState(true);

  const statusQuery = useQuery({
    queryKey: ["pipeline", "status"],
    queryFn: pipelineApi.status,
    refetchInterval: 5000, // fallback poll in case the WS drops — WS updates win via setQueryData
  });

  const configQuery = useQuery({ queryKey: ["system", "config"], queryFn: systemApi.config });

  const status = liveStatus ?? statusQuery.data;

  const discoveryMutation = useMutation({
    mutationFn: () => pipelineApi.triggerDiscovery({ use_sample_run: true, use_cache: useCache }),
    onSuccess: () => toast.success("Discovery pipeline triggered"),
    onError: (e: Error) => toast.error(e.message),
  });

  const retrainMutation = useMutation({
    mutationFn: () => pipelineApi.smartRetrain({}),
    onSuccess: () => toast.success("Smart retraining triggered — LLM will advise"),
    onError: (e: Error) => toast.error(e.message),
  });

  const resetMutation = useMutation({
    mutationFn: () => systemApi.resetAll(true),
    onSuccess: () => toast.success("System reset to initial state"),
    onError: (e: Error) => toast.error(e.message),
  });

  const discoveryRunning = status?.discovery.status === "running";
  const retrainingRunning = status?.retraining.status === "running";

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-muted-foreground">Trigger pipeline runs and watch them execute in real time.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Discovery</CardTitle>
            <CardDescription>Run YOLO + DINOv2 + FAISS + HDBSCAN + VLM on the sample run folder.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={useCache} onChange={(e) => setUseCache(e.target.checked)} />
              Use cache (fast demo, skips YOLO/VLM)
            </label>
            <Button disabled={discoveryRunning || discoveryMutation.isPending} onClick={() => discoveryMutation.mutate()}>
              {discoveryRunning ? "Running..." : "Trigger discovery"}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Retraining</CardTitle>
            <CardDescription>LLM-advised: analyzes the dataset, then decides hyperparameters.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <Button disabled={retrainingRunning || retrainMutation.isPending} onClick={() => retrainMutation.mutate()}>
              {retrainingRunning ? "Running..." : "Smart retrain"}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>System</CardTitle>
            <CardDescription>
              {configQuery.data
                ? `${configQuery.data.known_defect_names.length} known classes · ${configQuery.data.input_images_count} input images`
                : "Loading config..."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              variant="destructive"
              disabled={resetMutation.isPending}
              onClick={() => {
                if (confirm("Reset everything to the initial state? This deletes crops, clusters, and fine-tuning data.")) {
                  resetMutation.mutate();
                }
              }}
            >
              Factory reset
            </Button>
          </CardContent>
        </Card>
      </div>

      <LogPanel />
    </div>
  );
}
