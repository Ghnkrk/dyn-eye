"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { modelsApi, pipelineApi, upsamplingApi } from "@/lib/api";
import { useLiveStore } from "@/store/useLiveStore";

export default function FineTuningPage() {
  const liveStatus = useLiveStore((s) => s.pipelineStatus);
  const [baseModel, setBaseModel] = useState<string>("active");

  const statusQuery = useQuery({
    queryKey: ["pipeline", "status"],
    queryFn: pipelineApi.status,
    refetchInterval: 5000,
  });
  const versionsQuery = useQuery({ queryKey: ["models", "versions"], queryFn: modelsApi.list });
  const upsamplingQuery = useQuery({ queryKey: ["upsampling", "status"], queryFn: upsamplingApi.status });
  const upsampling = upsamplingQuery.data?.settings;
  // Calls the LLM, so only on demand
  const previewQuery = useQuery({
    queryKey: ["advisor-preview"],
    queryFn: pipelineApi.advisorPreview,
    enabled: false,
    retry: false,
  });

  const status = liveStatus ?? statusQuery.data;
  const retrainingRunning = status?.retraining.status === "running";
  const versions = versionsQuery.data?.versions ?? [];
  const baseModelLabels: Record<string, string> = { active: "Currently active model" };
  for (const v of versions) {
    baseModelLabels[v.path] = `${v.version_id}${v.status === "deployed" ? " (deployed)" : ""}`;
  }

  const preview = previewQuery.data;
  const rec = preview?.recommendation;

  const retrainMutation = useMutation({
    mutationFn: () =>
      pipelineApi.smartRetrain({
        base_model_path: baseModel === "active" ? null : baseModel,
      }),
    onSuccess: () =>
      toast.success(upsampling?.enabled ? "Fine-tuning started with class-balancing upsampling" : "Fine-tuning started"),
    onError: (e: Error) => toast.error(e.message),
  });

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">FineTuning</h1>
        <p className="text-muted-foreground">
          LLM-advised retraining: analyzes the labeled dataset, then decides hyperparameters.
        </p>
      </div>

      <div className="grid items-start gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Fine-tune a model</CardTitle>
            <CardDescription>
              Choose which weights to fine-tune from, then let the LLM advisor pick hyperparameters.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">Base model</label>
              <Select value={baseModel} onValueChange={(value) => setBaseModel(value ?? "active")}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Currently active model">
                    {(value: string | null) => (value ? (baseModelLabels[value] ?? value) : "Currently active model")}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="active">Currently active model</SelectItem>
                  {versions.map((v) => (
                    <SelectItem key={v.version_id} value={v.path}>
                      {v.version_id} {v.status === "deployed" ? "(deployed)" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Pick a baseline from the <a className="underline" href="/models">Models</a> registry to fine-tune a
                different version instead of the one currently deployed.
              </p>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border p-3 text-sm">
              <span>
                Upsampling:{" "}
                {upsampling ? (
                  upsampling.enabled ? (
                    <span className="font-medium">
                      on, minorities up to {upsampling.target_ratio * 100}% of the largest class (max{" "}
                      {upsampling.max_factor}×)
                    </span>
                  ) : (
                    <span className="font-medium">off</span>
                  )
                ) : (
                  "..."
                )}
              </span>
              <a className="text-xs underline" href="/upsampling">
                Configure on the Upsampling page
              </a>
            </div>

            <Button disabled={retrainingRunning || retrainMutation.isPending} onClick={() => retrainMutation.mutate()}>
              {retrainingRunning ? "Running..." : "Smart retrain"}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Advisor preview</CardTitle>
            <CardDescription>What the advisor would decide for the current named clusters, before you start.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <Button
              variant="outline"
              disabled={previewQuery.isFetching}
              onClick={() =>
                previewQuery.refetch().then((r) => {
                  if (r.error) toast.error(r.error.message);
                })
              }
            >
              {previewQuery.isFetching ? "Asking the advisor..." : preview ? "Analyze again" : "Analyze dataset"}
            </Button>

            {preview && rec && (
              <>
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge variant={rec.should_train ? "secondary" : "destructive"}>
                    {rec.should_train ? "Ready to train" : "Not ready to train"}
                  </Badge>
                </div>

                <dl className="flex flex-col gap-2 text-sm">
                  <div>
                    <dt className="text-xs text-muted-foreground">Training</dt>
                    <dd>{rec.reason}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-muted-foreground">Planned hyperparameters</dt>
                    <dd className="tabular-nums">
                      {rec.config.epochs} epochs · batch {rec.config.batch} · freeze {rec.config.freeze ?? "none"} ·{" "}
                      {rec.config.optimizer}
                    </dd>
                  </div>
                </dl>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
