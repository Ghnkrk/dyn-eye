"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { modelsApi, pipelineApi } from "@/lib/api";
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

  const status = liveStatus ?? statusQuery.data;
  const retrainingRunning = status?.retraining.status === "running";
  const versions = versionsQuery.data?.versions ?? [];
  const baseModelLabels: Record<string, string> = { active: "Currently active model" };
  for (const v of versions) {
    baseModelLabels[v.path] = `${v.version_id}${v.status === "deployed" ? " (deployed)" : ""}`;
  }

  const retrainMutation = useMutation({
    mutationFn: () =>
      pipelineApi.smartRetrain({
        base_model_path: baseModel === "active" ? null : baseModel,
      }),
    onSuccess: () => toast.success("Smart retraining triggered — LLM will advise"),
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

      <Card className="max-w-xl">
        <CardHeader>
          <CardTitle>Fine-tune a model</CardTitle>
          <CardDescription>Choose which weights to fine-tune from, then let the LLM advisor pick hyperparameters.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
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

          <Button disabled={retrainingRunning || retrainMutation.isPending} onClick={() => retrainMutation.mutate()}>
            {retrainingRunning ? "Running..." : "Smart retrain"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
