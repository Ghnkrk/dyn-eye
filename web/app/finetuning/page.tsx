"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { modelsApi, pipelineApi } from "@/lib/api";
import type { AdvisorPreview } from "@/lib/types";
import { useLiveStore } from "@/store/useLiveStore";

const TARGET_LABELS: Record<string, string> = {
  "1": "the size of the largest class",
  "0.75": "75% of the largest class",
  "0.5": "50% of the largest class",
};

function ClassBars({ preview }: { preview: AdvisorPreview }) {
  const { classes, class_balance: balance } = preview.metadata;
  const entries = Object.entries(classes).sort((a, b) => b[1] - a[1]);
  const largest = Math.max(1, ...entries.map(([, n]) => n));
  const minority = new Set(balance.minority_classes);

  if (entries.length === 0) {
    return <p className="text-sm text-muted-foreground">No named clusters yet. Name clusters on the Clusters page first.</p>;
  }

  return (
    <ul className="flex flex-col gap-2">
      {entries.map(([name, n]) => (
        <li key={name} className="flex flex-col gap-1">
          <div className="flex items-baseline justify-between gap-2 text-sm">
            <span className="truncate">{name}</span>
            <span className="tabular-nums text-muted-foreground">{n}</span>
          </div>
          <div className="h-2 rounded-full bg-muted">
            <div
              className={minority.has(name) ? "h-2 rounded-full bg-amber-500" : "h-2 rounded-full bg-primary"}
              style={{ width: `${(n / largest) * 100}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function FineTuningPage() {
  const liveStatus = useLiveStore((s) => s.pipelineStatus);
  const [baseModel, setBaseModel] = useState<string>("active");
  const [upsample, setUpsample] = useState(false);
  const [targetRatio, setTargetRatio] = useState<string>("1");

  const statusQuery = useQuery({
    queryKey: ["pipeline", "status"],
    queryFn: pipelineApi.status,
    refetchInterval: 5000,
  });
  const versionsQuery = useQuery({ queryKey: ["models", "versions"], queryFn: modelsApi.list });
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
  const maxFactor = preview?.metadata.class_balance.max_factor ?? 3;

  const retrainMutation = useMutation({
    mutationFn: () =>
      pipelineApi.smartRetrain({
        base_model_path: baseModel === "active" ? null : baseModel,
        upsample,
        upsample_target_ratio: upsample ? Number(targetRatio) : null,
      }),
    onSuccess: () =>
      toast.success(upsample ? "Fine-tuning started with class-balancing upsampling" : "Fine-tuning started"),
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

            <div className="flex flex-col gap-2 rounded-lg border p-3">
              <label htmlFor="upsample" className="flex items-start gap-2 text-sm font-medium">
                <input
                  id="upsample"
                  type="checkbox"
                  className="mt-0.5"
                  checked={upsample}
                  onChange={(e) => setUpsample(e.target.checked)}
                />
                <span className="flex flex-col gap-0.5">
                  Upsample minority classes before training
                  <span className="text-xs font-normal text-muted-foreground">
                    Adds flipped, brightness- and noise-varied copies of training images that contain under-represented
                    classes. The validation set is never changed.
                  </span>
                </span>
              </label>
              {upsample && (
                <div className="flex flex-col gap-1.5 pl-6">
                  <label className="text-xs text-muted-foreground">Bring smaller classes up to</label>
                  <Select value={targetRatio} onValueChange={(value) => setTargetRatio(value ?? "1")}>
                    <SelectTrigger className="w-full" size="sm">
                      <SelectValue>{(value: string | null) => TARGET_LABELS[value ?? "1"]}</SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(TARGET_LABELS).map(([value, label]) => (
                        <SelectItem key={value} value={value}>
                          {label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    A class is never grown past {maxFactor}× its own size, so very small classes may stay below the
                    target.
                  </p>
                </div>
              )}
              {!upsample && rec?.recommend_upsampling && (
                <div className="flex items-center justify-between gap-2 pl-6 text-xs text-amber-600 dark:text-amber-400">
                  <span>The advisor recommends upsampling for this dataset.</span>
                  <Button size="sm" variant="outline" onClick={() => setUpsample(true)}>
                    Turn on
                  </Button>
                </div>
              )}
            </div>

            <Button disabled={retrainingRunning || retrainMutation.isPending} onClick={() => retrainMutation.mutate()}>
              {retrainingRunning ? "Running..." : "Smart retrain"}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Dataset balance</CardTitle>
            <CardDescription>Boxes per named class, and what the advisor recommends before you start.</CardDescription>
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
                <ClassBars preview={preview} />

                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge variant="outline">{preview.metadata.class_balance.imbalance_ratio}× imbalance</Badge>
                  <Badge variant={rec.recommend_upsampling ? "default" : "secondary"}>
                    {rec.recommend_upsampling ? "Upsampling recommended" : "Upsampling not needed"}
                  </Badge>
                  <Badge variant={rec.should_train ? "secondary" : "destructive"}>
                    {rec.should_train ? "Ready to train" : "Not ready to train"}
                  </Badge>
                </div>

                <dl className="flex flex-col gap-2 text-sm">
                  <div>
                    <dt className="text-xs text-muted-foreground">Class balance</dt>
                    <dd>{rec.upsampling_reason}</dd>
                  </div>
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
