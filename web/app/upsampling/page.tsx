"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";

import { BoxedImage } from "@/components/dashboard/boxed-image";
import { acknowledge, useAttention } from "@/components/layout/attention";
import { UpsamplingClassDialog, originLabel } from "@/components/dashboard/upsampling-class-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { upsamplingApi } from "@/lib/api";
import { queryClient } from "@/lib/queryClient";
import type { UpsamplingSettings, UpsamplingTransforms } from "@/lib/types";

const RATIO_OPTIONS: Record<string, string> = {
  "0.25": "25% of the largest class",
  "0.5": "50% of the largest class",
  "0.75": "75% of the largest class",
  "1": "100% (the largest class's size)",
};
const FACTOR_OPTIONS: Record<string, string> = {
  "2": "2× its original boxes",
  "3": "3× its original boxes",
  "4": "4× its original boxes",
  "5": "5× its original boxes",
};
const THRESHOLD_OPTIONS: Record<string, string> = {
  "1.5": "≤ 1/1.5 of the largest (67%)",
  "2": "≤ 1/2 of the largest (50%)",
  "3": "≤ 1/3 of the largest (33%)",
  "4": "≤ 1/4 of the largest (25%)",
};
const TRANSFORMS: { key: keyof UpsamplingTransforms; label: string; hint: string }[] = [
  { key: "hflip", label: "Horizontal flip", hint: "geometric" },
  { key: "vflip", label: "Vertical flip", hint: "geometric" },
  { key: "rot180", label: "Rotate 180°", hint: "geometric" },
  { key: "rot90", label: "Rotate 90°", hint: "off by default: steel defects follow the rolling direction" },
  { key: "contrast", label: "Brightness / contrast", hint: "photometric" },
  { key: "gamma", label: "Gamma", hint: "photometric" },
  { key: "blur", label: "Light blur", hint: "photometric" },
  { key: "noise", label: "Sensor noise", hint: "photometric" },
];

function sameSettings(a: UpsamplingSettings | null, b: UpsamplingSettings | null) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function OptionSelect({
  label,
  help,
  value,
  options,
  onChange,
}: {
  label: string;
  help: string;
  value: number;
  options: Record<string, string>;
  onChange: (v: number) => void;
}) {
  const key = String(value);
  const all = key in options ? options : { ...options, [key]: `${key} (custom)` };
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-medium">{label}</label>
      <Select value={key} onValueChange={(v) => v && onChange(Number(v))}>
        <SelectTrigger className="w-full">
          <SelectValue>{(v: string | null) => all[v ?? key]}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {Object.entries(all).map(([v, l]) => (
            <SelectItem key={v} value={v}>
              {l}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">{help}</p>
    </div>
  );
}

function BalanceBars({
  before,
  after,
  minority,
}: {
  before: Record<string, number>;
  after: Record<string, number> | null;
  minority: Set<string>;
}) {
  const names = Object.keys(before).sort((a, b) => before[b] - before[a]);
  const largest = Math.max(1, ...names.map((n) => Math.max(before[n], after?.[n] ?? 0)));
  return (
    <ul className="flex flex-col gap-2.5">
      {names.map((n) => {
        const b = before[n];
        const a = after?.[n] ?? b;
        return (
          <li key={n} className="flex flex-col gap-1">
            <div className="flex items-baseline justify-between gap-2 text-sm">
              <span className="flex items-center gap-1.5 truncate">
                {n}
                {minority.has(n) && <span className="text-xs text-amber-600 dark:text-amber-400">minority</span>}
              </span>
              <span className="tabular-nums text-muted-foreground">
                {b}
                {a !== b && <span className="text-foreground"> → {a}</span>}
              </span>
            </div>
            <div className="relative h-2 rounded-full bg-muted">
              <div className="absolute inset-y-0 left-0 rounded-full bg-primary/35" style={{ width: `${(a / largest) * 100}%` }} />
              <div
                className={`absolute inset-y-0 left-0 rounded-full ${minority.has(n) ? "bg-amber-500" : "bg-primary"}`}
                style={{ width: `${(b / largest) * 100}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export default function UpsamplingPage() {
  // Local edits; until the user changes something, the saved settings are shown
  const [draft, setDraft] = useState<UpsamplingSettings | null>(null);
  const [openClass, setOpenClass] = useState<string | null>(null);
  const [version, setVersion] = useState(() => Date.now());
  const { items: attentionItems } = useAttention();

  const statusQuery = useQuery({ queryKey: ["upsampling", "status"], queryFn: upsamplingApi.status });
  const groupsQuery = useQuery({ queryKey: ["upsampling", "groups"], queryFn: upsamplingApi.groups });
  const status = statusQuery.data;
  const settings = draft ?? status?.settings ?? null;
  const setSettings = (fn: (s: UpsamplingSettings | null) => UpsamplingSettings | null) => setDraft(fn(settings));

  // Dry run: recomputed whenever an option changes; reads label files only
  const previewQuery = useQuery({
    queryKey: ["upsampling", "preview", settings],
    queryFn: () => upsamplingApi.preview(settings!),
    enabled: !!settings && !!status?.dataset_ready,
    retry: false,
  });

  function refresh() {
    queryClient.invalidateQueries({ queryKey: ["upsampling"] });
    queryClient.invalidateQueries({ queryKey: ["attention"] });
  }

  const onError = (e: Error) => toast.error(e.message);
  const prepareMutation = useMutation({
    mutationFn: upsamplingApi.prepare,
    onSuccess: (r) => {
      toast.success(
        `Training set rebuilt: ${r.total_mapped} cluster boxes, ${r.replay_images} replay and ${r.external_images} external images`
      );
      refresh();
    },
    onError,
  });
  const generateMutation = useMutation({
    mutationFn: () => upsamplingApi.generate(settings!),
    onSuccess: (r) => {
      toast.success(`Generated ${r.generated_images} augmented images. The next fine-tune will use these settings.`);
      setDraft((s) => (s ? { ...s, enabled: true } : s));
      setVersion(Date.now());
      refresh();
    },
    onError,
  });
  const clearMutation = useMutation({
    mutationFn: upsamplingApi.clear,
    onSuccess: (r) => {
      toast.success(`Removed ${r.removed_files} augmented files`);
      refresh();
    },
    onError,
  });
  const saveMutation = useMutation({
    mutationFn: (s: UpsamplingSettings) => upsamplingApi.saveSettings(s),
    onSuccess: (s) => {
      setDraft(s);
      toast.success(s.enabled ? "Upsampling will run in the next fine-tune" : "Upsampling turned off for fine-tunes");
      refresh();
    },
    onError,
  });

  const busy = prepareMutation.isPending || generateMutation.isPending || clearMutation.isPending;
  const update = (patch: Partial<UpsamplingSettings>) => setSettings((s) => (s ? { ...s, ...patch } : s));
  const toggleTransform = (key: keyof UpsamplingTransforms) =>
    setSettings((s) => (s ? { ...s, transforms: { ...s.transforms, [key]: !s.transforms[key] } } : s));

  const preview = previewQuery.data;
  const minority = new Set(preview?.minority_classes ?? status?.balance.minority_classes ?? []);
  const unsaved = status && settings && !sameSettings({ ...settings, enabled: false }, { ...status.settings, enabled: false });
  const anyTransform = settings ? Object.values(settings.transforms).some(Boolean) : true;
  const groups = groupsQuery.data;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Upsampling</h1>
          <p className="text-muted-foreground">
            Balance the training set before fine-tuning by adding augmented copies of full images that contain
            under-represented classes. Validation images are never changed.
          </p>
        </div>
        {status && (
          <Badge variant={status.settings.enabled ? "default" : "secondary"}>
            {status.settings.enabled ? "On for the next fine-tune" : "Off for fine-tunes"}
          </Badge>
        )}
      </div>

      {/* ── Training set ── */}
      <Card>
        <CardHeader>
          <CardTitle>Training set</CardTitle>
          <CardDescription>
            Built from your named clusters, replay images of known classes, and external datasets in{" "}
            <span className="font-mono">data/external</span>. The fine-tune rebuilds it the same way.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-2 text-sm">
          {status?.dataset_ready ? (
            <>
              <Badge variant="outline">{status.train_images} train images</Badge>
              <Badge variant="outline">{status.val_images} val images</Badge>
              <Badge variant="outline">{status.class_names.length} classes</Badge>
              {Object.entries(status.origins).map(([o, n]) => (
                <Badge key={o} variant="secondary">
                  {originLabel(o)}: {n}
                </Badge>
              ))}
              {status.origins.unknown ? (
                <span className="text-xs text-muted-foreground">
                  Sources are unknown because this set was built before source tracking; rebuild it to see them.
                </span>
              ) : null}
            </>
          ) : (
            <span className="text-muted-foreground">
              {statusQuery.isLoading ? "Loading..." : "No training set yet. Build it from your named clusters."}
            </span>
          )}
        </CardContent>
        <CardFooter>
          <Button variant="outline" disabled={busy} onClick={() => prepareMutation.mutate()}>
            {prepareMutation.isPending ? "Rebuilding..." : status?.dataset_ready ? "Rebuild training set" : "Build training set"}
          </Button>
        </CardFooter>
      </Card>

      <div className="grid items-start gap-4 lg:grid-cols-2">
        {/* ── Options ── */}
        <Card>
          <CardHeader>
            <CardTitle>Options</CardTitle>
            <CardDescription>
              Defaults suit this dataset: lift minority classes to half the largest class, never more than 3× their own
              size, using flips and lighting changes only.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {settings && (
              <>
                <OptionSelect
                  label="Minority threshold"
                  help="Only classes at or below this share of the largest class are upsampled."
                  value={settings.imbalance_threshold}
                  options={THRESHOLD_OPTIONS}
                  onChange={(v) => update({ imbalance_threshold: v })}
                />
                <OptionSelect
                  label="Bring minority classes up to"
                  help="Target box count, as a share of the largest class."
                  value={settings.target_ratio}
                  options={RATIO_OPTIONS}
                  onChange={(v) => update({ target_ratio: v })}
                />
                <OptionSelect
                  label="But never more than"
                  help="Stops a tiny class being cloned so often the model just memorises it."
                  value={settings.max_factor}
                  options={FACTOR_OPTIONS}
                  onChange={(v) => update({ max_factor: v })}
                />
                <fieldset className="flex flex-col gap-1.5">
                  <legend className="mb-1 text-sm font-medium">Transforms</legend>
                  <div className="grid gap-1.5 sm:grid-cols-2">
                    {TRANSFORMS.map((t) => (
                      <label key={t.key} className="flex items-start gap-2 text-sm" title={t.hint}>
                        <input
                          type="checkbox"
                          className="mt-0.5"
                          checked={settings.transforms[t.key]}
                          onChange={() => toggleTransform(t.key)}
                        />
                        <span>{t.label}</span>
                      </label>
                    ))}
                  </div>
                  {!anyTransform && <p className="text-xs text-destructive">Turn on at least one transform.</p>}
                </fieldset>
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="seed" className="text-sm font-medium">
                    Seed
                  </label>
                  <Input
                    id="seed"
                    type="number"
                    className="w-32"
                    value={settings.seed}
                    onChange={(e) => update({ seed: Number.parseInt(e.target.value || "0", 10) })}
                  />
                  <p className="text-xs text-muted-foreground">Same settings and seed always produce the same images.</p>
                </div>
              </>
            )}
          </CardContent>
          <CardFooter className="flex flex-wrap gap-2">
            <Button
              disabled={busy || !settings || !status?.dataset_ready || !anyTransform}
              onClick={() => generateMutation.mutate()}
            >
              {generateMutation.isPending ? "Generating..." : "Generate augmented images"}
            </Button>
            <Button
              variant="outline"
              disabled={busy || !status?.defaults}
              onClick={() => status && setDraft({ ...status.defaults, enabled: settings?.enabled ?? false })}
            >
              Reset to defaults
            </Button>
            <Button
              variant="outline"
              disabled={busy || !status?.augmented_images}
              onClick={() => clearMutation.mutate()}
            >
              Clear augmented images
            </Button>
          </CardFooter>
        </Card>

        {/* ── Balance ── */}
        <Card>
          <CardHeader>
            <CardTitle>Class balance</CardTitle>
            <CardDescription>Boxes per class in the training split, now → after upsampling with these options.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {status?.dataset_ready ? (
              <>
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge variant="outline">{status.balance.imbalance_ratio}× imbalance</Badge>
                  <Badge variant={status.recommendation.recommend ? "default" : "secondary"}>
                    {status.recommendation.recommend ? "Upsampling recommended" : "Upsampling not needed"}
                  </Badge>
                  {preview && <Badge variant="outline">+{preview.generated_images} images</Badge>}
                </div>
                <p className="text-sm text-muted-foreground">{status.recommendation.reason}</p>
                <BalanceBars before={status.counts} after={preview?.after ?? null} minority={minority} />
                {preview && (preview.capped_classes.length > 0 || preview.limited_classes.length > 0) && (
                  <p className="text-xs text-muted-foreground">
                    {preview.capped_classes.length > 0 && (
                      <>Held back by the {settings?.max_factor}× limit: {preview.capped_classes.join(", ")}. </>
                    )}
                    {preview.limited_classes.length > 0 && (
                      <>Ran out of distinct copies: {Array.from(new Set(preview.limited_classes)).join(", ")}.</>
                    )}
                  </p>
                )}
                {previewQuery.error && <p className="text-xs text-destructive">{previewQuery.error.message}</p>}
              </>
            ) : (
              <p className="text-sm text-muted-foreground">Build the training set to see the class balance.</p>
            )}
          </CardContent>
          {settings && status && (
            <CardFooter className="flex flex-wrap items-center gap-3 border-t pt-4">
              <label className="flex items-center gap-2 text-sm font-medium">
                <input
                  type="checkbox"
                  checked={status.settings.enabled}
                  disabled={saveMutation.isPending}
                  onChange={(e) => saveMutation.mutate({ ...settings, enabled: e.target.checked })}
                />
                Use upsampling in the next fine-tune
              </label>
              {unsaved && (
                <Button size="sm" variant="outline" disabled={saveMutation.isPending} onClick={() => saveMutation.mutate(settings)}>
                  Save options
                </Button>
              )}
              {unsaved && <span className="text-xs text-amber-600 dark:text-amber-400">Options changed since last save</span>}
            </CardFooter>
          )}
        </Card>
      </div>

      {/* ── Viewer ── */}
      <div className="flex flex-col gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Augmented images</h2>
          <p className="text-sm text-muted-foreground">
            {status?.augmented_images
              ? `${status.augmented_images} augmented images, grouped by the class they were made for. Open a class to see each original next to its copies.`
              : "Nothing generated yet. Choose options and generate to review the images here."}
          </p>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {groups?.classes.map((c) => {
            const sample = groups.groups.find((g) => g.for_classes.includes(c.class_name));
            const aug = sample?.augmented.find((a) => a.for_class === c.class_name);
            return (
              <Card key={c.class_name}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between gap-2">
                    <CardTitle className="truncate font-mono text-sm">{c.class_name}</CardTitle>
                    <Badge variant="outline">+{c.augmented}</Badge>
                  </div>
                </CardHeader>
                <CardContent className="flex flex-col gap-2">
                  {sample && aug && (
                    <div className="grid grid-cols-2 gap-1 rounded-md border bg-muted/30 p-1">
                      <BoxedImage src={upsamplingApi.imageUrl(sample.source_image)} boxes={sample.boxes} />
                      <BoxedImage src={upsamplingApi.imageUrl(aug.image, version)} boxes={aug.boxes} />
                    </div>
                  )}
                  <p className="text-xs text-muted-foreground tabular-nums">
                    {c.originals} originals · boxes {c.boxes_before} → {c.boxes_after}
                  </p>
                  <Button
                    type="button"
                    variant="link"
                    size="sm"
                    className="h-auto justify-start p-0 text-xs"
                    onClick={() => {
                      setOpenClass(c.class_name);
                      acknowledge("augmented-review:", attentionItems); // looking at them is the review
                    }}
                  >
                    View originals &amp; copies &rarr;
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      <UpsamplingClassDialog
        className={openClass}
        groups={groups?.groups ?? []}
        version={version}
        onOpenChange={(open) => !open && setOpenClass(null)}
      />
    </div>
  );
}
