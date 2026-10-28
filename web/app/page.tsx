"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { LogPanel } from "@/components/dashboard/log-panel";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { imagesApi, pipelineApi, systemApi } from "@/lib/api";
import { queryClient } from "@/lib/queryClient";
import { useLiveStore } from "@/store/useLiveStore";

const IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"];

function folderNameOf(files: File[]): string {
  const relPath = files[0]?.webkitRelativePath;
  return relPath ? relPath.split("/")[0] : "selected folder";
}

export default function OverviewPage() {
  const liveStatus = useLiveStore((s) => s.pipelineStatus);
  const [useCache, setUseCache] = useState(true);
  const [source, setSource] = useState<"sample" | "upload">("sample");
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [uploadedFolder, setUploadedFolder] = useState<string | null>(null);
  const [resetOpen, setResetOpen] = useState(false);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const statusQuery = useQuery({
    queryKey: ["pipeline", "status"],
    queryFn: pipelineApi.status,
    refetchInterval: 5000, // fallback poll in case the WS drops — WS updates win via setQueryData
  });

  const configQuery = useQuery({ queryKey: ["system", "config"], queryFn: systemApi.config });

  const status = liveStatus ?? statusQuery.data;

  function handleFolderSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const all = Array.from(e.target.files ?? []);
    const images = all.filter((f) => IMAGE_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext)));
    setPendingFiles(images);
    setUploadedFolder(null);
    if (all.length > 0 && images.length === 0) {
      toast.error("That folder has no recognized image files");
    }
  }

  const uploadMutation = useMutation({
    mutationFn: () => imagesApi.upload(pendingFiles),
    onSuccess: (res) => {
      toast.success(`Uploaded ${res.count} image(s)`);
      setUploadedFolder(folderNameOf(pendingFiles));
      queryClient.invalidateQueries({ queryKey: ["system", "config"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const discoveryMutation = useMutation({
    mutationFn: () =>
      pipelineApi.triggerDiscovery(
        source === "sample"
          ? { use_sample_run: true, use_cache: useCache }
          : { use_sample_run: false, use_cache: useCache }
      ),
    onSuccess: () => toast.success("Discovery pipeline triggered"),
    onError: (e: Error) => toast.error(e.message),
  });

  const resetMutation = useMutation({
    mutationFn: () => systemApi.resetAll(true),
    onSuccess: () => {
      toast.success("System reset to initial state");
      setResetOpen(false);
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const discoveryRunning = status?.discovery.status === "running";

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-muted-foreground">Trigger pipeline runs and watch them execute in real time.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Discovery</CardTitle>
            <CardDescription>Run YOLO + DINOv2 + FAISS + HDBSCAN + VLM on a folder of images.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5 text-sm">
              <label className="flex items-center gap-2">
                <input
                  type="radio"
                  name="discovery-source"
                  checked={source === "sample"}
                  onChange={() => setSource("sample")}
                />
                Sample run folder
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="radio"
                  name="discovery-source"
                  checked={source === "upload"}
                  onChange={() => setSource("upload")}
                />
                Select an image folder
              </label>
            </div>

            {source === "upload" && (
              <div className="flex flex-col gap-2 rounded-md border bg-muted/30 p-2">
                <input
                  ref={folderInputRef}
                  type="file"
                  multiple
                  accept="image/*"
                  className="hidden"
                  // webkitdirectory/directory are non-standard but broadly
                  // supported (Chromium, Firefox, Edge) for folder selection.
                  // @ts-expect-error -- not in React's input prop typings
                  webkitdirectory=""
                  directory=""
                  onChange={handleFolderSelect}
                />
                <Button type="button" variant="outline" size="sm" onClick={() => folderInputRef.current?.click()}>
                  Choose folder...
                </Button>
                {pendingFiles.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    {folderNameOf(pendingFiles)} — {pendingFiles.length} image(s)
                    {uploadedFolder && " · uploaded"}
                  </p>
                )}
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  disabled={pendingFiles.length === 0 || uploadMutation.isPending}
                  onClick={() => uploadMutation.mutate()}
                >
                  {uploadMutation.isPending ? "Uploading..." : `Upload ${pendingFiles.length || ""} image(s)`}
                </Button>
              </div>
            )}

            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={useCache} onChange={(e) => setUseCache(e.target.checked)} />
              Use cache (fast demo, skips YOLO/VLM)
            </label>
            <Button
              disabled={discoveryRunning || discoveryMutation.isPending || (source === "upload" && !uploadedFolder)}
              onClick={() => discoveryMutation.mutate()}
            >
              {discoveryRunning ? "Running..." : "Trigger discovery"}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>System</CardTitle>
            <CardDescription>
              {configQuery.data ? (
                `${configQuery.data.known_defect_names.length} known classes · ${configQuery.data.input_images_count} input images`
              ) : (
                <Skeleton className="h-4 w-48" />
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button variant="destructive" disabled={resetMutation.isPending} onClick={() => setResetOpen(true)}>
              Factory reset
            </Button>
            <AlertDialog open={resetOpen} onOpenChange={setResetOpen}>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Reset everything to the initial state?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This deletes crops, clusters, and fine-tuning data, and restores the baseline YOLO model and FAISS
                    index. This can&apos;t be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                    onClick={() => resetMutation.mutate()}
                  >
                    Reset everything
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </CardContent>
        </Card>
      </div>

      <LogPanel />
    </div>
  );
}
