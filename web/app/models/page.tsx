"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { toast } from "sonner";

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
import { TableSkeletonRows } from "@/components/dashboard/table-skeleton-rows";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { modelsApi } from "@/lib/api";
import { queryClient } from "@/lib/queryClient";

function statusBadge(status: string) {
  if (status === "deployed") return <Badge>deployed</Badge>;
  if (status === "retired") return <Badge variant="secondary">retired</Badge>;
  return <Badge variant="outline">{status}</Badge>;
}

export default function ModelsPage() {
  const versionsQuery = useQuery({ queryKey: ["models", "versions"], queryFn: modelsApi.list });
  const historyQuery = useQuery({ queryKey: ["models", "history"], queryFn: modelsApi.history });

  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadNotes, setUploadNotes] = useState("");

  const [replaceTarget, setReplaceTarget] = useState<string | null>(null);
  const [replaceFile, setReplaceFile] = useState<File | null>(null);

  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const uploadInputRef = useRef<HTMLInputElement>(null);
  const replaceInputRef = useRef<HTMLInputElement>(null);

  function refresh() {
    queryClient.invalidateQueries({ queryKey: ["models"] });
  }

  const deployMutation = useMutation({
    mutationFn: (versionId: string) => modelsApi.deploy(versionId),
    onSuccess: (res) => {
      if (res.success) toast.success("Deployed");
      else toast.error(res.error ?? "Deploy failed");
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const rollbackMutation = useMutation({
    mutationFn: (versionId: string) => modelsApi.rollback(versionId),
    onSuccess: (res) => {
      if (res.success) toast.success("Rolled back");
      else toast.error(res.error ?? "Rollback failed");
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const uploadMutation = useMutation({
    mutationFn: () => {
      if (!uploadFile) throw new Error("Choose a .pt file first");
      return modelsApi.upload(uploadFile, uploadNotes);
    },
    onSuccess: (entry) => {
      toast.success(`Registered ${entry.version_id}`);
      setUploadOpen(false);
      setUploadFile(null);
      setUploadNotes("");
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const replaceMutation = useMutation({
    mutationFn: () => {
      if (!replaceTarget || !replaceFile) throw new Error("Choose a .pt file first");
      return modelsApi.replace(replaceTarget, replaceFile);
    },
    onSuccess: (res) => {
      if (res.success) toast.success(`Replaced weights for ${replaceTarget}`);
      else toast.error(res.error ?? "Replace failed");
      setReplaceTarget(null);
      setReplaceFile(null);
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const deleteMutation = useMutation({
    mutationFn: (versionId: string) => modelsApi.remove(versionId),
    onSuccess: (res) => {
      if (res.success) toast.success(`Deleted ${deleteTarget}`);
      else toast.error(res.error ?? "Delete failed");
      setDeleteTarget(null);
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const versions = versionsQuery.data?.versions ?? [];
  const current = versionsQuery.data?.current;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Models</h1>
          <p className="text-muted-foreground">Active version: {current ?? "none"}</p>
        </div>
        <Button onClick={() => setUploadOpen(true)}>Upload model</Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Versions</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Version</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Classes</TableHead>
                <TableHead>mAP50</TableHead>
                <TableHead>Size</TableHead>
                <TableHead>Source</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {versionsQuery.isLoading && <TableSkeletonRows columns={7} />}
              {versions.map((v) => (
                <TableRow key={v.version_id}>
                  <TableCell className="font-mono text-xs">{v.version_id}</TableCell>
                  <TableCell>{statusBadge(v.status)}</TableCell>
                  <TableCell className="max-w-xs truncate text-xs text-muted-foreground">
                    {v.classes.join(", ")}
                  </TableCell>
                  <TableCell>{v.metrics?.map50 != null ? Number(v.metrics.map50).toFixed(3) : "—"}</TableCell>
                  <TableCell>{v.size_mb.toFixed(1)} MB</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{v.source}</TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      {v.status !== "deployed" && (
                        <>
                          <Button size="sm" variant="outline" onClick={() => rollbackMutation.mutate(v.version_id)}>
                            Rollback
                          </Button>
                          <Button size="sm" onClick={() => deployMutation.mutate(v.version_id)}>
                            Deploy
                          </Button>
                        </>
                      )}
                      <Button size="sm" variant="outline" onClick={() => setReplaceTarget(v.version_id)}>
                        Replace
                      </Button>
                      <Button
                        size="sm"
                        variant="destructive"
                        disabled={v.status === "deployed"}
                        title={v.status === "deployed" ? "Deploy another version before deleting this one" : undefined}
                        onClick={() => setDeleteTarget(v.version_id)}
                      >
                        Delete
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
              {!versionsQuery.isLoading && versions.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} className="text-center text-muted-foreground">
                    No model versions registered yet.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Deployment history</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>When</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Previous</TableHead>
                <TableHead>By</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {historyQuery.isLoading && <TableSkeletonRows columns={5} />}
              {(historyQuery.data?.history ?? []).map((h, i) => (
                <TableRow key={`${h.version_id}-${h.deployed_at}-${i}`}>
                  <TableCell className="text-xs text-muted-foreground">{new Date(h.deployed_at).toLocaleString()}</TableCell>
                  <TableCell>
                    <Badge variant={h.action === "rollback" ? "secondary" : "default"}>{h.action}</Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs">{h.version_id}</TableCell>
                  <TableCell className="font-mono text-xs text-muted-foreground">{h.previous_version ?? "—"}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{h.confirmed_by}</TableCell>
                </TableRow>
              ))}
              {!historyQuery.isLoading && (historyQuery.data?.history ?? []).length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-center text-muted-foreground">
                    No deployment events yet.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Create: upload a new .pt checkpoint and register it as a new version */}
      <Dialog
        open={uploadOpen}
        onOpenChange={(open) => {
          setUploadOpen(open);
          if (!open) {
            setUploadFile(null);
            setUploadNotes("");
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Upload model</DialogTitle>
            <DialogDescription>
              Register a new .pt checkpoint as a model version. It won&apos;t be deployed until you hit Deploy.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            <input
              ref={uploadInputRef}
              type="file"
              accept=".pt"
              className="hidden"
              onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
            />
            <Button type="button" variant="outline" size="sm" onClick={() => uploadInputRef.current?.click()}>
              {uploadFile ? uploadFile.name : "Choose .pt file..."}
            </Button>
            <Input
              placeholder="Notes (optional)"
              value={uploadNotes}
              onChange={(e) => setUploadNotes(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button
              disabled={!uploadFile || uploadMutation.isPending}
              onClick={() => uploadMutation.mutate()}
            >
              {uploadMutation.isPending ? "Uploading..." : "Register version"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Update: replace the weights file for an existing version */}
      <Dialog
        open={replaceTarget !== null}
        onOpenChange={(open) => {
          if (!open) {
            setReplaceTarget(null);
            setReplaceFile(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Replace weights — {replaceTarget}</DialogTitle>
            <DialogDescription>
              Upload a new .pt file to overwrite this version&apos;s weights in place. If this version is currently
              deployed, the live model is updated too.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            <input
              ref={replaceInputRef}
              type="file"
              accept=".pt"
              className="hidden"
              onChange={(e) => setReplaceFile(e.target.files?.[0] ?? null)}
            />
            <Button type="button" variant="outline" size="sm" onClick={() => replaceInputRef.current?.click()}>
              {replaceFile ? replaceFile.name : "Choose .pt file..."}
            </Button>
          </div>
          <DialogFooter>
            <Button
              disabled={!replaceFile || replaceMutation.isPending}
              onClick={() => replaceMutation.mutate()}
            >
              {replaceMutation.isPending ? "Replacing..." : "Replace weights"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete */}
      <AlertDialog open={deleteTarget !== null} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {deleteTarget}?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently removes the version entry and its weights file. This can&apos;t be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={deleteMutation.isPending}
              onClick={() => deleteTarget && deleteMutation.mutate(deleteTarget)}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
