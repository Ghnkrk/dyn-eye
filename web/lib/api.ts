// Typed fetch client for the DYN-EYE FastAPI backend.
//
// Requests go to /api/v2/* which next.config.ts rewrites to the backend
// (same-origin from the browser's point of view, no CORS involved).
import type {
  ClusterManifest,
  ClustersResponse,
  DeploymentEvent,
  InferenceCompareResponse,
  ModelVersion,
  ModelVersionsResponse,
  PipelineStatus,
  RunDetail,
  RunSummary,
  SystemConfig,
} from "./types";

const BASE = "/api/v2";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = body.detail ?? message;
    } catch {
      // response wasn't JSON — fall back to statusText
    }
    throw new ApiError(res.status, message);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

function postJson<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined });
}

// ── Pipeline ──────────────────────────────────────────────────

export const pipelineApi = {
  status: () => request<PipelineStatus>("/pipeline/status"),
  runs: () => request<RunSummary[]>("/pipeline/runs"),
  run: (runId: string) => request<RunDetail>(`/pipeline/runs/${encodeURIComponent(runId)}`),
  triggerDiscovery: (body: {
    use_sample_run?: boolean;
    input_images_dir?: string | null;
    yolo_model_path?: string | null;
    use_cache?: boolean;
  }) => postJson<{ message: string; status: string }>("/pipeline/discovery/trigger", body),
  triggerRetraining: (body: {
    project_id: number;
    epochs?: number | null;
    imgsz?: number | null;
    batch_size?: number | null;
    freeze?: number | null;
    base_model_path?: string | null;
  }) => postJson<{ message: string; status: string }>("/pipeline/retraining/trigger", body),
  smartRetrain: (body: {
    epochs?: number | null;
    imgsz?: number | null;
    batch_size?: number | null;
    freeze?: number | null;
    base_model_path?: string | null;
  }) => postJson<{ message: string; status: string }>("/pipeline/retraining/smart-trigger", body),
  advisorPreview: () =>
    request<{ metadata: Record<string, unknown>; recommendation: Record<string, unknown> }>(
      "/pipeline/retraining/advisor-preview"
    ),
  startOrchestrator: (projectId?: number | null) =>
    postJson<{ message: string }>("/pipeline/orchestrator/start", { project_id: projectId ?? null }),
  stopOrchestrator: () => postJson<{ message: string }>("/pipeline/orchestrator/stop"),
};

// ── Clusters ──────────────────────────────────────────────────

export const clustersApi = {
  list: () => request<ClustersResponse>("/clusters"),
  manifest: () => request<ClusterManifest>("/clusters/manifest"),
  name: (names: Record<string, string>) =>
    postJson<{ message: string; updated_clusters: string[]; total_labeled_crops: number }>("/clusters/name", {
      names,
    }),
  merge: (sourceCluster: string, targetCluster: string, mergedLabel?: string | null) =>
    postJson<{ message: string; target_cluster: string; merged_crop_count: number; total_labeled_crops: number }>(
      "/clusters/merge",
      { source_cluster: sourceCluster, target_cluster: targetCluster, merged_label: mergedLabel ?? null }
    ),
  batchEdit: (body: { crop_files: string[]; source_cluster: string; target_cluster?: string | null; action: "move" | "drop" }) =>
    postJson<{ message: string; total_labeled_crops: number }>("/clusters/batch-edit-crops", body),
  editCrop: (body: { crop_file: string; source_cluster: string; target_cluster?: string | null; action: "move" | "drop" }) =>
    postJson<{ message: string; total_labeled_crops: number }>("/clusters/edit-crop", body),
  cropImageUrl: (clusterName: string, filename: string) =>
    `${BASE}/clusters/${encodeURIComponent(clusterName)}/${encodeURIComponent(filename)}`,
};

// ── Models ────────────────────────────────────────────────────

export const modelsApi = {
  list: () => request<ModelVersionsResponse>("/models"),
  active: () =>
    request<{ active_version: string | null; entry: unknown; model_exists: boolean; model_path: string }>(
      "/models/active"
    ),
  history: () => request<{ history: DeploymentEvent[] }>("/models/history"),
  deploy: (versionId: string) => postJson<{ success: boolean; error?: string }>("/models/deploy-confirm", { version_id: versionId }),
  rollback: (versionId: string, confirmedBy = "dashboard") =>
    postJson<{ success: boolean; error?: string }>("/models/rollback", { version_id: versionId, confirmed_by: confirmedBy }),
  upload: (file: File, notes = "") => {
    const form = new FormData();
    form.append("file", file);
    form.append("notes", notes);
    return request<ModelVersion>("/models/upload", { method: "POST", body: form });
  },
  replace: (versionId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ success: boolean; error?: string; entry?: ModelVersion }>(
      `/models/${encodeURIComponent(versionId)}/replace`,
      { method: "POST", body: form }
    );
  },
  remove: (versionId: string) =>
    request<{ success: boolean; error?: string }>(`/models/${encodeURIComponent(versionId)}`, { method: "DELETE" }),
};

// ── FAISS ─────────────────────────────────────────────────────

export const faissApi = {
  setup: (knownCropsDir?: string | null, classSubdirs = true) =>
    postJson<{ message: string; count: number }>("/faiss/setup", { known_crops_dir: knownCropsDir ?? null, class_subdirs: classSubdirs }),
  reset: () => postJson<{ message: string; success: boolean }>("/faiss/reset"),
  rebuild: () => postJson<{ message: string; success: boolean; count: number }>("/faiss/rebuild"),
};

// ── System ────────────────────────────────────────────────────

export const systemApi = {
  config: () => request<SystemConfig>("/system/config"),
  resetAll: (deleteCrops = true) => postJson<{ message: string; success: boolean }>("/system/reset-all", { delete_crops: deleteCrops }),
};

// ── Images ────────────────────────────────────────────────────

export const imagesApi = {
  upload: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request<{ success: boolean; count: number }>("/images/upload", { method: "POST", body: form });
  },
  inputImages: () => request<string[]>("/input-images"),
  inputImageUrl: (filename: string) => `${BASE}/input-images/${encodeURIComponent(filename)}`,
  cropImageUrl: (filename: string) => `${BASE}/crops/${encodeURIComponent(filename)}`,
};

// ── Inference ─────────────────────────────────────────────────

export const inferenceApi = {
  compare: (file: File, conf = 0.1) => {
    const form = new FormData();
    form.append("file", file);
    return request<InferenceCompareResponse>(`/inference/compare?conf=${conf}`, { method: "POST", body: form });
  },
};

// ── VLM ───────────────────────────────────────────────────────

export const vlmApi = {
  score: () => request<Record<string, unknown>>("/vlm/score"),
};

export { ApiError };
