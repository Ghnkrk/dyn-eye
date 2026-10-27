// Mirrors the Pydantic/SQLModel response shapes in the FastAPI backend
// (src/api/routers/*.py, src/db/models.py). Kept hand-written and in sync
// rather than codegen'd — the API surface is small and stable.

export type PipelineRunStatus = "idle" | "running" | "complete" | "failed";

export interface DiscoveryStatus {
  status: PipelineRunStatus;
  run_id: string | null;
  result: Record<string, unknown> | null;
  started_at?: string;
  completed_at?: string;
  error?: string | null;
}

export interface RetrainingStatus {
  status: PipelineRunStatus;
  run_id: string | null;
  result: Record<string, unknown> | null;
  started_at?: string;
  completed_at?: string;
  error?: string | null;
}

export interface PipelineStatus {
  discovery: DiscoveryStatus;
  retraining: RetrainingStatus;
  orchestrator: { status: string };
  clusters_ready: boolean;
}

export interface Cluster {
  name: string;
  image_count: number;
  images: string[];
  defect_name: string | null;
  cohesion: number | null;
}

export interface ClustersResponse {
  clusters: Cluster[];
  pipeline_running?: boolean;
  global_icc?: number;
  global_silhouette?: number;
  mean_cohesion?: number;
}

export interface ClusterCrop {
  crop_file: string;
  crop_path: string;
  source_image: string;
  source_image_name: string;
  box_2d_pixels: number[];
  box_2d_raw: number[];
  physical_traits: string;
  crop_width: number;
  crop_height: number;
}

export interface ManifestClusterEntry {
  cluster_id: number;
  crop_count: number;
  defect_name: string | null;
  fingerprint_id?: string | null;
  cohesion?: number | null;
  crops: ClusterCrop[];
}

export interface ClusterManifest {
  run_id?: string;
  vlm_system_prompt?: string;
  global_icc?: number;
  clusters: Record<string, ManifestClusterEntry>;
}

export interface ModelVersion {
  version_id: string;
  version_num: number;
  path: string;
  original_path: string;
  timestamp: string;
  created_at: string | null;
  metrics: Record<string, number>;
  training_config: Record<string, unknown>;
  source: string;
  notes: string;
  classes: string[];
  dataset_stats: Record<string, unknown>;
  size_mb: number;
  status: "registered" | "deployed" | "retired";
}

export interface ModelVersionsResponse {
  versions: ModelVersion[];
  current: string | null;
}

export interface DeploymentEvent {
  version_id: string;
  previous_version: string | null;
  deployed_at: string;
  confirmed_by: string;
  action: "deploy" | "rollback";
}

export interface RunSummary {
  run_id: string;
  kind: string;
  status: string;
  steps: string[];
  summary: Record<string, string>;
}

export interface StepMetric {
  step_name: string;
  status: string;
  start_time: string | null;
  end_time: string | null;
  latency_seconds: number;
  items_processed: number;
  items_failed: number;
  metadata_json: Record<string, unknown>;
  error: string | null;
}

export interface RunDetail {
  run_id: string;
  kind: string;
  status: string;
  result: Record<string, unknown>;
  error: string | null;
  steps: Record<string, StepMetric>;
}

export interface LogEvent {
  ts: string;
  level: "info" | "warn" | "error" | "step" | "progress" | string;
  source: string;
  message: string;
  data?: Record<string, unknown>;
}

export interface InferenceDetection {
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

export interface InferenceResult {
  image_b64: string;
  detections: InferenceDetection[];
  num_detections: number;
  classes: string[];
  model: string;
}

export interface InferenceCompareResponse {
  baseline: InferenceResult;
  finetuned: InferenceResult;
}

export interface SystemConfig {
  known_defect_names: string[];
  faiss_novelty_threshold: number;
  hdbscan_min_cluster_size: number;
  yolo_model_exists: boolean;
  faiss_index_exists: boolean;
  input_images_count: number;
}

export type WsTopic = "logs" | "pipeline-status" | "clusters" | "models";

export interface WsFrame<T = unknown> {
  topic: WsTopic;
  event: string;
  data: T;
}
