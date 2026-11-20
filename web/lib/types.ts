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

export interface ClassBalance {
  imbalance_ratio: number;
  largest_class?: string;
  minority_classes: string[];
  suggest_upsampling: boolean;
  max_factor: number;
}

export interface AdvisorPreview {
  metadata: {
    classes: Record<string, number>;
    num_classes: number;
    total_crops: number;
    class_balance: ClassBalance;
  };
  recommendation: {
    should_train: boolean;
    reason: string;
    config: { epochs?: number; batch?: number; freeze?: number | null; optimizer?: string };
    augmentation_notes: string;
    recommend_upsampling: boolean;
    upsampling_reason: string;
  };
}

// ── Upsampling ────────────────────────────────────────────────

export interface UpsamplingTransforms {
  hflip: boolean;
  vflip: boolean;
  rot180: boolean;
  rot90: boolean;
  contrast: boolean;
  gamma: boolean;
  blur: boolean;
  noise: boolean;
}

export interface UpsamplingSettings {
  enabled: boolean;
  target_ratio: number;
  max_factor: number;
  imbalance_threshold: number;
  seed: number;
  transforms: UpsamplingTransforms;
}

export interface UpsamplingResult {
  target_ratio: number;
  max_factor: number;
  imbalance_threshold: number;
  transforms: UpsamplingTransforms;
  seed: number;
  dry_run: boolean;
  before: Record<string, number>;
  after: Record<string, number>;
  generated_images: number;
  minority_classes: string[];
  classes_upsampled: string[];
  capped_classes: string[];
  limited_classes: string[];
  note?: string;
}

export interface UpsamplingStatus {
  dataset_ready: boolean;
  class_names: string[];
  train_images: number;
  val_images: number;
  origins: Record<string, number>;
  counts: Record<string, number>;
  counts_with_augmented: Record<string, number>;
  augmented_images: number;
  last_run: UpsamplingResult | null;
  balance: ClassBalance;
  recommendation: { recommend: boolean; reason: string };
  settings: UpsamplingSettings;
  defaults: UpsamplingSettings;
}

export interface UpsamplingBox {
  class_id: number;
  class_name: string;
  xywhn: [number, number, number, number];
}

export interface UpsamplingAugmented {
  image: string;
  geometric: string;
  photometric: string[];
  for_class: string;
  boxes: UpsamplingBox[];
}

export interface UpsamplingGroup {
  source_image: string;
  origin: string;
  boxes: UpsamplingBox[];
  for_classes: string[];
  augmented: UpsamplingAugmented[];
}

export interface UpsamplingGroups {
  classes: { class_name: string; originals: number; augmented: number; boxes_before: number; boxes_after: number }[];
  groups: UpsamplingGroup[];
}

// ── Pipeline DAG (Overview graph) ─────────────────────────────

export interface PipelineGraph {
  kind: "discovery" | "retraining";
  run_id: string | null;
  run_status: PipelineRunStatus;
  started_at: string | null;
  error: string | null;
  nodes: { id: string; label: string; gate: string | null }[];
  steps: Record<
    string,
    {
      status: "pending" | "running" | "success" | "failed";
      start_time: string | null;
      end_time: string | null;
      latency_seconds: number;
      items_processed: number;
      items_failed: number;
      metadata: Record<string, unknown>;
      error: string | null;
    }
  >;
}

// ── Attention ("your input is needed") ────────────────────────

export interface AttentionItem {
  key: string;
  severity: "blocking" | "recommended";
  title: string;
  detail: string;
  href: string;
  action: string;
  since: string | null;
  page: string;
}
