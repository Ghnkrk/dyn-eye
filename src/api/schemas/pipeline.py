from pydantic import BaseModel, Field


class DiscoveryRequest(BaseModel):
    use_sample_run: bool = False
    input_images_dir: str | None = None
    yolo_model_path: str | None = None
    use_cache: bool = False


class RetrainingRequest(BaseModel):
    project_id: int
    epochs: int | None = None
    imgsz: int | None = None
    batch_size: int | None = None
    freeze: int | None = None
    base_model_path: str | None = None  # weights to fine-tune from; None = currently active model
    upsample: bool = False  # opt-in class-balancing upsampling before training
    upsample_target_ratio: float | None = Field(None, gt=0, le=1)  # None = config default


class SmartRetrainRequest(BaseModel):
    epochs: int | None = None
    imgsz: int | None = None
    batch_size: int | None = None
    freeze: int | None = None  # user override; None = let LLM decide
    base_model_path: str | None = None  # weights to fine-tune from; None = currently active model
    upsample: bool = False  # opt-in class-balancing upsampling before training
    upsample_target_ratio: float | None = Field(None, gt=0, le=1)  # None = config default


class OrchestratorRequest(BaseModel):
    project_id: int | None = None
