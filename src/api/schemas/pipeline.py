from pydantic import BaseModel


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


class SmartRetrainRequest(BaseModel):
    epochs: int | None = None
    imgsz: int | None = None
    batch_size: int | None = None
    freeze: int | None = None  # user override; None = let LLM decide


class OrchestratorRequest(BaseModel):
    project_id: int | None = None
