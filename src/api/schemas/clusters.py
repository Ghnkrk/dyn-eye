from pydantic import BaseModel


class ClusterNamingRequest(BaseModel):
    """Request to assign defect names to clusters."""
    names: dict[str, str]  # cluster_name -> defect_name


class MergeClustersRequest(BaseModel):
    """Merge all crops from source_cluster into target_cluster."""
    source_cluster: str   # cluster to dissolve
    target_cluster: str   # cluster to absorb into
    merged_label: str | None = None  # optional defect name for the merged cluster


class BatchEditCropsRequest(BaseModel):
    """Perform a batch action on multiple crops in a cluster."""
    crop_files: list[str]
    source_cluster: str
    target_cluster: str | None = None
    action: str  # "move" or "drop"


class EditCropRequest(BaseModel):
    """Reassign or drop a single crop image in a cluster."""
    crop_file: str
    source_cluster: str
    target_cluster: str | None = None
    action: str  # "move" or "drop"
